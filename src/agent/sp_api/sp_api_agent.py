"""
Build ReAct agent wired to SP-API order and listing tools.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, List, Optional

from ..models import AgentState
from ..react_agent import ReActAgent
from .sp_api_client import SPAPIClient, SPAPICredentials
from .tools import SpApiGetListingsTool, SpApiGetOrdersTool


class SpApiReActAgent(ReActAgent):
    """
    SP-API ReAct loop: uses tools to fetch data, then returns **server-built** YAML for orders.

    The LLM must not invent order fields (getOrder does not return line items; status must
    match Amazon). When ``sp_api_get_orders`` succeeded, the user-visible reply is only the
    authoritative ``orders_yaml`` from the tool, not the model's Final Answer prose/YAML.
    """

    _PROMPT_SUFFIX = (
        "\n\nSP-API rules:\n"
        "- Call sp_api_get_orders with the real Amazon order id(s). Do not fabricate JSON/YAML.\n"
        "- For SKU/listing questions, call sp_api_get_listings with real seller SKU(s).\n"
        "- After a successful tool call, stop: output Final Answer: OK (the service will attach "
        "the real Amazon data).\n"
    )
    _ORDER_ID_PATTERN = re.compile(r"\b\d{3}-\d{7}-\d{7}\b")
    _SKU_AFTER_KEYWORD_PATTERN = re.compile(r"(?i)\bsku\s*[:#]?\s*([A-Za-z0-9][A-Za-z0-9._-]{2,})")
    _SKU_TOKEN_PATTERN = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9-]*_[A-Za-z0-9][A-Za-z0-9_-]*\b")

    def _build_prompt(self, state: AgentState) -> str:
        """Append SP-API formatting rules so the model keeps full order payloads."""
        return super()._build_prompt(state) + self._PROMPT_SUFFIX

    def _extract_orders_yaml(self, state: AgentState) -> Optional[str]:
        """Return the latest orders_yaml string from a successful sp_api_get_orders call."""
        for _thought, _action, obs in reversed(state.history):
            if obs.tool_name != "sp_api_get_orders" or not obs.success:
                continue
            out = obs.output
            if isinstance(out, dict):
                raw = out.get("orders_yaml")
                if isinstance(raw, str) and raw.strip():
                    return raw.strip()
        return None

    def _extract_listings_yaml(self, state: AgentState) -> Optional[str]:
        """Return the latest listings_yaml string from a successful sp_api_get_listings call."""
        for _thought, _action, obs in reversed(state.history):
            if obs.tool_name != "sp_api_get_listings" or not obs.success:
                continue
            out = obs.output
            if isinstance(out, dict):
                raw = out.get("listings_yaml")
                if isinstance(raw, str) and raw.strip():
                    return raw.strip()
        return None

    def _extract_order_ids_from_query(self, query: str) -> List[str]:
        """
        Extract unique Amazon order IDs from user query text.

        Args:
            query: User input text.

        Returns:
            Stable de-duplicated order ID list (format: 123-1234567-1234567).
        """
        found = self._ORDER_ID_PATTERN.findall(query or "")
        seen: set[str] = set()
        out: List[str] = []
        for oid in found:
            if oid not in seen:
                seen.add(oid)
                out.append(oid)
        return out

    def _extract_skus_from_query(self, query: str) -> List[str]:
        """
        Extract likely seller SKUs from user query text.

        Args:
            query: User input text.

        Returns:
            Stable de-duplicated SKU list.
        """
        text = query or ""
        found: List[str] = []
        found.extend(self._SKU_AFTER_KEYWORD_PATTERN.findall(text))
        found.extend(self._SKU_TOKEN_PATTERN.findall(text))

        seen: set[str] = set()
        out: List[str] = []
        for sku in found:
            s = str(sku).strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def _run_direct_get_orders_when_possible(self, query: str) -> Optional[str]:
        """
        Deterministically call sp_api_get_orders when order IDs are explicit in query.

        This avoids model-only answers and guarantees the response is based on Amazon API
        payloads from the tool layer.

        Args:
            query: User natural-language query.

        Returns:
            Authoritative YAML answer when direct call succeeds; otherwise None.
        """
        ids = self._extract_order_ids_from_query(query)
        if not ids:
            return None
        tool = self._registry.get("sp_api_get_orders")
        if tool is None:
            return None
        try:
            output = tool.execute(order_ids=ids)
        except Exception:
            return None
        if not isinstance(output, dict):
            return None
        yaml_blob = output.get("orders_yaml")
        if not isinstance(yaml_blob, str) or not yaml_blob.strip():
            return None
        return (
            "Below is the Amazon Selling Partner API getOrder response, formatted as YAML. "
            "This data comes from the API only (not from the language model).\n\n"
            f"```yaml\n{yaml_blob.strip()}\n```"
        )

    def _run_direct_get_listings_when_possible(self, query: str) -> Optional[str]:
        """
        Deterministically call sp_api_get_listings when SKU(s) are explicit in query.

        Args:
            query: User natural-language query.

        Returns:
            Authoritative YAML answer when direct call succeeds; otherwise None.
        """
        skus = self._extract_skus_from_query(query)
        if not skus:
            return None
        tool = self._registry.get("sp_api_get_listings")
        if tool is None:
            return None
        try:
            output = tool.execute(skus=skus)
        except Exception:
            return None
        if not isinstance(output, dict):
            return None
        yaml_blob = output.get("listings_yaml")
        if not isinstance(yaml_blob, str) or not yaml_blob.strip():
            return None
        return (
            "Below is the Amazon Selling Partner API getListingsItem response, formatted as YAML. "
            "This data comes from the API only (not from the language model).\n\n"
            f"```yaml\n{yaml_blob.strip()}\n```"
        )

    def run(self, query: str) -> str:
        """
        Run the ReAct loop. If getOrder ran successfully, return **only** tool-built YAML.

        This avoids LLM hallucinations (e.g. fake ``items``, wrong dates, ``COMPLETED`` vs
        Amazon's ``OrderStatus``) being shown as if they were API data.

        Args:
            query: User natural-language question.

        Returns:
            Final answer string: authoritative YAML for orders, or normal agent text otherwise.
        """
        direct_answer = self._run_direct_get_orders_when_possible(query)
        if direct_answer:
            return direct_answer
        direct_answer = self._run_direct_get_listings_when_possible(query)
        if direct_answer:
            return direct_answer

        start_time = time.time()
        state = AgentState(query=query)

        for _ in range(self._max_iterations):
            if state.is_complete:
                break
            state = self.step(state)

        execution_time = time.time() - start_time
        self._logger.log_run_complete(state, execution_time)

        yaml_blob = self._extract_orders_yaml(state)
        if yaml_blob:
            # Single source of truth: matches getOrder + scripts/test_get_amazon_order.py shape.
            return (
                "Below is the Amazon Selling Partner API getOrder response, formatted as YAML. "
                "This data comes from the API only (not from the language model).\n\n"
                f"```yaml\n{yaml_blob}\n```"
            )
        listings_yaml_blob = self._extract_listings_yaml(state)
        if listings_yaml_blob:
            return (
                "Below is the Amazon Selling Partner API getListingsItem response, formatted as YAML. "
                "This data comes from the API only (not from the language model).\n\n"
                f"```yaml\n{listings_yaml_blob}\n```"
            )

        if state.is_complete and state.final_answer:
            return state.final_answer
        if state.is_complete:
            return self._generate_final_response(state)

        return (
            f"Reached maximum iterations ({self._max_iterations}) without completing. "
            f"Partial state: {len(state.history)} iterations completed."
        )


def build_sp_api_react_agent(
    client: SPAPIClient,
    credentials: SPAPICredentials,
    llm: Any,
    *,
    max_iterations: int | None = None,
) -> SpApiReActAgent:
    """
    Create a SpApiReActAgent with read-only SP-API tools.

    Args:
        client: Shared SPAPIClient (LWA + rate limits + GET).
        credentials: Loaded credentials (seller_id, marketplace for listings).
        llm: LangChain-compatible chat model or callable.
        max_iterations: ReAct loop cap; default from env SP_API_MAX_ITERATIONS or 8.

    Returns:
        Configured SpApiReActAgent instance.
    """
    try:
        cap = (
            int(os.environ.get("SP_API_MAX_ITERATIONS", "8"))
            if max_iterations is None
            else int(max_iterations)
        )
    except (TypeError, ValueError):
        cap = 8
    tools: List[Any] = [
        SpApiGetOrdersTool(client),
        SpApiGetListingsTool(client, credentials),
    ]
    return SpApiReActAgent(llm=llm, tools=tools, max_iterations=cap)
