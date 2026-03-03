"""
Seller Operations Agent - ReActAgent with SP-API tools and session memory.
"""
import time
from typing import Any, List, Optional, Tuple

from src.agent import ReActAgent
from src.agent.models import AgentState

from .memory import ConversationMemory
from .sp_api_client import SPAPIClient
from .tools import (
    ProductCatalogTool,
    InventoryTool,
    ListOrdersTool,
    OrderDetailsTool,
    ListShipmentsTool,
    CreateShipmentTool,
    FBAFeeTool,
    FBAEligibilityTool,
    FinancialsTool,
    ReportRequestTool,
)


def _invoke_llm(llm: Any, prompt: str) -> str:
    if callable(llm) and not hasattr(llm, "invoke"):
        return llm(prompt)
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


class SellerOperationsAgent(ReActAgent):
    """ReActAgent with 10 SP-API tools, intent classification, and Redis memory."""

    def __init__(
        self,
        llm: Any,
        sp_api_client: SPAPIClient,
        memory: ConversationMemory,
        max_iterations: int = 15,
        logger: Optional[Any] = None,
    ):
        tools = [
            ProductCatalogTool(sp_api_client),
            InventoryTool(sp_api_client),
            ListOrdersTool(sp_api_client),
            OrderDetailsTool(sp_api_client),
            ListShipmentsTool(sp_api_client),
            CreateShipmentTool(sp_api_client),
            FBAFeeTool(sp_api_client),
            FBAEligibilityTool(sp_api_client),
            FinancialsTool(sp_api_client),
            ReportRequestTool(sp_api_client),
        ]
        super().__init__(llm=llm, tools=tools, max_iterations=max_iterations, logger=logger)
        self._memory = memory
        self._sp_api_client = sp_api_client

    def _classify_intent(self, query: str) -> str:
        """Classify query as query, action, or report."""
        prompt = f'''Classify the user query into exactly one of: query, action, report.
- query: read-only lookup (catalog, inventory, orders, financials)
- action: write/mutation (create shipment, etc.)
- report: async report request
Query: "{query}"
Reply with only: query, action, or report'''
        result = _invoke_llm(self._llm, prompt).strip().lower()
        intent = "query"
        if "action" in result:
            intent = "action"
        elif "report" in result:
            intent = "report"
        if self._logger:
            self._logger.log_thought(0, f"Intent: {intent}")
        return intent

    def _run_with_state(self, query: str) -> Tuple[str, AgentState]:
        """Run ReAct loop and return (result, state)."""
        start_time = time.time()
        state = AgentState(query=query)
        for _ in range(self._max_iterations):
            if state.is_complete:
                break
            state = self.step(state)
        execution_time = time.time() - start_time
        if self._logger:
            self._logger.log_run_complete(state, execution_time)
        if state.is_complete and state.final_answer:
            return state.final_answer, state
        if state.is_complete:
            prompt = self._build_prompt(state)
            prompt += "\n\nBased on the above, provide a concise Final Answer summarizing the result."
            return _invoke_llm(self._llm, prompt), state
        return (
            f"Reached maximum iterations ({self._max_iterations}) without completing. "
            f"Partial state: {len(state.history)} iterations completed.",
            state,
        )

    def query(self, query: str, session_id: str) -> str:
        """Process query with session history, save turn to memory."""
        history = self._memory.get_history(session_id, last_n=10)
        enriched = query
        if history:
            ctx = "\n".join([f"Q: {h['query']}\nA: {h['response'][:200]}..." for h in history[-3:]])
            enriched = f"Previous context:\n{ctx}\n\nCurrent query: {query}"
        result, state = self._run_with_state(enriched)
        self._memory.save_turn(session_id, query, result, state)
        return result

    def run_streaming(self, query: str):
        """Generator yielding SSE chunks per thought/observation as they happen."""
        state = AgentState(query=query)
        for _ in range(self._max_iterations):
            if state.is_complete:
                final = state.final_answer or ""
                prompt = self._build_prompt(state)
                prompt += "\n\nBased on the above, provide a concise Final Answer summarizing the result."
                final = _invoke_llm(self._llm, prompt) if not state.final_answer else state.final_answer
                yield {"type": "final", "response": final}
                return
            state = self.step(state)
            if state.history:
                thought, action, obs = state.history[-1]
                if thought:
                    yield {"type": "thought", "content": thought}
                obs_content = str(obs.output) if obs.success and obs.output is not None else str(obs.error or "")
                if action and action.tool_name:
                    yield {"type": "observation", "content": obs_content}
        yield {"type": "final", "response": f"Reached max iterations ({self._max_iterations})."}
