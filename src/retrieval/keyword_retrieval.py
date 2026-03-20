"""
Keyword Retrieval — Public API Module (公共接口模块)

Public API:
  KeywordMatchResult   — 匹配结果数据类
  KeywordRetrieval     — 三阶段匹配器（字典精确 → 短语包含 → 正则），由调用方注入规则行
  LoadKeywordRule      — 从目录加载 dict / YAML / 正则 CSV 的公共类方法
  keyword_retrieve     — 一次性便捷函数（注入规则行，无状态）

Workflow (match order):
  dict exact → for-loop phrase in query → regex
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API — rule loading (class methods)
# ---------------------------------------------------------------------------


class LoadKeywordRule:
    """
    Public loaders for keyword rule files under a single data directory.

    Expected files (optional): dict_sentences.csv, frequent_variable_sentences.yml,
    regular_rules.csv.
    """

    # YAML top-level keys in processing order; amazon_business maps to amazon_docs
    FOR_LOOP_YAML_KEYS: Tuple[str, ...] = ("sp_api", "uds", "amazon_business")

    @staticmethod
    def _map_yaml_workflow_key(yaml_key: str) -> str:
        """Map YAML workflow key to gateway workflow name."""
        if yaml_key == "amazon_business":
            return "amazon_docs"
        return yaml_key

    @classmethod
    def load_dict_sentences_csv(cls, data_dir: Path) -> List[Tuple[str, str]]:
        """
        Load dict_sentences.csv: (canonical_query lower, workflow).

        Args:
            data_dir: Directory containing dict_sentences.csv.

        Returns:
            List of (canonical lower, workflow); empty if file missing or on read error.
        """
        path = data_dir / "dict_sentences.csv"
        rows: List[Tuple[str, str]] = []
        if not path.is_file():
            return rows
        try:
            with path.open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    return rows
                for row in reader:
                    try:
                        c = (row.get("canonical_query") or "").strip()
                        w = (row.get("workflow") or "").strip()
                        if c and w:
                            rows.append((c.lower(), w))
                    except (TypeError, AttributeError):
                        continue
        except OSError as exc:
            logger.debug("dict_sentences read failed: %s", exc)
        return rows

    @classmethod
    def load_frequent_variable_yml(cls, data_dir: Path) -> List[Tuple[str, str, str]]:
        """
        Load frequent_variable_sentences.yml as flat (phrase_lower, workflow, intent_name).

        Workflow order follows FOR_LOOP_YAML_KEYS; phrases sorted by length descending per key.

        Args:
            data_dir: Directory containing frequent_variable_sentences.yml.

        Returns:
            Flat phrase rows; empty if file missing, PyYAML absent, or parse error.
        """
        path = data_dir / "frequent_variable_sentences.yml"
        result: List[Tuple[str, str, str]] = []
        if not path.is_file():
            return result
        if yaml is None:
            logger.warning("PyYAML not installed; skipping frequent_variable_sentences.yml")
            return result
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("YAML load failed for %s: %s", path, exc)
            return result
        if not isinstance(data, dict):
            return result

        for yaml_key in cls.FOR_LOOP_YAML_KEYS:
            workflow = cls._map_yaml_workflow_key(yaml_key)
            intent_name = workflow
            phrases = data.get(yaml_key)
            if not isinstance(phrases, list):
                continue
            cleaned: List[str] = []
            for item in phrases:
                if isinstance(item, str) and item.strip():
                    cleaned.append(item.strip())
            cleaned.sort(key=len, reverse=True)
            for phrase in cleaned:
                result.append((phrase.lower(), workflow, intent_name))

        return result

    @classmethod
    def load_regular_rules_csv(cls, data_dir: Path) -> List[Tuple[re.Pattern, str, str]]:
        """
        Load regular_rules.csv: compiled pattern, workflow, intent_name.

        Args:
            data_dir: Directory containing regular_rules.csv.

        Returns:
            List of (compiled pattern, workflow, intent_name); invalid regex rows skipped.
        """
        path = data_dir / "regular_rules.csv"
        out: List[Tuple[re.Pattern, str, str]] = []
        if not path.is_file():
            return out
        try:
            with path.open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    return out
                for row in reader:
                    pat_raw = ""
                    try:
                        pat_raw = (row.get("pattern") or "").strip()
                        wf = (row.get("workflow") or "").strip()
                        intent = (row.get("intent_name") or "").strip()
                        if not pat_raw or not wf or not intent:
                            continue
                        compiled = re.compile(pat_raw, re.IGNORECASE)
                        out.append((compiled, wf, intent))
                    except re.error as exc:
                        logger.debug("Skip invalid regex %r: %s", pat_raw, exc)
                        continue
        except OSError as exc:
            logger.debug("regular_rules read failed: %s", exc)
        return out


# ---------------------------------------------------------------------------
# Public API — matching
# ---------------------------------------------------------------------------


@dataclass
class KeywordMatchResult:
    """Result of a single keyword/regex match (part of public API)."""

    workflow: str
    intent_name: str
    confidence: str = "high"
    source: str = "keyword"


class KeywordRetrieval:
    """
    Stateless three-stage matcher using caller-supplied rule rows.

    Stages:
      1. dict_rows: (canonical_query_lower, workflow) — full-string equality vs query (case-insensitive)
      2. for_loop_rows: (phrase_lower, workflow, intent_name) — phrase substring in query
      3. regex_rows: (compiled_pattern, workflow, intent_name) — pattern.search(query)
    """

    def __init__(
        self,
        dict_rows: List[Tuple[str, str]],
        for_loop_rows: List[Tuple[str, str, str]],
        regex_rows: List[Tuple[re.Pattern, str, str]],
    ) -> None:
        """
        Args:
            dict_rows: Exact-match entries (canonical lower, workflow).
            for_loop_rows: Phrase containment entries (phrase lower, workflow, intent_name).
            regex_rows: Regex entries (compiled pattern, workflow, intent_name).
        """
        # Defensive copies so external mutation does not affect matching.
        self._dict_rows: List[Tuple[str, str]] = list(dict_rows)
        self._for_loop_rows: List[Tuple[str, str, str]] = list(for_loop_rows)
        self._regex_rows: List[Tuple[re.Pattern, str, str]] = list(regex_rows)

    def match(self, query: str) -> Optional[KeywordMatchResult]:
        """
        Match in order: dict exact → for-loop phrase → regex.

        Args:
            query: User query text.

        Returns:
            KeywordMatchResult if any stage matches, else None.
        """
        if not query or not query.strip():
            return None
        q = query.strip()
        q_lower = q.lower()

        # Stage 1: dict exact match (case-insensitive full string)
        for canonical_lower, workflow in self._dict_rows:
            if q_lower == canonical_lower:
                return KeywordMatchResult(
                    workflow=workflow,
                    intent_name=workflow,
                    confidence="high",
                    source="keyword",
                )

        # Stage 2: phrase contained in query
        for phrase_lower, workflow, intent_name in self._for_loop_rows:
            if phrase_lower in q_lower:
                return KeywordMatchResult(
                    workflow=workflow,
                    intent_name=intent_name,
                    confidence="high",
                    source="keyword",
                )

        # Stage 3: regex
        for pattern, workflow, intent_name in self._regex_rows:
            if pattern.search(q):
                return KeywordMatchResult(
                    workflow=workflow,
                    intent_name=intent_name,
                    confidence="high",
                    source="keyword",
                )

        return None


def keyword_retrieve(
    query: str,
    dict_rows: List[Tuple[str, str]],
    for_loop_rows: List[Tuple[str, str, str]],
    regex_rows: List[Tuple[re.Pattern, str, str]],
) -> Optional[KeywordMatchResult]:
    """
    One-shot match using injected rule rows (no file I/O).

    Args:
        query: User query text.
        dict_rows: Exact-match rows.
        for_loop_rows: Phrase containment rows.
        regex_rows: Compiled regex rows.

    Returns:
        KeywordMatchResult or None.
    """
    return KeywordRetrieval(dict_rows, for_loop_rows, regex_rows).match(query)
