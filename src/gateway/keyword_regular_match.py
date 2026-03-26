"""
Shared keyword / regex matching for Route LLM Layer-1 style rules.

Business layout (three datasets, one data directory)
----------------------------------------------------
1. **clear_sentence** — ``amazon_business_intent_sentence.csv`` (``sentence``, ``workflow``).
   High-confidence whitelist for clarification **L1**; reusable by rewrite / intent L1.

2. **regular_patterns** — ``regular_patterns.csv`` (``pattern``, ``workflow``, ``source``, ``example``).
   UDS-oriented query-shape regexes (order / listing).

3. **clarification_signals** — ``clarification_signals.csv`` (L2 tiers A/B + **exclusion**).

Environment (clarification step only)
-------------------------------------
- ``GATEWAY_CLARIFICATION_LAYER1_L2_ENABLED`` — enables §2.3 L1+L2 gate before L3 LLM.
  **Documentation name:** ``GATEWAY_CLARIFICATION_LAYER1&2_ENABLED`` — ``&`` is invalid in
  Unix env keys, so the real variable uses ``_L1_L2_``.

- ``GATEWAY_CLARIFICATION_LAYER3_FORCE`` — always run L3 LLM (debug / incident).

Optional: ``GATEWAY_ROUTE_RULES_DATA_DIR`` — directory containing the three CSV files
(default: ``<repo>/external/IC-Self-Study/data``).

Classes
-------
- ``RouteRulesPathResolver`` — repo root + CSV directory resolution.
- ``ClarificationLayerEnv`` — clarification-related env flags.
- ``RouteRulesMatcher`` — load + match all three datasets.
- ``ClarificationL3SkipEvaluator`` — §2.3 skip-L3 decision from matcher state.
- ``DefaultRouteRulesMatcherCache`` — process-wide lazy singleton matcher.

Module-level functions remain as thin wrappers for stable imports.
"""

from __future__ import annotations

import csv
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# --- Env key constants (canonical names; see docstring for L1&2 display naming) ---
ENV_CLARIFICATION_LAYER1_L2_ENABLED = "GATEWAY_CLARIFICATION_LAYER1_L2_ENABLED"
ENV_CLARIFICATION_LAYER3_FORCE = "GATEWAY_CLARIFICATION_LAYER3_FORCE"
ENV_ROUTE_RULES_DATA_DIR = "GATEWAY_ROUTE_RULES_DATA_DIR"

CSV_CLEAR_SENTENCE = "amazon_business_intent_sentence.csv"
CSV_REGULAR_PATTERNS = "regular_patterns.csv"
CSV_CLARIFICATION_SIGNALS = "clarification_signals.csv"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


class RouteRulesPathResolver:
    """Resolve repository root and IC-Self-Study CSV directory paths."""

    @staticmethod
    def repo_root() -> Path:
        """IC-RAG-Agent root (``src/gateway/`` → parents[2])."""
        return Path(__file__).resolve().parents[2]

    @staticmethod
    def resolve_data_dir() -> Path:
        """
        Directory that holds the three CSV files.

        Precedence: ``GATEWAY_ROUTE_RULES_DATA_DIR``, else ``external/IC-Self-Study/data``.
        """
        raw = (os.getenv(ENV_ROUTE_RULES_DATA_DIR) or "").strip()
        if raw:
            p = Path(raw).expanduser()
            if not p.is_absolute():
                p = (RouteRulesPathResolver.repo_root() / p).resolve()
            else:
                p = p.resolve()
            return p
        return (
            RouteRulesPathResolver.repo_root()
            / "external"
            / "IC-Self-Study"
            / "data"
        ).resolve()


def default_rules_data_dir() -> Path:
    """Backward-compatible alias for ``RouteRulesPathResolver.resolve_data_dir``."""
    return RouteRulesPathResolver.resolve_data_dir()


class ClarificationLayerEnv:
    """Clarification fast-path feature flags (L1+L2 gate and L3 force)."""

    @staticmethod
    def is_layer1_l2_enabled() -> bool:
        """True when L1+L2 rule path may skip L3 (scheme §2.3)."""
        v = (os.getenv(ENV_CLARIFICATION_LAYER1_L2_ENABLED) or "").strip().lower()
        return v in _TRUE_VALUES

    @staticmethod
    def is_layer3_force() -> bool:
        """True when every request must use L3 LLM clarification."""
        v = (os.getenv(ENV_CLARIFICATION_LAYER3_FORCE) or "").strip().lower()
        return v in _TRUE_VALUES


def is_clarification_layer1_l2_enabled() -> bool:
    return ClarificationLayerEnv.is_layer1_l2_enabled()


def is_clarification_layer3_force() -> bool:
    return ClarificationLayerEnv.is_layer3_force()


@dataclass(frozen=True)
class ClearSentenceHit:
    """Single whitelist match from clear_sentence dataset."""

    sentence: str
    workflow: str


@dataclass(frozen=True)
class RegularPatternHit:
    """One regex match from regular_patterns dataset."""

    pattern: str
    workflow: str
    source: str
    example: str
    match: re.Match[str]


@dataclass(frozen=True)
class ClarificationSignalHit:
    """Matched clarification signal row (tier A / B / exclusion)."""

    rule_id: str
    tier: str
    signal_category: str
    pattern: str


@dataclass
class RouteRulesMatcher:
    """
    In-memory indexes: clear_sentence, regular_patterns, clarification_signals.

    Built once per load; matching is read-only after construction.
    """

    data_dir: Path
    _clear_by_norm: Dict[str, ClearSentenceHit] = field(repr=False, default_factory=dict)
    _regular: List[Tuple[re.Pattern[str], Dict[str, str]]] = field(
        repr=False, default_factory=list
    )
    _clar_ambiguous: List[Tuple[re.Pattern[str], Dict[str, str]]] = field(
        repr=False, default_factory=list
    )
    _clar_exclusion: List[Tuple[re.Pattern[str], Dict[str, str]]] = field(
        repr=False, default_factory=list
    )

    @classmethod
    def load(cls, data_dir: Optional[Path] = None) -> RouteRulesMatcher:
        """
        Load all three CSVs from ``data_dir``.

        Raises:
            FileNotFoundError: If a required CSV is missing.
            ValueError: If columns are wrong or a regex is invalid.
        """
        base = Path(data_dir) if data_dir is not None else RouteRulesPathResolver.resolve_data_dir()
        base = base.resolve()
        self = cls(data_dir=base)
        self._load_clear_sentence(base / CSV_CLEAR_SENTENCE)
        self._load_regular_patterns(base / CSV_REGULAR_PATTERNS)
        self._load_clarification_signals(base / CSV_CLARIFICATION_SIGNALS)
        logger.info(
            "RouteRulesMatcher loaded from %s (clear=%d regular=%d clar_ambig=%d clar_excl=%d)",
            base,
            len(self._clear_by_norm),
            len(self._regular),
            len(self._clar_ambiguous),
            len(self._clar_exclusion),
        )
        return self

    def _load_clear_sentence(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"clear_sentence CSV not found: {path}")
        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "sentence" not in reader.fieldnames:
                raise ValueError(f"{path}: missing sentence column")
            if "workflow" not in reader.fieldnames:
                raise ValueError(f"{path}: missing workflow column")
            for row in reader:
                sent = (row.get("sentence") or "").strip()
                wf = (row.get("workflow") or "").strip()
                if not sent or not wf:
                    continue
                key = _normalize_clear_sentence_key(sent)
                if key not in self._clear_by_norm:
                    self._clear_by_norm[key] = ClearSentenceHit(sentence=sent, workflow=wf)

    def _load_regular_patterns(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"regular_patterns CSV not found: {path}")
        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"{path}: empty CSV")
            required = {"pattern", "workflow", "source", "example"}
            if not required.issubset(set(reader.fieldnames)):
                raise ValueError(f"{path}: need columns {sorted(required)}")
            for idx, row in enumerate(reader, start=2):
                pat_s = (row.get("pattern") or "").strip()
                if not pat_s:
                    continue
                try:
                    cre = re.compile(pat_s)
                except re.error as exc:
                    raise ValueError(f"{path} row {idx}: invalid regex: {exc}") from exc
                self._regular.append((cre, {k: (row.get(k) or "").strip() for k in row}))

    def _load_clarification_signals(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"clarification_signals CSV not found: {path}")
        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "pattern" not in reader.fieldnames:
                raise ValueError(f"{path}: missing pattern column")
            if "tier" not in reader.fieldnames:
                raise ValueError(f"{path}: missing tier column")
            for idx, row in enumerate(reader, start=2):
                pat_s = (row.get("pattern") or "").strip()
                tier = (row.get("tier") or "").strip()
                if not pat_s or not tier:
                    continue
                try:
                    cre = re.compile(pat_s)
                except re.error as exc:
                    raise ValueError(f"{path} row {idx}: invalid regex: {exc}") from exc
                row_norm = {k: (row.get(k) or "").strip() for k in row}
                tlow = tier.lower()
                if tlow == "exclusion":
                    self._clar_exclusion.append((cre, row_norm))
                elif tlow in ("a", "b"):
                    self._clar_ambiguous.append((cre, row_norm))
                else:
                    logger.warning(
                        "Skipping clarification row with unknown tier=%r at %s:%s",
                        tier,
                        path,
                        idx,
                    )

    # --- clear_sentence (Layer-1 whitelist) --------------------------------
    def match_clear_sentence(self, query: str) -> Optional[ClearSentenceHit]:
        """Exact lookup after strip + case-fold on key; CSV casing preserved in hit."""
        key = _normalize_clear_sentence_key(query)
        if not key:
            return None
        return self._clear_by_norm.get(key)

    def iter_clear_sentence_workflows(self) -> Sequence[str]:
        return sorted({h.workflow for h in self._clear_by_norm.values()})

    # --- regular_patterns (UDS query-shape) ---------------------------------
    def match_regular_patterns(self, text: str) -> List[RegularPatternHit]:
        if not text:
            return []
        out: List[RegularPatternHit] = []
        for cre, row in self._regular:
            m = cre.search(text)
            if m:
                out.append(
                    RegularPatternHit(
                        pattern=row.get("pattern", ""),
                        workflow=row.get("workflow", ""),
                        source=row.get("source", ""),
                        example=row.get("example", ""),
                        match=m,
                    )
                )
        return out

    def match_first_regular_pattern(self, text: str) -> Optional[RegularPatternHit]:
        hits = self.match_regular_patterns(text)
        return hits[0] if hits else None

    # --- clarification_signals (L2 + exclusion) ------------------------------
    def match_clarification_exclusions(self, text: str) -> List[ClarificationSignalHit]:
        return _match_clar_tier_list(text, self._clar_exclusion)

    def match_clarification_ambiguous(
        self,
        text: str,
        *,
        has_conversation_history: bool,
    ) -> List[ClarificationSignalHit]:
        if not text:
            return []
        out: List[ClarificationSignalHit] = []
        for cre, row in self._clar_ambiguous:
            rh = (row.get("requires_history") or "").strip().lower() in _TRUE_VALUES
            if rh and not has_conversation_history:
                continue
            if cre.search(text):
                out.append(_clar_hit_from_row(row))
        return out


def _normalize_clear_sentence_key(query: str) -> str:
    return (query or "").strip().casefold()


def _clar_hit_from_row(row: Dict[str, str]) -> ClarificationSignalHit:
    return ClarificationSignalHit(
        rule_id=row.get("rule_id", ""),
        tier=row.get("tier", ""),
        signal_category=row.get("signal_category", ""),
        pattern=row.get("pattern", ""),
    )


def _match_clar_tier_list(
    text: str,
    rules: Sequence[Tuple[re.Pattern[str], Dict[str, str]]],
) -> List[ClarificationSignalHit]:
    if not text:
        return []
    out: List[ClarificationSignalHit] = []
    for cre, row in rules:
        if cre.search(text):
            out.append(_clar_hit_from_row(row))
    return out


class ClarificationL3SkipEvaluator:
    """
    Implements ``route_llm_optimization_scheme_new.md`` §2.3 / §2.4 branching.

    Decides whether the clarification step may skip the L3 LLM call.
    """

    @staticmethod
    def effective_l2_ambiguity(
        text: str,
        *,
        has_conversation_history: bool,
        matcher: RouteRulesMatcher,
    ) -> bool:
        """
        True when L2 signals say the query is context-dependent (needs L3).

        Tier A/B match, unless an **exclusion** row also matches (comparison-style queries).
        """
        amb = matcher.match_clarification_ambiguous(
            text, has_conversation_history=has_conversation_history
        )
        if not amb:
            return False
        if matcher.match_clarification_exclusions(text):
            return False
        return True

    @staticmethod
    def skip_l3_decision(
        query: str,
        *,
        has_conversation_history: bool,
        matcher: RouteRulesMatcher,
    ) -> Tuple[bool, Optional[str]]:
        """
        Returns:
            (True, ``l1_skip`` | ``l2_skip``) — skip L3.
            (False, None) — run L3 LLM.
        """
        q = (query or "").strip()
        if not q:
            return True, "l2_skip"

        l1 = matcher.match_clear_sentence(q)
        l2_eff = ClarificationL3SkipEvaluator.effective_l2_ambiguity(
            q, has_conversation_history=has_conversation_history, matcher=matcher
        )

        if has_conversation_history:
            if l2_eff:
                return False, None
            if l1:
                return True, "l1_skip"
            return False, None

        if l1:
            return True, "l1_skip"
        if not l2_eff:
            return True, "l2_skip"
        return False, None


def evaluate_l2_effective_ambiguity(
    text: str,
    *,
    has_conversation_history: bool,
    matcher: RouteRulesMatcher,
) -> bool:
    return ClarificationL3SkipEvaluator.effective_l2_ambiguity(
        text,
        has_conversation_history=has_conversation_history,
        matcher=matcher,
    )


def clarification_skip_l3_decision(
    query: str,
    *,
    has_conversation_history: bool,
    matcher: RouteRulesMatcher,
) -> Tuple[bool, Optional[str]]:
    return ClarificationL3SkipEvaluator.skip_l3_decision(
        query,
        has_conversation_history=has_conversation_history,
        matcher=matcher,
    )


class ClarificationLayer12Gate:
    """
    Clarification-step orchestration: L1 (clear_sentence) + L2 (clarification_signals) vs L3 LLM.

    Call ``try_resolve_without_l3`` from ``check_ambiguity`` before invoking the clarification LLM.
    """

    @staticmethod
    def try_resolve_without_l3(
        query_stripped: str,
        *,
        has_conversation_history: bool,
    ) -> Optional[Dict[str, Any]]:
        """
        If rules allow skipping L3, return a ``check_ambiguity``-compatible dict.

        Returns:
            Dict with ``needs_clarification=False`` and ``clarification_path`` in {l1_skip, l2_skip},
            or ``None`` when the caller must run L3 LLM (force flag, fast path off, no safe skip, or load error).
        """
        if ClarificationLayerEnv.is_layer3_force():
            logger.info(
                "ClarificationLayer12Gate: %s=true -> L3 LLM",
                ENV_CLARIFICATION_LAYER3_FORCE,
            )
            return None

        if not ClarificationLayerEnv.is_layer1_l2_enabled():
            return None

        matcher = DefaultRouteRulesMatcherCache.get()
        if matcher is None:
            logger.warning(
                "ClarificationLayer12Gate: RouteRulesMatcher unavailable -> L3 LLM fallback",
            )
            return None

        skip, path = ClarificationL3SkipEvaluator.skip_l3_decision(
            query_stripped,
            has_conversation_history=has_conversation_history,
            matcher=matcher,
        )
        if skip:
            logger.info("ClarificationLayer12Gate: skip L3 via %s", path)
            return {
                "needs_clarification": False,
                "clarification_backend": None,
                "clarification_path": path,
            }
        return None


class DefaultRouteRulesMatcherCache:
    """Thread-safe lazy singleton for ``RouteRulesMatcher.load()``."""

    _lock = threading.Lock()
    _cached: Optional[RouteRulesMatcher] = None
    _error: Optional[Exception] = None

    @classmethod
    def get(cls) -> Optional[RouteRulesMatcher]:
        with cls._lock:
            if cls._cached is not None:
                return cls._cached
            if cls._error is not None:
                return None
            try:
                cls._cached = RouteRulesMatcher.load()
                return cls._cached
            except Exception as exc:
                cls._error = exc
                logger.error(
                    "Failed to load RouteRulesMatcher from %s: %s",
                    RouteRulesPathResolver.resolve_data_dir(),
                    exc,
                    exc_info=True,
                )
                return None

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._cached = None
            cls._error = None


def get_default_route_rules_matcher() -> Optional[RouteRulesMatcher]:
    return DefaultRouteRulesMatcherCache.get()


def reset_default_route_rules_matcher_cache() -> None:
    DefaultRouteRulesMatcherCache.reset()
