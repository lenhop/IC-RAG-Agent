"""
Intent Classification for UDS Agent.

Classifies user queries into business domains (sales, inventory, financial,
product, comparison, general) using LLM with few-shot examples or keyword fallback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class IntentDomain(str, Enum):
    """Business domains for UDS queries."""

    SALES = "sales"
    INVENTORY = "inventory"
    FINANCIAL = "financial"
    PRODUCT = "product"
    COMPARISON = "comparison"
    GENERAL = "general"


@dataclass
class IntentResult:
    """Result of intent classification."""

    primary_domain: IntentDomain
    secondary_domains: List[IntentDomain]
    confidence: float  # 0.0 to 1.0
    keywords: List[str]
    suggested_tools: List[str]
    reasoning: str


class UDSIntentClassifier:
    """
    Classifies user queries into business domains.

    Uses LLM with few-shot examples when available, keyword fallback otherwise.
    """

    INTENT_EXAMPLES = {
        IntentDomain.SALES: [
            "What were total sales in October?",
            "Show me daily revenue trends",
            "Which day had the highest sales?",
            "What's the average order value?",
            "Sales growth rate this month",
        ],
        IntentDomain.INVENTORY: [
            "What products are low on stock?",
            "Show me inventory levels for SKU-123",
            "Which items need reordering?",
            "Current stock levels",
            "Inventory turnover rate",
        ],
        IntentDomain.FINANCIAL: [
            "What's our profit margin?",
            "Show me fee breakdown",
            "Calculate net revenue after fees",
            "Total fees paid this month",
            "Profitability analysis",
        ],
        IntentDomain.PRODUCT: [
            "Top 10 products by revenue",
            "Which products are underperforming?",
            "Product performance comparison",
            "Best selling items",
            "Product rankings",
        ],
        IntentDomain.COMPARISON: [
            "Compare week 1 vs week 2 sales",
            "How did October compare to September?",
            "Product A vs Product B performance",
            "Period over period growth",
            "Marketplace comparison",
        ],
        IntentDomain.GENERAL: [
            "What tables are available?",
            "Describe the amz_order table",
            "How many orders do we have?",
            "Show me table relationships",
            "What data is available?",
        ],
    }

    DOMAIN_KEYWORDS = {
        IntentDomain.SALES: [
            "sales",
            "revenue",
            "orders",
            "sold",
            "gmv",
            "selling",
            "purchase",
            "transaction",
            "order value",
            "aov",
        ],
        IntentDomain.INVENTORY: [
            "inventory",
            "stock",
            "quantity",
            "units",
            "warehouse",
            "fba",
            "fulfillment",
            "reorder",
            "stockout",
            "sku",
        ],
        IntentDomain.FINANCIAL: [
            "profit",
            "margin",
            "fee",
            "cost",
            "expense",
            "financial",
            "revenue",
            "net",
            "gross",
            "profitability",
            "settlement",
        ],
        IntentDomain.PRODUCT: [
            "product",
            "item",
            "asin",
            "sku",
            "catalog",
            "listing",
            "performance",
            "ranking",
            "top",
            "best",
            "worst",
        ],
        IntentDomain.COMPARISON: [
            "compare",
            "vs",
            "versus",
            "difference",
            "growth",
            "change",
            "period",
            "week",
            "month",
            "year over year",
        ],
        IntentDomain.GENERAL: [
            "table",
            "schema",
            "describe",
            "list",
            "show",
            "available",
            "what",
            "how many",
            "database",
        ],
    }

    DOMAIN_TOOLS = {
        IntentDomain.SALES: ["SalesTrendTool", "ExecuteQueryTool", "CreateChartTool"],
        IntentDomain.INVENTORY: ["InventoryAnalysisTool", "ExecuteQueryTool"],
        IntentDomain.FINANCIAL: ["FinancialSummaryTool", "ExecuteQueryTool"],
        IntentDomain.PRODUCT: ["ProductPerformanceTool", "ExecuteQueryTool"],
        IntentDomain.COMPARISON: ["ComparisonTool", "ExecuteQueryTool"],
        IntentDomain.GENERAL: ["ListTablesTool", "DescribeTableTool", "SearchColumnsTool"],
    }

    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize intent classifier.

        Args:
            llm_client: Optional LLM client with generate() or invoke() method
        """
        self.llm = llm_client
        # optional cache layer supporting get_intent/set_intent
        self.cache = None

    def classify(self, query: str) -> IntentResult:
        """
        Classify user query into business domain.

        Args:
            query: User's natural language question

        Returns:
            IntentResult with classification
        """
        # normalize and validate input
        key = str(query).strip() if query is not None else ""
        if not key:
            return IntentResult(
                primary_domain=IntentDomain.GENERAL,
                secondary_domains=[],
                confidence=0.0,
                keywords=[],
                suggested_tools=self.DOMAIN_TOOLS[IntentDomain.GENERAL],
                reasoning="Empty query",
            )

        # check cache first
        if self.cache is not None:
            cached = self.cache.get_intent(key)
            if cached is not None:
                logger.debug("Intent cache hit for query")
                try:
                    return IntentResult(
                        primary_domain=IntentDomain(cached["primary_domain"]),
                        secondary_domains=[IntentDomain(d) for d in cached.get("secondary_domains", [])],
                        confidence=cached.get("confidence", 0.0),
                        keywords=cached.get("keywords", []),
                        suggested_tools=cached.get("suggested_tools", []),
                        reasoning=cached.get("reasoning", ""),
                    )
                except Exception:
                    # fall through to re-classify if reconstruction fails
                    pass

        # perform classification using LLM if available
        result: IntentResult
        if self.llm:
            try:
                result = self._llm_classify(key)
            except Exception as e:
                logger.warning("LLM classification failed: %s, falling back to keywords", e)
                result = self._keyword_classify(key)
        else:
            result = self._keyword_classify(key)

        # cache the result for future calls
        if self.cache is not None:
            try:
                self.cache.set_intent(key, {
                    "primary_domain": result.primary_domain.value,
                    "secondary_domains": [d.value for d in result.secondary_domains],
                    "confidence": result.confidence,
                    "keywords": result.keywords,
                    "suggested_tools": result.suggested_tools,
                    "reasoning": result.reasoning,
                })
            except Exception:
                pass

        return result

    def _llm_classify(self, query: str) -> IntentResult:
        """
        Use LLM with few-shot examples for classification.

        Args:
            query: User query

        Returns:
            IntentResult
        """
        prompt = self._build_classification_prompt(query)

        if hasattr(self.llm, "generate"):
            response = self.llm.generate(prompt)
        elif hasattr(self.llm, "invoke"):
            response = self.llm.invoke(prompt)
        else:
            raise AttributeError("LLM client must have generate() or invoke()")

        if isinstance(response, str):
            response_text = response
        else:
            response_text = getattr(response, "content", str(response))

        return self._parse_llm_response(response_text, query)

    def _keyword_classify(self, query: str) -> IntentResult:
        """
        Keyword-based classification fallback.

        Args:
            query: User query

        Returns:
            IntentResult
        """
        query_lower = query.lower()

        domain_scores: Dict[IntentDomain, int] = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            domain_scores[domain] = score

        primary = max(domain_scores, key=domain_scores.get)
        primary_score = domain_scores[primary]
        if primary_score == 0:
            primary = IntentDomain.GENERAL
            primary_score = 1

        secondary = [
            d for d, score in domain_scores.items()
            if score > 0 and d != primary
        ]

        total_score = sum(domain_scores.values())
        confidence = primary_score / total_score if total_score > 0 else 0.5
        confidence = min(1.0, max(0.0, confidence))

        keywords = [
            kw for kw in self.DOMAIN_KEYWORDS[primary]
            if kw in query_lower
        ]

        return IntentResult(
            primary_domain=primary,
            secondary_domains=secondary,
            confidence=confidence,
            keywords=keywords,
            suggested_tools=self.DOMAIN_TOOLS[primary],
            reasoning=f"Keyword match: {', '.join(keywords)}" if keywords else "No keyword match, default to highest scoring domain",
        )

    def _build_classification_prompt(self, query: str) -> str:
        """Build few-shot prompt for LLM."""
        examples = []
        for domain, queries in self.INTENT_EXAMPLES.items():
            for q in queries[:2]:
                examples.append(f"Query: {q}\nDomain: {domain.value}")

        prompt = f"""Classify the following query into one of these business domains:
- sales: Revenue, orders, sales trends
- inventory: Stock levels, inventory management
- financial: Profitability, fees, costs
- product: Product performance, rankings
- comparison: Period/product comparisons
- general: Schema exploration, data questions

Examples:
{chr(10).join(examples)}

Query: {query}
Domain:"""
        return prompt

    def _parse_llm_response(self, response: str, query: str) -> IntentResult:
        """Parse LLM response into IntentResult."""
        key = str(query).strip()
        if not key:
            return self._keyword_classify(query)
        
        response_lower = response.lower()
        primary = None
        for domain in IntentDomain:
            if domain.value in response_lower:
                primary = domain
                break
        
        if not primary:
            return self._keyword_classify(query)

        # return classification result from LLM
        result = IntentResult(
            primary_domain=primary,
            secondary_domains=[],
            confidence=0.9,
            keywords=[],
            suggested_tools=self.DOMAIN_TOOLS.get(primary, []),
            reasoning="LLM classification",
        )

        # store in cache if available
        if self.cache is not None:
            try:
                self.cache.set_intent(key, {
                    "primary_domain": result.primary_domain.value,
                    "secondary_domains": [d.value for d in result.secondary_domains],
                    "confidence": result.confidence,
                    "keywords": result.keywords,
                    "suggested_tools": result.suggested_tools,
                    "reasoning": result.reasoning,
                })
            except Exception:
                pass

        return result
