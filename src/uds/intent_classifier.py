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

    def classify(self, query: str) -> IntentResult:
        """
        Classify user query into business domain.

        Args:
            query: User's natural language question

        Returns:
            IntentResult with classification
        """
        if not query or not str(query).strip():
            return IntentResult(
                primary_domain=IntentDomain.GENERAL,
                secondary_domains=[],
                confidence=0.0,
                keywords=[],
                suggested_tools=self.DOMAIN_TOOLS[IntentDomain.GENERAL],
                reasoning="Empty query",
            )

        if self.llm:
            try:
                return self._llm_classify(str(query).strip())
            except Exception as e:
                logger.warning("LLM classification failed: %s, falling back to keywords", e)

        return self._keyword_classify(str(query).strip())

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
        response_lower = response.lower().strip()

        primary = None
        for domain in IntentDomain:
            if domain.value in response_lower:
                primary = domain
                break

        if not primary:
            return self._keyword_classify(query)

        return IntentResult(
            primary_domain=primary,
            secondary_domains=[],
            confidence=0.9,
            keywords=[],
            suggested_tools=self.DOMAIN_TOOLS[primary],
            reasoning="LLM classification",
        )
