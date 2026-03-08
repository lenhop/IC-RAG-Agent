"""
Unit tests for UDS Intent Classifier.
"""

import pytest

from src.uds.intent_classifier import (
    UDSIntentClassifier,
    IntentResult,
    IntentDomain,
)


class TestIntentClassifierKeyword:
    """Tests for keyword-based classification (no LLM)."""

    @pytest.fixture
    def classifier(self):
        """Classifier without LLM - uses keyword fallback."""
        return UDSIntentClassifier(llm_client=None)

    def test_classify_sales(self, classifier):
        """Sales queries classify to sales domain."""
        result = classifier.classify("What were total sales in October?")
        assert result.primary_domain == IntentDomain.SALES
        assert "sales" in result.keywords or "revenue" in result.keywords
        assert result.confidence > 0
        assert "SalesTrendTool" in result.suggested_tools

    def test_classify_inventory(self, classifier):
        """Inventory queries classify to inventory domain."""
        result = classifier.classify("What products are low on stock?")
        assert result.primary_domain == IntentDomain.INVENTORY
        assert "stock" in result.keywords or "inventory" in result.keywords
        assert "InventoryAnalysisTool" in result.suggested_tools

    def test_classify_financial(self, classifier):
        """Financial queries classify to financial domain."""
        result = classifier.classify("What's our profit margin?")
        assert result.primary_domain == IntentDomain.FINANCIAL
        assert "FinancialSummaryTool" in result.suggested_tools

    def test_classify_product(self, classifier):
        """Product queries classify to product domain."""
        result = classifier.classify("Top 10 products by revenue")
        assert result.primary_domain == IntentDomain.PRODUCT
        assert "ProductPerformanceTool" in result.suggested_tools

    def test_classify_comparison(self, classifier):
        """Comparison queries classify to comparison domain."""
        result = classifier.classify("Compare week 1 vs week 2 sales")
        assert result.primary_domain == IntentDomain.COMPARISON
        assert "compare" in result.keywords or "vs" in result.keywords
        assert "ComparisonTool" in result.suggested_tools

    def test_classify_general(self, classifier):
        """Schema/table queries classify to general domain."""
        result = classifier.classify("What tables are available?")
        assert result.primary_domain == IntentDomain.GENERAL
        assert "ListTablesTool" in result.suggested_tools

    def test_classify_empty_returns_general(self, classifier):
        """Empty query returns general with zero confidence."""
        result = classifier.classify("")
        assert result.primary_domain == IntentDomain.GENERAL
        assert result.confidence == 0.0

    def test_classify_returns_intent_result(self, classifier):
        """Classify returns IntentResult with all fields."""
        result = classifier.classify("Show me revenue trends")
        assert isinstance(result, IntentResult)
        assert hasattr(result, "primary_domain")
        assert hasattr(result, "secondary_domains")
        assert hasattr(result, "confidence")
        assert hasattr(result, "keywords")
        assert hasattr(result, "suggested_tools")
        assert hasattr(result, "reasoning")

    def test_confidence_bounded(self, classifier):
        """Confidence is between 0 and 1."""
        for query in ["sales revenue", "inventory stock", "profit margin"]:
            result = classifier.classify(query)
            assert 0 <= result.confidence <= 1


class TestIntentClassifierLLM:
    """Tests for LLM-based classification (when LLM provided)."""

    def test_llm_fallback_on_error(self):
        """When LLM raises, falls back to keyword classification."""
        class FailingLLM:
            def generate(self, prompt):
                raise RuntimeError("LLM unavailable")

        classifier = UDSIntentClassifier(llm_client=FailingLLM())
        result = classifier.classify("What were sales in October?")
        assert result.primary_domain == IntentDomain.SALES
        assert "Keyword" in result.reasoning or "keyword" in result.reasoning

    def test_llm_success_parses_response(self):
        """LLM response is parsed correctly."""
        class MockLLM:
            def generate(self, prompt):
                return "sales"

        classifier = UDSIntentClassifier(llm_client=MockLLM())
        result = classifier.classify("Some query")
        assert result.primary_domain == IntentDomain.SALES
        assert result.confidence == 0.9

    def test_llm_invoke_interface(self):
        """LLM with invoke() works (LangChain style)."""
        class InvokeLLM:
            def invoke(self, prompt):
                return "inventory"

        classifier = UDSIntentClassifier(llm_client=InvokeLLM())
        result = classifier.classify("Stock levels")
        assert result.primary_domain == IntentDomain.INVENTORY


class TestIntentDomain:
    """Tests for IntentDomain enum."""

    def test_all_domains_defined(self):
        """All 6 domains exist."""
        assert IntentDomain.SALES.value == "sales"
        assert IntentDomain.INVENTORY.value == "inventory"
        assert IntentDomain.FINANCIAL.value == "financial"
        assert IntentDomain.PRODUCT.value == "product"
        assert IntentDomain.COMPARISON.value == "comparison"
        assert IntentDomain.GENERAL.value == "general"
