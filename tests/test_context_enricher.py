import pytest
from src.uds.context_enricher import ContextEnricher
from src.uds.intent_classifier import IntentResult, IntentDomain


def test_context_enricher_initialization():
    """Test context enricher initializes correctly."""
    enricher = ContextEnricher()
    assert enricher is not None
    # glossary and statistics should be loaded
    assert isinstance(enricher.glossary, dict)
    assert isinstance(enricher.statistics, dict)


def test_enrich_sales_context():
    """Test enriching context for sales query."""
    enricher = ContextEnricher()
    
    intent = IntentResult(
        primary_domain=IntentDomain.SALES,
        secondary_domains=[],
        confidence=0.9,
        keywords=['sales', 'revenue'],
        suggested_tools=['SalesTrendTool'],
        reasoning="Keyword match"
    )
    
    context = enricher.enrich("What were total sales?", intent)
    
    # Schema context should mention table
    assert 'amz_order' in context
    assert len(context) > 0


def test_get_metric_definition():
    """Test getting metric definition."""
    enricher = ContextEnricher()
    
    metric = enricher.get_metric_definition("Total Sales")
    
    assert 'definition' in metric
    assert 'sql' in metric


def test_token_budget():
    """Test context stays within token budget."""
    enricher = ContextEnricher()
    
    intent = IntentResult(
        primary_domain=IntentDomain.SALES,
        secondary_domains=[],
        confidence=0.9,
        keywords=[],
        suggested_tools=[],
        reasoning=""
    )
    
    context = enricher.enrich("test query", intent, max_tokens=100)
    
    # Rough check: 100 tokens ~= 400 chars
    assert len(context) <= 500


def test_get_term_definition():
    """Test looking up a glossary term."""
    enricher = ContextEnricher()
    term = enricher.get_term_definition("ASIN")
    assert term.get('full_name') == "Amazon Standard Identification Number"


def test_glossary_context_extraction():
    """Terms mentioned in query should appear in context."""
    enricher = ContextEnricher()
    intent = IntentResult(
        primary_domain=IntentDomain.PRODUCT,
        secondary_domains=[],
        confidence=0.5,
        keywords=[],
        suggested_tools=[],
        reasoning=""
    )
    context = enricher.enrich("Tell me about ASIN and SKU", intent, max_tokens=200)
    assert "ASIN" in context
    assert "SKU" in context
