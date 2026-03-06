from typing import Dict, Any, List
from src.uds.intent_classifier import IntentResult, IntentDomain
import json
import os

class ContextEnricher:
    """
    Enriches agent context with relevant information.
    Manages token budget to stay within LLM limits.
    """
    
    def __init__(
        self,
        schema_metadata_path: str = 'src/uds/uds_schema_metadata.json',
        glossary_path: str = 'src/uds/uds_business_glossary.json',
        statistics_path: str = 'src/uds/uds_statistics.json'
    ):
        """
        Initialize context enricher.
        
        Args:
            schema_metadata_path: Path to schema metadata
            glossary_path: Path to business glossary
            statistics_path: Path to data statistics
        """
        self.schema = self._load_json(schema_metadata_path)
        self.glossary = self._load_json(glossary_path)
        self.statistics = self._load_json(statistics_path)
        
    def _load_json(self, path: str) -> Dict[str, Any]:
        """Load JSON file."""
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
        return {}
    
    def enrich(
        self,
        query: str,
        intent: IntentResult,
        max_tokens: int = 2000
    ) -> str:
        """
        Build enriched context within token budget.
        
        Args:
            query: User query
            intent: Classified intent
            max_tokens: Maximum context tokens
            
        Returns:
            Enriched context string
        """
        context_parts = []
        
        # Priority 1: Schema for relevant tables
        schema_context = self._get_schema_context(intent)
        if schema_context:
            context_parts.append(("Schema Information:", schema_context))
        
        # Priority 2: Business glossary
        glossary_context = self._get_glossary_context(query)
        if glossary_context:
            context_parts.append(("Business Terminology:", glossary_context))
        
        # Priority 3: Data statistics
        stats_context = self._get_statistics_context(intent)
        if stats_context:
            context_parts.append(("Data Statistics:", stats_context))
        
        # Priority 4: Example queries
        examples_context = self._get_examples_context(intent)
        if examples_context:
            context_parts.append(("Similar Queries:", examples_context))
        
        # Build context within token budget
        return self._build_within_budget(context_parts, max_tokens)
    
    def _get_schema_context(self, intent: IntentResult) -> str:
        """
        Get schema info for relevant tables.
        
        Args:
            intent: Classified intent
            
        Returns:
            Schema context string
        """
        if not self.schema or 'tables' not in self.schema:
            return ""
        
        # Get relevant tables for domain
        relevant_tables = self._get_relevant_tables(intent.primary_domain)
        
        if not relevant_tables:
            return ""
        
        context_lines = []
        for table in relevant_tables[:3]:  # Limit to 3 tables
            if table in self.schema['tables']:
                table_info = self.schema['tables'][table]
                context_lines.append(f"- {table}: {table_info.get('description', 'N/A')}")
                
                # Add key columns
                if 'columns' in table_info:
                    # columns may be list of dicts; extract names
                    raw_cols = table_info['columns'][:5]
                    key_cols = []
                    for col in raw_cols:
                        if isinstance(col, dict) and 'name' in col:
                            key_cols.append(col['name'])
                        elif isinstance(col, str):
                            key_cols.append(col)
                        else:
                            key_cols.append(str(col))
                    context_lines.append(f"  Columns: {', '.join(key_cols)}")
        
        return "\n".join(context_lines)
    
    def _get_glossary_context(self, query: str) -> str:
        """
        Get relevant glossary terms.
        
        Args:
            query: User query
            
        Returns:
            Glossary context string
        """
        if not self.glossary or 'glossary' not in self.glossary:
            return ""
        
        query_lower = query.lower()
        relevant_terms = []
        
        # Find terms mentioned in query
        for term, info in self.glossary['glossary'].items():
            if term.lower() in query_lower:
                definition = info.get('definition', 'N/A')
                relevant_terms.append(f"- {term}: {definition}")
        
        if not relevant_terms:
            return ""
        
        return "\n".join(relevant_terms[:5])  # Limit to 5 terms
    
    def _get_statistics_context(self, intent: IntentResult) -> str:
        """
        Get data statistics.
        
        Args:
            intent: Classified intent
            
        Returns:
            Statistics context string
        """
        if not self.statistics:
            return ""
        
        # Get relevant stats for domain
        relevant_tables = self._get_relevant_tables(intent.primary_domain)
        
        context_lines = []
        for table in relevant_tables[:2]:  # Limit to 2 tables
            if table in self.statistics:
                stats = self.statistics[table]
                context_lines.append(f"- {table}: {stats.get('row_count', 0):,} rows")
                
                if 'date_range' in stats:
                    context_lines.append(f"  Date range: {stats['date_range']}")
        
        return "\n".join(context_lines)
    
    def _get_examples_context(self, intent: IntentResult) -> str:
        """
        Get example queries for domain.
        
        Args:
            intent: Classified intent
            
        Returns:
            Examples context string
        """
        if not self.glossary or 'common_questions' not in self.glossary:
            return ""
        
        domain = intent.primary_domain.value
        if domain in self.glossary['common_questions']:
            examples = self.glossary['common_questions'][domain][:3]  # 3 examples
            return "\n".join(f"- {ex}" for ex in examples)
        
        return ""
    
    def _get_relevant_tables(self, domain: IntentDomain) -> List[str]:
        """Get tables relevant to domain."""
        domain_tables = {
            IntentDomain.SALES: ['amz_order', 'amz_transaction'],
            IntentDomain.INVENTORY: ['amz_fba_inventory_all', 'amz_daily_inventory_ledger'],
            IntentDomain.FINANCIAL: ['amz_transaction', 'amz_statement', 'amz_fee'],
            IntentDomain.PRODUCT: ['amz_product', 'amz_order', 'amz_listing_item'],
            IntentDomain.COMPARISON: ['amz_order', 'amz_transaction'],
            IntentDomain.GENERAL: []
        }
        return domain_tables.get(domain, [])
    
    def _build_within_budget(
        self,
        context_parts: List[tuple],
        max_tokens: int
    ) -> str:
        """
        Build context within token budget.
        
        Args:
            context_parts: List of (title, content) tuples
            max_tokens: Maximum tokens
            
        Returns:
            Context string within budget
        """
        # Simple token estimation: ~4 chars per token
        max_chars = max_tokens * 4
        
        result_parts = []
        current_chars = 0
        
        for title, content in context_parts:
            part = f"\n{title}\n{content}\n"
            part_chars = len(part)
            
            if current_chars + part_chars <= max_chars:
                result_parts.append(part)
                current_chars += part_chars
            else:
                # Truncate this part to fit
                remaining = max_chars - current_chars
                if remaining > 100:  # Only add if meaningful space left
                    truncated = part[:remaining] + "..."
                    result_parts.append(truncated)
                break
        
        return "".join(result_parts)
    
    def get_metric_definition(self, metric_name: str) -> Dict[str, Any]:
        """
        Get definition for a metric.
        
        Args:
            metric_name: Name of metric
            
        Returns:
            Metric definition dict
        """
        if not self.glossary or 'metrics' not in self.glossary:
            return {}
        
        return self.glossary['metrics'].get(metric_name, {})
    
    def get_term_definition(self, term: str) -> Dict[str, Any]:
        """
        Get definition for a term.
        
        Args:
            term: Term to look up
            
        Returns:
            Term definition dict
        """
        if not self.glossary or 'glossary' not in self.glossary:
            return {}
        
        return self.glossary['glossary'].get(term, {})
