from .schema_tools import (
    ListTablesTool,
    DescribeTableTool,
    GetTableRelationshipsTool,
    SearchColumnsTool,
)
from .visualization_tools import (
    CreateChartTool,
    CreateDashboardTool,
    ExportVisualizationTool,
)
from .analysis_tools import (
    SalesTrendTool,
    InventoryAnalysisTool,
    ProductPerformanceTool,
    FinancialSummaryTool,
    ComparisonTool,
)
from .query_tools import (
    GenerateSQLTool,
    ExecuteQueryTool,
    ValidateQueryTool,
    ExplainQueryTool,
)

__all__ = [
    # Schema tools
    "ListTablesTool",
    "DescribeTableTool",
    "GetTableRelationshipsTool",
    "SearchColumnsTool",
    # Visualization tools
    "CreateChartTool",
    "CreateDashboardTool",
    "ExportVisualizationTool",
    # Analysis tools
    "SalesTrendTool",
    "InventoryAnalysisTool",
    "ProductPerformanceTool",
    "FinancialSummaryTool",
    "ComparisonTool",
    # Query tools
    "GenerateSQLTool",
    "ExecuteQueryTool",
    "ValidateQueryTool",
    "ExplainQueryTool",
]


class UDSToolRegistry:
    """Registry for all UDS tools."""
    
    @staticmethod
    def get_all_tools():
        """Get all available tools."""
        return [
            # Schema tools
            ListTablesTool(),
            DescribeTableTool(),
            GetTableRelationshipsTool(),
            SearchColumnsTool(),
            # Visualization tools
            CreateChartTool(),
            CreateDashboardTool(),
            ExportVisualizationTool(),
            # Analysis tools
            SalesTrendTool(),
            InventoryAnalysisTool(),
            ProductPerformanceTool(),
            FinancialSummaryTool(),
            ComparisonTool(),
            # Query tools
            GenerateSQLTool(),
            ExecuteQueryTool(),
            ValidateQueryTool(),
            ExplainQueryTool(),
        ]
    
    @staticmethod
    def get_schema_tools():
        """Get schema inspection tools."""
        return [
            ListTablesTool(),
            DescribeTableTool(),
            GetTableRelationshipsTool(),
            SearchColumnsTool(),
        ]
    
    @staticmethod
    def get_visualization_tools():
        """Get visualization tools."""
        return [
            CreateChartTool(),
            CreateDashboardTool(),
            ExportVisualizationTool(),
        ]
    
    @staticmethod
    def get_analysis_tools():
        """Get analysis tools."""
        return [
            SalesTrendTool(),
            InventoryAnalysisTool(),
            ProductPerformanceTool(),
            FinancialSummaryTool(),
            ComparisonTool(),
        ]
    
    @staticmethod
    def get_query_tools():
        """Get query generation tools."""
        return [
            GenerateSQLTool(),
            ExecuteQueryTool(),
            ValidateQueryTool(),
            ExplainQueryTool(),
        ]
