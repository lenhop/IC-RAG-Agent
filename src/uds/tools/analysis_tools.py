from ai_toolkit.tools import BaseTool, ToolParameter, ToolResult
from ..query_templates import QueryTemplateRegistry
from ..uds_client import UDSClient
from ..config import UDSConfig
import pandas as pd
import numpy as np


def _build_registry() -> QueryTemplateRegistry:
    """Create a query template registry backed by a configured UDS client."""
    client = UDSClient(
        host=UDSConfig.CH_HOST,
        port=UDSConfig.CH_PORT,
        user=UDSConfig.CH_USER,
        password=UDSConfig.CH_PASSWORD,
        database=UDSConfig.CH_DATABASE,
    )
    return QueryTemplateRegistry(client)


class _BaseAnalysisTool(BaseTool):
    """Shared initializer for analysis tools."""

    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.registry = _build_registry()


class SalesTrendTool(_BaseAnalysisTool):
    """
    Analyze sales trends over time with automatic insights.
    Calculates growth rates, identifies peaks, and detects patterns.
    """
    
    name = "analyze_sales_trend"
    description = "Analyze sales trends over time with automatic insights including growth rate, peak periods, and patterns"
    
    parameters = [
        ToolParameter(
            name="start_date",
            type="string",
            required=True,
            description="Start date (YYYY-MM-DD)"
        ),
        ToolParameter(
            name="end_date",
            type="string",
            required=True,
            description="End date (YYYY-MM-DD)"
        ),
        ToolParameter(
            name="granularity",
            type="string",
            required=False,
            description="Time granularity: 'daily', 'weekly', 'monthly' (default: 'daily')",
            default="daily"
        ),
        ToolParameter(
            name="marketplace",
            type="string",
            required=False,
            description="Filter by marketplace (optional)"
        )
    ]
    
    def __init__(self):
        super().__init__(self.name, self.description)
    
    def _calculate_insights(self, df: pd.DataFrame) -> dict:
        """Calculate automatic insights from sales data."""
        insights = {}
        
        # Basic metrics
        insights['total_revenue'] = float(df['total_revenue'].sum())
        insights['total_orders'] = int(df['order_count'].sum())
        insights['avg_daily_revenue'] = float(df['total_revenue'].mean())
        insights['avg_order_value'] = float(df['avg_order_value'].mean())
        
        # Trend analysis
        if len(df) > 1:
            first_revenue = df['total_revenue'].iloc[0]
            last_revenue = df['total_revenue'].iloc[-1]
            
            if first_revenue > 0:
                growth_rate = ((last_revenue - first_revenue) / first_revenue) * 100
                insights['growth_rate_pct'] = round(growth_rate, 2)
                insights['trend'] = 'increasing' if growth_rate > 0 else 'decreasing'
            else:
                insights['growth_rate_pct'] = 0
                insights['trend'] = 'stable'
        
        # Peak and low periods
        peak_idx = df['total_revenue'].idxmax()
        low_idx = df['total_revenue'].idxmin()
        
        insights['peak_day'] = {
            'date': str(df.loc[peak_idx, 'date']),
            'revenue': float(df.loc[peak_idx, 'total_revenue']),
            'orders': int(df.loc[peak_idx, 'order_count'])
        }
        
        insights['low_day'] = {
            'date': str(df.loc[low_idx, 'date']),
            'revenue': float(df.loc[low_idx, 'total_revenue']),
            'orders': int(df.loc[low_idx, 'order_count'])
        }
        
        # Volatility
        insights['revenue_std_dev'] = float(df['total_revenue'].std())
        insights['revenue_cv'] = float(df['total_revenue'].std() / df['total_revenue'].mean()) if df['total_revenue'].mean() > 0 else 0
        
        # Weekly pattern (if daily data)
        if 'date' in df.columns and len(df) >= 7:
            df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
            weekly_avg = df.groupby('day_of_week')['total_revenue'].mean()
            insights['best_day_of_week'] = int(weekly_avg.idxmax())
            insights['worst_day_of_week'] = int(weekly_avg.idxmin())
        
        return insights
    
    def execute(
        self,
        start_date: str,
        end_date: str,
        granularity: str = "daily",
        marketplace: str = None
    ) -> ToolResult:
        """
        Analyze sales trends.
        """
        try:
            # Get sales data using query template
            df = self.registry.sales.daily_sales_trend(
                start_date, end_date, marketplace
            )
            
            if df.empty:
                # follow the current ToolResult signature: success, output, error, metadata
                # output/metadata can include human readable messages if needed
                return ToolResult(
                    success=False,
                    error="No data found for the specified period",
                    metadata={
                        "message": "No sales data available"
                    }
                )
            
            # Calculate insights
            insights = self._calculate_insights(df)
            
            # Generate summary
            summary = f"""
Sales Trend Analysis ({start_date} to {end_date}):
- Total Revenue: ${insights['total_revenue']:,.2f}
- Total Orders: {insights['total_orders']:,}
- Average Daily Revenue: ${insights['avg_daily_revenue']:,.2f}
- Growth Rate: {insights.get('growth_rate_pct', 0):.2f}%
- Trend: {insights.get('trend', 'stable').capitalize()}
- Peak Day: {insights['peak_day']['date']} (${insights['peak_day']['revenue']:,.2f})
"""
            
            return ToolResult(
                success=True,
                output={
                    "sales_data": df.to_dict('records'),
                    "insights": insights,
                    "summary": summary.strip()
                },
                metadata={
                    "message": "Sales trend analysis completed"
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"message": "Failed to analyze sales trend"}
            )

    def validate_parameters(self, **kwargs):
        """Validate parameters for sales trend analysis."""
        # Basic validation - could be enhanced
        pass

    def _get_parameters(self):
        """Get parameter definitions."""
        return self.parameters


class InventoryAnalysisTool(_BaseAnalysisTool):
    """
    Analyze inventory levels, turnover, and generate alerts.
    Identifies low stock, slow-moving items, and stockout risks.
    """
    
    name = "analyze_inventory"
    description = "Analyze inventory levels, turnover rates, and generate low stock alerts"
    
    parameters = [
        ToolParameter(
            name="as_of_date",
            type="string",
            required=False,
            description="Date to analyze inventory (YYYY-MM-DD, defaults to latest)"
        ),
        ToolParameter(
            name="low_stock_threshold",
            type="integer",
            required=False,
            description="Threshold for low stock alert (default: 10)",
            default=10
        )
    ]
    
    def __init__(self):
        super().__init__(self.name, self.description)
    
    def execute(
        self,
        as_of_date: str = None,
        low_stock_threshold: int = 10
    ) -> ToolResult:
        """
        Analyze inventory.
        """
        try:
            # Get current inventory levels
            inventory_df = self.registry.inventory.current_inventory_levels(as_of_date)
            
            # Get low stock items
            low_stock_df = self.registry.inventory.low_stock_alert(
                threshold=low_stock_threshold,
                as_of_date=as_of_date
            )
            
            # Calculate insights
            insights = {
                'total_skus': len(inventory_df),
                'total_units': int(inventory_df['total_quantity'].sum()),
                'low_stock_items': len(low_stock_df),
                'avg_stock_per_sku': float(inventory_df['total_quantity'].mean()),
                'stockout_risk': len(low_stock_df[low_stock_df['total_quantity'] == 0])
            }
            
            # Top 10 highest stock items
            top_stock = inventory_df.nlargest(10, 'total_quantity')[
                ['sku', 'total_quantity']
            ].to_dict('records')
            
            # Generate alerts
            alerts = []
            if insights['low_stock_items'] > 0:
                alerts.append(f"⚠️ {insights['low_stock_items']} items below threshold ({low_stock_threshold} units)")
            if insights['stockout_risk'] > 0:
                alerts.append(f"🚨 {insights['stockout_risk']} items at risk of stockout (0 units)")
            
            summary = f"""
Inventory Analysis:
- Total SKUs: {insights['total_skus']:,}
- Total Units: {insights['total_units']:,}
- Average Stock per SKU: {insights['avg_stock_per_sku']:.0f}
- Low Stock Items: {insights['low_stock_items']}
- Stockout Risk: {insights['stockout_risk']}
"""
            
            return ToolResult(
                success=True,
                output={
                    "inventory_summary": insights,
                    "low_stock_items": low_stock_df.to_dict('records'),
                    "top_stock_items": top_stock,
                    "alerts": alerts,
                    "summary": summary.strip()
                },
                metadata={"message": "Inventory analysis completed"}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"message": "Failed to analyze inventory"}
            )

    def validate_parameters(self, **kwargs):
        """Validate parameters for inventory analysis."""
        pass

    def _get_parameters(self):
        """Get parameter definitions."""
        return self.parameters


class ProductPerformanceTool(_BaseAnalysisTool):
    """
    Analyze product performance metrics.
    Identifies top/bottom performers and provides insights.
    """
    
    name = "analyze_product_performance"
    description = "Analyze product performance including top sellers, revenue, and trends"
    
    parameters = [
        ToolParameter(
            name="start_date",
            type="string",
            required=True,
            description="Start date (YYYY-MM-DD)"
        ),
        ToolParameter(
            name="end_date",
            type="string",
            required=True,
            description="End date (YYYY-MM-DD)"
        ),
        ToolParameter(
            name="metric",
            type="string",
            required=False,
            description="Metric to analyze: 'revenue' or 'units' (default: 'revenue')",
            default="revenue"
        ),
        ToolParameter(
            name="limit",
            type="integer",
            required=False,
            description="Number of top/bottom products to return (default: 10)",
            default=10
        )
    ]
    
    def __init__(self):
        super().__init__(self.name, self.description)
    
    def execute(
        self,
        start_date: str,
        end_date: str,
        metric: str = "revenue",
        limit: int = 10
    ) -> ToolResult:
        """
        Analyze product performance.
        """
        try:
            # Get top products
            top_products = self.registry.products.top_selling_products(
                start_date, end_date, limit, metric
            )
            
            if top_products.empty:
                return ToolResult(
                    success=False,
                    error="No product data found",
                    metadata={"message": "No products found for the specified period"}
                )
            
            # Calculate insights
            insights = {
                'total_products': len(top_products),
                'total_revenue': float(top_products['total_revenue'].sum()),
                'total_units': int(top_products['units_sold'].sum()),
                'avg_price': float(top_products['avg_price'].mean()),
                'top_product': {
                    'sku': top_products.iloc[0]['sku'],
                    'name': top_products.iloc[0]['product_name'],
                    'revenue': float(top_products.iloc[0]['total_revenue']),
                    'units': int(top_products.iloc[0]['units_sold'])
                }
            }
            
            # Revenue concentration (top 20% products)
            top_20_pct_count = max(1, len(top_products) // 5)
            top_20_pct_revenue = top_products.head(top_20_pct_count)['total_revenue'].sum()
            insights['revenue_concentration'] = float(
                (top_20_pct_revenue / insights['total_revenue']) * 100
            )
            
            summary = f"""
Product Performance Analysis ({start_date} to {end_date}):
- Total Products Analyzed: {insights['total_products']}
- Total Revenue: ${insights['total_revenue']:,.2f}
- Total Units Sold: {insights['total_units']:,}
- Average Price: ${insights['avg_price']:.2f}
- Top Product: {insights['top_product']['name']} (${insights['top_product']['revenue']:,.2f})
- Revenue Concentration: {insights['revenue_concentration']:.1f}% from top 20% products
"""
            
            return ToolResult(
                success=True,
                output={
                    "top_products": top_products.to_dict('records'),
                    "insights": insights,
                    "summary": summary.strip()
                },
                metadata={"message": "Product performance analysis completed"}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"message": "Failed to analyze product performance"}
            )

    def validate_parameters(self, **kwargs):
        """Validate parameters for product performance analysis."""
        pass

    def _get_parameters(self):
        """Get parameter definitions."""
        return self.parameters


class FinancialSummaryTool(_BaseAnalysisTool):
    """
    Generate financial summary with revenue, fees, and profitability.
    """
    
    name = "financial_summary"
    description = "Generate financial summary including revenue, fees, and profitability analysis"
    
    parameters = [
        ToolParameter(
            name="start_date",
            type="string",
            required=True,
            description="Start date (YYYY-MM-DD)"
        ),
        ToolParameter(
            name="end_date",
            type="string",
            required=True,
            description="End date (YYYY-MM-DD)"
        )
    ]
    
    def __init__(self):
        super().__init__(self.name, self.description)
    
    def execute(self, start_date: str, end_date: str) -> ToolResult:
        """
        Generate financial summary.
        """
        try:
            # Get revenue summary
            revenue_df = self.registry.financial.revenue_summary(
                start_date, end_date, group_by='day'
            )
            
            # Get fee analysis
            fees_df = self.registry.financial.fee_analysis(start_date, end_date)
            
            # Calculate metrics
            total_revenue = float(revenue_df['gross_revenue'].sum())
            total_fees = float(fees_df['total_fees'].sum())
            net_revenue = total_revenue - total_fees
            profit_margin = (net_revenue / total_revenue * 100) if total_revenue > 0 else 0
            
            insights = {
                'total_revenue': total_revenue,
                'total_fees': total_fees,
                'net_revenue': net_revenue,
                'profit_margin_pct': round(profit_margin, 2),
                'avg_daily_revenue': float(revenue_df['gross_revenue'].mean()),
                'fee_breakdown': fees_df.to_dict('records')
            }
            
            summary = f"""
Financial Summary ({start_date} to {end_date}):
- Gross Revenue: ${total_revenue:,.2f}
- Total Fees: ${total_fees:,.2f}
- Net Revenue: ${net_revenue:,.2f}
- Profit Margin: {profit_margin:.2f}%
- Avg Daily Revenue: ${insights['avg_daily_revenue']:,.2f}
"""
            
            return ToolResult(
                success=True,
                output={
                    "financial_metrics": insights,
                    "revenue_trend": revenue_df.to_dict('records'),
                    "summary": summary.strip()
                },
                metadata={"message": "Financial summary generated"}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"message": "Failed to generate financial summary"}
            )

    def validate_parameters(self, **kwargs):
        """Validate parameters for financial summary."""
        pass

    def _get_parameters(self):
        """Get parameter definitions."""
        return self.parameters


class ComparisonTool(_BaseAnalysisTool):
    """
    Compare metrics across periods, products, or marketplaces.
    Calculates growth rates and statistical significance.
    """
    
    name = "compare_metrics"
    description = "Compare metrics across different periods, products, or marketplaces"
    
    parameters = [
        ToolParameter(
            name="comparison_type",
            type="string",
            required=True,
            description="Type of comparison: 'period', 'product', 'marketplace'"
        ),
        ToolParameter(
            name="period1_start",
            type="string",
            required=False,
            description="Period 1 start date (for period comparison)"
        ),
        ToolParameter(
            name="period1_end",
            type="string",
            required=False,
            description="Period 1 end date (for period comparison)"
        ),
        ToolParameter(
            name="period2_start",
            type="string",
            required=False,
            description="Period 2 start date (for period comparison)"
        ),
        ToolParameter(
            name="period2_end",
            type="string",
            required=False,
            description="Period 2 end date (for period comparison)"
        )
    ]
    
    def __init__(self):
        super().__init__(self.name, self.description)
    
    def execute(
        self,
        comparison_type: str,
        period1_start: str = None,
        period1_end: str = None,
        period2_start: str = None,
        period2_end: str = None
    ) -> ToolResult:
        """
        Compare metrics.
        """
        try:
            if comparison_type == 'period':
                # Period comparison
                df = self.registry.sales.sales_growth_rate(
                    period1_start, period1_end,
                    period2_start, period2_end
                )
                
                if len(df) == 2:
                    period1 = df.iloc[0]
                    period2 = df.iloc[1]
                    
                    revenue_growth = ((period2['total_revenue'] - period1['total_revenue']) / 
                                    period1['total_revenue'] * 100)
                    order_growth = ((period2['order_count'] - period1['order_count']) / 
                                  period1['order_count'] * 100)
                    
                    comparison = {
                        'period1': {
                            'revenue': float(period1['total_revenue']),
                            'orders': int(period1['order_count'])
                        },
                        'period2': {
                            'revenue': float(period2['total_revenue']),
                            'orders': int(period2['order_count'])
                        },
                        'growth': {
                            'revenue_pct': round(revenue_growth, 2),
                            'orders_pct': round(order_growth, 2)
                        }
                    }
                    
                    summary = f"""
Period Comparison:
Period 1 ({period1_start} to {period1_end}):
  - Revenue: ${comparison['period1']['revenue']:,.2f}
  - Orders: {comparison['period1']['orders']:,}

Period 2 ({period2_start} to {period2_end}):
  - Revenue: ${comparison['period2']['revenue']:,.2f}
  - Orders: {comparison['period2']['orders']:,}

Growth:
  - Revenue: {comparison['growth']['revenue_pct']:.2f}%
  - Orders: {comparison['growth']['orders_pct']:.2f}%
"""
                    
                    return ToolResult(
                        success=True,
                        output={
                            "comparison": comparison,
                            "summary": summary.strip()
                        },
                        metadata={"message": "Period comparison completed"}
                    )
            
            return ToolResult(
                success=False,
                error=f"Comparison type '{comparison_type}' not yet implemented",
                metadata={"message": "Only 'period' comparison is currently supported"}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"message": "Failed to compare metrics"}
            )

    def validate_parameters(self, **kwargs):
        """Validate parameters for comparison."""
        pass

    def _get_parameters(self):
        """Get parameter definitions."""
        return self.parameters
