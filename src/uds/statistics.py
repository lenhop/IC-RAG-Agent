"""
Statistical analysis for UDS data.
Provides summaries and patterns for agent context.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from .uds_client import UDSClient


class StatisticalAnalyzer:
    """
    Statistical analysis for UDS tables.
    """
    
    def __init__(self, client: UDSClient):
        self.client = client
    
    def get_table_statistics(self, table: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a table.
        
        Args:
            table: Table name
        
        Returns:
            Dictionary with statistical summaries
        """
        stats = {
            "table": table,
            "row_count": 0,
            "numeric_columns": {},
            "categorical_columns": {},
            "date_columns": {}
        }
        
        # Get row count
        count_query = f"SELECT COUNT(*) as count FROM ic_agent.{table}"
        stats["row_count"] = self.client.query(count_query)['count'][0]
        
        # Get schema to identify column types
        schema = self.client.get_table_schema(table)
        
        for col in schema['columns']:
            col_name = col['name']
            col_type = col['type']
            
            if 'Float' in col_type or 'Int' in col_type:
                # Numeric column statistics
                query = f"""
                SELECT 
                    MIN({col_name}) as min_val,
                    MAX({col_name}) as max_val,
                    AVG({col_name}) as avg_val,
                    quantile(0.25)({col_name}) as q25,
                    quantile(0.50)({col_name}) as median,
                    quantile(0.75)({col_name}) as q75,
                    stddevPop({col_name}) as std_dev
                FROM ic_agent.{table}
                WHERE {col_name} IS NOT NULL
                """
                result = self.client.query(query).iloc[0]
                stats["numeric_columns"][col_name] = result.to_dict()
            
            elif 'String' in col_type:
                # Categorical column statistics
                query = f"""
                SELECT 
                    COUNT(DISTINCT {col_name}) as unique_count,
                    COUNT(*) as total_count
                FROM ic_agent.{table}
                """
                result = self.client.query(query).iloc[0]
                stats["categorical_columns"][col_name] = {
                    "unique_count": result['unique_count'],
                    "cardinality": result['unique_count'] / result['total_count']
                }
            
            elif 'Date' in col_type:
                # Date column statistics
                query = f"""
                SELECT 
                    MIN({col_name}) as min_date,
                    MAX({col_name}) as max_date,
                    COUNT(DISTINCT {col_name}) as unique_dates
                FROM ic_agent.{table}
                """
                result = self.client.query(query).iloc[0]
                stats["date_columns"][col_name] = result.to_dict()
        
        return stats
    
    def analyze_time_series(self, table: str, date_col: str, metric_col: str) -> Dict[str, Any]:
        """
        Analyze time series patterns.
        
        Args:
            table: Table name
            date_col: Date column name
            metric_col: Metric to analyze
        
        Returns:
            Dictionary with time series analysis
        """
        query = f"""
        SELECT 
            {date_col} as date,
            COUNT(*) as count,
            SUM({metric_col}) as total,
            AVG({metric_col}) as average
        FROM ic_agent.{table}
        GROUP BY {date_col}
        ORDER BY {date_col}
        """
        df = self.client.query(query)
        
        # Calculate trends
        df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
        
        analysis = {
            "total_days": len(df),
            "total_count": df['count'].sum(),
            "avg_daily_count": df['count'].mean(),
            "peak_day": df.loc[df['count'].idxmax(), 'date'],
            "low_day": df.loc[df['count'].idxmin(), 'date'],
            "trend": "increasing" if df['count'].iloc[-1] > df['count'].iloc[0] else "decreasing",
            "weekly_pattern": df.groupby('day_of_week')['count'].mean().to_dict()
        }
        
        return analysis
    
    def find_correlations(self, table: str, columns: List[str]) -> pd.DataFrame:
        """
        Find correlations between numeric columns.
        
        Args:
            table: Table name
            columns: List of numeric columns
        
        Returns:
            Correlation matrix as DataFrame
        """
        query = f"""
        SELECT {', '.join(columns)}
        FROM ic_agent.{table}
        WHERE {' AND '.join([f'{col} IS NOT NULL' for col in columns])}
        """
        df = self.client.query(query)
        return df.corr()
