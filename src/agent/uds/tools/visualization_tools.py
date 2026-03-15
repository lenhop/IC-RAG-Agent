"""
Visualization Tools for UDS Agent.

Provides 3 tools for creating and exporting charts: CreateChartTool,
CreateDashboardTool, ExportVisualizationTool.

All tools inherit from BaseTool (ai-toolkit) and return ToolResult.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from ai_toolkit.tools import BaseTool, ToolParameter, ToolResult
from ai_toolkit.errors import ValidationError

logger = logging.getLogger(__name__)


class CreateChartTool(BaseTool):
    """
    Create charts from data.

    Supports line, bar, pie, and scatter charts.
    """

    SUPPORTED_TYPES = ("line", "bar", "pie", "scatter")

    def __init__(self):
        super().__init__(
            name="create_chart",
            description="Create interactive charts (line, bar, pie, scatter) from data",
        )

    def _get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="data",
                type="object",
                description="Data to visualize (dict or list of dicts)",
                required=True,
            ),
            ToolParameter(
                name="chart_type",
                type="string",
                description="Type of chart: 'line', 'bar', 'pie', 'scatter'",
                required=True,
            ),
            ToolParameter(
                name="x_column",
                type="string",
                description="Column for x-axis (or names for pie)",
                required=False,
            ),
            ToolParameter(
                name="y_column",
                type="string",
                description="Column for y-axis (or values for pie)",
                required=False,
            ),
            ToolParameter(
                name="title",
                type="string",
                description="Chart title",
                required=False,
            ),
        ]

    def validate_parameters(
        self,
        data: Any = None,
        chart_type: str = None,
        x_column: str = None,
        y_column: str = None,
        **kwargs,
    ) -> None:
        """Validate parameters."""
        if data is None:
            raise ValidationError(
                message="data is required",
                field_name="data",
            )
        if not chart_type or not str(chart_type).strip():
            raise ValidationError(
                message="chart_type is required",
                field_name="chart_type",
            )
        chart_type = str(chart_type).strip().lower()
        if chart_type not in self.SUPPORTED_TYPES:
            raise ValidationError(
                message=f"chart_type must be one of: {', '.join(self.SUPPORTED_TYPES)}",
                field_name="chart_type",
                field_value=chart_type,
            )
        if chart_type != "pie" and (not x_column or not y_column):
            raise ValidationError(
                message="x_column and y_column are required for non-pie charts",
                field_name="x_column",
            )
        if chart_type == "pie" and (not x_column or not y_column):
            raise ValidationError(
                message="x_column (names) and y_column (values) are required for pie charts",
                field_name="x_column",
            )

    def execute(
        self,
        data: Any,
        chart_type: str,
        x_column: str = None,
        y_column: str = None,
        title: str = None,
        **kwargs,
    ) -> ToolResult:
        """
        Create a chart.

        Args:
            data: Data to visualize (dict or list of dicts)
            chart_type: Type of chart
            x_column: X-axis column (or names for pie)
            y_column: Y-axis column (or values for pie)
            title: Chart title

        Returns:
            ToolResult with chart HTML
        """
        try:
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data

            if df.empty:
                return ToolResult(
                    success=False,
                    error="Data is empty",
                    metadata={"message": "Cannot create chart from empty data"},
                )

            chart_type = str(chart_type).strip().lower()
            title = str(title).strip() if title else None

            if chart_type == "line":
                fig = px.line(df, x=x_column, y=y_column, title=title)
            elif chart_type == "bar":
                fig = px.bar(df, x=x_column, y=y_column, title=title)
            elif chart_type == "pie":
                fig = px.pie(df, names=x_column, values=y_column, title=title)
            elif chart_type == "scatter":
                fig = px.scatter(df, x=x_column, y=y_column, title=title)
            else:
                return ToolResult(
                    success=False,
                    error=f"Unsupported chart type: {chart_type}",
                    metadata={"message": f"Supported types: {', '.join(self.SUPPORTED_TYPES)}"},
                )

            chart_html = fig.to_html(include_plotlyjs="cdn")
            figure_json = fig.to_json()

            return ToolResult(
                success=True,
                output={
                    "chart_html": chart_html,
                    "figure_json": figure_json,
                    "chart_type": chart_type,
                    "title": title,
                },
                metadata={"message": f"{chart_type.capitalize()} chart created successfully"},
            )

        except Exception as e:
            logger.exception("Failed to create chart")
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"message": "Failed to create chart"},
            )


class CreateDashboardTool(BaseTool):
    """
    Create a dashboard with multiple charts.

    Arranges charts in a grid layout.
    """

    SUPPORTED_LAYOUTS = ("2x2", "1x3", "3x1", "2x1", "1x2")

    def __init__(self):
        super().__init__(
            name="create_dashboard",
            description="Create a dashboard with multiple charts arranged in a grid",
        )

    def _get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="charts",
                type="array",
                description="List of chart configurations (each with data, type, x_column, y_column, title)",
                required=True,
            ),
            ToolParameter(
                name="title",
                type="string",
                description="Dashboard title",
                required=False,
            ),
            ToolParameter(
                name="layout",
                type="string",
                description="Layout: '2x2', '1x3', '3x1' (default: '2x2')",
                required=False,
                default="2x2",
            ),
        ]

    def validate_parameters(
        self,
        charts: List = None,
        title: str = None,
        layout: str = "2x2",
        **kwargs,
    ) -> None:
        """Validate parameters."""
        if not charts or not isinstance(charts, list):
            raise ValidationError(
                message="charts is required and must be a list",
                field_name="charts",
            )
        if layout and str(layout).strip().lower() not in self.SUPPORTED_LAYOUTS:
            raise ValidationError(
                message=f"layout must be one of: {', '.join(self.SUPPORTED_LAYOUTS)}",
                field_name="layout",
                field_value=layout,
            )

    def execute(
        self,
        charts: List[Dict[str, Any]],
        title: str = "Dashboard",
        layout: str = "2x2",
        **kwargs,
    ) -> ToolResult:
        """
        Create a dashboard.

        Args:
            charts: List of chart configs
            title: Dashboard title
            layout: Grid layout

        Returns:
            ToolResult with dashboard HTML
        """
        try:
            from plotly.subplots import make_subplots

            layout = str(layout).strip().lower() or "2x2"
            if layout not in self.SUPPORTED_LAYOUTS:
                layout = "2x2"

            rows, cols = map(int, layout.split("x"))
            max_charts = rows * cols
            chart_configs = charts[:max_charts] if charts else []

            if not chart_configs:
                return ToolResult(
                    success=False,
                    error="No chart configurations provided",
                    metadata={"message": "At least one chart is required"},
                )

            subplot_titles = [c.get("title", f"Chart {i+1}") for i, c in enumerate(chart_configs)]
            fig = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=subplot_titles,
            )

            for idx, chart_config in enumerate(chart_configs):
                row = (idx // cols) + 1
                col = (idx % cols) + 1

                chart_type = chart_config.get("type", "line")
                chart_data = chart_config.get("data", {})
                if isinstance(chart_data, list):
                    data_df = pd.DataFrame(chart_data)
                else:
                    data_df = pd.DataFrame(chart_data)

                x_col = chart_config.get("x_column")
                y_col = chart_config.get("y_column")

                if data_df.empty or not x_col or not y_col:
                    continue

                if chart_type == "line":
                    fig.add_trace(
                        go.Scatter(
                            x=data_df[x_col],
                            y=data_df[y_col],
                            mode="lines",
                            name=chart_config.get("title", ""),
                        ),
                        row=row,
                        col=col,
                    )
                elif chart_type == "bar":
                    fig.add_trace(
                        go.Bar(
                            x=data_df[x_col],
                            y=data_df[y_col],
                            name=chart_config.get("title", ""),
                        ),
                        row=row,
                        col=col,
                    )
                elif chart_type == "scatter":
                    fig.add_trace(
                        go.Scatter(
                            x=data_df[x_col],
                            y=data_df[y_col],
                            mode="markers",
                            name=chart_config.get("title", ""),
                        ),
                        row=row,
                        col=col,
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=data_df[x_col],
                            y=data_df[y_col],
                            mode="lines",
                            name=chart_config.get("title", ""),
                        ),
                        row=row,
                        col=col,
                    )

            fig.update_layout(title_text=title or "Dashboard", showlegend=False)
            dashboard_html = fig.to_html(include_plotlyjs="cdn")
            figure_json = fig.to_json()

            return ToolResult(
                success=True,
                output={
                    "dashboard_html": dashboard_html,
                    "figure_json": figure_json,
                    "chart_count": len(chart_configs),
                    "layout": layout,
                },
                metadata={"message": f"Dashboard with {len(chart_configs)} charts created"},
            )

        except Exception as e:
            logger.exception("Failed to create dashboard")
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"message": "Failed to create dashboard"},
            )


def _extract_figure_from_html(html: str):
    """
    Extract Plotly figure from HTML string.

    Plotly's to_html() embeds the figure in a Plotly.newPlot() call.
    We extract the data and layout JSON to reconstruct the figure.
    """
    import plotly.io as pio

    # Try to find the JSON in the Plotly.newPlot call (last 64KB to avoid huge matches)
    match = re.search(
        r"Plotly\.newPlot\s*\(\s*[^,]+,\s*(\[.*?\])\s*,\s*(\{.*?\})\s*\)",
        html[-65536:],
        re.DOTALL,
    )
    if match:
        data_str = match.group(1)
        layout_str = match.group(2)
        try:
            data = json.loads(data_str)
            layout = json.loads(layout_str)
            fig_dict = {"data": data, "layout": layout}
            return pio.from_json(json.dumps(fig_dict))
        except json.JSONDecodeError:
            pass

    # Alternative: look for script type="application/json"
    match = re.search(
        r'<script type="application/json"[^>]*>(\s*\{.*?\}\s*)</script>',
        html,
        re.DOTALL,
    )
    if match:
        try:
            fig_dict = json.loads(match.group(1).strip())
            return pio.from_json(json.dumps(fig_dict))
        except json.JSONDecodeError:
            pass

    return None


class ExportVisualizationTool(BaseTool):
    """
    Export visualizations as PNG or PDF files.

    Requires kaleido package for image export.
    """

    SUPPORTED_FORMATS = ("png", "pdf")

    def __init__(self):
        super().__init__(
            name="export_visualization",
            description="Export charts or dashboards as PNG or PDF files",
        )

    def _get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="chart_html",
                type="string",
                description="Chart HTML to export (from create_chart or create_dashboard). Optional if figure_json provided.",
                required=False,
            ),
            ToolParameter(
                name="figure_json",
                type="string",
                description="Figure JSON from create_chart/create_dashboard output (preferred for reliable export)",
                required=False,
            ),
            ToolParameter(
                name="format",
                type="string",
                description="Export format: 'png' or 'pdf'",
                required=True,
            ),
            ToolParameter(
                name="filename",
                type="string",
                description="Output filename (default: 'chart.png' or 'chart.pdf')",
                required=False,
            ),
        ]

    def validate_parameters(
        self,
        chart_html: str = None,
        figure_json: str = None,
        format: str = None,
        filename: str = None,
        **kwargs,
    ) -> None:
        """Validate parameters."""
        has_html = chart_html and str(chart_html).strip()
        has_json = figure_json and str(figure_json).strip()
        if not has_html and not has_json:
            raise ValidationError(
                message="Either chart_html or figure_json is required",
                field_name="chart_html",
            )
        if not format or not str(format).strip():
            raise ValidationError(
                message="format is required",
                field_name="format",
            )
        fmt = str(format).strip().lower()
        if fmt not in self.SUPPORTED_FORMATS:
            raise ValidationError(
                message=f"format must be one of: {', '.join(self.SUPPORTED_FORMATS)}",
                field_name="format",
                field_value=format,
            )

    def execute(
        self,
        chart_html: str = None,
        figure_json: str = None,
        format: str = None,
        filename: str = None,
        **kwargs,
    ) -> ToolResult:
        """
        Export visualization.

        Args:
            chart_html: Chart HTML from create_chart or create_dashboard
            figure_json: Figure JSON (preferred, from create_chart output)
            format: Export format (png, pdf)
            filename: Output filename

        Returns:
            ToolResult with file path
        """
        try:
            import plotly.io as pio

            fmt = str(format).strip().lower()
            if fmt not in self.SUPPORTED_FORMATS:
                return ToolResult(
                    success=False,
                    error=f"Unsupported format: {format}",
                    metadata={"message": f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"},
                )

            if not filename or not str(filename).strip():
                filename = f"chart.{fmt}"
            else:
                filename = str(filename).strip()
                if not filename.lower().endswith(f".{fmt}"):
                    filename = f"{filename}.{fmt}"

            fig = None
            if figure_json and str(figure_json).strip():
                try:
                    fig = pio.from_json(str(figure_json))
                except Exception:
                    pass
            if fig is None and chart_html:
                fig = _extract_figure_from_html(chart_html)
            if fig is None:
                return ToolResult(
                    success=False,
                    error="Could not extract figure. Provide figure_json from create_chart output, or valid chart_html.",
                    metadata={"message": "Invalid chart HTML or figure JSON"},
                )

            # Resolve output path (project root / output)
            output_dir = Path(__file__).resolve().parent.parent.parent.parent / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / filename

            fig.write_image(str(output_path), format=fmt)

            return ToolResult(
                success=True,
                output={
                    "filename": filename,
                    "format": fmt,
                    "path": str(output_path),
                    "message": f"Chart exported to {output_path}",
                },
                metadata={"message": f"Visualization exported as {fmt}"},
            )

        except ImportError as e:
            logger.exception("kaleido not installed")
            return ToolResult(
                success=False,
                error="kaleido package is required for export. Install with: pip install kaleido",
                metadata={"message": "Missing dependency"},
            )
        except Exception as e:
            logger.exception("Failed to export visualization")
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"message": "Failed to export visualization"},
            )
