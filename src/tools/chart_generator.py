"""
ChartGeneratorTool -- Generates premium statistical charts using Plotly.
Produces interactive, visually rich charts: histograms, box plots, scatter plots,
heatmaps, bar charts, area-gradient, sunburst, radar, violin, treemap, donut, and QQ plots.
"""

import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from scipy import stats
from crewai.tools import tool

from src.config.settings import get_settings

# Premium dark theme for all charts
CHART_TEMPLATE = "plotly_dark"
COLOR_PALETTE = px.colors.sequential.Plasma
DISCRETE_COLORS = px.colors.qualitative.Set2
GRADIENT_COLORS = ["#667eea", "#764ba2", "#f093fb", "#f5576c", "#4facfe", "#00f2fe"]


def _adaptive_size(n_rows: int, n_items: int = 1) -> tuple[int, int]:
    """Scale chart dimensions based on data volume."""
    # Width scales with number of items (categories, columns)
    width = max(800, min(1600, 800 + n_items * 40))
    # Height scales mildly with row count
    height = max(500, min(900, 500 + int(n_rows ** 0.3) * 10))
    return width, height


def _save_chart(fig, filepath: str, title: str, width: int = 1200, height: int = 700):
    """Apply consistent styling and save the chart as PNG and HTML."""
    fig.update_layout(
        template=CHART_TEMPLATE,
        font=dict(family="Inter, Arial, sans-serif", size=13),
        title=dict(text=title, font=dict(size=18, color="#e0e0e0"), x=0.5, xanchor="center"),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        margin=dict(l=60, r=40, t=80, b=60),
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
            font=dict(color="#e0e0e0"),
        ),
    )
    # Save as PNG
    fig.write_image(filepath, width=width, height=height, scale=2)
    # Also save interactive HTML version
    html_path = filepath.replace(".png", ".html")
    fig.write_html(html_path, include_plotlyjs="cdn")


@tool("Chart Generator")
def chart_generator_tool(file_path: str, chart_instructions: str) -> str:
    """
    Generate premium statistical charts from cleaned CSV data using Plotly.

    Args:
        file_path: Absolute path to the cleaned CSV file.
        chart_instructions: JSON string describing the charts to generate.
            You can include an optional "output_dir" key to specify where to save charts.
            Supported chart types:
            - "histogram": Distribution of numeric columns
            - "boxplot": Box plot, optionally grouped by a category
            - "scatter": Scatter plot between two numeric columns
            - "heatmap": Correlation heatmap of all numeric columns
            - "bar": Frequency counts for a categorical column
            - "area": Area chart with gradient fill for numeric trends
            - "sunburst": Hierarchical breakdown of categorical columns
            - "radar": Spider/radar chart comparing metrics across categories
            - "violin": Violin plot showing distribution shape
            - "treemap": Hierarchical treemap of categorical data
            - "donut": Donut/pie chart for category proportions
            - "qq": QQ plot for normality assessment

            Example: {
                "output_dir": "D:/path/to/charts",
                "charts": [
                    {"type": "histogram", "columns": ["age", "salary"], "title": "Distribution"},
                    {"type": "scatter", "x": "age", "y": "salary", "hue": "dept", "title": "Age vs Salary"},
                    {"type": "heatmap", "title": "Correlation Matrix"},
                    {"type": "area", "columns": ["revenue", "cost"], "title": "Revenue Trends"},
                    {"type": "sunburst", "path": ["region", "city"], "values": "sales", "title": "Sales by Region"},
                    {"type": "radar", "categories": "department", "metrics": ["avg_salary", "headcount"], "title": "Dept Comparison"},
                    {"type": "violin", "column": "salary", "group_by": "dept", "title": "Salary Distribution"},
                    {"type": "donut", "column": "status", "title": "Status Split"}
                ]
            }
    """
    try:
        settings = get_settings()
        df = pd.read_csv(file_path)

        # Parse instructions
        try:
            config = json.loads(chart_instructions)
        except (json.JSONDecodeError, TypeError):
            config = {"charts": []}

        # Determine output directory -- use per-run dir if provided, else default
        charts_dir = config.get("output_dir", None)
        if charts_dir:
            charts_dir = str(charts_dir)
        else:
            # Derive from CSV filename
            csv_stem = Path(file_path).stem.replace("cleaned_", "")
            charts_dir = str(settings.run_charts_dir(csv_stem))

        # Ensure the directory exists
        os.makedirs(charts_dir, exist_ok=True)

        chart_specs = config.get("charts", [])
        generated = []
        n_rows = len(df)

        for i, spec in enumerate(chart_specs):
            chart_type = spec.get("type", "").lower()
            title = spec.get("title", f"Chart {i + 1}")
            safe_title = "".join(c if c.isalnum() or c == "_" else "_" for c in title)
            filename = f"{i + 1}_{safe_title}.png"
            filepath = os.path.join(charts_dir, filename)

            try:
                fig = None

                # --- HISTOGRAM ---
                if chart_type == "histogram":
                    columns = spec.get("columns", [])
                    valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
                    if valid_cols:
                        fig = go.Figure()
                        for j, col in enumerate(valid_cols):
                            color = GRADIENT_COLORS[j % len(GRADIENT_COLORS)]
                            fig.add_trace(go.Histogram(
                                x=df[col].dropna(), name=col, opacity=0.75,
                                marker=dict(color=color, line=dict(color="white", width=0.5)),
                                nbinsx=30,
                            ))
                        fig.update_layout(barmode="overlay", xaxis_title="Value", yaxis_title="Frequency")

                # --- BOXPLOT ---
                elif chart_type == "boxplot":
                    col = spec.get("column", "")
                    group_by = spec.get("group_by", "")
                    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                        if group_by and group_by in df.columns:
                            fig = px.box(
                                df, x=group_by, y=col, color=group_by,
                                color_discrete_sequence=DISCRETE_COLORS,
                                notched=True, points="outliers",
                            )
                        else:
                            fig = px.box(
                                df, y=col,
                                color_discrete_sequence=[GRADIENT_COLORS[0]],
                                notched=True, points="outliers",
                            )

                # --- SCATTER ---
                elif chart_type == "scatter":
                    x_col = spec.get("x", "")
                    y_col = spec.get("y", "")
                    hue_col = spec.get("hue", None)
                    if x_col in df.columns and y_col in df.columns:
                        scatter_kwargs = dict(
                            data_frame=df, x=x_col, y=y_col, opacity=0.7,
                            color_discrete_sequence=DISCRETE_COLORS,
                            trendline="ols",
                        )
                        if hue_col and hue_col in df.columns:
                            scatter_kwargs["color"] = hue_col
                        fig = px.scatter(**scatter_kwargs)
                        fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color="white")))

                # --- HEATMAP ---
                elif chart_type == "heatmap":
                    numeric_df = df.select_dtypes(include=["number"])
                    if len(numeric_df.columns) >= 2:
                        corr = numeric_df.corr()
                        fig = go.Figure(data=go.Heatmap(
                            z=corr.values,
                            x=corr.columns.tolist(),
                            y=corr.columns.tolist(),
                            colorscale="RdBu_r",
                            zmid=0,
                            text=np.round(corr.values, 2),
                            texttemplate="%{text}",
                            textfont=dict(size=11, color="white"),
                            hoverongaps=False,
                        ))
                        size = max(600, len(corr.columns) * 80)
                        fig.update_layout(width=size, height=size)

                # --- BAR ---
                elif chart_type == "bar":
                    col = spec.get("column", "")
                    if col in df.columns:
                        counts = df[col].value_counts().head(15).reset_index()
                        counts.columns = [col, "count"]
                        fig = px.bar(
                            counts, x=col, y="count",
                            color="count", color_continuous_scale="Plasma",
                            text="count",
                        )
                        fig.update_traces(
                            textposition="outside",
                            marker_line_color="white", marker_line_width=0.5,
                        )
                        fig.update_layout(xaxis_tickangle=-45)

                # --- AREA (gradient fill) ---
                elif chart_type == "area":
                    columns = spec.get("columns", [])
                    valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
                    if valid_cols:
                        fig = go.Figure()
                        for j, col in enumerate(valid_cols):
                            color = GRADIENT_COLORS[j % len(GRADIENT_COLORS)]
                            fig.add_trace(go.Scatter(
                                x=list(range(len(df))), y=df[col].values,
                                mode="lines", name=col, fill="tozeroy",
                                line=dict(color=color, width=2),
                                fillcolor=color.replace(")", ",0.3)").replace("rgb", "rgba") if "rgb" in color else color + "4D",
                            ))
                        fig.update_layout(xaxis_title="Index", yaxis_title="Value")

                # --- SUNBURST ---
                elif chart_type == "sunburst":
                    path_cols = spec.get("path", [])
                    values_col = spec.get("values", None)
                    valid_path = [c for c in path_cols if c in df.columns]
                    if len(valid_path) >= 1:
                        sunburst_kwargs = dict(
                            data_frame=df, path=valid_path,
                            color_discrete_sequence=px.colors.qualitative.Pastel,
                        )
                        if values_col and values_col in df.columns:
                            sunburst_kwargs["values"] = values_col
                        fig = px.sunburst(**sunburst_kwargs)
                        fig.update_traces(textinfo="label+percent parent")

                # --- RADAR / SPIDER ---
                elif chart_type == "radar":
                    cat_col = spec.get("categories", "")
                    metrics = spec.get("metrics", [])
                    valid_metrics = [m for m in metrics if m in df.columns and pd.api.types.is_numeric_dtype(df[m])]
                    if cat_col in df.columns and valid_metrics:
                        grouped = df.groupby(cat_col)[valid_metrics].mean()
                        fig = go.Figure()
                        for j, (cat, row) in enumerate(grouped.head(8).iterrows()):
                            # Normalize values to 0-1 for radar
                            vals = row.values
                            max_vals = grouped.max().values
                            max_vals[max_vals == 0] = 1
                            norm_vals = vals / max_vals
                            color = DISCRETE_COLORS[j % len(DISCRETE_COLORS)]
                            fig.add_trace(go.Scatterpolar(
                                r=list(norm_vals) + [norm_vals[0]],
                                theta=valid_metrics + [valid_metrics[0]],
                                fill="toself", name=str(cat),
                                line=dict(color=color),
                                fillcolor=color + "4D" if not color.startswith("rgb") else color,
                            ))
                        fig.update_layout(
                            polar=dict(
                                bgcolor="rgba(0,0,0,0.3)",
                                radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.2)"),
                                angularaxis=dict(gridcolor="rgba(255,255,255,0.2)"),
                            ),
                        )

                # --- VIOLIN ---
                elif chart_type == "violin":
                    col = spec.get("column", "")
                    group_by = spec.get("group_by", "")
                    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                        if group_by and group_by in df.columns:
                            fig = px.violin(
                                df, x=group_by, y=col, color=group_by,
                                color_discrete_sequence=DISCRETE_COLORS,
                                box=True, points="outliers",
                            )
                        else:
                            fig = px.violin(
                                df, y=col,
                                color_discrete_sequence=[GRADIENT_COLORS[0]],
                                box=True, points="outliers",
                            )

                # --- TREEMAP ---
                elif chart_type == "treemap":
                    path_cols = spec.get("path", [])
                    values_col = spec.get("values", None)
                    valid_path = [c for c in path_cols if c in df.columns]
                    if len(valid_path) >= 1:
                        treemap_kwargs = dict(
                            data_frame=df, path=valid_path,
                            color_discrete_sequence=px.colors.qualitative.Pastel,
                        )
                        if values_col and values_col in df.columns:
                            treemap_kwargs["values"] = values_col
                        fig = px.treemap(**treemap_kwargs)

                # --- DONUT / PIE ---
                elif chart_type in ("donut", "pie"):
                    col = spec.get("column", "")
                    if col in df.columns:
                        counts = df[col].value_counts().head(10)
                        hole = 0.45 if chart_type == "donut" else 0
                        fig = go.Figure(data=[go.Pie(
                            labels=counts.index.tolist(),
                            values=counts.values.tolist(),
                            hole=hole,
                            marker=dict(colors=px.colors.qualitative.Set2, line=dict(color="white", width=2)),
                            textinfo="percent+label",
                            textfont=dict(size=12),
                            pull=[0.05] * min(3, len(counts)) + [0] * max(0, len(counts) - 3),
                        )])

                # --- QQ PLOT ---
                elif chart_type == "qq":
                    col = spec.get("column", "")
                    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                        clean = df[col].dropna()
                        if len(clean) > 2:
                            theoretical, sample = stats.probplot(clean, dist="norm")[:2]
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=theoretical[0], y=theoretical[1],
                                mode="markers", name="Data",
                                marker=dict(color=GRADIENT_COLORS[0], size=5, opacity=0.7),
                            ))
                            # Reference line
                            x_line = np.array([theoretical[0].min(), theoretical[0].max()])
                            y_line = sample[0] * x_line + sample[1]
                            fig.add_trace(go.Scatter(
                                x=x_line, y=y_line,
                                mode="lines", name="Normal Reference",
                                line=dict(color="#f5576c", dash="dash", width=2),
                            ))
                            fig.update_layout(
                                xaxis_title="Theoretical Quantiles",
                                yaxis_title="Sample Quantiles",
                            )

                else:
                    continue

                if fig is not None:
                    w, h = _adaptive_size(n_rows, len(df.columns))
                    _save_chart(fig, filepath, title, width=w, height=h)
                    generated.append({
                        "chart_type": chart_type,
                        "title": title,
                        "file_path": filepath,
                        "html_path": filepath.replace(".png", ".html"),
                    })
                else:
                    generated.append({
                        "chart_type": chart_type,
                        "title": title,
                        "error": "Insufficient data for this chart type",
                    })

            except Exception as chart_err:
                generated.append({
                    "chart_type": chart_type,
                    "title": title,
                    "error": str(chart_err),
                })

        result = {
            "charts_generated": len([c for c in generated if "error" not in c]),
            "charts": generated,
            "output_directory": charts_dir,
        }
        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})
