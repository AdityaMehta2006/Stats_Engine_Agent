"""
Stats-driven chart generator -- Runs statistical tests first, then creates
Plotly charts that visualize the significant findings. Charts are directly
tied to statistical evidence, not generic exploration.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from itertools import combinations

# Fixed seed for reproducible results
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# Premium theme
THEME = {
    "bg": "#1a1a2e",
    "plot_bg": "#16213e",
    "text": "#e0e0e0",
    "font": "Inter, Arial, sans-serif",
    "gradient": ["#667eea", "#764ba2", "#f093fb", "#f5576c", "#4facfe", "#00f2fe"],
    "discrete": px.colors.qualitative.Set2,
}


def _style(fig, title: str, width=None, height=None):
    """Apply consistent premium dark styling."""
    layout = dict(
        template="plotly_dark",
        font=dict(family=THEME["font"], size=13, color=THEME["text"]),
        title=dict(text=title, font=dict(size=18), x=0.5, xanchor="center"),
        paper_bgcolor=THEME["bg"],
        plot_bgcolor=THEME["plot_bg"],
        margin=dict(l=60, r=40, t=80, b=60),
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
        ),
    )
    if width:
        layout["width"] = width
    if height:
        layout["height"] = height
    fig.update_layout(**layout)
    return fig


def _safe_round(val, n=4):
    """Safely round a value, handling NaN/Inf."""
    if val is None or np.isnan(val) or np.isinf(val):
        return None
    return round(float(val), n)


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICAL ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_data(df: pd.DataFrame) -> dict:
    """
    Run a comprehensive statistical analysis on the DataFrame.
    Returns structured findings that drive chart generation.
    """
    findings = {
        "normality": [],
        "correlations": [],
        "group_comparisons": [],
        "chi_squared": [],
        "distributions": [],
        "categorical_counts": [],
    }

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Filter out constant or near-constant columns
    numeric_cols = [c for c in numeric_cols if df[c].nunique() > 1]
    cat_cols_low = [c for c in cat_cols if 1 < df[c].nunique() <= 20]

    # ── NORMALITY TESTS ─────────────────────────────────────────────────────
    for col in numeric_cols[:8]:
        clean = df[col].dropna()
        if len(clean) < 8:
            continue
        try:
            if len(clean) < 5000:
                stat, p = stats.shapiro(clean)
                test = "Shapiro-Wilk"
            else:
                stat, p = stats.normaltest(clean)
                test = "D'Agostino-Pearson"
            findings["normality"].append({
                "column": col,
                "test": test,
                "p_value": _safe_round(p),
                "is_normal": p > 0.05,
            })
        except Exception:
            pass

    # ── CORRELATIONS (find significant pairs) ────────────────────────────────
    if len(numeric_cols) >= 2:
        for col1, col2 in combinations(numeric_cols[:10], 2):
            clean = df[[col1, col2]].dropna()
            if len(clean) < 10:
                continue
            try:
                r, p = stats.pearsonr(clean[col1], clean[col2])
                if abs(r) > 0.3 and p < 0.05:  # Only significant correlations
                    findings["correlations"].append({
                        "x": col1, "y": col2,
                        "r": _safe_round(r), "p_value": _safe_round(p),
                        "strength": "strong" if abs(r) > 0.7 else "moderate",
                    })
            except Exception:
                pass
        # Sort by absolute correlation
        findings["correlations"].sort(key=lambda x: abs(x["r"]), reverse=True)

    # ── GROUP COMPARISONS (t-test / ANOVA) ───────────────────────────────────
    for cat_col in cat_cols_low[:5]:
        groups = df.groupby(cat_col)
        n_groups = df[cat_col].nunique()
        if n_groups < 2 or n_groups > 10:
            continue

        for num_col in numeric_cols[:6]:
            group_data = [g[num_col].dropna().values for _, g in groups if len(g[num_col].dropna()) >= 3]
            if len(group_data) < 2:
                continue

            try:
                if n_groups == 2:
                    stat, p = stats.mannwhitneyu(group_data[0], group_data[1], alternative="two-sided")
                    test = "Mann-Whitney U"
                else:
                    stat, p = stats.kruskal(*group_data)
                    test = "Kruskal-Wallis"

                if p < 0.05:  # Only significant differences
                    findings["group_comparisons"].append({
                        "numeric": num_col, "group_by": cat_col,
                        "test": test, "statistic": _safe_round(stat),
                        "p_value": _safe_round(p), "n_groups": n_groups,
                    })
            except Exception:
                pass
        # Sort by p-value
        findings["group_comparisons"].sort(key=lambda x: x["p_value"])

    # ── CHI-SQUARED (categorical associations) ──────────────────────────────
    if len(cat_cols_low) >= 2:
        for col1, col2 in combinations(cat_cols_low[:6], 2):
            try:
                ct = pd.crosstab(df[col1], df[col2])
                if ct.shape[0] < 2 or ct.shape[1] < 2:
                    continue
                chi2, p, dof, expected = stats.chi2_contingency(ct)
                if p < 0.05:  # Only significant associations
                    # Cramér's V for effect size
                    n = ct.sum().sum()
                    v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
                    findings["chi_squared"].append({
                        "col1": col1, "col2": col2,
                        "chi2": _safe_round(chi2), "p_value": _safe_round(p),
                        "cramers_v": _safe_round(v),
                        "strength": "strong" if v > 0.3 else "moderate" if v > 0.1 else "weak",
                    })
            except Exception:
                pass
        findings["chi_squared"].sort(key=lambda x: x["p_value"])

    # ── DISTRIBUTION INFO ───────────────────────────────────────────────────
    for col in numeric_cols[:6]:
        clean = df[col].dropna()
        if len(clean) < 5:
            continue
        findings["distributions"].append({
            "column": col,
            "mean": _safe_round(clean.mean()),
            "median": _safe_round(clean.median()),
            "std": _safe_round(clean.std()),
            "skewness": _safe_round(clean.skew()),
            "kurtosis": _safe_round(clean.kurtosis()),
        })

    # ── CATEGORICAL COUNTS ──────────────────────────────────────────────────
    for col in cat_cols_low[:4]:
        counts = df[col].value_counts().head(15)
        findings["categorical_counts"].append({
            "column": col,
            "n_unique": df[col].nunique(),
            "top_values": counts.to_dict(),
        })

    return findings


# ─────────────────────────────────────────────────────────────────────────────
# CHART GENERATION (driven by statistical findings)
# ─────────────────────────────────────────────────────────────────────────────

def generate_charts(csv_path: str) -> list[dict]:
    """
    Run statistical tests and generate charts that visualize the significant findings.

    Returns:
        List of dicts with: title, fig (Plotly Figure), description, stat_info
    """
    df = pd.read_csv(csv_path)
    findings = _analyze_data(df)
    charts = []

    # ── 1. CORRELATION SCATTER PLOTS (for significant correlations) ──────────
    for corr in findings["correlations"][:3]:
        x, y = corr["x"], corr["y"]
        r, p = corr["r"], corr["p_value"]

        # Add color by a categorical column if available
        cat_cols = [f["column"] for f in findings["categorical_counts"]]
        color_col = cat_cols[0] if cat_cols else None

        scatter_kwargs = dict(data_frame=df, x=x, y=y, opacity=0.6)
        if color_col:
            scatter_kwargs["color"] = color_col
            scatter_kwargs["color_discrete_sequence"] = THEME["discrete"]
        else:
            scatter_kwargs["color_discrete_sequence"] = [THEME["gradient"][0]]

        try:
            fig = px.scatter(**scatter_kwargs, trendline="ols")
        except Exception:
            fig = px.scatter(**scatter_kwargs)

        fig.update_traces(marker=dict(size=5, line=dict(width=0.3, color="white")))
        title = f"{x} vs {y} (r={r}, p={p})"
        _style(fig, title)
        charts.append({
            "title": f"Correlation: {x} vs {y}",
            "fig": fig,
            "description": f"Pearson r = {r} (p = {p}). {corr['strength'].title()} {'positive' if r > 0 else 'negative'} correlation.",
            "stat_info": corr,
        })

    # ── 2. GROUP COMPARISON CHARTS (for significant t-test / ANOVA) ──────────
    for comp in findings["group_comparisons"][:3]:
        num_col = comp["numeric"]
        cat_col = comp["group_by"]
        p = comp["p_value"]
        test = comp["test"]

        fig = px.violin(
            df, x=cat_col, y=num_col, color=cat_col,
            color_discrete_sequence=THEME["discrete"],
            box=True, points="outliers",
        )
        title = f"{num_col} by {cat_col} ({test}, p={p})"
        _style(fig, title)
        charts.append({
            "title": f"Group Test: {num_col} by {cat_col}",
            "fig": fig,
            "description": f"{test}: p = {p}. There is a statistically significant difference in '{num_col}' across '{cat_col}' groups.",
            "stat_info": comp,
        })

    # ── 3. CHI-SQUARED ASSOCIATION CHARTS (stacked bars) ─────────────────────
    for chi in findings["chi_squared"][:2]:
        col1, col2 = chi["col1"], chi["col2"]
        p = chi["p_value"]
        v = chi["cramers_v"]

        ct = pd.crosstab(df[col1], df[col2])
        ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

        fig = go.Figure()
        for j, c in enumerate(ct_pct.columns):
            color = THEME["discrete"][j % len(THEME["discrete"])]
            fig.add_trace(go.Bar(
                x=ct_pct.index.tolist(), y=ct_pct[c].values,
                name=str(c), marker_color=color,
            ))
        fig.update_layout(barmode="stack", xaxis_title=col1, yaxis_title="Percentage (%)")
        title = f"{col1} vs {col2} (χ²: p={p}, V={v})"
        _style(fig, title)
        charts.append({
            "title": f"Association: {col1} vs {col2}",
            "fig": fig,
            "description": f"Chi-squared: p = {p}, Cramér's V = {v} ({chi['strength']} association).",
            "stat_info": chi,
        })

    # ── 4. DISTRIBUTION CHARTS (normality results) ──────────────────────────
    normal_cols = [n for n in findings["normality"] if n["is_normal"]]
    non_normal_cols = [n for n in findings["normality"] if not n["is_normal"]]

    # Show distributions of top columns with normality annotations
    dist_cols = findings["distributions"][:4]
    if dist_cols:
        fig = go.Figure()
        for j, dist in enumerate(dist_cols):
            col = dist["column"]
            color = THEME["gradient"][j % len(THEME["gradient"])]
            vals = df[col].dropna()

            # Find normality result for this column
            norm_info = next((n for n in findings["normality"] if n["column"] == col), None)
            label = f"{col} ({'Normal' if norm_info and norm_info['is_normal'] else 'Non-Normal'})"

            fig.add_trace(go.Histogram(
                x=vals, name=label, opacity=0.7,
                marker=dict(color=color, line=dict(color="white", width=0.5)),
                nbinsx=min(50, max(10, len(vals) // 20)),
            ))
        fig.update_layout(barmode="overlay", xaxis_title="Value", yaxis_title="Frequency")
        _style(fig, "Distributions (Normality Test Results)")

        norm_summary = ", ".join([f"{n['column']}(p={n['p_value']})" for n in findings["normality"][:4]])
        charts.append({
            "title": "Distributions & Normality",
            "fig": fig,
            "description": f"Normality tests: {norm_summary}. Columns labelled Normal/Non-Normal based on p > 0.05.",
            "stat_info": findings["normality"][:4],
        })

    # ── 5. QQ PLOT (for a key non-normal column) ────────────────────────────
    if non_normal_cols:
        col = non_normal_cols[0]["column"]
        clean = df[col].dropna()
        if len(clean) > 10:
            theoretical, sample = stats.probplot(clean, dist="norm")[:2]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=theoretical[0], y=theoretical[1],
                mode="markers", name="Data",
                marker=dict(color=THEME["gradient"][0], size=4, opacity=0.6),
            ))
            x_line = np.array([theoretical[0].min(), theoretical[0].max()])
            y_line = sample[0] * x_line + sample[1]
            fig.add_trace(go.Scatter(
                x=x_line, y=y_line,
                mode="lines", name="Normal Reference",
                line=dict(color="#f5576c", dash="dash", width=2),
            ))
            fig.update_layout(xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
            p_val = non_normal_cols[0]["p_value"]
            _style(fig, f"QQ Plot: {col} (p={p_val})")
            charts.append({
                "title": f"QQ Plot: {col}",
                "fig": fig,
                "description": f"QQ plot for '{col}'. Normality test p = {p_val} -- column deviates from normal distribution.",
                "stat_info": non_normal_cols[0],
            })

    # ── 6. CORRELATION HEATMAP (if enough numeric cols) ─────────────────────
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if df[c].nunique() > 1]
    if len(numeric_cols) >= 3:
        corr = df[numeric_cols].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale="RdBu_r",
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=10, color="white"),
        ))
        size = max(600, len(corr.columns) * 65)
        _style(fig, "Correlation Matrix", width=size, height=size)
        charts.append({
            "title": "Correlation Matrix",
            "fig": fig,
            "description": "Full correlation matrix. Significant pairs highlighted in the scatter plots above.",
            "stat_info": {"n_significant": len(findings["correlations"])},
        })

    # ── 7. CATEGORICAL BREAKDOWN (donut for primary category) ───────────────
    if findings["categorical_counts"]:
        cat = findings["categorical_counts"][0]
        col = cat["column"]
        counts = df[col].value_counts().head(10)
        fig = go.Figure(data=[go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            hole=0.45,
            marker=dict(colors=THEME["discrete"], line=dict(color="white", width=2)),
            textinfo="percent+label",
            textfont=dict(size=11),
            pull=[0.05] * min(3, len(counts)) + [0] * max(0, len(counts) - 3),
        )])
        _style(fig, f"{col} Distribution")
        charts.append({
            "title": f"{col} Breakdown",
            "fig": fig,
            "description": f"Proportion breakdown of '{col}' ({cat['n_unique']} unique values).",
            "stat_info": cat,
        })

    # ── 8. RADAR (if group comparisons found) ───────────────────────────────
    if findings["group_comparisons"] and len(numeric_cols) >= 3:
        # Use the first significant grouping
        best_cat = findings["group_comparisons"][0]["group_by"]
        metrics = [c["numeric"] for c in findings["group_comparisons"] if c["group_by"] == best_cat]
        if len(metrics) < 3:
            metrics = numeric_cols[:5]

        grouped = df.groupby(best_cat)[metrics].mean()
        if len(grouped) >= 2:
            max_vals = grouped.max().values
            max_vals[max_vals == 0] = 1

            fig = go.Figure()
            for j, (cat_val, row) in enumerate(grouped.head(6).iterrows()):
                norm_vals = row.values / max_vals
                color = THEME["discrete"][j % len(THEME["discrete"])]
                fig.add_trace(go.Scatterpolar(
                    r=list(norm_vals) + [norm_vals[0]],
                    theta=metrics + [metrics[0]],
                    fill="toself", name=str(cat_val),
                    line=dict(color=color),
                ))

            fig.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0.3)",
                    radialaxis=dict(visible=True, range=[0, 1.1], gridcolor="rgba(255,255,255,0.2)"),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.2)"),
                ),
            )
            _style(fig, f"Radar: Metrics by {best_cat}")
            charts.append({
                "title": f"Radar: {best_cat} Comparison",
                "fig": fig,
                "description": f"Normalized comparison of statistically significant metrics across '{best_cat}' groups.",
                "stat_info": {"group_by": best_cat, "metrics": metrics},
            })

    # ── 9. SUNBURST (if chi-squared found associations) ─────────────────────
    if findings["chi_squared"]:
        chi = findings["chi_squared"][0]
        path = [chi["col1"], chi["col2"]]
        try:
            fig = px.sunburst(
                df, path=path,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig.update_traces(textinfo="label+percent parent")
            _style(fig, f"Hierarchy: {path[0]} -> {path[1]}")
            charts.append({
                "title": f"Sunburst: {path[0]} -> {path[1]}",
                "fig": fig,
                "description": f"Hierarchical breakdown based on chi-squared association (p={chi['p_value']}).",
                "stat_info": chi,
            })
        except Exception:
            pass

    return charts


def save_charts_as_png(charts: list[dict], output_dir: str):
    """Save all chart figures as PNG files."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    for i, chart in enumerate(charts):
        safe_title = "".join(c if c.isalnum() or c == "_" else "_" for c in chart["title"])
        filepath = os.path.join(output_dir, f"{i + 1}_{safe_title}.png")
        try:
            chart["fig"].write_image(filepath, width=1200, height=700, scale=2)
            chart["file_path"] = filepath
        except Exception as e:
            print(f"Warning: Could not save {filepath}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION-SPECIFIC CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def generate_regression_charts(csv_path: str, regression_results: dict = None) -> list[dict]:
    """
    Generate regression-focused charts. If regression_results are not provided,
    runs a fresh OLS regression internally.

    Returns:
        List of dicts with: title, fig (Plotly Figure), description, stat_info
    """
    df = pd.read_csv(csv_path)
    charts = []

    # If no regression results provided, run OLS internally
    if regression_results is None:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if df[c].nunique() > 1]
        if len(numeric_cols) < 2:
            return charts

        # Auto-select target (highest avg correlation)
        corr_matrix = df[numeric_cols].corr().abs()
        target = max(numeric_cols, key=lambda c: corr_matrix[c].drop(c).mean())
        predictors = [c for c in numeric_cols if c != target]

        import statsmodels.api as sm
        analysis_df = df[[target] + predictors].dropna()
        if len(analysis_df) < len(predictors) + 2:
            return charts

        y = analysis_df[target]
        X = sm.add_constant(analysis_df[predictors])
        model = sm.OLS(y, X).fit()

        regression_results = {
            "target_variable": target,
            "predictors": predictors,
            "regression_data": {
                "actual": y.tolist(),
                "predicted": model.fittedvalues.tolist(),
                "residuals": model.resid.tolist(),
            },
            "model_summary": {
                "r_squared": round(model.rsquared, 4),
                "adj_r_squared": round(model.rsquared_adj, 4),
            },
            "coefficients": [
                {
                    "variable": name,
                    "coefficient": round(float(coef), 4),
                    "p_value": round(float(p), 4),
                    "ci_lower": round(float(ci_low), 4),
                    "ci_upper": round(float(ci_high), 4),
                    "significant": p < 0.05,
                }
                for name, coef, p, ci_low, ci_high in zip(
                    model.params.index,
                    model.params.values,
                    model.pvalues.values,
                    model.conf_int()[0].values,
                    model.conf_int()[1].values,
                )
            ],
            "vif": [],
            "acf_pacf": {},
        }

        # VIF
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            X_vif = sm.add_constant(analysis_df[predictors])
            for i, col in enumerate(X_vif.columns):
                if col != "const":
                    vif_val = variance_inflation_factor(X_vif.values, i)
                    regression_results["vif"].append({
                        "variable": col,
                        "vif": round(float(vif_val), 2),
                    })
        except Exception:
            pass

        # ACF/PACF
        try:
            from statsmodels.tsa.stattools import acf, pacf
            residuals = model.resid.values
            n_lags = min(20, len(residuals) // 3)
            if n_lags >= 2:
                regression_results["acf_pacf"] = {
                    "n_lags": n_lags,
                    "acf_values": acf(residuals, nlags=n_lags, fft=True).tolist(),
                    "pacf_values": pacf(residuals, nlags=n_lags, method="ywm").tolist(),
                    "confidence_band": round(1.96 / np.sqrt(len(residuals)), 4),
                }
        except Exception:
            pass

    # Extract data
    target = regression_results.get("target_variable", "Y")
    reg_data = regression_results.get("regression_data", {})
    actual = np.array(reg_data.get("actual", []))
    predicted = np.array(reg_data.get("predicted", []))
    residuals = np.array(reg_data.get("residuals", []))
    r2 = regression_results.get("model_summary", {}).get("r_squared", 0)
    adj_r2 = regression_results.get("model_summary", {}).get("adj_r_squared", 0)

    if len(actual) == 0:
        return charts

    # ── 1. ACTUAL vs PREDICTED ─────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual, y=predicted,
        mode="markers", name="Data Points",
        marker=dict(color=THEME["gradient"][0], size=5, opacity=0.6,
                    line=dict(width=0.3, color="white")),
    ))
    # Perfect prediction line
    line_range = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    fig.add_trace(go.Scatter(
        x=line_range, y=line_range,
        mode="lines", name="Perfect Prediction",
        line=dict(color="#f5576c", dash="dash", width=2),
    ))
    fig.update_layout(xaxis_title="Actual", yaxis_title="Predicted")
    _style(fig, f"Actual vs Predicted (R²={r2})")
    charts.append({
        "title": f"Actual vs Predicted: {target}",
        "fig": fig,
        "description": f"Scatter plot of actual vs predicted values. R² = {r2}, Adjusted R² = {adj_r2}. Points close to the diagonal line indicate accurate predictions.",
        "stat_info": {"r_squared": r2, "adj_r_squared": adj_r2},
    })

    # ── 2. RESIDUALS vs FITTED ─────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=predicted, y=residuals,
        mode="markers", name="Residuals",
        marker=dict(color=THEME["gradient"][1], size=5, opacity=0.6,
                    line=dict(width=0.3, color="white")),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#f5576c", line_width=2)
    fig.update_layout(xaxis_title="Fitted Values", yaxis_title="Residuals")
    _style(fig, "Residuals vs Fitted (Homoscedasticity Check)")
    charts.append({
        "title": "Residuals vs Fitted",
        "fig": fig,
        "description": "Residuals should scatter randomly around zero with no pattern. A funnel shape indicates heteroscedasticity.",
        "stat_info": {},
    })

    # ── 3. QQ PLOT of RESIDUALS ────────────────────────────────────────────
    if len(residuals) > 5:
        theoretical, sample = stats.probplot(residuals, dist="norm")[:2]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=theoretical[0], y=theoretical[1],
            mode="markers", name="Residuals",
            marker=dict(color=THEME["gradient"][0], size=4, opacity=0.7),
        ))
        x_line = np.array([theoretical[0].min(), theoretical[0].max()])
        y_line = sample[0] * x_line + sample[1]
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode="lines", name="Normal Reference",
            line=dict(color="#f5576c", dash="dash", width=2),
        ))
        fig.update_layout(xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
        _style(fig, "QQ Plot of Residuals (Normality Check)")
        charts.append({
            "title": "QQ Plot: Residuals",
            "fig": fig,
            "description": "Points along the reference line indicate normally distributed residuals. Deviations suggest non-normality.",
            "stat_info": regression_results.get("residual_diagnostics", {}).get("jarque_bera", {}),
        })

    # ── 4. RESIDUAL DISTRIBUTION ───────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=residuals, name="Residuals", opacity=0.7,
        marker=dict(color=THEME["gradient"][2], line=dict(color="white", width=0.5)),
        nbinsx=min(50, max(10, len(residuals) // 15)),
    ))
    fig.update_layout(xaxis_title="Residual Value", yaxis_title="Frequency")
    _style(fig, "Residual Distribution")
    charts.append({
        "title": "Residual Distribution",
        "fig": fig,
        "description": "Distribution of regression residuals. Should approximate a bell curve for valid OLS assumptions.",
        "stat_info": {},
    })

    # ── 5. ACF PLOT ────────────────────────────────────────────────────────
    acf_data = regression_results.get("acf_pacf", {})
    if "acf_values" in acf_data:
        acf_vals = acf_data["acf_values"]
        conf = acf_data.get("confidence_band", 0.1)
        n_lags = len(acf_vals)

        fig = go.Figure()
        for i, val in enumerate(acf_vals):
            fig.add_trace(go.Bar(
                x=[i], y=[val], name=f"Lag {i}",
                marker=dict(color=THEME["gradient"][4] if abs(val) <= conf or i == 0 else "#f5576c"),
                showlegend=False,
            ))
        fig.add_hline(y=conf, line_dash="dash", line_color="rgba(255,255,255,0.5)", line_width=1)
        fig.add_hline(y=-conf, line_dash="dash", line_color="rgba(255,255,255,0.5)", line_width=1)
        fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=0.5)
        fig.update_layout(xaxis_title="Lag", yaxis_title="Autocorrelation", bargap=0.3)
        _style(fig, "ACF of Residuals (Autocorrelation)")
        charts.append({
            "title": "ACF: Residuals",
            "fig": fig,
            "description": f"Autocorrelation function of residuals. Bars outside the confidence band (±{conf:.3f}) indicate significant autocorrelation.",
            "stat_info": acf_data,
        })

    # ── 6. PACF PLOT ───────────────────────────────────────────────────────
    if "pacf_values" in acf_data:
        pacf_vals = acf_data["pacf_values"]
        conf = acf_data.get("confidence_band", 0.1)

        fig = go.Figure()
        for i, val in enumerate(pacf_vals):
            fig.add_trace(go.Bar(
                x=[i], y=[val], name=f"Lag {i}",
                marker=dict(color=THEME["gradient"][5] if abs(val) <= conf or i == 0 else "#f5576c"),
                showlegend=False,
            ))
        fig.add_hline(y=conf, line_dash="dash", line_color="rgba(255,255,255,0.5)", line_width=1)
        fig.add_hline(y=-conf, line_dash="dash", line_color="rgba(255,255,255,0.5)", line_width=1)
        fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=0.5)
        fig.update_layout(xaxis_title="Lag", yaxis_title="Partial Autocorrelation", bargap=0.3)
        _style(fig, "PACF of Residuals (Partial Autocorrelation)")
        charts.append({
            "title": "PACF: Residuals",
            "fig": fig,
            "description": f"Partial autocorrelation function. Shows direct lag correlations after removing intermediate effects. Bars outside ±{conf:.3f} are significant.",
            "stat_info": acf_data,
        })

    # ── 7. VIF BAR CHART ──────────────────────────────────────────────────
    vif_data = regression_results.get("vif", [])
    vif_data = [v for v in vif_data if "error" not in v]
    if vif_data:
        vif_df = pd.DataFrame(vif_data)
        colors = [
            "#f5576c" if v > 10 else "#ffa502" if v > 5 else THEME["gradient"][0]
            for v in vif_df["vif"]
        ]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=vif_df["variable"], y=vif_df["vif"],
            marker=dict(color=colors, line=dict(color="white", width=0.5)),
            text=[f"{v:.1f}" for v in vif_df["vif"]],
            textposition="outside",
        ))
        fig.add_hline(y=5, line_dash="dash", line_color="#ffa502", line_width=2,
                      annotation_text="VIF = 5 (moderate)")
        fig.add_hline(y=10, line_dash="dash", line_color="#f5576c", line_width=2,
                      annotation_text="VIF = 10 (severe)")
        fig.update_layout(xaxis_title="Variable", yaxis_title="VIF Score", xaxis_tickangle=-45)
        _style(fig, "Variance Inflation Factor (Multicollinearity)")
        charts.append({
            "title": "VIF: Multicollinearity",
            "fig": fig,
            "description": "VIF scores for each predictor. VIF > 5 suggests moderate multicollinearity, VIF > 10 is severe. Consider removing or combining correlated predictors.",
            "stat_info": vif_data,
        })

    # ── 8. COEFFICIENT PLOT ────────────────────────────────────────────────
    coef_data = regression_results.get("coefficients", [])
    coef_data = [c for c in coef_data if c["variable"] != "const"]
    if coef_data:
        coef_df = pd.DataFrame(coef_data)
        colors = [
            THEME["gradient"][0] if sig else "rgba(100,100,100,0.5)"
            for sig in coef_df["significant"]
        ]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=coef_df["variable"], x=coef_df["coefficient"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="white", width=0.5)),
            error_x=dict(
                type="data",
                symmetric=False,
                array=(coef_df["ci_upper"] - coef_df["coefficient"]).tolist(),
                arrayminus=(coef_df["coefficient"] - coef_df["ci_lower"]).tolist(),
                color="rgba(255,255,255,0.5)",
            ),
            text=[f"β={c:.3f}" for c in coef_df["coefficient"]],
            textposition="outside",
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.5)", line_width=1)
        fig.update_layout(xaxis_title="Coefficient (β)", yaxis_title="Variable")
        _style(fig, "Regression Coefficients with 95% CI")
        charts.append({
            "title": "Coefficient Plot",
            "fig": fig,
            "description": "Regression coefficients with 95% confidence intervals. Colored bars are statistically significant (p < 0.05). Bars crossing zero are not significant.",
            "stat_info": coef_data,
        })

    # ── 9. CORRELATION HEATMAP (reused from general charts) ────────────────
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if df[c].nunique() > 1]
    if len(numeric_cols) >= 3:
        corr = df[numeric_cols].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale="RdBu_r",
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=10, color="white"),
        ))
        size = max(600, len(corr.columns) * 65)
        _style(fig, "Predictor Correlation Matrix", width=size, height=size)
        charts.append({
            "title": "Correlation Matrix",
            "fig": fig,
            "description": "Correlation matrix of all numeric variables. High inter-predictor correlations contribute to multicollinearity (see VIF chart).",
            "stat_info": {},
        })

    return charts
