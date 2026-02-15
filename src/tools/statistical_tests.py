"""
StatisticalTestTool -- Runs appropriate statistical tests on cleaned data.
Automatically selects tests based on data characteristics (normality, groups, types).
"""

import json
import warnings
import pandas as pd
import numpy as np
from scipy import stats
from crewai.tools import tool

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _check_normality(series: pd.Series) -> dict:
    """Run normality test on a numeric series."""
    clean = series.dropna()
    if len(clean) < 8:
        return {"test": "skipped", "p_value": None, "is_normal": False, "reason": "too few values"}

    if len(clean) < 5000:
        stat, p = stats.shapiro(clean)
        test_name = "Shapiro-Wilk"
    else:
        stat, p = stats.normaltest(clean)  # D'Agostino-Pearson
        test_name = "D'Agostino-Pearson"

    return {
        "test": test_name,
        "statistic": round(float(stat), 6),
        "p_value": round(float(p), 6),
        "is_normal": p > 0.05,
    }


def _correlation_tests(df: pd.DataFrame, numeric_cols: list) -> list:
    """Run Pearson or Spearman correlations between numeric columns."""
    results = []
    if len(numeric_cols) < 2:
        return results

    tested = set()
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i + 1:]:
            pair_key = tuple(sorted([col1, col2]))
            if pair_key in tested:
                continue
            tested.add(pair_key)

            clean = df[[col1, col2]].dropna()
            if len(clean) < 5:
                continue

            # Check normality of both columns to decide test type
            norm1 = _check_normality(clean[col1])
            norm2 = _check_normality(clean[col2])

            if norm1.get("is_normal") and norm2.get("is_normal"):
                corr, p = stats.pearsonr(clean[col1], clean[col2])
                test_name = "Pearson Correlation"
            else:
                corr, p = stats.spearmanr(clean[col1], clean[col2])
                test_name = "Spearman Correlation"

            strength = "negligible"
            abs_corr = abs(corr)
            if abs_corr >= 0.7:
                strength = "strong"
            elif abs_corr >= 0.4:
                strength = "moderate"
            elif abs_corr >= 0.2:
                strength = "weak"

            direction = "positive" if corr > 0 else "negative"

            results.append({
                "test_name": test_name,
                "columns_tested": [col1, col2],
                "statistic": round(float(corr), 6),
                "p_value": round(float(p), 6),
                "significant": p < 0.05,
                "interpretation": f"{strength} {direction} correlation (r={corr:.3f}, p={p:.4f})",
            })

    return results


def _group_comparison_tests(df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> list:
    """Run group comparison tests (t-test, ANOVA, Mann-Whitney, Kruskal-Wallis)."""
    results = []

    for cat_col in categorical_cols:
        groups = df[cat_col].dropna().unique()
        if len(groups) < 2 or len(groups) > 20:
            continue

        for num_col in numeric_cols:
            group_data = [df[df[cat_col] == g][num_col].dropna() for g in groups]
            group_data = [g for g in group_data if len(g) >= 3]

            if len(group_data) < 2:
                continue

            # Check normality
            all_normal = all(_check_normality(g).get("is_normal", False) for g in group_data)

            if len(group_data) == 2:
                if all_normal:
                    stat, p = stats.ttest_ind(group_data[0], group_data[1])
                    test_name = "Independent t-test"
                else:
                    stat, p = stats.mannwhitneyu(group_data[0], group_data[1], alternative="two-sided")
                    test_name = "Mann-Whitney U"
            else:
                if all_normal:
                    stat, p = stats.f_oneway(*group_data)
                    test_name = "One-way ANOVA"
                else:
                    stat, p = stats.kruskal(*group_data)
                    test_name = "Kruskal-Wallis"

            sig_text = "significant" if p < 0.05 else "not significant"
            results.append({
                "test_name": test_name,
                "columns_tested": [cat_col, num_col],
                "statistic": round(float(stat), 6),
                "p_value": round(float(p), 6),
                "significant": p < 0.05,
                "interpretation": f"{test_name}: {sig_text} difference in {num_col} across {cat_col} groups (p={p:.4f})",
            })

    return results


def _categorical_tests(df: pd.DataFrame, categorical_cols: list) -> list:
    """Run Chi-squared tests between categorical columns."""
    results = []
    if len(categorical_cols) < 2:
        return results

    tested = set()
    for i, col1 in enumerate(categorical_cols):
        for col2 in categorical_cols[i + 1:]:
            pair_key = tuple(sorted([col1, col2]))
            if pair_key in tested:
                continue
            tested.add(pair_key)

            # Limit to columns with reasonable cardinality
            if df[col1].nunique() > 20 or df[col2].nunique() > 20:
                continue

            contingency = pd.crosstab(df[col1], df[col2])
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                continue

            chi2, p, dof, expected = stats.chi2_contingency(contingency)

            sig_text = "significant" if p < 0.05 else "not significant"
            results.append({
                "test_name": "Chi-squared test",
                "columns_tested": [col1, col2],
                "statistic": round(float(chi2), 6),
                "p_value": round(float(p), 6),
                "significant": p < 0.05,
                "interpretation": f"Chi-squared: {sig_text} association between {col1} and {col2} (chi2={chi2:.3f}, p={p:.4f}, dof={dof})",
            })

    return results


@tool("Statistical Tests")
def statistical_test_tool(file_path: str) -> str:
    """
    Run a comprehensive suite of statistical tests on cleaned CSV data.
    Automatically selects appropriate tests based on data types and distributions.

    Tests performed:
    - Normality tests (Shapiro-Wilk / D'Agostino-Pearson)
    - Correlations (Pearson for normal data, Spearman for non-normal)
    - Group comparisons (t-test/ANOVA for normal, Mann-Whitney/Kruskal-Wallis for non-normal)
    - Chi-squared tests for categorical associations

    Args:
        file_path: Absolute path to the cleaned CSV file.
    """
    try:
        df = pd.read_csv(file_path)

        # Guard: empty or single-row DataFrame
        if df.empty:
            return json.dumps({"error": "DataFrame is empty, no data to analyze."})
        if len(df) < 2:
            return json.dumps({"error": "Need at least 2 rows for statistical analysis."})

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Filter out high-cardinality categoricals (likely IDs)
        categorical_cols = [c for c in categorical_cols if df[c].nunique() <= 50]

        # Filter out constant numeric columns (zero variance -- no useful stats)
        numeric_cols = [c for c in numeric_cols if df[c].nunique() > 1]

        # Classify data
        total = len(numeric_cols) + len(categorical_cols)
        if total == 0:
            return json.dumps({"error": "No analyzable columns found (all constant or high-cardinality)."})

        numeric_ratio = len(numeric_cols) / total if total > 0 else 0
        if numeric_ratio > 0.7:
            classification = "mostly numeric"
        elif numeric_ratio < 0.3:
            classification = "mostly categorical"
        else:
            classification = "mixed"

        all_results = []

        # Normality tests
        normality_results = []
        for col in numeric_cols:
            norm = _check_normality(df[col])
            normality_results.append({
                "column": col,
                **norm,
            })

        # Correlations
        correlation_results = _correlation_tests(df, numeric_cols)
        all_results.extend(correlation_results)

        # Group comparisons
        group_results = _group_comparison_tests(df, numeric_cols, categorical_cols)
        all_results.extend(group_results)

        # Categorical associations
        chi2_results = _categorical_tests(df, categorical_cols)
        all_results.extend(chi2_results)

        # Descriptive stats with NaN/Inf protection
        def _safe_round(val, decimals=4):
            """Round a float, returning 0.0 if NaN or Inf."""
            if val is None or np.isnan(val) or np.isinf(val):
                return 0.0
            return round(float(val), decimals)

        descriptive = {}
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue
            descriptive[col] = {
                "mean": _safe_round(series.mean()),
                "median": _safe_round(series.median()),
                "std": _safe_round(series.std()),
                "min": _safe_round(series.min()),
                "max": _safe_round(series.max()),
                "skewness": _safe_round(series.skew()),
                "kurtosis": _safe_round(series.kurtosis()),
            }

        output = {
            "data_classification": classification,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "normality_tests": normality_results,
            "descriptive_statistics": descriptive,
            "test_results": all_results,
            "total_tests_run": len(all_results),
            "significant_findings": len([r for r in all_results if r.get("significant")]),
        }

        return json.dumps(output, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})
