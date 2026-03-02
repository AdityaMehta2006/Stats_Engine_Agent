"""
RegressionAnalysisTool -- Runs OLS linear regression with full diagnostics.
Includes VIF (multicollinearity), ACF/PACF (residual autocorrelation),
Durbin-Watson, Jarque-Bera, Breusch-Pagan, and coefficient analysis.
"""

import json
import warnings
import pandas as pd
import numpy as np
from scipy import stats
from crewai.tools import tool

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _safe_round(val, decimals=4):
    """Round a float, returning 0.0 if NaN or Inf."""
    if val is None or np.isnan(val) or np.isinf(val):
        return 0.0
    return round(float(val), decimals)


def _auto_select_target(df: pd.DataFrame, numeric_cols: list) -> str | None:
    """
    Auto-select the best dependent variable for regression.
    Heuristics: pick the numeric column with the highest average absolute
    correlation with other numeric columns (most 'explained' by others).
    """
    if len(numeric_cols) < 2:
        return None

    best_col = None
    best_avg_corr = -1

    corr_matrix = df[numeric_cols].corr().abs()
    for col in numeric_cols:
        # Average correlation with all other columns (excluding self)
        avg_corr = corr_matrix[col].drop(col).mean()
        if avg_corr > best_avg_corr:
            best_avg_corr = avg_corr
            best_col = col

    return best_col


def _run_ols(df: pd.DataFrame, target: str, predictors: list) -> dict:
    """Run OLS regression and return comprehensive results."""
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.stattools import durbin_watson, jarque_bera

    y = df[target].dropna()
    X = df[predictors].loc[y.index].dropna()
    y = y.loc[X.index]

    if len(y) < len(predictors) + 2:
        return {"error": f"Not enough observations ({len(y)}) for {len(predictors)} predictors."}

    # Add constant (intercept)
    X_const = sm.add_constant(X)

    # Fit OLS
    model = sm.OLS(y, X_const).fit()

    # ── MODEL SUMMARY ──────────────────────────────────────────────────────
    model_summary = {
        "r_squared": _safe_round(model.rsquared),
        "adj_r_squared": _safe_round(model.rsquared_adj),
        "f_statistic": _safe_round(model.fvalue),
        "f_pvalue": _safe_round(model.f_pvalue),
        "aic": _safe_round(model.aic),
        "bic": _safe_round(model.bic),
        "n_observations": int(model.nobs),
        "n_predictors": len(predictors),
        "log_likelihood": _safe_round(model.llf),
    }

    # ── COEFFICIENTS ───────────────────────────────────────────────────────
    coefficients = []
    for name, coef, se, t, p, ci_low, ci_high in zip(
        model.params.index,
        model.params.values,
        model.bse.values,
        model.tvalues.values,
        model.pvalues.values,
        model.conf_int()[0].values,
        model.conf_int()[1].values,
    ):
        coefficients.append({
            "variable": name,
            "coefficient": _safe_round(coef),
            "std_error": _safe_round(se),
            "t_statistic": _safe_round(t),
            "p_value": _safe_round(p),
            "ci_lower": _safe_round(ci_low),
            "ci_upper": _safe_round(ci_high),
            "significant": bool(p < 0.05),
        })

    # ── VIF (Variance Inflation Factor) ────────────────────────────────────
    vif_results = []
    if len(predictors) > 1:
        try:
            X_vif = X.copy()
            X_vif_const = sm.add_constant(X_vif)
            for i, col in enumerate(X_vif_const.columns):
                vif_val = variance_inflation_factor(X_vif_const.values, i)
                if col != "const":
                    vif_results.append({
                        "variable": col,
                        "vif": _safe_round(vif_val, 2),
                        "multicollinearity": (
                            "severe" if vif_val > 10
                            else "moderate" if vif_val > 5
                            else "low"
                        ),
                    })
        except Exception as e:
            vif_results = [{"error": str(e)}]

    # ── RESIDUAL DIAGNOSTICS ───────────────────────────────────────────────
    residuals = model.resid.values
    fitted = model.fittedvalues.values

    # Durbin-Watson (autocorrelation)
    dw = durbin_watson(residuals)
    dw_interpretation = (
        "positive autocorrelation" if dw < 1.5
        else "no autocorrelation" if dw <= 2.5
        else "negative autocorrelation"
    )

    # Jarque-Bera (normality of residuals)
    jb_stat, jb_p, jb_skew, jb_kurt = jarque_bera(residuals)

    # Breusch-Pagan (heteroscedasticity)
    try:
        bp_stat, bp_p, bp_fstat, bp_fp = het_breuschpagan(residuals, X_const)
    except Exception:
        bp_stat, bp_p, bp_fstat, bp_fp = 0, 1, 0, 1

    residual_diagnostics = {
        "durbin_watson": {
            "statistic": _safe_round(dw),
            "interpretation": dw_interpretation,
        },
        "jarque_bera": {
            "statistic": _safe_round(jb_stat),
            "p_value": _safe_round(jb_p),
            "residuals_normal": bool(jb_p > 0.05),
            "skewness": _safe_round(jb_skew),
            "kurtosis": _safe_round(jb_kurt),
        },
        "breusch_pagan": {
            "statistic": _safe_round(bp_stat),
            "p_value": _safe_round(bp_p),
            "homoscedastic": bool(bp_p > 0.05),
        },
        "residual_stats": {
            "mean": _safe_round(np.mean(residuals)),
            "std": _safe_round(np.std(residuals)),
            "min": _safe_round(np.min(residuals)),
            "max": _safe_round(np.max(residuals)),
        },
    }

    # ── ACF / PACF of residuals ────────────────────────────────────────────
    acf_pacf = {}
    try:
        from statsmodels.tsa.stattools import acf, pacf

        n_lags = min(20, len(residuals) // 3)
        if n_lags >= 2:
            acf_values = acf(residuals, nlags=n_lags, fft=True)
            pacf_values = pacf(residuals, nlags=n_lags, method="ywm")

            acf_pacf = {
                "n_lags": n_lags,
                "acf_values": [_safe_round(v) for v in acf_values.tolist()],
                "pacf_values": [_safe_round(v) for v in pacf_values.tolist()],
                "confidence_band": _safe_round(1.96 / np.sqrt(len(residuals))),
            }
    except Exception as e:
        acf_pacf = {"error": str(e)}

    # ── STORE RAW DATA for chart generation ────────────────────────────────
    regression_data = {
        "actual": y.tolist(),
        "predicted": fitted.tolist(),
        "residuals": residuals.tolist(),
    }

    return {
        "target_variable": target,
        "predictors": predictors,
        "model_summary": model_summary,
        "coefficients": coefficients,
        "vif": vif_results,
        "residual_diagnostics": residual_diagnostics,
        "acf_pacf": acf_pacf,
        "regression_data": regression_data,
        "model_text_summary": model.summary().as_text(),
    }


@tool("Regression Analysis")
def regression_analysis_tool(file_path: str, target_column: str = "") -> str:
    """
    Run comprehensive OLS linear regression analysis on cleaned CSV data.
    Automatically selects target variable if not specified, runs VIF,
    residual diagnostics (Durbin-Watson, Jarque-Bera, Breusch-Pagan),
    and ACF/PACF analysis.

    Tests performed:
    - OLS Regression with R², Adjusted R², F-statistic, AIC, BIC
    - Coefficient table with confidence intervals and significance
    - VIF for multicollinearity detection
    - Durbin-Watson for residual autocorrelation
    - Jarque-Bera for residual normality
    - Breusch-Pagan for heteroscedasticity
    - ACF/PACF of residuals

    Args:
        file_path: Absolute path to the cleaned CSV file.
        target_column: Optional. Name of the dependent variable. If empty,
                       auto-selects the best target based on correlation analysis.
    """
    try:
        df = pd.read_csv(file_path)

        if df.empty:
            return json.dumps({"error": "DataFrame is empty."})

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if df[c].nunique() > 1]

        if len(numeric_cols) < 2:
            return json.dumps({
                "error": "Need at least 2 numeric columns for regression analysis.",
                "available_numeric_columns": numeric_cols,
            })

        # Determine target variable
        target = target_column.strip() if target_column else ""
        if not target or target not in numeric_cols:
            target = _auto_select_target(df, numeric_cols)
            if not target:
                return json.dumps({"error": "Could not auto-select a target variable."})

        # Predictors = all other numeric columns
        predictors = [c for c in numeric_cols if c != target]

        # Drop rows with NaN in target or predictor columns
        analysis_cols = [target] + predictors
        df_clean = df[analysis_cols].dropna()

        if len(df_clean) < len(predictors) + 2:
            return json.dumps({
                "error": f"Not enough complete rows ({len(df_clean)}) for regression.",
            })

        # Run OLS regression
        ols_results = _run_ols(df_clean, target, predictors)

        if "error" in ols_results:
            return json.dumps(ols_results)

        # Also run basic correlation matrix for context
        corr_matrix = df_clean[analysis_cols].corr()
        correlations_with_target = {
            col: _safe_round(corr_matrix[target][col])
            for col in predictors
        }

        # Descriptive stats
        descriptive = {}
        for col in analysis_cols:
            series = df_clean[col]
            descriptive[col] = {
                "mean": _safe_round(series.mean()),
                "median": _safe_round(series.median()),
                "std": _safe_round(series.std()),
                "min": _safe_round(series.min()),
                "max": _safe_round(series.max()),
                "skewness": _safe_round(series.skew()),
                "kurtosis": _safe_round(series.kurtosis()),
            }

        # Categorical columns for context
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        categorical_cols = [c for c in categorical_cols if df[c].nunique() <= 50]

        output = {
            "analysis_mode": "regression",
            "target_variable": target,
            "predictors": predictors,
            "correlations_with_target": correlations_with_target,
            "descriptive_statistics": descriptive,
            "categorical_columns": categorical_cols,
            **ols_results,
        }

        # Remove the large regression_data from the text output to avoid overwhelming the agent
        # (it's used internally for chart generation)
        agent_output = {k: v for k, v in output.items() if k != "regression_data"}
        agent_output["note"] = (
            "Raw prediction data (actual, predicted, residuals) is available "
            "internally for chart generation but omitted from this summary."
        )

        return json.dumps(agent_output, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})
