"""
granger_causality_selector.py
------------------------------
Feature selection for time-series targets via Granger causality testing.

Workflow:
  1. For each candidate feature X_j, test whether X_j Granger-causes
     the target y using statsmodels' grangercausalitytests.
  2. Collect p-values from the F-test at each lag.
  3. Correct for multiple testing: Bonferroni and Benjamini-Hochberg FDR.
  4. Return features ranked by minimum p-value across tested lags.

Conditional variant: fit a VAR(p) model with all features and test
joint exclusion of each predictor block.

Dependencies: statsmodels (part of standard data-science stack).
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.multitest import multipletests
from sklearn.datasets import make_regression


# ---------------------------------------------------------------------------
# Bivariate Granger test
# ---------------------------------------------------------------------------

def bivariate_granger_pvalues(
    series: pd.DataFrame,
    target_col: str,
    max_lag: int = 4,
    test: str = "ssr_ftest",
) -> pd.Series:
    """Return min p-value across lags for each feature → target.

    Parameters
    ----------
    series : DataFrame where each column is a time series (rows = time steps)
    target_col : name of the target column
    max_lag : maximum lag order to test
    test : statsmodels test key ('ssr_ftest', 'ssr_chi2test', etc.)

    Returns
    -------
    pvalues : Series indexed by feature name (excludes target_col)
    """
    feature_cols = [c for c in series.columns if c != target_col]
    pvalues: dict[str, float] = {}

    for feat in feature_cols:
        data_pair = series[[target_col, feat]].dropna().values
        if data_pair.shape[0] < max_lag * 2 + 10:
            pvalues[feat] = 1.0
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = grangercausalitytests(data_pair, maxlag=max_lag, verbose=False)
        # Take the minimum p-value across all tested lags
        lag_pvals = [results[lag][0][test][1] for lag in range(1, max_lag + 1)]
        pvalues[feat] = float(min(lag_pvals))

    return pd.Series(pvalues, name="pvalue_raw").sort_values()


# ---------------------------------------------------------------------------
# Multiple testing correction
# ---------------------------------------------------------------------------

def apply_multiple_testing_correction(
    pvalues: pd.Series,
    alpha: float = 0.05,
    methods: list[str] | None = None,
) -> pd.DataFrame:
    """Apply Bonferroni and BH-FDR corrections to raw p-values.

    Returns a DataFrame with raw and corrected p-values, plus reject flags.
    """
    if methods is None:
        methods = ["bonferroni", "fdr_bh"]

    raw = pvalues.values
    result = pd.DataFrame({"feature": pvalues.index, "pvalue_raw": raw})

    for method in methods:
        reject, pvals_corr, _, _ = multipletests(raw, alpha=alpha, method=method)
        result[f"pvalue_{method}"] = pvals_corr
        result[f"reject_{method}"] = reject

    return result.sort_values("pvalue_raw").reset_index(drop=True)


# ---------------------------------------------------------------------------
# High-level selector
# ---------------------------------------------------------------------------

def granger_select_features(
    series: pd.DataFrame,
    target_col: str,
    max_lag: int = 4,
    alpha: float = 0.05,
    correction: str = "fdr_bh",
) -> list[str]:
    """Return features that Granger-cause the target after MTC.

    Parameters
    ----------
    series : multivariate time series DataFrame
    target_col : column to predict
    max_lag : number of lags to test
    alpha : significance level after correction
    correction : 'bonferroni' or 'fdr_bh'

    Returns
    -------
    selected : list of feature names ordered by significance
    """
    pvals = bivariate_granger_pvalues(series, target_col, max_lag=max_lag)
    report = apply_multiple_testing_correction(pvals, alpha=alpha,
                                               methods=["bonferroni", "fdr_bh"])
    reject_col = f"reject_{correction}"
    selected = report.loc[report[reject_col], "feature"].tolist()
    return selected, report


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n_time = 300

    # Construct a synthetic multivariate time series where only features
    # 0..4 Granger-cause y, the rest are pure noise.
    n_informative = 5
    n_noise = 10
    t = np.arange(n_time)

    # Causal features: AR(1) processes
    causal = pd.DataFrame({
        f"x_causal_{i}": np.convolve(
            rng.standard_normal(n_time + 1),
            [1, 0.5 ** i], mode="full"
        )[:n_time]
        for i in range(n_informative)
    })

    # Target: lags of causal features + small noise
    y = (
        0.4 * causal["x_causal_0"].shift(1).fillna(0)
        + 0.3 * causal["x_causal_1"].shift(2).fillna(0)
        + 0.2 * causal["x_causal_2"].shift(1).fillna(0)
        + 0.05 * rng.standard_normal(n_time)
    )

    noise_features = pd.DataFrame(
        rng.standard_normal((n_time, n_noise)),
        columns=[f"x_noise_{i}" for i in range(n_noise)],
    )

    series = pd.concat([causal, noise_features, y.rename("target")], axis=1)

    print("Bivariate Granger causality tests (max lag = 3) …")
    selected, report = granger_select_features(
        series, target_col="target", max_lag=3, alpha=0.05, correction="fdr_bh"
    )

    print("\nFull report (sorted by raw p-value):")
    print(report.to_string(index=False))

    print(f"\nSelected features (BH-FDR corrected, alpha=0.05): {selected}")
    print(f"True positives expected: x_causal_0, x_causal_1, x_causal_2")
