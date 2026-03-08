"""
feature_drift_detector.py
--------------------------
Production feature drift monitoring.

Metrics computed per feature:
  - Population Stability Index (PSI): measures distributional shift
      PSI < 0.1   → no drift
      PSI 0.1–0.2 → minor drift (monitor)
      PSI > 0.2   → significant drift (alert)
  - Kolmogorov-Smirnov statistic + p-value

Workflow:
  1. Establish a reference distribution (training data).
  2. At serving time, compare each incoming batch against the reference.
  3. Generate a structured drift report.
  4. Raise alerts for features exceeding thresholds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# PSI computation
# ---------------------------------------------------------------------------

def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-8,
) -> float:
    """Compute Population Stability Index between reference and current distributions.

    PSI = sum( (P_current - P_reference) * ln(P_current / P_reference) )

    Parameters
    ----------
    reference : 1-D array of reference values (training set)
    current : 1-D array of current values (serving batch)
    n_bins : number of equal-width bins (based on reference quantiles)
    epsilon : small constant to avoid log(0)

    Returns
    -------
    psi : float
    """
    # Build bins from reference quantiles so bins are equally populated
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(reference, quantiles)
    # Ensure unique edges (edge case: constant column)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0  # constant feature — no drift measurable

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = ref_counts / (ref_counts.sum() + epsilon)
    cur_pct = cur_counts / (cur_counts.sum() + epsilon)

    # Replace zeros with epsilon to avoid log(0)
    ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
    cur_pct = np.where(cur_pct == 0, epsilon, cur_pct)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


# ---------------------------------------------------------------------------
# KS test wrapper
# ---------------------------------------------------------------------------

def ks_drift_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Return (KS statistic, p-value) for reference vs current."""
    stat, pvalue = stats.ks_2samp(reference, current)
    return float(stat), float(pvalue)


# ---------------------------------------------------------------------------
# Drift detector
# ---------------------------------------------------------------------------

PSI_MINOR_THRESHOLD = 0.1
PSI_MAJOR_THRESHOLD = 0.2
KS_PVALUE_THRESHOLD = 0.05


def _psi_severity(psi: float) -> str:
    if psi < PSI_MINOR_THRESHOLD:
        return "ok"
    if psi < PSI_MAJOR_THRESHOLD:
        return "minor"
    return "ALERT"


def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Generate a per-feature drift report.

    Parameters
    ----------
    reference_df : DataFrame of reference (training) feature values
    current_df : DataFrame of current (serving) feature values
    n_bins : PSI bins

    Returns
    -------
    report : DataFrame with columns [feature, psi, psi_severity,
                                      ks_stat, ks_pvalue, ks_drift]
    """
    if not set(reference_df.columns) == set(current_df.columns):
        raise ValueError("reference_df and current_df must have identical columns")

    rows = []
    for col in reference_df.columns:
        ref = reference_df[col].dropna().values
        cur = current_df[col].dropna().values

        psi = compute_psi(ref, cur, n_bins=n_bins)
        ks_stat, ks_pval = ks_drift_test(ref, cur)

        rows.append({
            "feature": col,
            "psi": round(psi, 5),
            "psi_severity": _psi_severity(psi),
            "ks_stat": round(ks_stat, 5),
            "ks_pvalue": round(ks_pval, 5),
            "ks_drift": ks_pval < KS_PVALUE_THRESHOLD,
        })

    report = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    return report


def print_drift_summary(report: pd.DataFrame) -> None:
    """Print a concise drift summary with alerts highlighted."""
    n_alert = (report["psi_severity"] == "ALERT").sum()
    n_minor = (report["psi_severity"] == "minor").sum()
    n_ks = report["ks_drift"].sum()

    print(f"Drift Report — {len(report)} features analysed")
    print(f"  PSI ALERT (>0.2)  : {n_alert} features")
    print(f"  PSI minor (0.1-0.2): {n_minor} features")
    print(f"  KS drift detected  : {n_ks} features")
    print()
    print(report.to_string(index=False))

    if n_alert > 0:
        alerted = report[report["psi_severity"] == "ALERT"]["feature"].tolist()
        print(f"\n[ALERT] Significant drift in: {alerted}")


# ---------------------------------------------------------------------------
# Demo — simulate drift on breast cancer dataset
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    data = load_breast_cancer()
    X_raw = data.data
    feature_names = list(data.feature_names)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    df = pd.DataFrame(X, columns=feature_names)

    # Reference: first 400 samples
    reference_df = df.iloc[:400].copy()

    # Simulate drift: take last 169 samples and add noise + mean shift to some features
    current_df = df.iloc[400:].copy()
    drifted_features = feature_names[:5]
    print(f"Simulating drift in: {drifted_features}\n")

    for feat in drifted_features:
        shift = rng.uniform(0.8, 1.5)
        noise_scale = rng.uniform(0.5, 1.5)
        current_df[feat] = current_df[feat] * noise_scale + shift

    report = generate_drift_report(reference_df, current_df, n_bins=10)
    print_drift_summary(report)
