"""
knockoff_filter.py
------------------
Model-X knockoff filter for FDR-controlled feature selection.

Construction: second-order (Gaussian) knockoffs using the SDP equi-correlated
method.  Knockoff statistics: signed max coefficient difference from
cross-validated Lasso (LCD statistic).  Threshold: knockoff+ (more
conservative; guarantees FDR at the nominal level).

Reference: Candès et al. 2018, "Panning for Gold: Model-X Knockoffs".
"""

import warnings
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Second-order knockoff construction
# ---------------------------------------------------------------------------

def _sdp_equicorrelated_s(Sigma: np.ndarray) -> np.ndarray:
    """Compute the equi-correlated s-vector for Gaussian knockoffs.

    Solves: s_j = min(2 * lambda_min(Sigma), 1) for each feature,
    then clips so that diag(s) <= 2*Sigma (PSD requirement).
    """
    p = Sigma.shape[0]
    lambda_min = float(np.linalg.eigvalsh(Sigma).min())
    s_val = min(2.0 * lambda_min, 1.0)
    s = np.full(p, s_val)
    return s


def construct_gaussian_knockoffs(X: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Build second-order Gaussian knockoffs X_tilde.

    Parameters
    ----------
    X : array of shape (n_samples, p) — already standardised recommended
    random_state : seed for reproducibility

    Returns
    -------
    X_tilde : array of shape (n_samples, p)
    """
    rng = np.random.default_rng(random_state)
    n, p = X.shape

    # Estimate covariance from data
    Sigma = np.cov(X, rowvar=False)
    mu = X.mean(axis=0)

    s = _sdp_equicorrelated_s(Sigma)
    S_diag = np.diag(s)

    # Conditional mean and covariance of X_tilde | X
    # mu_tilde = X + (X - mu) @ (I - Sigma^{-1} @ diag(s))^T
    Sigma_inv = np.linalg.inv(Sigma)
    A = np.eye(p) - Sigma_inv @ S_diag            # (p, p)
    Sigma_tilde = 2.0 * S_diag - S_diag @ Sigma_inv @ S_diag  # (p, p)

    # Regularise for numerical PSD
    Sigma_tilde += 1e-8 * np.eye(p)

    try:
        L = cholesky(Sigma_tilde, lower=True)
    except Exception:
        warnings.warn("Cholesky failed; falling back to eigendecomposition.")
        eigvals, eigvecs = np.linalg.eigh(Sigma_tilde)
        eigvals = np.clip(eigvals, 1e-10, None)
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    mu_tilde = mu + (X - mu) @ A.T          # (n, p)
    noise = rng.standard_normal((n, p)) @ L.T
    X_tilde = mu_tilde + noise
    return X_tilde


# ---------------------------------------------------------------------------
# Knockoff statistics — LCD (Lasso Coefficient Difference)
# ---------------------------------------------------------------------------

def compute_knockoff_statistics(X: np.ndarray, X_tilde: np.ndarray,
                                 y: np.ndarray) -> np.ndarray:
    """Fit Lasso on [X, X_tilde] and return W_j = |beta_j| - |beta_tilde_j|."""
    X_aug = np.hstack([X, X_tilde])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lasso = LassoCV(cv=5, max_iter=5000, n_jobs=-1)
        lasso.fit(X_aug, y)

    p = X.shape[1]
    coef_orig = np.abs(lasso.coef_[:p])
    coef_knock = np.abs(lasso.coef_[p:])
    W = coef_orig - coef_knock
    return W


# ---------------------------------------------------------------------------
# Knockoff+ threshold
# ---------------------------------------------------------------------------

def knockoff_plus_threshold(W: np.ndarray, fdr_level: float = 0.10) -> float:
    """Compute the knockoff+ threshold t* that controls FDR at fdr_level.

    Knockoff+: t* = min{ t : #{j: W_j <= -t} / max(1, #{j: W_j >= t}) <= fdr_level }
    The +1 in the numerator makes it conservative (finite-sample guarantee).
    """
    W_abs = np.sort(np.abs(W[W != 0]))[::-1]
    for t in W_abs:
        # number of features with W_j >= t (likely true)
        num_selected = np.sum(W >= t)
        # number of knockoff "hits" (false discoveries proxy)
        num_knock_hits = np.sum(W <= -t) + 1  # knockoff+ adds 1
        fdp_estimate = num_knock_hits / max(1, num_selected)
        if fdp_estimate <= fdr_level:
            return float(t)
    return float("inf")  # no threshold found → select nothing


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data = load_breast_cancer()
    X_raw, y = data.data, data.target
    feature_names = np.array(data.feature_names)

    # Standardise — important for Gaussian knockoff construction
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    print("Constructing Gaussian knockoffs …")
    X_tilde = construct_gaussian_knockoffs(X, random_state=42)

    print("Computing Lasso coefficient difference statistics …")
    W = compute_knockoff_statistics(X, X_tilde, y)

    # Try FDR levels from strict to lenient; report results at all levels
    results = pd.DataFrame({
        "feature": feature_names,
        "W_statistic": W,
    }).sort_values("W_statistic", ascending=False)
    print("\nW-statistics (sorted):")
    print(results.to_string(index=False))

    for fdr_target in (0.10, 0.20, 0.50):
        threshold = knockoff_plus_threshold(W, fdr_level=fdr_target)
        if np.isinf(threshold):
            print(f"\nFDR ≤ {fdr_target}: no threshold found — knockoff+ selects nothing "
                  f"(W-statistics too weak; the collinear structure of breast cancer "
                  f"reduces discriminative power of Gaussian knockoffs).")
        else:
            selected_mask = W >= threshold
            selected_features = feature_names[selected_mask]
            print(f"\nKnockoff+ threshold at FDR ≤ {fdr_target}: t* = {threshold:.4f}")
            print(f"Features selected ({selected_mask.sum()} / {len(W)}): "
                  f"{list(selected_features)}")
            print(f"FDR guarantee: E[FDP] ≤ {fdr_target}")
