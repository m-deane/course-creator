"""
Module 01: MIDAS Fundamentals — Self-Check Exercises

Tests understanding of the MIDAS equation, weight functions,
and U-MIDAS vs. restricted MIDAS.

Run: python 01_midas_fundamentals_self_check.py
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'module_00_foundations', 'resources')


def load_csv(filename):
    path = os.path.join(RESOURCES_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path, index_col='date', parse_dates=True).squeeze()


def check(cond, desc, hint=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {desc}")
    if not cond and hint:
        print(f"         {hint}")
    return cond


# ============================================================
# Exercise 1: MIDAS Weight Properties
# ============================================================

def exercise_1_weight_properties():
    """Verify that Beta polynomial weights have correct properties."""
    print("\nExercise 1: Beta Polynomial Weight Properties")
    print("-" * 50)

    all_pass = True

    def beta_weights(K, t1, t2):
        if t1 <= 0 or t2 <= 0:
            return np.ones(K) / K
        x = (np.arange(K) + 0.5) / K
        raw = beta_dist.pdf(1 - x, t1, t2)
        s = raw.sum()
        return raw / s if s > 1e-12 else np.ones(K) / K

    # Test 1: Weights sum to 1
    for t1, t2 in [(1.0, 1.0), (1.5, 4.0), (2.0, 2.0), (1.0, 8.0)]:
        w = beta_weights(9, t1, t2)
        all_pass &= check(
            abs(w.sum() - 1.0) < 1e-8,
            f"Beta({t1}, {t2}) weights sum to 1 (sum={w.sum():.10f})"
        )

    # Test 2: All weights non-negative
    for t1, t2 in [(0.5, 3.0), (1.5, 4.0)]:
        w = beta_weights(12, t1, t2)
        all_pass &= check(
            (w >= 0).all(),
            f"Beta({t1}, {t2}) all weights >= 0 (min={w.min():.6f})"
        )

    # Test 3: Beta(1,1) ≈ uniform
    w_uniform = beta_weights(9, 1.0, 1.0)
    max_dev = np.abs(w_uniform - 1/9).max()
    all_pass &= check(
        max_dev < 0.02,
        f"Beta(1,1) ≈ uniform (max deviation from 1/9: {max_dev:.4f})"
    )

    # Test 4: Beta(1, 5) is front-loaded (w[0] > w[8])
    w_decline = beta_weights(9, 1.0, 5.0)
    all_pass &= check(
        w_decline[0] > w_decline[8],
        f"Beta(1,5) front-loaded: w[0]={w_decline[0]:.4f} > w[8]={w_decline[8]:.4f}",
        "theta2 > theta1 should produce declining weights (more weight on recent lags)."
    )

    # Test 5: Beta(5, 1) is back-loaded (w[0] < w[8])
    w_back = beta_weights(9, 5.0, 1.0)
    all_pass &= check(
        w_back[0] < w_back[8],
        f"Beta(5,1) back-loaded: w[0]={w_back[0]:.4f} < w[8]={w_back[8]:.4f}",
        "theta1 > theta2 should put more weight on older lags."
    )

    return all_pass


# ============================================================
# Exercise 2: Almon Polynomial Properties
# ============================================================

def exercise_2_almon_properties():
    """Verify Almon polynomial weight properties."""
    print("\nExercise 2: Almon Polynomial Weight Properties")
    print("-" * 50)

    all_pass = True

    def almon_weights(K, t1, t2):
        j = np.arange(K, dtype=float)
        lx = t1 * j + t2 * j**2
        lx -= lx.max()
        raw = np.exp(lx)
        return raw / raw.sum()

    # Test 1: Weights sum to 1
    for t1, t2 in [(0.0, 0.0), (-0.3, 0.0), (-0.1, -0.02)]:
        w = almon_weights(9, t1, t2)
        all_pass &= check(
            abs(w.sum() - 1.0) < 1e-8,
            f"Almon({t1}, {t2}) weights sum to 1"
        )

    # Test 2: theta1=0, theta2=0 → uniform
    w_unif = almon_weights(9, 0.0, 0.0)
    max_dev = np.abs(w_unif - 1/9).max()
    all_pass &= check(
        max_dev < 1e-8,
        f"Almon(0,0) = uniform (max deviation: {max_dev:.2e})",
        "exp(0) is constant, so all lags get equal weight."
    )

    # Test 3: theta1 < 0 → declining (w[0] > w[8])
    w_decline = almon_weights(9, -0.3, 0.0)
    all_pass &= check(
        w_decline[0] > w_decline[4] > w_decline[8],
        f"Almon(-0.3, 0) declining: {w_decline[0]:.3f} > {w_decline[4]:.3f} > {w_decline[8]:.3f}",
        "Negative theta1 gives geometric decay: w[j] ∝ exp(theta1 * j)."
    )

    # Test 4: Hump shape — peak at interior point
    w_hump = almon_weights(12, -0.1, -0.02)
    peak_idx = np.argmax(w_hump)
    all_pass &= check(
        0 < peak_idx < 11,
        f"Almon(-0.1, -0.02) hump peak at lag {peak_idx} (interior, not endpoint)",
        "The theoretical peak is at j* = -theta1 / (2*theta2) = 2.5."
    )

    # Test 5: Peak location formula
    theoretical_peak = -(-0.1) / (2 * (-0.02))  # = 2.5
    all_pass &= check(
        abs(peak_idx - round(theoretical_peak)) <= 1,
        f"Almon hump peak at {peak_idx} ≈ theoretical {theoretical_peak:.1f}",
        "j* = -theta1 / (2*theta2) should predict the peak lag."
    )

    return all_pass


# ============================================================
# Exercise 3: MIDAS vs. Aggregation
# ============================================================

def exercise_3_midas_vs_aggregation():
    """Test MIDAS model superiority over equal-weight aggregation."""
    print("\nExercise 3: MIDAS vs. Equal-Weight Aggregation")
    print("-" * 50)

    all_pass = True

    from scipy.optimize import minimize
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    try:
        gdp = load_csv('gdp_quarterly.csv')
        ip = load_csv('industrial_production_monthly.csv')
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return False

    gdp.index = pd.PeriodIndex(gdp.index, freq='Q')
    ip.index = pd.PeriodIndex(ip.index, freq='M')

    def beta_weights(K, t1, t2):
        if t1 <= 0 or t2 <= 0:
            return np.ones(K) / K
        x = (np.arange(K) + 0.5) / K
        raw = beta_dist.pdf(1 - x, t1, t2)
        s = raw.sum()
        return raw / s if s > 1e-12 else np.ones(K) / K

    # Build simple MIDAS matrix (K=6: 2 quarterly lags × 3 months)
    K = 6
    T = len(gdp)
    X = np.full((T, K), np.nan)
    for t, q in enumerate(gdp.index):
        lm = q.end_time.to_period('M')
        for j in range(K):
            tgt = lm - j
            if tgt in ip.index:
                X[t, j] = ip.iloc[ip.index.get_loc(tgt)]

    Y = gdp.values
    valid = ~(np.isnan(X).any(1) | np.isnan(Y))
    Y, X = Y[valid], X[valid]

    # OLS on equal-weight aggregate
    x_agg = X.mean(axis=1)
    ols = LinearRegression().fit(x_agg.reshape(-1,1), Y)
    r2_ols = r2_score(Y, ols.predict(x_agg.reshape(-1,1)))

    # MIDAS with optimized Beta weights
    def sse(params, Y, X):
        a, b, t1, t2 = params
        if t1 <= 0 or t2 <= 0:
            return 1e10
        w = beta_weights(X.shape[1], t1, t2)
        return np.sum((Y - a - b * (X @ w))**2)

    b0 = np.cov(Y, x_agg)[0,1] / np.var(x_agg)
    a0 = Y.mean() - b0 * x_agg.mean()

    best_sse = np.inf
    for p0 in [[a0, b0, 1.0, 5.0], [a0, b0, 1.5, 4.0], [a0, b0, 1.0, 1.0]]:
        r = minimize(sse, p0, args=(Y, X), method='Nelder-Mead',
                     options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-6})
        if r.fun < best_sse:
            best_sse = r.fun
            best_params = r.x

    a, b, t1, t2 = best_params
    t1, t2 = max(t1, 0.01), max(t2, 0.01)
    w_opt = beta_weights(K, t1, t2)
    y_hat_midas = a + b * (X @ w_opt)
    r2_midas = r2_score(Y, y_hat_midas)

    all_pass &= check(
        r2_midas >= r2_ols - 0.005,
        f"MIDAS R² ({r2_midas:.4f}) ≥ OLS R² ({r2_ols:.4f})",
        "MIDAS should always fit at least as well as OLS on the aggregate (in sample)."
    )

    all_pass &= check(
        abs(w_opt.sum() - 1.0) < 1e-8,
        f"MIDAS weights sum to 1 (sum={w_opt.sum():.8f})"
    )

    # Uniform Beta(1,1) should reproduce OLS aggregate R²
    w_uniform = beta_weights(K, 1.0, 1.0)
    x_w_uniform = X @ w_uniform
    model_u = LinearRegression().fit(x_w_uniform.reshape(-1,1), Y)
    r2_uniform_midas = r2_score(Y, model_u.predict(x_w_uniform.reshape(-1,1)))

    all_pass &= check(
        abs(r2_uniform_midas - r2_ols) < 0.001,
        f"Beta(1,1) MIDAS ≈ OLS aggregate R² ({r2_uniform_midas:.4f} ≈ {r2_ols:.4f})",
        "Beta(1,1) weights are approximately equal = equal-weight aggregation."
    )

    print(f"\n  Summary: OLS R²={r2_ols:.4f}, MIDAS R²={r2_midas:.4f}, "
          f"Gain={r2_midas - r2_ols:.4f}, Optimal θ=({t1:.2f}, {t2:.2f})")

    return all_pass


# ============================================================
# Exercise 4: U-MIDAS vs. Restricted MIDAS
# ============================================================

def exercise_4_umidas_comparison():
    """Test the U-MIDAS vs restricted MIDAS tradeoff."""
    print("\nExercise 4: U-MIDAS vs. Restricted MIDAS Comparison")
    print("-" * 50)

    all_pass = True

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    try:
        gdp = load_csv('gdp_quarterly.csv')
        ip = load_csv('industrial_production_monthly.csv')
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return False

    gdp.index = pd.PeriodIndex(gdp.index, freq='Q')
    ip.index = pd.PeriodIndex(ip.index, freq='M')

    K = 9  # 3 quarterly lags × 3 months
    T = len(gdp)
    X = np.full((T, K), np.nan)
    for t, q in enumerate(gdp.index):
        lm = q.end_time.to_period('M')
        for j in range(K):
            tgt = lm - j
            if tgt in ip.index:
                X[t, j] = ip.iloc[ip.index.get_loc(tgt)]

    Y = gdp.values
    valid = ~(np.isnan(X).any(1) | np.isnan(Y))
    Y, X = Y[valid], X[valid]
    T_valid = len(Y)

    # U-MIDAS: OLS on all K lags
    umidas = LinearRegression().fit(X, Y)
    r2_umidas = r2_score(Y, umidas.predict(X))
    sse_umidas = np.sum((Y - umidas.predict(X))**2)

    # OLS aggregate for comparison
    x_agg = X.mean(axis=1)
    ols_agg = LinearRegression().fit(x_agg.reshape(-1,1), Y)
    r2_ols = r2_score(Y, ols_agg.predict(x_agg.reshape(-1,1)))
    sse_ols = np.sum((Y - ols_agg.predict(x_agg.reshape(-1,1)))**2)

    # AIC/BIC comparison
    def aic(sse, k, n): return n * np.log(sse/n) + 2*k
    def bic(sse, k, n): return n * np.log(sse/n) + k*np.log(n)

    aic_ols = aic(sse_ols, 2, T_valid)
    bic_ols = bic(sse_ols, 2, T_valid)
    aic_umidas = aic(sse_umidas, K+1, T_valid)
    bic_umidas = bic(sse_umidas, K+1, T_valid)

    print(f"  OLS-aggregate (k=2): R²={r2_ols:.4f}, AIC={aic_ols:.2f}, BIC={bic_ols:.2f}")
    print(f"  U-MIDAS (k={K+1}): R²={r2_umidas:.4f}, AIC={aic_umidas:.2f}, BIC={bic_umidas:.2f}")

    # U-MIDAS should have higher in-sample R²
    all_pass &= check(
        r2_umidas >= r2_ols - 0.001,
        f"U-MIDAS R² ({r2_umidas:.4f}) ≥ OLS aggregate R² ({r2_ols:.4f})",
        "OLS on more regressors always improves or maintains in-sample fit."
    )

    # BIC should penalize U-MIDAS more (it has more params)
    all_pass &= check(
        bic_umidas > bic_ols,
        f"BIC penalizes U-MIDAS ({bic_umidas:.2f}) more than OLS ({bic_ols:.2f})",
        "BIC = n*log(SSE/n) + k*log(n). U-MIDAS has k=K+1 vs k=2 for OLS."
    )

    # K/T ratio check
    kt_ratio = K / T_valid
    all_pass &= check(
        kt_ratio < 0.20,
        f"K/T ratio = {K}/{T_valid} = {kt_ratio:.3f} — U-MIDAS marginally viable",
        "K/T < 0.10 is ideal for U-MIDAS. Between 0.10-0.20 is borderline."
    )

    return all_pass


# ============================================================
# Main runner
# ============================================================

def main():
    print("=" * 55)
    print("Module 01 Self-Check: MIDAS Fundamentals")
    print("=" * 55)

    results = [
        exercise_1_weight_properties(),
        exercise_2_almon_properties(),
        exercise_3_midas_vs_aggregation(),
        exercise_4_umidas_comparison(),
    ]

    n_passed = sum(results)
    n_total = len(results)

    print("\n" + "=" * 55)
    print(f"Overall: {n_passed}/{n_total} exercises passed")

    if n_passed == n_total:
        print("All exercises passed. Ready for Module 02 (Estimation & Inference).")
    else:
        failed = [i + 1 for i, r in enumerate(results) if not r]
        print(f"Exercises {failed} need review.")
        print("Re-read the corresponding guide sections before continuing.")

    print("=" * 55)
    return 0 if n_passed == n_total else 1


if __name__ == '__main__':
    sys.exit(main())
