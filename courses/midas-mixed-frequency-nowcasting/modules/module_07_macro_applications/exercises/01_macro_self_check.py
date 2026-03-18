"""
Module 07 — Macroeconomic Applications: Self-Check Exercise
=============================================================

Four tasks covering GDP nowcasting, ragged-edge handling, inflation nowcasting,
and labour market forecasting.

  1. Implement ragged-edge carry-forward fill
  2. Build a MIDAS GDP nowcast and compute news decomposition
  3. Implement energy CPI nowcast with daily oil signal
  4. Claims-payrolls correlation and MIDAS-W model

Run: python 01_macro_self_check.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

np.random.seed(2024)

# ──────────────────────────────────────────────────────────────────────────────
# Data generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_test_data():
    """Generate macro test data for all four tasks."""
    dates_m = pd.date_range('2005-01-01', '2022-12-01', freq='MS')
    dates_q = pd.date_range('2005-01-01', '2022-10-01', freq='QS')
    dates_w = pd.date_range('2005-01-01', '2022-12-31', freq='W-THU')
    T_m, T_q, T_w = len(dates_m), len(dates_q), len(dates_w)

    # Common factor
    factor = np.zeros(T_m)
    for t in range(1, T_m):
        factor[t] = 0.85 * factor[t-1] + np.random.randn()

    # Monthly indicators
    monthly_df = pd.DataFrame(index=dates_m)
    monthly_df['INDPRO'] = 0.7 * factor + 0.4 * np.random.randn(T_m)
    monthly_df['PAYEMS'] = 0.6 * factor + 0.3 * np.random.randn(T_m)
    monthly_df['RETAILSL'] = 0.5 * factor + 0.5 * np.random.randn(T_m)
    monthly_df['UMCSENT'] = 0.4 * factor + 0.6 * np.random.randn(T_m)

    # Quarterly GDP
    gdp_vals = []
    for q in range(T_q):
        m = q * 3
        if m + 2 >= T_m:
            break
        f_q = factor[m:m+3].mean()
        gdp_vals.append(2.0 + 0.8 * f_q + 0.4 * np.random.randn())
    T_q_eff = len(gdp_vals)
    gdp = pd.Series(gdp_vals, index=dates_q[:T_q_eff])

    # Daily oil returns
    daily_dates = pd.date_range('2005-01-01', '2022-12-31', freq='B')
    T_d = len(daily_dates)
    oil_ret = np.zeros(T_d)
    for t in range(1, T_d):
        oil_ret[t] = 0.97 * oil_ret[t-1] + 0.02 * np.random.randn()
    oil_daily = pd.Series(oil_ret, index=daily_dates)

    # Monthly energy CPI (driven by oil)
    energy_cpi = np.zeros(T_m)
    for m in range(T_m):
        d_start = min(m * 22, T_d - 22)
        oil_m = oil_ret[d_start:d_start+22].sum()
        energy_cpi[m] = 0.3 + 0.6 * oil_m + 0.15 * np.random.randn()
    energy_cpi_series = pd.Series(energy_cpi, index=dates_m)

    # Weekly initial claims
    claims = np.zeros(T_w)
    for i, wd in enumerate(dates_w):
        mi = min((wd.year - 2005) * 12 + wd.month - 1, T_m - 1)
        claims[i] = max(150, 220 - 35 * factor[mi] + 12 * np.random.randn())
    claims_series = pd.Series(claims, index=dates_w)

    # Monthly payrolls
    payrolls = 150 + 50 * factor + 20 * np.random.randn(T_m)
    payrolls_series = pd.Series(payrolls, index=dates_m)

    return monthly_df, gdp, oil_daily, energy_cpi_series, claims_series, payrolls_series


monthly_df, gdp, oil_daily, energy_cpi, claims_series, payrolls = generate_test_data()

print("Data ready:")
print(f"  Monthly indicators: {monthly_df.shape}")
print(f"  Quarterly GDP: {len(gdp)}")
print(f"  Daily oil: {len(oil_daily)}")
print(f"  Monthly energy CPI: {len(energy_cpi)}")
print(f"  Weekly claims: {len(claims_series)}")
print(f"  Monthly payrolls: {len(payrolls)}")
print()

# ──────────────────────────────────────────────────────────────────────────────
# TASK 1: Ragged-edge handling
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TASK 1: Ragged-edge carry-forward fill")
print("=" * 60)

# Simulate a ragged edge: last 2 months of one indicator are NaN
df_ragged = monthly_df.copy()
df_ragged.iloc[-2:, 0] = np.nan  # INDPRO: last 2 months missing
df_ragged.iloc[-1, 2] = np.nan   # RETAILSL: last month missing

print(f"Ragged-edge NaN count: {df_ragged.isna().sum().to_dict()}")


def fill_ragged_edge_forward(df):
    """
    Fill ragged edge using carry-forward (ffill) within each series.
    Only fills trailing NaN values, not internal gaps.
    """
    df_filled = df.copy()
    for col in df.columns:
        series = df_filled[col]
        # Find last valid index
        last_valid = series.last_valid_index()
        if last_valid is None:
            continue
        # Fill values after last_valid with the last_valid value
        nan_mask = (df_filled.index > last_valid) & df_filled[col].isna()
        if nan_mask.any():
            df_filled.loc[nan_mask, col] = series[last_valid]
    return df_filled


df_filled = fill_ragged_edge_forward(df_ragged)

# VERIFY Task 1
assert df_filled.isna().sum().sum() == 0, \
    f"Still {df_filled.isna().sum().sum()} NaN values after fill. Check your implementation."
assert df_filled.iloc[-1, 0] == monthly_df.iloc[-3, 0], \
    f"INDPRO should be filled with value from -3 position. Got {df_filled.iloc[-1,0]} vs expected {monthly_df.iloc[-3,0]}"
print(f"Filled NaN count: {df_filled.isna().sum().sum()}")
print("TASK 1 PASSED: Ragged-edge filled correctly with carry-forward.")
print()

# ──────────────────────────────────────────────────────────────────────────────
# TASK 2: GDP nowcast with news decomposition
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TASK 2: GDP nowcast and news decomposition")
print("=" * 60)

# Build simple MIDAS features (3 monthly lags per indicator)
def build_gdp_features(monthly_df, gdp_dates, n_lags=3):
    rows = []
    dates = []
    for qdate in gdp_dates:
        avail = monthly_df[monthly_df.index < qdate]
        if len(avail) < n_lags:
            continue
        row = {}
        for col in monthly_df.columns:
            lags = avail[col].diff().dropna().tail(n_lags).values[::-1]
            if len(lags) < n_lags:
                lags = np.zeros(n_lags)
            for j, v in enumerate(lags, start=1):
                row[f'{col}_L{j}'] = v
        rows.append(row)
        dates.append(qdate)
    X = pd.DataFrame(rows, index=dates).fillna(0)
    return X


X_gdp = build_gdp_features(monthly_df, gdp.index)
y_gdp = gdp.reindex(X_gdp.index).values
T_gdp = len(y_gdp)

# Train on first 70%
train_end_gdp = int(T_gdp * 0.7)
sc_gdp = StandardScaler()
X_tr_s = sc_gdp.fit_transform(X_gdp.values[:train_end_gdp])
X_te_s = sc_gdp.transform(X_gdp.values)

model_gdp = Ridge(alpha=1.0)
model_gdp.fit(X_tr_s, y_gdp[:train_end_gdp])
gdp_nowcasts = model_gdp.predict(X_te_s)
oos_rmse = np.sqrt(mean_squared_error(y_gdp[train_end_gdp:], gdp_nowcasts[train_end_gdp:]))

# News decomposition for last quarter
coefs = model_gdp.coef_
feature_names = X_gdp.columns.tolist()
indicators = list(monthly_df.columns)
n_lags = 3
X_last = X_te_s[-1]

news = {}
for ind in indicators:
    ind_idx = [i for i, f in enumerate(feature_names) if f.startswith(ind)]
    contribution = np.sum(coefs[ind_idx] * X_last[ind_idx])
    news[ind] = contribution

print(f"GDP Nowcast OOS RMSE: {oos_rmse:.4f}%")
print(f"\nNews decomposition (last quarter):")
for ind, contrib in sorted(news.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {ind}: {contrib:+.4f}%")
print(f"  Intercept: {model_gdp.intercept_:+.4f}%")
print(f"  Total nowcast: {gdp_nowcasts[-1]:.4f}%")
print(f"  Actual: {y_gdp[-1]:.4f}%")

# VERIFY Task 2
assert oos_rmse < 5.0, f"GDP RMSE too high: {oos_rmse:.4f}"
assert len(news) == len(indicators), f"Should have {len(indicators)} indicators, got {len(news)}"
assert all(isinstance(v, float) for v in news.values()), "News contributions should be floats"
print("\nTASK 2 PASSED: GDP nowcast and news decomposition computed.")
print()

# ──────────────────────────────────────────────────────────────────────────────
# TASK 3: Energy CPI nowcast with daily oil
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TASK 3: Energy CPI nowcast with daily oil returns")
print("=" * 60)

K_oil = 22

def build_oil_features(oil_daily, target_dates, K=22):
    rows = []
    dates = []
    for date in target_dates:
        avail = oil_daily[oil_daily.index < date]
        if len(avail) < K:
            continue
        lags = avail.tail(K).values[::-1]
        rows.append(lags)
        dates.append(date)
    return np.array(rows), pd.DatetimeIndex(dates)


X_oil_arr, oil_dates = build_oil_features(oil_daily, energy_cpi.index, K=K_oil)
y_en = energy_cpi.reindex(oil_dates).values

# Fit simple regression: monthly sum of oil returns → energy CPI
oil_monthly_sum = X_oil_arr.sum(axis=1)
X_lin = np.column_stack([np.ones(len(y_en)), oil_monthly_sum])
beta_oil, _, _, _ = np.linalg.lstsq(X_lin, y_en, rcond=None)
fitted_oil = X_lin @ beta_oil
linear_rmse = np.sqrt(mean_squared_error(y_en, fitted_oil))

# AR(1) benchmark
ar_en = np.full(len(y_en), np.nan)
for t in range(1, len(y_en)):
    X_ar = np.column_stack([np.ones(t), y_en[:t]])
    b, _, _, _ = np.linalg.lstsq(X_ar, y_en[:t], rcond=None)
    if len(b) >= 2:
        ar_en[t] = b[0] + b[1] * y_en[t-1]

valid_ar = ~np.isnan(ar_en)
ar_rmse = np.sqrt(mean_squared_error(y_en[valid_ar], ar_en[valid_ar]))

# Correlation: monthly oil sum vs energy CPI
corr_oil_cpi = np.corrcoef(oil_monthly_sum, y_en)[0, 1]

print(f"Oil-to-CPI correlation: {corr_oil_cpi:.4f}")
print(f"AR(1) energy CPI RMSE: {ar_rmse:.4f}")
print(f"Linear oil → CPI RMSE: {linear_rmse:.4f}")
print(f"Improvement over AR(1): {(1 - linear_rmse/ar_rmse)*100:.1f}%")

# VERIFY Task 3
assert abs(corr_oil_cpi) > 0.3, f"Oil-CPI correlation should be > 0.3, got {corr_oil_cpi:.4f}"
assert linear_rmse < ar_rmse, f"Oil regression should beat AR(1): {linear_rmse:.4f} vs {ar_rmse:.4f}"
print("\nTASK 3 PASSED: Energy CPI nowcast with oil signal.")
print()

# ──────────────────────────────────────────────────────────────────────────────
# TASK 4: Claims-payrolls MIDAS-W
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TASK 4: Claims-payrolls MIDAS-W model")
print("=" * 60)

K_claims = 4

def build_claims_features(weekly_claims, monthly_dates, K=4):
    rows = []
    dates = []
    for date in monthly_dates:
        avail = weekly_claims[weekly_claims.index < date]
        if len(avail) < K:
            continue
        lags = avail.tail(K).values[::-1]
        rows.append(lags)
        dates.append(date)
    return np.array(rows), pd.DatetimeIndex(dates)


X_c, c_dates = build_claims_features(claims_series, payrolls.index, K=K_claims)
y_p = payrolls.reindex(c_dates).values

def beta_weights(K, t1, t2):
    x = np.linspace(0.001, 0.999, K)
    u = x**(t1 - 1) * (1 - x)**(t2 - 1)
    return u / u.sum()


def claims_midas_sse(params, X, y):
    mu, phi, t1, t2 = params
    if t1 <= 0 or t2 <= 0:
        return 1e10
    w = beta_weights(K_claims, t1, t2)
    return np.sum((y - mu - phi * (X @ w))**2)


train_end_c = int(len(y_p) * 0.7)

result = minimize(
    claims_midas_sse,
    x0=[150, -0.8, 1.0, 1.0],
    args=(X_c[:train_end_c], y_p[:train_end_c]),
    method='L-BFGS-B',
    bounds=[(None, None), (None, 0), (0.01, 20), (0.01, 20)]
)

mu_c, phi_c, t1_c, t2_c = result.x
w_c = beta_weights(K_claims, t1_c, t2_c)
fitted_c = mu_c + phi_c * (X_c @ w_c)
oos_rmse_c = np.sqrt(mean_squared_error(y_p[train_end_c:], fitted_c[train_end_c:]))

# Simple average baseline
avg_claims = X_c.mean(axis=1)
X_avg_c = np.column_stack([np.ones(train_end_c), avg_claims[:train_end_c]])
beta_avg_c, _, _, _ = np.linalg.lstsq(X_avg_c, y_p[:train_end_c], rcond=None)
avg_fitted_c = beta_avg_c[0] + beta_avg_c[1] * avg_claims
avg_rmse_c = np.sqrt(mean_squared_error(y_p[train_end_c:], avg_fitted_c[train_end_c:]))

print(f"MIDAS-W estimates: μ={mu_c:.1f}, φ={phi_c:.4f}, θ₁={t1_c:.3f}, θ₂={t2_c:.3f}")
print(f"Beta weights: {np.round(w_c, 4)}")
print(f"Simple avg claims OOS RMSE: {avg_rmse_c:.1f}k")
print(f"MIDAS-W OOS RMSE:           {oos_rmse_c:.1f}k")
print(f"Phi sign: {'negative (correct — high claims → lower payrolls)' if phi_c < 0 else 'POSITIVE (unexpected)'}")

# VERIFY Task 4
assert phi_c < 0, f"Claims coefficient should be negative, got {phi_c:.4f}"
assert oos_rmse_c < 500, f"RMSE too high: {oos_rmse_c:.1f}k"
assert abs(w_c.sum() - 1.0) < 1e-8, "Weights should sum to 1"
print("\nTASK 4 PASSED: Claims-payrolls MIDAS-W computed.")
print()

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("ALL TASKS COMPLETED")
print("=" * 60)
print()
print("Summary:")
print(f"  T1: Ragged-edge fill → {df_filled.isna().sum().sum()} NaN remaining (should be 0)")
print(f"  T2: GDP MIDAS RMSE = {oos_rmse:.4f}%")
print(f"  T3: Energy CPI improvement = {(1 - linear_rmse/ar_rmse)*100:.1f}% over AR(1)")
print(f"  T4: Claims-payrolls φ = {phi_c:.3f} (negative = correct sign)")
