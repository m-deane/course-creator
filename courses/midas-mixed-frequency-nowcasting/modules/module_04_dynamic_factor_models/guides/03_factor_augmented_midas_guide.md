# Factor-Augmented MIDAS

> **Reading time:** ~18 min | **Module:** 04 — Dynamic Factor Models | **Prerequisites:** Module 3


## In Brief


<div class="callout-key">

**Key Concept Summary:** Factor-Augmented MIDAS (FA-MIDAS) combines the dimensionality reduction of DFMs with the mixed-frequency handling of MIDAS. Extract a common factor from a panel of monthly indicators via PCA, then use

</div>

Factor-Augmented MIDAS (FA-MIDAS) combines the dimensionality reduction of DFMs with the mixed-frequency handling of MIDAS. Extract a common factor from a panel of monthly indicators via PCA, then use the factor time series as the MIDAS predictor. The result: better nowcast accuracy than single-indicator MIDAS with fewer parameters than multi-indicator MIDAS.

## Key Insight

<div class="callout-insight">

**Insight:** Factor models compress many noisy indicators into a few latent factors, which acts as a form of regularization. This is why DFM-based nowcasts often outperform single-indicator MIDAS models.

</div>


The factor acts as a noise-reduced aggregate of all available monthly indicators. By running MIDAS on the factor rather than on raw IP, we automatically incorporate information from employment, retail sales, and other indicators while maintaining the parsimonious 4-parameter MIDAS structure.

---

## Implementation

<div class="callout-warning">

**Warning:** Be cautious about extrapolating MIDAS performance from stable periods to crisis periods. The relationship between high-frequency indicators and the low-frequency target can shift dramatically during regime changes.

</div>


### Step 1: Load and Standardize the Monthly Panel


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_monthly_panel(series_dict):
    """
    Load multiple monthly series and return a standardized panel.

    Parameters
    ----------
    series_dict : dict of str -> pd.Series
        Monthly indicator series (e.g., {'IP': ip_series, 'Payrolls': pay_series})

    Returns
    -------
    panel_std : pd.DataFrame (T_months, N) — standardized panel
    scaler : StandardScaler — for inverse transform
    """
    panel = pd.DataFrame(series_dict)
    panel = panel.dropna()  # Drop rows with any missing values (balanced panel)

    scaler = StandardScaler()
    panel_std = pd.DataFrame(
        scaler.fit_transform(panel),
        index=panel.index,
        columns=panel.columns
    )
    return panel_std, scaler
```

</div>
</div>

### Step 2: Extract Common Factor


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def extract_factors(panel_std, n_factors=1, verbose=True):
    """
    Extract n_factors common factors from standardized monthly panel via PCA.

    Returns
    -------
    factors : pd.DataFrame (T_months, n_factors)
    loadings : pd.DataFrame (N, n_factors)
    var_explained : np.ndarray
    """
    pca = PCA(n_components=n_factors)
    F = pca.fit_transform(panel_std.values)  # (T, q)

    factor_cols = [f'F{i+1}' for i in range(n_factors)]
    factors = pd.DataFrame(F, index=panel_std.index, columns=factor_cols)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=panel_std.columns,
        columns=factor_cols
    )

    if verbose:
        print(f"Factor extraction: q={n_factors}")
        print(f"Variance explained: {pca.explained_variance_ratio_.round(4)}")
        print(f"Total: {pca.explained_variance_ratio_.sum():.4f}")
        print("\nFactor loadings:")
        print(loadings.round(4).to_string())

    return factors, loadings, pca.explained_variance_ratio_


def normalize_factor_sign(factors, loadings, reference_indicator):
    """
    Normalize factor sign so reference indicator has positive loading.
    This ensures consistent interpretation across expanding-window estimations.
    """
    if loadings.loc[reference_indicator, 'F1'] < 0:
        factors['F1'] = -factors['F1']
        loadings['F1'] = -loadings['F1']
    return factors, loadings
```

</div>
</div>

### Step 3: Build MIDAS Matrix from Factor Series


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def build_midas_matrix_from_factor(Y_quarterly, F_monthly, K, m=3):
    """
    Build MIDAS matrix using the monthly factor as the high-frequency predictor.

    Parameters
    ----------
    Y_quarterly : pd.Series (quarterly GDP growth)
    F_monthly : pd.Series (monthly factor series)
    K : int — number of monthly lags

    Returns
    -------
    Y : np.ndarray (T,)
    X : np.ndarray (T, K)
    """
    if hasattr(Y_quarterly.index, 'to_period'):
        y_q = Y_quarterly.copy()
        y_q.index = Y_quarterly.index.to_period('Q')
    else:
        y_q = Y_quarterly.copy()
        y_q.index = pd.PeriodIndex(Y_quarterly.index, freq='Q')

    if hasattr(F_monthly.index, 'to_period'):
        f_m = F_monthly.copy()
        f_m.index = F_monthly.index.to_period('M')
    else:
        f_m = F_monthly.copy()
        f_m.index = pd.PeriodIndex(F_monthly.index, freq='M')

    rows_Y, rows_X = [], []
    for q in y_q.index:
        last_month = q.asfreq('M', how='end')
        lags = [last_month - i for i in range(K)]
        if any(lag not in f_m.index for lag in lags):
            continue
        rows_Y.append(y_q[q])
        rows_X.append([f_m[lag] for lag in lags])

    return np.array(rows_Y), np.array(rows_X)
```


### Step 4: Estimate FA-MIDAS

```python
def estimate_fa_midas(Y_quarterly, F_monthly, K=12):
    """
    Full Factor-Augmented MIDAS estimation pipeline.

    1. Build MIDAS matrix from monthly factor series
    2. Estimate Beta MIDAS by profile NLS
    3. Return results with factor context
    """
    Y, X = build_midas_matrix_from_factor(Y_quarterly, F_monthly, K)
    est = estimate_midas(Y, X)

    r2 = 1 - est['sse'] / np.sum((Y - Y.mean())**2)
    print(f"FA-MIDAS (K={K}):")
    print(f"  beta   = {est['beta']:.4f}")
    print(f"  theta  = ({est['theta1']:.4f}, {est['theta2']:.4f})")
    print(f"  R²     = {r2:.4f}")

    return est, Y, X
```

---

## Expanding-Window FA-MIDAS

The full nowcasting pipeline with expanding-window evaluation:

```python
def fa_midas_expanding_window(Y_quarterly, monthly_panel, K, n_factors=1, min_train=30):
    """
    Expanding-window FA-MIDAS nowcast evaluation.

    At each step:
    1. Re-estimate factor loadings on training data
    2. Project test month onto factor space
    3. Build MIDAS matrix including test observation
    4. Nowcast with estimated MIDAS model

    Returns: list of squared forecast errors.
    """
    T = len(Y_quarterly)
    Y_arr = Y_quarterly.values if hasattr(Y_quarterly, 'values') else Y_quarterly
    sq_errors = []

    # Pre-build monthly factor for all time (re-estimated each step)
    for t in range(min_train, T):
        # Get training quarterly indices
        # Re-extract factor on training monthly data
        panel_train = monthly_panel[monthly_panel.index <= monthly_panel.index[t * 3]]

        if len(panel_train) < min_train * 3 + K:
            sq_errors.append(np.nan)
            continue

        scaler = StandardScaler()
        panel_std = scaler.fit_transform(panel_train.values)

        pca = PCA(n_components=n_factors)
        F_train = pca.fit_transform(panel_std)

        # Sign normalization: first column of pca.components_ should have positive IP loading
        if pca.components_[0, 0] < 0:
            pca.components_[0] = -pca.components_[0]
            F_train[:, 0] = -F_train[:, 0]

        F_series = pd.Series(F_train[:, 0], index=panel_train.index)

        # Build MIDAS matrix from the factor series
        Y_tr = Y_arr[:t]
        Y_pd = pd.Series(Y_tr, index=Y_quarterly.index[:t])
        Y_al, X_al = build_midas_matrix_from_factor(Y_pd, F_series, K)

        if len(Y_al) < 15:
            sq_errors.append(np.nan)
            continue

        est = estimate_midas(Y_al, X_al, starts=[(1.0, 5.0), (1.5, 4.0)])

        # Generate nowcast for period t
        # Use factor at the test observation
        x_test_raw = scaler.transform(monthly_panel.iloc[t*3:t*3+K].values[::-1])
        x_test_factor = pca.transform(x_test_raw)[:, 0][:K]

        w = beta_weights(K, est['theta1'], est['theta2'])
        xw_test = float(x_test_factor @ w)
        y_hat = est['alpha'] + est['beta'] * xw_test
        sq_errors.append((Y_arr[t] - y_hat)**2)

    valid = [e for e in sq_errors if not np.isnan(e)]
    rmse = np.sqrt(np.mean(valid)) if valid else np.nan
    print(f"FA-MIDAS RMSE: {rmse:.4f} ({len(valid)} valid observations)")
    return sq_errors
```

---

## Comparison with Single-Indicator MIDAS

```python
def compare_midas_fa_midas(Y_quarterly, ip_monthly, monthly_panel,
                            K=12, n_factors=1, min_train=30):
    """
    Side-by-side comparison of standard MIDAS vs FA-MIDAS.
    Returns RMSE for both models.
    """
    # Standard MIDAS (IP only)
    Y, X_ip = build_midas_matrix_from_factor(Y_quarterly, ip_monthly, K)
    sq_err_midas = []
    for t in range(min_train, len(Y)):
        est = estimate_midas(Y[:t], X_ip[:t], starts=[(1.0, 5.0), (1.5, 4.0)])
        w = beta_weights(K, est['theta1'], est['theta2'])
        y_hat = est['alpha'] + est['beta'] * float(X_ip[t] @ w)
        sq_err_midas.append((Y[t] - y_hat)**2)

    rmse_midas = np.sqrt(np.mean(sq_err_midas))

    # FA-MIDAS (factor from multiple indicators)
    sq_err_fa = fa_midas_expanding_window(Y_quarterly, monthly_panel, K, n_factors, min_train)
    valid_fa = [e for e in sq_err_fa if not np.isnan(e)]
    rmse_fa = np.sqrt(np.mean(valid_fa)) if valid_fa else np.nan

    print(f"\nModel Comparison:")
    print(f"  MIDAS (IP only):  RMSE = {rmse_midas:.4f}")
    print(f"  FA-MIDAS (q={n_factors}):  RMSE = {rmse_fa:.4f}")
    if rmse_fa < rmse_midas:
        improvement = (rmse_midas - rmse_fa) / rmse_midas * 100
        print(f"  FA-MIDAS improves by {improvement:.1f}%")
    else:
        degradation = (rmse_fa - rmse_midas) / rmse_midas * 100
        print(f"  MIDAS is better by {degradation:.1f}% (factor not helpful here)")

    return rmse_midas, rmse_fa
```

---

## Common Pitfalls

**Pitfall 1: Forgetting sign normalization.** PCA signs are arbitrary — the same factor can be extracted with opposite sign in different expanding-window iterations. Always normalize: ensure the IP loading (or another reference indicator) is positive.

**Pitfall 2: Using test-period data to extract the factor.** The factor must be extracted using only training data. Projecting the test observation onto the training-period factor space is correct; re-extracting with the test data included is look-ahead bias.

**Pitfall 3: Not accounting for factor estimation error.** The two-step approach treats estimated factors as if they were known. This understates parameter uncertainty. For inference, use bootstrap that includes the factor extraction step.

---

## Connections

<div class="callout-danger">

**Danger:** Never use future information when constructing the high-frequency regressor matrix. In a real-time nowcasting context, you only have data up to the current date -- using the full quarter of monthly data when nowcasting mid-quarter is a look-ahead bias that invalidates your results.



- **Builds on:** Guide 01 (PCA factor extraction), Guide 02 (mixed-frequency DFM)
- **Leads to:** Notebooks 01-03 (implementation and comparison)
- **Related to:** Bai-Ng (2006) factor-augmented regression, Stock-Watson FAVAR

---



## Practice Problems

1. Sketch the FA-MIDAS expanding-window algorithm. At step $t$ (estimating on quarters 1 to $t-1$, predicting quarter $t$): (a) what data goes into the PCA? (b) how is the factor for the test quarter computed? (c) where is look-ahead bias possible?

2. If the factor loading of S&P 500 is negative ($\hat{\lambda}_{SP500} = -0.42$), what does this imply about the relationship between equity markets and the business cycle in our PCA normalization?

3. The expanding-window FA-MIDAS re-estimates the factor loadings at each step. In early steps (t=30), the loadings are estimated on only ~90 monthly observations. In later steps (t=90), they're estimated on ~270 observations. How might this affect the factor stability and the consequent MIDAS estimates?


---

## Cross-References

<a class="link-card" href="./01_factor_models_guide.md">
  <div class="link-card-title">01 Factor Models</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_factor_models_slides.md">
  <div class="link-card-title">01 Factor Models — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./02_dfm_mixed_frequency_guide.md">
  <div class="link-card-title">02 Dfm Mixed Frequency</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_dfm_mixed_frequency_slides.md">
  <div class="link-card-title">02 Dfm Mixed Frequency — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

