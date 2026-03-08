# Granger Causality for Feature Selection in Time Series

## In Brief

Granger causality tests whether the past values of one time series improve forecasts of another beyond what can be predicted from the target's own history. For feature selection, it provides a statistically grounded ranking of which candidate features carry directional predictive information about the target.

## Key Insight

Granger causality is not causality in the philosophical sense — it is *predictive precedence*. Feature $X$ Granger-causes target $Y$ if knowing the past of $X$ reduces forecast error for future values of $Y$, after accounting for $Y$'s own past. This distinction is critical: a barometer Granger-causes rain because falling pressure precedes rain, not because pressure causes rain.

---

## 1. The Bivariate Granger Causality Test

### 1.1 Formal Setup

Let $y_t$ be the target series and $x_t$ be a candidate feature, both stationary. We compare two autoregressive models:

**Restricted model** (target's own history only):

$$y_t = \alpha_0 + \sum_{l=1}^{L} \alpha_l y_{t-l} + \varepsilon_t^R$$

**Unrestricted model** (target history + feature history):

$$y_t = \alpha_0 + \sum_{l=1}^{L} \alpha_l y_{t-l} + \sum_{l=1}^{L} \beta_l x_{t-l} + \varepsilon_t^U$$

where $L$ is the lag order and $\varepsilon_t^R$, $\varepsilon_t^U$ are residuals.

### 1.2 The F-Test

$X$ Granger-causes $Y$ if the joint null hypothesis $H_0: \beta_1 = \beta_2 = \cdots = \beta_L = 0$ is rejected.

The test statistic is:

$$F = \frac{(RSS_R - RSS_U) / L}{RSS_U / (T - 2L - 1)} \sim F(L, T - 2L - 1)$$

where $RSS_R$ and $RSS_U$ are the residual sums of squares from the restricted and unrestricted models respectively, and $T$ is the sample size.

**Decision rule:** Reject $H_0$ at significance level $\alpha$ if $F > F_{\alpha}(L, T - 2L - 1)$.

### 1.3 Lag Order Selection

The lag order $L$ must be chosen before testing. Common approaches:

| Criterion | Formula | Preference |
|---|---|---|
| AIC | $-2\ln \hat{L} + 2k$ | Larger models; good prediction |
| BIC | $-2\ln \hat{L} + k \ln T$ | Smaller models; consistent |
| HQIC | $-2\ln \hat{L} + 2k \ln \ln T$ | Between AIC and BIC |

where $k$ is the number of parameters. For financial time series, lag orders of 1–5 (daily), 1–12 (monthly), or 1–4 (quarterly) are typical starting points.

### 1.4 Python Implementation: Bivariate Granger Test

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def bivariate_granger_test(
    target: pd.Series,
    feature: pd.Series,
    max_lag: int = 5,
    verbose: bool = False
) -> dict:
    """
    Perform bivariate Granger causality test.

    Parameters
    ----------
    target : pd.Series
        The target variable (Y).
    feature : pd.Series
        The candidate feature (X).
    max_lag : int
        Maximum lag order to test (1 through max_lag).
    verbose : bool
        Print detailed output.

    Returns
    -------
    dict with keys:
        'min_pvalue'   : float  - smallest p-value across lags
        'best_lag'     : int    - lag yielding smallest p-value
        'pvalues'      : dict   - {lag: p_value} for all lags tested
        'f_stats'      : dict   - {lag: F_statistic}
    """
    # Align and drop NaNs
    data = pd.concat([target.rename('y'), feature.rename('x')], axis=1).dropna()

    results = grangercausalitytests(data[['y', 'x']], maxlag=max_lag, verbose=verbose)

    pvalues = {}
    f_stats = {}
    for lag, res in results.items():
        # res[0]['ssr_ftest'] is (F-stat, p-value, df_denom, df_num)
        f_stat, p_val, _, _ = res[0]['ssr_ftest']
        pvalues[lag] = p_val
        f_stats[lag] = f_stat

    best_lag = min(pvalues, key=pvalues.get)

    return {
        'min_pvalue': pvalues[best_lag],
        'best_lag': best_lag,
        'pvalues': pvalues,
        'f_stats': f_stats,
    }


def granger_feature_ranking(
    target: pd.Series,
    features: pd.DataFrame,
    max_lag: int = 5,
) -> pd.DataFrame:
    """
    Rank all features by Granger causality p-value against target.

    Returns DataFrame sorted by ascending p-value (most predictive first).
    """
    records = []
    for col in features.columns:
        result = bivariate_granger_test(target, features[col], max_lag=max_lag)
        records.append({
            'feature': col,
            'min_pvalue': result['min_pvalue'],
            'best_lag': result['best_lag'],
            'f_stat_at_best_lag': result['f_stats'][result['best_lag']],
        })

    ranking = pd.DataFrame(records).sort_values('min_pvalue').reset_index(drop=True)
    return ranking
```

---

## 2. Vector Autoregression (VAR) Formulation

### 2.1 The Multivariate VAR Model

The bivariate test generalises to a $K$-dimensional VAR($L$) system:

$$\mathbf{y}_t = \boldsymbol{\nu} + \mathbf{A}_1 \mathbf{y}_{t-1} + \mathbf{A}_2 \mathbf{y}_{t-2} + \cdots + \mathbf{A}_L \mathbf{y}_{t-L} + \boldsymbol{\varepsilon}_t$$

where $\mathbf{y}_t \in \mathbb{R}^K$, $\mathbf{A}_l \in \mathbb{R}^{K \times K}$ are lag coefficient matrices, and $\boldsymbol{\varepsilon}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$.

The $(i,j)$ element of $\mathbf{A}_l$ captures the effect of variable $j$ at lag $l$ on variable $i$ at time $t$.

### 2.2 Granger Non-Causality in VAR

Variable $X_j$ does **not** Granger-cause $X_i$ if and only if all elements in the $i$-th row, $j$-th column of every lag matrix are zero:

$$[A_l]_{ij} = 0 \quad \forall l \in \{1, \ldots, L\}$$

This is tested with a Wald chi-squared test on the joint restriction.

### 2.3 Python: VAR-Based Granger Testing

```python
from statsmodels.tsa.api import VAR


def var_granger_matrix(
    data: pd.DataFrame,
    target_col: str,
    max_lag: int = 5,
) -> pd.DataFrame:
    """
    Fit VAR model and extract Granger causality p-values for all
    features predicting target_col.

    Parameters
    ----------
    data : pd.DataFrame
        Stationary multivariate time series (features + target).
    target_col : str
        Name of the target column.
    max_lag : int
        Max lag order for VAR selection.

    Returns
    -------
    pd.DataFrame with columns ['feature', 'chi2', 'pvalue', 'df'].
    """
    # Fit VAR with lag selected by BIC
    model = VAR(data)
    lag_order_result = model.select_order(maxlags=max_lag)
    optimal_lag = lag_order_result.bic

    fitted = model.fit(optimal_lag)

    # Test each feature as cause of target
    records = []
    feature_cols = [c for c in data.columns if c != target_col]

    for feat in feature_cols:
        test = fitted.test_causality(
            caused=target_col,
            causing=feat,
            kind='wald',
        )
        records.append({
            'feature': feat,
            'chi2': test.test_statistic,
            'pvalue': test.pvalue,
            'df': test.df,
        })

    return pd.DataFrame(records).sort_values('pvalue').reset_index(drop=True)
```

---

## 3. Conditional Granger Causality

### 3.1 Motivation: Controlling for Confounders

Bivariate Granger causality can be spurious when a confounder $Z$ drives both $X$ and $Y$. Conditional Granger causality (CGC) controls for a set of conditioning variables $\mathbf{Z}$:

**Restricted model** (target history + conditioning variables):

$$y_t = \sum_{l=1}^{L} \alpha_l y_{t-l} + \sum_{l=1}^{L} \gamma_l z_{t-l} + \varepsilon_t^R$$

**Unrestricted model** (adds feature $X$):

$$y_t = \sum_{l=1}^{L} \alpha_l y_{t-l} + \sum_{l=1}^{L} \beta_l x_{t-l} + \sum_{l=1}^{L} \gamma_l z_{t-l} + \varepsilon_t^U$$

$X$ conditionally Granger-causes $Y$ given $\mathbf{Z}$ if $H_0: \beta_1 = \cdots = \beta_L = 0$ is rejected.

### 3.2 Python: Conditional Granger Test

```python
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def conditional_granger_test(
    target: pd.Series,
    feature: pd.Series,
    conditioning: pd.DataFrame,
    lag: int = 3,
) -> dict:
    """
    Conditional Granger causality: test whether feature predicts
    target after conditioning on a set of control variables.

    Parameters
    ----------
    target : pd.Series
        Target variable Y.
    feature : pd.Series
        Feature to test X.
    conditioning : pd.DataFrame
        Control variables Z (e.g., other selected features).
    lag : int
        Lag order.

    Returns
    -------
    dict with 'f_stat', 'pvalue', 'df_num', 'df_denom'.
    """
    # Build lagged design matrix
    def build_lags(series: pd.Series, name: str, lag: int) -> pd.DataFrame:
        return pd.concat(
            {f'{name}_lag{l}': series.shift(l) for l in range(1, lag + 1)},
            axis=1,
        )

    T = len(target)

    # Lagged target
    y_lags = build_lags(target, 'y', lag)
    # Lagged conditioning
    z_lags = pd.concat([build_lags(conditioning[c], c, lag) for c in conditioning.columns], axis=1)
    # Lagged feature
    x_lags = build_lags(feature, 'x', lag)

    # Combine and drop NaN rows
    full_data = pd.concat([target.rename('y'), y_lags, z_lags, x_lags], axis=1).dropna()
    y = full_data['y']

    # Restricted model: Y lags + Z lags
    restricted_cols = [c for c in full_data.columns if c.startswith('y_lag') or c.startswith(tuple(conditioning.columns))]
    X_restricted = add_constant(full_data[restricted_cols])
    res_restricted = OLS(y, X_restricted).fit()

    # Unrestricted model: + X lags
    x_lag_cols = [c for c in full_data.columns if c.startswith('x_lag')]
    X_unrestricted = add_constant(full_data[restricted_cols + x_lag_cols])
    res_unrestricted = OLS(y, X_unrestricted).fit()

    # F-test
    rss_r = res_restricted.ssr
    rss_u = res_unrestricted.ssr
    df_num = lag
    df_denom = len(y) - X_unrestricted.shape[1]
    f_stat = ((rss_r - rss_u) / df_num) / (rss_u / df_denom)
    p_value = 1 - stats.f.cdf(f_stat, df_num, df_denom)

    return {
        'f_stat': f_stat,
        'pvalue': p_value,
        'df_num': df_num,
        'df_denom': df_denom,
    }
```

---

## 4. Nonlinear Granger Causality

### 4.1 Limitations of Linear Granger Testing

Standard Granger tests assume linear relationships. Financial and economic systems exhibit well-documented nonlinearities: volatility clustering, regime switching, threshold effects. A feature with a nonlinear predictive relationship may pass undetected by the linear F-test.

### 4.2 Kernel-Based Nonlinear Granger Causality

The Kernel Granger Causality (KGC) approach replaces linear regression with kernel ridge regression in a reproducing kernel Hilbert space (RKHS):

$$y_t = f(\mathbf{y}_{t-1:t-L}, \mathbf{x}_{t-1:t-L}) + \varepsilon_t$$

where $f$ is estimated in a RKHS defined by a kernel $\kappa$. The test compares prediction error with and without $\mathbf{x}$ lags, using a permutation test for significance:

```python
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score


def kernel_granger_test(
    target: pd.Series,
    feature: pd.Series,
    lag: int = 3,
    kernel: str = 'rbf',
    alpha: float = 0.1,
    n_permutations: int = 200,
    random_state: int = 42,
) -> dict:
    """
    Nonlinear Granger causality via kernel ridge regression
    and permutation testing.

    Returns dict with 'test_stat', 'pvalue'.
    """
    rng = np.random.default_rng(random_state)

    def make_lags(series: np.ndarray, lag: int) -> np.ndarray:
        """Stack lagged vectors row-wise."""
        rows = []
        for t in range(lag, len(series)):
            rows.append(series[t - lag:t])
        return np.array(rows)

    y_arr = target.values
    x_arr = feature.values

    # Align
    n = min(len(y_arr), len(x_arr))
    y_arr, x_arr = y_arr[:n], x_arr[:n]

    y_lags = make_lags(y_arr, lag)
    x_lags = make_lags(x_arr, lag)
    y_target = y_arr[lag:]

    # Model 1 (restricted): predict from y lags only
    model_r = KernelRidge(kernel=kernel, alpha=alpha)
    scores_r = cross_val_score(model_r, y_lags, y_target, cv=5, scoring='neg_mean_squared_error')
    mse_r = -scores_r.mean()

    # Model 2 (unrestricted): predict from y + x lags
    Xy = np.hstack([y_lags, x_lags])
    model_u = KernelRidge(kernel=kernel, alpha=alpha)
    scores_u = cross_val_score(model_u, Xy, y_target, cv=5, scoring='neg_mean_squared_error')
    mse_u = -scores_u.mean()

    # Observed statistic: fractional reduction in MSE
    obs_stat = (mse_r - mse_u) / mse_r

    # Permutation test: shuffle x_lags and recompute
    null_stats = []
    for _ in range(n_permutations):
        x_lags_perm = rng.permutation(x_lags)
        Xy_perm = np.hstack([y_lags, x_lags_perm])
        scores_perm = cross_val_score(model_u, Xy_perm, y_target, cv=5,
                                      scoring='neg_mean_squared_error')
        mse_perm = -scores_perm.mean()
        null_stats.append((mse_r - mse_perm) / mse_r)

    p_value = np.mean(np.array(null_stats) >= obs_stat)

    return {
        'test_stat': obs_stat,
        'pvalue': p_value,
        'mse_restricted': mse_r,
        'mse_unrestricted': mse_u,
    }
```

### 4.3 Neural Network-Based Granger Causality

Neural Granger causality (NGC) uses neural networks with group sparsity regularisation to discover nonlinear Granger structure. The approach (Tank et al., 2018) fits a multi-task recurrent or MLP model where each input group corresponds to a feature; features with zero-group weights do not Granger-cause the target.

The training objective penalises each feature's contribution:

$$\mathcal{L}(\theta) = \sum_t \| y_t - f_\theta(\mathbf{y}_{t-L:t-1}, \mathbf{x}_{t-L:t-1}) \|^2 + \lambda \sum_j \| W_{:,j} \|_2$$

The group-lasso penalty $\lambda \sum_j \| W_{:,j} \|_2$ drives entire feature groups to zero, performing implicit feature selection.

---

## 5. Spectral Granger Causality

### 5.1 Frequency-Domain Decomposition

Spectral (frequency-domain) Granger causality (Geweke, 1982) decomposes causal influence by frequency band. This is valuable when features are predictive only at specific oscillatory frequencies — e.g., a weekly cycle in demand may Granger-cause sales only at the weekly frequency.

From the VAR($L$) model, the spectral representation of causality from $X$ to $Y$ at frequency $\lambda \in [0, \pi]$ is:

$$\mathcal{F}_{X \to Y}(\lambda) = \ln \frac{S_{YY}(\lambda)}{S_{YY}(\lambda) - [\Sigma_{XX} - \Sigma_{XY}^2 / \Sigma_{YY}] |H_{YX}(\lambda)|^2}$$

where $H(\lambda)$ is the transfer function matrix of the VAR (Fourier transform of impulse responses), and $S_{YY}(\lambda)$ is the spectral density of $Y$.

Total causality integrates across all frequencies:

$$\mathcal{F}_{X \to Y} = \frac{1}{2\pi} \int_{-\pi}^{\pi} \mathcal{F}_{X \to Y}(\lambda) \, d\lambda$$

### 5.2 Implementation via `spectral_connectivity`

```python
# spectral Granger causality with mne-connectivity
# pip install mne-connectivity

import numpy as np
from numpy.fft import fft, fftfreq


def spectral_granger_via_var(
    target: np.ndarray,
    feature: np.ndarray,
    lag: int = 5,
    fs: float = 1.0,
) -> dict:
    """
    Approximate spectral Granger causality by computing the
    transfer function of a fitted VAR model.

    Parameters
    ----------
    target, feature : np.ndarray
        Stationary 1-D time series.
    lag : int
        VAR lag order.
    fs : float
        Sampling frequency (Hz). Use 1.0 for unit-time series.

    Returns
    -------
    dict with 'freqs', 'causality_X_to_Y', 'total_causality'.
    """
    from statsmodels.tsa.api import VAR

    data = np.column_stack([target, feature])
    model = VAR(data)
    fitted = model.fit(lag)

    # Coefficient matrices A_1, ..., A_L  shape (L, K, K)
    coefs = fitted.coefs  # shape (L, 2, 2)

    # Build transfer function H(freq)
    n_freqs = 256
    freqs = np.linspace(0, fs / 2, n_freqs)
    H = np.zeros((n_freqs, 2, 2), dtype=complex)

    for i, f in enumerate(freqs):
        omega = 2 * np.pi * f / fs
        A_sum = np.eye(2)
        for l in range(lag):
            A_sum -= coefs[l] * np.exp(-1j * omega * (l + 1))
        H[i] = np.linalg.inv(A_sum)

    sigma = fitted.sigma_u  # (2,2) noise covariance

    # Spectral density S = H * Sigma * H^H
    S = np.einsum('fij,jk,flk->fil', H, sigma, np.conj(H))

    # Spectral causality X->Y (index 1->0)
    S_YY = S[:, 0, 0].real
    H_YX = H[:, 0, 1]
    intrinsic = sigma[1, 1] - sigma[0, 1] ** 2 / sigma[0, 0]
    noise_contrib = intrinsic * np.abs(H_YX) ** 2
    causality = np.log(S_YY / (S_YY - noise_contrib).clip(min=1e-10)).real

    return {
        'freqs': freqs,
        'causality_X_to_Y': causality,
        'total_causality': np.trapz(causality, freqs),
    }
```

---

## 6. Pitfalls and Failure Modes

### 6.1 Non-Stationarity

Granger causality tests require **stationary** time series. Running tests on integrated series (I(1) or higher) produces spurious results — the asymptotic $F$ and chi-squared distributions are invalid.

**Remediation protocol:**
1. Apply Augmented Dickey-Fuller (ADF) or KPSS tests to each series.
2. If non-stationary: difference once, re-test; or test in levels if series are cointegrated (use VECM).
3. For financial returns: use log-returns, not price levels.

```python
from statsmodels.tsa.stattools import adfuller, kpss

def stationarity_check(series: pd.Series, name: str = '') -> dict:
    """ADF and KPSS stationarity tests."""
    adf_stat, adf_p, _, _, adf_crit, _ = adfuller(series.dropna())
    kpss_stat, kpss_p, _, kpss_crit = kpss(series.dropna(), regression='c', nlags='auto')

    is_stationary = (adf_p < 0.05) and (kpss_p > 0.05)

    return {
        'series': name,
        'adf_pvalue': adf_p,
        'kpss_pvalue': kpss_p,
        'is_stationary': is_stationary,
        'recommendation': 'stationary' if is_stationary else 'difference or transform',
    }
```

### 6.2 Spurious Causality from Common Trends

Two series driven by a common latent trend (e.g., both trending upward with GDP) will appear to Granger-cause each other even after differencing. Conditional Granger causality controlling for trend proxies or cointegration residuals resolves this.

### 6.3 Instantaneous Effects

The standard Granger test uses strictly lagged values and misses same-period effects. If $X_t$ and $Y_t$ are contemporaneously correlated (e.g., intraday trading), instantaneous causality tests (Geweke decomposition) or structural VAR identification are needed.

### 6.4 Lag Misspecification

Choosing too few lags misses slow-moving effects; too many lags inflates model complexity, reduces power, and may introduce multicollinearity. Always select lag order using information criteria, not manual inspection.

---

## 7. Multiple Testing Correction

When testing $p$ features simultaneously for Granger causality, the familywise error rate (FWER) inflates. With $p = 100$ features at $\alpha = 0.05$, approximately 5 features will be false positives even under the null.

### 7.1 Bonferroni Correction

The most conservative approach: adjust the threshold to $\alpha / p$.

$$\text{Reject } H_0^{(j)} \text{ if } p_j < \frac{\alpha}{p}$$

Controls FWER at level $\alpha$ but loses power rapidly as $p$ grows.

### 7.2 Benjamini-Hochberg FDR

Controls the **false discovery rate** (FDR) — the expected proportion of false positives among all rejections. For time series feature selection where some false positives are tolerable, BH is preferred over Bonferroni.

**Procedure:**
1. Sort p-values: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(p)}$.
2. Find the largest $k$ such that $p_{(k)} \leq \frac{k}{p} \alpha$.
3. Reject all $H_0^{(j)}$ for $j \leq k$.

```python
from statsmodels.stats.multitest import multipletests


def apply_multiple_testing_correction(
    pvalues: pd.Series,
    method: str = 'fdr_bh',
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Apply multiple testing correction to a Series of p-values.

    Parameters
    ----------
    pvalues : pd.Series
        P-values indexed by feature name.
    method : str
        'bonferroni', 'fdr_bh' (Benjamini-Hochberg), or 'fdr_by'.
    alpha : float
        Target FWER or FDR level.

    Returns
    -------
    pd.DataFrame with original and corrected significance.
    """
    reject, pvals_corrected, _, _ = multipletests(
        pvalues.values, alpha=alpha, method=method
    )

    return pd.DataFrame({
        'feature': pvalues.index,
        'raw_pvalue': pvalues.values,
        'corrected_pvalue': pvals_corrected,
        'reject_h0': reject,
    }).sort_values('corrected_pvalue').reset_index(drop=True)
```

### 7.3 Adaptive Thresholding

For exploratory feature selection, rather than a hard reject/fail-to-reject decision, rank features by corrected p-value and select the top-$k$ or those passing $p_{\text{corrected}} < 0.1$.

---

## 8. Complete Feature Selection Pipeline

```python
def granger_feature_selection_pipeline(
    target: pd.Series,
    features: pd.DataFrame,
    max_lag: int = 5,
    alpha: float = 0.05,
    correction: str = 'fdr_bh',
    n_conditioning: int = 0,
) -> dict:
    """
    Full Granger causality feature selection pipeline.

    Steps:
    1. Stationarity checks and differencing.
    2. Bivariate Granger tests for all features.
    3. Multiple testing correction.
    4. Optional: conditional Granger test for top features.

    Returns ranked feature list with corrected p-values.
    """
    # Step 1: Stationarity
    print("Checking stationarity...")
    target_check = stationarity_check(target, 'target')
    if not target_check['is_stationary']:
        print(f"  Target not stationary — differencing.")
        target = target.diff().dropna()

    stationary_features = {}
    for col in features.columns:
        check = stationarity_check(features[col], col)
        if check['is_stationary']:
            stationary_features[col] = features[col]
        else:
            stationary_features[col] = features[col].diff().dropna()

    features_clean = pd.DataFrame(stationary_features).dropna()
    target_clean = target.loc[features_clean.index].dropna()
    features_clean = features_clean.loc[target_clean.index]

    # Step 2: Bivariate Granger tests
    print(f"Running Granger tests for {len(features.columns)} features...")
    ranking = granger_feature_ranking(target_clean, features_clean, max_lag=max_lag)

    # Step 3: Multiple testing correction
    pval_series = ranking.set_index('feature')['min_pvalue']
    corrected = apply_multiple_testing_correction(pval_series, method=correction, alpha=alpha)

    selected_features = corrected[corrected['reject_h0']]['feature'].tolist()
    print(f"Selected {len(selected_features)} features after {correction} correction.")

    return {
        'ranking': corrected,
        'selected': selected_features,
        'n_tested': len(features.columns),
        'n_selected': len(selected_features),
    }
```

---

## Common Pitfalls

- **Testing in price levels:** Returns or differences required for valid inference.
- **Ignoring instantaneous correlation:** Bivariate tests assume no contemporaneous feedback.
- **Single lag order for all pairs:** Different feature pairs may have optimal predictive lags; always test a range.
- **Mistaking statistical significance for practical significance:** A p-value < 0.01 with tiny F-statistic may not improve forecast accuracy meaningfully.
- **Treating Granger causality as true causality:** It is predictive precedence, not mechanism.

---

## Connections

- **Builds on:** Module 02 mutual information; stationarity testing (ADF, KPSS); VAR models
- **Leads to:** Module 07 walk-forward validation (testing Granger features over time); Module 09 causal feature selection
- **Related to:** Transfer entropy (information-theoretic Granger); directed acyclic graphs; cointegration

---

## Further Reading

- Granger, C.W.J. (1969). "Investigating Causal Relations by Econometric Models and Cross-spectral Methods." *Econometrica*, 37(3), 424–438. — Original paper.
- Geweke, J. (1982). "Measurement of Linear Dependence and Feedback Between Multiple Time Series." *JASA*, 77(378), 304–313. — Spectral extension.
- Tank, A. et al. (2018). "Neural Granger Causality." *arXiv:1802.05842*. — Neural network approach.
- Barnett, L. & Seth, A.K. (2014). "The MVGC multivariate Granger causality toolbox." *Journal of Neuroscience Methods*, 223, 50–68. — Comprehensive toolbox reference.
- de Prado, M.L. (2018). *Advances in Financial Machine Learning*. Wiley. — Chapter on feature importance in financial ML.
