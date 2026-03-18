---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Dynamic Factor Models

## Theory, Estimation, and Factor Extraction

**Mixed-Frequency Models: MIDAS Regression and Nowcasting**
Module 04 — Guide 01

<!-- Speaker notes: This guide introduces Dynamic Factor Models (DFMs) as the next step beyond single-indicator MIDAS. The key motivation: with many indicators (N>>1), we need dimensionality reduction before nowcasting. DFMs extract a small number of common factors that summarize comovement across the panel. The main skill is: extract factors via PCA, determine how many to use (Bai-Ng or scree plot), and use them in a regression. The Kalman filter for the full DFM is covered conceptually but not implemented in detail — we focus on the two-step PCA approach. -->

---

## Why Dynamic Factor Models?

**The problem with many indicators:**

| Indicators | U-MIDAS params | Feasible? |
|-----------|---------------|-----------|
| 1 (K=12) | 13 | Yes (K/T=0.13) |
| 5 (K=12 each) | 61 | Maybe |
| 20 (K=12 each) | 241 | No (T~100) |

**Solution:** Extract $q=2$ factors from all 20 indicators, use 2 predictors.

$$\mathbf{x}_t = \mathbf{\Lambda}\mathbf{f}_t + \mathbf{e}_t, \quad q \ll N$$

<!-- Speaker notes: The practical problem is clear: with 20 monthly indicators at K=12 lags each, U-MIDAS would require 241 parameters with only T=100 quarterly observations. The K/T ratio is 2.4 — way above the 0.05 threshold for U-MIDAS viability. Factor models solve this by projecting the N-dimensional data onto a q-dimensional factor space. The key insight is that most economic indicators share a common business cycle component — the first principal component of the indicator panel captures this. -->

---

## The Factor Model

**Observation equation:**
$$\mathbf{x}_t = \mathbf{\Lambda}\mathbf{f}_t + \mathbf{e}_t$$

**State equation (dynamics):**
$$\mathbf{f}_t = \mathbf{A}\mathbf{f}_{t-1} + \mathbf{u}_t$$

Key assumptions:
- $E[\mathbf{e}_{it}\mathbf{e}_{jt}] \approx 0$ for $i \neq j$ (idiosyncratic errors uncorrelated)
- $N$ large enough so factor is identified
- $q \ll N$ (few common factors)

<!-- Speaker notes: The factor model decomposes each indicator into a common component (Lambda_i * f_t) and an idiosyncratic component (e_it). The common component captures comovement across indicators — when the economy booms, most indicators rise together. The idiosyncratic component captures indicator-specific variation. The key identifying assumption is that idiosyncratic errors are approximately uncorrelated across indicators — otherwise, you can't separate common from idiosyncratic. This is realistic for diversified indicator panels (IP, employment, retail sales, confidence), where no two indicators share the same idiosyncratic shock. -->

---

## Estimation: Principal Components

```python
from sklearn.decomposition import PCA
import numpy as np

def extract_factors_pca(X, n_factors):
    """
    X: (T, N) panel of standardized monthly indicators
    Returns F (T, q), Lambda (N, q), variance explained.
    """
    pca = PCA(n_components=n_factors)
    F = pca.fit_transform(X)        # (T, q) factor series
    Lambda = pca.components_.T      # (N, q) loadings
    var_exp = pca.explained_variance_ratio_
    return F, Lambda, var_exp

# IMPORTANT: standardize X before PCA!
X_std = (X - X.mean(axis=0)) / X.std(axis=0)
F, Lambda, var_exp = extract_factors_pca(X_std, n_factors=2)
```

PCA is consistent for the factor space as $N, T \to \infty$ (Stock & Watson 2002).

<!-- Speaker notes: The PCA estimator is surprisingly powerful for factor extraction. Stock and Watson (2002) showed that PCA gives a consistent estimate of the factor space even without knowing the true number of factors, as long as N and T both grow to infinity. The key requirement is standardization: divide each indicator by its standard deviation so that high-variance indicators don't dominate the first factor just because they're more volatile. After standardization, the first PC captures the direction of maximum covariance across indicators, which is the common business cycle factor. -->

---

## Scree Plot and Variance Explained

```
Variance explained by each principal component:

PC1 | ████████████████ 38%
PC2 | ████████ 18%
PC3 | ████ 10%
PC4 | ██ 7%
PC5 | ██ 6%
...
     ─────────────────
     Cumulative PC1-3: 66%

Elbow at PC3 → select q=2 or q=3
```

**Rule of thumb:** Select q where cumulative variance explained exceeds 50-60%.

<!-- Speaker notes: The scree plot is the most common visual tool for determining the number of factors. The "elbow" is where the curve bends from steep to flat — each additional factor explains much less variance than the previous ones. In practice for a panel of 10-20 US economic indicators, the scree plot typically shows a sharp elbow at q=2 or q=3, meaning the first 2-3 factors explain the bulk of comovement while the remaining factors are largely idiosyncratic noise. The 50-60% variance explained threshold is a practical heuristic — there's no magic number, and the Bai-Ng criterion provides a more principled alternative. -->

---

## Bai-Ng Information Criteria

$$IC_{p2}(q) = \log \hat{\sigma}^2(q) + q \cdot \frac{N+T}{NT} \cdot \log \min(N,T)$$

```python
def select_n_factors_bai_ng(X, max_factors=8):
    T, N = X.shape
    g_NT = (N + T) / (N * T) * np.log(min(N, T))
    ic_vals = []
    for q in range(1, max_factors + 1):
        F, Lambda, _ = extract_factors_pca(X, q)
        X_hat = F @ Lambda.T
        sigma2 = np.mean((X - X_hat)**2)
        ic_vals.append(np.log(sigma2) + q * g_NT)
    return np.argmin(ic_vals) + 1
```

Consistent for $q$ as $N, T \to \infty$.

<!-- Speaker notes: The Bai-Ng (2002) information criterion is the standard formal method for selecting the number of factors. It's analogous to BIC for regression: the first term measures fit (log of unexplained variance) and the second term penalizes for the number of factors. The penalty g_NT = (N+T)/(NT)*log(min(N,T)) is designed to be consistent — it shrinks to zero slowly enough that the true number of factors is always selected asymptotically. In finite samples with N=15 and T=100, the criterion typically works well. For very small N (<10), it can underselect. -->

---

## Factor Loadings: Economic Interpretation

The loadings $\hat{\lambda}_i$ tell us how each indicator relates to the common factor:

```
Factor 1 loadings (business cycle factor):
  IP growth:        +0.72  (procyclical)
  Employment:       +0.68  (procyclical)
  Retail sales:     +0.61  (procyclical)
  Consumer conf.:   +0.58  (procyclical)
  Treasury spread:  -0.41  (countercyclical — inverts in recessions)
  VIX:             -0.55  (countercyclical — rises in recessions)
```

Positive loadings = procyclical. Negative = countercyclical.

**Sign is identified by normalization** (we choose the sign that makes IP loading positive).

<!-- Speaker notes: The economic interpretation of factor loadings is one of the most informative outputs of the DFM. After extracting the first factor, we look at the loadings to understand what it represents. When all major real indicators (IP, employment, retail sales) have large positive loadings, we call it the "business cycle factor." This factor is essentially a weighted average of the indicators, with weights proportional to the loadings. The sign identification requires a normalization — we typically set the IP loading to be positive so that the factor rises during expansions. Treasury spread and VIX having negative loadings confirms the factor is countercyclical in the way expected. -->

---

## Factor-Augmented Regression for GDP

After extracting factors, use them in a GDP regression:

$$y_t = \alpha + \sum_{j=1}^{q} \beta_j f_{jt} + \varepsilon_t$$

This is **Factor-Augmented MIDAS (FA-MIDAS)** when combined with the MIDAS weight structure.

**Advantages:**
- Uses all $N$ indicators via 2-3 factor proxies
- Noise reduction: idiosyncratic errors average out
- Parsimonious: $q+1$ parameters regardless of $N$

<!-- Speaker notes: Factor-augmented regression (FAVAR) is now standard in empirical macro. The idea is simple: instead of using each indicator separately (which requires many parameters and overfits), we use the estimated factors as predictors. The factors capture the common information across all indicators, so we get the informational benefit of using all N indicators while only adding q regression coefficients. The FA-MIDAS extension is our contribution — we apply the MIDAS weight structure to the factor time series, allowing different lags of the factor to receive different weights based on the Beta polynomial. -->

---

## DFM vs. MIDAS: When to Use Each

<div class="columns">

<div>

**Use MIDAS when:**
- Single dominant indicator (IP for GDP)
- Want interpretable weight function
- T > 60, K/T < 0.1
- Need to show which lag matters

</div>

<div>

**Use DFM when:**
- Many indicators (N > 5)
- Want to combine information
- Indicators are noisy
- "Kitchen sink" nowcasting

</div>

</div>

**Best practice:** Start with MIDAS. If RMSE doesn't improve beyond 2-3 indicators, add factor extraction.

<!-- Speaker notes: The decision between MIDAS and DFM depends on the number of indicators and the research question. For a single dominant predictor like IP for GDP, MIDAS is preferred because it's interpretable and the weight function can be visualized and tested. For large datasets (central bank nowcasting with N=50+ indicators), DFMs are necessary. The practical recommendation is to start with MIDAS on the 2-3 most important indicators, check the RMSE, and only move to a DFM if RMSE keeps improving as you add indicators. If RMSE flattens after 2-3 indicators, the additional information is not helping. -->

---

## Summary: DFM Key Results

| Component | Estimate | Interpretation |
|-----------|---------|----------------|
| Factor $\hat{f}_t$ | PCA first component | Business cycle index |
| Loading $\hat{\lambda}_{IP}$ | 0.72 | IP tracks cycle closely |
| Loading $\hat{\lambda}_{SP500}$ | 0.48 | Markets lead cycle slightly |
| Variance explained (q=1) | 45% | One factor dominates |
| Variance explained (q=2) | 63% | Second factor adds 18% |

**Next:** Guide 02 — Mixed-frequency DFM: handling the ragged edge with the Kalman filter.

<!-- Speaker notes: The summary table gives typical values from a 3-4 indicator DFM for US quarterly data. The first factor explaining 45% of variance in a panel of 3-4 indicators is substantial — in a well-diversified panel of 15+ indicators, you'd typically see the first factor explaining 25-35%. The loading of 0.72 for IP means IP and the business cycle factor have a correlation of 0.72, which is strong but not perfect — some IP variation is idiosyncratic. The second factor (18% additional variance) often captures a sector-specific or financial dimension that the first factor misses. -->
