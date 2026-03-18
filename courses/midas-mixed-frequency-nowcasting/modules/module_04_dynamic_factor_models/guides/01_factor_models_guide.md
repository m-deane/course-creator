# Dynamic Factor Models: Theory and Intuition

## In Brief

A Dynamic Factor Model (DFM) extracts a small number of common factors $f_t$ from a large panel of economic indicators. Each indicator $x_{it}$ loads on the common factors, and the factors summarize the comovement across the panel. For nowcasting, DFMs extend naturally to mixed-frequency data: different indicators are observed at different frequencies, and the common factor evolves at the highest available frequency.

## Key Insight

The DFM solves the "curse of dimensionality" problem in nowcasting: when you have dozens of monthly indicators, unrestricted regression (U-MIDAS) with all of them has too many parameters. The DFM reduces the $N \times T$ data matrix to a $q \times T$ factor matrix ($q \ll N$), then uses the factors in a low-dimensional forecasting model.

---

## The Static Factor Model

The simplest factor model decomposes each indicator into a common and idiosyncratic component:

$$x_{it} = \lambda_i f_t + e_{it}$$

where:
- $x_{it}$: observation of indicator $i$ at time $t$ (standardized: mean 0, variance 1)
- $f_t$: common factor (scalar for one-factor model, vector for multi-factor)
- $\lambda_i$: loading of indicator $i$ on the factor
- $e_{it}$: idiosyncratic error, $\text{Cov}(e_{it}, e_{jt}) \approx 0$ for $i \neq j$

With $N$ indicators and $q$ factors, the model is:

$$\mathbf{x}_t = \mathbf{\Lambda} \mathbf{f}_t + \mathbf{e}_t$$

where $\mathbf{x}_t \in \mathbb{R}^N$, $\mathbf{\Lambda} \in \mathbb{R}^{N \times q}$, $\mathbf{f}_t \in \mathbb{R}^q$.

### Identification

The model is only identified up to rotation. We normalize $\mathbf{f}_t' \mathbf{f}_t / T = \mathbf{I}_q$ and $\mathbf{\Lambda}' \mathbf{\Lambda}$ is diagonal (PC normalization).

### Estimation: Principal Components

The factors are estimated as the first $q$ principal components of the data matrix:

1. Standardize each $x_{it}$ (subtract mean, divide by standard deviation)
2. Compute the $N \times N$ covariance matrix (or $T \times T$ if $N > T$)
3. Extract the top $q$ eigenvectors
4. Estimated factors: $\hat{\mathbf{F}} = \mathbf{X} \hat{\mathbf{V}}_q / N$ where $\hat{\mathbf{V}}_q$ contains the top-$q$ eigenvectors

```python
import numpy as np
from sklearn.decomposition import PCA

def extract_factors_pca(X, n_factors):
    """
    Extract n_factors common factors from panel X via PCA.

    Parameters
    ----------
    X : np.ndarray, shape (T, N) — T time periods, N indicators
        Each column should be standardized (mean 0, std 1).
    n_factors : int

    Returns
    -------
    F : np.ndarray, shape (T, n_factors) — extracted factor series
    Lambda : np.ndarray, shape (N, n_factors) — factor loadings
    var_explained : np.ndarray, shape (n_factors,) — variance explained
    """
    pca = PCA(n_components=n_factors)
    F = pca.fit_transform(X)   # (T, n_factors)
    Lambda = pca.components_.T  # (N, n_factors)
    var_exp = pca.explained_variance_ratio_
    return F, Lambda, var_exp
```

---

## The Dynamic Factor Model

The DFM adds dynamics to the static model. The factors follow a VAR process:

$$\mathbf{f}_t = \mathbf{A}_1 \mathbf{f}_{t-1} + \cdots + \mathbf{A}_p \mathbf{f}_{t-p} + \mathbf{u}_t, \quad \mathbf{u}_t \sim \mathcal{N}(0, \mathbf{Q})$$

$$\mathbf{x}_t = \mathbf{\Lambda} \mathbf{f}_t + \mathbf{e}_t, \quad \mathbf{e}_t \sim \mathcal{N}(0, \mathbf{R})$$

This state-space representation is estimated via the Kalman filter. The state vector $\mathbf{f}_t$ is unobserved; the Kalman filter provides the optimal linear estimate $\hat{\mathbf{f}}_{t|t} = E[\mathbf{f}_t | \mathbf{x}_1, \ldots, \mathbf{x}_t]$.

### Two-Step Estimation (Giannone, Reichlin, Small 2008)

1. **Step 1:** Estimate loadings $\hat{\Lambda}$ by PCA on the balanced data (all $N$ indicators observed)
2. **Step 2:** Run the Kalman filter using $\hat{\Lambda}$ as fixed parameters, updating the factor estimate as new data arrives

This is computationally tractable even for $N = 100$ indicators and $T = 200$ time periods.

---

## Number of Factors: Selection

How many factors to extract? Three methods:

### 1. Scree Plot
Plot the eigenvalues in decreasing order. The "elbow" suggests the number of factors. Factors beyond the elbow explain little additional variance.

### 2. Bai-Ng Information Criteria (2002)
$$IC_p(q) = \log \hat{\sigma}^2(q) + q \cdot g(N, T)$$

where $g(N, T) = \frac{N + T}{NT} \log \min(N, T)$ for the IC_p2 criterion.

### 3. Variance Explained
Common rule of thumb: select $q$ such that the first $q$ factors explain at least 50–70% of total variance.

For US quarterly macro data (N=15–30 indicators): typically $q = 2$ or $q = 3$ factors suffice.

```python
def select_n_factors_bai_ng(X, max_factors=10):
    """
    Bai-Ng IC_p2 criterion for number of factors.

    Returns the number of factors minimizing IC_p2.
    """
    T, N = X.shape
    g_NT = (N + T) / (N * T) * np.log(min(N, T))

    ic_values = []
    for q in range(1, max_factors + 1):
        pca = PCA(n_components=q)
        F = pca.fit_transform(X)
        Lambda = pca.components_.T
        X_hat = F @ Lambda.T
        resid = X - X_hat
        sigma2 = np.mean(resid**2)
        ic = np.log(sigma2) + q * g_NT
        ic_values.append(ic)

    return np.argmin(ic_values) + 1, np.array(ic_values)
```

---

## Factor-Augmented Regression

Once factors are extracted, use them in a standard regression for GDP:

$$y_t = \alpha + \beta_1 f_{1t} + \beta_2 f_{2t} + \cdots + \beta_q f_{qt} + \varepsilon_t$$

This is the Factor-Augmented MIDAS (FA-MIDAS) model when the factors are constructed from mixed-frequency data.

**Advantages over single-indicator MIDAS:**
- Uses information from all $N$ indicators simultaneously
- Factor extraction provides implicit noise reduction
- Naturally handles "news" — surprises in any indicator update the factor estimate

---

## Mixed-Frequency DFM

For nowcasting, indicators are observed at different frequencies. The key challenge: how to define $\mathbf{x}_t$ when some components are monthly and others are quarterly?

**Two approaches:**

1. **Aggregation before factor extraction:** Average monthly series to quarterly, then extract factors from the quarterly panel. Simple but discards within-quarter variation.

2. **Factor at monthly frequency:** Define the state vector at monthly frequency. Quarterly variables are observed every 3 months with a known aggregation relationship. The Kalman filter handles the missing observations naturally.

For the course's "small" DFM application, we use approach 1 (quarterly aggregation) and then build on MIDAS to handle the mixed-frequency structure.

---

## Small DFM for Course Application

For our course dataset (3 indicators: IP growth, payrolls growth, S&P 500 returns), a one-factor model suffices:

$$\begin{pmatrix} IP_t \\ Payrolls_t \\ SP500_t \end{pmatrix} = \begin{pmatrix} \lambda_{IP} \\ \lambda_{Payrolls} \\ \lambda_{SP500} \end{pmatrix} f_t + \mathbf{e}_t$$

The single factor $f_t$ captures the common business cycle component. By construction, this factor correlates strongly with GDP growth and can be used as the high-frequency predictor in a MIDAS-style nowcasting model.

---

## Common Pitfalls

**Pitfall 1: Not standardizing indicators.** PCA is scale-dependent. If IP growth is in percent and S&P 500 returns are in decimals, the S&P 500 will dominate the first factor purely due to scale. Always standardize to unit variance.

**Pitfall 2: Ignoring the timing of factor "news."** The factor estimate updates each month as new releases arrive. The nowcast update equals $\hat{\beta} \cdot \partial\hat{f}_t / \partial x_{new}$ — the sensitivity of the factor to the new data point.

**Pitfall 3: Using too many factors.** For T=100 and N=5 indicators, extracting q=4 factors is essentially U-MIDAS in disguise — overfitting. Use Bai-Ng or the scree plot to determine q.

---

## Connections

- **Builds on:** Module 01 (MIDAS), Module 03 (nowcasting)
- **Leads to:** Guide 02 (DFM mixed frequency), Guide 03 (Factor-augmented MIDAS)
- **Related to:** Stock-Watson (2002), Giannone-Reichlin-Small (2008), Bai-Ng (2002)

---

## Practice Problems

1. The first principal component of a (T=100, N=20) data matrix explains 35% of total variance. How much variance do the first 3 PCs explain if each subsequent PC explains half as much as the previous?

2. Write out the Bai-Ng IC_p2 criterion for $N=15$, $T=80$, $q=1$ vs. $q=2$. By how much must $\hat{\sigma}^2(2) < \hat{\sigma}^2(1)$ for the criterion to prefer $q=2$?

3. The loading of IP growth on the first factor is $\hat{\lambda}_{IP} = 0.72$, and the factor has unit variance. If a new IP observation is 0.5 standard deviations above the current factor estimate, by how much does the factor estimate update? (Use the Kalman filter weight formula if you know it, or approximate it as the OLS coefficient.)
