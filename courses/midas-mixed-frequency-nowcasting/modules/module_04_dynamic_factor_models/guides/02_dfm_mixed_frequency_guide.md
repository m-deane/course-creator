# Mixed-Frequency DFMs

> **Reading time:** ~13 min | **Module:** 04 — Dynamic Factor Models | **Prerequisites:** Module 3


## In Brief

<div class="flow">
<div class="flow-step mint">1. Collect Data</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">2. Identify Frequencies</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">3. Align Time Indices</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Build MIDAS Regressors</div>
</div>


<div class="callout-key">

**Key Concept Summary:** Mixed-frequency DFMs (MF-DFMs) handle panels where some series are monthly and others are quarterly. The key innovation is treating quarterly series as monthly series observed only every third mont...

</div>

Mixed-frequency DFMs (MF-DFMs) handle panels where some series are monthly and others are quarterly. The key innovation is treating quarterly series as monthly series observed only every third month. The state-space representation with the Kalman filter naturally handles these missing observations.

## Key Insight

<div class="callout-insight">

**Insight:** Factor models compress many noisy indicators into a few latent factors, which acts as a form of regularization. This is why DFM-based nowcasts often outperform single-indicator MIDAS models.

</div>


In the state-space framework, "missing" quarterly data is simply a missing observation in the monthly factor system. The Kalman filter propagates uncertainty through the missing periods and updates when a quarterly observation arrives. This is the theoretically clean solution to the mixed-frequency nowcasting problem.

---

## The Quarterly-Monthly Aggregation Constraint

<div class="callout-warning">

**Warning:** Be cautious about extrapolating MIDAS performance from stable periods to crisis periods. The relationship between high-frequency indicators and the low-frequency target can shift dramatically during regime changes.

</div>


Let $y_t^Q$ denote quarterly GDP growth and $y_{m}^M$ the corresponding monthly factor index. The aggregation constraint is:

$$y_t^Q = \frac{1}{3}(y_{3t}^M + y_{3t-1}^M + y_{3t-2}^M) + u_t$$

(for flow variables like GDP growth; stock variables use different aggregation).

In the state-space formulation, define the state vector for the quarterly series as:

$$\begin{pmatrix} y_{m}^M \\ y_{m-1}^M \\ y_{m-2}^M \end{pmatrix}$$

The quarterly GDP observation appears every 3 months with the constraint:
$$y_t^Q = \begin{pmatrix} 1/3 & 1/3 & 1/3 \end{pmatrix} \begin{pmatrix} y_{m}^M \\ y_{m-1}^M \\ y_{m-2}^M \end{pmatrix}$$

---

## The State-Space Representation

**State equation (monthly):**
$$\mathbf{s}_m = \mathbf{T} \mathbf{s}_{m-1} + \mathbf{R} \boldsymbol{\eta}_m$$

where the state $\mathbf{s}_m$ includes the factors $\mathbf{f}_m$ and any lagged values.

**Observation equation (mixed frequency):**
$$\mathbf{y}_m = \mathbf{Z}_m \mathbf{s}_m + \boldsymbol{\varepsilon}_m$$

where $\mathbf{Z}_m$ is the measurement matrix that changes each month:
- In months 1 and 2 of a quarter: only monthly indicators load onto the state
- In month 3 of a quarter: both monthly indicators AND quarterly GDP load onto the state

The Kalman filter handles this automatically by setting $\mathbf{Z}_m$ appropriately.

---

## Practical Implementation: Two-Step Approach

For the course, we use a simplified two-step approach that avoids full Kalman filter implementation:

**Step 1: Extract factors from monthly panel**

<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def extract_monthly_factors(monthly_panel, n_factors):
    """
    Extract common factors from a panel of monthly indicators.
    Monthly panel: DataFrame (T_months, N_monthly_indicators), standardized.
    """
    from sklearn.decomposition import PCA
    import numpy as np

    pca = PCA(n_components=n_factors)
    F_monthly = pca.fit_transform(monthly_panel.values)
    loadings = pca.components_.T
    var_exp = pca.explained_variance_ratio_
    return F_monthly, loadings, var_exp
```

</div>
</div>

**Step 2: Aggregate factors to quarterly frequency**

<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def monthly_to_quarterly_factors(F_monthly, quarterly_dates, monthly_dates):
    """
    Aggregate monthly factors to quarterly by end-of-quarter value or average.
    Returns quarterly factor matrix aligned with quarterly GDP.
    """
    import pandas as pd
    import numpy as np

    # Convert to DataFrame with monthly index
    F_df = pd.DataFrame(F_monthly, index=monthly_dates,
                        columns=[f'F{i+1}' for i in range(F_monthly.shape[1])])
    F_df.index = pd.to_datetime(F_df.index)

    # Resample to quarterly (last month of quarter = end-of-quarter value)
    F_quarterly = F_df.resample('QE').last()

    # Align with quarterly GDP dates
    F_aligned = F_quarterly.reindex(quarterly_dates)
    return F_aligned.values
```

</div>
</div>

**Step 3: MIDAS regression using factor as predictor**

<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def factor_augmented_midas(Y, F_monthly, K, n_factors=1):
    """
    Factor-augmented MIDAS: extract factor from monthly panel, use as MIDAS predictor.

    Instead of a single indicator (IP), uses the first common factor.
    """
    # Build MIDAS matrix from the first factor series
    # (treat factor time series exactly like any other monthly indicator)
    F1_monthly = F_monthly[:, 0]  # First factor
    Y_aligned, X_factor = build_midas_matrix_from_series(Y, F1_monthly, K)
    return estimate_midas(Y_aligned, X_factor)
```

</div>
</div>

---

## Factor Nowcasting: Information Flow

In the MF-DFM, the nowcast updates each time a new monthly indicator is released:

```
Month 1 releases (Feb 15):
  IP released → update factor estimate → update GDP nowcast

Month 2 releases (Mar 15):
  Employment released → update factor → update nowcast
  Retail sales released → update factor → update nowcast

Month 3 releases (Apr 15):
  Full quarter IP, employment, retail sales → final factor update → final nowcast
```

Each release contributes "news" proportional to:
1. How much the indicator's loading $\hat{\lambda}_i$ is
2. How surprising the release is (actual minus the prior)

---

## Comparison: MIDAS vs. DFM Nowcast

| Model | Indicators | OOS RMSE (typical) | Interpretation |
|-------|-----------|-------------------|----------------|
| AR(1) | None | 0.85 | Baseline |
| MIDAS (IP only) | 1 | 0.70 | IP carries most info |
| MIDAS (IP + payrolls) | 2 | 0.66 | 6% improvement |
| FA-MIDAS (q=1 factor) | 3+ | 0.64 | Factor noise-reduces |
| Full MF-DFM | All | 0.61 | Marginal gain |

**Practical finding:** Most of the forecast improvement comes from the first 2-3 indicators. Adding more through a DFM gives diminishing returns.

---

## When to Use MF-DFM vs. Multi-Indicator MIDAS

**Use MF-DFM when:**
- $N > 5$ indicators (DFM reduces dimensionality)
- Indicators are noisy and correlated
- Indicators published at different times within the month (staggered releases)
- You want a principled treatment of the ragged edge across many series

**Use multi-indicator MIDAS when:**
- $N \leq 5$ indicators
- Interpretability is important
- You want to assign economic meaning to each predictor
- Sample size is small ($T < 80$)

---

## Common Pitfalls

**Pitfall 1: Not re-estimating factors in the expanding window.** Factor loadings should be re-estimated as the sample grows, not kept fixed from the initial estimation. In practice, loadings are fairly stable, but re-estimation is the correct procedure.

**Pitfall 2: Inconsistent sign normalization.** The PCA sign is arbitrary. Normalize consistently across all expanding-window estimations (e.g., IP loading is always positive). A sign flip mid-sample would invert the factor and destroy the regression.

**Pitfall 3: Mixing real-time and revised data.** IP and employment are revised substantially. The factor extracted from revised data is smoother and more informative than what was available in real time.

---

## Connections

<div class="callout-danger">

**Danger:** Never use future information when constructing the high-frequency regressor matrix. In a real-time nowcasting context, you only have data up to the current date -- using the full quarter of monthly data when nowcasting mid-quarter is a look-ahead bias that invalidates your results.

</div>


- **Builds on:** Guide 01 (factor models theory, PCA extraction)
- **Leads to:** Guide 03 (Factor-augmented MIDAS implementation)
- **Related to:** Bańbura-Rünstler (2011), Doz-Giannone-Reichlin (2012)

---

## Practice Problems

1. For a monthly panel with N=5 indicators (IP, employment, retail sales, consumer confidence, PMI) and T_months=300, write out the dimensions of $\mathbf{F}$, $\mathbf{\Lambda}$, and $\mathbf{e}$ for a q=2 factor model.

2. If the IP loading on the first factor is $\hat{\lambda}_{IP}=0.72$ and the factor standard deviation is 1, what is the variance of IP growth explained by the first factor? What is the idiosyncratic variance if total IP variance is 1?

3. The MF-DFM nowcast updates when employment data is released. If the employment loading is $\hat{\lambda}_{emp}=0.65$ and the employment "surprise" (actual minus Kalman prediction) is +0.8%, estimate the factor update (approximately proportional to $\hat{\lambda}_{emp} \times \text{surprise}$).


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

<a class="link-card" href="./03_factor_augmented_midas_guide.md">
  <div class="link-card-title">03 Factor Augmented Midas</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_factor_augmented_midas_slides.md">
  <div class="link-card-title">03 Factor Augmented Midas — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

