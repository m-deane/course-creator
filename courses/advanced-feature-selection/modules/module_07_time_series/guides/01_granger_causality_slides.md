---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->
<!-- Speaker notes: Welcome to Module 7. This deck covers Granger causality as a principled framework for feature selection in time series. Emphasize early on that "Granger causality" is statistical predictive precedence, not mechanistic causality — this distinction prevents a common misinterpretation. -->

# Granger Causality for Feature Selection
## Temporal Dependencies and Directed Information Flow

### Module 07 — Feature Selection for Time Series

*Which features actually predict the future?*

---

<!-- Speaker notes: Open with motivation. Ask the audience: "If you have 50 candidate features for a financial return forecast, how do you decide which ones carry information about the future?" Standard correlation is symmetric and ignores temporal order. Granger causality is explicitly about the future. -->

## The Time Series Feature Selection Problem

Standard feature selection ignores time structure:

- Correlation is **symmetric** — it cannot tell you which variable leads
- Mutual information ignores **temporal ordering** of observations
- Cross-validation assumes **iid samples** — violated by serial dependence

**The question we need to answer:**

> Does knowing the past of feature $X$ help predict future values of $Y$, beyond what $Y$'s own past already tells us?

Granger causality answers this directly.

---

<!-- Speaker notes: This slide contains the core definition. Stress the word "beyond" — this is what makes Granger causality useful. The test controls for autocorrelation in Y before asking whether X adds information. Walk through the two equations carefully. -->

## What is Granger Causality?

**Definition (Granger, 1969):** $X$ Granger-causes $Y$ if past values of $X$ improve forecasts of $Y$ *beyond what $Y$'s own history provides.*

<div class="columns">

**Restricted model** (Y's own history):
$$y_t = \sum_{l=1}^{L} \alpha_l y_{t-l} + \varepsilon_t^R$$

**Unrestricted model** (add X lags):
$$y_t = \sum_{l=1}^{L} \alpha_l y_{t-l} + \sum_{l=1}^{L} \beta_l x_{t-l} + \varepsilon_t^U$$

</div>

**Test:** $H_0: \beta_1 = \beta_2 = \cdots = \beta_L = 0$

If rejected: $X$ Granger-causes $Y$.

---

<!-- Speaker notes: Walk through the F-test formula. Emphasize that this is just a standard F-test for adding a block of regressors. The key choice is lag order L — spend a moment on AIC vs BIC tradeoff: AIC for prediction, BIC for parsimony. -->

## The F-Test for Granger Causality

$$F = \frac{(RSS_R - RSS_U) / L}{RSS_U / (T - 2L - 1)} \sim F(L,\; T - 2L - 1)$$

| Symbol | Meaning |
|--------|---------|
| $RSS_R$ | Residual sum of squares, restricted model |
| $RSS_U$ | Residual sum of squares, unrestricted model |
| $L$ | Lag order (number of lags tested) |
| $T$ | Sample size |

**Lag order selection:**

| Criterion | Preference |
|-----------|-----------|
| AIC | Prediction accuracy (larger models) |
| BIC | Parsimony, consistent (smaller models) |
| HQIC | Between AIC and BIC |

---

<!-- Speaker notes: This is the practical slide. Show how this maps to code. The statsmodels grangercausalitytests function does the work — the key is that you test multiple lags and take the minimum p-value, then correct for multiple comparisons. Note: always check stationarity first. -->

## From Test to Feature Ranking

**Pipeline:**

```
For each feature X_j:
    1. Check stationarity (ADF test) — difference if needed
    2. Run Granger test at lags 1, ..., L_max
    3. Record min p-value across lags

Rank features by ascending p-value
Apply multiple testing correction (BH-FDR)
Select features with corrected p < alpha
```

**Key implementation point:** `statsmodels.tsa.stattools.grangercausalitytests` handles the F-test; you loop over features and collect results.

---

<!-- Speaker notes: Move to multivariate — bivariate tests miss interactions. In a VAR, variable j causes variable i if the (i,j) block across all lag matrices is jointly non-zero. The Wald test checks this block restriction. Optimal lag by BIC is standard practice. -->

## VAR Formulation: Multivariate Granger

For $K$ variables simultaneously, the VAR($L$) model:

$$\mathbf{y}_t = \boldsymbol{\nu} + \sum_{l=1}^{L} \mathbf{A}_l \mathbf{y}_{t-l} + \boldsymbol{\varepsilon}_t, \quad \boldsymbol{\varepsilon}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$$

**Granger non-causality from $X_j$ to $X_i$:**

$$[\mathbf{A}_l]_{ij} = 0 \quad \forall\, l \in \{1, \ldots, L\}$$

Tested jointly with a **Wald chi-squared test:**

$$W = \vec{R}\hat{\boldsymbol{\beta}} \left[\vec{R} \text{Cov}(\hat{\boldsymbol{\beta}}) \vec{R}^\top \right]^{-1} \vec{R}\hat{\boldsymbol{\beta}} \sim \chi^2(L)$$

**Advantage over bivariate:** accounts for correlations among features in a single joint model.

---

<!-- Speaker notes: Conditional Granger causality is the key tool for avoiding spurious selection from confounders. The classic example: a leading economic indicator may appear to Granger-cause GDP, but only because it correlates with a true cause. Conditioning removes this. -->

## Conditional Granger Causality

**Problem:** Feature $X$ appears to Granger-cause $Y$, but only because both are driven by confounder $Z$.

**Solution:** Conditional Granger causality — control for $\mathbf{Z}$:

<div class="columns">

**Restricted** (Y history + Z):
$$y_t = \sum_{l} \alpha_l y_{t-l} + \sum_{l} \gamma_l z_{t-l} + \varepsilon_t^R$$

**Unrestricted** (+ X):
$$y_t = \sum_{l} \alpha_l y_{t-l} + \sum_{l} \beta_l x_{t-l} + \sum_{l} \gamma_l z_{t-l} + \varepsilon_t^U$$

</div>

$X$ conditionally Granger-causes $Y$ given $\mathbf{Z}$ iff $H_0: \boldsymbol{\beta} = \mathbf{0}$ is rejected.

---

<!-- Speaker notes: Nonlinearity is the gap in standard Granger testing. For financial data — think volatility, options implied vol, credit spreads — relationships are often highly nonlinear. The kernel approach is elegant: replace OLS with kernel ridge regression, then permutation test for significance. -->

## Nonlinear Granger Causality

Linear Granger tests miss nonlinear predictive relationships:

- Volatility regime effects
- Threshold crossing dynamics
- Fat-tail events driving correlated jumps

**Kernel Granger Causality (KGC):**

Replace OLS with kernel ridge regression in a reproducing kernel Hilbert space:

$$f(\mathbf{y}_{t-L:t-1}, \mathbf{x}_{t-L:t-1}) = \sum_i \alpha_i \kappa([\mathbf{y}, \mathbf{x}]_i, \cdot)$$

**Significance:** permutation test (shuffle $\mathbf{x}$ lags, recompute MSE reduction)

**Advantage:** captures any measurable relationship, not just linear.

---

<!-- Speaker notes: Neural Granger causality is the deep learning frontier. Tank et al. (2018) use group-sparse regularisation — the group lasso drives entire feature groups to zero, giving automatic feature selection. Good for very high-dimensional feature sets where you have enough data. -->

## Neural Network Granger Causality

**Tank et al. (2018) — Neural Granger Causality:**

Train a neural network with **group-lasso penalty** on input feature groups:

$$\mathcal{L}(\theta) = \sum_t \| y_t - f_\theta(\mathbf{y}_{t-L:t}, \mathbf{X}_{t-L:t}) \|^2 + \lambda \sum_j \| \mathbf{W}_{:,j} \|_2$$

The group-lasso term $\lambda \| \mathbf{W}_{:,j} \|_2$ drives the entire weight group for feature $j$ to zero if it is not predictive.

**Interpretation:** Selected features = those with non-zero input weight groups after training.

**When to use:** Large datasets ($T > 1000$), suspected nonlinear relationships, high-dimensional feature spaces.

---

<!-- Speaker notes: Spectral Granger causality is less common but powerful. It answers: "At which frequencies does X predict Y?" Useful for economic data with mixed cycles — monthly, quarterly, annual effects can be separated. The integral of spectral causality over all frequencies equals total linear Granger causality. -->

## Spectral Granger Causality

Decompose causal influence **by frequency band**:

$$\mathcal{F}_{X \to Y}(\lambda) = \ln \frac{S_{YY}(\lambda)}{S_{YY}(\lambda) - \sigma_{XX}|H_{YX}(\lambda)|^2}$$

where $H(\lambda)$ is the VAR transfer function (Fourier transform of impulse responses).

**Total causality** integrates across all frequencies:

$$\mathcal{F}_{X \to Y} = \frac{1}{2\pi} \int_{-\pi}^{\pi} \mathcal{F}_{X \to Y}(\lambda)\, d\lambda$$

**Use cases:**
- Identify features that predict only at seasonal frequencies
- Separate short-run vs long-run predictive information
- Detect spurious apparent causality from shared cycles

---

<!-- Speaker notes: Non-stationarity is the most common failure mode. I(1) series (random walks) produce spurious Granger causality — the asymptotic chi-squared distribution for the Wald test does not apply. The fix is differencing or using a VECM for cointegrated systems. -->

## Pitfall 1: Non-Stationarity

Granger tests require **stationary** time series. Testing on integrated series ($I(1)$, $I(2)$) produces spurious results — the $F$-distribution is invalid.

```
Stationarity Protocol:
  1. ADF test (H0: unit root) + KPSS test (H0: stationary)
  2. Stationary if: ADF rejects AND KPSS does not reject
  3. If non-stationary: difference once and re-test
  4. If cointegrated: use VECM, test in levels with error correction
```

**Financial data:** always use **log-returns**, not price levels.

$$r_t = \ln P_t - \ln P_{t-1}$$

---

<!-- Speaker notes: Common trends is the subtle second pitfall. Two trending series — say, two commodity prices in a supercycle — will appear to Granger-cause each other even after differencing if they share a stochastic trend. Conditioning on the cointegrating residual (error correction term) removes this. -->

## Pitfall 2: Spurious Causality from Common Trends

When two non-stationary series share a common stochastic trend (cointegration), bivariate Granger tests produce spurious rejections even in first differences.

**Example:** Crude oil and natural gas prices during commodity supercycles.

```
Detection:
  1. Engle-Granger or Johansen cointegration test
  2. If cointegrated: test in VECM framework
  3. Or: include error correction term as conditioning variable
     in conditional Granger test
```

**Why it matters for feature selection:** You may select features that appear predictive only because they share a long-run trend with the target, not because they carry short-run information.

---

<!-- Speaker notes: Multiple testing is critical when you have many candidate features. With p=100 at alpha=0.05, you expect 5 false positives. Bonferroni is too conservative for feature selection — BH-FDR is the right tradeoff between precision and recall. -->

## Multiple Testing Correction

Testing $p$ features simultaneously inflates false discovery rate.

**With $p = 100$ features at $\alpha = 0.05$:** expect $\approx 5$ false positives under $H_0$.

<div class="columns">

**Bonferroni (FWER):**
$$\text{Reject if } p_j < \frac{\alpha}{p}$$
Controls: familywise error rate
Tradeoff: very conservative, high false negatives

**Benjamini-Hochberg (FDR):**
$$\text{Reject } p_{(k)} < \frac{k}{p}\alpha, \text{ for largest } k$$
Controls: false discovery rate
Tradeoff: allows some false positives, higher power

</div>

**Recommendation:** Use BH-FDR (`fdr_bh` in statsmodels) for feature selection. Bonferroni for strict hypothesis testing.

---

<!-- Speaker notes: Bring it all together. This slide shows the complete pipeline. Walk through each step. Emphasise that the order matters: stationarity first, then testing, then correction, then optional conditional tests for the selected features. -->

## Complete Pipeline

```
Input: target series Y, candidate features X_1, ..., X_p

Step 1: Stationarity
  ADF + KPSS for target and all features
  Difference non-stationary series

Step 2: Bivariate Granger Tests
  For j = 1, ..., p:
    Test X_j -> Y at lags 1, ..., L_max
    Record min p-value

Step 3: Multiple Testing Correction
  Apply BH-FDR at alpha = 0.05
  Selected = {j : p_j_corrected < alpha}

Step 4: Optional Refinement
  Conditional Granger test among selected features
  Remove features with no conditional Granger causality
```

---

<!-- Speaker notes: This comparison slide helps learners place Granger causality in context. The key messages: Granger is the only method that explicitly tests temporal predictive direction. MI is univariate but stronger for nonlinearity. Correlation is fastest but weakest. -->

## Granger vs Correlation vs Mutual Information

| Property | Correlation | Mutual Information | Granger Causality |
|---|---|---|---|
| Temporal direction | No | No | Yes |
| Nonlinear | No | Yes | With extensions |
| Multiple testing | Easy | Easy | Easy |
| Stationarity required | No | No | Yes |
| Computational cost | Low | Medium | Medium–High |
| Interpretability | High | Medium | High |

**Recommendation:** Use Granger as the **primary filter** for time series feature selection. Use MI as a complementary check for nonlinear associations.

---

<!-- Speaker notes: End with a summary of what to take away. The three big ideas: (1) test predictive precedence not correlation; (2) conditional tests for confounders; (3) always correct for multiple comparisons. The notebook exercises make these concrete. -->

## Summary

**Key Takeaways**

1. Granger causality tests **predictive precedence** — which features improve forecasts of $Y$ beyond its own history.

2. The bivariate F-test is a restricted vs unrestricted AR comparison — simple, well-calibrated, and interpretable.

3. **Conditional Granger causality** removes confounders — essential when features are correlated with each other.

4. **Nonlinear extensions** (kernel, neural) capture financial regime effects and threshold dynamics.

5. **Always correct** for multiple testing — BH-FDR balances discovery power with false positive control.

6. **Non-stationarity** invalidates test distributions — difference or transform before testing.

---

<!-- Speaker notes: Point learners to the notebook where they will implement bivariate Granger tests on a real financial dataset, compute the directed information flow network, and compare Granger ranking to correlation and MI rankings. Emphasise that the multiple testing correction step is critical. -->

## Notebook 01: Granger Feature Ranking

**What you will build:**

- Load a real multivariate financial time series
- Compute Granger causality p-values for all features vs target
- Apply BH-FDR correction and compare with Bonferroni
- Compare Granger ranking against correlation and mutual information
- Visualise the **directed information flow network** using NetworkX

**Key skill:** You will see that correlation and Granger rankings differ substantially — features highly correlated with the target may have no Granger predictive power, and vice versa.

*See `notebooks/01_granger_feature_ranking.ipynb`*

---

<!-- Speaker notes: Further reading for advanced learners. The Granger 1969 original is readable and short. Barnett & Seth is the definitive computational reference. Tank 2018 is the neural extension. De Prado chapter 8 covers financial application context. -->

## Further Reading

- Granger, C.W.J. (1969). "Investigating Causal Relations..." *Econometrica* 37(3). — The original paper; readable in one sitting.

- Geweke, J. (1982). "Measurement of Linear Dependence and Feedback." *JASA* 77(378). — Spectral extension.

- Tank, A. et al. (2018). "Neural Granger Causality." *arXiv:1802.05842*. — Group-sparse neural approach.

- Barnett, L. & Seth, A.K. (2014). "The MVGC Toolbox." *J. Neuroscience Methods* 223. — Comprehensive VAR-based reference.

- de Prado, M.L. (2018). *Advances in Financial Machine Learning*. Wiley. — Feature importance in financial ML context.
