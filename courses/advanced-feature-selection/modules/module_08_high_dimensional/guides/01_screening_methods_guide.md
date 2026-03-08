# Sure Independence Screening and Ultra-High Dimensional Feature Selection

## In Brief

Sure Independence Screening (SIS) is a marginal correlation-based procedure that reduces ultra-high dimensional feature spaces ($p \gg n$) to a manageable size before applying penalised regression. Introduced by Fan & Lv (2008), SIS carries a theoretical guarantee — the *sure screening property* — that the true model features survive the screening step with probability approaching one.

## Key Insight

When $p$ is in the tens of thousands, even computing a Lasso path is expensive and numerically unstable. Marginal screening exploits a key empirical regularity: truly relevant features tend to have higher marginal correlation with the target than irrelevant ones. Screen first to get $p$ down to $O(n)$, then apply your method of choice on the reduced set.

---

## 1. The Ultra-High Dimensional Setting

### Problem Setup

Let $\mathbf{X} \in \mathbb{R}^{n \times p}$ and $\mathbf{y} \in \mathbb{R}^n$ with $p \gg n$. The standard high-dimensional regime assumes $p \sim n^\alpha$ for some $\alpha > 1$. The *ultra-high dimensional* regime allows $p = O(\exp(n^\kappa))$ for some $\kappa \in (0, 1)$ — exponential growth in $p$ relative to $n$.

Examples from practice:
- Genomics: $n = 200$ patients, $p = 500{,}000$ SNPs
- Text classification: $n = 1{,}000$ documents, $p = 100{,}000$ word n-grams
- Financial signals: $n = 252$ trading days, $p = 8{,}000$ derived features

In this regime, the Lasso suffers from:
1. **Computational cost**: $O(p^2 n)$ per iteration of coordinate descent when $p$ is very large
2. **Sign inconsistency**: The irrepresentable condition fails when features are highly correlated
3. **Numerical instability**: Gram matrix $\mathbf{X}^\top \mathbf{X}$ is ill-conditioned

SIS addresses all three by discarding irrelevant features before Lasso runs.

---

## 2. Sure Independence Screening (SIS)

### 2.1 Marginal Correlation Scores

For each feature $j = 1, \ldots, p$, compute the marginal correlation:

$$\hat{\omega}_j = |\text{Cor}(\mathbf{x}_j, \mathbf{y})| = \frac{|\mathbf{x}_j^\top \mathbf{y}|}{\|\mathbf{x}_j\|_2 \|\mathbf{y}\|_2}$$

where $\mathbf{x}_j$ is the $j$-th column of $\mathbf{X}$ (assumed centered).

For standardised features and centered outcomes, this simplifies to:

$$\hat{\omega}_j = \frac{1}{n} |\mathbf{x}_j^\top \mathbf{y}|$$

### 2.2 The Screening Step

Retain the features with the $d_n$ largest scores:

$$\hat{\mathcal{M}} = \left\{ j : \hat{\omega}_j \text{ is among the top } d_n \text{ of } \hat{\omega}_1, \ldots, \hat{\omega}_p \right\}$$

Fan & Lv (2008) recommend $d_n = \lfloor n / \log n \rfloor$ as the canonical choice, which keeps the reduced model Lasso-tractable.

### 2.3 The Sure Screening Property

**Definition (Minimum Signal Strength):** Let $\mathcal{M}^* = \{j : \beta_j^* \neq 0\}$ be the true active set, with $|\mathcal{M}^*| = s$.

**Condition (A1) — Minimum Marginal Correlation:** There exists a constant $c > 0$ such that:

$$\min_{j \in \mathcal{M}^*} |\text{Cor}(X_j, Y)| \geq c \cdot n^{-\kappa}$$

for some $\kappa \in [0, 1/2)$.

**Condition (A2) — Moment Condition:** The features and errors have bounded moments:

$$\mathbb{E}\left[\exp\left(t X_j^2\right)\right] \leq C < \infty \quad \text{for some } t > 0$$

**Theorem (Fan & Lv, 2008 — Sure Screening Property):** Under conditions (A1) and (A2), if $d_n \geq \lfloor n / \log n \rfloor$ and $\log p = O(n^{1-2\kappa})$, then:

$$P\left(\mathcal{M}^* \subseteq \hat{\mathcal{M}}\right) \geq 1 - O\left(p \cdot \exp(-n^{1-2\kappa} c^2 / 2)\right) \to 1$$

as $n \to \infty$.

**Interpretation:** With high probability, all truly active features survive screening. False negatives — discarding a relevant feature — become exponentially rare. False positives (irrelevant features that pass) are acceptable since Lasso will remove them in the second step.

### 2.4 When Does the Sure Screening Property Hold?

The key condition is (A1): each active feature must have a non-negligible *marginal* correlation with $y$. This fails when:

1. **Suppressor variables**: Feature $j$ is relevant conditionally but has zero marginal correlation (e.g., $X_1$ and $X_2$ are jointly important but individually cancel).
2. **High multicollinearity**: Two active features $X_j$ and $X_k$ are highly correlated and their marginal signals partially cancel.
3. **Interaction effects**: $X_j$ and $X_k$ interact but neither is marginally relevant individually.

---

## 3. Iterative SIS (ISIS)

### 3.1 Motivation

Marginal screening fails on suppressor variables and interactions. Iterative SIS (ISIS, Fan & Lv 2008; Fan, Samworth & Wu 2009) addresses this by conditioning on already-selected features in each iteration.

### 3.2 ISIS Algorithm

```
Algorithm: Iterative SIS (ISIS)

Input: X (n × p), y (n-vector), target model size d_n, max iterations T
Output: Selected feature set M_hat

1. Initialise: M_0 = {} (empty selected set)

2. For t = 1, 2, ..., T:
   a. Compute residuals: r = y - X_{M_{t-1}} * beta_hat_{t-1}
      (beta_hat_{t-1} = OLS coefficients on currently selected features)
   b. Compute marginal correlations of each remaining feature with residuals:
      omega_j = |Cor(x_j, r)| for j not in M_{t-1}
   c. Screen: add top d_t features to get M_t = M_{t-1} union {top d_t features}
   d. Apply Lasso on M_t to prune: M_t = {j in M_t : beta_hat_j != 0}
   e. If M_t == M_{t-1}: stop (converged)

3. Return M_T
```

### 3.3 Why Residual Screening Recovers Suppressors

Consider a suppressor variable scenario:
- True model: $y = \beta_1 x_1 + \beta_2 x_2 + \varepsilon$
- Structure: $\text{Cor}(x_1, y) \approx 0$ (suppressor), $\text{Cor}(x_2, y) > 0$ (strong)

Iteration 1: SIS selects $x_2$. Residual $r_1 = y - \hat{y}_{\{2\}}$ now contains the unexplained variation driven by $x_1$.

Iteration 2: Marginal correlation of $x_1$ with $r_1$ is now non-zero. SIS selects $x_1$.

The residualisation "unmasks" suppressor features.

---

## 4. Screening Variants

### 4.1 DC-SIS: Distance Correlation Screening

Distance correlation (Székely et al., 2007) captures both linear and nonlinear dependence. DC-SIS (Li, Zhong & Zhu, 2012) replaces marginal Pearson correlation with marginal distance correlation.

For vectors $\mathbf{x}_j$ and $\mathbf{y}$, the sample distance covariance is:

$$\widehat{\text{dCov}}^2(\mathbf{x}_j, \mathbf{y}) = \frac{1}{n^2} \sum_{k,l} A_{kl} B_{kl}$$

where $A_{kl}$ and $B_{kl}$ are doubly-centred distance matrices:

$$A_{kl} = a_{kl} - \bar{a}_{k\cdot} - \bar{a}_{\cdot l} + \bar{a}_{\cdot\cdot}$$

with $a_{kl} = |x_{jk} - x_{jl}|$ (Euclidean distance between observations $k$ and $l$ for feature $j$).

The distance correlation is:

$$\widehat{\text{dCor}}(\mathbf{x}_j, \mathbf{y}) = \frac{\widehat{\text{dCov}}(\mathbf{x}_j, \mathbf{y})}{\sqrt{\widehat{\text{dCov}}(\mathbf{x}_j, \mathbf{x}_j) \cdot \widehat{\text{dCov}}(\mathbf{y}, \mathbf{y})}}$$

**Key property:** $\widehat{\text{dCor}} = 0$ if and only if $X_j \perp Y$ (for continuous distributions). Pearson correlation only detects linear dependence.

**Sure screening property for DC-SIS** holds under weaker moment conditions than SIS — no Gaussian assumption required.

### 4.2 RRCS: Rank-Based Correlation Screening

RRCS (Li, Peng & Zhu, 2012) uses Kendall's $\tau$ or Spearman's $\rho$ as the marginal score. This provides robustness to:
- Heavy-tailed distributions (financial returns, commodity prices)
- Outliers in the feature space
- Monotone nonlinear relationships

Kendall's $\tau$ between $\mathbf{x}_j$ and $\mathbf{y}$:

$$\hat{\tau}_j = \binom{n}{2}^{-1} \sum_{k < l} \text{sign}\left[(x_{jk} - x_{jl})(y_k - y_l)\right]$$

### 4.3 NIS: Nonparametric Independence Screening

NIS (Fan, Feng & Song, 2011) uses additive model fitting. For each feature $j$, fit:

$$\hat{m}_j(x) = \arg\min_{m} \sum_{i=1}^n \left(y_i - m(x_{ij})\right)^2 + \lambda \int [m''(x)]^2 \, dx$$

The screening score is the fitted $R^2$:

$$\hat{\omega}_j^{\text{NIS}} = 1 - \frac{\sum_i (y_i - \hat{m}_j(x_{ij}))^2}{\sum_i (y_i - \bar{y})^2}$$

NIS detects complex nonlinear relationships that SIS and DC-SIS may miss but is computationally heavier: $O(p \cdot n \log n)$ vs $O(p \cdot n)$ for SIS.

### 4.4 Comparison Table

| Method | Dependence type detected | Computation | Assumption |
|--------|--------------------------|-------------|------------|
| SIS | Linear | $O(pn)$ | Sub-Gaussian |
| DC-SIS | Any (via distance correlation) | $O(pn^2)$ | Finite moments |
| RRCS | Monotone nonlinear | $O(pn \log n)$ | None |
| NIS | Any smooth nonlinear | $O(pn \log n)$ | Smoothness |
| ISIS | Linear + interactions | $O(Tpn)$ | Sub-Gaussian |

---

## 5. Threshold Selection

### 5.1 Predetermined Thresholds

Fan & Lv (2008) advocate $d_n = \lfloor n / \log n \rfloor$:

| $n$ | $d_n$ |
|-----|-------|
| 100 | 21 |
| 200 | 38 |
| 500 | 80 |
| 1000 | 144 |

This ensures the reduced problem remains amenable to Lasso while discarding the vast majority of the $p$ features.

### 5.2 Extended BIC (EBIC) for Threshold Selection

Chen & Chen (2008) propose the Extended BIC:

$$\text{EBIC}_\gamma(\hat{\mathcal{M}}) = -2\ell(\hat{\boldsymbol{\beta}}_{\hat{\mathcal{M}}}) + |\hat{\mathcal{M}}| \log n + 2\gamma \log \binom{p}{|\hat{\mathcal{M}}|}$$

where $\gamma \in [0, 1]$ controls penalisation of model complexity relative to $p$. For $\gamma = 0$, EBIC reduces to standard BIC. For $\gamma = 1$, the penalty grows as $|\hat{\mathcal{M}}| \log p$, which is appropriate for $p \gg n$.

**Practical recipe:** Run SIS across a grid of $d_n$ values, apply Lasso on each reduced set, and select the $d_n$ that minimises EBIC on the full dataset.

### 5.3 Cross-Validated Screening

A pragmatic alternative: $K$-fold cross-validation on the SIS + Lasso pipeline.

```
For d_n in [n/4, n/2, n, 2n, 5n]:
    For fold k in 1..K:
        Screen: M_hat_k = SIS(X_train_k, y_train_k, d_n)
        Fit: beta_hat_k = Lasso(X_train_k[:, M_hat_k], y_train_k)
        Score: err_k = MSE(y_val_k, X_val_k[:, M_hat_k] @ beta_hat_k)
    CV_score[d_n] = mean(err_k over folds)
Select: d_n* = argmin CV_score
```

**Caveat:** CV threshold selection is computationally expensive — $O(K \cdot |\text{grid}| \cdot pn)$. Prefer EBIC for very large $p$.

---

## 6. When Screening Fails

### 6.1 Multicollinearity

If active features $X_j$ and $X_k$ are highly correlated ($\rho_{jk} \approx 1$), their marginal correlations with $y$ may be very similar. SIS will likely retain one but not necessarily both. ISIS iterative residualisation partially resolves this.

**Diagnostic:** Compute pairwise correlations within $\hat{\mathcal{M}}$ after screening. If VIF > 10 for retained features, apply group screening or proceed with elastic net instead of Lasso.

### 6.2 Suppressor Variables

A suppressor variable $X_j$ satisfies $\text{Cor}(X_j, Y) \approx 0$ but $\text{Cor}(X_j, Y | X_{-j}) \neq 0$. SIS will discard $X_j$ in the first pass.

**Remedy:** Use ISIS with $T \geq 3$ iterations. After each iteration, residualise $y$ with respect to already-selected features. The suppressor's conditional relevance becomes visible through residual correlation.

### 6.3 Interaction-Only Effects

If $X_j$ is relevant only through its interaction with $X_k$ (neither has a main effect), both will have zero marginal correlation with $y$.

**Remedy:** Augment $\mathbf{X}$ with pairwise products $\{x_j x_k\}$ before screening, or use NIS which can detect these via nonparametric fits. Note: augmentation increases $p$ to $p + \binom{p}{2}$, which may be infeasible for very large $p$.

### 6.4 The Two-Step Pipeline and Its Guarantee

```
Ultra-high dimensional data (p >> n)
         |
         v
   Screening Step (SIS/ISIS/DC-SIS)
   Output: M_hat with |M_hat| = d_n ~ n/log(n)
         |
         v
   Selection Step (Lasso/SCAD/MCP)
   on X[:, M_hat] only
         |
         v
   Final model: sparse subset of M_hat
```

Provided the sure screening property holds, the oracle rate of the second-step estimator is preserved: the two-step estimator achieves the same rate as if we had known $\mathcal{M}^*$ in advance.

---

## Common Pitfalls

- **Applying SIS with too small $d_n$**: Setting $d_n < s$ (smaller than the true sparsity) guarantees false negatives. Always verify $d_n \geq$ your prior estimate of $s$.
- **Forgetting standardisation**: SIS uses marginal correlation which is scale-sensitive. Standardise $\mathbf{X}$ and center $\mathbf{y}$ before computing scores.
- **Single-pass SIS for complex data**: Financial and genomic data regularly exhibit suppressor structures. Default to ISIS with $T = 3$ iterations in practice.
- **Treating screening as selection**: The screened set $\hat{\mathcal{M}}$ still requires a second-step selector. Never report $\hat{\mathcal{M}}$ directly as the final model.

---

## Connections

- **Builds on:** Lasso (Module 4), information theory screening (Module 1), filter methods (Module 1)
- **Leads to:** Post-selection inference (Guide 03), structured sparsity (Guide 02)
- **Related to:** Forward stepwise regression, stability selection (Module 10)

---

## Further Reading

- **Fan, J. & Lv, J. (2008).** "Sure Independence Screening for Ultra-High Dimensional Feature Space." *Journal of the Royal Statistical Society: Series B*, 70(5), 849–911. — The foundational paper. Read Section 2 for the sure screening theorem, Section 4 for the ISIS extension.
- **Li, R., Zhong, W. & Zhu, L. (2012).** "Feature Screening via Distance Correlation Learning." *Journal of the American Statistical Association*, 107(499), 1129–1139. — DC-SIS with weaker assumptions.
- **Fan, J., Samworth, R. & Wu, Y. (2009).** "Ultrahigh Dimensional Feature Selection: Beyond the Linear Model." *Journal of Machine Learning Research*, 10, 2013–2038. — ISIS and nonlinear extensions.
- **Chen, J. & Chen, Z. (2008).** "Extended Bayesian Information Criteria for Model Selection with Large Model Spaces." *Biometrika*, 95(3), 759–771. — EBIC derivation and theory.
