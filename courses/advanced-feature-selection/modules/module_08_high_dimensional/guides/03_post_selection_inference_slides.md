---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Post-Selection Inference
## Valid Confidence Intervals After Feature Selection

Module 8 · Advanced Feature Selection

<!-- Speaker notes: This is one of the most practically important topics in applied statistics that most practitioners get wrong. When you use data to select features and then estimate their effects using the same data, your confidence intervals do not have the stated coverage. This is not a subtle asymptotic issue — in realistic settings, a "95%" CI based on Lasso-selected OLS can have actual coverage of 45%. We cover three solutions: the Debiased Lasso, selective inference, and data splitting. -->

---

## The Problem: Why Naive CIs Fail

**Standard workflow (WRONG for inference):**
1. Run Lasso on full data → select features $\hat{\mathcal{M}}$
2. Run OLS on $\mathbf{X}_{\hat{\mathcal{M}}}$ → get $\hat{\boldsymbol{\beta}}^{\text{OLS}}$
3. Report $\hat{\beta}_j \pm 1.96 \hat{\sigma}_j$ as 95% CI

**Why this fails — selection bias:**

A feature $j$ is selected because $|\hat{\beta}_j|$ is large *in this sample*. Reporting a CI centred on the value that was large enough to trigger selection inflates coverage.

| Method | Stated coverage | Actual coverage |
|--------|----------------|-----------------|
| Naive OLS on selected set | 95% | ~50% |
| Debiased Lasso | 95% | ~94% |
| Data splitting | 95% | ~96% |

<!-- Speaker notes: The empirical coverage numbers here come from simulation studies with n=100, p=500, s=10 active features. The naive approach has coverage around 50% — roughly half the stated level. This is not a small-sample artifact. It persists even with n=1000. The mechanism is clear: features are selected because their observed coefficients are large. The observed coefficient is an overestimate of the true coefficient due to noise — this is the winner's curse or regression to the mean. Reporting a CI based on the overestimated value produces intervals that miss the true value systematically. -->

---

## The Selection Event is a Constraint on $\mathbf{y}$

The Lasso selection $\hat{\mathcal{M}} = \mathcal{M}$ is equivalent to:

$$\left\{ \hat{\mathcal{M}}(\mathbf{y}) = \mathcal{M} \right\} = \left\{ \mathbf{A}_{\mathcal{M}} \mathbf{y} \leq \mathbf{b}_{\mathcal{M}} \right\}$$

A **polyhedral set** in $\mathbf{y}$-space (Lee et al., 2016).

**Consequence for inference:**

$$\mathbb{E}[\hat{\beta}_j^{\text{OLS}} \mid \mathcal{E}_\lambda] \neq \beta_j^*$$

The OLS estimate, *conditional on being selected*, is biased upward. The bias does not vanish with $n$ at the same rate as the standard error.

Three solutions:
1. **Debiased Lasso** — correct the bias directly
2. **Selective inference** — account for conditioning on $\mathcal{E}_\lambda$
3. **Data splitting** — use separate data for selection and inference

<!-- Speaker notes: The polyhedral geometry insight from Lee et al. 2016 is the key theoretical advance. The Lasso selection event is not some abstract concept — it corresponds to a concrete set of linear inequalities on y. Each of these inequalities arises from the KKT conditions of the Lasso optimisation. If we know which constraints are active, we know exactly what the conditional distribution of y looks like, and we can compute exact conditional p-values and confidence intervals. This is beautiful but computationally intensive. -->

---

## Solution 1: Debiased Lasso

**The bias correction (van de Geer et al., 2014; Zhang & Zhang, 2014):**

$$\hat{\boldsymbol{\beta}}^{\text{d}} = \underbrace{\hat{\boldsymbol{\beta}}^{\text{Lasso}}}_{\text{biased}} + \underbrace{\frac{1}{n}\hat{\mathbf{\Theta}}\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}^{\text{Lasso}})}_{\text{bias correction}}$$

$\hat{\mathbf{\Theta}} \approx (\mathbf{X}^\top\mathbf{X}/n)^{-1}$: approximate precision matrix

The correction term = precision matrix $\times$ Lasso residuals

**Asymptotic distribution:**

$$\frac{\sqrt{n}(\hat{\beta}_j^{\text{d}} - \beta_j^*)}{\hat{\sigma}_j} \xrightarrow{d} \mathcal{N}(0,1)$$

Valid $(1-\alpha)$ CI: $\hat{\beta}_j^{\text{d}} \pm z_{1-\alpha/2} \cdot \hat{\sigma}_j/\sqrt{n}$

<!-- Speaker notes: The debiased Lasso formula looks simple but hides significant computation in the hat Theta matrix. When p > n, we cannot compute the standard inverse of X^T X / n. Instead, we compute an approximate inverse using nodewise regression. The residuals y - X beta_hat are the Lasso residuals — these are not small, since the Lasso shrinks coefficients toward zero leaving systematic unexplained variation. The bias correction amplifies these residuals through the approximate precision matrix to remove the shrinkage bias. -->

---

## The Nodewise Lasso: Computing $\hat{\mathbf{\Theta}}$

When $p > n$, $\mathbf{X}^\top\mathbf{X}$ is singular. Compute column $j$ of $\hat{\mathbf{\Theta}}$ via a Lasso regression of $\mathbf{x}_j$ on all other features:

```python
def nodewise_lasso(X, mu=None):
    n, p = X.shape
    Theta_hat = np.zeros((p, p))

    for j in range(p):
        # Regress x_j on all other features
        X_minus_j = np.delete(X, j, axis=1)
        if mu is None:
            mu = np.sqrt(np.log(p) / n)

        lasso = Lasso(alpha=mu).fit(X_minus_j, X[:, j])
        gamma_j = lasso.coef_

        # Residual of x_j unexplained by others
        m_j = X[:, j] - X_minus_j @ gamma_j
        tau_j = m_j @ X[:, j] / n  # normalisation

        # Fill column j of Theta_hat
        Theta_hat[j, j] = 1.0 / tau_j
        idx = np.delete(np.arange(p), j)
        Theta_hat[j, idx] = -gamma_j / tau_j

    return Theta_hat
```

Cost: $p$ Lasso regressions → parallelisable.

<!-- Speaker notes: The nodewise Lasso is the computational heart of the debiased Lasso procedure. For each feature j, we regress x_j on all other features using Lasso. The residual of this regression, m_j, represents the part of x_j that cannot be explained by other features — essentially the unique information in x_j. The normalisation constant tau_j is the inner product of m_j with x_j, which measures how much unique information j provides. The approximate precision matrix column j is then constructed from the regression coefficients and tau_j. For p=500 this requires 500 Lasso regressions, which is feasible. For p=50,000 this requires parallelisation across a cluster. -->

---

## Solution 2: Selective Inference

**The Polyhedral Lemma (Lee et al., 2016):**

For any linear statistic $\eta^\top\mathbf{y}$, its conditional distribution given $\mathcal{E}_\lambda$ is **truncated normal**:

$$\eta^\top\mathbf{y} \;\Big|\; \mathcal{E}_\lambda \sim \text{TruncNorm}\!\left(\eta^\top\mathbf{X}\boldsymbol{\beta}^*, \sigma^2\|\eta\|^2, [\mathcal{V}^-, \mathcal{V}^+]\right)$$

**Truncation limits** $[\mathcal{V}^-, \mathcal{V}^+]$ computed from the polyhedral constraints.

**Conditional p-value:**

$$p_j = P\!\left(|T| \geq |\eta^\top\mathbf{y}| \;\Big|\; \mathcal{E}_\lambda\right)$$

where $T \sim \text{TruncNorm}(0, 1, [\mathcal{V}^-/\sigma, \mathcal{V}^+/\sigma])$ under $H_0: \beta_j^* = 0$.

These p-values are **exactly valid** for any $n$ and $p$.

<!-- Speaker notes: The selective inference approach is theoretically exact, not asymptotic. For finite n and any p, the conditional p-values have the correct uniform distribution under the null. The key computation is finding the truncation limits V_minus and V_plus. These are derived from the polyhedral constraints A_M y <= b_M by considering which constraints would become violated as we move y in the direction of eta. The implementation requires computing these limits for each feature in the selected set, which involves solving a set of linear programs. The selectiveInference R package automates this. -->

---

## Selective Inference: When It Applies

**Strengths:**
- Exact finite-sample validity (no asymptotics)
- Applies to Lasso, forward stepwise, LARS
- Naturally handles unknown selection threshold

**Limitations:**
- Requires known $\sigma^2$ (or estimated from data)
- Computationally intensive for large $|\hat{\mathcal{M}}|$
- Power loss from conditioning on the selection event
- Conservative when many features are selected

**Use selective inference when:**
- $n < 200$ (asymptotics unreliable for debiased Lasso)
- Exact statements are needed (regulatory or clinical settings)
- $|\hat{\mathcal{M}}| \leq 20$ (tractable conditioning)

<!-- Speaker notes: The choice between debiased Lasso and selective inference is essentially a choice between asymptotic and exact guarantees, at the cost of power. Selective inference is exact but tends to produce wider intervals because conditioning on the selection event reduces the effective sample size. Debiased Lasso is asymptotic but often more powerful because it does not condition on the selection event — it directly corrects the bias in the estimator. In practice, for n > 300 I almost always recommend the debiased Lasso. For smaller n or regulated applications, selective inference is the appropriate tool. -->

---

## Solution 3: Data Splitting

**The simplest valid approach:**

```
All n observations
    ├── First n/2  → Feature selection (Lasso/SIS)
    │              → Output: M_hat
    └── Second n/2 → OLS on X[:, M_hat]
                   → Standard CIs (valid, no selection bias)
```

**Why it works:** The second-half data is independent of $\hat{\mathcal{M}}$ (which was computed on the first half). Classical OLS theory applies — no post-selection correction needed.

**Cost:** Each step uses only $n/2$ observations.

**Multi-split to reduce variability (Meinshausen et al., 2009):**
Repeat $B = 100$ times with different splits, aggregate p-values across splits.

<!-- Speaker notes: Data splitting is the conceptually simplest approach and often the most practical for large datasets. The key advantage is that it requires no modification to existing software — any Lasso implementation for selection and any OLS implementation for inference will give valid results. The cost is efficiency: with n/2 observations for each step, confidence intervals are wider by a factor of sqrt(2) compared to methods that use all n observations. Multi-splitting reduces the variance from the random split but requires running the full pipeline B times. For n > 1000, data splitting with B=50 splits is my default recommendation. -->

---

## Multi-Split: Aggregating p-values

```python
import numpy as np
from sklearn.linear_model import LassoCV, LinearRegression
from scipy import stats

def multi_split_pvalues(X, y, B=100, gamma=0.05):
    n, p = X.shape
    all_pvals = {j: [] for j in range(p)}

    for b in range(B):
        idx = np.random.permutation(n)
        i_sel, i_inf = idx[:n//2], idx[n//2:]

        # Select on first half
        lasso = LassoCV(cv=3).fit(X[i_sel], y[i_sel])
        sel = np.where(lasso.coef_ != 0)[0]
        if len(sel) == 0:
            continue

        # OLS inference on second half
        ols = LinearRegression().fit(X[np.ix_(i_inf, sel)], y[i_inf])
        resid = y[i_inf] - ols.predict(X[np.ix_(i_inf, sel)])
        s2 = resid.var(ddof=len(sel)+1)
        XtXinv = np.linalg.pinv(X[np.ix_(i_inf, sel)].T @ X[np.ix_(i_inf, sel)])
        se = np.sqrt(s2 * np.diag(XtXinv))
        t_stat = ols.coef_ / se
        pv = 2 * stats.t.sf(np.abs(t_stat), df=n//2 - len(sel) - 1)

        for idx_f, j in enumerate(sel):
            all_pvals[j].append(min(1, pv[idx_f] * len(sel)))  # Bonferroni within split

    # Aggregate: quantile aggregation
    agg = {}
    for j in range(p):
        if all_pvals[j]:
            agg[j] = min(1, np.quantile(all_pvals[j], gamma) / gamma)
    return agg
```

<!-- Speaker notes: This implementation of the multi-split procedure follows Meinshausen, Meier and Bühlmann 2009. The key steps are: within each split, apply Bonferroni correction for the number of selected features (multiplying p-values by the selection set size); then aggregate across splits using a quantile rather than a mean. The gamma-quantile aggregation provides a valid p-value that accounts for the randomness of the split while maintaining the correct type-I error rate. Using gamma=0.05 gives a conservative but valid aggregation. -->

---

## Knockoff+: FDR Control in High Dimensions

**Knockoff features** $\tilde{\mathbf{X}}$ satisfy:
- Same column correlations: $\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}} = \mathbf{X}^\top\mathbf{X}$
- Cross-correlations match: $\mathbf{X}^\top\tilde{\mathbf{X}} = \text{diag}(\mathbf{s})$

**Knockoff statistic:** $W_j = Z_j - \tilde{Z}_j$ where $Z_j, \tilde{Z}_j$ are Lasso entry times.

**Selection threshold (Barber & Candès, 2015):**

$$T = \min\left\{t > 0: \frac{\#\{j: W_j \leq -t\} + 1}{\#\{j: W_j \geq t\}} \leq q\right\}$$

**Theorem:** $\text{FDR} = \mathbb{E}\!\left[\frac{\text{false positives}}{\text{selected} \vee 1}\right] \leq q$ in **finite samples**.

No distribution assumptions on $\boldsymbol{\beta}^*$ — valid for any sparsity level.

<!-- Speaker notes: The Knockoff procedure is remarkable for its finite-sample FDR guarantee. Unlike the debiased Lasso and selective inference which rely on asymptotic arguments, Knockoffs control FDR exactly for any n and p. The mechanism is elegant: for each feature j, construct a knockoff that has the same correlations with all other features but is a controlled null. Run Lasso on the augmented design matrix [X, X_tilde]. Features that enter before their knockoffs have evidence for relevance; features that enter after their knockoffs are likely noise. The threshold T is chosen so that the expected false discovery rate is at most q. -->

---

## Choosing a Post-Selection Inference Method

| Setting | Recommended method |
|---------|-------------------|
| $n > 300$, any $p$ | Debiased Lasso |
| $n < 200$, $|\hat{\mathcal{M}}| < 20$ | Selective inference |
| Simple implementation needed | Data splitting |
| FDR control required | Knockoff+ |
| $p > 10{,}000$, missing data | Multi-split + stability |

**Universal rule:** Never report OLS p-values on Lasso-selected features without correction. The bias can exceed 10× in realistic settings.

<!-- Speaker notes: This comparison table is the practical decision guide. In most real-world settings with n > 300, the debiased Lasso is the right default — it is asymptotically valid, computationally feasible, and provides individual confidence intervals for all selected features. Data splitting is the right choice when you need a simple, explainable procedure or when the sample size is large enough that losing half the data for selection is not costly. Knockoffs are the right choice when you want to report a feature list with a guaranteed false discovery rate — this is the appropriate framework for discovery studies in genomics or drug development. -->

---

## Feature Selection with Missing Data

**Problem:** $p = 5{,}000$ features, 15% values missing. Complete-case analysis drops 80% of rows.

**Solution: Multiple Imputation + Stability Selection**

```
For m = 1..M imputation rounds:
    X_m = MICE(X)  # impute missing values
    For b = 1..B subsamples:
        Run Lasso on subsample of X_m
        Record selected features
    stability_score_m = mean selections across B

Aggregate: stability_score = mean over M rounds
Select: features with score > 0.7
```

Features robust across imputation rounds and subsamples → unlikely to be selection artefacts.

<!-- Speaker notes: Missing data in high-dimensional settings is a compounding problem: the more features, the more likely any given observation has at least one missing value. Complete-case analysis is not viable. The multiple imputation approach deals with this by running the full selection procedure on multiple imputed datasets and aggregating the results. The stability selection layer — subsampling within each imputed dataset — controls for selection instability due to both the missing data and the high-dimensional selection process. Features that consistently appear across imputation rounds are robust to the specific imputation model. -->

---

## The Complete Screen → Select → Infer Pipeline

```python
# 1. Screen: reduce p to d_n
screened = sis_screen(X, y, d_n=int(n/np.log(n)))
X_sc = X[:, screened]

# 2. Select: Lasso on screened features
lasso = LassoCV(cv=5).fit(X_sc, y)
selected_in_sc = np.where(lasso.coef_ != 0)[0]
selected = screened[selected_in_sc]  # original feature indices

# 3. Infer: debiased Lasso on selected features
X_sel = X[:, selected]
beta_lasso = lasso.coef_[selected_in_sc]
Theta = nodewise_lasso(X_sel)
correction = Theta @ X_sel.T @ (y - X_sel @ beta_lasso) / n
beta_debiased = beta_lasso + correction

# 4. Confidence intervals
sigma_hat = np.std(y - X_sel @ beta_lasso)
se = sigma_hat * np.sqrt(np.diag(Theta)) / np.sqrt(n)
ci_lower = beta_debiased - 1.96 * se
ci_upper = beta_debiased + 1.96 * se
```

<!-- Speaker notes: This is the complete pipeline connecting all three guides in Module 8. Step 1 uses SIS from Guide 01 to reduce p from thousands to tens. Step 2 uses standard Lasso on the reduced feature set. Step 3 applies the debiased Lasso correction from this guide. Step 4 computes confidence intervals from the debiased estimates. The key thing to note is that the nodewise Lasso in Step 3 runs on the selected feature set, not on all p features — this is appropriate because we are doing inference conditional on having passed the screening step, and the selected set is small enough for nodewise regression to be fast. -->

---

## Summary: Post-Selection Inference

1. **Naive OLS on selected features** gives coverage ~50% for stated 95% CIs — never do this

2. **Debiased Lasso**: asymptotically valid CIs for all coefficients simultaneously; requires nodewise regression for $\hat{\mathbf{\Theta}}$

3. **Selective inference**: exact finite-sample CIs conditional on the Lasso selection event; based on the truncated normal distribution

4. **Data splitting**: conceptually simple, always valid, costs half the data; multi-split reduces split variability

5. **Knockoff+**: finite-sample FDR control; requires constructing exchangeable knockoff features

<!-- Speaker notes: The key takeaway is that valid post-selection inference is not optional — it is necessary for any scientific or business conclusion drawn from feature selection. The choice among methods depends on sample size, computational budget, and the type of guarantee needed. Debiased Lasso is the default for large n. Selective inference for small n or exact guarantees. Data splitting for simplicity. Knockoffs for FDR control. In the notebook, we will demonstrate all four methods on the same simulated dataset and compare their coverage rates empirically. -->

---

<!-- _class: lead -->

## Module 8 Complete

You can now:
- Screen $p \gg n$ features with SIS/ISIS/DC-SIS
- Apply structured penalties (Group Lasso, Graph Lasso, Sparse PCA)
- Construct valid confidence intervals after selection
- Choose the right inference method for your setting

**Notebooks:**
- `01_sis_screening.ipynb` — Implement screening methods
- `02_sparse_pca_selection.ipynb` — Structured sparsity in practice
- `03_post_selection_inference.ipynb` — Valid inference after selection

<!-- Speaker notes: This completes Module 8. The three guides and three notebooks cover the full pipeline from raw ultra-high dimensional data to valid inferential statements about selected features. The exercises in exercises/01_high_dim_exercises.py give you practice implementing DC-SIS, Group Lasso, and the coverage comparison. Module 9 will connect this to causal feature selection — asking not just which features predict y but which features causally influence y. -->
