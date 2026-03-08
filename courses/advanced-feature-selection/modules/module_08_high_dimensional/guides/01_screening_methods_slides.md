---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->
<!-- Speaker notes: Welcome to Module 8. This deck covers the theory and practice of screening methods for ultra-high dimensional data — settings where p is in the thousands to millions and n is in the hundreds. The core idea is elegantly simple: use cheap marginal statistics to reduce p to a tractable size, then apply your expensive selection method on the reduced set. By the end you will understand when this is guaranteed to work, when it fails, and how to fix the failures. -->

# Feature Selection for High-Dimensional Data
## Sure Independence Screening and Its Variants

Module 8 · Advanced Feature Selection

---

## The Ultra-High Dimensional Problem

$$p \gg n: \quad p = O\!\left(\exp(n^\kappa)\right), \quad \kappa \in (0, 1)$$

| Domain | $n$ | $p$ | Ratio $p/n$ |
|--------|-----|-----|-------------|
| Genomics (GWAS) | 500 | 500,000 | 1,000× |
| Text (bag-of-words) | 1,000 | 100,000 | 100× |
| Financial signals | 252 | 8,000 | 32× |
| Proteomics | 80 | 20,000 | 250× |

**Why Lasso alone struggles here:**
- $O(p^2 n)$ compute per coordinate descent iteration
- Irrepresentable condition fails under strong collinearity
- Gram matrix $\mathbf{X}^\top \mathbf{X}$ is rank-deficient ($\text{rank} \leq n < p$)

<!-- Speaker notes: The ultra-high dimensional regime is genuinely different from merely high-dimensional. When p is in the hundred-thousands, the Gram matrix X^T X — which Lasso implicitly uses — is not just ill-conditioned, it is singular. Every row of the design matrix lies in an n-dimensional subspace, so we cannot even estimate all p coefficients simultaneously. The examples here are real: GWAS studies routinely have 500k SNPs with a few hundred patients, and commodity trading desks generate thousands of technical indicators daily. -->

---

## Sure Independence Screening: The Core Idea

**Two-step pipeline:**

```
Ultra-high dimensional (p >> n)
        │
        ▼
   [Step 1: Screen]
   Marginal correlation scores → keep top dₙ features
        │
        ▼
   [Step 2: Select]
   Lasso / SCAD / MCP on reduced X̃ ∈ ℝⁿˣᵈⁿ
        │
        ▼
   Final sparse model
```

**Key intuition:** Active features tend to have higher marginal correlation with $y$ than noise features. Use this cheap signal to discard obvious noise before the expensive step.

<!-- Speaker notes: The intuition is almost embarrassingly simple, which is why it took until 2008 for the theoretical guarantee to appear. Fan and Lv showed that under mild conditions, this intuition is not just heuristic — it is a provable property. The marginal screening step runs in O(pn) time, which is linear in p. The subsequent Lasso runs on a dataset with only d_n ~ n/log(n) features, which is now tractable. -->

---

## The Marginal Correlation Score

For standardised features and centred outcome:

$$\hat{\omega}_j = \frac{1}{n} \left| \mathbf{x}_j^\top \mathbf{y} \right| = |\widehat{\text{Cor}}(X_j, Y)|$$

**Screening rule:** Retain the $d_n$ features with the largest $\hat{\omega}_j$:

$$\hat{\mathcal{M}} = \left\{ j : \hat{\omega}_j \in \text{top-}d_n \text{ of } \hat{\omega}_1, \ldots, \hat{\omega}_p \right\}$$

**Canonical threshold** (Fan & Lv, 2008):

$$d_n = \left\lfloor \frac{n}{\log n} \right\rfloor$$

| $n$ | 100 | 200 | 500 | 1000 |
|-----|-----|-----|-----|------|
| $d_n$ | 21 | 38 | 80 | 144 |

<!-- Speaker notes: The score omega_j is nothing more than the absolute marginal correlation — a one-line computation per feature. The threshold d_n = floor(n / log n) has a beautiful property: it grows with n but slowly enough that the reduced problem remains tractable for Lasso. For n=200 we keep at most 38 features regardless of whether p is 5,000 or 500,000. This is the screening step's whole computational budget. -->

---

## The Sure Screening Property

**Condition A1 (Minimum Signal Strength):**
$$\min_{j \in \mathcal{M}^*} |\text{Cor}(X_j, Y)| \geq c \cdot n^{-\kappa}, \quad \kappa \in [0, 1/2)$$

Each active feature must have at least a weak marginal signal.

**Condition A2 (Moment Condition):** Sub-Gaussian features and errors.

**Theorem (Fan & Lv, 2008):** Under A1, A2, if $\log p = O(n^{1-2\kappa})$:

$$\boxed{P\!\left(\mathcal{M}^* \subseteq \hat{\mathcal{M}}\right) \geq 1 - O\!\left(p \cdot e^{-n^{1-2\kappa} c^2 / 2}\right) \to 1}$$

All true features survive screening with probability approaching 1.

<!-- Speaker notes: This theorem is the core theoretical contribution of Fan and Lv 2008. The key condition is A1: each active feature must have a non-trivial marginal correlation with y. The condition allows this correlation to shrink with n — specifically at rate n to the minus kappa for any kappa less than one-half. The probability of missing a true feature shrinks exponentially in n, while p can grow exponentially in n. That is the remarkable result: we can screen reliably even when p is vastly larger than n. -->

---

## Where SIS Fails: Three Scenarios

<div class="columns">

**Suppressor Variables**
- $X_1, X_2$ jointly relevant
- $\text{Cor}(X_1, Y) \approx 0$
- SIS discards $X_1$ — false negative

**High Multicollinearity**
- $X_j \approx X_k$, both active
- Marginal signals nearly identical
- SIS picks one, misses the other

</div>

**Interaction-Only Effects**
- True model: $Y = \beta_{12} X_1 X_2 + \varepsilon$
- $\text{Cor}(X_1, Y) = 0$ and $\text{Cor}(X_2, Y) = 0$
- Neither feature passes marginal screening

**Remedy for all three: Iterative SIS (ISIS)**

<!-- Speaker notes: Fan and Lv were well aware of these failure modes — they introduced ISIS in the same 2008 paper to address them. Suppressor variables are particularly common in finance: imagine two technical indicators that individually have zero predictive power but whose ratio or difference is highly predictive. Neither passes marginal screening. The iterative residualisation in ISIS is designed specifically to unmask such features. -->

---

## Iterative SIS (ISIS)

At iteration $t$, compute residuals from the current model, then screen features against residuals:

$$\mathbf{r}^{(t)} = \mathbf{y} - \mathbf{X}_{\hat{\mathcal{M}}^{(t-1)}} \hat{\boldsymbol{\beta}}^{(t-1)}$$

$$\hat{\omega}_j^{(t)} = \left|\text{Cor}(\mathbf{x}_j, \mathbf{r}^{(t)})\right| \quad \text{for } j \notin \hat{\mathcal{M}}^{(t-1)}$$

**Why residual screening recovers suppressors:**

After selecting $X_2$ in step 1, residual $\mathbf{r}^{(1)}$ contains the variance driven by $X_1$.

$$\text{Cor}(X_1, \mathbf{r}^{(1)}) \neq 0 \implies X_1 \text{ is detected in step 2}$$

Typically $T = 3$ iterations suffices in practice.

<!-- Speaker notes: The ISIS algorithm is elegant: residualise y with respect to the already-selected features, then run marginal screening against the residual. Any feature that was masked by a correlated feature in step 1 will now reveal its marginal association with the residual. Fan and Lv show that the sure screening property still holds for ISIS under slightly stronger conditions. In practice, 3 iterations is almost always sufficient — I have never seen a real dataset where more than 5 iterations changed the outcome. -->

---

## DC-SIS: Distance Correlation Screening

Replace Pearson correlation with **distance correlation** (Székely et al., 2007):

$$\widehat{\text{dCor}}(X_j, Y) = \frac{\widehat{\text{dCov}}(X_j, Y)}{\sqrt{\widehat{\text{dCov}}(X_j, X_j) \cdot \widehat{\text{dCov}}(Y, Y)}}$$

where $\widehat{\text{dCov}}^2(\mathbf{x}, \mathbf{y}) = \frac{1}{n^2}\sum_{k,l} A_{kl} B_{kl}$ and $A, B$ are doubly-centred distance matrices.

**Key property:**
$$\widehat{\text{dCor}} = 0 \iff X_j \perp Y \quad \text{(for continuous distributions)}$$

Detects **any** statistical dependence — linear or nonlinear. Sure screening property holds under weaker assumptions than SIS (no sub-Gaussian required).

<!-- Speaker notes: Distance correlation was introduced by Székely, Rizzo and Bakirov in 2007, and Li, Zhong and Zhu adapted it for screening in 2012. The computational cost is O(n^2) per feature rather than O(n) for Pearson correlation — which means DC-SIS total cost is O(p n^2). For p=5000 and n=200 that is still 200 billion operations, which is expensive but feasible on modern hardware. The key advantage is that the sure screening property holds under much weaker moment conditions and detects genuinely nonlinear dependencies. -->

---

## Screening Variants: Comparison

| Method | Detects | Cost | Key assumption |
|--------|---------|------|----------------|
| **SIS** | Linear | $O(pn)$ | Sub-Gaussian |
| **ISIS** | Linear + interactions | $O(Tpn)$ | Sub-Gaussian |
| **DC-SIS** | Any (distance corr.) | $O(pn^2)$ | Finite moments |
| **RRCS** | Monotone nonlinear | $O(pn \log n)$ | None |
| **NIS** | Any smooth nonlinear | $O(pn \log n)$ | Smoothness |

**Practical hierarchy:**

```
Is n < 500?  → SIS/ISIS (fastest, well-understood)
Nonlinear?   → DC-SIS (rigorous) or RRCS (robust)
Heavy tails? → RRCS (rank-based, outlier resistant)
Black-box?   → NIS (nonparametric, slowest but most flexible)
```

<!-- Speaker notes: In most financial and ML applications, SIS with ISIS is the right starting point. It is fast, theoretically clean, and the failure modes are well understood. DC-SIS is the go-to when you have strong prior belief that the outcome-feature relationship is nonlinear — e.g., option payoffs, volatility regimes. RRCS is excellent for commodity price data which has fat tails that violate the sub-Gaussian assumption. NIS is rarely used directly but informs the nonparametric independence tests we cover in Module 9. -->

---

## Threshold Selection: Three Approaches

### 1. Predetermined: $d_n = \lfloor n / \log n \rfloor$

Simple, theoretically justified, no data-adaptive tuning.

### 2. Extended BIC (Chen & Chen, 2008)

$$\text{EBIC}_\gamma = -2\ell(\hat{\boldsymbol{\beta}}) + |\hat{\mathcal{M}}| \log n + 2\gamma \log \binom{p}{|\hat{\mathcal{M}}|}$$

Set $\gamma = 1$ when $p \gg n$. Screen across a grid of $d_n$, select minimum EBIC.

### 3. Cross-Validated Screening

$K$-fold CV on the full SIS + Lasso pipeline. Expensive but directly minimises prediction error.

**Recommendation:** Start with $d_n = \lfloor n / \log n \rfloor$, validate with EBIC.

<!-- Speaker notes: The predetermined threshold is almost always a good starting point. The Fan and Lv paper shows it is rate-optimal. Extended BIC is the right tool when you want to be more data-adaptive without the computational cost of full cross-validation. The gamma parameter in EBIC controls how aggressively you penalise model complexity relative to p — for p/n ratios above 10, use gamma = 1. Cross-validated screening is mainly useful as a diagnostic to check whether the EBIC-selected threshold is reasonable. -->

---

## The Two-Step Pipeline: Complete Workflow

```python
from sklearn.linear_model import LassoCV
import numpy as np

def sis_lasso_pipeline(X, y, d_n=None):
    n, p = X.shape
    if d_n is None:
        d_n = int(n / np.log(n))  # Fan & Lv canonical threshold

    # Step 1: SIS screening — O(pn)
    X_std = (X - X.mean(0)) / X.std(0)
    y_ctr = y - y.mean()
    omega = np.abs(X_std.T @ y_ctr) / n
    screened_idx = np.argsort(omega)[::-1][:d_n]

    # Step 2: Lasso on reduced set — O(d_n^2 * n)
    X_reduced = X[:, screened_idx]
    lasso = LassoCV(cv=5, max_iter=5000).fit(X_reduced, y)
    active_in_reduced = np.where(lasso.coef_ != 0)[0]

    return screened_idx[active_in_reduced], lasso.coef_[active_in_reduced]
```

<!-- Speaker notes: This is the complete working implementation. Note the two critical steps: first, standardise X and centre y before computing omega — the marginal correlation is scale-dependent. Second, the returned indices are in the original feature space, not the reduced space, which is why we index screened_idx by active_in_reduced. This pipeline reduces a p=5000 problem to a d_n=38 problem for n=200, making Lasso trivially fast and numerically stable. -->

---

## Practical Failure Modes and Diagnostics

**Check 1: How many features pass with marginal signal?**
```python
# Distribution of omega scores — should see clear gap
plt.hist(omega, bins=100)
plt.axvline(omega[screened_idx[-1]], color='red', label=f'Threshold (d_n={d_n})')
```
If no clear gap exists, marginal correlations are all small — check for weak signal.

**Check 2: Post-screening VIF**
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(X_reduced, j) for j in range(X_reduced.shape[1])]
# VIF > 10 suggests multicollinearity — switch to elastic net
```

**Check 3: ISIS vs SIS agreement**
If ISIS selects substantially different features than SIS, suppressor variables are present.

<!-- Speaker notes: These three diagnostics should be run as a standard part of every SIS application. The histogram of omega scores is the most informative: a good screening scenario shows a fat body of near-zero scores with a clear elbow, and the threshold sits in the gap. When the gap is absent, you are in a difficult signal setting where ISIS and DC-SIS will perform better. The VIF check is essential because SIS does not handle multicollinearity — it screens correlated features based purely on their marginal associations. -->

---

## Real-World Application: Genomics Pipeline

**Scenario:** $n = 300$ patients, $p = 50{,}000$ SNPs, binary disease outcome.

```python
# 1. SIS with logistic marginal score
from scipy.stats import pointbiserialr

omega = np.array([
    abs(pointbiserialr(X[:, j], y).statistic)
    for j in range(p)
])
d_n = int(n / np.log(n))  # = 55 for n=300
screened = np.argsort(omega)[::-1][:d_n]

# 2. Logistic Lasso on screened features
from sklearn.linear_model import LogisticRegressionCV
clf = LogisticRegressionCV(penalty='l1', solver='saga', cv=5)
clf.fit(X[:, screened], y)
```

**Speed comparison:**
- Direct Lasso on $50{,}000$ features: ~45 minutes
- SIS + Lasso on 55 features: ~8 seconds

<!-- Speaker notes: The genomics application is the canonical motivating example from Fan and Lv's original paper. The speedup of 300x is real — computing the full Lasso path on 50,000 features with only 300 samples is not just slow, it is numerically problematic because the coordinate descent algorithm must cycle through all 50,000 features repeatedly. Marginal screening reduces this to a well-conditioned 300 × 55 problem that Lasso handles in milliseconds. The key change for classification is using point-biserial or logistic marginal scores instead of Pearson correlation. -->

---

## Summary: When to Use Each Method

```
p > 10 × n?
├── YES → Must screen before selection
│   ├── Linear relationships expected → SIS → Lasso
│   ├── Nonlinear relationships likely → DC-SIS → Random Forest
│   ├── Heavy-tailed data → RRCS → Robust regression
│   └── Interactions known → ISIS (T=3) → Lasso
└── NO  → Direct Lasso/elastic net without screening
```

**Critical rule:** Never report the screened set as the final model. Screening reduces $p$ for computational tractability; selection identifies the true active set.

<!-- Speaker notes: The decision tree here is the practical take-home. The dividing line is whether p exceeds 10 times n — below that, modern Lasso implementations are fast and stable enough to run directly. Above that line, screening is essential. The choice among SIS variants depends on your prior beliefs about the signal structure. When in doubt, run both SIS and ISIS and compare — disagreement between them is informative. And always remember: the screened set typically contains many false positives that the second-step selector must remove. -->

---

## Module 8 Roadmap

| Guide | Topic | Key methods |
|-------|-------|-------------|
| **01 (this)** | Screening | SIS, ISIS, DC-SIS, RRCS |
| 02 | Structured sparsity | Sparse PCA, Group Lasso, graph penalties |
| 03 | Post-selection inference | Debiased Lasso, selective inference, data splitting |

**Notebooks:**
- `01_sis_screening.ipynb` — Implement and compare screening methods
- `02_sparse_pca_selection.ipynb` — Structured sparsity in practice
- `03_post_selection_inference.ipynb` — Valid inference after selection

<!-- Speaker notes: This guide sets up the computational foundation for Module 8. Guide 02 addresses a complementary problem: when features have known structure — groups, graphs, hierarchies — how do we incorporate that structure into selection? Guide 03 addresses the statistical consequence of selection: once we have selected features, how do we make valid confidence intervals and p-values? All three guides connect at the end in Notebook 03 which walks through a complete screen → select → infer pipeline. -->

---

<!-- _class: lead -->

## Key Takeaways

1. SIS reduces $p \gg n$ to $d_n \sim n/\log n$ in $O(pn)$ time
2. The **sure screening property** guarantees $\mathcal{M}^* \subseteq \hat{\mathcal{M}}$ with high probability under minimum signal strength
3. **ISIS** recovers suppressor variables through iterative residualisation
4. **DC-SIS** detects nonlinear dependence; **RRCS** is robust to heavy tails
5. Always follow screening with a second selection step — screening is not selection

**Reference:** Fan & Lv (2008), *JRSS-B* 70(5): 849–911

<!-- Speaker notes: These five points are the essential summary. If a student remembers nothing else from this slide deck, they should remember that screening is a dimensionality reduction step with a theoretical guarantee, not a selection step. The sure screening property guarantees recall — we do not miss true features. It says nothing about precision — we may retain many false positives. That is why the two-step pipeline is essential: screening for recall, selection for precision. -->
