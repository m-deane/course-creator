# Stability Selection and Structured Sparsity

## In Brief

Stability selection (Meinshausen & Bühlmann 2010) wraps any regularisation method in a subsampling loop, measuring how often each feature is selected. Features with high selection probability across random subsamples are declared stable. This provides finite-sample false discovery rate control without parametric assumptions. Structured sparsity methods — Group Lasso, Sparse Group Lasso, Fused Lasso — extend the L1 penalty to enforce selection at the group or adjacent-feature level.

## Key Insight

A single Lasso fit at a fixed $\lambda$ gives a binary decision for each feature. Stability selection converts this into a probability, naturally handling the uncertainty that comes from correlated features and finite samples. The theoretical guarantee is that the expected number of falsely selected features is controlled by the stability threshold, subsampling fraction, and the regularisation parameter.

---

## 1. Stability Selection: Motivation and Setup

### The Problem with Single Lasso Fits

Running Lasso once at the CV-optimal $\lambda$ produces a sparse model, but this selection is fragile:

- Removing or adding a few observations can completely change which features are selected
- Among correlated features, the choice is nearly arbitrary
- The CV-optimal $\lambda$ optimises prediction, not support recovery

### The Stability Selection Idea

Run the regularisation method $B$ times on random subsamples of the data. For each subsample and each feature, record whether the feature is selected. The **stability probability** for feature $j$ is:

$$\hat{\Pi}_j^\lambda = \mathbb{P}^*[\text{feature } j \text{ selected at } \lambda]$$

where $\mathbb{P}^*$ is the empirical distribution over subsamples.

A feature is declared **stable** if its selection probability exceeds a threshold $\pi_{\text{thr}}$:

$$\hat{S}^{\text{stable}} = \{j : \hat{\Pi}_j^\lambda \geq \pi_{\text{thr}} \text{ for some } \lambda \in \Lambda\}$$

### Algorithm

```
Input: Data (X, y), subsample fraction q, threshold π_thr, lambda grid Λ, B subsamples

For b = 1 to B:
    1. Draw subsample I_b of size ⌊qn⌋ without replacement
    2. Fit Lasso path on (X_{I_b}, y_{I_b}) over grid Λ
    3. For each λ in Λ, record S^b(λ) = {j : β̂_j ≠ 0}

For each feature j:
    Π̂_j^λ = (1/B) Σ_b 1[j ∈ S^b(λ)]   (fraction of subsamples selecting j at λ)

Return: Ŝ^stable = {j : max_λ Π̂_j^λ ≥ π_thr}
```

---

## 2. Theoretical Guarantees: Per-Family Error Rate Control

### Main Theorem (Meinshausen & Bühlmann 2010)

Under the assumption that the subsampling fraction is $q \leq 1/2$ and the base selector has a certain exchangeability property:

$$\mathbb{E}[|\hat{S}^{\text{stable}} \cap \text{Irr}|] \leq \frac{1}{2\pi_{\text{thr}} - 1} \cdot \frac{q^2}{1 - q} \cdot \frac{\mathbb{E}[|\hat{S}^\lambda|]^2}{p}$$

where:
- $\hat{S}^{\text{stable}} \cap \text{Irr}$: stably selected features that are truly irrelevant
- $\hat{S}^\lambda$: average number of features selected at $\lambda$ over subsamples
- $\pi_{\text{thr}}$: the stability threshold
- $p$: total number of features

### Interpreting the Bound

**Expected number of false positives** is bounded by a quantity that depends on:
1. $\pi_{\text{thr}}$: higher threshold → fewer false positives (but more false negatives)
2. $\mathbb{E}[|\hat{S}^\lambda|]$: average model size at the chosen $\lambda$ — smaller models → fewer false positives
3. $p$: more features → looser bound (but bound still holds)

The bound is distribution-free — no Gaussianity or sparsity assumptions on the true model.

### Practical Guidance on Parameter Selection

**Subsample fraction $q$:** Use $q = 0.5$ (half the data per subsample). Smaller $q$ increases variance of stability estimates; larger $q$ reduces independence between subsamples.

**Number of subsamples $B$:** Use $B \geq 100$. The stability probabilities $\hat{\Pi}_j^\lambda$ are averages; $B = 50$ gives SE $\approx 0.07$, $B = 100$ gives SE $\approx 0.05$.

**Threshold $\pi_{\text{thr}}$:** Values between $0.6$ and $0.9$ are standard. Use $\pi_{\text{thr}} = 0.75$ as a starting point; increase to reduce false positives, decrease to improve recall.

**Lambda range $\Lambda$:** Stability selection is most informative at $\lambda$ values where the model is neither too full nor too empty. Use a geometric grid from $\lambda_{\max}$ (empty model) to $0.05 \lambda_{\max}$ (sparse but not trivially empty).

### Comparison: Stability Selection vs. Lasso Selection

| Property | Single Lasso | Stability Selection |
|-----------|-------------|---------------------|
| Output | Binary selection | Selection probability |
| Correlated features | Arbitrary choice | Both receive intermediate probability |
| False positive control | No guarantee | Finite-sample bound |
| Computational cost | One fit | $B \times |\Lambda|$ fits |
| Interpretability | Feature in/out | Feature selection probability |

---

## 3. Group Lasso: Selecting Groups of Related Features

### Motivation

Many feature sets have natural group structure:
- One-hot encoded categorical variables (dummy group)
- Polynomial or interaction features derived from one base feature
- Measurements from the same sensor at different lags
- Genes in the same biological pathway

Group Lasso selects **entire groups** rather than individual features.

### The Group Lasso Penalty

Partition features into $G$ non-overlapping groups $\mathcal{G}_1, \ldots, \mathcal{G}_G$ with group sizes $p_g = |\mathcal{G}_g|$:

$$\mathcal{L}_{\text{GL}}(\beta) = \|y - X\beta\|_2^2 + \lambda \sum_{g=1}^G \sqrt{p_g} \|\beta_{\mathcal{G}_g}\|_2$$

The penalty on group $g$ is the L2 norm of the group's coefficients (weighted by $\sqrt{p_g}$ to account for group size).

### Why This Produces Group Sparsity

The subdifferential of $\|\beta_g\|_2$ at $\beta_g = 0$ is the entire unit ball $\{\xi : \|\xi\|_2 \leq 1\}$. The KKT condition for group $g$ to be inactive is:

$$\|X_{\mathcal{G}_g}^\top r\|_2 \leq \lambda \sqrt{p_g}$$

This is a single condition on the entire group — either all group members have small residual correlation (group is inactive) or the group is active (all members included). There is no mechanism for partial group inclusion.

### Block Coordinate Descent for Group Lasso

The problem decouples across groups given the other groups. For group $g$, the update is a block soft-thresholding operation:

$$\hat{\beta}_g \leftarrow \left(1 - \frac{\lambda \sqrt{p_g}}{\|S_g\|_2}\right)_+ S_g$$

where $S_g = X_{\mathcal{G}_g}^\top r_g$ and $r_g = y - \sum_{g' \neq g} X_{\mathcal{G}_{g'}} \hat{\beta}_{g'}$ is the partial residual.

### Implementation

```python
# Group Lasso via group_lasso package or manual block coordinate descent
# sklearn does not natively support Group Lasso

from group_lasso import GroupLasso

# Define groups: feature indices belonging to each group
groups = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])  # 3 groups

gl = GroupLasso(
    groups=groups,
    group_reg=0.05,      # lambda for group penalty
    l1_reg=0,            # no within-group sparsity (pure group lasso)
    n_iter=1000,
    tol=1e-5
)
gl.fit(X_train, y_train)

# Check which groups are active
for g in np.unique(groups):
    group_coef = gl.coef_[groups == g]
    active = np.any(group_coef != 0)
    print(f"Group {g}: {'active' if active else 'inactive'}, coef_norm={np.linalg.norm(group_coef):.4f}")
```

---

## 4. Sparse Group Lasso: Group-Level and Within-Group Sparsity

### The Penalty

$$\mathcal{L}_{\text{SGL}}(\beta) = \|y - X\beta\|_2^2 + (1-\alpha)\lambda \sum_{g=1}^G \sqrt{p_g}\|\beta_{\mathcal{G}_g}\|_2 + \alpha\lambda \|\beta\|_1$$

This combines the Group Lasso (selects groups) with the Lasso (selects within groups). The mixing parameter $\alpha$ controls the balance:
- $\alpha = 0$: pure Group Lasso (group selection only)
- $\alpha = 1$: pure Lasso (individual feature selection)
- $\alpha \in (0,1)$: both group-level and within-group sparsity

### KKT Conditions

Feature $j$ in group $g$ is zero if and only if:

$$|x_j^\top r - (1-\alpha)\lambda \frac{\hat{\beta}_j}{\|\hat{\beta}_g\|_2}| \leq \alpha\lambda$$

For the entire group to be zero:

$$\|X_{\mathcal{G}_g}^\top r\|_2 \leq (1-\alpha)\lambda\sqrt{p_g}$$

### When to Use Sparse Group Lasso

Use when: groups are known, but not all group members are expected to be active. For example:
- Temporal lags of a feature (group = feature, within-group = which lags matter)
- Polynomial expansion (group = base variable, within-group = which degrees matter)

---

## 5. Fused Lasso: Adjacent Coefficient Similarity

### Motivation

In time series and genomics applications, features have a natural ordering (time, chromosomal position). Adjacent features are expected to have similar effects. The Fused Lasso encourages both sparsity and smoothness along the feature sequence.

### The Fused Lasso Penalty

$$\mathcal{L}_{\text{FL}}(\beta) = \|y - X\beta\|_2^2 + \lambda_1 \|\beta\|_1 + \lambda_2 \sum_{j=2}^{p} |\beta_j - \beta_{j-1}|$$

- $\lambda_1$: L1 penalty for overall sparsity
- $\lambda_2$: fusion penalty for adjacent coefficient differences

The fusion penalty $\sum |\beta_j - \beta_{j-1}|$ is the total variation of the coefficient sequence. Minimising it encourages **piecewise constant** coefficients — long runs of equal values separated by sharp changes.

### Connection to Total Variation Denoising

Without the L1 term, the Fused Lasso reduces to Total Variation (TV) denoising on the coefficient sequence. The $\lambda_1$ term additionally pushes coefficients to zero, combining sparsity with smoothness.

### The Signal Approximator / 1D Fused Lasso

The 1D Fused Lasso (signal approximator) solves:

$$\min_\beta \frac{1}{2}\|y - \beta\|_2^2 + \lambda \sum_{j=2}^{p} |\beta_j - \beta_{j-1}|$$

This has an efficient solution via the FLSA algorithm (Friedman et al. 2007) in $O(p)$ time.

### Applications in Time Series Feature Selection

For time series data where features are measurements at times $t-1, t-2, \ldots, t-p$:
- The Fused Lasso encourages selecting **consecutive lags** rather than isolated lags
- Produces contiguous blocks of selected lags — interpretable lag windows
- Particularly useful for financial data where autocorrelation structure is present

### Implementation via CVXPY

```python
import cvxpy as cp
import numpy as np

def fused_lasso(X, y, lambda1, lambda2):
    """
    Solve the Fused Lasso via convex programming.

    Parameters
    ----------
    X : array (n, p)
    y : array (n,)
    lambda1 : float — sparsity penalty
    lambda2 : float — fusion (smoothness) penalty

    Returns
    -------
    beta : array (p,) — coefficient vector
    """
    n, p = X.shape
    beta = cp.Variable(p)

    # Fused Lasso objective
    data_fit = 0.5 * cp.sum_squares(y - X @ beta)
    l1_penalty = lambda1 * cp.norm1(beta)
    fusion_penalty = lambda2 * cp.norm1(beta[1:] - beta[:-1])

    objective = cp.Minimize(data_fit + l1_penalty + fusion_penalty)
    problem = cp.Problem(objective)
    problem.solve(solver=cp.ECOS)

    return beta.value
```

---

## 6. Stability Path Plots

### Visualising Stability Paths

The stability path plots feature selection probability $\hat{\Pi}_j^\lambda$ against $\lambda$ (or equivalently, average model size $\mathbb{E}[|\hat{S}^\lambda|]$).

Interpretation:
- **High, flat curves**: features selected consistently across all $\lambda$ — very stable
- **Rising curves that plateau**: features enter at a specific $\lambda$ and remain — stable entry point
- **Noisy, low curves**: features selected near-randomly — unstable, likely false positives
- **Crossing curves**: features competing — correlated, one entering when the other exits

### The Stability Path vs. the Regularisation Path

| Property | Regularisation Path | Stability Path |
|----------|--------------------|-----------------|
| y-axis | Coefficient value | Selection probability |
| Captures uncertainty | No | Yes |
| Handles correlated features | Poorly | Naturally |
| Requires thresholding | On $\beta_j = 0$ | On $\hat{\Pi}_j$ |
| Produces gradations | Coefficient magnitude | Selection probability |

---

## 7. Practical Guidance

### Choosing the Stability Threshold $\pi_{\text{thr}}$

Apply the Meinshausen-Bühlmann bound to set $\pi_{\text{thr}}$ given a target false positive budget:

$$\mathbb{E}[\text{false positives}] \leq V$$

Solving for $\pi_{\text{thr}}$:

$$\pi_{\text{thr}} \geq \frac{1}{2}\left(1 + \frac{\mathbb{E}[|\hat{S}^\lambda|]^2}{p \cdot V}\right)$$

**Example:** $p = 100$, $\mathbb{E}[|\hat{S}^\lambda|] = 10$ (average 10 features selected), target $V = 1$ false positive:

$$\pi_{\text{thr}} \geq \frac{1}{2}\left(1 + \frac{100}{100 \cdot 1}\right) = 1.0$$

This means you cannot guarantee 1 false positive with 10-feature average selection — increase $\lambda$ to reduce $\mathbb{E}[|\hat{S}^\lambda|]$.

### Choosing Group Definitions

Group Lasso requires pre-specified groups. Sources:
1. **Domain knowledge**: gene pathways, economic sectors, physical measurement types
2. **Correlation clustering**: cluster features by correlation, use clusters as groups
3. **Factor loadings**: group by dominant principal component
4. **Feature origin**: one-hot encodings of one categorical variable form a natural group

### Base Estimator Choices for Stability Selection

Stability selection works with any sparse selector:
- **Lasso**: standard choice, well-understood path
- **ElasticNet**: better for correlated feature groups
- **Randomised Lasso**: add L1 perturbation for better randomisation
- **Random Forests** (threshold importance): non-parametric alternative

---

## Common Pitfalls

- **Too few subsamples**: Use $B \geq 100$. With $B = 20$, stability estimates have high variance.
- **Treating stability probability as a p-value**: $\hat{\Pi}_j$ is not a p-value. The false positive bound applies to the set of stable features, not individual probabilities.
- **Wrong group structure in Group Lasso**: Incorrect groups produce worse results than ungrouped Lasso. Use domain knowledge or data-driven grouping.
- **Fused Lasso without ordering**: The fusion penalty only makes sense when features have a meaningful order. Do not apply to unordered feature sets.
- **Single $\lambda$ for stability selection**: Report the stability path across $\lambda$, not just at one value. The path reveals the stability of the selection across regularisation strengths.

---

## Connections

- **Builds on:** Lasso regularisation (Guide 01), subsampling methods, convex optimisation
- **Leads to:** Knockoff filter (Guide 03), production selection pipelines (Module 11)
- **Related to:** Bootstrap selection, bagging, random subspace methods

---

## Further Reading

- Meinshausen & Bühlmann (2010) "Stability Selection" — the founding paper with theoretical guarantees
- Yuan & Lin (2006) "Model Selection and Estimation in Regression with Grouped Variables" — Group Lasso
- Simon et al. (2013) "A Sparse-Group Lasso" — Sparse Group Lasso with efficient algorithm
- Tibshirani et al. (2005) "Sparsity and Smoothness via the Fused Lasso" — Fused Lasso original paper
- Shah & Samworth (2013) "Variable Selection with Error Control: Another Look at Stability Selection" — improved theoretical analysis
