---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->
<!-- Speaker notes: Welcome to Module 4 on embedded feature selection methods. This deck covers regularisation paths — the mathematical heart of embedded selection. The key question to frame for students: how does penalising the loss function produce sparsity, and how can we compute all solutions at once? -->

# Regularisation Paths
## L1, L2, ElasticNet, and the LARS Algorithm

### Module 04 — Embedded Methods

How penalty geometry forces sparsity — and how to compute the complete path in one pass

---

<!-- Speaker notes: Start with the unified framing. Embedded methods are neither filter (no model) nor wrapper (external model calls). The penalty IS the selection mechanism. This is computationally efficient and theoretically well-understood. -->

## Embedded Methods: Selection Inside the Loss

Feature selection embedded directly into the optimisation objective.

$$\mathcal{L}(\beta) = \underbrace{\|y - X\beta\|_2^2}_{\text{data fit}} + \lambda \underbrace{P(\beta)}_{\text{selection penalty}}$$

| Method | Penalty $P(\beta)$ | Sparsity? |
|--------|-------------------|-----------|
| Ridge | $\|\beta\|_2^2$ | No |
| Lasso | $\|\beta\|_1$ | Yes |
| ElasticNet | $\alpha\|\beta\|_1 + \frac{1-\alpha}{2}\|\beta\|_2^2$ | Yes |

> The choice of $P(\beta)$ determines whether selection happens at all.

---

<!-- Speaker notes: Ridge has a closed form because the L2 penalty is smooth. Walk through the derivation step by step. The key insight is that the gradient of L2 at zero is zero — no force pushes coefficients to exact zero. -->

## Ridge Regression: Shrinkage Without Selection

$$\mathcal{L}_{\text{Ridge}}(\beta) = \|y - X\beta\|_2^2 + \lambda \|\beta\|_2^2$$

**Setting gradient to zero:**

$$-2X^\top(y - X\beta) + 2\lambda\beta = 0$$

$$\hat{\beta}_{\text{Ridge}} = (X^\top X + \lambda I)^{-1} X^\top y$$

**Why no zeros?** Gradient of penalty at $\beta_j = 0$ is $2\lambda \cdot 0 = 0$. No force drives coefficients to exact zero for finite $\lambda$.

---

<!-- Speaker notes: The geometric explanation is the most intuitive. Draw this on the board: an ellipse (contours of squared loss) and a circle (L2 ball). They touch at a smooth point, not a corner. Contrast this with the L1 ball on the next slide. -->

## The Geometry of Regularisation

<div class="columns">

**L2 Ball (Ridge): Smooth sphere**

$$\{\beta : \|\beta\|_2^2 \leq t\}$$

Loss contours touch the ball at a **smooth point** — no coordinate is forced to zero.

**L1 Ball (Lasso): Diamond/cross-polytope**

$$\{\beta : \|\beta\|_1 \leq t\}$$

Loss contours generically touch the ball at a **corner** — one coordinate is zero.

</div>

> The geometry of the constraint set determines whether sparsity can emerge.

---

<!-- Speaker notes: This is the mathematical heart of why Lasso works. The subdifferential at zero is an interval [-1, +1], which means there is a "range of zero-forcing". Walk through the three cases carefully. Students often struggle with the subdifferential — spend time here. -->

## KKT Conditions for Lasso

The L1 penalty $|\beta_j|$ is not differentiable at 0. Use the **subdifferential**:

$$\partial|\beta_j| = \begin{cases} \{+1\} & \beta_j > 0 \\ [-1, +1] & \beta_j = 0 \\ \{-1\} & \beta_j < 0 \end{cases}$$

**Stationarity conditions** (with residual $r = y - X\hat{\beta}$):

$$\boxed{\begin{cases} x_j^\top r = \frac{\lambda}{2} \cdot \text{sign}(\hat{\beta}_j) & \hat{\beta}_j \neq 0 \\ |x_j^\top r| \leq \frac{\lambda}{2} & \hat{\beta}_j = 0 \end{cases}}$$

---

<!-- Speaker notes: The KKT conditions have a beautiful interpretation. Active features are "maximally correlated" with the residual — they're the most informative features given what the model has already fit. This is exactly the LARS criterion. -->

## Reading the KKT Conditions

$$|x_j^\top r| \leq \frac{\lambda}{2}$$

**Active feature** ($\hat{\beta}_j \neq 0$): inner product with residual equals $\lambda/2$.

**Inactive feature** ($\hat{\beta}_j = 0$): inner product with residual is strictly less than $\lambda/2$.

As $\lambda$ **decreases**:
1. More features "reach" the correlation threshold
2. They enter the model sequentially
3. Residual $r$ shrinks as more variance is explained

> Features enter the model in order of their marginal correlation with the current residual.

---

<!-- Speaker notes: ElasticNet adds the L2 term to fix Lasso's failure mode with correlated features. The grouping effect theorem is crucial — state it clearly and give the intuition: if two features are perfectly correlated, ElasticNet splits the coefficient equally. -->

## ElasticNet: Combining L1 and L2

$$\mathcal{L}_{\text{EN}}(\beta) = \|y - X\beta\|_2^2 + \lambda\left[\alpha\|\beta\|_1 + \frac{1-\alpha}{2}\|\beta\|_2^2\right]$$

**Mixing ratio** $\alpha \in [0,1]$: $\alpha=1$ is Lasso, $\alpha=0$ is Ridge.

**Grouping effect theorem** (Zou & Hastie 2005):

$$|\hat{\beta}_j - \hat{\beta}_k| \leq \frac{1}{\lambda(1-\alpha)} \|x_j - x_k\|_2 \|y\|_2$$

Correlated features receive **similar coefficients** — the L2 term stabilises correlated groups.

---

<!-- Speaker notes: LARS is the algorithmic breakthrough that makes Lasso practical. Before LARS, you had to run coordinate descent separately for each lambda. LARS gives you the complete path in one pass. Emphasise: same cost as a single OLS fit. -->

## The LARS Algorithm: Motivation

**Problem:** Computing the Lasso at $K$ values of $\lambda$ requires $K$ separate optimisations.

**Insight (Efron et al. 2004):** The Lasso path is **piecewise linear**. Between consecutive knots, each coefficient changes linearly in $\lambda$.

**LARS gives the complete path in one pass:**
- Complexity: $O(p^3 + np^2)$ — equivalent to a single OLS fit
- $p$ steps total (one per feature entering/leaving)
- At each step: solve one small linear system

> LARS: Least Angle Regression and Shrinkage

---

<!-- Speaker notes: Walk through the LARS step carefully. The equiangular direction is the key concept — it is the direction in response space that maintains equal correlation with all active features. This is why LARS is called "least angle regression" — it moves at equal angles. -->

## LARS: The Equiangular Direction

With active set $\mathcal{A}$ and signs $s_j = \text{sign}(x_j^\top r)$:

**Step 1:** Build the signed design matrix $\tilde{X}_{\mathcal{A}} = X_{\mathcal{A}} \cdot \text{diag}(s_{\mathcal{A}})$

**Step 2:** Compute Gram matrix $G_{\mathcal{A}} = \tilde{X}_{\mathcal{A}}^\top \tilde{X}_{\mathcal{A}}$

**Step 3:** Normalisation constant $A_{\mathcal{A}} = (\mathbf{1}^\top G_{\mathcal{A}}^{-1} \mathbf{1})^{-1/2}$

**Step 4:** Equiangular direction $u_{\mathcal{A}} = \tilde{X}_{\mathcal{A}} G_{\mathcal{A}}^{-1} \mathbf{1} \cdot A_{\mathcal{A}}$

$$X_{\mathcal{A}}^\top u_{\mathcal{A}} = A_{\mathcal{A}} \cdot \mathbf{1}_{|\mathcal{A}|}$$

Direction $u_{\mathcal{A}}$ makes **equal angles** with all active features.

---

<!-- Speaker notes: The step length formula is where features enter the active set. For each inactive feature, compute the lambda at which it would reach the current maximum correlation. Take the minimum positive step. This is the geometric condition for the next knot in the path. -->

## LARS: Step Length and Feature Entry

**Residual evolution:** $r(\gamma) = r - \gamma \cdot u_{\mathcal{A}}$

**For each inactive feature $j \notin \mathcal{A}$**, compute when it reaches maximum correlation:

$$\gamma_j = \min\left\{\frac{C - c_j}{A_{\mathcal{A}} - a_j},\ \frac{C + c_j}{A_{\mathcal{A}} + a_j}\right\}_{+}$$

where $c_j = x_j^\top r$, $a_j = x_j^\top u_{\mathcal{A}}$, $C = \max_j |c_j|$

**Take minimum positive step:** $\gamma^* = \min_{j \notin \mathcal{A}} \gamma_j > 0$

**Lasso modification:** Also check if any active coefficient hits zero — if so, remove it first.

---

<!-- Speaker notes: The regularisation path plot is one of the most useful diagnostic tools in feature selection. Spend time on how to read it. The entry order, slope after entry, and sign changes all carry information. Use a concrete example from sklearn. -->

## Reading Regularisation Path Plots

```
Coefficient
    |
  β₁|‾‾‾‾‾‾‾‾‾‾‾\
    |              \
  β₂|    /‾‾‾‾‾‾‾‾‾\___
    |   /
    |__/
  β₃|_________________________ → 0
    |
    +--+--+--+--+--+--+--→ λ (decreasing →)
   max                    0
```

**What to read:**
- **Entry order** = importance rank
- **Steep slope** = strong effect size
- **Sign change** = multicollinearity artifact
- **Early plateau** = correlated feature absorbs variance

---

<!-- Speaker notes: Cross-validation selects the optimal lambda. The one-standard-error rule is important in practice — it selects a sparser model when performance is similar. Show the CV error curve shape: U-shaped with the minimum in the middle. -->

## Cross-Validated Lambda Selection

**LassoCV** searches over $\lambda$ grid using $k$-fold CV:

$$\hat{\lambda} = \argmin_\lambda \overline{\text{CV-MSE}}(\lambda)$$

**One-standard-error rule** (prefer sparser models):

$$\hat{\lambda}_{1SE} = \max\{\lambda : \text{CV-MSE}(\lambda) \leq \text{CV-MSE}(\hat{\lambda}) + \text{SE}(\hat{\lambda})\}$$

**ElasticNetCV** adds an outer loop over $\alpha$:

```python
enet_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
    cv=10, n_alphas=100
)
```

---

<!-- Speaker notes: The irrepresentable condition is the theoretical limit of Lasso. When correlated features violate this condition, Lasso fails to consistently select the true support. This motivates ElasticNet, Group Lasso, and stability selection. -->

## When Lasso Fails: The Irrepresentable Condition

Lasso consistently selects the true support $S$ **if and only if:**

$$\|X_{\bar{S}}^\top X_S (X_S^\top X_S)^{-1} \text{sign}(\beta_S)\|_\infty < 1$$

**Violation:** $X_{\bar{S}}$ (inactive features) are too correlated with $X_S$ (active features).

**Consequences of failure:**
- Arbitrary selection among correlated feature groups
- Coefficient instability across bootstrap samples
- True features excluded; correlated impostors selected

---

<!-- Speaker notes: Give concrete guidance on when to use each method. This is the practical takeaway. Lasso for uncorrelated features or when exact sparsity matters. ElasticNet for correlated groups. Group Lasso (next guide) when group structure is known. -->

## Decision Framework: Which Regulariser?

| Scenario | Recommended Method |
|----------|-------------------|
| Features are approximately uncorrelated | **Lasso** |
| Features form correlated groups, want representatives | **Lasso** |
| Features form correlated groups, want all group members | **ElasticNet** |
| $n < p$ and need more than $n$ features | **ElasticNet** |
| Groups are semantically meaningful | **Group Lasso** (Guide 02) |
| Want uncertainty-aware selection | **Stability Selection** (Guide 02) |

---

<!-- Speaker notes: Summarise the key mathematical results: the geometry (L1 ball corners), the KKT conditions (correlation thresholds), LARS (piecewise linear path), and cross-validation (lambda selection). These four pillars give a complete understanding of regularisation-based feature selection. -->

## Summary

**Geometry:** L1 ball corners force sparsity; L2 ball does not.

**KKT conditions:** Active features maintain $|x_j^\top r| = \lambda/2$; inactive features are below threshold.

**LARS:** Computes the complete Lasso path in $O(p^3)$ — one pass through all $p$ feature entries.

**Cross-validation:** Selects $\lambda$ for prediction; one-SE rule for sparsity.

**Failure mode:** Correlated features violate the irrepresentable condition — use ElasticNet or Group Lasso.

---

<!-- Speaker notes: Preview the next guide. Stability selection addresses the uncertainty in Lasso selection by asking: which features are selected consistently across bootstrap subsamples? This gives probabilistic selection rather than a hard threshold. -->

<!-- _class: lead -->

## Next: Stability Selection and Structured Sparsity

Guide 02 covers:
- Stability selection (Meinshausen & Bühlmann 2010)
- Theoretical guarantees on false discovery rate
- Group Lasso for structured selection
- Fused Lasso for adjacent features (time series)
