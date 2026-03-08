# Regularisation Paths: L1, L2, ElasticNet, and the LARS Algorithm

## In Brief

Regularisation-based feature selection embeds the selection criterion directly into the model's loss function. The penalty term shrinks coefficients toward zero, and L1 regularisation (Lasso) sets coefficients exactly to zero — performing selection implicitly. The entire selection path as the penalty strength varies can be computed in one pass using the LARS algorithm.

## Key Insight

Lasso does not simply shrink all coefficients uniformly. The geometry of the L1 ball forces the optimum to land at a corner — a sparse solution. Understanding why sparsity emerges from L1 but not L2 requires reading the KKT conditions carefully.

---

## 1. L2 Regularisation (Ridge): Shrinkage Without Selection

### Loss Function

$$\mathcal{L}_{\text{Ridge}}(\beta) = \underbrace{\|y - X\beta\|_2^2}_{\text{data fit}} + \lambda \underbrace{\|\beta\|_2^2}_{\text{L2 penalty}}$$

### Closed-Form Solution

Because the L2 penalty is differentiable everywhere, the optimum has a closed form:

$$\hat{\beta}_{\text{Ridge}} = (X^\top X + \lambda I)^{-1} X^\top y$$

Setting $\nabla_\beta \mathcal{L} = 0$:

$$-2X^\top(y - X\beta) + 2\lambda\beta = 0 \implies (X^\top X + \lambda I)\beta = X^\top y$$

### Why Ridge Does Not Select Features

The gradient of the penalty at $\beta_j = 0$ is $2\lambda \cdot 0 = 0$. There is no force pushing $\beta_j$ exactly to zero from above and below simultaneously. Ridge contracts all coefficients toward zero but never reaches zero (unless $\lambda \to \infty$).

Geometrically: the L2 ball $\{\beta : \|\beta\|_2^2 \leq t\}$ is a smooth sphere. The contours of the squared loss (ellipses) touch the sphere at a generic point — not at an axis-aligned corner.

### Selection Properties

- All features remain in the model for any finite $\lambda$
- Correlated features receive similar, shared coefficients (stable)
- Useful for prediction; not for selection
- Effective degrees of freedom: $\text{df}(\lambda) = \text{tr}[X(X^\top X + \lambda I)^{-1}X^\top]$

---

## 2. L1 Regularisation (Lasso): Sparsity from Geometry

### Loss Function

$$\mathcal{L}_{\text{Lasso}}(\beta) = \|y - X\beta\|_2^2 + \lambda \|\beta\|_1 = \|y - X\beta\|_2^2 + \lambda \sum_{j=1}^p |\beta_j|$$

### KKT Conditions Derivation

The L1 penalty is not differentiable at $\beta_j = 0$. We use the subdifferential. Define the subdifferential of $|\beta_j|$:

$$\partial|\beta_j| = \begin{cases} \{+1\} & \beta_j > 0 \\ [-1, +1] & \beta_j = 0 \\ \{-1\} & \beta_j < 0 \end{cases}$$

Stationarity (subgradient = 0) gives the KKT conditions:

$$-2x_j^\top(y - X\hat{\beta}) + \lambda s_j = 0 \quad \forall j$$

where $s_j \in \partial|\hat{\beta}_j|$. This yields:

$$\begin{cases} x_j^\top r = \frac{\lambda}{2} \cdot \text{sign}(\hat{\beta}_j) & \text{if } \hat{\beta}_j \neq 0 \\ |x_j^\top r| \leq \frac{\lambda}{2} & \text{if } \hat{\beta}_j = 0 \end{cases}$$

where $r = y - X\hat{\beta}$ is the residual vector.

**Reading the KKT conditions:**

- An active feature $j$ has inner product $|x_j^\top r| = \lambda/2$ — it is "maximally correlated" with the residual.
- An inactive feature $j$ has $|x_j^\top r| < \lambda/2$ — it is not correlated enough to "enter" the model.
- As $\lambda$ decreases, more features become active.

### Why Lasso Produces Sparsity

The L1 ball $\{\beta : \|\beta\|_1 \leq t\}$ is a diamond (in 2D) or cross-polytope (in $p$ dimensions). It has corners at the axes. The contours of the squared loss generically touch the ball at a corner — meaning one coordinate is zero.

In $p$ dimensions: the L1 ball has $2p$ corners, each with exactly $p-1$ coordinates equal to zero. The probability that the loss ellipse is tangent at a face (non-sparse point) is zero for generic data.

### Regularisation Path

As $\lambda$ decreases from $\lambda_{\max}$ to 0:
- At $\lambda_{\max} = 2\max_j |x_j^\top y|$: all coefficients are zero.
- Coefficients enter the model sequentially as their inner product with the residual exceeds the threshold.
- At $\lambda = 0$: Lasso recovers the ordinary least squares solution (if $p \leq n$).

---

## 3. ElasticNet: Combining L1 and L2

### Loss Function

$$\mathcal{L}_{\text{EN}}(\beta) = \|y - X\beta\|_2^2 + \lambda_1 \|\beta\|_1 + \lambda_2 \|\beta\|_2^2$$

Equivalently parameterised with mixing ratio $\alpha \in [0,1]$ and overall penalty $\lambda$:

$$\mathcal{L}_{\text{EN}}(\beta) = \|y - X\beta\|_2^2 + \lambda \left[\alpha \|\beta\|_1 + \frac{1-\alpha}{2} \|\beta\|_2^2\right]$$

- $\alpha = 1$: pure Lasso
- $\alpha = 0$: pure Ridge
- $\alpha \in (0,1)$: ElasticNet

### Selection Properties of ElasticNet

The L2 component stabilises the solution when features are correlated. The **grouping effect**: if features $x_j$ and $x_k$ are strongly correlated, ElasticNet tends to select both with similar coefficients, rather than arbitrarily selecting one (as Lasso does).

Theorem (Zou & Hastie 2005): For any two features $j, k$:

$$|\hat{\beta}_j - \hat{\beta}_k| \leq \frac{1}{\lambda_2} \|x_j - x_k\|_2 \|y\|_2$$

As $\lambda_2$ increases, correlated features receive increasingly similar coefficients.

### KKT Conditions for ElasticNet

$$\begin{cases} x_j^\top r = \lambda_1 \cdot \text{sign}(\hat{\beta}_j) + 2\lambda_2 \hat{\beta}_j & \hat{\beta}_j \neq 0 \\ |x_j^\top r| \leq \lambda_1 & \hat{\beta}_j = 0 \end{cases}$$

---

## 4. The LARS Algorithm: Full Path in One Pass

### Motivation

Computing the Lasso solution at a single $\lambda$ requires iterative coordinate descent. Computing solutions at $K$ values of $\lambda$ requires $K$ separate runs. The LARS algorithm (Efron et al. 2004) computes the complete piecewise-linear Lasso path in $O(p^3 + np^2)$ time — equivalent to a single OLS fit.

### The LARS Algorithm: Full Derivation

**Setup:**
- Current active set: $\mathcal{A}$ (features currently in the model)
- Current residual: $r = y - X_{\mathcal{A}}\hat{\beta}_{\mathcal{A}}$
- Current maximum correlation: $C = \max_j |x_j^\top r|$

**Algorithm:**

**Step 1: Initialisation**

Start with $\hat{\beta} = 0$, $r = y$, $\mathcal{A} = \emptyset$.

Compute correlations $c_j = x_j^\top r$ for all $j$.

The first feature to enter is $j^* = \argmax_j |c_j|$. Set $\mathcal{A} = \{j^*\}$, $C = |c_{j^*}|$.

**Step 2: Equiangular Direction**

With active set $\mathcal{A}$, compute the equiangular vector — the direction in feature space that makes equal angles with all active feature vectors:

$$s_j = \text{sign}(x_j^\top r), \quad j \in \mathcal{A}$$

$$X_{\mathcal{A}} = X[\mathcal{A}], \quad \tilde{X}_{\mathcal{A}} = X_{\mathcal{A}} \cdot \text{diag}(s_{\mathcal{A}})$$

$$G_{\mathcal{A}} = \tilde{X}_{\mathcal{A}}^\top \tilde{X}_{\mathcal{A}}, \quad \mathbf{1}_{\mathcal{A}} = \mathbf{1}_{|\mathcal{A}|}$$

Normalisation constant:

$$A_{\mathcal{A}} = (\mathbf{1}_{\mathcal{A}}^\top G_{\mathcal{A}}^{-1} \mathbf{1}_{\mathcal{A}})^{-1/2}$$

Equiangular direction (in coefficient space):

$$w_{\mathcal{A}} = A_{\mathcal{A}} G_{\mathcal{A}}^{-1} \mathbf{1}_{\mathcal{A}}$$

Equiangular vector (in response space):

$$u_{\mathcal{A}} = \tilde{X}_{\mathcal{A}} w_{\mathcal{A}}$$

The vector $u_{\mathcal{A}}$ satisfies $X_{\mathcal{A}}^\top u_{\mathcal{A}} = A_{\mathcal{A}} \cdot \mathbf{1}_{|\mathcal{A}|}$ — equal angles with all active features.

**Step 3: Step Length**

Move the coefficient vector along the equiangular direction by step $\gamma$:

$$\hat{\beta}_{\mathcal{A}}(\gamma) = \hat{\beta}_{\mathcal{A}} + \gamma \cdot s_{\mathcal{A}} \odot w_{\mathcal{A}}$$

The residual evolves as: $r(\gamma) = r - \gamma \cdot u_{\mathcal{A}}$

Compute the step at which an inactive feature $j \notin \mathcal{A}$ reaches the current maximum correlation:

$$a_j = x_j^\top u_{\mathcal{A}}$$

$$\gamma_j = \frac{C - c_j}{A_{\mathcal{A}} - a_j} \quad \text{or} \quad \gamma_j = \frac{C + c_j}{A_{\mathcal{A}} + a_j}$$

Take the minimum positive step: $\gamma = \min_{j \notin \mathcal{A}} \{\gamma_j > 0\}$

**Step 4: Update**

Move: $\hat{\beta} \leftarrow \hat{\beta} + \gamma \cdot s_{\mathcal{A}} \odot w_{\mathcal{A}}$, $r \leftarrow r - \gamma \cdot u_{\mathcal{A}}$

Add the new feature $j^*$ (that reached the maximum correlation) to $\mathcal{A}$.

**Lasso Modification:** At each step, also check if any active coefficient crosses zero. If so, remove it from $\mathcal{A}$ and continue. This produces the exact Lasso path.

**Step 5: Terminate**

When $|\mathcal{A}| = \min(n-1, p)$ or $\lambda = 0$.

### Complexity

- $p$ steps (one per feature entering the model)
- At each step: solve a $|\mathcal{A}| \times |\mathcal{A}|$ linear system — $O(|\mathcal{A}|^3)$ naively, $O(|\mathcal{A}|^2)$ with rank-1 updates
- Total: $O(p \cdot p^2) = O(p^3)$ for the Gram matrix manipulations plus $O(np)$ for initial correlations

---

## 5. Regularisation Path Plots: Interpreting Coefficient Trajectories

### What to Look For

A regularisation path plot shows $\hat{\beta}_j(\lambda)$ for each feature $j$ as $\lambda$ varies (typically plotted on log scale or as fraction of $\lambda_{\max}$).

**Key features to read:**

1. **Entry order**: Features that enter at larger $\lambda$ are more important. The first feature to enter has the highest marginal correlation with $y$ given the current residual.

2. **Slope after entry**: A steep slope indicates the feature absorbs a large proportion of the explained variance.

3. **Sign changes**: A coefficient crossing zero indicates the feature's role reverses — possible multicollinearity artifact.

4. **Plateaus**: A flat coefficient after entry indicates the feature competes with a correlated feature that entered later.

5. **Stability of selection**: If the boundary between selected and deselected is sharp as $\lambda$ varies, selection is stable. Fuzzy boundaries indicate correlated features interchangeable in the model.

### The Knot Structure

The Lasso path is piecewise linear — the path is a sequence of line segments joined at **knots**. Each knot corresponds to a $\lambda$ value where:
- A new feature enters the active set, or
- An active feature's coefficient hits zero (drops out)

The number of knots equals the total number of features (counting entries and exits separately).

---

## 6. Cross-Validated Regularisation Selection

### LassoCV

`LassoCV` searches over a grid of $\lambda$ values using $k$-fold cross-validation. For each fold:

1. Fit the Lasso path on the training fold
2. Compute prediction error on the validation fold
3. Average across folds

Select $\hat{\lambda} = \argmin_\lambda \overline{\text{CV-MSE}}(\lambda)$.

**The one-standard-error rule:** Select the largest $\lambda$ whose CV error is within one standard error of the minimum. This favours sparser models when they perform comparably to the minimum-error model.

$$\hat{\lambda}_{1SE} = \max\{\lambda : \text{CV-MSE}(\lambda) \leq \text{CV-MSE}(\hat{\lambda}) + \text{SE}(\hat{\lambda})\}$$

### ElasticNetCV

Adds an outer loop over $\alpha$ (the L1/L2 mixing ratio). For each $\alpha$:
- Run LassoCV to find optimal $\lambda$ given that $\alpha$
- Record minimum CV error

Select the $(\alpha^*, \lambda^*)$ pair with the lowest CV error overall.

### Implementation Pattern

```python
from sklearn.linear_model import LassoCV, ElasticNetCV
import numpy as np

# LassoCV with 10-fold CV
lasso_cv = LassoCV(
    cv=10,
    n_alphas=100,       # 100 lambda values on log scale
    max_iter=10000,
    random_state=42
)
lasso_cv.fit(X_train, y_train)

optimal_lambda = lasso_cv.alpha_
selected_features = np.where(lasso_cv.coef_ != 0)[0]
print(f"Optimal lambda: {optimal_lambda:.4f}")
print(f"Selected features: {len(selected_features)}")

# ElasticNetCV with alpha grid
enet_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
    cv=10,
    n_alphas=100,
    max_iter=10000,
    random_state=42
)
enet_cv.fit(X_train, y_train)
print(f"Optimal l1_ratio: {enet_cv.l1_ratio_:.2f}")
print(f"Optimal alpha: {enet_cv.alpha_:.4f}")
```

---

## 7. When Lasso Fails: Correlated Features

### The Irrepresentable Condition

Lasso consistently selects the true support $S$ if and only if the **irrepresentable condition** holds:

$$\|X_{\bar{S}}^\top X_S (X_S^\top X_S)^{-1} \text{sign}(\beta_S)\|_\infty < 1$$

where $S$ is the true support and $\bar{S}$ is its complement.

This condition fails when $X_{\bar{S}}$ is highly correlated with $X_S$ — the inactive features are too correlated with active features for Lasso to distinguish them.

### Consequences of Failure

1. **Arbitrary selection:** Among a group of highly correlated features, Lasso arbitrarily selects one (or a few), discarding the rest. The selected feature may not be the causally relevant one.

2. **Coefficient instability:** Small perturbations in the data change which correlated feature is selected.

3. **False negatives:** True features in a correlated group may be excluded if a correlated impostor absorbs the variance first.

### When to Use ElasticNet Instead

- Features known to form groups (gene pathways, financial factor clusters)
- High pairwise correlations ($|r_{jk}| > 0.7$) between candidate features
- Prediction performance is the goal (not necessarily the exact support)
- $n < p$ settings where Lasso can select at most $n$ features

### Grouped Selection Needed

When the goal is to select entire groups of features together, Lasso and ElasticNet are insufficient. Group Lasso (covered in Guide 02) extends the penalty to enforce group-level sparsity.

---

## Common Pitfalls

- **Not standardising features:** The L1/L2 penalties are scale-dependent. Always standardise (zero mean, unit variance) before fitting.
- **Using Lasso for inference:** Lasso-selected features cannot be used for classical hypothesis testing without correction (see post-selection inference literature).
- **Treating the optimal $\lambda$ as the true $\lambda$:** CV selects $\lambda$ for prediction, not necessarily for correct support recovery.
- **Ignoring the path:** Looking only at the CV-optimal solution misses information about which features are robustly selected across $\lambda$ values.

---

## Connections

- **Builds on:** Linear regression, convex optimisation, KKT conditions (mathematical prerequisites); Module 01 and 02 (filter-based relevance scores motivate why embedded methods are an improvement — they optimise directly for the downstream model); Module 03 wrapper methods (embedded methods trade the wrapper's explicit CV loop for an implicit regularisation path)
- **Leads to:** Stability selection (Guide 02), Group Lasso (Guide 02), Knockoff filter (Guide 03)
- **Related to:** Coordinate descent, proximal gradient methods, ADMM

---

## Further Reading

- Tibshirani (1996) "Regression Shrinkage and Selection via the Lasso" — the original Lasso paper
- Efron et al. (2004) "Least Angle Regression" — LARS derivation and connection to Lasso
- Zou & Hastie (2005) "Regularization and Variable Selection via the Elastic Net" — ElasticNet with grouping effect
- Bühlmann & van de Geer (2011) "Statistics for High-Dimensional Data" — theory of Lasso selection consistency
- Zhao & Yu (2006) "On Model Selection Consistency of Lasso" — irrepresentable condition
