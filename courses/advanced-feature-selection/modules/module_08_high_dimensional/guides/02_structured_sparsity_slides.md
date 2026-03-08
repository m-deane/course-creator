---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Structured Sparsity
## Random Projections, Sparse PCA, and Graph-Guided Penalties

Module 8 · Advanced Feature Selection

<!-- Speaker notes: This deck covers methods that go beyond element-wise sparsity to exploit structure in the feature space. The key insight is that domain knowledge about feature relationships — groups, graphs, hierarchies — can be encoded directly into the penalty function, leading to more interpretable and often more accurate feature selection. We cover three major approaches: random projections for dimensionality reduction, Sparse PCA for unsupervised structure discovery, and graph-guided penalties for incorporating known feature relationships. -->

---

## Why Structure Matters

**Standard Lasso treats all features as independent:**
- No preference for selecting $X_1$ and $X_2$ together
- Selection results may be unstable under correlated features
- Ignores domain knowledge about feature relationships

**Structured sparsity encodes relationships:**

```
Finance: [Energy sector] [Tech sector] [Commodities]
              ↓                ↓              ↓
         [Oil] [Gas]    [Hardware][SW]  [Gold][Wheat]
              ↓                ↓              ↓
         Individual stocks/futures/options
```

Group penalties select entire branches — scientifically meaningful units.

<!-- Speaker notes: The motivating observation is simple: in most real applications, we have prior knowledge about which features belong together. Genes cluster into pathways. Stocks cluster into sectors. Technical indicators cluster into families — momentum, mean-reversion, volatility. The Lasso ignores all of this. Structured sparsity methods say: if we select one gene from a pathway, we probably care about the whole pathway, so let us apply selection at the pathway level rather than the gene level. This leads to more interpretable and often more stable solutions. -->

---

## Random Projections: The Johnson-Lindenstrauss Lemma

**Lemma (JL, 1984):** For $m$ points in $\mathbb{R}^p$ and distortion $\varepsilon \in (0,1)$, a random map:

$$f: \mathbb{R}^p \to \mathbb{R}^k, \quad k = O\!\left(\frac{\log m}{\varepsilon^2}\right)$$

preserves all pairwise distances within factor $(1 \pm \varepsilon)$:

$$(1-\varepsilon)\|\mathbf{z}_i - \mathbf{z}_j\|^2 \leq \|f(\mathbf{z}_i) - f(\mathbf{z}_j)\|^2 \leq (1+\varepsilon)\|\mathbf{z}_i - \mathbf{z}_j\|^2$$

**For $n$ observations and 10% distortion:** $k = O(\log n / 0.01)$

| $n$ | 100 | 500 | 1000 | 10000 |
|-----|-----|-----|------|-------|
| $k$ | 460 | 621 | 690 | 921 |

<!-- Speaker notes: The Johnson-Lindenstrauss lemma is one of the most surprising results in high-dimensional geometry. It says that any set of m points in arbitrarily high dimensions can be projected into k = O(log m) dimensions while preserving all pairwise distances. The projection is random — typically a Gaussian matrix — and works with high probability. For feature selection, this means we can run selection algorithms in the projected k-dimensional space and expect the results to approximately reflect structure in the original p-dimensional space. -->

---

## Random Projection for Feature Selection

**Random Lasso (Meinshausen & Bühlmann, 2010):**

```python
def random_lasso(X, y, B=100, k=None):
    n, p = X.shape
    k = k or max(10, int(np.log(n)))
    scores = np.zeros(p)

    for b in range(B):
        # Random Gaussian projection matrix
        Phi = np.random.randn(k, p) / np.sqrt(k)
        Z = X @ Phi.T          # n × k projected design

        # Lasso on projected problem
        lasso = LassoCV(cv=3).fit(Z, y)
        gamma = lasso.coef_   # k-vector

        # Back-project importance to original features
        scores += np.abs(Phi.T @ gamma)  # p-vector

    return scores / B
```

Features with highest average back-projected importance are selected.

<!-- Speaker notes: The Random Lasso is elegant: instead of one Lasso on p features, we run B Lasso problems on k-dimensional projections. Each projection is different, so different features dominate in different projections. Averaging the back-projected importances gives a stable, ensemble-style feature importance score. The key parameter is k — it controls the information retained in each projection. Setting k to log(n) is conservative but safe. For n=200 that means k ≈ 5, so each Lasso is on a 5-feature problem — trivially fast. -->

---

## Sparse PCA: The Setup

**Standard PCA loading problem:**

$$\mathbf{v}_1 = \arg\max_{\|\mathbf{v}\|=1} \mathbf{v}^\top \hat{\boldsymbol{\Sigma}} \mathbf{v}$$

All $p$ features have non-zero loadings — uninterpretable for large $p$.

**Sparse PCA (Zou, Hastie & Tibshirani, 2006):**

$$\min_{\mathbf{A}, \mathbf{B}} \|\mathbf{X} - \mathbf{X}\mathbf{B}\mathbf{A}^\top\|_F^2 + \lambda \sum_{j=1}^k \|\mathbf{b}_j\|_1$$

subject to $\mathbf{A}^\top \mathbf{A} = \mathbf{I}_k$

- $\mathbf{B} \in \mathbb{R}^{p \times k}$: **sparse loadings** — most entries zero
- $\mathbf{A} \in \mathbb{R}^{n \times k}$: **score vectors** — orthonormal

<!-- Speaker notes: The key idea of Sparse PCA is to replace the dense PCA loadings with sparse ones by adding an L1 penalty. This is a matrix factorisation problem with two sets of variables — the scores A and the loadings B. The constraint that A is orthonormal is what makes it a PCA rather than just any low-rank factorisation. The L1 penalty on the columns of B forces each principal component to be explained by a small subset of features, making the components interpretable. -->

---

## SPCA Algorithm: Alternating Regression

**Alternating minimisation (block coordinate descent):**

**Step A — Fix $\mathbf{B}$, update $\mathbf{A}$:**
$$\mathbf{X}\mathbf{B} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top \implies \mathbf{A}^* = \mathbf{U}\mathbf{V}^\top$$
(SVD of $\mathbf{X}\mathbf{B}$, take left × right singular vectors)

**Step B — Fix $\mathbf{A}$, update $\mathbf{B}$ column by column:**
$$\mathbf{b}_j^* = \arg\min_{\mathbf{b}} \|\mathbf{X}^\top \mathbf{a}_j - \mathbf{b}\|^2 + \lambda_j \|\mathbf{b}\|_1$$

This is a **Lasso** with response $\mathbf{X}^\top \mathbf{a}_j$ — solved instantly by soft-thresholding.

**Convergence:** Objective is non-increasing at each step, bounded below $\Rightarrow$ guaranteed convergence.

<!-- Speaker notes: The SPCA algorithm is a classical block coordinate descent scheme. Step A is a standard orthogonal Procrustes problem — given B, find the best orthonormal A. The solution is the product of left and right singular vectors of XB. Step B is k independent Lasso problems — one per component. Each Lasso has response X^T a_j and design matrix the identity, so it reduces to simple soft-thresholding: b_j = sign(X^T a_j) * max(|X^T a_j| - lambda, 0). The algorithm is fast and reliable. -->

---

## Feature Selection from Sparse PCA Loadings

After fitting SPCA with $k$ components and loading matrix $\mathbf{B}^* \in \mathbb{R}^{p \times k}$:

<div class="columns">

**Union selection:**
Select $j$ if $B^*_{jl} \neq 0$ for any $l$.
Maximises recall of important features.

**Importance-weighted:**
Rank by $\sum_l (B^*_{jl})^2 \cdot \sigma_l^2$
where $\sigma_l^2$ is variance of component $l$.

</div>

**Visualising the sparsity pattern:**

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(12, 4))
ax.imshow(spca_loadings.T != 0, cmap='Blues', aspect='auto')
ax.set_xlabel('Features (p=5000)')
ax.set_ylabel('Sparse PC')
ax.set_title('Non-zero loading pattern across components')
```

White = zero loading. Blue = active feature.

<!-- Speaker notes: The sparsity pattern visualisation is one of the most useful diagnostics for SPCA. For a dataset with known group structure — say genomic pathways — you expect to see the non-zero entries clustered in blocks corresponding to pathways. If the pattern is scattered randomly, the sparsity parameter lambda may need tuning. The importance-weighted ranking is better than simple union selection when the components explain very different amounts of variance — you want to up-weight features that are active in high-variance components. -->

---

## Sparse Autoencoders: Selection-Adjacent Methods

**Architecture:**
$$\mathbf{x} \xrightarrow{\mathbf{W}_1, \text{ReLU}} \mathbf{h} \xrightarrow{\mathbf{W}_2} \hat{\mathbf{x}}$$

**Sparsity loss (KL divergence from target sparsity $\rho$):**

$$\mathcal{L} = \underbrace{\|\mathbf{x} - \hat{\mathbf{x}}\|^2}_{\text{reconstruction}} + \beta \sum_{j=1}^k \underbrace{\text{KL}\!\left(\rho \,\|\, \hat{\rho}_j\right)}_{\text{sparsity}}$$

where $\hat{\rho}_j = \frac{1}{n}\sum_i h_{ij}$ (mean activation of unit $j$).

**Feature importance from first-layer weights:**
$$\text{importance}_j = \left\|\mathbf{W}_1[:, j]\right\|_2 \quad \text{(column norm)}$$

Features with near-zero column norms in $\mathbf{W}_1$ are unimportant.

<!-- Speaker notes: Sparse autoencoders are technically reconstruction methods, not feature selectors, but their first-layer weights reveal which input features the network finds informative. The KL divergence term forces the hidden units to be mostly inactive for any given input — only a few units fire for each example. This means the encoder has learned a sparse code. The column norm of the first weight matrix measures how much the encoder relies on each input feature to construct this sparse code. Features with near-zero norms are those the encoder can ignore. -->

---

## Graph-Guided Penalties: Encoding Feature Relationships

**Setup:** Feature graph $G = (V, E)$ with edge weights $w_{jk}$.

**Graph Lasso penalty (Grace, Li & Li, 2008):**

$$\mathcal{L}_{\text{Grace}} = \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda_1 \|\boldsymbol{\beta}\|_1 + \frac{\lambda_2}{2} \sum_{(j,k) \in E} w_{jk}(\beta_j - \beta_k)^2$$

The graph Laplacian form: $\frac{\lambda_2}{2} \boldsymbol{\beta}^\top \mathbf{L} \boldsymbol{\beta}$

**Effect:** If $(j,k) \in E$ and $\beta_j \neq 0$, the penalty encourages $\beta_k \neq 0$.

Connected features are selected together.

<!-- Speaker notes: The Grace penalty is the most intuitive graph-guided method. The third term is the graph Laplacian quadratic form, which penalises differences between coefficients of adjacent features. If two genes are connected in a regulatory network and gene j is relevant to the outcome, the penalty encourages gene k to also be included. This makes sense biologically — if one gene in a regulatory pathway matters, the genes it regulates probably also matter. The key parameters are lambda1 controlling sparsity and lambda2 controlling graph smoothness. -->

---

## Group Lasso: Group-Level Selection

**Setting:** Features partitioned into $L$ groups $G_1, \ldots, G_L$ (disjoint).

**Penalty (Yuan & Lin, 2006):**

$$\mathcal{L}_{\text{GL}} = \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \sum_{l=1}^L \lambda_l \sqrt{|G_l|} \|\boldsymbol{\beta}_{G_l}\|_2$$

**Selection property:** $\|\boldsymbol{\beta}_{G_l}\|_2 = 0$ or $\|\boldsymbol{\beta}_{G_l}\|_2 > 0$ — entire group in or out.

**Example — sector-based selection:**
```
Groups: Energy(50), Tech(80), Materials(30), Financials(60)
Group Lasso selects entire sectors, not individual stocks
```

The $\sqrt{|G_l|}$ scaling ensures equal penalisation per group.

<!-- Speaker notes: Group Lasso is the most widely implemented structured penalty. The key property is group-level sparsity: the L2 norm of each group's coefficient vector is either exactly zero or strictly positive. This creates an all-or-nothing selection at the group level. The square root scaling by group size is critical — without it, larger groups are penalised more heavily and you get size-biased selection. In financial applications, groups often correspond to sectors, asset classes, or factor families. -->

---

## Overlapping Groups: Sparse Group Lasso

**Problem:** Features belonging to multiple groups (e.g., a stock in multiple indices).

**Sparse Group Lasso (Simon et al., 2013):**

$$\mathcal{L}_{\text{SGL}} = \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + (1-\alpha)\lambda \sum_l \sqrt{|G_l|} \|\boldsymbol{\beta}_{G_l}\|_2 + \alpha \lambda \|\boldsymbol{\beta}\|_1$$

**Mixing parameter $\alpha$:**

| $\alpha$ | Effect |
|----------|--------|
| 0 | Pure Group Lasso (group sparsity only) |
| 0.5 | Balanced group + individual sparsity |
| 1 | Pure Lasso (no group structure) |

Selects sparse subsets *within* selected groups.

<!-- Speaker notes: The Sparse Group Lasso combines the L1 and group L2 penalties with a mixing parameter. This is analogous to elastic net mixing Lasso and ridge, but at the group level. The alpha parameter controls whether you care more about group-level or individual-level sparsity. For genomics where you expect only a few genes from each relevant pathway to actually matter, alpha around 0.5 works well. For finance where you expect entire factor groups to contribute or not, alpha closer to 0 is better. The sgl package in R and the group-lasso package in Python implement this efficiently. -->

---

## Tree-Guided Sparsity: Hierarchical Selection

**Structure:** Features arranged in a tree $T$ — coarse to fine.

```
Root (all features)
├── Macro (GDP, rates, FX)
│   ├── Rates (2Y, 5Y, 10Y, 30Y)
│   └── FX (EURUSD, GBPUSD, USDJPY)
└── Commodity
    ├── Energy (WTI, Brent, NatGas)
    └── Metals (Gold, Silver, Copper)
```

**Hierarchical Lasso (Zhao et al., 2009):** Each node $g \in T$ has penalty:

$$\sum_{g \in T} \lambda_g \|\boldsymbol{\beta}_g\|_2$$

**Selection rule:** Selecting leaf $j$ forces all ancestors selected. Parents have larger $\lambda_g$ — coarse selection is more heavily penalised.

<!-- Speaker notes: Tree-guided sparsity is particularly natural for commodity trading data, where the hierarchy from asset class to sector to individual instrument is well-defined. The selection propagates upward: to select a leaf feature (individual WTI contract), the algorithm must overcome the penalty at every ancestor node (Energy sector, Commodity class, Root). This means the algorithm prefers selecting broad groups first and then drilling down to specific instruments only if their individual contribution justifies the additional penalty. -->

---

## Practical Guide: Choosing a Structured Penalty

```
Do features have known groups? ──No──→ Standard Lasso / Elastic Net
         │
         Yes
         │
Are groups disjoint? ──No──→ Latent Group Lasso / Sparse Group Lasso
         │
         Yes
         │
Is hierarchical structure present? ──Yes──→ Hierarchical Lasso
         │
         No
         │
Do features form a graph? ──Yes──→ Grace / Network-constrained
         │
         No
         │
Are groups homogeneous? ──Yes──→ Group Lasso
         │
         No
         └───────────────────→ Sparse Group Lasso (alpha ~ 0.5)
```

<!-- Speaker notes: This decision tree gives a practical starting point. In most applications, the group structure is known from domain expertise — sector classifications, pathway databases, factor taxonomies. The key question is whether the groups overlap. If they don't overlap, Group Lasso or Hierarchical Lasso is appropriate depending on whether a tree structure is present. If they do overlap, Sparse Group Lasso handles this without duplication of features. Graph penalties (Grace) are most powerful when the graph encodes genuine dependency structure, not just similarity. -->

---

## Implementation: Group Lasso in Practice

```python
from group_lasso import GroupLasso
import numpy as np

# Define group membership for p=20 features in 4 groups of 5
groups = np.array([0]*5 + [1]*5 + [2]*5 + [3]*5)

# Fit Group Lasso
gl = GroupLasso(
    groups=groups,
    group_reg=0.05,    # lambda for group L2 penalty
    l1_reg=0.01,       # optional L1 for within-group sparsity
    standardize=True,
    fit_intercept=True
)
gl.fit(X_train, y_train)

# Inspect group-level selection
for g in range(4):
    group_mask = groups == g
    group_norm = np.linalg.norm(gl.coef_[group_mask])
    selected = "SELECTED" if group_norm > 1e-6 else "zeroed"
    print(f"Group {g}: norm={group_norm:.4f} → {selected}")
```

<!-- Speaker notes: This code uses the group-lasso Python package which implements both Group Lasso and Sparse Group Lasso with efficient proximal gradient optimisation. The groups array specifies which group each feature belongs to. The group_reg parameter is lambda for the group L2 penalty. Setting l1_reg > 0 activates the Sparse Group Lasso variant. The key diagnostic after fitting is checking the L2 norm of each group's coefficients — groups with norm below 1e-6 are effectively zeroed out. -->

---

## Sparse PCA vs Group Lasso: When to Use Which

| Question | Sparse PCA | Group Lasso |
|----------|-----------|-------------|
| Is the outcome $y$ used? | No (unsupervised) | Yes (supervised) |
| Goal | Discover feature structure | Predict $y$ with groups |
| Primary output | Sparse loading matrix | Sparse coefficient vector |
| When to use | Exploratory analysis, preprocessing | Prediction with known groups |
| Typical $k$ | 5–20 components | Determined by group structure |

**Combined workflow:**
1. Sparse PCA on $\mathbf{X}$ → identify latent feature groups
2. Use groups from SPCA as input to Group Lasso for prediction

<!-- Speaker notes: This is a common confusion point. Sparse PCA is fundamentally unsupervised — it finds structure in X without looking at y. Group Lasso is supervised — it uses y to select groups. The combined workflow is powerful: use Sparse PCA to discover the latent group structure from unlabelled data (e.g., all historical price data), then use those groups as input to Group Lasso fitted on the labelled outcome (e.g., future returns). This is a natural two-stage approach that leverages both supervised and unsupervised information. -->

---

## Summary: Structured Sparsity Methods

1. **Random projections** (JL lemma): Reduce $p$ to $k = O(\log n)$ while preserving distances — enables fast selection on projected problems

2. **Sparse PCA** (Zou et al., 2006): Unsupervised sparse loadings reveal which features drive variance — useful for exploratory analysis and group discovery

3. **Sparse autoencoders**: First-layer weight norms provide feature importance for deep feature sets

4. **Graph penalties** (Grace): Encode feature graph into Laplacian penalty — connected features selected together

5. **Group Lasso** (Yuan & Lin, 2006): Group-level sparsity for disjoint groups

6. **Sparse Group Lasso** (Simon et al., 2013): Combined group + individual sparsity for within-group heterogeneity

<!-- Speaker notes: The key insight tying all these methods together is that structure in the feature space — whether it comes from a known graph, a known grouping, or a latent structure discovered by Sparse PCA — can be encoded into the selection objective to produce more interpretable and often more stable results. The Lasso is powerful but blind to structure. Structured penalties are the right tool when domain knowledge provides structure. -->

---

<!-- _class: lead -->

## Next: Post-Selection Inference

After screening and structured selection, the critical question is:

**"How do we make valid confidence intervals after we have selected features?"**

Naively reusing the same data for selection and inference gives confidence intervals with wrong coverage.

Guide 03 covers: Debiased Lasso, selective inference, and data splitting.

<!-- Speaker notes: This is the bridge slide to Guide 03. The problem is fundamental: if we use the data to select features and then use the same data to estimate their effects, our confidence intervals are anti-conservative — they do not contain the true value with the stated probability. This is the post-selection inference problem, and it has been the subject of intense methodological development since 2009. Guide 03 covers the three main approaches: the Debiased Lasso which corrects the Lasso estimator, selective inference which conditions on the selection event, and data splitting which avoids the problem by using held-out data for inference. -->
