# Structured Sparsity: Random Projections, Sparse PCA, and Graph-Guided Penalties

## In Brief

Structured sparsity methods exploit known relationships among features — groups, graphs, hierarchies — to impose selection patterns that go beyond element-wise sparsity. When domain knowledge encodes which features belong together, structured penalties outperform the Lasso by selecting interpretable, coherent subsets rather than arbitrary sparse solutions.

## Key Insight

The Lasso treats every feature independently: it has no preference for selecting $X_1$ and $X_2$ together just because they measure related phenomena. Group Lasso and graph-guided penalties encode this preference explicitly. The result is selection aligned with domain structure — features selected in interpretable, scientifically meaningful groups.

---

## 1. Random Projection Methods

### 1.1 Motivation: Dimension Reduction Before Selection

When $p$ is very large, even computing pairwise feature correlations ($O(p^2 n)$) is expensive. Random projections reduce $p$ to a low-dimensional space $k \ll p$ in $O(p k n)$ time while approximately preserving pairwise distances.

### 1.2 Johnson-Lindenstrauss Lemma

**Lemma (Johnson & Lindenstrauss, 1984):** For any set of $m$ points $\{\mathbf{z}_1, \ldots, \mathbf{z}_m\} \subset \mathbb{R}^p$ and $\varepsilon \in (0, 1)$, there exists a map $f: \mathbb{R}^p \to \mathbb{R}^k$ with:

$$k = O\!\left(\frac{\log m}{\varepsilon^2}\right)$$

such that for all $i, j$:

$$(1-\varepsilon) \|\mathbf{z}_i - \mathbf{z}_j\|^2 \leq \|f(\mathbf{z}_i) - f(\mathbf{z}_j)\|^2 \leq (1+\varepsilon) \|\mathbf{z}_i - \mathbf{z}_j\|^2$$

The map $f$ can be a random Gaussian matrix $\mathbf{\Phi} \in \mathbb{R}^{k \times p}$ with entries $\Phi_{ij} \sim \mathcal{N}(0, 1/k)$.

**For feature selection:** If $n$ observations are mapped into $k = O(\log n / \varepsilon^2)$ dimensions, their pairwise distances are approximately preserved. A selection method operating in the $k$-dimensional space can then be used to identify which of the original $p$ features contributed most to the projection.

### 1.3 Selection on Projections

**Algorithm (Meinshausen & Bühlmann, 2010 — Random Lasso):**

```
For b = 1, ..., B bootstrap iterations:
    1. Draw random projection: Phi_b ~ N(0, 1/k)^{k x p}
    2. Compute projected design: Z_b = X @ Phi_b.T  (n x k)
    3. Run Lasso on Z_b to get gamma_b (k-vector)
    4. Back-project: score_b = |Phi_b.T @ gamma_b|  (p-vector)

Stability score: omega_j = mean over b of score_b[j]
Select: features with omega_j > threshold
```

The random projection diversifies which features contribute to each Lasso solve, and aggregation over projections provides a stability-weighted feature importance.

### 1.4 Sparse Random Projections

Achlioptas (2003) showed that $\Phi_{ij}$ can be drawn from $\{+1, 0, -1\}$ with probabilities $\{1/6, 2/3, 1/6\}$ (scaled by $\sqrt{3}$) while preserving the JL guarantee. This yields a factor of 3× speedup in projection computation and is implemented in `sklearn.random_projection.SparseRandomProjection`.

---

## 2. Sparse PCA for Feature Identification

### 2.1 Standard PCA and Its Feature Selection Problem

Standard PCA finds principal components $\mathbf{v}_1, \ldots, \mathbf{v}_k$ maximising explained variance:

$$\mathbf{v}_1 = \arg\max_{\|\mathbf{v}\|=1} \mathbf{v}^\top \hat{\boldsymbol{\Sigma}} \mathbf{v}$$

where $\hat{\boldsymbol{\Sigma}} = \mathbf{X}^\top \mathbf{X} / n$. The loadings $v_{1j}$ are non-zero for all $j = 1, \ldots, p$ — every feature contributes to every component. This is interpretable for $p = 10$ but not for $p = 5{,}000$.

### 2.2 Sparse PCA Formulation

Sparse PCA (Zou, Hastie & Tibshirani, 2006) adds an $\ell_1$ penalty to the loading vectors:

$$(\mathbf{A}^*, \mathbf{B}^*) = \arg\min_{\mathbf{A}, \mathbf{B}} \|\mathbf{X} - \mathbf{X}\mathbf{B}\mathbf{A}^\top\|_F^2 + \lambda \sum_{j=1}^k \|\mathbf{b}_j\|_1$$

subject to $\mathbf{A}^\top \mathbf{A} = \mathbf{I}_k$ (orthonormality of score vectors), where $\mathbf{B} \in \mathbb{R}^{p \times k}$ contains the sparse loadings and $\mathbf{A} \in \mathbb{R}^{n \times k}$ contains the scores.

The $\ell_1$ penalty drives loading vectors $\mathbf{b}_j$ to be sparse — each component is explained by a small subset of features.

### 2.3 The SPCA Algorithm (Zou, Hastie & Tibshirani, 2006)

The SPCA algorithm alternates between two steps:

**Step A (Fix B, optimise A):** With loadings $\mathbf{B}$ fixed, the optimal score matrix is the left singular vectors of $\mathbf{X}\mathbf{B}$:
$$\mathbf{A} = \mathbf{U}\mathbf{V}^\top \quad \text{where} \quad \mathbf{X}\mathbf{B} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$$

**Step B (Fix A, optimise B):** With scores $\mathbf{A}$ fixed, each loading $\mathbf{b}_j$ is found by solving a Lasso:
$$\mathbf{b}_j = \arg\min_{\mathbf{b}} \|\mathbf{X}^\top \mathbf{a}_j - \mathbf{b}\|^2 + \lambda_j \|\mathbf{b}\|_1$$

where $\mathbf{a}_j$ is the $j$-th column of $\mathbf{A}$. This is a standard Lasso problem solvable in closed form via soft-thresholding.

**Convergence:** The objective is non-increasing at each step and bounded below, so the algorithm converges.

### 2.4 GPower Algorithm

An alternative to SPCA, GPower (Journ\'ee et al., 2010) uses a gradient-based approach on the non-convex formulation:

$$\max_{\|\mathbf{v}\|=1} \mathbf{v}^\top \hat{\boldsymbol{\Sigma}} \mathbf{v} - \lambda \|\mathbf{v}\|_1$$

**GPower update:** At each iteration, apply a gradient step followed by sparse projection:
$$\mathbf{v}^{(t+1)} = \text{SparseProject}_k\!\left(\hat{\boldsymbol{\Sigma}} \mathbf{v}^{(t)}\right)$$

where $\text{SparseProject}_k$ retains the $k$ largest entries in absolute value and re-normalises.

GPower is faster than SPCA for very large $p$ since it avoids the SVD step.

### 2.5 Feature Selection from Sparse PCA Loadings

Once sparse loadings $\mathbf{B}^*$ are computed:

1. **Union selection:** Select any feature $j$ with $B^*_{jl} \neq 0$ for at least one component $l$.
2. **Threshold selection:** Select features with $\max_l |B^*_{jl}| > \tau$.
3. **Importance-weighted selection:** Rank features by $\sum_l (B^*_{jl})^2 \cdot \sigma_l^2$ where $\sigma_l^2$ is the variance explained by component $l$.

**Advantage over Lasso:** Sparse PCA is unsupervised — it identifies features that explain variance in $\mathbf{X}$ regardless of the outcome $y$. This is valuable for exploratory analysis and for discovering latent structure in the feature space.

---

## 3. Sparse Autoencoders as Selection-Adjacent Methods

### 3.1 Autoencoder Architecture

An autoencoder learns a low-dimensional representation of $\mathbf{X}$ through an encoder-decoder pair:

$$\text{Encoder: } \mathbf{h} = f_\theta(\mathbf{x}) \in \mathbb{R}^k, \quad k \ll p$$
$$\text{Decoder: } \hat{\mathbf{x}} = g_\phi(\mathbf{h}) \in \mathbb{R}^p$$

Minimising reconstruction loss $\|\mathbf{x} - \hat{\mathbf{x}}\|^2$ forces the bottleneck $\mathbf{h}$ to preserve the most informative structure of $\mathbf{x}$.

### 3.2 Enforcing Sparsity in the Bottleneck

A **sparse autoencoder** adds a sparsity penalty to the hidden activations:

$$\mathcal{L} = \sum_i \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|^2 + \beta \sum_j \text{KL}(\rho \| \hat{\rho}_j)$$

where $\hat{\rho}_j = \frac{1}{n}\sum_i h_{ij}$ is the mean activation of hidden unit $j$, and $\rho$ is the target sparsity level (e.g., 0.05).

The KL term pushes most hidden units to be near-zero for any given input — only a few activate.

### 3.3 Connection to Feature Selection

Sparse autoencoders are not direct feature selectors but serve as selection-adjacent tools:

- **First-layer weights** $\mathbf{W}_1 \in \mathbb{R}^{k \times p}$ reveal which input features each hidden unit attends to
- **Activation patterns** identify which features drive the representation
- **Gradient-based attribution**: Compute $\partial h_j / \partial x_i$ to attribute hidden activation to input features

For a strict feature selection interpretation, apply $\ell_1$ regularisation to the first layer weights directly:

$$\mathcal{L} = \|\mathbf{X} - \hat{\mathbf{X}}\|_F^2 + \lambda \|\mathbf{W}_1\|_1$$

Features with near-zero first-layer weight norms across all hidden units are identified as unimportant.

---

## 4. Graph-Guided Penalties

### 4.1 Motivation: Feature Relationship Graphs

Domain knowledge often specifies which features are related. Examples:
- **Genomics**: Gene regulatory networks — edges indicate interaction
- **Finance**: Sector membership — stocks in same sector behave similarly
- **Neuroscience**: Brain connectivity — anatomically connected regions co-activate

Graph-guided penalties incorporate this graph $G = (V, E)$ to encourage selecting connected subsets of features.

### 4.2 Graph Lasso Penalty (Grace)

The Graph-constrained Estimation (Grace, Li & Li, 2008) adds a graph-smoothing penalty:

$$\mathcal{L}_{\text{Grace}} = \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda_1 \|\boldsymbol{\beta}\|_1 + \frac{\lambda_2}{2} \sum_{(j,k) \in E} w_{jk}(\beta_j - \beta_k)^2$$

The third term penalises differences between coefficients of connected features: if $(j, k) \in E$ and $\beta_j \neq 0$, the penalty encourages $\beta_k \approx \beta_j$ (and thus $\beta_k \neq 0$).

**Matrix form:** The penalty term equals $\boldsymbol{\beta}^\top \mathbf{L} \boldsymbol{\beta}$ where $\mathbf{L}$ is the graph Laplacian:

$$L_{jk} = \begin{cases} \sum_l w_{jl} & j = k \\ -w_{jk} & (j,k) \in E \\ 0 & \text{otherwise} \end{cases}$$

### 4.3 Network-Constrained Regularisation

Chen et al. (2010) propose an alternative graph penalty that couples the $\ell_1$ term with the Laplacian:

$$\mathcal{L}_{\text{Net}} = \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda_1 \|\boldsymbol{\beta}\|_1 + \lambda_2 \boldsymbol{\beta}^\top \mathbf{L} \boldsymbol{\beta}$$

This is solved efficiently by augmenting the design matrix. Let $\tilde{\mathbf{X}} = \begin{pmatrix} \mathbf{X} \\ \sqrt{n\lambda_2} \mathbf{L}^{1/2} \end{pmatrix}$. Then:

$$\mathcal{L}_{\text{Net}} = \frac{1}{2\tilde{n}}\|\tilde{\mathbf{y}} - \tilde{\mathbf{X}}\boldsymbol{\beta}\|^2 + \lambda_1 \|\boldsymbol{\beta}\|_1$$

which is a standard Lasso problem on the augmented data $(\tilde{\mathbf{X}}, \tilde{\mathbf{y}})$.

---

## 5. Tree-Guided Sparsity

### 5.1 Hierarchical Feature Groups

Features often have natural hierarchy: broad groups (sectors) contain sub-groups (industries) containing individual features (stocks). Tree-guided sparsity enforces that selecting a fine-grained feature implies selecting its parent group.

**Tree structure:** Each feature $j$ belongs to a path from the root to a leaf in a tree $T$. Let $\mathcal{G}$ denote the set of all groups (nodes in $T$).

### 5.2 Hierarchical Lasso (Zhao et al., 2009)

$$\mathcal{L}_{\text{HLasso}} = \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \sum_{g \in \mathcal{G}} \lambda_g \|\boldsymbol{\beta}_g\|_2$$

where $\boldsymbol{\beta}_g$ is the vector of coefficients for features in group $g$, and the penalty weight $\lambda_g$ is larger for higher-level (coarser) groups.

**Selection property:** If group $g$ is not selected ($\|\boldsymbol{\beta}_g\|_2 = 0$), all descendant groups are also not selected. Selection propagates downward through the tree.

This is appropriate for commodity trading where broad asset class exposure ($\lambda$ large) is penalised more heavily than instrument-specific exposure ($\lambda$ small).

---

## 6. Overlapping Group Penalties

### 6.1 Group Lasso for Non-Overlapping Groups

Standard Group Lasso (Yuan & Lin, 2006) handles disjoint groups $\mathcal{G} = \{G_1, \ldots, G_L\}$ where $G_l \cap G_{l'} = \emptyset$:

$$\mathcal{L}_{\text{GL}} = \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \sum_{l=1}^L \lambda_l \|\boldsymbol{\beta}_{G_l}\|_2$$

**Selection property:** The $\ell_2$ norm on $\boldsymbol{\beta}_{G_l}$ is zero or non-zero for the entire group — the Group Lasso performs group-level selection. Either all features in $G_l$ are included or all are excluded.

### 6.2 Overlapping Groups: The Problem

When feature $j$ belongs to multiple groups (e.g., a gene participates in several pathways), non-overlapping Group Lasso is misspecified. Naively applying it requires duplicating feature $j$ in each group — which inflates $p$ and creates inconsistent coefficient estimates.

### 6.3 Latent Group Lasso (Jacob et al., 2009)

The Latent Group Lasso introduces a latent representation $\boldsymbol{\beta} = \sum_{l: j \in G_l} \mathbf{v}^{(l)}$ where $\mathbf{v}^{(l)}$ is the contribution of group $l$ to $\boldsymbol{\beta}$:

$$\mathcal{L}_{\text{LGL}} = \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda \sum_{l=1}^L \|\mathbf{v}^{(l)}_{G_l}\|_2$$

subject to $\boldsymbol{\beta}_j = \sum_{l: j \in G_l} v_j^{(l)}$ for all $j$.

This penalises each group's contribution separately, allowing feature $j$ to be selected through one group while being suppressed through another.

### 6.4 Sparse Group Lasso (Simon et al., 2013)

The Sparse Group Lasso combines element-wise sparsity ($\ell_1$) with group-level sparsity ($\ell_{2}$ per group):

$$\mathcal{L}_{\text{SGL}} = \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + (1-\alpha)\lambda \sum_{l=1}^L \sqrt{|G_l|} \|\boldsymbol{\beta}_{G_l}\|_2 + \alpha \lambda \|\boldsymbol{\beta}\|_1$$

The mixing parameter $\alpha \in [0, 1]$ controls the trade-off:
- $\alpha = 0$: Group Lasso (group-level selection only)
- $\alpha = 1$: Lasso (individual-level selection only)
- $\alpha = 0.5$: Equal weight on both

This is useful when groups overlap and within-group sparsity is expected — e.g., selecting a gene pathway does not imply selecting all genes in the pathway.

---

## Common Pitfalls

- **Confusing Sparse PCA with feature selection for prediction**: Sparse PCA finds features that explain variance in $\mathbf{X}$, not in $y$. Always follow it with a supervised step if prediction is the goal.
- **Misspecifying group boundaries in Group Lasso**: Incorrect group structure can actively harm performance relative to standard Lasso. Validate group assignments with domain experts before applying Group Lasso.
- **Ignoring the $\sqrt{|G_l|}$ scaling factor**: The Group Lasso penalty should scale with $\sqrt{|G_l|}$ to ensure equal penalisation per group regardless of group size. Many implementations omit this scaling.
- **Using graph penalties with an incorrect graph**: Grace and network-constrained penalties degrade performance when the graph is misspecified. If the graph is uncertain, consider ensemble approaches that aggregate over multiple graphs.

---

## Connections

- **Builds on:** Lasso and elastic net (Module 4), PCA (prerequisite), stability selection (Module 10)
- **Leads to:** Post-selection inference with structured models (Guide 03)
- **Related to:** Multi-task learning, transfer learning, Bayesian variable selection with hierarchical priors

---

## Further Reading

- **Zou, H., Hastie, T. & Tibshirani, R. (2006).** "Sparse Principal Component Analysis." *Journal of Computational and Graphical Statistics*, 15(2), 265–286. — SPCA derivation and the elastic net connection.
- **Yuan, M. & Lin, Y. (2006).** "Model selection and estimation in regression with grouped variables." *Journal of the Royal Statistical Society: Series B*, 68(1), 49–67. — The foundational Group Lasso paper.
- **Jacob, L., Obozinski, G. & Vert, J.-P. (2009).** "Group Lasso with overlap and graph Lasso." *ICML 2009*, 433–440. — Latent Group Lasso for overlapping groups.
- **Simon, N., Friedman, J., Hastie, T. & Tibshirani, R. (2013).** "A Sparse-Group Lasso." *Journal of Computational and Graphical Statistics*, 22(2), 231–245. — SGL with efficient implementation.
- **Li, C. & Li, H. (2008).** "Network-constrained regularization and variable selection for analysis of genomic data." *Bioinformatics*, 24(9), 1175–1182. — Grace penalty derivation.
- **Johnson, W.B. & Lindenstrauss, J. (1984).** "Extensions of Lipschitz maps into a Hilbert space." *Contemporary Mathematics*, 26, 189–206. — The original JL lemma.
