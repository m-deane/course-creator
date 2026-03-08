---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->
<!-- Speaker notes: This deck covers stability selection and structured sparsity. The big idea: single Lasso fits are fragile. Stability selection asks "which features are consistently selected?" across many random subsamples. Structured sparsity (Group Lasso, Fused Lasso) enforces prior knowledge about feature relationships. -->

# Stability Selection and Structured Sparsity
## From Fragile to Robust Feature Selection

### Module 04 — Embedded Methods

Subsampling + regularisation for FDR control, group selection, and adjacent smoothness

---

<!-- Speaker notes: Open with the fragility problem. Run Lasso twice on slightly different data — you can get very different feature sets. This motivates stability selection: measure how often each feature is selected, rather than whether it is selected in one run. -->

## The Problem: Single Lasso Fits Are Fragile

Run Lasso on full data → feature set $A$

Remove 5 observations, refit → feature set $B$

$$A \neq B \quad \text{even for large } n$$

**Why?** Among correlated features, the KKT conditions are near-degenerate. Small data perturbations tip the balance between competing features.

**Solution:** Measure **selection probability** across many random subsamples.

$$\hat{\Pi}_j^\lambda = \mathbb{P}^*[\text{feature } j \text{ selected at penalty } \lambda]$$

---

<!-- Speaker notes: The stability selection algorithm is simple: subsample, fit, record. Repeat 100 times. The selection probability is just the fraction of times each feature was included. Then threshold: features above pi_thr are declared stable. -->

## Stability Selection: The Algorithm

```
Input: Data (X, y), fraction q, threshold π_thr, lambda grid Λ, B subsamples

For b = 1 to B:
  1. Draw subsample I_b of size ⌊qn⌋ without replacement
  2. Fit Lasso path on (X_{I_b}, y_{I_b}) over Λ
  3. Record: S^b(λ) = {j : β̂_j ≠ 0} for each λ

Selection probability:  Π̂_j^λ = (1/B) Σ_b 1[j ∈ S^b(λ)]

Stable set:  Ŝ^stable = {j : max_λ Π̂_j^λ ≥ π_thr}
```

**Default parameters:** $q = 0.5$, $B \geq 100$, $\pi_{\text{thr}} \in [0.6, 0.9]$

---

<!-- Speaker notes: The main theorem gives a finite-sample bound on the expected number of false positives. Walk through the components: pi_thr (threshold), E[|S_lambda|] (average model size), p (total features). The key insight is that the bound is distribution-free -- no Gaussianity required. -->

## Theoretical Guarantee (Meinshausen & Bühlmann 2010)

$$\mathbb{E}[|\hat{S}^{\text{stable}} \cap \text{Irr}|] \leq \frac{1}{2\pi_{\text{thr}} - 1} \cdot \frac{q^2}{1-q} \cdot \frac{\mathbb{E}[|\hat{S}^\lambda|]^2}{p}$$

| Symbol | Meaning |
|--------|---------|
| $|\hat{S}^{\text{stable}} \cap \text{Irr}|$ | Number of falsely selected stable features |
| $\pi_{\text{thr}}$ | Selection probability threshold |
| $\mathbb{E}[|\hat{S}^\lambda|]$ | Average model size at chosen $\lambda$ |
| $p$ | Total number of features |

**Distribution-free:** No Gaussianity, no sparsity assumption on the true model.

---

<!-- Speaker notes: Give concrete numbers. With p=100 features, average model size 10, and target <= 1 false positive, what threshold do we need? Work through the algebra. Then show that if the average model size is too large, no threshold can give the guarantee. -->

## Setting the Threshold: Worked Example

**Setup:** $p = 100$ features, $q = 0.5$, target $\leq 1$ false positive

$$\mathbb{E}[\text{FP}] \leq \frac{1}{2\pi_{\text{thr}} - 1} \cdot 1 \cdot \frac{\mathbb{E}[|\hat{S}^\lambda|]^2}{100} \leq 1$$

Solving with $\mathbb{E}[|\hat{S}^\lambda|] = 5$ (sparse average model):

$$\pi_{\text{thr}} \geq \frac{1}{2}\left(1 + \frac{25}{100}\right) = 0.625$$

With $\mathbb{E}[|\hat{S}^\lambda|] = 15$ (denser average):

$$\pi_{\text{thr}} \geq \frac{1}{2}\left(1 + \frac{225}{100}\right) = 1.625 \quad \text{(impossible!)}$$

> **Control model size** to make the guarantee achievable.

---

<!-- Speaker notes: The stability path is the primary output of stability selection. Show how to read it: high flat curves are stable, noisy low curves are unstable. Crossing curves indicate correlated features competing for selection. The threshold is a horizontal line across the plot. -->

## Reading Stability Path Plots

```
Selection
Probability
  1.0 |  feature 3 ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
  0.9 |
  0.8 |--threshold------------------------------------------
  0.7 |          feature 1 ‾‾‾‾‾\_____
  0.6 |
  0.5 |              feature 5 /‾‾‾‾‾\___
  0.3 |  ~~~feature 7 (noise)~~~~~~~~~~~~~~~
  0.0 +----------------------------------------→ λ (↓)
```

- **High, flat** = robustly selected = stable feature
- **Rising plateau** = enters at specific $\lambda$, stays = stable entry
- **Noisy, low** = noise feature — excluded by threshold
- **Crossing** = correlated pair — both get ~0.5 probability

---

<!-- Speaker notes: Group Lasso extends the L1 penalty to act on groups rather than individual coefficients. The key formula is the L2 norm of each group's coefficients in the penalty. This produces all-in or all-out group selection. -->

## Group Lasso: Selecting Groups Together

**Setup:** Partition features into $G$ groups $\mathcal{G}_1, \ldots, \mathcal{G}_G$ with sizes $p_g$.

$$\mathcal{L}_{\text{GL}}(\beta) = \|y - X\beta\|_2^2 + \lambda \sum_{g=1}^G \sqrt{p_g} \|\beta_{\mathcal{G}_g}\|_2$$

**KKT condition for group $g$ to be zero:**

$$\|X_{\mathcal{G}_g}^\top r\|_2 \leq \lambda \sqrt{p_g}$$

One condition on the **whole group** → all-in or all-out selection.

---

<!-- Speaker notes: The block coordinate descent update for Group Lasso is a block soft-thresholding operation. This is the vector analogue of the scalar soft-thresholding for standard Lasso. Make sure students see the connection: shrink toward zero, but in the L2 norm sense. -->

## Group Lasso: Block Soft-Thresholding

The coordinate descent update for group $g$ (given all other groups):

$$S_g = X_{\mathcal{G}_g}^\top r_g \quad \text{(residual correlation for group } g\text{)}$$

$$\hat{\beta}_g \leftarrow \left(1 - \frac{\lambda \sqrt{p_g}}{\|S_g\|_2}\right)_+ S_g$$

This is **block soft-thresholding**: shrink $S_g$ toward zero in L2 norm.

Compare to scalar Lasso update: $\hat{\beta}_j \leftarrow \text{sign}(S_j)(|S_j| - \lambda)_+$

---

<!-- Speaker notes: When to use Group Lasso vs Lasso. Give concrete examples: dummy variables from a categorical feature must be included/excluded as a group. Gene pathway analysis: select whole pathways. Time series lags: select whole lag windows. -->

## When to Use Group Lasso

| Scenario | Group Definition |
|----------|-----------------|
| One-hot encoded categorical | All dummies from one variable |
| Polynomial features ($x, x^2, x^3$) | All powers of one base feature |
| Time series lags ($x_{t-1}, \ldots, x_{t-k}$) | Lags of one underlying variable |
| Gene pathway analysis | Genes in same biological pathway |
| Financial factors | Features loading on same risk factor |

**Requirement:** Groups must be pre-specified. Group Lasso cannot discover groups.

---

<!-- Speaker notes: Sparse Group Lasso adds within-group sparsity. The mixing parameter alpha controls the balance between group selection and within-group selection. This is useful when groups are known but not all group members are expected to be relevant. -->

## Sparse Group Lasso: Two Levels of Sparsity

$$\mathcal{L}_{\text{SGL}}(\beta) = \|y - X\beta\|_2^2 + (1-\alpha)\lambda \sum_{g=1}^G \sqrt{p_g}\|\beta_{\mathcal{G}_g}\|_2 + \alpha\lambda \|\beta\|_1$$

<div class="columns">

**$\alpha = 0$ (Group Lasso)**
- Selects whole groups
- All group members in or all out
- No within-group selection

**$\alpha = 1$ (Lasso)**
- Selects individual features
- No group structure enforced
- Groups can be partially selected

</div>

**$\alpha \in (0,1)$:** Group-level and within-group sparsity simultaneously.

---

<!-- Speaker notes: Fused Lasso is for ordered features (time series, genomics). The fusion penalty encourages adjacent coefficients to be similar, producing piecewise constant coefficient sequences. This is Total Variation regularisation on the coefficient sequence. -->

## Fused Lasso: Adjacent Coefficient Similarity

**Setup:** Features have a natural ordering (time $t-1, t-2, \ldots$ or chromosomal position).

$$\mathcal{L}_{\text{FL}}(\beta) = \|y - X\beta\|_2^2 + \lambda_1 \|\beta\|_1 + \lambda_2 \sum_{j=2}^{p} |\beta_j - \beta_{j-1}|$$

- $\lambda_1$: global sparsity (push coefficients to zero)
- $\lambda_2$: fusion (adjacent coefficients similar — piecewise constant)

**Interpretation:** Select **contiguous windows** of lags rather than scattered individual lags.

---

<!-- Speaker notes: Visualise what the Fused Lasso does to a coefficient sequence. Without fusion: scattered non-zero coefficients. With fusion: contiguous blocks. For time series lag selection, this is much more interpretable -- you find the relevant lag window rather than isolated lags. -->

## Fused Lasso: Coefficient Profile

Without fusion ($\lambda_2 = 0$):
```
β: 0  0  0.5  0  0  -0.3  0  0.4  0  0
   ← scattered, hard to interpret
```

With fusion ($\lambda_2 > 0$):
```
β: 0  0  0.4  0.4  0.4  0  0  0  0  0
   ← contiguous block of selected lags
```

**For lag selection:** Fused Lasso identifies the relevant lag window, not scattered lags.

Efficient solution via FLSA algorithm: $O(p)$ time (Friedman et al. 2007).

---

<!-- Speaker notes: Summarise the comparison between methods. The table helps students decide which method to use in practice. Key decision points: is there group structure? Is there ordering? Do you need uncertainty quantification? -->

## Comparison: Which Method for Which Scenario?

| Method | Group Structure | Ordering | Uncertainty |
|--------|----------------|----------|-------------|
| Lasso | No | No | No |
| ElasticNet | Implicit (grouping effect) | No | No |
| Group Lasso | Yes, pre-specified | No | No |
| Sparse Group Lasso | Yes + within-group | No | No |
| Fused Lasso | No | Yes | No |
| **Stability Selection** | Optional | Optional | **Yes** |

> Stability selection wraps any of the above and adds uncertainty quantification.

---

<!-- Speaker notes: Summarise key practical parameters. Students should leave with concrete numbers they can use immediately. Emphasise that stability selection adds computational cost but provides guarantees that single Lasso fits cannot. -->

## Practical Parameter Summary

**Stability Selection:**
- Subsample fraction $q = 0.5$
- Subsamples $B \geq 100$
- Threshold $\pi_{\text{thr}} \in [0.6, 0.9]$; start at $0.75$
- Lambda grid: geometric from $\lambda_{\max}$ to $0.05\lambda_{\max}$

**Group Lasso:**
- Define groups from domain knowledge or correlation clustering
- $\sqrt{p_g}$ normalisation is essential for fair group comparison
- Start $\lambda$ at a value selecting 3–5 groups

**Fused Lasso:**
- Only use when features have meaningful ordering
- $\lambda_2 / \lambda_1$ ratio controls smoothness vs. sparsity

---

<!-- Speaker notes: Preview the next guide. Tree-based importance methods and the knockoff filter are the other major embedded approaches. The knockoff filter provides rigorous FDR control using a completely different approach -- constructing "decoy" features that act as negative controls. -->

<!-- _class: lead -->

## Next: Tree Importance and Knockoff Filter

Guide 03 covers:
- Impurity-based vs. permutation-based vs. SHAP importance
- Known biases and corrections
- Model-X knockoffs: FDR control without stability subsampling
- Attention-based importance in modern architectures
