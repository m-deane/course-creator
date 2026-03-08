---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->
<!-- Speaker notes: This is the third guide of Module 4. We cover three topics: tree-based importance methods (and their biases), the knockoff filter (FDR-controlled selection), and attention-based importance in deep learning. The connective thread is: each method gives a different answer to "which features matter?" and has different failure modes. -->

# Tree Importance, Knockoffs, and Attention
## Which Method for Which Problem?

### Module 04 — Embedded Methods

Impurity, permutation, SHAP, knockoffs, and attention-based importance compared

---

<!-- Speaker notes: Start with the three types of tree importance. MDI is what sklearn gives by default. MDA (permutation) is more reliable. SHAP is the gold standard for interpretability. Each has known failure modes — don't let students assume MDI is "the" feature importance. -->

## Three Flavours of Tree Importance

| Method | Computes | When Computed |
|--------|----------|--------------|
| MDI (Gini) | Avg impurity decrease at splits | During training |
| MDA (Permutation) | Accuracy drop when feature permuted | After training |
| SHAP | Shapley value — marginal contribution | After training |

All three can give **different rankings** on the same model and dataset.

Understanding why reveals what each is actually measuring.

---

<!-- Speaker notes: MDI formula. Walk through the components: p(v) is the fraction of samples at a node (higher nodes have more samples and thus more weight), DeltaI(v) is the impurity decrease. The bias arises from the interaction between the number of possible split points and the expected impurity decrease. -->

## Mean Decrease in Impurity (MDI)

$$\text{MDI}(j) = \frac{1}{T} \sum_{t=1}^T \sum_{v \in t : \text{split on } j} p(v) \cdot \Delta I(v)$$

- $p(v)$: fraction of samples reaching node $v$
- $\Delta I(v)$: impurity decrease at node $v$
- $T$: number of trees

**Fast:** Computed during training at no extra cost.

**Problem:** This metric is biased. Let's see why.

---

<!-- Speaker notes: The cardinality bias is the most important MDI failure mode. A continuous feature with 1000 unique values can find a better split than a binary feature even if both are equally predictive. Under the null hypothesis (feature independent of y), continuous features have higher expected MDI than binary features -- this is the bias. -->

## MDI Bias 1: High-Cardinality Features

**Under the null** (feature $j$ independent of $y$), expected MDI increases with the number of unique values.

**Why:** More unique values = more split point candidates = higher chance of finding a "lucky" split that decreases impurity.

**Consequence:** Continuous features are systematically ranked above binary features even when both are equally useless.

```python
# Demonstration: useless continuous vs useless binary
X_null = pd.DataFrame({
    'continuous_null': np.random.normal(0, 1, 1000),   # useless
    'binary_null': np.random.randint(0, 2, 1000)        # useless
})
# MDI will rank continuous_null above binary_null!
```

---

<!-- Speaker notes: Correlation bias is more subtle. If two features are correlated, the tree alternately uses one or the other. The one that appears first in the tree (higher up, larger p(v)) gets more credit. This is arbitrary -- it depends on which tree happened to split on which feature first. -->

## MDI Bias 2: Correlated Features

If $X_j$ and $X_k$ are correlated, the tree can use either interchangeably.

- In some trees, $X_j$ splits first → $X_j$ gets full credit (large $p(v)$ at high node)
- In other trees, $X_k$ splits first → $X_k$ gets full credit

**Result:** Importance is split between $j$ and $k$, but not proportionally to their actual predictive contribution — it depends on tree-building randomness.

**Strobl et al. (2007):** Documented both biases empirically. Recommended conditional permutation importance instead.

---

<!-- Speaker notes: Permutation importance fixes the cardinality bias. The logic: permuting a feature breaks its relationship with y while preserving its marginal distribution. The accuracy drop measures how much the model relied on that feature. No cardinality advantage because we're measuring model performance, not split quality. -->

## Permutation Importance (MDA): Fixing Cardinality Bias

$$\text{MDA}(j) = \frac{1}{B}\sum_{b=1}^B \left[\text{Acc}(X_{\text{oob}}) - \text{Acc}(\tilde{X}_{\text{oob},j})\right]$$

Permuting feature $j$ **breaks its relationship with $y$** while preserving its marginal distribution.

**Cardinality-unbiased:** Under the null, permuting any feature (binary or continuous) yields zero expected accuracy drop.

**Remaining issue:** Permuting $X_j$ creates **out-of-distribution samples** when $X_j$ is correlated with other features.

---

<!-- Speaker notes: SHAP is the current gold standard. The Shapley value from game theory provides the unique attribution satisfying efficiency (values sum to prediction), symmetry, dummy (null features get zero), and additivity. TreeSHAP computes exact values efficiently for trees. -->

## SHAP Values: Theoretically Grounded Attribution

$$f(x) = \phi_0 + \sum_{j=1}^p \phi_j(x)$$

Each $\phi_j(x)$ is the Shapley value — the unique attribution satisfying:

- **Efficiency:** $\sum_j \phi_j(x) = f(x) - \mathbb{E}[f(X)]$
- **Symmetry:** Features with equal contributions get equal attribution
- **Dummy:** Null features get $\phi_j = 0$
- **Additivity:** Consistent across ensemble members

**Global importance:** $\text{SHAP-Importance}(j) = \frac{1}{n}\sum_i |\phi_j(x_i)|$

**TreeSHAP:** $O(TLD^2)$ — exact for trees, efficient.

---

<!-- Speaker notes: Show the comparison on a concrete correlated feature set. MDI overcounts high-cardinality features. Permutation undercounts correlated features. SHAP is closest to ground truth (measured via data generating process). This is the empirical motivation for using SHAP in high-stakes settings. -->

## Comparison on Correlated Features

Setup: 20 features, 5 truly relevant, 10 correlated with relevant features, 5 pure noise.

| Method | Correlated feature rank (expected) | Noise feature rank |
|--------|----------------------------------|-------------------|
| MDI | Inflated (shares rank with true feature) | Also inflated if continuous |
| Permutation | Deflated (model uses correlated proxy) | Near zero |
| SHAP | Correctly split between correlated pair | Near zero |

> No method is always best. Know the biases for your data structure.

---

<!-- Speaker notes: The knockoff filter is a completely different paradigm. Instead of asking "does the importance score exceed a threshold?", we ask "is the real feature's importance score higher than its knockoff copy?" The knockoff is a synthetic negative control -- we know it's null by construction. -->

## The Knockoff Filter: A Different Paradigm

**Standard importance:** "Is feature $j$'s importance score large enough?"

**Problem:** No reference distribution for "large enough". Threshold is arbitrary.

**Knockoff approach:**
1. Construct **knockoff copies** $\tilde{X}_j$ for each feature — synthetic nulls
2. Fit model on $[X, \tilde{X}]$
3. Feature $j$ is selected if its score exceeds its knockoff's score by enough

$\tilde{X}_j$ is null **by construction**: $\tilde{X}_j \perp y \mid X$

---

<!-- Speaker notes: The pairwise exchangeability condition is the key mathematical property of knockoffs. Swapping X_j and tilde_X_j for any subset S doesn't change the joint distribution. This means a null feature's original and knockoff versions are interchangeable -- the model has no reason to prefer one over the other. -->

## Knockoff Construction: Exchangeability

**Required property (Model-X knockoffs, Candès et al. 2018):**

For any subset $S \subseteq [p]$, swapping $X_j$ and $\tilde{X}_j$ for $j \in S$:

$$(X, \tilde{X})_{\text{swap}(S)} \overset{d}{=} (X, \tilde{X})$$

**And:** $\tilde{X} \perp y \mid X$

**Consequence:** For any null feature $j$ (truly independent of $y$), the statistic $W_j = |Z_j| - |\tilde{Z}_j|$ has a symmetric distribution around zero.

True features will have $W_j > 0$ (original beats knockoff).

---

<!-- Speaker notes: Second-order knockoffs use Gaussian construction. The SDP chooses the diagonal matrix S to maximise the minimum gap between original and knockoff -- making them as different as possible while maintaining exchangeability. More different = higher power to detect true features. -->

## Second-Order Knockoffs for Gaussian Features

If $X \sim N(\mu, \Sigma)$, construct:

$$\begin{pmatrix} X \\ \tilde{X} \end{pmatrix} \sim N\left(\begin{pmatrix}\mu \\ \mu\end{pmatrix}, \begin{pmatrix}\Sigma & \Sigma - S \\ \Sigma - S & \Sigma\end{pmatrix}\right)$$

Choose $S = \text{diag}(s_1, \ldots, s_p)$ by SDP to **maximise distinguishability**:

$$\max_S \min_j s_j \quad \text{subject to} \quad 2\Sigma - S \succeq 0,\ s_j \geq 0$$

Larger $s_j$ → more different knockoff from original → higher detection power.

---

<!-- Speaker notes: The selection threshold tau is the key formula. The numerator counts features where the knockoff scored higher than the original (these are likely false positives, since true features should beat their knockoffs). The denominator counts selected features. The ratio is a conservative FDR estimate. -->

## The Knockoff+ Threshold

**Feature statistic:** $W_j = |Z_j| - |\tilde{Z}_j|$ (original minus knockoff importance)

**Threshold $\tau$ at target FDR level $q$:**

$$\tau = \min\left\{t > 0 : \frac{1 + |\{j : W_j \leq -t\}|}{|\{j : W_j \geq t\}| \vee 1} \leq q\right\}$$

**Numerator:** Features where knockoff beat original (estimated false positives + 1)

**Denominator:** Features where original beat knockoff (selected features)

**Select:** $\hat{S} = \{j : W_j \geq \tau\}$

**Guarantee:** $\mathbb{E}[\text{FDR}] \leq q$ for any model and any distribution of $X$.

---

<!-- Speaker notes: Attention-based importance from TabNet and FT-Transformer. These are elegant but have a key limitation: attention measures routing, not causal importance. A feature can receive high attention simply because it is a reliable proxy for another correlated feature. Do not conflate attention weight with causal importance. -->

## Attention-Based Importance: TabNet

TabNet uses sequential attention masks $M^{(k)}$ at each step:

$$I_j = \sum_k \eta_k M^{(k)}_j \quad \text{(cumulative attention across steps)}$$

**Advantage:** The attention mask is sparse by design — TabNet has a sparsity regulariser that limits how many features can be active per step.

**Limitation:** Attention weight ≠ causal importance. High attention to $X_j$ means the model routes through $X_j$, not that $X_j$ has a direct causal effect on $y$.

**Use for:** Interpretability in deep learning pipelines; not for rigorous feature selection.

---

<!-- Speaker notes: The comparison table is the practical takeaway. Students should leave knowing which method to reach for in which scenario. FDR control is the key differentiator for the knockoff filter. SHAP is best for interpretability. MDI is only for quick-and-dirty exploration. -->

## Decision Matrix: Which Importance Method?

| Scenario | Method |
|----------|--------|
| Quick initial exploration | MDI (fast, use cautiously) |
| General purpose, uncorrelated features | Permutation importance |
| Correlated features, care about attribution | SHAP |
| Correlated features, need causal interpretation | Conditional permutation / SAGE |
| Formal FDR control required | **Knockoff filter** |
| Uncertainty quantification over selection | **Stability selection** |
| Deep learning model | Attention importance + SHAP |

---

<!-- Speaker notes: Summarise the key messages from all three guides in Module 4. Embedded methods build selection into the model. The choice of method depends on the goals: prediction vs. selection, correlated vs. uncorrelated features, need for FDR control. -->

## Module 04 Summary: Embedded Methods

**Regularisation (Guide 01):**
- L1 geometry forces sparsity; LARS computes the complete path
- ElasticNet for correlated groups; use CV to select $\lambda$

**Stability & Structure (Guide 02):**
- Stability selection: selection probability with FDR bound
- Group/Sparse Group/Fused Lasso: structured sparsity

**Tree Importance & Knockoffs (Guide 03):**
- MDI < Permutation < SHAP in bias hierarchy
- Knockoff filter: only method with exact FDR control

---

<!-- Speaker notes: Preview Module 5 on genetic algorithms for feature selection. The key contrast: regularisation methods are continuous (penalty-based), while GAs are combinatorial (search-based). GAs do not require convexity, work with any model, and can handle non-monotone feature interactions. -->

<!-- _class: lead -->

## Next: Module 05 — Genetic Algorithms

Combinatorial search for feature selection:
- No convexity requirement
- Any black-box model
- Non-monotone feature interactions
- Multi-objective selection (accuracy + parsimony)
