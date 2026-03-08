# Tree-Based Importance, Knockoff Filter, and Attention-Based Selection

## In Brief

Tree-based models provide three flavours of feature importance: impurity-based (fast but biased), permutation-based (model-agnostic but expensive), and SHAP-based (consistent, additive, theoretically grounded). Each has known failure modes. The knockoff filter (Barber & Candès 2015, extended to Model-X knockoffs by Candès et al. 2018) provides exact false discovery rate control for any feature importance method by constructing synthetic "knockoff" copies of each feature that act as built-in negative controls.

## Key Insight

Impurity-based importance in scikit-learn is not measuring what most practitioners think. It is biased toward high-cardinality features and correlated features independently of their actual predictive value. Permutation importance corrects the cardinality bias but not the correlation bias. Conditional permutation importance (SAGE, PIMP) corrects for correlations. The knockoff filter bypasses all of these by constructing a test with provable FDR control.

---

## 1. Impurity-Based Importance (Mean Decrease in Impurity)

### Definition

For a Random Forest with $T$ trees, the impurity-based importance of feature $j$ is:

$$\text{MDI}(j) = \frac{1}{T} \sum_{t=1}^T \sum_{\text{nodes } v \in t : \text{split on } j} p(v) \cdot \Delta I(v)$$

where:
- $p(v)$: fraction of samples reaching node $v$
- $\Delta I(v)$: decrease in impurity (Gini or entropy) at node $v$ due to the split

### Why Impurity Importance Is Biased

**Bias toward high-cardinality features (Strobl et al. 2007):**

A feature with many unique values can make finer splits, producing a larger average impurity decrease than a coarser feature — even if both are equally predictive. Continuous features are systematically preferred over binary features.

**Mathematical argument:** For a feature $X_j$ with support $\{v_1, \ldots, v_k\}$, the expected impurity decrease is proportional to $\log k$ under the null hypothesis that $X_j$ is independent of $y$. This means a uniformly random continuous feature ($k \to \infty$) has higher expected MDI than a useless binary feature.

**Bias toward correlated features:**

If $X_j$ and $X_k$ are correlated, the tree can split on either one interchangeably. In some trees, $X_j$ appears first and gets full credit; in others, $X_k$ does. The total importance is shared, but not proportionally to the marginal contribution — whichever appears higher in the tree gets a larger share due to the larger $p(v)$ at higher nodes.

### The Strobl et al. Correction

Strobl et al. (2007) proposed using **conditional permutation importance** (see Section 3) or using the **unbiased tree** implementation that samples features and split points from appropriate distributions. In practice, use permutation importance or SHAP instead of MDI when feature cardinality or correlation is a concern.

---

## 2. Permutation-Based Importance (Mean Decrease in Accuracy)

### Definition

Permutation importance (Breiman 2001) measures how much model accuracy decreases when feature $j$'s values are randomly shuffled:

$$\text{MDA}(j) = \frac{1}{B} \sum_{b=1}^B \left[\text{Acc}(X_{\text{oob}}^b, y_{\text{oob}}^b) - \text{Acc}(\tilde{X}_{\text{oob},j}^b, y_{\text{oob}}^b)\right]$$

where $\tilde{X}_{\text{oob},j}^b$ is the out-of-bag test set with feature $j$'s values permuted.

Permuting feature $j$ breaks its relationship with $y$ while preserving its marginal distribution. A large drop in accuracy indicates feature $j$ was important.

### Advantages Over MDI

- **Cardinality-unbiased:** Permuting a high-cardinality feature yields no advantage over permuting a binary feature under the null hypothesis.
- **Post-hoc:** Computed after training; works with any model, not just trees.
- **Interpretable scale:** Expressed in the same units as the performance metric.

### Remaining Bias: Correlation

If $X_j$ and $X_k$ are correlated, permuting $X_j$ creates an out-of-distribution sample: $X_j$ is now independent of $X_k$ even though in the training data they were correlated. The model may be able to recover $y$ using $X_k$ even with $X_j$ permuted, underestimating $X_j$'s true importance.

This means permutation importance **underestimates** the importance of features that have correlated counterparts in the feature set.

### Permutation Importance with sklearn

```python
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Compute permutation importance on validation set
result = permutation_importance(
    rf, X_val, y_val,
    n_repeats=30,          # repeat permutation 30 times for stability
    random_state=42,
    scoring='r2'
)

importances = result.importances_mean
importances_std = result.importances_std

# Features with importance significantly above zero
significant = importances > 2 * importances_std
```

---

## 3. Conditional Permutation Importance

### Motivation

Standard permutation importance permutes feature $j$ unconditionally, creating out-of-distribution samples when features are correlated. Conditional permutation importance permutes $X_j$ **within strata** defined by the values of correlated features, preserving the conditional distribution $p(X_j | X_{-j})$.

### The PIMP Procedure (Altmann et al. 2010)

Permutation Importance Permutation test: under the null hypothesis that feature $j$ is independent of $y$, the distribution of permutation importance can be estimated by repeatedly:
1. Permuting $y$ (not $X_j$)
2. Retraining the forest
3. Computing the importance of feature $j$

The resulting null distribution is used to compute a p-value for the observed importance.

### SAGE: Shapley Additive Global importancE

SAGE (Covert et al. 2020) computes feature importance as the marginal contribution to model performance, averaged over all possible feature subsets. This is equivalent to the Shapley value from cooperative game theory, applied to the model's expected accuracy.

For a model $f$ and dataset $D$:

$$\text{SAGE}(j) = \sum_{S \subseteq [p] \setminus \{j\}} \frac{|S|!(p-|S|-1)!}{p!} \left[v(S \cup \{j\}) - v(S)\right]$$

where $v(S)$ is the expected model loss when only features in $S$ are available (others marginalised out via conditional distribution).

---

## 4. SHAP-Based Feature Importance

### SHAP Values: Consistent, Additive Attribution

SHAP (SHapley Additive exPlanations, Lundberg & Lee 2017) decomposes each prediction into additive contributions from each feature:

$$f(x) = \phi_0 + \sum_{j=1}^p \phi_j(x)$$

where $\phi_j(x)$ is the Shapley value of feature $j$ for instance $x$:

$$\phi_j(x) = \sum_{S \subseteq [p] \setminus \{j\}} \frac{|S|!(p-|S|-1)!}{p!} \left[f_{S \cup \{j\}}(x) - f_S(x)\right]$$

### Global SHAP Importance

Aggregate over all instances to get global feature importance:

$$\text{SHAP-Importance}(j) = \frac{1}{n} \sum_{i=1}^n |\phi_j(x_i)|$$

### TreeSHAP: Efficient Exact Computation for Trees

For tree-based models, TreeSHAP computes exact SHAP values in $O(TLD^2)$ time (where $T$ = trees, $L$ = leaves, $D$ = depth), compared to $O(TLD \cdot 2^p)$ for the naive approach.

### Advantages of SHAP

1. **Consistency:** If model $A$ assigns higher importance to feature $j$ than model $B$ for all datasets, then SHAP(j, A) ≥ SHAP(j, B).
2. **Local + Global:** Both instance-level explanation and global importance.
3. **No cardinality or correlation bias** (in the marginal SHAP formulation).
4. **Handles interactions:** SHAP interaction values decompose pairwise feature interactions.

### SHAP Implementation

```python
import shap

# TreeSHAP for tree-based models
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_val)

# Global importance: mean |SHAP|
shap_importance = np.abs(shap_values).mean(axis=0)

# SHAP beeswarm plot (feature importance + effect direction)
shap.summary_plot(shap_values, X_val, feature_names=feature_names)

# SHAP bar plot (global importance ranking)
shap.summary_plot(shap_values, X_val, plot_type='bar', feature_names=feature_names)
```

---

## 5. The Knockoff Filter: FDR-Controlled Feature Selection

### The FDR Framework

False Discovery Rate (FDR) is the expected fraction of selected features that are truly null:

$$\text{FDR} = \mathbb{E}\left[\frac{|\hat{S} \cap H_0|}{|\hat{S}| \vee 1}\right]$$

Standard feature selection methods do not control FDR. The knockoff filter provides exact FDR control at any pre-specified level $q$.

### The Original Knockoff Filter (Barber & Candès 2015)

For Gaussian linear models, construct knockoff features $\tilde{X}$ satisfying:

1. **Pairwise exchangeability:** $(X, \tilde{X})$ has the same joint distribution as $(X_{\text{swap}(S)}, \tilde{X}_{\text{swap}(S)})$ for any subset $S$ (swapping original and knockoff versions of features in $S$).
2. **Conditional independence:** $\tilde{X} \perp y \mid X$ — knockoffs do not carry additional information about $y$ beyond what $X$ already provides.

### Model-X Knockoffs (Candès et al. 2018)

The Model-X framework extends knockoffs to arbitrary models and distributions, requiring only that the joint distribution $p(X)$ of the features is known.

**Construction:** For each feature $j$, construct $\tilde{X}_j$ such that:

$$(\tilde{X}_1, \ldots, \tilde{X}_p) \overset{d}{=} (X_1, \ldots, X_p) \quad \text{(same marginal)}$$

$$\tilde{X}_j \perp y \mid X \quad \text{(knockoff is null by construction)}$$

$$\text{Corr}(\tilde{X}_j, X_j) \approx 1 \quad \text{(knockoff closely mirrors the original)}$$

### The Knockoff Procedure

1. Construct knockoffs $\tilde{X}$
2. Fit any model (e.g., Lasso) on the augmented matrix $[X, \tilde{X}]$
3. Compute feature statistics $W_j = |Z_j| - |\tilde{Z}_j|$ where $Z_j$ and $\tilde{Z}_j$ are the model's importance scores for $X_j$ and $\tilde{X}_j$
4. Choose threshold $\tau$ using the knockoff+ procedure:

$$\tau = \min\left\{t > 0 : \frac{1 + |\{j : W_j \leq -t\}|}{|\{j : W_j \geq t\}| \vee 1} \leq q\right\}$$

5. Select features: $\hat{S} = \{j : W_j \geq \tau\}$

### Why FDR Is Controlled

The key insight: true null features have $W_j$ symmetrically distributed around 0 (because swapping $X_j$ and $\tilde{X}_j$ leaves the joint null distribution unchanged). The numerator $1 + |\{j : W_j \leq -t\}|$ counts features with large negative $W_j$ — these are features where the knockoff scored higher than the original, which cannot happen for truly important features. This gives a conservative estimate of the number of false discoveries.

### Second-Order Knockoffs

When $X \sim N(\mu, \Sigma)$, construct $\tilde{X}$ from the conditional:

$$\tilde{X} \mid X \sim N(\mu_{\tilde{X}|X}, \Sigma_{\tilde{X}|X})$$

Using the block matrix structure:

$$\begin{pmatrix} X \\ \tilde{X} \end{pmatrix} \sim N\left(\begin{pmatrix}\mu \\ \mu\end{pmatrix}, \begin{pmatrix}\Sigma & \Sigma - S \\ \Sigma - S & \Sigma\end{pmatrix}\right)$$

where $S = \text{diag}(s_1, \ldots, s_p)$ is chosen to maximise $\min_j s_j$ (minimum reconstructability) subject to $2\Sigma - S \succeq 0$.

The diagonal matrix $S$ is found by semidefinite programming:

```python
# SDP for optimal knockoff construction
import cvxpy as cp

Sigma = np.cov(X_train.T)
p = Sigma.shape[0]

s = cp.Variable(p)
constraints = [
    s >= 0,
    2 * Sigma - cp.diag(s) >> 0  # positive semidefinite
]
objective = cp.Maximize(cp.sum(s))
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.SCS)

S_diag = s.value
```

### Deep Knockoffs (Romano et al. 2019)

For non-Gaussian $X$, construct knockoffs using a deep generative model trained to satisfy the exchangeability condition. The network is trained with a penalty that measures violation of the pairwise exchangeability property.

---

## 6. Attention-Based Feature Importance

### TabNet: Sequential Attention for Feature Selection

TabNet (Arik & Pfister 2021) uses a sequential attention mechanism to select features at each decision step. The attention mask $M^{(k)}$ at step $k$ assigns weights to features:

$$h^{(k)} = f^{(k)}\left(M^{(k)} \odot a^{(k)}\right)$$

$$M^{(k)}_j = \frac{\exp(h^{(k-1)}_j \cdot P^{(k)}_j)}{\sum_{j'} \exp(h^{(k-1)}_{j'} \cdot P^{(k)}_{j'})}$$

The **feature importance** is the cumulative attention across steps:

$$I_j = \sum_k \eta_k M^{(k)}_j$$

where $\eta_k$ is the relative contribution of step $k$ to the final prediction.

### FT-Transformer: Feature Tokenisation

FT-Transformer (Gorishniy et al. 2021) treats each tabular feature as a "token" (analogous to a word in NLP). The attention weights in the transformer layers provide natural feature importance scores.

### SAINT: Self-Attention and Intersample Attention

SAINT (Somepalli et al. 2021) applies attention both across features (row-wise) and across samples (column-wise), producing importance scores that account for sample-level interactions.

### Limitations of Attention-Based Importance

- Attention weights measure **routing** in the model, not causal importance
- High attention to feature $j$ means the model uses $j$, not that $j$ is causally important
- Correlated features can have attention split arbitrarily between them
- Attention-based importance does not provide FDR control

---

## 7. Comparison Matrix: Which Method for Which Scenario

| Method | Cardinality Bias | Correlation Bias | FDR Control | Cost | Best For |
|--------|-----------------|-----------------|-------------|------|----------|
| MDI (Gini) | High | Medium | No | Low | Quick baseline only |
| Permutation | None | Medium | No | Medium | General use, uncorrelated features |
| SHAP | None | Low | No | High | Interpretability, publication |
| Conditional Permutation | None | None | No | High | Correlated feature sets |
| Knockoff Filter | None | None | **Yes** | High | Rigorous FDR control needed |
| Attention (TabNet/FT) | None | Medium | No | High | Deep learning models |
| Stability Selection | None | Low | Approximate | High | Uncertainty quantification |

### Decision Guide

**Use MDI only:** As a fast initial ranking when features are known to be uncorrelated and roughly equal cardinality.

**Use Permutation:** When you need a quick importance ranking that is not biased by cardinality; acceptable when feature correlations are moderate.

**Use SHAP:** When you need instance-level explanations alongside global importance, or when presenting results to non-technical stakeholders.

**Use Conditional Permutation / SAGE:** When features are highly correlated and you need importance that reflects the marginal contribution given all other features.

**Use Knockoff Filter:** When you need formal FDR control — regulatory settings, biomarker discovery, any application where false positives have serious consequences.

**Use Stability Selection:** When you want selection with uncertainty quantification and the knockoff covariate distribution is unknown.

---

## Common Pitfalls

- **Reporting MDI from sklearn as "feature importance" without qualification**: Always note the cardinality and correlation biases.
- **Computing permutation importance on training data**: Always use out-of-bag or held-out validation data.
- **Using knockoffs with small $n$**: Knockoff FDR guarantees require sufficient power. With $n < 3p$, knockoff power is low and few features will be selected.
- **Treating SHAP values as causal effects**: SHAP measures prediction attribution, not causal importance. Correlated features can have their true causal effect redistributed across correlated proxies.
- **Using knockoffs when $p(X)$ is misspecified**: The knockoff FDR guarantee depends on correct specification of the feature distribution. Use empirical second-order knockoffs or deep knockoffs when the distribution is unknown.

---

## Connections

- **Builds on:** Random forests (Module 03), SHAP theory, multiple testing (Benjamini-Hochberg)
- **Leads to:** Causal feature selection (Module 09), production pipelines (Module 11)
- **Related to:** Multiple hypothesis testing, Benjamini-Hochberg procedure, conditional independence testing

---

## Further Reading

- Breiman (2001) "Random Forests" — original permutation importance definition
- Strobl et al. (2007) "Bias in Random Forest Variable Importance Measures" — cardinality bias documentation
- Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions" — SHAP theory
- Barber & Candès (2015) "Controlling the False Discovery Rate via Knockoffs" — original knockoff filter
- Candès et al. (2018) "Panning for Gold: Model-X Knockoffs for High-Dimensional Controlled Variable Selection" — Model-X knockoffs
- Romano et al. (2019) "Deep Knockoffs" — neural network knockoff construction
- Arik & Pfister (2021) "TabNet: Attentive Interpretable Tabular Learning" — attention-based selection
