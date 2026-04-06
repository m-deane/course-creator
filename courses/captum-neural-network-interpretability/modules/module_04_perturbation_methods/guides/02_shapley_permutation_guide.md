# Guide 02: Shapley Values and Permutation Feature Importance

> **Reading time:** ~8 min | **Module:** 4 — Perturbation Methods | **Prerequisites:** Module 0 Foundations


## Overview

Shapley values provide the game-theoretic foundation for feature attribution. Unlike Occlusion (which measures feature removal one at a time), Shapley values consider all possible feature coalitions, capturing interaction effects and providing uniquely fair attribution under four axioms. This guide covers Shapley Value Sampling in Captum and its connection to SHAP, as well as Permutation Feature Importance for global model understanding.


<div class="callout-key">

<strong>Key Concept Summary:</strong> This guide covers the core concepts of guide 02: shapley values and permutation feature importance.

</div>

---

## 1. Shapley Values: Game Theory Foundation

Shapley values originate from cooperative game theory (Shapley, 1953). In the attribution context:

- **Players:** features $F_1, F_2, \ldots, F_n$
- **Game:** the prediction function $f(x)$
- **Payoff:** the prediction score $f(x)$ minus the baseline prediction $f(x')$
- **Shapley value of feature $i$:** fair attribution of the payoff to player $i$

The Shapley value is defined as:

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! (n - |S| - 1)!}{n!} \left[ v(S \cup \{i\}) - v(S) \right]$$

where:
- $N = \{1, 2, \ldots, n\}$ is the set of all features
- $S$ ranges over all subsets that do not include feature $i$
- $v(S)$ is the model's prediction when only features in $S$ are present (others replaced by baseline)

**Interpretation:** The Shapley value of feature $i$ is its average marginal contribution across all possible orderings in which features are added.

---

## 2. The Four Shapley Axioms

Shapley values are the unique attribution method satisfying all four axioms simultaneously:

### Axiom 1: Efficiency (Completeness)
$$\sum_{i=1}^n \phi_i = f(x) - f(x')$$

The attributions sum to the total prediction change. Same as IG's completeness property.

### Axiom 2: Symmetry
If two features $i$ and $j$ have the same marginal contribution in all coalitions they appear in together, they receive the same attribution:
$$v(S \cup \{i\}) = v(S \cup \{j\}) \implies \phi_i = \phi_j$$

### Axiom 3: Dummy
If a feature contributes nothing in any coalition (its presence never changes the output):
$$v(S \cup \{i\}) = v(S) \text{ for all } S \implies \phi_i = 0$$

### Axiom 4: Additivity
For two games (models) $f$ and $g$ that can be combined:
$$\phi_i^{f+g} = \phi_i^f + \phi_i^g$$

These axioms make Shapley values the **uniquely fair** attribution: no other method satisfies all four simultaneously.

---

## 3. Computational Challenge

Exact Shapley value computation requires evaluating $v(S)$ for all $2^n$ subsets. For $n=11$ tabular features, this is $2^{11} = 2048$ model calls. For $n=100$ features, $2^{100}$ is infeasible.
<div class="callout-warning">

<strong>Warning:</strong> Exact Shapley value computation requires evaluating $v(S)$ for all $2^n$ subsets. For $n=11$ tabular features, this is $2^{11} = 2048$ model calls. For $n=100$ features, $2^{100}$ is infeasible.

</div>


**Solution: Shapley Value Sampling**

Estimate Shapley values by sampling random orderings:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>
<div class="code-body">

```python
# Conceptual algorithm
shapley_estimates = {i: 0 for i in range(n_features)}
for _ in range(n_samples):
    # Random ordering of features
    ordering = random_permutation(range(n_features))
    # For each feature, compute its marginal contribution
    # in this ordering (considering all features before it as present)
    for k, feature_i in enumerate(ordering):
        coalition_before = ordering[:k]
        shapley_estimates[feature_i] += (
            v(coalition_before | {feature_i}) - v(coalition_before)
        )
# Average over samples
shapley_estimates = {i: v / n_samples for i, v in shapley_estimates.items()}
```

</div>

</div>

Each sample requires $n$ model evaluations (one per position in the ordering). With $n\_\text{samples}$ Monte Carlo samples: total cost = $n \times n\_\text{samples}$ forward passes.

---

## 4. Captum ShapleyValueSampling API

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>
<div class="code-body">
<div class="callout-key">

<strong>Key Point:</strong> from captum.attr import ShapleyValueSampling

</div>


```python
from captum.attr import ShapleyValueSampling

model = trained_model.eval()

svs = ShapleyValueSampling(model)

# Compute Shapley value estimates
attributions = svs.attribute(
    input_tensor,           # Input: (1, n_features) for tabular
    baselines=baseline,     # Reference: training mean
    target=class_idx,       # Target class (for classifiers)
    n_samples=200           # Monte Carlo samples (default: 25)
)
# attributions: same shape as input
```

</div>

</div>

**n_samples trade-off:**
- `n_samples=25`: fast, high variance — good for exploration
- `n_samples=200`: slower, lower variance — good for reporting
- `n_samples=500+`: publication quality for important decisions

### Error Estimate

Captum does not provide variance estimates directly, but you can estimate via bootstrap:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>
<div class="code-body">

```python
from scipy.stats import sem

# Run multiple times and compute standard error
runs = [
    svs.attribute(input_tensor, baselines=bl, target=cls, n_samples=50)
    .squeeze().detach().numpy()
    for _ in range(10)
]
shapley_mean = np.stack(runs).mean(axis=0)
shapley_stderr = sem(np.stack(runs), axis=0)
```

</div>

</div>

---

## 5. Shapley Values vs. Integrated Gradients

Shapley values and IG both satisfy completeness. Their differences:

| Property | IG | Shapley Values |
|----------|-----|----------------|
| Completeness | Yes | Yes |
| Sensitivity | Yes | Yes |
| Symmetry | No (path-dependent) | Yes |
| Dummy feature | Approximately | Yes |
| Interaction effects | No (linear path) | Yes (all coalitions) |
| Speed | n_steps passes | n × n_samples passes |
| Model-agnostic | No (needs gradients) | Yes |

**Key difference: interaction effects.** IG attributes each feature independently along a linear path. Shapley values measure each feature's *marginal contribution* across all possible subsets of other features, capturing interactions.

**Example:** If feature A matters only when feature B is also present (interaction effect), IG may give A near-zero attribution (the path doesn't capture the interaction), while Shapley correctly attributes A as important.

---

## 6. SHAP vs. ShapleyValueSampling

SHAP (SHapley Additive exPlanations) is a family of Shapley-based methods:

- **KernelSHAP:** Model-agnostic Shapley approximation (any model)
- **TreeSHAP:** Exact Shapley for tree models (XGBoost, LightGBM)
- **DeepSHAP:** Fast approximation for neural networks (uses IG-like path)
- **LinearSHAP:** Exact Shapley for linear models

Captum's `ShapleyValueSampling` is equivalent to the Monte Carlo version of KernelSHAP. For neural networks in PyTorch, Captum's implementation is more convenient than the standalone `shap` library.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>
<div class="code-body">

```python
# Captum (PyTorch native, works with arbitrary nn.Module)
from captum.attr import ShapleyValueSampling
svs = ShapleyValueSampling(model)
attr = svs.attribute(x, baselines=bl, target=cls, n_samples=200)

# vs. shap library (more general, supports many model types)
import shap
explainer = shap.KernelExplainer(model_fn, background_data)
shap_vals = explainer.shap_values(x)
```

</div>

</div>

---

## 7. Feature Permutation Importance: Global Attribution
<div class="callout-key">

<strong>Key Point:</strong> While Occlusion and Shapley values explain individual predictions (local), **Permutation Feature Importance** provides global feature importance across a dataset:

</div>


While Occlusion and Shapley values explain individual predictions (local), **Permutation Feature Importance** provides global feature importance across a dataset:

**Algorithm:**
1. Measure baseline model performance on validation set: $\text{Score}_0$
2. For each feature $i$:
   a. Randomly shuffle feature $i$ across all validation samples
   b. Measure model performance with feature $i$ permuted: $\text{Score}_i$
   c. Importance of feature $i$ = $\text{Score}_0 - \text{Score}_i$
3. Features with high importance decrease performance when permuted

**Key property:** No need to define a baseline value — permuting breaks the relationship between feature and prediction without removing the feature from the input space.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>
<div class="code-body">

```python
def permutation_importance(model, X_val, y_val, n_repeats=5):
    """
    Compute permutation feature importance for a neural network.

    Parameters
    ----------
    model : callable  f(X) → scores
    X_val : Tensor  (n_samples, n_features)
    y_val : Tensor  (n_samples,)  ground truth
    n_repeats : int  Number of permutations per feature

    Returns
    -------
    importances_mean : ndarray (n_features,)
    importances_std  : ndarray (n_features,)
    """
    with torch.no_grad():
        baseline_preds = model(X_val)
        baseline_score = compute_metric(baseline_preds, y_val)

    n_features = X_val.shape[1]
    all_importances = np.zeros((n_repeats, n_features))

    for rep in range(n_repeats):
        for feat_idx in range(n_features):
            X_permuted = X_val.clone()
            # Shuffle this feature across all samples
            perm = torch.randperm(X_val.shape[0])
            X_permuted[:, feat_idx] = X_val[perm, feat_idx]

            with torch.no_grad():
                perm_preds = model(X_permuted)
                perm_score = compute_metric(perm_preds, y_val)

            all_importances[rep, feat_idx] = baseline_score - perm_score

    return all_importances.mean(axis=0), all_importances.std(axis=0)
```

</div>

</div>

---

## 8. Captum KernelShap

Captum also provides `KernelShap` as a specific implementation following the original KernelSHAP methodology with a weighted regression approach:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>
<div class="code-body">

```python
from captum.attr import KernelShap

ks = KernelShap(model)

# For tabular data
attr = ks.attribute(
    input_tensor,          # (1, n_features)
    baselines=baseline,
    target=class_idx,
    n_samples=100          # Samples for regression
)
```

</div>

</div>

`KernelShap` vs `ShapleyValueSampling`:
- Both approximate Shapley values
- KernelShap uses a weighted regression formulation
- ShapleyValueSampling uses Monte Carlo permutation averaging
- Both converge to the same values with enough samples

---

## 9. Local vs. Global Attribution Summary

| Method | Scope | Question | Cost |
|--------|-------|----------|------|
| IG | Local (one input) | Which features matter for *this* prediction? | n_steps passes |
| Occlusion | Local | Which regions affect *this* prediction? | HW/s² passes |
| ShapleyValueSampling | Local | Fair per-feature attribution with interactions? | n×n_samples passes |
| Permutation Importance | Global | Which features are important *overall*? | n_features × n_repeats × n_samples passes |

**When to use which:**
- **Explain one prediction:** IG (fastest, per-pixel) or Shapley (slowest, most principled)
- **Explain batch of predictions:** Run local methods in batches, then aggregate
- **Understand model globally:** Permutation Importance on validation set
- **Regulatory/documentation use:** Shapley values (axiomatic foundation) or IG (completeness + sensitivity)

---

## 10. Convergence of Shapley Estimates

ShapleyValueSampling is a stochastic estimate — it has variance that decreases with n_samples. Monitor convergence:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>
<div class="code-body">

```python
def shapley_with_convergence_check(model, input_x, baseline, target,
                                    max_samples=500, tolerance=0.01):
    """
    Run ShapleyValueSampling with increasing n_samples until convergence.
    Convergence: max change in attributions between iterations < tolerance.
    """
    svs = ShapleyValueSampling(model)
    prev_attr = None

    for n_samples in [25, 50, 100, 200, 300, 500]:
        attr = svs.attribute(input_x, baselines=baseline,
                              target=target, n_samples=n_samples)
        attr_np = attr.squeeze().detach().numpy()

        if prev_attr is not None:
            max_change = np.abs(attr_np - prev_attr).max()
            print(f"n_samples={n_samples:4d}: max_change={max_change:.5f}")
            if max_change < tolerance:
                print(f"  Converged at n_samples={n_samples}")
                return attr

        prev_attr = attr_np

    print("  Warning: did not converge within max_samples")
    return attr
```

</div>

</div>

---


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Shapley Values: Game Theory Foundation" and why it matters in practice.

2. Given a real-world scenario involving guide 02: shapley values and permutation feature importance, what would be your first three steps to apply the techniques from this guide?

</div>

## Summary

1. **Shapley values** are the unique attribution satisfying Efficiency, Symmetry, Dummy, and Additivity axioms
2. **Exact computation** requires $2^n$ model calls — infeasible for large $n$
3. **ShapleyValueSampling** approximates via Monte Carlo permutation averaging: cost = $n \times n\_\text{samples}$ passes
4. **Captures interactions:** unlike IG's linear path, Shapley considers all feature coalitions
5. **Permutation Feature Importance** is the global (dataset-level) counterpart: measures importance by shuffling features
6. **Captum provides:** `ShapleyValueSampling`, `KernelShap`, and `PermutationFeatureImportance`

---

## Further Reading

- Shapley, "A Value for n-person Games", 1953 — original Shapley value paper
- Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions (SHAP)", NeurIPS 2017
- Strumbelj & Kononenko, "Explaining prediction models and individual predictions with feature contributions", Knowledge and Information Systems 2014
- Captum ShapleyValueSampling: https://captum.ai/api/shapley_value_sampling.html

---

## Cross-References

<a class="link-card" href="../notebooks/01_occlusion_image.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
