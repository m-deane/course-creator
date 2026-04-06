# SHAP Theory: Shapley Values and KernelSHAP

> **Reading time:** ~7 min | **Module:** 5 — SHAP Methods | **Prerequisites:** Module 4 Perturbation Methods


## Learning Objectives

By the end of this guide, you will be able to:
1. Derive the Shapley value formula from cooperative game theory axioms
2. Explain why SHAP satisfies efficiency, symmetry, dummy, and linearity properties
3. Describe how KernelSHAP approximates Shapley values using weighted linear regression
4. Identify the computational trade-offs between exact SHAP and approximate methods
5. Implement KernelSHAP via Captum's `KernelShap` class


<div class="callout-key">

<strong>Key Concept Summary:</strong> This guide covers the core concepts of shap theory: shapley values and kernelshap.

</div>

---

## 1. Cooperative Game Theory Background

SHAP (SHapley Additive exPlanations) originates from cooperative game theory, where the central question is: **given a group of players who collectively produce some value, how should we fairly distribute that value among the players?**

In the machine learning context:
- **Players** = input features $x_1, x_2, \ldots, x_d$
- **Coalition** = a subset $S \subseteq \{1, \ldots, d\}$ of features used to make a prediction
- **Value function** $v(S)$ = the model's expected output when only features in $S$ are known

The value function integrates over the missing features:

$$v(S) = \mathbb{E}_{X_{\bar{S}}}[f(X) \mid X_S = x_S]$$

where $\bar{S} = \{1, \ldots, d\} \setminus S$ is the complement set.

---

## 2. The Shapley Value Formula

The Shapley value for feature $i$ is the average marginal contribution of feature $i$ across all possible orderings (or equivalently, all possible coalitions):

$$\phi_i(v) = \sum_{S \subseteq \{1,\ldots,d\} \setminus \{i\}} \frac{|S|!\,(d - |S| - 1)!}{d!} \left[v(S \cup \{i\}) - v(S)\right]$$

**Interpretation of the weighting factor:**

The factor $\frac{|S|!\,(d - |S| - 1)!}{d!}$ counts the fraction of all $d!$ orderings in which:
1. All features in $S$ appear before feature $i$
2. Feature $i$ appears before all features not in $S \cup \{i\}$

This gives equal weight to every possible ordering of features, making the attribution fair across all coalition sizes.

**Decomposition property:** The Shapley values sum to the model's output minus the baseline:

$$\sum_{i=1}^{d} \phi_i = f(x) - \mathbb{E}[f(X)]$$

---

## 3. The Four Shapley Axioms

The Shapley value is the **unique** attribution method satisfying all four axioms simultaneously:
<div class="callout-warning">

<strong>Warning:</strong> The Shapley value is the **unique** attribution method satisfying all four axioms simultaneously:

</div>


### Axiom 1: Efficiency
The attributions sum exactly to the difference between the model output and the expected output:

$$\sum_{i=1}^{d} \phi_i(f, x) = f(x) - \mathbb{E}[f(X)]$$

No attribution is "wasted" or "double-counted."

### Axiom 2: Symmetry
If two features $i$ and $j$ make identical marginal contributions to every coalition:

$$v(S \cup \{i\}) = v(S \cup \{j\}) \quad \forall S \subseteq \{1,\ldots,d\} \setminus \{i,j\}$$

Then they receive equal attribution: $\phi_i = \phi_j$.

### Axiom 3: Dummy (Null Player)
If feature $i$ contributes nothing to any coalition:

$$v(S \cup \{i\}) = v(S) \quad \forall S \subseteq \{1,\ldots,d\} \setminus \{i\}$$

Then $\phi_i = 0$. Truly irrelevant features receive zero attribution.

### Axiom 4: Linearity (Additivity)
For two games $v$ and $w$:

$$\phi_i(v + w) = \phi_i(v) + \phi_i(w)$$

Attributions from combined models equal the sum of individual attributions.

---

## 4. Why Exact Shapley Values Are Expensive

The number of possible coalitions for $d$ features is $2^d$. For each coalition, evaluating $v(S)$ requires integrating over the missing features — typically approximated by averaging over background samples.

For a model with $d = 20$ features and $100$ background samples:
- Number of coalitions: $2^{20} = 1{,}048{,}576$
- Model evaluations: $\approx 10^8$

This is computationally intractable for most real models. KernelSHAP provides an efficient approximation.

---

## 5. KernelSHAP: Shapley Values as Weighted Linear Regression
<div class="callout-insight">

<strong>Insight:</strong> Lundberg and Lee (2017) showed that Shapley values can be computed by solving a specially weighted linear regression problem.

</div>


Lundberg and Lee (2017) showed that Shapley values can be computed by solving a specially weighted linear regression problem.

### The SHAP Explanation Model

Any additive explanation $g$ has the form:

$$g(z') = \phi_0 + \sum_{i=1}^{d} \phi_i z'_i$$

where $z' \in \{0,1\}^d$ is a binary coalition vector ($z'_i = 1$ means feature $i$ is "present").

### The SHAP Kernel

KernelSHAP finds the $\phi_i$ values that minimize:

$$\mathcal{L}(f, g, \pi_x) = \sum_{z' \in \{0,1\}^d} \pi_x(z') \left[f(h_x(z')) - g(z')\right]^2$$

with the SHAP kernel weight:

$$\pi_x(z') = \frac{d - 1}{\binom{d}{|z'|}|z'|(d - |z'|)}$$

**Key insight:** This kernel assigns higher weight to coalitions of size 0, 1, $d-1$, and $d$ — the coalitions that matter most for Shapley value computation.

### Why This Works

Lundberg and Lee proved that the unique solution to this weighted regression (subject to the efficiency constraint $\sum \phi_i = f(x) - \mathbb{E}[f(X)]$) equals the exact Shapley values.

In practice, KernelSHAP samples a subset of coalitions rather than enumerating all $2^d$, making it tractable for moderate $d$.

---

## 6. Missing Feature Simulation

A critical design choice in SHAP is: **what does it mean for a feature to be "absent"?**

For tabular data, features absent from coalition $S$ are replaced by randomly sampled values from the background dataset:

$$v(S) \approx \frac{1}{n_{bg}} \sum_{k=1}^{n_{bg}} f(x_S, x^{(k)}_{\bar{S}})$$

This marginalizes over the distribution of missing features, matching the theoretical definition.

**Implications:**
- Background dataset should represent the data distribution (typically 50-200 samples)
- Correlated features can receive unintuitive attributions if correlations aren't captured
- For images and text, different masking strategies are used (superpixels, token masking)

---

## 7. Captum KernelSHAP Implementation


<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import torch
from captum.attr import KernelShap

# Load pretrained model
model = torch.load("model.pt")
model.eval()

# KernelSHAP requires a function that takes masked inputs
kernel_shap = KernelShap(model)

# Input and baseline
input_tensor = torch.tensor([[...]], dtype=torch.float32)
baseline = torch.zeros_like(input_tensor)  # or mean of training data

# Compute attributions
attributions = kernel_shap.attribute(
    input_tensor,
    n_samples=500,          # number of coalition samples
    perturbations_per_eval=64,  # batch size for efficiency
    target=class_idx,
    baselines=baseline,
)
# attributions shape: same as input_tensor
```

</div>

</div>

The `n_samples` parameter controls the approximation quality — more samples yield more accurate Shapley value estimates.

---

## 8. KernelSHAP vs. Other SHAP Variants

| Method | Model Type | Speed | Accuracy | Key Assumption |
|--------|-----------|-------|----------|----------------|
| Exact Shapley | Any | Very slow ($O(2^d)$) | Exact | None |
| KernelSHAP | Any (black-box) | Moderate | Approximate | Linear approximation |
| TreeSHAP | Tree ensembles | Fast ($O(TLD^2)$) | Exact | Tree structure |
| LinearSHAP | Linear models | Very fast | Exact | Linear model |
| GradientSHAP | Differentiable | Fast | Approximate | Gradient + expectation |
| DeepLIFT SHAP | Neural networks | Fast | Approximate | Propagation rules |

---

## 9. Practical Considerations

### Background Dataset Selection
- Use 50-200 representative samples from the training distribution
- K-means clustering on training data provides compact, representative backgrounds
- Avoid using test data in the background (data leakage risk)

### Interpreting SHAP Values
- Positive $\phi_i$: feature pushes prediction above the baseline
- Negative $\phi_i$: feature pushes prediction below the baseline
- Magnitude: strength of the feature's influence

### Limitations
- KernelSHAP assumes feature independence when sampling; correlated features require careful interpretation
- Computing SHAP for neural networks with many input dimensions (e.g., images) is slow
- GradientSHAP and DeepLIFTSHAP are preferred for deep networks

---

## 10. Mathematical Connection to Other Methods

SHAP provides a unifying framework. Several existing methods are special cases:

- **LIME** = KernelSHAP with a different (incorrect) kernel weight
- **Integrated Gradients** ≈ GradientSHAP (expectation over baselines)
- **DeepLIFT** ≈ DeepLIFT SHAP (single baseline vs. expectation)
- **Occlusion** ≈ approximate Shapley values with coalition size 1

This unification is one of SHAP's major theoretical contributions.

---


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Cooperative Game Theory Background" and why it matters in practice.

2. Given a real-world scenario involving shap theory: shapley values and kernelshap, what would be your first three steps to apply the techniques from this guide?

</div>

## Summary

| Concept | Key Formula |
|---------|-------------|
| Shapley value | $\phi_i = \sum_{S} \frac{\|S\|!(d-\|S\|-1)!}{d!}[v(S\cup\{i\}) - v(S)]$ |
| Efficiency | $\sum_i \phi_i = f(x) - \mathbb{E}[f(X)]$ |
| SHAP kernel | $\pi_x(z') = \frac{d-1}{\binom{d}{\|z'\|}\|z'\|(d-\|z'\|)}$ |
| Value function | $v(S) = \mathbb{E}_{X_{\bar{S}}}[f(X) \mid X_S = x_S]$ |

---

## Further Reading

- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Shapley, L. S. (1953). A value for n-person games. *Contributions to the Theory of Games*.
- Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*.
- Captum KernelShap documentation: https://captum.ai/api/kernel_shap.html

---

## Cross-References

<a class="link-card" href="../notebooks/01_kernelshap_vs_gradientshap.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
