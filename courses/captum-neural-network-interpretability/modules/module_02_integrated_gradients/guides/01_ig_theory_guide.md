# Integrated Gradients: Theory and Axiomatic Derivation

> **Reading time:** ~8 min | **Module:** 2 — Integrated Gradients | **Prerequisites:** Module 1 Gradient Methods


## In Brief

Integrated Gradients (Sundararajan, Taly, Yan, 2017) is the uniquely principled gradient-based attribution method — the only one satisfying both the sensitivity and implementation invariance axioms simultaneously. It computes attribution by integrating gradients along a straight-line path from a baseline to the input.

## Key Insight

Saliency evaluates the gradient at one point. Integrated Gradients evaluates the gradient at every point along the path from baseline to input and accumulates (integrates) all contributions. This single change resolves both the saturation problem and converts an approximation into an exact decomposition.


<div class="callout-key">

<strong>Key Concept Summary:</strong> Integrated Gradients (Sundararajan, Taly, Yan, 2017) is the uniquely principled gradient-based attribution method — the only one satisfying both the sensitivity and implementation invariance axioms simultaneously.

</div>

---

## 1. Motivation: The Gap in Gradient Methods

All gradient methods from Module 01 fail at least one of the two fundamental axioms:

**Sensitivity:** If feature $i$ affects the model output when varied from baseline to input, it should receive non-zero attribution.

**Implementation Invariance:** Two models that produce identical outputs for all inputs should receive identical attributions.

Saliency fails sensitivity (saturation problem). Guided Backprop fails both (architecture-dependent).

The question: can we construct an attribution method satisfying both axioms? Sundararajan et al. answered yes, and proved that Integrated Gradients is the *unique* such method in a natural function class.

---

## 2. The Integration Idea

### From Approximation to Exactness
<div class="callout-warning">

<strong>Warning:</strong> For a linear model $f(x) = w^T x$, the exact decomposition is:

</div>


For a linear model $f(x) = w^T x$, the exact decomposition is:

$$f(x) - f(0) = \sum_i w_i x_i = \sum_i x_i \frac{\partial f}{\partial x_i}$$

For non-linear models, Input×Gradient provides only a first-order approximation:

$$f(x) - f(x') \approx \sum_i (x_i - x'_i) \frac{\partial f(x)}{\partial x_i}$$

This approximation is exact for linear functions but wrong for non-linear ones.

### The Path Integral

Instead of evaluating the gradient at one point, integrate it along the straight-line path from baseline $x'$ to input $x$:

$$\text{IG}_i(x) = (x_i - x'_i) \int_0^1 \frac{\partial f(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$

This integral is exact — not an approximation — by the Fundamental Theorem of Calculus.

### Connection to FTC

The FTC states that for a function $g: [0,1] \rightarrow \mathbb{R}$:

$$\int_0^1 g'(\alpha) d\alpha = g(1) - g(0)$$

Let $g(\alpha) = f(x' + \alpha(x - x'))$. Then:

$$g(1) = f(x), \quad g(0) = f(x')$$

$$g'(\alpha) = \sum_i \frac{\partial f(x' + \alpha(x - x'))}{\partial x_i} \cdot (x_i - x'_i)$$

Therefore:

$$f(x) - f(x') = \int_0^1 \sum_i \frac{\partial f(x' + \alpha(x - x'))}{\partial x_i} \cdot (x_i - x'_i) d\alpha$$

$$= \sum_i (x_i - x'_i) \int_0^1 \frac{\partial f(x' + \alpha(x - x'))}{\partial x_i} d\alpha = \sum_i \text{IG}_i(x)$$

**The sum of attributions equals exactly $f(x) - f(x')$.** This is the completeness (conservation) property.

---

## 3. Formal Axioms

### Axiom 1: Sensitivity

For any two inputs $x$ and $x'$ that differ in exactly one feature $i$, and where $f(x) \neq f(x')$:
<div class="callout-key">

<strong>Key Point:</strong> For any two inputs $x$ and $x'$ that differ in exactly one feature $i$, and where $f(x) \neq f(x')$:

</div>


$$\phi_i(x, x') \neq 0$$

**IG satisfies this:** if changing feature $i$ from $x'_i$ to $x_i$ changes the output, then the integral $\int_0^1 \frac{\partial f}{\partial x_i}$ along the path cannot be identically zero.

### Axiom 2: Implementation Invariance

For any two models $f$ and $g$ with $f(x) = g(x)$ for all $x$:

$$\text{IG}_i^f(x, x') = \text{IG}_i^g(x, x')$$

**IG satisfies this:** IG depends only on the model's input-output behavior (via gradients), not on how the computation is internally organized. Two models with identical outputs have identical gradients, hence identical attributions.

### Derived Properties

**Completeness (Conservation):**
$$\sum_i \text{IG}_i(x) = f(x) - f(x')$$

This is the practical validation check — if the sum of attributions does not equal $f(x) - f(x')$, the numerical approximation has too much error.

**Linearity:**
If $f = a \cdot f_1 + b \cdot f_2$, then $\text{IG}_i^f = a \cdot \text{IG}_i^{f_1} + b \cdot \text{IG}_i^{f_2}$.

**Dummy:**
If $f$ does not depend on feature $i$ at all, then $\text{IG}_i = 0$.

**Symmetry:**
If swapping features $i$ and $j$ does not change the model (symmetric in $i$ and $j$), then $\text{IG}_i = \text{IG}_j$.

---

## 4. Numerical Approximation

The integral is approximated by a Riemann sum with $m$ steps:
<div class="callout-insight">

<strong>Insight:</strong> The integral is approximated by a Riemann sum with $m$ steps:

</div>


$$\text{IG}_i^{\text{approx}}(x) \approx (x_i - x'_i) \cdot \frac{1}{m} \sum_{k=1}^{m} \frac{\partial f\left(x' + \frac{k}{m}(x - x')\right)}{\partial x_i}$$

### Algorithm

```

1. Choose baseline x' (zero image, blurred, etc.)
2. Generate m interpolation points:
   x^(k) = x' + (k/m) * (x - x'),  k = 1, ..., m
3. For each x^(k), compute ∂f/∂x_i via backpropagation
4. Average gradients: (1/m) * sum_k grad_i(x^(k))
5. Multiply by (x - x') to get attribution
```

### Convergence

The approximation error is:

$$\left| \text{IG}_i(x) - \text{IG}_i^{\text{approx}}(x) \right| \leq O(1/m)$$

The error decreases with $m$ (linearly for a standard Riemann sum; quadratically if using the trapezoidal rule). In practice:
- $m = 20$: fast, adequate for visual inspection
- $m = 50$: standard quality (Captum default)
- $m = 300$: high quality, recommended for validation

**The convergence delta** measures the total approximation error:

$$\delta = \sum_i \text{IG}_i^{\text{approx}}(x) - (f(x) - f(x'))$$

In Captum: `return_convergence_delta=True` returns this delta. A value near 0 indicates a good approximation.

---

## 5. Uniqueness of Integrated Gradients

Sundararajan et al. prove that among all attribution methods satisfying the sensitivity axiom, implementation invariance, linearity, and a few technical regularity conditions, the **only** valid attribution is Integrated Gradients along any path from $x'$ to $x$.

The straight-line path is not the only valid path — any path from $x'$ to $x$ produces a valid attribution. The straight-line path is the most natural choice and is what Captum implements.

Other paths produce different but equally valid attributions. This motivates the **Integrated Hessians** extension which uses a 2D path, and the **Expected Gradients** (SHAP's deep explainer) which averages over multiple baseline samples.

---

## 6. Interpreting IG Attributions

### Sign Interpretation
<div class="callout-key">

<strong>Key Point:</strong> - **Positive attribution** $(\text{IG}_i > 0)$: feature $i$ contributed positively to $f(x)$ relative to the baseline

</div>


- **Positive attribution** $(\text{IG}_i > 0)$: feature $i$ contributed positively to $f(x)$ relative to the baseline
- **Negative attribution** $(\text{IG}_i < 0)$: feature $i$ contributed negatively to $f(x)$ relative to the baseline
- **Zero attribution**: feature $i$ made no contribution along the path

### Scale Interpretation

The scale is meaningful: attribution values sum to $f(x) - f(x')$. If the model output on the baseline is 0.1 and on the input is 0.9, the total attribution sums to 0.8. Individual attributions can be read as fractional contributions.

### The Role of the Baseline

The baseline $x'$ defines "no information" — the reference point from which contributions are measured. Attribution measures: "compared to having no information, how much did feature $i$ contribute?"

Different baselines answer different questions:
- **Black image baseline** ($x' = 0$): "which pixels contributed compared to a black image?"
- **Blurred baseline**: "which local patterns contributed beyond the blurred background?"
- **Mean training sample**: "how does this input differ from the average input?"

---

## 7. Comparison with Related Methods

### IG vs Input × Gradient

| Property | Input × Gradient | Integrated Gradients |
|----------|-----------------|---------------------|
| Computation | 1 gradient | m gradients |
| Correctness | First-order approx | Exact (FTC) |
| Sensitivity | No | Yes |
| Completeness | Approximate | Exact |
| Baseline required | No (implicit: 0) | Yes (explicit) |

### IG vs SHAP

Shapley values and Integrated Gradients are related but distinct:

- **Shapley values** distribute the prediction among all features using game-theoretic fairness axioms (efficiency, symmetry, dummy, linearity)
- **IG** satisfies a subset of these (linearity, dummy, plus sensitivity and implementation invariance)
- **Expected Gradients** (used in DeepSHAP/SHAP library) averages IG over a distribution of baselines, approximating Shapley values

For tabular data with independent features, SHAP provides stronger game-theoretic guarantees. For images and text, IG is more natural and computationally efficient.

---

## Common Pitfalls

- **Too few steps:** $m = 5$ is often insufficient. Always check convergence delta.
- **Wrong baseline:** Using a baseline that is informative (e.g., a blurred version of the same image) can artificially inflate or deflate attributions.
- **Interpreting attribution without baseline context:** "$x_i$ has large IG" means it contributes relative to the baseline, not in an absolute sense.
- **Ignoring negative attributions:** Negative attributions are as informative as positive ones — they show which features oppose the predicted class.

---

## Connections

- **Builds on:** Module 01 (gradient theory and saturation failure), Fundamental Theorem of Calculus
- **Leads to:** Guide 02 (baseline choices, convergence), Module 02 notebooks
- **Related to:** SHAP (game-theoretic perspective), path methods in general

---


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Key Insight" and why it matters in practice.

2. Given a real-world scenario involving integrated gradients: theory and axiomatic derivation, what would be your first three steps to apply the techniques from this guide?

</div>

## Further Reading

- Sundararajan, Taly & Yan (2017). Axiomatic Attribution for Deep Networks. *ICML 2017* — The foundational paper. Required reading.
- Shrikumar et al. (2017). Learning Important Features Through Propagating Activation Differences. *ICML 2017* — DeepLIFT, closely related to IG.
- Erion et al. (2021). Improving Performance of Deep Learning Models with Axiomatic Attribution Priors and Expected Gradients. *Nature Machine Intelligence* — Expected Gradients (SHAP connection to IG).
- Kapishnikov et al. (2021). Guided Integrated Gradients. *CVPR 2021* — Path choice and guided variants.

---

## Cross-References

<a class="link-card" href="../notebooks/01_ig_image_classification.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
