# GradientSHAP and DeepLIFT SHAP

> **Reading time:** ~8 min | **Module:** 5 — SHAP Methods | **Prerequisites:** Module 4 Perturbation Methods


## Learning Objectives

By the end of this guide, you will be able to:
1. Explain how GradientSHAP combines Integrated Gradients with SHAP's baseline expectation
2. Describe DeepLIFT's propagation rules and how they differ from backpropagation
3. Implement DeepLIFT and DeepLIFTSHAP in Captum for any differentiable model
4. Identify when to prefer propagation-based methods over KernelSHAP
5. Interpret the differences between gradient and propagation attributions


<div class="callout-key">

<strong>Key Concept Summary:</strong> This guide covers the core concepts of gradientshap and deeplift shap.

</div>

---

## 1. The Speed-Theory Trade-off

KernelSHAP satisfies all four Shapley axioms for any black-box model but requires $O(M \cdot n_{bg})$ model evaluations, where $M$ is the number of sampled coalitions. For neural networks with millions of parameters and image inputs, this is prohibitively slow.

GradientSHAP and DeepLIFT SHAP exploit the differentiable structure of neural networks to compute approximate Shapley values with far fewer model evaluations — often just one forward/backward pass per baseline sample.

---

## 2. GradientSHAP: Integrated Gradients Meets SHAP

### The Core Idea
<div class="callout-insight">

<strong>Insight:</strong> Integrated Gradients (IG) computes attributions for a single baseline $x'$:

</div>


Integrated Gradients (IG) computes attributions for a single baseline $x'$:

$$\text{IG}_i(x) = (x_i - x'_i) \int_0^1 \frac{\partial f(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$

GradientSHAP extends this to an **expectation over multiple baselines**:

$$\phi_i^{\text{GradSHAP}}(x) = \mathbb{E}_{x' \sim \mathcal{D}_{bg}, \alpha \sim U(0,1)} \left[ \frac{\partial f(x' + \alpha(x - x'))}{\partial x_i} \cdot (x_i - x'_i) \right]$$

where $\mathcal{D}_{bg}$ is the distribution of baseline samples.

### Connection to Shapley Values

Lundberg and Lee (2017) showed that for models $f$ where:
$$\mathbb{E}_{x' \sim \mathcal{D}_{bg}}[\text{IG}(x, x')] \approx \phi^{\text{SHAP}}(x)$$

This approximation holds exactly when features are independent, and practically well when the background distribution approximates the marginal.

### Monte Carlo Estimation

In practice, GradientSHAP:
1. Samples $n$ baselines $x'_1, \ldots, x'_n$ from the background distribution
2. For each baseline, samples a scalar $\alpha_k \sim U(0,1)$
3. Evaluates the gradient at the interpolated point $x'_k + \alpha_k(x - x'_k)$
4. Returns the average: $\frac{1}{n}\sum_k \nabla_{x_i} f(x'_k + \alpha_k(x - x'_k)) \cdot (x_i - x'_{k,i})$

---

## 3. GradientSHAP in Captum


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import torch
from captum.attr import GradientShap

model.eval()
grad_shap = GradientShap(model)

# Background: multiple representative samples
background = X_train[:100]  # shape: (100, num_features)

# Test input
x_test = X_test[[0]]  # shape: (1, num_features)

# Compute attributions
attributions, delta = grad_shap.attribute(
    inputs=x_test,
    baselines=background,
    target=predicted_class,
    n_samples=50,        # number of Monte Carlo samples
    stdevs=0.0,          # optional Gaussian noise for SmoothGrad variant
    return_convergence_delta=True,
)

# delta: convergence error (should be close to 0)
print(f"Convergence delta: {delta.mean().item():.4f}")
```

</div>

</div>

The `return_convergence_delta=True` argument returns the approximation error — a useful quality check.

### Convergence Delta Interpretation

The convergence delta measures how well the attributions satisfy the efficiency axiom:

$$\delta = \left| \sum_i \phi_i - (f(x) - \mathbb{E}[f(X')]) \right|$$

A delta close to 0 indicates reliable attributions. Large delta values suggest increasing `n_samples`.

---

## 4. DeepLIFT: Propagation-Based Attribution

### Motivation
<div class="callout-key">

<strong>Key Point:</strong> Gradient-based methods have a fundamental problem: when a network saturates (outputs near 0 or 1), gradients vanish even though the saturated feature clearly influences the prediction. DeepLIFT was designed to overcome this.

</div>


Gradient-based methods have a fundamental problem: when a network saturates (outputs near 0 or 1), gradients vanish even though the saturated feature clearly influences the prediction. DeepLIFT was designed to overcome this.

### DeepLIFT's Reference Activation Difference

DeepLIFT (Deep Learning Important FeaTures) defines attributions using **activation differences** rather than gradients:

$$\Delta x_i = x_i - x'_i \quad \text{(input difference from reference)}$$

$$\Delta h_l = h_l(x) - h_l(x') \quad \text{(activation difference at layer } l\text{)}$$

The DeepLIFT attribution $C_{\Delta x_i \Delta y}$ is defined via the **summation-to-delta** rule:

$$\sum_i C_{\Delta x_i \Delta y} = \Delta y = f(x) - f(x')$$

This is similar to the efficiency axiom but uses a single reference $x'$ rather than an expectation.

### Propagation Rules

DeepLIFT backpropagates through the network using modified rules:

**Linear rule** (for linear layers):
$$C_{\Delta x_i \Delta t} = \Delta x_i \cdot w_i$$

where $w_i$ is the weight connecting $x_i$ to neuron $t$.

**Rescale rule** (for nonlinear activations):
$$C_{\Delta x_i \Delta t} = \frac{\Delta x_i}{\Delta t} \cdot C_{\Delta x_i \Delta y}$$

This rescales contributions proportionally to the activation difference, avoiding vanishing gradients.

**RevealCancel rule** (for neurons receiving both positive and negative inputs):
Separates positive and negative contributions to prevent cancellation artifacts.

---

## 5. DeepLIFT vs. Gradient: The Saturation Example

Consider a sigmoid activation with:
- Input $x = 5.0$ (deeply saturated)
- Reference $x' = 0.0$ (neutral point)
- $\sigma(5) \approx 0.993$, $\sigma(0) = 0.5$


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Gradient at saturated input: near zero (vanishing gradient)
gradient = sigma(5) * (1 - sigma(5)) ≈ 0.007
gradient_attribution = gradient * (5 - 0) = 0.035

# DeepLIFT: uses activation difference
delta_y = sigma(5) - sigma(0) = 0.993 - 0.5 = 0.493
deeplift_attribution = delta_y = 0.493  # much larger, more meaningful
```

</div>

</div>

DeepLIFT correctly attributes the large change in output to the input difference, while the gradient essentially reports "nothing changed" because of saturation.

---

## 6. DeepLIFT in Captum


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import torch
from captum.attr import DeepLift

model.eval()
deep_lift = DeepLift(model)

# Single reference baseline (e.g., zero or mean of training data)
baseline = torch.zeros(1, num_features)

# Alternative: use mean baseline

# baseline = X_train.mean(dim=0, keepdim=True)

x_test = X_test[[0]]

# Compute attributions
attributions, delta = deep_lift.attribute(
    inputs=x_test,
    baselines=baseline,
    target=predicted_class,
    return_convergence_delta=True,
)

print(f"Attributions: {attributions}")
print(f"Convergence delta: {delta.item():.6f}")  # should be ~0
```

</div>

</div>

**Important:** DeepLIFT requires the model to be a standard sequential network or use hooks-compatible architecture. Custom activation functions may require additional configuration.

---

## 7. DeepLIFT SHAP: Combining Both Approaches

DeepLIFT SHAP combines:
- DeepLIFT's efficient propagation (no coalition sampling)
- SHAP's expectation over multiple baselines (approaches Shapley values)

$$\phi_i^{\text{DeepLIFT-SHAP}} = \frac{1}{n_{bg}} \sum_{k=1}^{n_{bg}} C_{\Delta x_i \Delta y}(x, x'_k)$$

This runs DeepLIFT once per background sample, then averages. For $n_{bg} = 50$ background samples, this requires 50 forward-backward passes — much cheaper than KernelSHAP's hundreds of passes but more expensive than single-baseline DeepLIFT.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
from captum.attr import DeepLiftShap

model.eval()
deep_lift_shap = DeepLiftShap(model)

# Multiple background baselines
background = X_train[:50]  # shape: (50, num_features)

attributions, delta = deep_lift_shap.attribute(
    inputs=x_test,
    baselines=background,
    target=predicted_class,
    return_convergence_delta=True,
)
```

</div>

</div>

---

## 8. Handling Custom Architectures

DeepLIFT uses custom backpropagation hooks and may encounter issues with:

### Unsupported Operations
Captum raises `TorchFunctionalAPIError` for operations like `torch.einsum`, `F.gelu`, or custom layers.

**Solution:** Use `GradientShap` instead, which uses standard autograd.

### Batch Normalization
BatchNorm statistics differ between training and eval mode. Always ensure `model.eval()` before running DeepLIFT.

### Residual Connections
Residual connections in ResNets/Transformers require the `DeepLiftShap` variant which handles skip connections more robustly.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# For models with residual connections
from captum.attr import DeepLiftShap

# DeepLiftShap handles more complex architectures
attrs = DeepLiftShap(model).attribute(
    x, baselines=background, target=target
)
```

</div>

</div>

---

## 9. GradientSHAP vs DeepLIFT SHAP: When to Use Which

| Criterion | GradientSHAP | DeepLIFT SHAP |
|-----------|-------------|---------------|
| Architecture compatibility | Any differentiable model | Standard layers only |
| Speed (n baselines) | O(n) gradient computations | O(n) forward+backward |
| Saturation handling | No (gradient vanishes) | Yes (rescale rule) |
| Convergence delta | Approximate | Exact (efficiency guaranteed) |
| Custom activations | Always works | May require workarounds |
| Theoretical basis | IG + expectation | DeepLIFT + expectation |

**Rule of thumb:**
- Use **DeepLIFT SHAP** when you have standard architectures (ReLU, sigmoid) and need exact efficiency
- Use **GradientSHAP** when you have custom layers, GELU activations, attention mechanisms, or Transformers

---

## 10. Visualizing GradientSHAP and DeepLIFT Outputs

### For Tabular Data


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import matplotlib.pyplot as plt
import numpy as np

feature_names = ["income", "age", "debt_ratio", "employment", "credit_score"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (attrs, title) in zip(axes, [
    (grad_shap_attrs.squeeze().numpy(), "GradientSHAP"),
    (deeplift_attrs.squeeze().numpy(), "DeepLIFT SHAP"),
]):
    colors = ["#d73027" if a > 0 else "#4575b4" for a in attrs]
    bars = ax.barh(feature_names, attrs, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Attribution value (φᵢ)")
    ax.set_title(title)

plt.tight_layout()
plt.savefig("gradient_vs_deeplift_attributions.png", dpi=150, bbox_inches="tight")
```

</div>


### For Images


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
from captum.attr import visualization as viz
import numpy as np

# Convert attributions to numpy for visualization
attrs_np = attributions.squeeze().permute(1, 2, 0).detach().numpy()
original_np = x_test.squeeze().permute(1, 2, 0).detach().numpy()

# Use Captum's built-in visualization
fig, _ = viz.visualize_image_attr_multiple(
    attrs_np,
    original_np,
    methods=["original_image", "heat_map", "masked_image"],
    signs=["all", "all", "positive"],
    titles=["Original", "GradientSHAP", "Positive Regions"],
    show_colorbar=True,
)
```



---

## 11. Numerical Comparison

When comparing methods, always check that attributions are on the same scale:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python

# Normalize by L2 norm for fair comparison
def normalize_attrs(attrs):
    norm = attrs.norm().item()
    return attrs / norm if norm > 0 else attrs

attrs_ks_norm = normalize_attrs(attrs_kernel_shap)
attrs_gs_norm = normalize_attrs(attrs_gradient_shap)
attrs_dl_norm = normalize_attrs(attrs_deeplift_shap)

# Correlation between methods
from scipy.stats import spearmanr

corr_gs_dl, _ = spearmanr(attrs_gs_norm.flatten(), attrs_dl_norm.flatten())
print(f"GradientSHAP vs DeepLIFT SHAP (Spearman r): {corr_gs_dl:.3f}")
```



High correlation ($r > 0.9$) between methods on the same input indicates robust, model-supported attributions.

---


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "The Speed-Theory Trade-off" and why it matters in practice.

2. Given a real-world scenario involving gradientshap and deeplift shap, what would be your first three steps to apply the techniques from this guide?


## Summary

| Method | Formula | Model Requirements | Key Strength |
|--------|---------|-------------------|--------------|
| GradientSHAP | $\mathbb{E}_{x',\alpha}[\nabla f \cdot (x - x')]$ | Any differentiable | Works everywhere |
| DeepLIFT | Propagation rules using $\Delta h$ | Standard layers | Handles saturation |
| DeepLIFT SHAP | Average DeepLIFT over backgrounds | Standard layers | Efficiency guarantee |

All three methods:
- Are faster than KernelSHAP for neural networks
- Require a background distribution or reference point
- Produce attributions approximately aligned with Shapley values

---

## Further Reading

- Shrikumar, A., et al. (2017). Learning important features through propagating activation differences. *ICML*.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Ancona, M., et al. (2018). Towards better understanding of gradient-based attribution methods for deep neural networks. *ICLR*.
- Captum DeepLift documentation: https://captum.ai/api/deep_lift.html

---

## Cross-References

<a class="link-card" href="../notebooks/01_kernelshap_vs_gradientshap.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
