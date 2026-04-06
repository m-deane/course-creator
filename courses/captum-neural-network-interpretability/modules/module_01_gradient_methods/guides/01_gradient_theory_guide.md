# Gradient-Based Attribution Methods: Theory

> **Reading time:** ~8 min | **Module:** 1 — Gradient Methods | **Prerequisites:** Module 0 Foundations


## In Brief

Gradient-based attribution methods use the partial derivatives of a neural network's output with respect to its input to measure feature importance. They are the fastest attribution methods (single backward pass) and form the foundation for more principled methods like Integrated Gradients.

## Key Insight

The gradient $\frac{\partial f}{\partial x_i}$ tells you: "if I increase input feature $x_i$ slightly, how much does the output change?" This local sensitivity is an intuitive measure of feature importance — but it has well-documented failure modes that motivate the methods in Module 02.


<div class="callout-key">

<strong>Key Concept Summary:</strong> Gradient-based attribution methods use the partial derivatives of a neural network's output with respect to its input to measure feature importance.

</div>

---

## 1. Vanilla Gradients (Saliency Maps)

### Definition
<div class="callout-insight">

<strong>Insight:</strong> Given a model $f: \mathbb{R}^d \rightarrow \mathbb{R}$ and an input $x$, the **saliency attribution** for feature $i$ is:

</div>


Given a model $f: \mathbb{R}^d \rightarrow \mathbb{R}$ and an input $x$, the **saliency attribution** for feature $i$ is:

$$\phi_i^{\text{saliency}}(x) = \left| \frac{\partial f(x)}{\partial x_i} \right|$$

The absolute value is taken to measure *magnitude* of sensitivity, regardless of direction.

### Intuition

The gradient is the direction of steepest ascent in the output surface. Features with large gradients are those where the model's output is most sensitive to small changes — intuitively, the model is "paying attention" to these features.

### Implementation

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>
<div class="code-body">

```python
import torch
from captum.attr import Saliency

model.eval()
saliency = Saliency(model)

# Input must require gradients
input_tensor = preprocess(image).unsqueeze(0).requires_grad_(True)

# attr_method: use abs (absolute_value) for magnitude-only visualization
attributions = saliency.attribute(input_tensor, target=class_idx)
# attributions.shape == input_tensor.shape
# Each value = |∂f/∂x_i|
```

</div>

</div>

### The Gradient Computation (Manual)

Understanding what Captum does internally:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>
<div class="code-body">

```python
# Captum's Saliency is equivalent to:
input_tensor.requires_grad_(True)
output = model(input_tensor)
output[0, class_idx].backward()  # Compute ∂f/∂x
gradients = input_tensor.grad.abs()  # Take absolute value
```

</div>

</div>

### Known Failure: Saturation

The most important failure of saliency maps is **gradient saturation**. ReLU networks have large flat regions where the gradient is exactly zero, even for inputs that strongly influence the output.

Consider a ReLU with input $x = 3$: the output is 3, but the gradient is 1. However, if we change to input $x = 5$ vs $x = 3$, the output changes — the feature *is* relevant. The gradient at $x = 3$ correctly reports sensitivity, but if the network has "already saturated" (e.g., a feature was clamped), the gradient is zero despite relevance.

This violates the sensitivity axiom: a feature can be relevant (changing it changes the output) but receive zero attribution from saliency.

---

## 2. Input × Gradient

### Definition

The **Input × Gradient** method multiplies each gradient by the corresponding input value:
<div class="callout-warning">

<strong>Warning:</strong> The **Input × Gradient** method multiplies each gradient by the corresponding input value:

</div>


$$\phi_i^{\text{I×G}}(x) = x_i \cdot \frac{\partial f(x)}{\partial x_i}$$

### Motivation

Pure gradients measure *local sensitivity*. Multiplying by the input value creates a measure of "how much the input feature actually contributes" — a feature with zero value contributes nothing to the output even if the gradient is large.

Think of it as: gradient tells you the rate of change, input tells you the magnitude. Their product approximates the contribution.

### Mathematical Perspective

Input×Gradient is a first-order Taylor approximation of the difference between the model output at the input and at the zero baseline:

$$f(x) - f(0) \approx \sum_i x_i \cdot \frac{\partial f(x)}{\partial x_i} = \sum_i \phi_i^{\text{I×G}}(x)$$

This approximation is exact for linear models and approximate for non-linear models. For Integrated Gradients (Module 02), this approximation becomes exact by integrating over all intermediate points.

### Implementation

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>
<div class="code-body">

```python
from captum.attr import InputXGradient

ixg = InputXGradient(model)
attributions = ixg.attribute(input_tensor, target=class_idx)
# Each value = x_i * ∂f/∂x_i
```

</div>

</div>

### Known Failure: Input Dependency

Input×Gradient inherits saliency's saturation problem. Additionally, because it multiplies by the input value, it can produce large attributions for large-valued inputs even when the model ignores them — and zero attributions for small-valued inputs that the model relies on heavily.

---

## 3. Guided Backpropagation

### Definition

Guided Backpropagation (Springenberg et al., 2014) modifies the standard gradient backpropagation through ReLU layers. During the backward pass, a standard gradient propagates negative values through ReLUs (zeroing them out). Guided Backprop additionally zeroes out gradients in the backward pass where the *gradient itself* is negative:
<div class="callout-key">

<strong>Key Point:</strong> Guided Backpropagation (Springenberg et al., 2014) modifies the standard gradient backpropagation through ReLU layers. During the backward pass, a standard gradient propagates negative values through ReLUs (zeroing them out). Guided Backprop additionally zeroes out gradients in the backward pass where the *gradient itself* is negative:

</div>


**Standard ReLU backward:**
$$\delta^l = \frac{\partial f}{\partial h^l} \cdot \mathbf{1}[h^l > 0]$$

**Guided Backprop backward:**
$$\delta^l = \frac{\partial f}{\partial h^l} \cdot \mathbf{1}[h^l > 0] \cdot \mathbf{1}\left[\frac{\partial f}{\partial h^l} > 0\right]$$

The second indicator clips negative gradients in the backward pass.

### Effect

This modification produces cleaner, sharper visualizations by removing gradient noise. Guided Backprop visualizations often look more like the object itself rather than a diffuse heatmap.

### Implementation

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>
<div class="code-body">

```python
from captum.attr import GuidedBackprop

gbp = GuidedBackprop(model)
attributions = gbp.attribute(input_tensor, target=class_idx)
```

</div>

</div>

### Critical Failure: Not Attribution

Adebayo et al. (2018) — "Sanity Checks for Saliency Maps" — demonstrated that Guided Backprop produces nearly identical visualizations for:
- A fully trained model
- A randomly initialized model with the same architecture

This means Guided Backprop visualizations reflect the **network architecture** (specifically, the pattern of ReLU nonlinearities), not the **model's learned weights**. The visualizations are visually appealing but do not represent attribution.

This violates implementation invariance: two models with different weights (one trained, one random) produce similar attributions, which is impossible if the attribution is measuring model behavior.

**Practical implication:** Guided Backprop is useful for visualizing what an architecture *can* represent, but is not a reliable attribution method. Do not use it for model debugging or validation.

---

## 4. Deconvolution

### Definition

Deconvolution (Zeiler & Fergus, 2014) is similar to Guided Backpropagation but uses a different rule: during backward pass through ReLU, deconvolution zeroes out negative gradient values but NOT based on the forward activation.
<div class="callout-insight">

<strong>Insight:</strong> Deconvolution (Zeiler & Fergus, 2014) is similar to Guided Backpropagation but uses a different rule: during backward pass through ReLU, deconvolution zeroes out negative gradient values but NOT based on the forward activation.

</div>


**Deconvolution backward:**
$$\delta^l = \frac{\partial f}{\partial h^l} \cdot \mathbf{1}\left[\frac{\partial f}{\partial h^l} > 0\right]$$

(No condition on $h^l > 0$, only on the gradient sign.)

### Comparison with Guided Backprop

| Rule | Condition |
|------|-----------|
| Standard gradient | $h^l > 0$ (forward activation) |
| Deconvolution | $\frac{\partial f}{\partial h^l} > 0$ (gradient positive) |
| Guided Backprop | Both conditions must hold |

### Known Issue

Like Guided Backprop, deconvolution fails implementation invariance and is not a principled attribution method. It suffers from the same architectural dependence described above.

### Implementation

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>
<div class="code-body">

```python
from captum.attr import Deconvolution

deconv = Deconvolution(model)
attributions = deconv.attribute(input_tensor, target=class_idx)
```

</div>

</div>

---

## 5. Side-by-Side Method Comparison

### What Each Method Computes
<div class="callout-warning">

<strong>Warning:</strong> None of these methods satisfies both axioms. This motivates Integrated Gradients.

</div>


| Method | Formula | Key Property |
|--------|---------|--------------|
| Saliency | $\|{\nabla_x f(x)}\|$ | Local sensitivity magnitude |
| Input×Gradient | $x \odot \nabla_x f(x)$ | Input-weighted sensitivity |
| Guided Backprop | Modified $\nabla$ | Zeroes negative gradients |
| Deconvolution | Modified $\nabla$ | Zeroes negative gradient flow |

### Axiom Compliance

| Method | Sensitivity | Impl. Invariance | Completeness |
|--------|------------|-----------------|--------------|
| Saliency | No | Yes | No |
| Input×Gradient | No | Yes | Approx (linear approx) |
| Guided Backprop | No | **No** | No |
| Deconvolution | No | **No** | No |

None of these methods satisfies both axioms. This motivates Integrated Gradients.

### When to Use Which

- **Saliency:** Quick debugging, fastest computation, when only relative magnitudes matter
- **Input×Gradient:** When input scale matters; better localization than saliency for contrast
- **Guided Backprop:** When you need a clean visual for presentation (but NOT for validation — results are architecture-dependent)
- **Deconvolution:** Historical reference; rarely preferred over Guided Backprop in modern practice

---

## 6. The Noisy Gradient Problem

Gradients of deep networks are notoriously noisy. The gradient at a specific input is highly sensitive to local perturbations, producing "salt and pepper" noise in saliency maps.
<div class="callout-key">

<strong>Key Point:</strong> Gradients of deep networks are notoriously noisy. The gradient at a specific input is highly sensitive to local perturbations, producing "salt and pepper" noise in saliency maps.

</div>


### Illustration

For a small perturbation $\epsilon$, the gradient at $x$ and $x + \epsilon$ can be very different:

$$\left|\frac{\partial f(x)}{\partial x_i} - \frac{\partial f(x + \epsilon)}{\partial x_i}\right| \gg 0 \text{ for small } \|\epsilon\|$$

This is especially problematic near ReLU boundaries where the gradient is discontinuous.

### SmoothGrad

SmoothGrad (Smilkov et al., 2017) addresses this by averaging gradients over inputs perturbed with Gaussian noise:

$$\phi_i^{\text{SmoothGrad}}(x) = \frac{1}{n} \sum_{k=1}^n \frac{\partial f(x + \epsilon_k)}{\partial x_i}, \quad \epsilon_k \sim \mathcal{N}(0, \sigma^2 I)$$

In Captum, this is implemented via `NoiseTunnel`:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>
<div class="code-body">

```python
from captum.attr import Saliency, NoiseTunnel

saliency = Saliency(model)
nt = NoiseTunnel(saliency)

smooth_attr = nt.attribute(
    input_tensor,
    nt_type='smoothgrad',
    nt_samples=20,
    stdevs=0.1,
    target=class_idx
)
```

</div>

</div>

SmoothGrad is covered in depth in Module 02 (Notebook 03).

---

## Common Pitfalls

- **Confusing saliency with importance:** Saliency measures local sensitivity, not global importance. A feature with small gradient can still be important if the model is far from saturation.
- **Trusting Guided Backprop for validation:** Adebayo et al. (2018) showed it is architecture-dependent, not weight-dependent.
- **Not checking signs:** Taking absolute value discards the direction of attribution (positive = increases prediction, negative = decreases it).
- **Comparing across images:** Gradient magnitude scales vary between images. Normalize before comparing.

---

## Connections

- **Builds on:** Neural network backpropagation, chain rule, ReLU activation
- **Leads to:** Integrated Gradients (Module 02) — the path-integral solution to gradient limitations
- **Related to:** SmoothGrad, VarGrad, noise-based variance reduction

---


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Key Insight" and why it matters in practice.

2. Given a real-world scenario involving gradient-based attribution methods: theory, what would be your first three steps to apply the techniques from this guide?

</div>

## Further Reading

- Simonyan et al. (2014). Deep Inside Convolutional Networks: Visualising Image Classification Models. *arXiv* — Original saliency maps paper.
- Springenberg et al. (2014). Striving for Simplicity: The All Convolutional Net. *ICLR Workshop* — Guided Backpropagation.
- Zeiler & Fergus (2014). Visualizing and Understanding Convolutional Networks. *ECCV 2014* — Deconvolution method.
- Adebayo et al. (2018). Sanity Checks for Saliency Maps. *NeurIPS 2018* — Critical evaluation showing Guided Backprop is architecture-dependent.
- Smilkov et al. (2017). SmoothGrad: removing noise by adding noise. *arXiv* — Noise averaging for cleaner gradient visualizations.

---

## Cross-References

<a class="link-card" href="../notebooks/01_gradient_methods_cnn.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
