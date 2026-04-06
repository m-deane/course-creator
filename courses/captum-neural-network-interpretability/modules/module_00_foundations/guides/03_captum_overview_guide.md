# Captum Library Architecture and Philosophy

> **Reading time:** ~10 min | **Module:** 0 — Foundations | **Prerequisites:** PyTorch basics, neural network fundamentals


## In Brief

Captum is PyTorch's official interpretability library, developed by Facebook AI Research. It provides a unified, extensible framework for attribution methods that works on any PyTorch model. This guide covers the library's design philosophy, API conventions, and comparison with alternatives.

## Key Insight

Captum's design principle — every method implements a single `.attribute()` interface — makes it trivial to swap methods and run comparisons. This unified interface is more valuable than any individual method it contains.


<div class="callout-key">
<strong>Key Concept Summary:</strong> Captum is PyTorch's official interpretability library, developed by Facebook AI Research.
</div>

---

## 1. Library Philosophy and Design Goals

Captum (from Latin: "to perceive, understand") was released in 2019 and is maintained by Meta AI Research. Its design reflects four core principles:

**Unified interface:** Every attribution method, regardless of its underlying algorithm, accepts the same inputs and returns attributions in the same format. You can replace `IntegratedGradients` with `Saliency` with a one-word change.

**Composability:** Methods can be combined. `NoiseTunnel` wraps any method with variance reduction. `LayerIntegratedGradients` combines Integrated Gradients with layer attribution. New methods can be built on existing primitives.

**Native PyTorch:** No separate computation graph. Captum uses PyTorch's autograd system directly, which means it works with dynamic graphs, custom layers, and any model that has a `.forward()` method.

**Attribution contract:** Captum enforces a consistent contract for what an attribution means. All methods return tensors of the same shape as the input (or layer), representing contribution scores.

---

## 2. The Core API

### Basic Attribution Pattern

Every Captum method follows this pattern:
<div class="callout-warning">
<strong>Warning:</strong> Every Captum method follows this pattern:
</div>


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
from captum.attr import IntegratedGradients

# 1. Instantiate the method with your model
ig = IntegratedGradients(model)

# 2. Compute attributions with .attribute()
attributions = ig.attribute(
    inputs,          # The input tensor requiring explanation
    baselines=baseline,  # Reference point (method-dependent)
    target=class_idx     # Which output to explain
)

# 3. attributions has the same shape as inputs
assert attributions.shape == inputs.shape
```

</div>
</div>

The `attribute()` method signature varies slightly between methods, but the pattern is always:
1. Create method instance with model
2. Call `.attribute(inputs, ...)`
3. Receive tensor of same shape as input

### Common Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `inputs` | Tensor | Input to explain. Must require grad for gradient methods. |
| `baselines` | Tensor or None | Reference point representing "no information" |
| `target` | int or Tensor | Output neuron to explain (for classification: the class index) |
| `additional_forward_args` | tuple | Extra arguments passed to `model.forward()` |
| `return_convergence_delta` | bool | For IG: return the approximation error |
| `n_steps` | int | For IG: number of integral approximation steps |

### Preparing Inputs

Captum requires inputs to have gradients enabled. The standard pattern:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
import torch
from torchvision import transforms
from PIL import Image

# Load and preprocess image
image = Image.open("image.jpg")
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

# Enable gradients — required by Captum
input_tensor = input_tensor.requires_grad_(True)
```

</div>
</div>

---

## 3. Method Families in Captum

### Primary Attribution Methods
<div class="callout-key">
<strong>Key Point:</strong> from captum.attr import Saliency              # Vanilla gradient magnitude
</div>


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
# Gradient-based
from captum.attr import Saliency              # Vanilla gradient magnitude
from captum.attr import InputXGradient        # Input × gradient
from captum.attr import GuidedBackprop        # Modified gradient backprop
from captum.attr import Deconvolution         # Deconvolution-based
from captum.attr import IntegratedGradients   # Path integral attribution

# Propagation-based
from captum.attr import GuidedGradCam         # Guided GradCAM

# Layer-based
from captum.attr import LayerGradCam          # Gradient-weighted class activation maps
from captum.attr import LayerConductance      # Attribution through a layer
from captum.attr import LayerActivation       # Raw layer activations
from captum.attr import InternalInfluence     # Attribution to internal state

# Neuron-based
from captum.attr import NeuronConductance     # Attribution to specific neuron
from captum.attr import NeuronIntegratedGradients  # IG for a neuron

# Perturbation-based
from captum.attr import FeatureAblation       # Systematic feature removal
from captum.attr import Occlusion             # Sliding window masking
from captum.attr import ShapleyValueSampling  # Approximate Shapley values
from captum.attr import FeaturePermutation    # Feature permutation importance

# Concept-based
from captum.attr import TCAV                  # Testing with concept activation vectors
```

</div>
</div>

### The NoiseTunnel Wrapper

`NoiseTunnel` adds variance reduction to any attribution method:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
from captum.attr import IntegratedGradients, NoiseTunnel

ig = IntegratedGradients(model)
nt = NoiseTunnel(ig)

# SmoothGrad: average attributions over inputs + noise
attributions_smooth = nt.attribute(
    inputs,
    nt_type='smoothgrad',     # Average over noisy samples
    nt_samples=20,            # Number of noise samples
    stdevs=0.1,               # Noise standard deviation
    baselines=baseline,
    target=class_idx
)

# VarGrad: use variance of attributions
attributions_var = nt.attribute(
    inputs,
    nt_type='vargrad',        # Variance of attributions
    nt_samples=20,
    stdevs=0.1,
    baselines=baseline,
    target=class_idx
)
```

</div>
</div>

---

## 4. Captum Visualization Utilities

Captum includes `captum.attr.visualization` for common attribution visualizations:
<div class="callout-insight">
<strong>Insight:</strong> Captum includes `captum.attr.visualization` for common attribution visualizations:
</div>


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
from captum.attr import visualization as viz
import numpy as np

# Convert attribution tensor to numpy for visualization
attribution_np = attributions.squeeze(0).permute(1, 2, 0).detach().numpy()
image_np = input_tensor.squeeze(0).permute(1, 2, 0).detach().numpy()

# Denormalize image for display
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
image_display = image_np * std + mean
image_display = np.clip(image_display, 0, 1)

# Visualize attributions overlaid on image
fig, axes = viz.visualize_image_attr_multiple(
    attribution_np,                        # Attribution heatmap
    image_display,                         # Original image
    methods=["original_image",             # Show original
             "heat_map",                    # Attribution heatmap
             "blended_heat_map",            # Overlay on image
             "masked_image"],               # Mask low-attribution regions
    signs=["all", "positive", "all", "positive"],
    titles=["Original", "Heatmap", "Overlay", "Masked"],
    show_colorbar=True,
    fig_size=(18, 4)
)
```

</div>
</div>

### Available Visualization Modes

| Mode | Description |
|------|-------------|
| `original_image` | Display the input image |
| `heat_map` | Attribution heatmap |
| `blended_heat_map` | Heatmap overlaid on image |
| `masked_image` | Image with low-attribution regions masked |
| `alpha_scaling` | Image with opacity proportional to attribution |

---

## 5. Comparison with SHAP and LIME

### SHAP (SHapley Additive exPlanations)

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
# SHAP usage (separate library)
import shap

explainer = shap.DeepExplainer(model, background_data)
shap_values = explainer.shap_values(input_data)
shap.image_plot(shap_values, input_data)
```

</div>
</div>

**SHAP strengths:**
- Game-theoretically grounded Shapley values
- Consistent, additive property guarantees
- Good ecosystem for tabular/NLP/vision
- Built-in plots for various data types

**SHAP limitations:**
- Separate library (not PyTorch-native)
- `DeepExplainer` is KernelSHAP with model-specific optimizations — not exact Shapley values
- Computationally expensive for large models
- Less coverage of layer/neuron attribution

**Captum vs SHAP for deep learning:**
- Captum: better layer/neuron attribution, native PyTorch, gradient methods
- SHAP: better theoretical foundation, better for tabular, better ecosystem

### LIME (Local Interpretable Model-agnostic Explanations)

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
# LIME usage (separate library)
from lime import lime_image

explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(
    image,
    model_predict_fn,
    num_samples=1000
)
```

</div>
</div>

**LIME strengths:**
- Truly model-agnostic (only needs input-output pairs)
- Works without gradients
- Intuitive local surrogate model interpretation

**LIME limitations:**
- High variance (sample-dependent explanations)
- Slow (requires 1000+ model forward passes)
- Segmentation-dependent for images
- Local linearity assumption may not hold

**Captum vs LIME:**
- Captum: faster (gradient-based), more precise, layer attribution
- LIME: model-agnostic (works on non-differentiable models), simpler interpretation

#
---

## Practice Questions

<div class="callout-info">
<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Key Insight" and why it matters in practice.

2. Given a real-world scenario involving captum library architecture and philosophy, what would be your first three steps to apply the techniques from this guide?
</div>

## Summary Comparison

| Feature | Captum | SHAP | LIME |
|---------|--------|------|------|
| PyTorch-native | Yes | No | No |
| Gradient methods | Yes | No | No |
| Layer attribution | Yes | No | No |
| Neuron attribution | Yes | No | No |
| Tabular support | Yes | Best | Yes |
| Axiom compliance (IG) | Yes | Yes (approx) | No |
| Model-agnostic | Partial | Partial | Yes |
| Speed | Fast | Medium | Slow |
| Concept attribution | Yes (TCAV) | No | No |

**Recommendation:** Use Captum as your primary tool for PyTorch models. Use SHAP for final reporting (additive property is useful for business communication) and for tabular models. Use LIME only when you need model-agnostic explanations or cannot use gradients.

---

## 6. Working with Pretrained Models

Captum works with any pretrained PyTorch model out of the box:
<div class="callout-key">
<strong>Key Point:</strong> Captum works with any pretrained PyTorch model out of the box:
</div>


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
import torch
import torchvision.models as models

# ImageNet pretrained models
resnet50 = models.resnet50(weights='IMAGENET1K_V1')
vgg16 = models.vgg16(weights='IMAGENET1K_V1')
efficientnet = models.efficientnet_b0(weights='IMAGENET1K_V1')

# All work with Captum with zero modification
from captum.attr import IntegratedGradients

ig = IntegratedGradients(resnet50.eval())
```

</div>
</div>

### HuggingFace Transformers

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

# Captum's LayerIntegratedGradients works on transformer embeddings
from captum.attr import LayerIntegratedGradients

lig = LayerIntegratedGradients(model, model.distilbert.embeddings)
```

</div>
</div>

### Accessing Named Layers

For layer-based attribution, you need to specify which layer to attribute to:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
# For ResNet50
target_layer = resnet50.layer4[-1].conv2  # Last conv in layer 4

# For VGG16
target_layer = vgg16.features[28]          # Last conv layer

# For EfficientNet
target_layer = efficientnet.features[-1]   # Last feature block

from captum.attr import LayerGradCam
lgc = LayerGradCam(resnet50.eval(), target_layer)
```

</div>
</div>

---

## 7. Performance and Practical Considerations

### Memory Management
<div class="callout-insight">
<strong>Insight:</strong> Attribution computation creates a full backward pass (for gradient methods). For large batches or high-resolution images, this can exceed GPU memory:
</div>


Attribution computation creates a full backward pass (for gradient methods). For large batches or high-resolution images, this can exceed GPU memory:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
# For memory-constrained settings, process one image at a time
results = []
for i in range(batch_size):
    attr = ig.attribute(
        inputs[i:i+1],  # Single image
        baselines=baseline,
        target=targets[i]
    )
    results.append(attr.detach().cpu())
    torch.cuda.empty_cache()  # Free gradient memory

attributions = torch.cat(results, dim=0)
```

</div>
</div>

### Evaluation Mode

Always call `model.eval()` before computing attributions. Training mode enables dropout and batch norm in training behavior, which makes attributions non-deterministic and unreliable:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
model.eval()  # CRITICAL: disables dropout, uses running stats for BN
with torch.no_grad():  # Use no_grad for the forward model pass
    pass  # But NOT for the attribution call itself

# Correct pattern:
model.eval()
attributions = ig.attribute(inputs, baselines=baseline, target=class_idx)
# Captum handles gradient computation internally
```

</div>
</div>

### Internal Batching

For Integrated Gradients with many steps, Captum supports internal batching to control memory:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
attributions = ig.attribute(
    inputs,
    baselines=baseline,
    target=class_idx,
    n_steps=300,              # More steps = better approximation
    internal_batch_size=50    # Process 50 steps at a time
)
```

</div>
</div>

---

## 8. Quick Validation: Is Your Attribution Working?

Before trusting attributions, run these sanity checks:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
# 1. Completeness check (for Integrated Gradients)
attributions, delta = ig.attribute(
    inputs, baselines=baseline, target=class_idx,
    return_convergence_delta=True
)
print(f"Convergence delta: {delta.item():.4f}")  # Should be near 0

# 2. Attribution sum check
# For IG: sum of attributions should equal f(input) - f(baseline)
model.eval()
with torch.no_grad():
    pred_input = model(inputs)[0, class_idx].item()
    pred_baseline = model(baseline)[0, class_idx].item()

attr_sum = attributions.sum().item()
expected = pred_input - pred_baseline
print(f"Attribution sum:     {attr_sum:.4f}")
print(f"f(input) - f(base):  {expected:.4f}")  # Should match

# 3. Randomization check
# Attributions for a randomly initialized model should look like noise
import copy
random_model = copy.deepcopy(resnet50)
for layer in random_model.modules():
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()

ig_random = IntegratedGradients(random_model.eval())
attr_random = ig_random.attribute(inputs, baselines=baseline, target=class_idx)
# This should look like noise, not structured attribution
```

</div>
</div>

---

## Common Pitfalls

- **Forgetting `model.eval()`:** Dropout during attribution produces different results each run.
- **Wrong target class:** `target` is the class index, not the class name or one-hot vector.
- **Gradient-enabled baseline:** The baseline should NOT require gradients.
- **Batch dimension confusion:** Inputs should have a batch dimension: `(1, C, H, W)` not `(C, H, W)`.
- **Not detaching:** Call `.detach()` before converting to numpy to avoid retaining gradient graph.

---

## Connections

- **Builds on:** Guide 01 (why interpretability), Guide 02 (taxonomy), PyTorch basics
- **Leads to:** Module 01 (gradient methods hands-on), Module 02 (Integrated Gradients)
- **Related to:** SHAP ecosystem, LIME, model explanations for production

---

## Further Reading

- Kokhlikyan et al. (2020). Captum: A unified and generic model interpretability library for PyTorch. *arXiv:2009.07896* — Original Captum paper.
- Captum documentation: captum.ai/docs — Full API reference.
- PyTorch Captum tutorials: captum.ai/tutorials — Official worked examples.
- Adebayo et al. (2018). Sanity Checks for Saliency Maps. *NeurIPS 2018* — Critical evaluation of attribution methods including randomization tests.

---

## Cross-References

<a class="link-card" href="../notebooks/01_environment_setup.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
