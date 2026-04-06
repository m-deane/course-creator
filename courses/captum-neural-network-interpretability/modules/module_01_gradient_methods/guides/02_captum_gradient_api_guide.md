# Captum Gradient API: Practical Usage

> **Reading time:** ~8 min | **Module:** 1 — Gradient Methods | **Prerequisites:** Module 0 Foundations


## In Brief

This guide covers the Captum API for all four gradient-based attribution methods. The focus is on practical patterns: correct input preparation, visualization, side-by-side comparison, and the sanity checks that validate attribution quality.

## Key Insight

The uniformity of Captum's API means you can compare all four methods with a single function that takes the method as a parameter. Write your attribution pipeline once, parameterize the method.


<div class="callout-key">

<strong>Key Concept Summary:</strong> This guide covers the Captum API for all four gradient-based attribution methods.

</div>

---

## 1. Common Setup Pattern

Every gradient attribution in this module follows this setup:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 1. Load and prepare the model
model = models.resnet50(weights='IMAGENET1K_V1')
model.eval()  # CRITICAL

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(DEVICE)

# 2. Standard ImageNet preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 3. Load image and preprocess
image = Image.open("path/to/image.jpg").convert('RGB')
input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

# 4. Enable gradients — required for gradient-based methods
input_for_attr = input_tensor.requires_grad_(True)

# 5. Get the predicted class
with torch.no_grad():
    probs = torch.softmax(model(input_tensor), dim=1)
    top_class = probs.argmax().item()
```

</div>

</div>

---

## 2. Saliency (Vanilla Gradient)

### API


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
from captum.attr import Saliency

saliency = Saliency(model)

attributions = saliency.attribute(
    inputs=input_for_attr,
    target=top_class,
    abs=True  # Default True: returns absolute gradient values
)
```

</div>

</div>

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `inputs` | required | Input tensor, must have `requires_grad=True` |
| `target` | required | Class index to explain |
| `abs` | `True` | If True, return `|∂f/∂x|`; if False, return signed gradient |
| `additional_forward_args` | `None` | Extra args passed to `model.forward()` |

### Output Shape

`attributions.shape == inputs.shape` — e.g., `(1, 3, 224, 224)` for a single image.

### Visualization Pattern


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
from captum.attr import visualization as viz

# Captum viz expects numpy (H, W, C) with values in [0, 1]
attr_np = attributions.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
img_np = denormalize(input_tensor.squeeze(0).cpu()).permute(1, 2, 0).numpy()

fig, axes = viz.visualize_image_attr_multiple(
    attr_np, img_np,
    methods=["heat_map", "blended_heat_map"],
    signs=["absolute_value", "absolute_value"],
    titles=["Saliency Heatmap", "Saliency Overlay"],
    show_colorbar=True,
    fig_size=(10, 4)
)
```

</div>

</div>

---

## 3. Input × Gradient

### API


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
from captum.attr import InputXGradient

ixg = InputXGradient(model)

attributions = ixg.attribute(
    inputs=input_for_attr,
    target=top_class
)

# Returns: x * ∂f/∂x (signed, no absolute value taken)
```

</div>

</div>

### Sign Information

Unlike Saliency (which takes absolute value), Input×Gradient returns signed attributions:
- **Positive values:** feature increases the predicted class score
- **Negative values:** feature decreases the predicted class score

For visualization, you can show all, only positive, or absolute value:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Show only positive attributions (evidence FOR the class)
fig, axes = viz.visualize_image_attr_multiple(
    attr_np, img_np,
    methods=["heat_map", "heat_map"],
    signs=["positive", "all"],  # positive only vs. full range
    titles=["Positive Evidence", "All Evidence"],
    fig_size=(10, 4)
)
```

</div>

</div>

---

## 4. Guided Backpropagation

### API


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
from captum.attr import GuidedBackprop

gbp = GuidedBackprop(model)

attributions = gbp.attribute(
    inputs=input_for_attr,
    target=top_class
)
```

</div>

</div>

### Important Note

GuidedBackprop registers **hooks** on the model's ReLU layers to modify the backward pass. These hooks are registered when the method is instantiated and removed when the attribution is complete. This is handled transparently by Captum.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Captum handles hook registration/removal automatically
gbp = GuidedBackprop(model)  # Registers hooks on ReLU layers
attributions = gbp.attribute(inputs, target=class_idx)  # Runs modified backward

# Hooks are removed after .attribute() completes
```

</div>

</div>

### Sanity Check: Randomization Test

This is the critical validation that reveals Guided Backprop's failure mode:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import copy

def compute_gbp_attribution(model_to_use, inputs, target):
    """Compute GBP attribution for a given model."""
    gbp = GuidedBackprop(model_to_use)
    return gbp.attribute(inputs.requires_grad_(True), target=target)

# Attribution with trained model
attr_trained = compute_gbp_attribution(model, input_tensor, top_class)

# Attribution with randomly initialized model (same architecture)
random_model = copy.deepcopy(model)
for layer in random_model.modules():
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()
random_model.eval()
attr_random = compute_gbp_attribution(random_model, input_tensor, top_class)

# Compare: they should look VERY different if GBP were faithful

# In reality, they often look similar — the architecture dominates
correlation = np.corrcoef(
    attr_trained.squeeze().detach().numpy().flatten(),
    attr_random.squeeze().detach().numpy().flatten()
)[0, 1]
print(f"Correlation between trained and random model attributions: {correlation:.3f}")

# High correlation (> 0.5) indicates architecture dependence
```

</div>

</div>

---

## 5. Deconvolution

### API


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
from captum.attr import Deconvolution

deconv = Deconvolution(model)

attributions = deconv.attribute(
    inputs=input_for_attr,
    target=top_class
)
```

</div>

</div>

Like GuidedBackprop, Deconvolution uses hooks to modify the backward pass through ReLU layers.

---

## 6. Side-by-Side Comparison Pattern

The canonical comparison pattern for all four methods:
<div class="callout-key">

<strong>Key Point:</strong> The canonical comparison pattern for all four methods:

</div>


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
from captum.attr import Saliency, InputXGradient, GuidedBackprop, Deconvolution

def compute_all_gradient_attributions(model, input_tensor, target_class):
    """
    Compute all four gradient-based attributions for comparison.

    Parameters
    ----------
    model : nn.Module
        Pretrained model in eval() mode
    input_tensor : Tensor
        Preprocessed input image, shape (1, C, H, W)
    target_class : int
        Class index to explain

    Returns
    -------
    dict : {method_name: attribution_tensor}
    """
    # Enable gradients for attribution computation
    inp = input_tensor.requires_grad_(True)

    methods = {
        'Saliency':         Saliency(model),
        'Input × Gradient': InputXGradient(model),
        'Guided Backprop':  GuidedBackprop(model),
        'Deconvolution':    Deconvolution(model),
    }

    attributions = {}
    for name, method in methods.items():
        # Fresh requires_grad for each method
        inp = input_tensor.clone().requires_grad_(True)
        attr = method.attribute(inp, target=target_class)
        attributions[name] = attr.detach()

    return attributions


def plot_attribution_comparison(attributions, image_np, predicted_class):
    """
    Plot all four attributions side-by-side.

    Parameters
    ----------
    attributions : dict
        {name: (1, C, H, W) attribution tensor}
    image_np : ndarray
        Original image as (H, W, C) array in [0, 1]
    predicted_class : str
        Class name for title
    """
    method_names = list(attributions.keys())
    n_methods = len(method_names)

    fig, axes = plt.subplots(3, n_methods, figsize=(5 * n_methods, 12))

    for col, name in enumerate(method_names):
        attr = attributions[name]
        # Convert to (H, W, C) numpy array
        attr_np = attr.squeeze(0).permute(1, 2, 0).numpy()

        # Row 0: Heatmap
        attr_mag = np.abs(attr_np).mean(axis=-1)
        im = axes[0, col].imshow(attr_mag, cmap='hot')
        axes[0, col].set_title(f"{name}\n(heatmap)", fontsize=10)
        axes[0, col].axis('off')
        plt.colorbar(im, ax=axes[0, col], fraction=0.046)

        # Row 1: Overlay on original image
        axes[1, col].imshow(image_np)
        axes[1, col].imshow(
            attr_mag, alpha=0.6, cmap='hot',
            vmin=np.percentile(attr_mag, 85)
        )
        axes[1, col].set_title(f"{name}\n(overlay)", fontsize=10)
        axes[1, col].axis('off')

        # Row 2: Distribution of attribution values
        axes[2, col].hist(attr_np.flatten(), bins=80,
                          color='steelblue', alpha=0.7, edgecolor='none')
        axes[2, col].set_title(f"{name}\n(value distribution)", fontsize=10)
        axes[2, col].set_xlabel('Attribution value')
        axes[2, col].set_ylabel('Count')
        axes[2, col].set_yscale('log')

    plt.suptitle(
        f"Gradient Attribution Comparison — Predicted: {predicted_class}",
        fontsize=14, y=1.01
    )
    plt.tight_layout()
    plt.show()
```

</div>

</div>

---

## 7. Quantitative Method Comparison

Beyond visual inspection, quantitative metrics help compare attribution quality:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def attribution_statistics(attributions, image_np):
    """
    Compute quantitative statistics for each attribution method.

    Metrics:
    - Sparsity: fraction of pixels with attribution > 90th percentile
    - Signal-to-noise: std(top 10%) / std(bottom 90%)
    - Attribution range: max - min
    - Background/foreground ratio: mean attr in background vs foreground
      (requires a simple background mask)
    """
    stats = {}
    for name, attr in attributions.items():
        attr_np = np.abs(attr.squeeze(0).permute(1, 2, 0).numpy()).mean(axis=-1)

        threshold_90 = np.percentile(attr_np, 90)
        top10 = attr_np[attr_np >= threshold_90]
        bottom90 = attr_np[attr_np < threshold_90]

        stats[name] = {
            'sparsity': (attr_np >= threshold_90).mean(),
            'snr': top10.std() / (bottom90.std() + 1e-8),
            'range': attr_np.max() - attr_np.min(),
            'mean': attr_np.mean(),
        }

    return stats
```

</div>

</div>

---

## 8. Common API Mistakes and Fixes

### Mistake 1: Forgetting `requires_grad_(True)`
<div class="callout-warning">

<strong>Warning:</strong> input_tensor = preprocess(image).unsqueeze(0)

</div>


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python

# WRONG: will raise RuntimeError about grad requirement
input_tensor = preprocess(image).unsqueeze(0)
saliency = Saliency(model)
attr = saliency.attribute(input_tensor, target=0)

# CORRECT: enable gradients on input
input_tensor = preprocess(image).unsqueeze(0).requires_grad_(True)
attr = saliency.attribute(input_tensor, target=0)
```



### Mistake 2: Model in Training Mode


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python

# WRONG: dropout produces different gradients each run
model = models.resnet50(weights='IMAGENET1K_V1')

# (eval() not called)
attr = saliency.attribute(input_tensor, target=0)  # Non-deterministic!

# CORRECT
model.eval()  # Call eval() before any attribution
attr = saliency.attribute(input_tensor, target=0)  # Deterministic
```



### Mistake 3: Wrong Target Type


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python

# WRONG: target should be int, not float or one-hot
attr = saliency.attribute(input_tensor, target=0.0)   # TypeError
attr = saliency.attribute(input_tensor, target=[1, 0])  # Wrong type

# CORRECT: integer class index
attr = saliency.attribute(input_tensor, target=0)      # Class index 0
attr = saliency.attribute(input_tensor, target=281)    # Class index 281
```



### Mistake 4: Batch Target Mismatch


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python

# For batch attribution, target can be a list/tensor
inputs = torch.stack([img1, img2, img3])  # (3, C, H, W)

# WRONG: single target for batch
attr = saliency.attribute(inputs, target=0)  # Explains class 0 for ALL images

# CORRECT: per-image target
attr = saliency.attribute(inputs, target=[0, 281, 483])  # Different class per image
```



---

## 9. Normalizing Attributions for Display

Different attribution methods produce values at different scales. Normalize for fair visual comparison:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
def normalize_attribution(attr_np, percentile=99):
    """
    Normalize attribution to [0, 1] for display.

    Uses percentile clipping to handle outliers.

    Parameters
    ----------
    attr_np : ndarray (H, W) or (H, W, C)
        Attribution values (may be any range)
    percentile : float
        Values above this percentile are clipped

    Returns
    -------
    ndarray: normalized to [0, 1]
    """
    if attr_np.ndim == 3:
        attr_np = np.abs(attr_np).mean(axis=-1)  # Reduce channels

    # Clip outliers using percentile
    vmax = np.percentile(np.abs(attr_np), percentile)
    vmin = 0

    attr_clipped = np.clip(np.abs(attr_np), vmin, vmax)
    return attr_clipped / (vmax + 1e-8)
```



---

## Common Pitfalls

- **NoiseTunnel with GuidedBackprop:** SmoothGrad applied to GBP still produces architecture-dependent results — the variance reduction does not fix the faithfulness problem.
- **Comparing attribution magnitudes across images:** Scale varies by image and method. Always normalize before comparing.
- **Using gradient methods on transformer models without special handling:** Standard Saliency works on transformers, but attributions at the embedding level are in embedding space, not token space. Use `LayerIntegratedGradients` for token-level attribution.

---

## Connections

- **Builds on:** Guide 01 (gradient theory), Module 00 (Captum API basics)
- **Leads to:** Module 01 notebooks (hands-on comparison), Module 02 (Integrated Gradients)
- **Related to:** SmoothGrad (NoiseTunnel wrapper), VarGrad (variance of attributions)

---


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Key Insight" and why it matters in practice.

2. Given a real-world scenario involving captum gradient api: practical usage, what would be your first three steps to apply the techniques from this guide?


## Further Reading

- Captum API docs for gradient methods: captum.ai/api/#attribution — Full parameter documentation.
- Adebayo et al. (2018). Sanity Checks for Saliency Maps. *NeurIPS* — The randomization test every practitioner should run.
- Smilkov et al. (2017). SmoothGrad: removing noise by adding noise. *arXiv* — Variance reduction via noise averaging.

---

## Cross-References

<a class="link-card" href="../notebooks/01_gradient_methods_cnn.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
