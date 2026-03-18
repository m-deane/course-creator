# Guide 01: Occlusion and Feature Ablation

## Overview

Perturbation methods are a fundamentally different paradigm from gradient-based attribution. Instead of backpropagating through the model to measure sensitivity, they directly observe what happens to the model's output when features are removed or replaced. This makes them:

1. **Model-agnostic:** work on any function, not just differentiable neural networks
2. **Conceptually simple:** directly measure the counterfactual "what if this feature was absent?"
3. **Expensive:** require many forward passes rather than a single backward pass
4. **Robust:** not susceptible to gradient vanishing or saturation issues

This guide covers two closely related perturbation methods: Occlusion (sliding window) and Feature Ablation (arbitrary feature groups).

---

## 1. The Perturbation Principle

The core question all perturbation methods ask:

$$\text{Importance}(F_i) = f(x) - f(x \text{ with } F_i \text{ replaced by baseline})$$

If removing feature $F_i$ (replacing it with a baseline value) decreases the model's confidence, then $F_i$ is positively important. If removing it increases confidence, $F_i$ is negatively important (it was hurting the prediction).

This is a **causal** definition of importance: we measure the actual effect of removing a feature on the output, not a local linear approximation of the gradient.

---

## 2. Occlusion (Sliding Window)

Occlusion was introduced by Zeiler & Fergus (2014) as a way to explain CNN predictions. The algorithm:

1. Select a rectangular sliding window of size $W \times W$ pixels
2. For each position $(u, v)$ of the window:
   a. Replace pixels in the window with a baseline value (e.g., 0 or mean color)
   b. Compute the model's prediction score for the target class
   c. The attribution at position $(u, v)$ is: $\text{attr}_{uv} = f(x) - f(x \text{ with window at }(u,v))$
3. Stride the window by $s$ pixels

**Computational cost:** $\lceil H/s \rceil \times \lceil W/s \rceil$ forward passes per image.

For a 224×224 image with window size 15×15 and stride 8: approximately $28 \times 28 = 784$ forward passes.

### Occlusion Resolution vs. Cost Trade-off

| Window size | Stride | Resolution | Forward passes | Localization |
|-------------|--------|------------|----------------|--------------|
| 7×7 | 3 | High (fine) | ~5,600 | Precise |
| 15×15 | 8 | Medium | ~784 | Good |
| 30×30 | 15 | Low (coarse) | ~225 | Approximate |

**Recommendation:** Start with stride=8, window=15×15 for quick exploration. Use stride=3, window=7×7 for publication-quality maps.

---

## 3. Captum Occlusion API

```python
from captum.attr import Occlusion

model = resnet50(weights='IMAGENET1K_V1').eval()

occ = Occlusion(model)

attributions = occ.attribute(
    input_tensor,                 # (1, 3, 224, 224)
    strides=(3, 8, 8),            # Stride per dimension (channels, H, W)
    target=class_idx,             # Target class
    sliding_window_shapes=(3, 15, 15),  # Window size (covers all channels)
    baselines=0                   # Baseline value (scalar or tensor)
)
# attributions: (1, 3, 224, 224) — same shape as input
```

### Parameter Details

**`strides`:** A tuple `(channel_stride, h_stride, w_stride)`. For images, always use `strides=(3, s, s)` where `3` is the number of color channels (occlude full pixel) and `s` is the spatial stride.

**`sliding_window_shapes`:** A tuple `(channels, height, width)`. Use `(3, W, W)` to occlude all color channels at once (full pixel occlusion).

**`baselines`:** The value used to fill the occluded region. Scalar `0` = black pixels. Can also be a tensor matching input shape for image-specific baselines.

### Postprocessing for Visualization

```python
import numpy as np

# Average across color channels for 2D heatmap
attr_2d = attributions.abs().mean(dim=1).squeeze().detach().cpu().numpy()

# Percentile normalization (robust to outliers)
vmin = np.percentile(attr_2d, 1)
vmax = np.percentile(attr_2d, 99)
attr_norm = np.clip((attr_2d - vmin) / (vmax - vmin + 1e-8), 0, 1)
```

---

## 4. Interpretation: What Occlusion Reveals

**High positive attribution:** Occluding this region significantly reduces the model's confidence. This region is crucial for the prediction.

**Near-zero attribution:** Occluding this region barely changes confidence. This region is irrelevant for this prediction.

**High negative attribution:** Occluding this region *increases* confidence. The model is actually confused by this region — removing it helps! This often indicates spurious background correlations.

### Common Patterns

- **Focal prediction:** High attribution concentrated on a small object region (e.g., a single bird in a large image)
- **Texture-based prediction:** High attribution distributed across a textured surface (e.g., leopard spots)
- **Background attribution:** High attribution in the background region — potential Clever Hans effect

---

## 5. Feature Ablation: Generalizing Occlusion

Occlusion uses a sliding window to define feature groups. **Feature Ablation** generalizes this to arbitrary feature groupings:

- For images: superpixels (semantically coherent regions) instead of rectangular windows
- For tabular: individual features or groups of correlated features
- For text: individual tokens, n-grams, or semantic chunks

Feature Ablation is the most general perturbation method in Captum:

```python
from captum.attr import FeatureAblation

fa = FeatureAblation(model)

# Option 1: Ablate each feature independently
attributions = fa.attribute(
    input_tensor,           # (1, 3, 224, 224)
    target=class_idx,
    baselines=0             # Replace each feature with 0
)

# Option 2: Ablate feature groups using a mask
# mask[i, j] = k means pixel (i,j) belongs to group k
mask = get_superpixel_mask(input_tensor)  # (1, 1, 224, 224) with integer group IDs
attributions_grouped = fa.attribute(
    input_tensor,
    target=class_idx,
    baselines=0,
    feature_mask=mask       # One value per group
)
# attributions_grouped: (1, 1, 224, 224) with one value per group
```

---

## 6. Superpixel-Based Feature Ablation

For images, superpixel segmentation provides semantically meaningful feature groups. SLIC (Simple Linear Iterative Clustering) segments the image into compact, color-coherent regions:

```python
from skimage.segmentation import slic
import torch

def get_superpixel_mask(image_np, n_segments=50):
    """
    Compute SLIC superpixel segmentation and return as a tensor mask.

    Parameters
    ----------
    image_np : ndarray (H, W, 3), values in [0, 1]
    n_segments : int
        Approximate number of superpixels

    Returns
    -------
    mask : Tensor (1, 1, H, W) of dtype int
        Each pixel assigned to a superpixel group ID
    """
    segments = slic(image_np, n_segments=n_segments, compactness=10,
                    start_label=0)
    mask = torch.tensor(segments, dtype=torch.long).unsqueeze(0).unsqueeze(0)
    return mask

# Usage with FeatureAblation
mask = get_superpixel_mask(img_np, n_segments=50)
fa = FeatureAblation(model)
attr = fa.attribute(
    input_tensor, target=class_idx,
    baselines=0, feature_mask=mask
)
```

Superpixel ablation produces cleaner heatmaps than pixel-level occlusion: each colored region is a semantically coherent area, and the attribution reflects whether that semantic region matters for the prediction.

---

## 7. Tabular Feature Ablation

For tabular models, Feature Ablation is the standard perturbation attribution method. Each feature is ablated individually:

```python
import torch
import torch.nn as nn
import pandas as pd

# Assume wine_model: nn.Module, input shape (1, 11)
# And a wine quality prediction task

# Feature names
feature_names = [
    'fixed_acidity', 'volatile_acidity', 'citric_acid',
    'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
    'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol'
]

# Create input and baseline
x = torch.tensor(sample_values, dtype=torch.float32).unsqueeze(0)
baseline = torch.tensor(X_train_mean, dtype=torch.float32).unsqueeze(0)

fa = FeatureAblation(wine_model)
attr = fa.attribute(
    x,
    baselines=baseline,   # "Typical wine" as reference
    target=None           # No target class for regression
)
# attr: (1, 11) — importance of each feature

# Bar chart of feature importances
importances = attr.squeeze().detach().numpy()
sorted_idx = importances.argsort()
plt.barh([feature_names[i] for i in sorted_idx], importances[sorted_idx])
plt.title('Feature Ablation Importances — Wine Quality Prediction')
```

---

## 8. Occlusion vs IG: When to Choose Which

| Criterion | Occlusion | Integrated Gradients |
|-----------|-----------|---------------------|
| Model type | Any (gradient-free) | Differentiable only |
| Computational cost | $O(H \cdot W / s^2)$ passes | $O(\text{n\_steps})$ passes |
| Resolution | Window size | Full pixel |
| Saturation issue | None | Yes (if not using IG) |
| Axiom compliance | None formal | Sensitivity + Impl. Inv. |
| Handles non-linearities | Yes (exactly) | Yes (via integration) |
| Feature interactions | Captures main effects | Linear path approximation |

**Use Occlusion when:**
- Model is not differentiable (e.g., random forests, XGBoost wrappers, API models)
- You want to confirm IG results with a gradient-free method
- The task requires explaining coarse spatial regions rather than pixel-level features

**Use IG when:**
- Model is differentiable and you need fast per-pixel attribution
- Axiom compliance is required (regulatory/documentation use)
- You need text or tabular attribution at the feature level

---

## 9. Comprehensive Perturbation Baseline Choice

Like IG, Occlusion and Feature Ablation require a baseline to define "absent feature":

| Data type | Baseline | Interpretation |
|-----------|----------|----------------|
| Images | `0` (black pixels) | "No visual information" |
| Images | Mean pixel value | "Average pixel from dataset" |
| Images | Blurred version | "Global scene without local detail" |
| Tabular | Training mean | "Typical sample from distribution" |
| Text | [PAD] token embedding | "No token at this position" |

The baseline choice changes what question you're asking. Always select the baseline that represents your domain's "no information" reference point.

---

## 10. Multi-Input Perturbation

For models with multiple input modalities (e.g., image + metadata):

```python
# Model with two inputs: image and metadata vector
def multimodal_model(image, metadata):
    img_feat = vision_encoder(image)
    meta_feat = fc_encoder(metadata)
    combined = torch.cat([img_feat, meta_feat], dim=1)
    return classifier(combined)

fa = FeatureAblation(multimodal_model)

# Ablate image pixels
attr_image = fa.attribute(
    (image, metadata),           # Tuple of inputs
    target=class_idx,
    baselines=(torch.zeros_like(image), metadata),  # Only ablate image
    feature_mask=(image_mask, None)
)

# Ablate metadata features
attr_meta = fa.attribute(
    (image, metadata),
    target=class_idx,
    baselines=(image, torch.zeros_like(metadata))   # Only ablate metadata
)
```

---

## Summary

1. **Perturbation methods** directly measure the causal effect of removing features — no gradients required
2. **Occlusion** slides a rectangular window over the image; high attribution = occluding that region hurts prediction
3. **Feature Ablation** generalizes occlusion to arbitrary feature groups (superpixels, tabular features)
4. **Computational cost** is the main drawback: many forward passes per input ($O(HW/s^2)$ for images)
5. **Model-agnostic:** work on any model, including non-differentiable ones
6. **Baseline choice** applies the same way as in IG — represents "absent feature"

---

## Further Reading

- Zeiler & Fergus, "Visualizing and Understanding Convolutional Networks", ECCV 2014 — original Occlusion paper
- Ribeiro et al., "Why Should I Trust You?: LIME", KDD 2016 — related perturbation approach
- Captum Occlusion documentation: https://captum.ai/api/occlusion.html
- Captum FeatureAblation documentation: https://captum.ai/api/feature_ablation.html
