# Guide 01: GradCAM, Guided GradCAM, and LayerGradCam

> **Reading time:** ~8 min | **Module:** 3 — Layer & Neuron Attribution | **Prerequisites:** Module 2 Integrated Gradients


## Overview

Gradient-weighted Class Activation Mapping (GradCAM) answers a different question than input-level attribution methods like IG. Instead of asking "which input pixels mattered?", GradCAM asks "which spatial regions in a convolutional feature map drove this prediction?" The result is a coarse heatmap aligned to the image, produced by analyzing the gradients flowing into the final convolutional layer.

GradCAM is the most widely used CNN attribution method in practice because it:
1. Produces spatially interpretable heatmaps without per-pixel noise
2. Works on any CNN without architectural modification
3. Runs in a single forward-backward pass (fast)
4. Highlights entire object regions rather than scattered pixels


<div class="callout-key">

<strong>Key Concept Summary:</strong> This guide covers the core concepts of guide 01: gradcam, guided gradcam, and layergradcam.

</div>

---

## 1. Motivation: Why Convolutional Feature Maps?

A convolutional neural network (CNN) builds hierarchical representations:
- **Early layers:** edges, textures
- **Middle layers:** parts (eyes, wheels, windows)
- **Final conv layer:** high-level semantic regions ("dog face", "car body")

The final convolutional layer retains spatial information before the global average pooling and fully connected classifier collapse it to a single logit. GradCAM exploits this spatial structure.

**Key insight:** The gradient of the output class score with respect to the final conv layer's feature maps tells us how much each spatial location contributed to the decision.

---

## 2. GradCAM Derivation

Let $A^k$ denote the $k$-th feature map of the final convolutional layer, with spatial dimensions $u \times v$. Let $y^c$ be the score (pre-softmax) for class $c$.
<div class="callout-insight">

<strong>Insight:</strong> Let $A^k$ denote the $k$-th feature map of the final convolutional layer, with spatial dimensions $u \times v$. Let $y^c$ be the score (pre-softmax) for class $c$.

</div>


### Step 1: Compute Gradient Weights

For each feature map $k$, compute the global average of gradients over the spatial dimensions:

$$\alpha_k^c = \frac{1}{Z} \sum_u \sum_v \frac{\partial y^c}{\partial A^k_{uv}}$$

where $Z = u \times v$ is the total number of spatial locations.

$\alpha_k^c$ is the **importance weight** for feature map $k$ with respect to class $c$. It measures how much, on average, each spatial location in that feature map influences the class score.

### Step 2: Weighted Combination

Form a weighted sum of the feature maps, then apply ReLU:

$$L^c_{\text{GradCAM}} = \text{ReLU}\left(\sum_k \alpha_k^c \cdot A^k\right)$$

The **ReLU** is critical: it removes negative contributions (regions that *reduce* the class score) and retains only regions that *increase* it. The result is a spatial map of the same size as the feature maps (e.g., 7×7 for ResNet-50's final conv layer).

### Step 3: Upsample

Bilinearly upsample $L^c_{\text{GradCAM}}$ from 7×7 to the input image size (224×224) for overlay visualization.

---

## 3. Limitations of GradCAM

GradCAM has two known weaknesses:

**1. Coarseness:** The 7×7 spatial resolution of ResNet-50's final layer produces a coarse heatmap. Fine-grained attribution (e.g., "which whiskers?") is not possible with GradCAM alone.

**2. Class discriminativeness:** GradCAM can sometimes highlight regions relevant to multiple classes (e.g., two dogs in an image may both be highlighted when explaining "dog"). The ReLU partially addresses this but does not fully solve it.

---

## 4. Guided GradCAM

**Guided GradCAM** combines GradCAM's spatial localization with Guided Backpropagation's fine-grained resolution.

The combination:
1. Compute Guided Backpropagation attributions (high-resolution, fine-grained)
2. Compute GradCAM heatmap (low-resolution, spatially coherent)
3. Upsample the GradCAM heatmap to input resolution
4. Element-wise multiply the two maps

$$L^c_{\text{Guided GradCAM}} = L^c_{\text{GuidedBP}} \odot \text{upsample}(L^c_{\text{GradCAM}})$$

The result preserves the fine detail of Guided Backpropagation while focusing it on the spatially coherent regions identified by GradCAM.

**Important caveat:** As noted in Module 01, Guided Backpropagation does not satisfy implementation invariance, so Guided GradCAM inherits this limitation. The combination is visually appealing but not axiomatically sound. Use it for visual communication, not rigorous attribution analysis.

---

## 5. LayerGradCam in Captum

Captum implements GradCAM as `LayerGradCam`, which can target any convolutional layer:
<div class="callout-insight">

<strong>Insight:</strong> Captum implements GradCAM as `LayerGradCam`, which can target any convolutional layer:

</div>



<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
from captum.attr import LayerGradCam, LayerAttribution

# Target the final convolutional layer of ResNet-50
# (the last layer before global average pooling)
target_layer = model.layer4[-1]  # ResNet-50: BasicBlock/Bottleneck

lg = LayerGradCam(model, target_layer)

# Compute GradCAM attribution
attr = lg.attribute(
    input_tensor,   # (1, 3, 224, 224)
    target=class_idx
)
# attr shape: (1, 2048, 7, 7)  — one value per spatial location per channel

# Upsample to input resolution
attr_upsampled = LayerAttribution.interpolate(
    attr,
    interpolate_dims=(224, 224),
    interpolate_mode='bilinear'
)
# attr_upsampled shape: (1, 2048, 224, 224)
```

</div>

</div>

### Aggregating Channels

The raw output has shape `(1, channels, 7, 7)`. To get a single 2D heatmap:


<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import torch
import numpy as np

# Option 1: Sum across channels (most common)
heatmap = attr.sum(dim=1).squeeze(0)

# Option 2: ReLU + sum (ignore negative contributions)
heatmap = torch.relu(attr).sum(dim=1).squeeze(0)

# Normalize to [0, 1]
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
```

</div>

</div>

---

## 6. Intermediate Layer GradCAM

One powerful feature of Captum's `LayerGradCam` is the ability to target intermediate layers, not just the final conv layer. This reveals what earlier layers "see":


<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Early layer: edges, textures
early_layer = model.layer1[-1]
lg_early = LayerGradCam(model, early_layer)
attr_early = lg_early.attribute(input_tensor, target=class_idx)

# Middle layer: parts
mid_layer = model.layer2[-1]
lg_mid = LayerGradCam(model, mid_layer)
attr_mid = lg_mid.attribute(input_tensor, target=class_idx)

# Final layer: semantic regions
final_layer = model.layer4[-1]
lg_final = LayerGradCam(model, final_layer)
attr_final = lg_final.attribute(input_tensor, target=class_idx)
```

</div>

</div>

Comparing GradCAM across layers shows the hierarchical nature of CNN representations: early layers respond to local edges, middle layers to object parts, final layers to whole object regions.

---

## 7. GradCAM for Multiple Classes

A key advantage of GradCAM over class-agnostic methods is that it can produce different heatmaps for different classes. For an image containing both a dog and a cat:


<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Heatmap explaining "dog" prediction
attr_dog = lg.attribute(input_tensor, target=dog_class_idx)

# Heatmap explaining "cat" prediction
attr_cat = lg.attribute(input_tensor, target=cat_class_idx)
```

</div>

</div>

The dog heatmap should highlight the dog region; the cat heatmap should highlight the cat region. If they are the same, the model may be using shared features (texture, background) rather than class-specific object features.

---

## 8. Quantitative GradCAM: Insertion/Deletion Score

Beyond visual inspection, GradCAM can be evaluated quantitatively using the **Insertion** and **Deletion** metrics (Petsiuk et al., 2018):
<div class="callout-insight">

<strong>Insight:</strong> Beyond visual inspection, GradCAM can be evaluated quantitatively using the **Insertion** and **Deletion** metrics (Petsiuk et al., 2018):

</div>


**Deletion:** Starting from the full image, progressively replace pixels with baseline (black) in order of decreasing attribution importance. Measure how fast the model's confidence drops. Faster drop = better attribution.

**Insertion:** Starting from the baseline, progressively reveal pixels in order of decreasing attribution importance. Measure how fast confidence rises. Faster rise = better attribution.


<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def deletion_score(model, image, heatmap, n_steps=100, target_class=None):
    """
    Compute deletion AUC — lower is better (confidence drops fast).
    """
    pixels = [(v, u) for u in range(224) for v in range(224)]
    # Sort by attribution importance (descending)
    flat_heatmap = heatmap.flatten()
    sorted_idxs = flat_heatmap.argsort(descending=True)

    scores = []
    modified = image.clone()
    step_size = len(pixels) // n_steps

    for step in range(n_steps):
        start = step * step_size
        end = start + step_size
        for flat_idx in sorted_idxs[start:end]:
            row = flat_idx // 224
            col = flat_idx % 224
            modified[0, :, row, col] = 0.0  # Set to baseline

        with torch.no_grad():
            prob = torch.softmax(model(modified), dim=1)[0, target_class].item()
        scores.append(prob)

    return np.trapz(scores) / n_steps  # AUC — lower is better
```

</div>

</div>

---

## 9. GradCAM vs. Integrated Gradients: Choosing the Right Tool

| Criterion | GradCAM | Integrated Gradients |
|-----------|---------|---------------------|
| Resolution | Coarse (feature map size) | Full input resolution |
| Computational cost | Single forward-backward pass | n_steps passes |
| Axiom compliance | None formally | Sensitivity + Implementation Invariance |
| Best for | Spatial localization | Per-pixel attribution accuracy |
| Works on text/tabular | No (conv layers required) | Yes (any differentiable model) |
| Negative attributions | No (ReLU removes them) | Yes (signed attributions) |
| Interpretability | Which regions | Which exact features |

**Rule of thumb:**
- Use GradCAM for CNN image models when you want to know "where" the model is looking
- Use IG when you need provable attribution correctness or non-spatial data

---

## 10. Common Pitfalls

**1. Wrong target layer:**
Choosing a non-convolutional layer (e.g., a fully connected layer) will produce errors or meaningless results. Always target the last layer that maintains spatial structure — typically the layer before global average pooling.

**2. Batch normalization interference:**
Some models with aggressive batch normalization produce noisy GradCAM results. If the heatmap is uniform or random, try targeting an earlier layer.

**3. Forgetting `model.eval()`:**
In training mode, batch normalization uses batch statistics, which changes the gradients. Always call `model.eval()` before attribution.

**4. Ignoring negative GradCAM:**
The ReLU in GradCAM removes negative contributions. To see the full picture (what the model actively suppresses), compute GradCAM without ReLU and visualize both positive and negative regions.


<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Full signed GradCAM (no ReLU in aggregation)
heatmap_signed = attr.sum(dim=1).squeeze(0)
positive_map = torch.relu(heatmap_signed)
negative_map = torch.relu(-heatmap_signed)
```

</div>

</div>

---


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Motivation: Why Convolutional Feature Maps?" and why it matters in practice.

2. Given a real-world scenario involving guide 01: gradcam, guided gradcam, and layergradcam, what would be your first three steps to apply the techniques from this guide?

</div>

## Summary

GradCAM provides fast, spatially coherent class activation maps by weighting convolutional feature maps by their gradient importance. Its key properties:

1. **Spatial localization:** Highlights which image regions the model attends to for a specific class
2. **Layer-selective:** Can target any convolutional layer to reveal hierarchical representations
3. **Class-discriminative:** Different classes produce different heatmaps for the same image
4. **Fast:** Single forward-backward pass, unlike IG's n_steps passes
5. **Limitation:** Coarse resolution and no formal axiom guarantees

In practice, GradCAM is the first tool to reach for when explaining CNN predictions on images. It answers "where is the model looking?" quickly and clearly.

---

## Further Reading

- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017
- Chattopadhay et al., "Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks", WACV 2018
- Petsiuk et al., "RISE: Randomized Input Sampling for Explanation of Black-box Models", BMVC 2018
- Captum LayerGradCam documentation: https://captum.ai/api/layer.html#layergradcam

---

## Cross-References

<a class="link-card" href="../notebooks/01_gradcam_resnet.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
