# Model Debugging with Attribution Methods

> **Reading time:** ~9 min | **Module:** 8 — Production Pipelines | **Prerequisites:** Modules 1-7


## Learning Objectives

By the end of this guide, you will be able to:
1. Identify spurious correlations using attribution analysis
2. Detect data leakage artifacts with saliency maps
3. Use attribution to generate hypotheses for model improvement
4. Produce attribution-based evidence for regulatory reporting
5. Build a systematic debugging workflow for classification models


<div class="callout-key">

<strong>Key Concept Summary:</strong> This guide covers the core concepts of model debugging with attribution methods.

</div>

---

## 1. Why Attribution-Based Debugging?

Traditional model evaluation answers: "How accurate is this model?"

Attribution-based debugging answers: "Why does this model make these predictions?"

These are different questions. A model can achieve high accuracy on test data while relying on shortcuts — background textures, watermarks, metadata artifacts — that fail catastrophically on distribution shifts.

**Famous examples of shortcut learning:**
- Pneumonia classifiers that learned hospital-specific scanner artifacts rather than disease features
- Wolf vs. husky classifiers that learned snow background rather than animal features
- Skin lesion classifiers that learned surgical skin markers as a "malignant" signal
- NLP models that learned annotation artifacts (hypothesis-only classification in NLI)

Attribution methods make these shortcuts visible before deployment.

---

## 2. The Debugging Workflow

```

1. Train model → evaluate metrics (accuracy, AUC, F1)
2. Select representative examples per class
3. Compute attributions for correct AND incorrect predictions
4. Look for:
   - Attributions on background / irrelevant regions
   - Attributions on metadata (borders, watermarks, text)
   - Consistent attribution patterns across wrong predictions
   - Class-specific spurious features
5. Generate hypotheses about the shortcut
6. Test hypotheses with counterfactual examples
7. Fix: data augmentation, reweighting, debiasing, retraining
8. Repeat
```

---

## 3. Detecting Spurious Correlations

### Setup: Biased CIFAR Classifier


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from captum.attr import IntegratedGradients, Saliency

# Load a pretrained model for demonstration
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.eval()

ig = IntegratedGradients(lambda x: model(x))
```

</div>

</div>

### Attribution on Correct Predictions


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
def attribution_for_prediction(model, inputs, target_class, n_steps=50):
    """Compute IG attribution and return signed + unsigned versions."""
    ig = IntegratedGradients(lambda x: model(x))
    baseline = torch.zeros_like(inputs)

    attrs, delta = ig.attribute(
        inputs, baseline,
        target=target_class,
        n_steps=n_steps,
        return_convergence_delta=True,
    )
    # Aggregate over channel dimension → (H, W)
    attrs_unsigned = attrs.abs().sum(dim=1).squeeze(0).detach()
    attrs_signed   = attrs.sum(dim=1).squeeze(0).detach()

    return attrs_signed, attrs_unsigned, delta.item()
```

</div>

</div>

---

## 4. Attribution Heatmap Analysis

### Visualizing What the Model Relies On


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def debug_attribution_plot(image_tensor, attrs_unsigned, title=""):
    """Overlay attribution heatmap on image for debugging."""
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (image_tensor.squeeze(0) * std + mean).clamp(0, 1)
    img_np = img.permute(1, 2, 0).numpy()

    # Normalize attribution to [0, 1]
    attr_np = attrs_unsigned.numpy()
    attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(attr_np, cmap="hot")
    axes[1].set_title("Attribution Heatmap")
    axes[1].axis("off")

    # Overlay: blend image and heatmap
    heatmap_rgb = cm.hot(attr_np)[:, :, :3]
    overlay = 0.6 * img_np + 0.4 * heatmap_rgb
    axes[2].imshow(overlay.clip(0, 1))
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig
```

</div>

</div>

---

## 5. Spurious Correlation Detection Protocol

### Step 1: Compute Attribution Statistics Per Class


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
def class_attribution_statistics(
    model, dataloader, class_idx, n_examples=50, n_steps=30
):
    """Compute mean attribution map for a specific class."""
    ig = IntegratedGradients(lambda x: model(x))
    attribution_accumulator = None
    count = 0

    for images, labels in dataloader:
        for img, label in zip(images, labels):
            if label.item() != class_idx or count >= n_examples:
                continue
            img = img.unsqueeze(0)
            baseline = torch.zeros_like(img)
            attrs = ig.attribute(img, baseline, target=class_idx, n_steps=n_steps)
            attr_map = attrs.abs().sum(dim=1).squeeze(0)

            if attribution_accumulator is None:
                attribution_accumulator = attr_map
            else:
                attribution_accumulator += attr_map
            count += 1

    if attribution_accumulator is not None and count > 0:
        attribution_accumulator /= count

    return attribution_accumulator, count
```

</div>

</div>

### Step 2: Compare Mean Attributions Across Classes


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
def compare_class_attributions(model, dataloader, class_names, n_steps=30):
    """Find spatial regions consistently attributed differently across classes."""
    class_attr_maps = {}

    for class_idx, class_name in enumerate(class_names):
        mean_map, n = class_attribution_statistics(
            model, dataloader, class_idx, n_examples=30, n_steps=n_steps
        )
        if mean_map is not None:
            class_attr_maps[class_name] = mean_map
            print(f"  {class_name}: {n} examples, max attr = {mean_map.max():.4f}")

    return class_attr_maps
```

</div>

</div>

---

## 6. Common Spurious Patterns and Their Signatures

### Pattern 1: Background Attribution
<div class="callout-warning">

<strong>Warning:</strong> **Symptom:** High attribution in sky, grass, water, snow — not the object.

</div>


**Symptom:** High attribution in sky, grass, water, snow — not the object.

**Diagnosis code:**


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
def measure_object_vs_background_attribution(attrs_map, segmentation_mask):
    """
    Compute fraction of total attribution inside vs. outside object mask.
    segmentation_mask: binary tensor, 1 = object region.
    """
    total_attr = attrs_map.sum().item()
    object_attr = (attrs_map * segmentation_mask).sum().item()
    background_attr = total_attr - object_attr

    object_fraction = object_attr / (total_attr + 1e-8)
    background_fraction = background_attr / (total_attr + 1e-8)

    return {
        "object_fraction": object_fraction,
        "background_fraction": background_fraction,
        "diagnosis": "spurious" if background_fraction > 0.5 else "plausible",
    }
```

</div>

</div>

**Fix:** Background augmentation (random background replacement), Grad-CAM guided cropping, or object-centric training data collection.

---

### Pattern 2: Metadata / Artifact Attribution

**Symptom:** High attribution on image borders, watermarks, hospital system overlays, or systematic pixel patterns.

**Diagnosis code:**


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
def measure_border_attribution(attrs_map, border_fraction=0.1):
    """
    Measure what fraction of attribution falls in border region.
    border_fraction: e.g. 0.1 = outermost 10% of each dimension.
    """
    H, W = attrs_map.shape
    border_h = int(H * border_fraction)
    border_w = int(W * border_fraction)

    # Create border mask
    mask = torch.zeros_like(attrs_map)
    mask[:border_h, :] = 1
    mask[-border_h:, :] = 1
    mask[:, :border_w] = 1
    mask[:, -border_w:] = 1

    total = attrs_map.sum().item()
    border_attr = (attrs_map * mask).sum().item()
    return border_attr / (total + 1e-8)


# Flag examples where >20% attribution is in the border
def screen_for_border_artifacts(model, dataloader, threshold=0.20, n_steps=30):
    ig = IntegratedGradients(lambda x: model(x))
    flagged = []

    for batch_idx, (images, labels) in enumerate(dataloader):
        for i, (img, label) in enumerate(zip(images, labels)):
            img = img.unsqueeze(0)
            baseline = torch.zeros_like(img)
            with torch.no_grad():
                pred = model(img).argmax(dim=1).item()
            attrs = ig.attribute(img, baseline, target=pred, n_steps=n_steps)
            attr_map = attrs.abs().sum(dim=1).squeeze(0)
            border_frac = measure_border_attribution(attr_map)

            if border_frac > threshold:
                flagged.append({
                    "batch_idx": batch_idx, "sample_idx": i,
                    "pred_class": pred, "border_attribution": border_frac,
                })

    return flagged
```

</div>

</div>

---

### Pattern 3: Texture vs. Shape Bias

**Symptom:** For object classifiers, attribution concentrates on textures (fur, fabric) rather than shape outlines.

**Diagnosis:** Use TCAV with texture-concept images (see Module 06) or compare attribution across stylized variants of the same image.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
def texture_vs_shape_score(attrs_map, edge_mask):
    """
    edge_mask: binary mask from Canny edge detection — represents shape.
    High score = model relies on shape.
    Low score = model relies on texture.
    """
    total = attrs_map.sum().item()
    shape_attr = (attrs_map * edge_mask).sum().item()
    return shape_attr / (total + 1e-8)
```

</div>

</div>

---

## 7. Debugging Failed Predictions

### Attribution Comparison: Correct vs. Incorrect Predictions


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
def debug_misclassification(model, image, true_label, pred_label, n_steps=50):
    """
    Compute attributions targeting true class and predicted class.
    Reveals what features pushed the model toward the wrong answer.
    """
    ig = IntegratedGradients(lambda x: model(x))
    baseline = torch.zeros_like(image)

    # Attribution toward TRUE class (what was missing?)
    attrs_true = ig.attribute(image, baseline, target=true_label, n_steps=n_steps)

    # Attribution toward PREDICTED class (what fired incorrectly?)
    attrs_pred = ig.attribute(image, baseline, target=pred_label, n_steps=n_steps)

    return {
        "attrs_toward_true": attrs_true.abs().sum(dim=1).squeeze(0),
        "attrs_toward_pred": attrs_pred.abs().sum(dim=1).squeeze(0),
    }
```

</div>

</div>

**Interpretation:**
- `attrs_toward_true`: Features present in the image that support the correct class but were insufficient
- `attrs_toward_pred`: Features that drove the wrong prediction — often the spurious signal

---

## 8. Attribution-Based Regulatory Reporting

### What Regulators Require
<div class="callout-insight">

<strong>Insight:</strong> For high-stakes models (credit, hiring, medical), regulators may require:

</div>


For high-stakes models (credit, hiring, medical), regulators may require:
- Per-prediction explanation of which features influenced the outcome
- Evidence that decisions are not based on protected characteristics
- Consistent explanations across similar inputs (stability)

### Attribution Report Structure


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
from dataclasses import dataclass, field
from typing import List, Optional
import json
from datetime import datetime

@dataclass
class AttributionReport:
    report_id: str
    model_id: str
    model_version: str
    prediction_timestamp: str
    input_hash: str                    # SHA256 of raw input
    predicted_class: str
    predicted_confidence: float
    attribution_method: str
    top_features: List[dict]           # [{name, attribution, rank}, ...]
    attribution_sum: float             # for completeness check
    convergence_delta: Optional[float]
    baseline_description: str

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)

    def completeness_check(self, tolerance: float = 0.05) -> bool:
        """Verify attribution sum matches f(x) - f(baseline)."""
        # In practice, compare attribution_sum to model output difference
        return abs(self.convergence_delta or 0) < tolerance


def generate_attribution_report(
    model, tokenizer_or_transform,
    input_data, input_metadata: dict,
    model_id: str, model_version: str,
    attribution_method: str = "integrated_gradients",
    n_steps: int = 100,
) -> AttributionReport:
    """Generate a compliance-grade attribution report."""
    import hashlib
    import uuid

    # Compute attribution
    ig = IntegratedGradients(lambda x: model(x))
    baseline = torch.zeros_like(input_data)

    with torch.no_grad():
        logits = model(input_data)
        probs = torch.softmax(logits, dim=1)[0]
        pred_class = probs.argmax().item()
        confidence = probs[pred_class].item()

    attrs, delta = ig.attribute(
        input_data, baseline, target=pred_class,
        n_steps=n_steps, return_convergence_delta=True
    )

    # Flatten and rank features
    attrs_flat = attrs.squeeze(0).detach().cpu().numpy().flatten()
    feature_names = input_metadata.get("feature_names",
                                        [f"feature_{i}" for i in range(len(attrs_flat))])

    ranked = sorted(
        zip(feature_names, attrs_flat),
        key=lambda x: abs(x[1]), reverse=True
    )
    top_features = [
        {"name": name, "attribution": float(val), "rank": i+1}
        for i, (name, val) in enumerate(ranked[:10])
    ]

    input_hash = hashlib.sha256(
        input_data.cpu().numpy().tobytes()
    ).hexdigest()

    return AttributionReport(
        report_id=str(uuid.uuid4()),
        model_id=model_id,
        model_version=model_version,
        prediction_timestamp=datetime.utcnow().isoformat() + "Z",
        input_hash=input_hash,
        predicted_class=input_metadata.get("class_names", [str(pred_class)])[pred_class],
        predicted_confidence=round(confidence, 4),
        attribution_method=attribution_method,
        top_features=top_features,
        attribution_sum=float(attrs_flat.sum()),
        convergence_delta=float(delta.item()),
        baseline_description="zero vector (all zeros)",
    )
```

</div>

</div>

---

## 9. Monitoring Attribution Drift

Attribution patterns shift when the data distribution shifts. Monitoring attribution statistics over time detects distribution drift before accuracy degrades.
<div class="callout-warning">

<strong>Warning:</strong> Attribution patterns shift when the data distribution shifts. Monitoring attribution statistics over time detects distribution drift before accuracy degrades.

</div>


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
import numpy as np
from scipy.stats import wasserstein_distance

class AttributionDriftMonitor:
    """Monitor changes in attribution distributions over time."""

    def __init__(self, model, n_steps=20):
        self.model = model
        self.n_steps = n_steps
        self.ig = IntegratedGradients(lambda x: model(x))
        self.reference_distributions = {}

    def record_reference(self, dataloader, class_idx, n_samples=200):
        """Record reference attribution distribution for a class."""
        attributions = []
        count = 0
        for images, labels in dataloader:
            for img, label in zip(images, labels):
                if label.item() != class_idx or count >= n_samples:
                    continue
                img = img.unsqueeze(0)
                baseline = torch.zeros_like(img)
                attrs = self.ig.attribute(img, baseline,
                                          target=class_idx, n_steps=self.n_steps)
                attributions.append(attrs.abs().mean().item())
                count += 1

        self.reference_distributions[class_idx] = np.array(attributions)

    def measure_drift(self, dataloader, class_idx, n_samples=200):
        """Compute Wasserstein distance between current and reference distributions."""
        if class_idx not in self.reference_distributions:
            raise ValueError(f"No reference recorded for class {class_idx}")

        current = []
        count = 0
        for images, labels in dataloader:
            for img, label in zip(images, labels):
                if label.item() != class_idx or count >= n_samples:
                    continue
                img = img.unsqueeze(0)
                baseline = torch.zeros_like(img)
                attrs = self.ig.attribute(img, baseline,
                                          target=class_idx, n_steps=self.n_steps)
                current.append(attrs.abs().mean().item())
                count += 1

        w_dist = wasserstein_distance(
            self.reference_distributions[class_idx],
            np.array(current)
        )
        return {"class": class_idx, "wasserstein_distance": w_dist,
                "drift_detected": w_dist > 0.1}
```

</div>

</div>

---


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Why Attribution-Based Debugging?" and why it matters in practice.

2. Given a real-world scenario involving model debugging with attribution methods, what would be your first three steps to apply the techniques from this guide?

</div>

## Summary

| Debugging Task | Tool | What to Look For |
|----------------|------|-----------------|
| Spurious correlations | IG / Saliency heatmap | Attribution on background, not object |
| Metadata artifacts | Border attribution fraction | >20% attribution in border zone |
| Texture vs. shape bias | TCAV / edge mask | Low overlap with Canny edges |
| Misclassification root cause | Dual-class attribution | Features firing for wrong class |
| Regulatory compliance | AttributionReport | Top features, completeness delta |
| Distribution drift | AttributionDriftMonitor | Wasserstein distance from reference |

---

## Further Reading

- "Shortcut Learning in Deep Neural Networks" — Geirhos et al. (2020)
- "Right for the Wrong Reasons" — Ross et al. (2017)
- LIME explanations for debugging: https://github.com/marcotcr/lime
- EU AI Act explainability requirements: https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai

---

## Cross-References

<a class="link-card" href="../notebooks/01_captum_insights_demo.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
