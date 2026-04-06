# Captum Insights: Interactive Web-Based Visualization

> **Reading time:** ~5 min | **Module:** 8 — Production Pipelines | **Prerequisites:** Modules 1-7


## Learning Objectives

By the end of this guide, you will be able to:
1. Set up Captum Insights for any classification model
2. Configure input transformers and baseline functions for Insights
3. Navigate the Insights interface to explore attributions interactively
4. Use Insights for multi-class attribution comparison
5. Export and share attribution visualizations from Insights


<div class="callout-key">

<strong>Key Concept Summary:</strong> This guide covers the core concepts of captum insights: interactive web-based visualization.

</div>

---

## 1. What Is Captum Insights?

Captum Insights is a web-based visualization interface built on top of Captum's attribution methods. It provides an interactive dashboard for exploring model predictions and attributions without writing any visualization code.

**Key capabilities:**
- Interactive exploration of individual predictions
- Side-by-side comparison of attribution methods (IG, Gradient×Input, Saliency, etc.)
- Image, text, and tabular input visualization
- Real-time attribution computation as you explore examples
- Export visualizations to PNG

**Use cases:**
- Model debugging during development
- Presenting model explanations to non-technical stakeholders
- Comparing attribution methods on your specific data
- Quick exploratory analysis without visualization code

---

## 2. Installation

Captum Insights requires Flask and additional dependencies:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.sh</span>

</div>
<div class="code-body">

```bash
pip install captum[insights]
# or
pip install captum flask flask-compress
```

</div>

</div>

---

## 3. Core API: `AttributionVisualizer`

The main class is `AttributionVisualizer`, which wraps your model and attribution methods:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>
<div class="code-body">

```python
from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature, TextFeature, GeneralFeature

visualizer = AttributionVisualizer(
    models=[model],
    score_func=score_func,        # converts model output to scores
    classes=class_names,           # list of class name strings
    features=[...],                # feature descriptors
    dataset=data_iter,             # iterable of Batch objects
    num_examples=4,                # examples per batch in UI
)
```

</div>

</div>

---

## 4. Image Classification Setup

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>
<div class="code-body">

```python
import torch
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as T
from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature

# Load model
weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)
model.eval()

# Score function: converts logits to probabilities
def score_func(model_output):
    return torch.softmax(model_output, dim=1)

# Class names (first 10 for simplicity)
class_names = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
    "electric ray", "stingray", "cock", "hen", "ostrich"
]

# Image preprocessing (must match training preprocessing)
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])

# Baseline: black image
def baseline_func(input_tensor):
    return torch.zeros_like(input_tensor)

# Feature descriptor: tells Insights how to display/process the input
image_feature = ImageFeature(
    name="Input Image",
    baseline_transforms=[baseline_func],
    input_transforms=[transform],
)
```

</div>

</div>

---

## 5. Dataset for Insights

Captum Insights expects an iterable of `Batch` objects:
<div class="callout-insight">

<strong>Insight:</strong> Captum Insights expects an iterable of `Batch` objects:

</div>


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>
<div class="code-body">

```python
from captum.insights import Batch

def load_imagenet_samples(image_dir: str, transform, class_names: list, n=20):
    """Yield Batch objects from a directory of class-labeled images."""
    from PIL import Image
    from pathlib import Path

    for class_idx, class_name in enumerate(class_names):
        class_dir = Path(image_dir) / class_name
        if not class_dir.exists():
            continue
        for img_path in sorted(class_dir.glob("*.jpg"))[:2]:
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0)
            label_tensor = torch.tensor([class_idx])
            yield Batch(inputs=img_tensor, labels=label_tensor)
```

</div>

</div>

For a quick demo with synthetic images:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>
<div class="code-body">

```python
import torch
import numpy as np

def synthetic_image_dataset(n_batches=10, n_classes=10):
    """Generate synthetic image batches for Insights demo."""
    transform = T.Compose([T.Resize(224), T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    for i in range(n_batches):
        label = i % n_classes
        # Create a random image with label-correlated color
        arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        arr[:, :, label % 3] = np.random.randint(150, 250)  # color signal
        from PIL import Image
        img = Image.fromarray(arr)
        img_t = transform(img).unsqueeze(0)
        yield Batch(inputs=img_t, labels=torch.tensor([label]))
```

</div>

</div>

---

## 6. Starting the Insights Server

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>
<div class="code-body">

```python
visualizer = AttributionVisualizer(
    models=[model],
    score_func=score_func,
    classes=class_names,
    features=[image_feature],
    dataset=list(synthetic_image_dataset(n_batches=20)),
    num_examples=4,
)

# Start the Flask server
visualizer.serve(debug=False, port=5001)
```

</div>

</div>

Navigate to `http://localhost:5001` in your browser to access the Insights dashboard.

**Jupyter Notebook usage:**

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>
<div class="code-body">

```python
from captum.insights.widget import CaptumInsightsWidget

# In a Jupyter cell, render as widget
widget = CaptumInsightsWidget(visualizer)
widget.render()
```

</div>

</div>

---

## 7. Insights Interface Navigation

### Main Controls
- **Model selector:** Switch between multiple models (if registered)
- **Attribution method:** IG, Gradient×Input, Saliency, etc.
- **Target class:** Which output class to attribute to
- **Example navigation:** Browse through dataset batches

### Attribution Overlays
- **Heatmap:** Color-coded attribution over the input
- **Positive/negative masks:** Show only positive or negative contributions
- **Magnitude threshold:** Filter out low-attribution regions

### Comparison View
Run two attribution methods simultaneously and view side by side.

---

## 8. Text Classification Setup

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>
<div class="code-body">

```python
from captum.insights.attr_vis.features import TextFeature
from captum.attr import LayerIntegratedGradients

def text_score_func(model_output):
    return torch.softmax(model_output, dim=1)

def text_baseline_func(input_embeds):
    """Return zero-embedding baseline for text input."""
    return torch.zeros_like(input_embeds)

text_feature = TextFeature(
    name="Input Text",
    baseline_transforms=[text_baseline_func],
    input_transforms=[],  # tokenization handled separately
    visualization_transform=lambda ids: tokenizer.convert_ids_to_tokens(ids[0].tolist()),
)

text_visualizer = AttributionVisualizer(
    models=[bert_model],
    score_func=text_score_func,
    classes=["NEGATIVE", "POSITIVE"],
    features=[text_feature],
    dataset=list(text_dataset_iter()),
    num_examples=4,
)
```

</div>

</div>

---

## 9. Limitations and Workarounds

### Browser Requirements
Captum Insights requires a modern browser (Chrome, Firefox, Edge). Safari has compatibility issues with some visualization components.

### Large Models
For models with large inputs (high-resolution images, long documents), attribution computation in real-time can be slow. Solutions:
- Cache attributions on disk and load them
- Use a lighter attribution method (Saliency instead of IG)
- Reduce image resolution for interactive use

### Custom Architectures
Insights uses Captum's attribution methods internally. Models with unsupported layers (custom activations, non-standard connections) may encounter errors. Test your attribution method standalone before wiring into Insights.

### Deployment
Captum Insights is designed for local development and exploration, not production deployment. For serving attributions to end users, build a custom API (see guide 02).

---


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "What Is Captum Insights?" and why it matters in practice.

2. Given a real-world scenario involving captum insights: interactive web-based visualization, what would be your first three steps to apply the techniques from this guide?

</div>

## Summary

| Component | Purpose |
|-----------|---------|
| `AttributionVisualizer` | Main class wrapping model + attribution |
| `ImageFeature` | Descriptor for image inputs |
| `TextFeature` | Descriptor for text inputs |
| `GeneralFeature` | Descriptor for tabular/general inputs |
| `Batch` | Data container for Insights dataset |
| `visualizer.serve()` | Start Flask development server |
| `CaptumInsightsWidget` | Jupyter notebook embedded view |

---

## Further Reading

- Captum Insights documentation: https://captum.ai/docs/captum_insights
- Captum Insights GitHub examples: https://github.com/pytorch/captum/tree/master/captum/insights

---

## Cross-References

<a class="link-card" href="../notebooks/01_captum_insights_demo.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
