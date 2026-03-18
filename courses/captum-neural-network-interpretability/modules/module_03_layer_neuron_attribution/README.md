# Module 03: Layer & Neuron Attribution

## Overview

While input-level attribution methods (IG, saliency) explain the relationship between input features and predictions, layer and neuron attribution methods ask a deeper question: *what happens inside the network?* This module covers three complementary techniques for attributing predictions to intermediate representations:

1. **GradCAM / LayerGradCam** — which spatial regions in a convolutional layer drive the prediction?
2. **Layer Conductance** — which layer in a deep network is doing the most work?
3. **Neuron Conductance** — which specific neurons are responsible, and what do they detect?

## Learning Objectives

By completing this module, you will:

1. Apply `LayerGradCam` to produce class activation heatmaps for any convolutional layer
2. Compare GradCAM heatmaps across multiple layers to observe the CNN representation hierarchy
3. Compute `LayerConductance` across all ResNet-50 layers to identify the primary decision layer
4. Verify the completeness property of Layer Conductance across layers
5. Use `NeuronConductance` to understand what individual neurons in a layer detect
6. Classify neurons as class-selective vs shared using conductance-based selectivity scores

## Prerequisites

- Module 02 (Integrated Gradients) completed
- PyTorch familiarity with CNN architecture (residual blocks, conv layers)

## Estimated Time

- Guides: 45 minutes (2 guides × ~22 min each)
- Notebooks: 45 minutes (15 min × 3)
- Exercise: 25 minutes
- **Total: ~115 minutes**

## Contents

### Guides

| File | Topic | Time |
|------|-------|------|
| `guides/01_gradcam_guide.md` | GradCAM derivation, Guided GradCAM, LayerGradCam API | 22 min |
| `guides/01_gradcam_slides.md` | Companion slide deck (16 slides) | — |
| `guides/02_conductance_guide.md` | Layer Conductance, Neuron Conductance, Internal Influence | 23 min |
| `guides/02_conductance_slides.md` | Companion slide deck (14 slides) | — |

### Notebooks

| File | Topic | Time |
|------|-------|------|
| `notebooks/01_gradcam_resnet.ipynb` | GradCAM heatmaps across layers and classes, class IoU | 15 min |
| `notebooks/02_layer_conductance.ipynb` | Layer-level conductance comparison, spatial maps | 15 min |
| `notebooks/03_neuron_conductance.ipynb` | Individual neuron attribution, selectivity analysis | 15 min |

### Exercises

| File | Topic | Questions |
|------|-------|-----------|
| `exercises/01_layer_attribution_self_check.py` | GradCAM, Layer Conductance, Neuron Conductance API | 24 |

## Key Equations

**GradCAM Importance Weights:**
$$\alpha_k^c = \frac{1}{Z} \sum_u \sum_v \frac{\partial y^c}{\partial A^k_{uv}}$$

**GradCAM Class Activation Map:**
$$L^c_{\text{GradCAM}} = \text{ReLU}\!\left(\sum_k \alpha_k^c \cdot A^k\right)$$

**Layer Conductance:**
$$\text{Cond}_i^l = (h_i(x) - h_i(x')) \cdot \int_0^1 \frac{\partial y^c}{\partial h_i(x(\alpha))} \, d\alpha$$

**Completeness (every layer):**
$$\sum_i \text{Cond}_i^l = f(x) - f(x')$$

## Captum API Quick Reference

```python
from captum.attr import LayerGradCam, LayerAttribution
from captum.attr import LayerConductance, NeuronConductance

# GradCAM
lg = LayerGradCam(model, model.layer4[-1])
attr = lg.attribute(input_tensor, target=class_idx)
attr_up = LayerAttribution.interpolate(attr, (224, 224), 'bilinear')
heatmap = torch.relu(attr_up.sum(1)).squeeze()

# Layer Conductance
lc = LayerConductance(model, model.layer4[-1])
attr = lc.attribute(input_tensor, baselines=baseline,
                     target=class_idx, n_steps=50)
# attr.sum().item() ≈ f(x) - f(baseline)  [completeness]

# Neuron Conductance
nc = NeuronConductance(model, model.layer4[-1])
neuron_attr = nc.attribute(
    input_tensor,
    neuron_selector=(channel_idx, h, w),
    target=class_idx, baselines=baseline, n_steps=50
)
# neuron_attr: (1, 3, 224, 224) — input attribution for this neuron
```

## Method Comparison

| Method | Question | Speed | Axioms | Output |
|--------|----------|-------|--------|--------|
| GradCAM | Where? | Fast (1 pass) | None formal | Spatial heatmap |
| Layer Conductance | Which layer? | n_steps passes | Completeness | Per-neuron values |
| Neuron Conductance | Which neuron does what? | n_steps passes | Completeness | Input attribution |

## ResNet-50 Layer Reference

| Layer | Channels | Spatial | Notes |
|-------|----------|---------|-------|
| `layer1[-1]` | 256 | 56×56 | Edges, textures |
| `layer2[-1]` | 512 | 28×28 | Patterns, corners |
| `layer3[-1]` | 1024 | 14×14 | Object parts |
| `layer4[-1]` | 2048 | 7×7 | Semantic regions — best for GradCAM |

## Next Module

**Module 04 — Perturbation Methods:** Occlusion, Feature Ablation, and Shapley Value Sampling — gradient-free attribution methods that work on any model, including non-differentiable ones.
