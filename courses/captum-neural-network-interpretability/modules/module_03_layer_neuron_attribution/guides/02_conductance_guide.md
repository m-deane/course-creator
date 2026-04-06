# Guide 02: Layer Conductance, Neuron Conductance, and Internal Influence

> **Reading time:** ~9 min | **Module:** 3 — Layer & Neuron Attribution | **Prerequisites:** Module 2 Integrated Gradients


## Overview

GradCAM tells us *where* in the image the model looks. Conductance methods tell us *which layer* and *which neurons* within that layer are doing the most work. Layer Conductance decomposes the model's output prediction across all layers in the network. Neuron Conductance identifies which specific neurons in a layer are responsible for a prediction. Together, they provide an inside-out view of neural network computation.


<div class="callout-key">
<strong>Key Concept Summary:</strong> This guide covers the core concepts of guide 02: layer conductance, neuron conductance, and internal influence.
</div>

---

## 1. Motivation: What Happens Between Input and Output?

Attribution methods like IG and GradCAM explain the relationship between input and output. But neural networks are deep chains of transformations. Understanding *which intermediate representations* are responsible for a prediction is a separate and important question.

**Example:** A ResNet-50 has 50 layers. When it correctly classifies a Golden Retriever, is the decision primarily made in layer4 (semantic regions) or earlier? Does layer3 contribute substantially? Answering this requires *internal attribution* — attributing the output to intermediate layers, not to input features.

This is the problem Layer Conductance solves.

---

## 2. Internal Influence: The Building Block

Before Conductance, consider **Internal Influence** — a simpler concept.

For a neuron $h_i$ in layer $l$, the internal influence on output $y^c$ is:

$$\text{II}_i = \frac{\partial y^c}{\partial h_i^l}$$

This is simply the gradient of the output with respect to the hidden unit's activation. It answers: "if $h_i$ changed by a small amount, how would $y^c$ change?"

**Limitation:** Like saliency at the input level, Internal Influence only looks at the local gradient. It does not account for how much $h_i$ actually changes between the baseline and the input.

---

## 3. Layer Conductance

Layer Conductance extends Integrated Gradients to intermediate layers. For each neuron $h_i$ in layer $l$:
<div class="callout-warning">
<strong>Warning:</strong> Layer Conductance extends Integrated Gradients to intermediate layers. For each neuron $h_i$ in layer $l$:
</div>


$$\text{Cond}_i = \underbrace{(h_i(x) - h_i(x'))}_{\text{activation change}} \cdot \int_0^1 \frac{\partial y^c}{\partial h_i(x(\alpha))} \, d\alpha$$

where $x(\alpha) = x' + \alpha(x - x')$ is the interpolation path from baseline $x'$ to input $x$.

This is precisely IG applied at the hidden layer, rather than at the input. The two terms capture:
1. **How much the neuron's activation changes** when moving from baseline to input
2. **How much that change matters** to the output (via the integrated gradient)

### Completeness of Layer Conductance

Layer Conductance satisfies a generalized completeness property:

$$\sum_i \text{Cond}_i^l = f(x) - f(x')$$

for **every layer** $l$ in the network. This means that summing the conductance values across all neurons in any single layer recovers the total prediction difference. This is a remarkable property: it means you can decompose the output by layer without double-counting.

### Proof sketch

By the chain rule, the IG integral telescopes through the network:

$$\int_0^1 \frac{\partial y^c}{\partial x_i} d\alpha = \sum_j \int_0^1 \frac{\partial y^c}{\partial h_j^l} \frac{\partial h_j^l}{\partial x_i} d\alpha$$

Summing over neurons $j$ in layer $l$ and applying the FTC to the layer activation change recovers the same integral as input-level IG.

---

## 4. Captum Layer Conductance API

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
from captum.attr import LayerConductance

model = resnet50(weights='IMAGENET1K_V1').eval()

# Measure conductance at the third residual block
target_layer = model.layer3[-1]

lc = LayerConductance(model, target_layer)

# Compute layer conductance
attr = lc.attribute(
    input_tensor,          # (1, 3, 224, 224)
    baselines=baseline,    # Zero baseline
    target=class_idx,      # Target class
    n_steps=50
)
# attr shape: (1, C, H, W) for convolutional layers
# attr shape: (1, N)       for fully connected layers
```

</div>
</div>

### Interpreting the Output

For a convolutional layer (e.g., `model.layer3[-1]`):
- Output shape: `(1, 1024, 14, 14)` for ResNet-50 layer3
- Each value: conductance of that specific neuron at that spatial location
- Aggregate to compare layers:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
# Total conductance of this layer (sum of absolute values)
layer_total_conductance = attr.abs().sum().item()

# Spatial map: which spatial locations have high conductance?
spatial_map = attr.abs().sum(dim=1).squeeze()  # (14, 14)

# Channel map: which channels have high conductance?
channel_importance = attr.abs().mean(dim=(-2, -1)).squeeze()  # (1024,)
```

</div>
</div>

---

## 5. Comparing Conductance Across Layers

The completeness property allows fair comparison across layers:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
layers_to_examine = {
    'layer1': model.layer1[-1],
    'layer2': model.layer2[-1],
    'layer3': model.layer3[-1],
    'layer4': model.layer4[-1],
}

layer_conductances = {}
for name, layer in layers_to_examine.items():
    lc = LayerConductance(model, layer)
    attr = lc.attribute(
        input_tensor, baselines=baseline,
        target=class_idx, n_steps=50
    )
    # Total signed conductance (sum, not abs)
    layer_conductances[name] = attr.sum().item()

# Which layer contributes most to the prediction?
print(layer_conductances)
```

</div>
</div>

Because conductance is complete, the values across all layers should be consistent with the total prediction difference $f(x) - f(x')$.

**Expected finding for ResNet-50:** layer4 typically has the highest conductance for ImageNet classification, confirming it is where the semantic decision is made. However, for fine-grained tasks (e.g., distinguishing dog breeds), layer3 and layer4 may have comparable conductance.

---

## 6. Neuron Conductance

While Layer Conductance measures the importance of entire layers, **Neuron Conductance** measures the importance of individual neurons:
<div class="callout-warning">
<strong>Warning:</strong> While Layer Conductance measures the importance of entire layers, **Neuron Conductance** measures the importance of individual neurons:
</div>


$$\text{NeuronCond}_i = (h_i(x) - h_i(x')) \cdot \int_0^1 \frac{\partial y^c}{\partial h_i(x(\alpha))} \, d\alpha$$

This is the same formula as Layer Conductance, but interpreted at the level of a single neuron rather than aggregated across the whole layer.

### What Can Neuron Conductance Tell Us?

1. **Which neurons are polysemantic?** A neuron with high conductance for both "dog" and "cat" is responding to shared features (fur texture, animal shape). A neuron with high conductance only for "dog" is class-specific.

2. **Which neurons are dormant for a given input?** Neurons with near-zero conductance are not contributing to this prediction — regardless of their weights.

3. **Are certain neurons specialized?** Some neurons in language models and CNNs are known to correspond to interpretable concepts ("curve detectors", "head detectors"). Neuron conductance identifies these for a given prediction.

---

## 7. Captum Neuron Conductance API

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">
<div class="callout-key">
<strong>Key Point:</strong> from captum.attr import NeuronConductance
</div>


```python
from captum.attr import NeuronConductance

# Examine a specific layer
target_layer = model.layer4[-1]

nc = NeuronConductance(model, target_layer)

# Conductance for a specific neuron
# For a conv layer, neuron index is (channel, height, width)
neuron_attr = nc.attribute(
    input_tensor,
    neuron_selector=(42, 3, 3),  # Channel 42, spatial location (3, 3)
    target=class_idx,
    baselines=baseline,
    n_steps=50
)
# neuron_attr: (1, 3, 224, 224) — attribution on INPUT for this neuron
```

</div>
</div>

Neuron Conductance returns the *input attribution* for a single intermediate neuron. It answers: "which input features caused neuron 42 at position (3,3) in layer4 to activate as it did?"

### Finding the Most Important Neurons

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
def top_neurons_for_prediction(model, layer, input_tensor, baseline,
                                class_idx, n_neurons=10):
    """
    Find the n_neurons most important for a prediction.
    Returns list of (conductance, channel, h, w) tuples.
    """
    lc = LayerConductance(model, layer)
    attr = lc.attribute(
        input_tensor, baselines=baseline,
        target=class_idx, n_steps=50
    )
    # attr: (1, C, H, W)

    # Get absolute conductance per neuron
    cond_flat = attr.abs().squeeze(0).view(-1)  # (C*H*W,)
    top_flat_idxs = cond_flat.topk(n_neurons).indices

    results = []
    C, H, W = attr.squeeze(0).shape
    for flat_idx in top_flat_idxs:
        c = (flat_idx // (H * W)).item()
        hw = flat_idx % (H * W)
        h = (hw // W).item()
        w = (hw % W).item()
        cond = attr[0, c, h, w].item()
        results.append((abs(cond), c, h, w))

    return sorted(results, reverse=True)
```

</div>
</div>

---

## 8. Layer Activation Analysis

Before computing conductance, it is useful to understand the distribution of activations in each layer. Neurons with zero activation have zero conductance regardless of weights.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
# Get activations without computing attribution
activations = {}

def make_hook(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

hooks = []
for name, layer in layers_to_examine.items():
    hooks.append(layer.register_forward_hook(make_hook(name)))

with torch.no_grad():
    _ = model(input_tensor)

for hook in hooks:
    hook.remove()

# Analyze activation statistics
for name, act in activations.items():
    n_active = (act.abs() > 0.01).float().mean().item()
    print(f"{name}: {n_active:.1%} neurons active, "
          f"mean={act.mean():.3f}, std={act.std():.3f}")
```

</div>
</div>

---

## 9. Conductance vs. Activation Patching

Conductance measures importance via gradients. An alternative is **activation patching** (causal mediation analysis):

**Activation patching:**
1. Run model on input A → record layer $l$ activations $h_l(A)$
2. Run model on input B → at layer $l$, *replace* activations with $h_l(A)$
3. Measure output change

If patching layer $l$ of input B with activations from input A changes the output toward A's prediction, layer $l$ is causally responsible for the difference.

| | Layer Conductance | Activation Patching |
|--|--|--|
| Method | Gradient-based | Intervention-based |
| Interpretation | Sensitivity at each layer | Causal responsibility |
| Computational cost | n_steps passes | 1 pass per patch |
| Axioms | Completeness | Causal faithfulness |

Activation patching is popular in mechanistic interpretability research (circuits). Layer Conductance is standard for production explainability.

---

## 10. InternalInfluence in Captum

Captum also provides `InternalInfluence`, which is the simpler gradient-based internal attribution (no integration):

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
from captum.attr import InternalInfluence

ii = InternalInfluence(model, target_layer)

attr = ii.attribute(
    input_tensor,
    target=class_idx
)
# Gradient of output w.r.t. layer activations
```

</div>
</div>

`InternalInfluence` is faster than `LayerConductance` (no integration) but does not satisfy completeness. Use it for rapid exploration; use `LayerConductance` for rigorous analysis.

---

## 11. Connecting Layer Attribution to Module 02 (IG)

Layer Conductance generalizes IG:

| Method | What it attributes | Attribution target |
|--------|-------------------|-------------------|
| IG | Output to input features | Input pixels/tokens |
| Layer Conductance | Output to layer neurons | Any intermediate layer |
| Neuron Conductance | Output to input features for one neuron | Input, for single neuron |

All three share the same mathematical foundation (IG integral), completeness property, and the same baseline requirement. If you understand IG, you understand Layer Conductance — it is the same computation applied at a different layer.

---


---

## Practice Questions

<div class="callout-info">
<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Motivation: What Happens Between Input and Output?" and why it matters in practice.

2. Given a real-world scenario involving guide 02: layer conductance, neuron conductance, and internal influence, what would be your first three steps to apply the techniques from this guide?
</div>

## Summary

1. **Internal Influence** = gradient of output w.r.t. hidden activations (fast, no integration)
2. **Layer Conductance** = IG applied at intermediate layers, satisfies completeness for every layer
3. **Neuron Conductance** = conductance for a single neuron, returns input attribution for that neuron
4. **Completeness:** $\sum_i \text{Cond}_i^l = f(x) - f(x')$ for any layer $l$
5. **Use case:** find the "decision-making" layer, identify specialized neurons, debug deep networks

---

## Further Reading

- Dhamdhere et al., "How Important is a Neuron?", ICLR 2019 — original Layer Conductance paper
- Elhage et al., "A Mathematical Framework for Transformer Circuits", Anthropic 2021 — mechanistic interpretability
- Bau et al., "Network Dissection: Quantifying Interpretability of Deep Visual Representations", CVPR 2017 — neuron concept detection
- Captum Layer Attribution documentation: https://captum.ai/api/layer.html

---

## Cross-References

<a class="link-card" href="../notebooks/01_gradcam_resnet.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
