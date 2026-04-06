# Attribution Method Selection: A Comprehensive Decision Framework

> **Reading time:** ~11 min | **Module:** 8 — Production Pipelines | **Prerequisites:** Modules 1-7


## Learning Objectives

By the end of this guide, you will be able to:
1. Select the appropriate attribution method given model architecture and use case
2. Understand the trade-offs between speed, accuracy, and theoretical guarantees
3. Apply the decision framework to novel model types
4. Choose the right baseline strategy for each method and domain
5. Combine methods strategically for comprehensive interpretability coverage


<div class="callout-key">

<strong>Key Concept Summary:</strong> This guide covers the core concepts of attribution method selection: a comprehensive decision framework.

</div>

---

## 1. The Core Trade-Off Space

Every attribution method makes trade-offs across four dimensions:

| Dimension | Fast end | Slow end |
|-----------|----------|----------|
| **Computation** | Saliency (1 backward pass) | KernelSHAP (1000+ forward passes) |
| **Faithfulness** | Saliency (no completeness) | IG (exact completeness) |
| **Stability** | IG (deterministic) | GradientSHAP (stochastic) |
| **Scope** | Input attribution | Concept attribution (TCAV) |

No method dominates all dimensions. The choice depends on what matters most for your use case.

---

## 2. The Primary Decision Tree

### Question 1: What kind of input does your model take?

```

Input type?
├── Image (CNN or ViT) → Section 3
├── Text (Transformer) → Section 4
├── Tabular / structured data → Section 5
└── Audio / time series → Section 6
```

### Question 2: What are you trying to do with the attribution?

```

Purpose?
├── Debug the model during development → Section 7
├── Explain a single prediction → Section 8
├── Regulatory / compliance reporting → Section 9
├── Research / method comparison → Section 10
└── Real-time production serving → Section 11
```

---

## 3. Image Models

### CNNs (ResNet, VGG, EfficientNet, MobileNet)
<div class="callout-warning">

<strong>Warning:</strong> **Architecture note:** GradCAM requires a convolutional layer. For the last conv layer (e.g., `model.layer4[-1]` in ResNet), it produces a class activation map that is upsampled to input resolution. For pixel-level attribution, use IG or GradientSHAP.

</div>


| Use Case | Recommended Method | Why |
|----------|--------------------|-----|
| Fast debugging | GradCAM | One backward pass, spatial resolution, no baseline |
| Single-example explanation | Integrated Gradients | Completeness guarantee, pixel-level |
| Baseline comparison | Saliency vs IG vs GradCAM | See which best captures known object regions |
| Stakeholder presentation | GradCAM overlay | Intuitive, coarse but clear |
| Compliance report | IG + convergence delta | Auditable attribution sum |

**Architecture note:** GradCAM requires a convolutional layer. For the last conv layer (e.g., `model.layer4[-1]` in ResNet), it produces a class activation map that is upsampled to input resolution. For pixel-level attribution, use IG or GradientSHAP.

### Vision Transformers (ViT, DeiT, Swin)

| Use Case | Recommended Method | Why |
|----------|--------------------|-----|
| Patch attribution | IG on input embeddings | Treats patches as features |
| Attention analysis | Attention rollout | Visualizes information flow |
| Layer importance | LayerConductance on transformer blocks | Identifies which blocks matter |
| Comparison | IG vs attention rollout | These often disagree — report both |

**Architecture note:** GradCAM is inapplicable to ViTs (no conv layers). Use `LayerGradCam` on attention layers only with care — attention is not activation in the GradCAM sense.

---

## 4. Text / Transformer Models

| Task | Recommended Method | Notes |
|------|--------------------|-------|
| Token attribution | LayerIntegratedGradients on embedding layer | Standard approach for BERT-family |
| Subword handling | Aggregate `##` tokens after attribution | Sum or mean of subword attribution scores |
| Baseline choice | `[PAD]` token ID for BERT, zero-embedding for GPT | Match pretraining convention |
| Attention exploration | Last-layer mean attention | Exploratory only — not attribution |
| Layer importance | LayerConductance over encoder layers | Identifies task-relevant depth |
| Multi-token output (seq2seq) | LayerIG with target per output step | Run separately per output token |
<div class="callout-key">

<strong>Key Point:</strong> Attribution and attention serve different purposes:

</div>


### The Attention Warning

Attribution and attention serve different purposes:

- **Attention:** Information routing weights. Tells you where the model looks during computation.
- **Attribution (IG, LayerIG):** Counterfactual feature importance. Tells you what features changed the output.

These disagree on negation, adversarial inputs, and compositional sentences. Use attention for exploration; use IG for explanation.

### Baseline Selection for Text


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# BERT-family: PAD baseline
baseline = torch.full_like(input_ids, tokenizer.pad_token_id)

# GPT-family: zero embedding baseline (cannot use token ID baseline)
def zero_embedding_baseline(inputs_embeds):
    return torch.zeros_like(inputs_embeds)

# Uniform random tokens (second choice for BERT)
baseline = torch.randint_like(input_ids, 0, tokenizer.vocab_size)
```

</div>

</div>

**Prefer PAD for BERT:** BERT was trained with `[PAD]` tokens in attention masking, so PAD produces near-neutral predictions — a well-defined reference point. GPT models have no PAD concept; use zero embeddings.

---

## 5. Tabular / Structured Data Models

| Architecture | Recommended Method | Notes |
|--------------|--------------------|-------|
| MLP (feedforward) | DeepLIFT or DeepLIFTSHAP | Exact propagation rules, fast |
| MLP | KernelSHAP | Model-agnostic, slower but SHAP-guaranteed |
| Gradient Boosted Trees | KernelSHAP | Cannot use gradient methods |
| Random Forest | KernelSHAP | Cannot use gradient methods |
| Tabular Transformer | IG on embedding inputs | Treat each feature embedding as the attribution target |

### Baseline for Tabular Data


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Option 1: Zero vector (if features are normalized)
baseline = torch.zeros_like(inputs)

# Option 2: Training set mean (most interpretable baseline)
baseline = train_features.mean(dim=0, keepdim=True).expand_as(inputs)

# Option 3: Median (robust to outliers)
baseline = train_features.median(dim=0).values.unsqueeze(0).expand_as(inputs)

# Option 4: Distribution sample (GradientSHAP)

# Use 50-100 random training examples as background
background = train_features[torch.randperm(len(train_features))[:100]]
```

</div>

</div>

**Recommendation:** Use training-set mean as baseline. It represents "a typical example" and makes attribution differences directly interpretable as deviation from average behavior.

---

## 6. Audio and Time Series

| Architecture | Recommended Method | Attribution Target |
|--------------|--------------------|--------------------|
| 1D CNN (wav2vec-style) | GradCAM on last conv layer | Time × channel heatmap |
| LSTM / GRU | IG on input embeddings | Per-timestep contribution |
| Temporal Transformer | LayerIG on embedding | Per-token (timestep) attribution |
| Spectrogram → CNN | IG or GradCAM | Time-frequency attribution |

**Key difference:** For sequential models, the baseline is often a silence vector (zeros for audio) or a seasonal mean (for time series). The attribution then shows which timesteps deviate from baseline behavior.

---

## 7. Method Selection by Use Case

### Development and Debugging
<div class="callout-key">

<strong>Key Point:</strong> **Goal:** Find spurious correlations quickly across many examples.

</div>


**Goal:** Find spurious correlations quickly across many examples.

**Recommended sequence:**
1. **Saliency** — 1 forward + 1 backward pass. Scan 50-100 examples in minutes.
2. **GradCAM** — Same speed, but coarser. Better for spatial pattern detection.
3. **IG (n_steps=50)** — Deeper dive on suspicious examples. Check completeness delta.

### Comparative Research

**Goal:** Understand method differences on your data.

**Recommended sequence:**
1. Run Saliency, IG, GradientSHAP, GradCAM on the same examples.
2. Compute Spearman rank correlation between unsigned attribution maps.
3. Focus analysis on examples where methods disagree (low correlation).
4. Disagreement reveals which aspects of the model each method captures differently.

---

### Regulatory Compliance / Audit

**Goal:** Produce auditable per-prediction explanations with verifiable properties.

**Requirements:**
- Completeness: attribution sums to $f(x) - f(x')$
- Determinism: same input → same attribution
- Stability: similar inputs → similar attributions

**Only IG satisfies all three requirements.**


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Compliance-grade configuration
ig = IntegratedGradients(model)
attrs, delta = ig.attribute(
    inputs, baseline,
    target=pred_class,
    n_steps=200,                   # more steps → smaller delta
    return_convergence_delta=True,
)
assert abs(delta.item()) < 0.05, f"Completeness violation: delta={delta.item()}"
```

</div>

</div>

### Real-Time Production Serving

**Goal:** Serve explanations with <100ms latency.

| Latency Budget | Method | Notes |
|----------------|--------|-------|
| <10ms | Saliency or GradCAM | Single backward pass |
| <50ms | GradientSHAP (n=10) | Stochastic, less stable |
| <200ms | IG (n_steps=20-30) | Reduced quality but fast |
| <500ms | IG (n_steps=50) | Production-quality |
| >500ms | IG (n_steps=100+) | Compliance/audit only |

---

## 8. Baseline Strategy Guide

The baseline is as important as the method. It defines the reference point: "What prediction does the model make when this feature is absent?"
<div class="callout-insight">

<strong>Insight:</strong> The baseline is as important as the method. It defines the reference point: "What prediction does the model make when this feature is absent?"

</div>


### Baseline Selection Matrix

| Domain | Standard Baseline | Alternative Baseline | When to Use Alternative |
|--------|------------------|----------------------|------------------------|
| Image (ImageNet) | Black image (zeros) | Blurred image | When black artifacts cause attribution on edges |
| Image (medical) | Mean image over dataset | Gaussian noise | When dataset has consistent background |
| Text (BERT) | PAD token baseline | MASK token baseline | MASK gives slightly more neutral predictions |
| Text (GPT) | Zero embeddings | Random embeddings | Random is noisier but forces stochastic coverage |
| Tabular | Training mean | All-zeros | Use zeros only if features are centered |
| Audio | Silence (zeros) | Pink noise | Pink noise when silence is unnatural for the domain |

### Multiple Baselines: DeepLIFTSHAP and GradientSHAP

For GradientSHAP and DeepLIFTSHAP, provide 50-200 background samples:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
from captum.attr import GradientShap

grad_shap = GradientShap(model)

# background: (n_background, *input_shape)
background = train_data[torch.randperm(len(train_data))[:100]]
attrs, delta = grad_shap.attribute(
    inputs, background, n_samples=50, target=pred_class,
    return_convergence_delta=True,
)
```



Multiple baselines reduce sensitivity to the choice of reference point.

---

## 9. The Complete Decision Flowchart

```

START: What is your model architecture?
│
├── CNN (ResNet, EfficientNet, MobileNet, VGG)
│   ├── Need spatial heatmap quickly? → GradCAM (Layer: last conv layer)
│   ├── Need pixel-level + completeness? → IG (baseline: black image)
│   ├── Need speed + Shapley properties? → GradientSHAP (bg: 50-100 train images)
│   └── Non-differentiable or tree-based? → KernelSHAP
│
├── Vision Transformer (ViT, DeiT, Swin)
│   ├── Patch-level attribution? → IG on input patch embeddings
│   ├── Attention visualization? → Attention rollout (exploration only)
│   └── Layer importance? → LayerConductance on transformer blocks
│
├── Text Transformer (BERT, RoBERTa, GPT, T5)
│   ├── Token attribution? → LayerIntegratedGradients (embedding layer)
│   ├── Layer importance? → LayerConductance over encoder/decoder layers
│   └── Neuron attribution? → NeuronIntegratedGradients on attention heads
│
└── Tabular / MLP
    ├── Need speed + exact? → DeepLIFT (baseline: training mean)
    ├── Need Shapley guarantees? → DeepLIFTSHAP (bg: training samples)
    ├── Need model-agnostic? → KernelSHAP (n_samples: 100-500)
    └── Non-differentiable? → KernelSHAP only

THEN: What is your purpose?
│
├── Debugging → Start with Saliency/GradCAM, then IG on suspicious cases
├── Single explanation → IG with completeness check (delta < 0.05)
├── Compliance → IG, n_steps=200, generate AttributionReport with delta
├── Real-time serving → Saliency or GradCAM (< 10ms); IG if budget allows
├── Research comparison → Run all methods, compute Spearman correlation
└── Concept analysis → TCAV (Module 06) complementing token/pixel attribution
```

---

## 10. When Methods Disagree

Attribution methods frequently produce different outputs on the same input. This is expected — they formalize different notions of "importance."

**Understanding disagreement:**

| Method A | Method B | Likely reason for disagreement |
|----------|----------|-------------------------------|
| Saliency | IG | Saliency is local gradient; IG integrates over path. Disagree in saturated regions. |
| IG | GradCAM | IG is pixel-level; GradCAM is class-activation upsampled. Different granularity. |
| IG | GradientSHAP | GradientSHAP is stochastic. Run multiple times and take mean for stability. |
| Attention | IG | Attention captures routing; IG captures counterfactual impact. Fundamentally different quantities. |

**When methods strongly disagree:** use the method with stronger theoretical guarantees for final reporting (IG > GradientSHAP > Saliency).

**When methods strongly agree:** that region is robustly important regardless of formalism — high confidence for reporting.

---

## 11. Quick-Reference Summary

| Method | Architecture | Completeness | Speed | Baseline Needed | SHAP Axioms |
|--------|-------------|--------------|-------|-----------------|-------------|
| Saliency | Any differentiable | No | Fastest | No | No |
| GradCAM | CNN only | No | Fastest | No | No |
| Gradient×Input | Any differentiable | No | Fast | Yes | Partial |
| Integrated Gradients | Any differentiable | Yes | Medium | Yes | Sensitivity + Linearity |
| DeepLIFT | MLP / CNN | Yes | Fast | Yes | Partial |
| DeepLIFTSHAP | MLP / CNN | Yes (expected) | Fast | Multiple | Yes |
| GradientSHAP | Any differentiable | Yes (expected) | Medium | Multiple | Yes |
| KernelSHAP | Any (black box) | Yes (expected) | Slow | Multiple | Full |
| LayerIG | Any + target layer | Yes | Medium | Yes | Sensitivity + Linearity |
| LayerConductance | Any + target layer | Yes | Medium | Yes | Sensitivity + Linearity |
| TCAV | Any + concept images | No | Slow | No | N/A |

---


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "The Core Trade-Off Space" and why it matters in practice.

2. Given a real-world scenario involving attribution method selection: a comprehensive decision framework, what would be your first three steps to apply the techniques from this guide?




## Further Reading

- "Axiomatic Attribution for Deep Networks" (Sundararajan et al., 2017) — IG foundations
- "A Unified Approach to Interpreting Model Predictions" (Lundberg & Lee, 2017) — SHAP
- "Grad-CAM" (Selvaraju et al., 2017) — GradCAM
- "Learning Important Features Through Propagating Activation Differences" (Shrikumar et al., 2017) — DeepLIFT
- Captum documentation: https://captum.ai/docs/algorithms_comparison_matrix

---

## Cross-References

<a class="link-card" href="../notebooks/01_captum_insights_demo.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
