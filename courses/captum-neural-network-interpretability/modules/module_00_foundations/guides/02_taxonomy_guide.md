# Taxonomy of Interpretability Methods

> **Reading time:** ~10 min | **Module:** 0 — Foundations | **Prerequisites:** PyTorch basics, neural network fundamentals


## In Brief

The interpretability landscape contains dozens of methods that answer fundamentally different questions. Choosing the wrong method produces an answer to a question nobody asked. This guide maps the full taxonomy and provides decision rules for selecting the right tool.

## Key Insight

Every interpretability method makes implicit choices about: what level of abstraction to explain (input, layer, or concept), whose predictions to explain (one instance or the whole model), and what relationship to maintain with the underlying model (use internal gradients or treat as black box). Understanding these dimensions eliminates confusion about which method to apply.


<div class="callout-key">

<strong>Key Concept Summary:</strong> The interpretability landscape contains dozens of methods that answer fundamentally different questions.

</div>

---

## 1. The Primary Taxonomy Dimensions

### Dimension 1: Intrinsic vs Post-hoc
<div class="callout-insight">

<strong>Insight:</strong> **Intrinsic interpretability** — The model *is* the explanation.

</div>


**Intrinsic interpretability** — The model *is* the explanation.

Intrinsically interpretable models are designed so their parameters directly encode human-readable knowledge:
- Linear regression: coefficients are the explanation
- Decision tree: the tree structure is the explanation
- Generalized Additive Models (GAMs): shape functions are the explanation

**Trade-offs:**
- Directly interpretable with no additional tooling
- May sacrifice predictive power on complex problems
- Architecture choice constrains the model

**Post-hoc interpretability** — Explanation is computed *after* training.

Post-hoc methods take a trained model and produce a separate explanation artifact:
- Saliency maps (gradient-based)
- SHAP values (game-theoretic)
- LIME (local surrogate models)
- Captum methods (the focus of this course)

**Trade-offs:**
- Works on any model regardless of complexity
- Explanation is an approximation, not the model itself
- Quality depends on method assumptions

### Dimension 2: Local vs Global

**Local explanation** — Explains one specific prediction for one specific input.

> "Why did the model classify *this particular image* as a cat?"

Local explanations are conditional on the input. Change the input, change the explanation.

**Use cases:**
- Debugging a specific wrong prediction
- Auditing a specific high-stakes decision
- User-facing explanations ("your loan was denied because...")

**Global explanation** — Explains the model's general behavior across all inputs.

> "Which features does the model consistently rely on across all predictions?"

Global explanations describe the aggregate behavior pattern, not any single decision.

**Use cases:**
- Model validation and conceptual soundness checks
- Regulatory documentation
- Feature engineering decisions
- Understanding the learned representation

**Relationship:** Global explanations can sometimes be constructed by aggregating local explanations (e.g., mean SHAP values over a dataset).

### Dimension 3: Model-specific vs Model-agnostic

**Model-specific methods** — Exploit the internal structure of a specific architecture.

Examples:
- GradCAM: requires convolutional layers with spatial structure
- Integrated Gradients: requires differentiability (gradient computation)
- Layer Conductance: requires access to intermediate activations

**Advantages:**
- Computationally efficient
- Can provide more precise, mechanistic explanations
- Often more faithful to actual model computation

**Limitations:**
- Only works on models with the required structure
- Different methods needed for CNNs vs transformers vs tabular networks

**Model-agnostic methods** — Treat the model as a black box and probe it with inputs/outputs.

Examples:
- LIME: builds a local linear approximation by perturbing inputs
- SHAP (KernelSHAP): estimates Shapley values by sampling input subsets
- Occlusion: systematically masks parts of the input

**Advantages:**
- Works on any model (even non-differentiable)
- Directly comparable across different model architectures
- Often more intuitive for non-technical stakeholders

**Limitations:**
- Computationally expensive (many forward passes required)
- Approximation quality depends on sampling strategy

---

## 2. Attribution Types Within Post-hoc Methods

Within post-hoc interpretability, there is a further taxonomy based on *what is being attributed*:
<div class="callout-warning">

<strong>Warning:</strong> Within post-hoc interpretability, there is a further taxonomy based on *what is being attributed*:

</div>


### Input Attribution (Feature Attribution)

Assigns importance scores to individual input features for a specific prediction.

$$\phi: \mathbb{R}^d \rightarrow \mathbb{R}^d$$

The attribution $\phi(x)_i$ represents how much input feature $i$ contributed to the prediction $f(x)$.

**Methods:** Integrated Gradients, SHAP, LIME, Saliency, Input×Gradient

**Output:** Same shape as input — a heatmap for images, per-token scores for text, per-feature scores for tabular data.

### Layer Attribution

Attributes importance to neurons or feature maps in a specific intermediate layer.

$$\phi_l: \mathbb{R}^d \rightarrow \mathbb{R}^{|h_l|}$$

where $h_l$ is the activation at layer $l$.

**Methods:** GradCAM, Layer Conductance, Internal Influence

**Output:** One importance score per unit in the target layer.

### Neuron Attribution

Attributes importance to input features *for a specific neuron's activation*, rather than the final output.

$$\phi_n: \mathbb{R}^d \rightarrow \mathbb{R}^d$$

where the target is maximizing neuron $n$'s activation rather than the model output.

**Methods:** Neuron Conductance, Neuron Integrated Gradients

**Output:** Input-space attribution map explaining what makes a neuron activate.

---

## 3. Full Taxonomy Map

```

Interpretability Methods
├── Intrinsic
│   ├── Linear models (coefficients)
│   ├── Decision trees (tree structure)
│   ├── GAMs (shape functions)
│   └── Attention weights (debated)
│
└── Post-hoc
    ├── Local
    │   ├── Gradient-based (model-specific)
    │   │   ├── Saliency (vanilla gradient)
    │   │   ├── Input × Gradient
    │   │   ├── Guided Backpropagation
    │   │   └── Integrated Gradients ← Captum focus
    │   │
    │   ├── Propagation-based (model-specific)
    │   │   ├── GradCAM ← Captum focus
    │   │   ├── Layer Conductance ← Captum focus
    │   │   └── DeepLIFT
    │   │
    │   └── Perturbation-based (model-agnostic)
    │       ├── LIME
    │       ├── SHAP (KernelSHAP)
    │       ├── Occlusion ← Captum focus
    │       └── Feature Ablation ← Captum focus
    │
    └── Global
        ├── Aggregate attribution (mean |SHAP|)
        ├── Concept-based (TCAV) ← Captum focus
        ├── Probing classifiers
        └── Activation maximization
```

---

## 4. Decision Guide: When to Use Each Method

### For Image Classification

| Goal | Method | Why |
|------|--------|-----|
| Quick pixel attribution | Saliency / Guided Backprop | Fast, no baseline needed |
| High-fidelity attribution | Integrated Gradients | Satisfies axioms, with baseline |
| Spatial heatmap | GradCAM | Coarser but robust to noise |
| Debugging wrong prediction | GradCAM + IG overlay | GradCAM locates region, IG explains it |
| Verify model focus | Occlusion | Model-agnostic, intuitive |

### For Text Classification

| Goal | Method | Why |
|------|--------|-----|
| Token importance | Integrated Gradients | Works naturally on embeddings |
| Attention visualization | Attention rollout | Direct from transformer internals |
| Model-agnostic token importance | LIME | No gradient required |

### For Tabular Data

| Goal | Method | Why |
|------|--------|-----|
| Feature importance ranking | Feature Ablation | Direct, interpretable |
| Prediction explanation | SHAP (KernelSHAP) | Game-theoretically grounded |
| Sensitivity analysis | Input × Gradient | Fast, continuous |
| Regulatory documentation | Shapley values | Consistent, additive decomposition |

### By Computational Budget

| Budget | Method |
|--------|--------|
| Very fast (single backward pass) | Saliency, Input×Gradient |
| Fast (few backward passes) | Integrated Gradients (50-300 steps) |
| Medium (many forward passes) | Occlusion, Feature Ablation |
| Expensive (many model evaluations) | SHAP, LIME |

---

## 5. Method Comparison by Property

### Formal Axiomatic Properties

Sundararajan et al. (2017) identified two axioms that any attribution method should satisfy:

**Sensitivity (Completeness):** If changing input feature $i$ changes the output, then $\phi_i \neq 0$.

**Implementation Invariance:** Two models that produce identical outputs for all inputs should receive identical attributions, regardless of their internal implementation.

| Method | Sensitivity | Implementation Invariance |
|--------|------------|--------------------------|
| Integrated Gradients | Yes | Yes |
| Saliency | No | Yes |
| Input × Gradient | No | Yes |
| Guided Backprop | No | No |
| Occlusion | Yes | Yes |
| SHAP | Yes | Yes |

Integrated Gradients is uniquely satisfying both axioms among gradient-based methods.

---

## 6. Practical Considerations

### The Baseline Problem
<div class="callout-key">

<strong>Key Point:</strong> Gradient-based attribution methods require a reference point (baseline) representing the "absence of information." The choice of baseline affects all attributions.

</div>


Gradient-based attribution methods require a reference point (baseline) representing the "absence of information." The choice of baseline affects all attributions.

Common baseline choices:
- **Zero baseline:** All input values set to 0. Simple but may not correspond to a meaningful "no information" state.
- **Blurred image baseline:** A heavily blurred version of the image. Removes local structure while preserving global statistics.
- **Random noise baseline:** Average over multiple random baselines to reduce dependence on any single choice.
- **Domain-specific baselines:** A black image for visual models, a [MASK] token for text models, feature mean for tabular data.

### Computational Cost Comparison

For a single prediction explanation with a ResNet-50 on a single GPU:

| Method | Approximate Time | Notes |
|--------|-----------------|-------|
| Saliency | ~1ms | Single backward pass |
| Input × Gradient | ~1ms | Single backward pass |
| Integrated Gradients (50 steps) | ~50ms | 50 forward+backward passes |
| GradCAM | ~5ms | One forward pass + hook |
| Occlusion (8×8 window) | ~2-5 seconds | Many forward passes |
| SHAP (KernelSHAP) | ~10-60 seconds | Many forward passes with sampling |

### Faithfulness vs Interpretability

There is a tension between methods that are faithful to the model and methods that produce clean, human-interpretable visualizations:

- **Faithful methods** (Integrated Gradients, SHAP) accurately represent model computation but may produce noisy attributions
- **Clean visualization methods** (GradCAM, Guided Backprop) produce smooth, visually clean heatmaps but may not accurately reflect model computation

In practice: use faithful methods for validation/debugging, use clean methods for stakeholder communication.

---

## 7. Captum's Coverage of the Taxonomy

Captum (PyTorch's interpretability library) provides implementations spanning the full taxonomy:

**Gradient-based:**
- `Saliency` — vanilla gradient magnitude
- `InputXGradient` — input times gradient
- `GuidedBackprop` — modified gradient backpropagation
- `Deconvolution` — deconvolution-based attribution
- `IntegratedGradients` — path integral attribution

**Layer-based:**
- `LayerGradCam` — gradient-weighted class activation maps
- `LayerConductance` — conductance through a layer
- `InternalInfluence` — attribution to internal layer outputs
- `LayerActivation` — raw activation values

**Neuron-based:**
- `NeuronConductance` — conductance to a specific neuron
- `NeuronIntegratedGradients` — IG targeting a neuron

**Perturbation-based:**
- `FeatureAblation` — systematic feature removal
- `Occlusion` — sliding window masking
- `ShapleyValueSampling` — approximate Shapley values
- `FeaturePermutation` — permutation importance

**Concept-based:**
- `TCAV` — testing with concept activation vectors

**Noise Tunnel:**
- `NoiseTunnel` — wraps any method with SmoothGrad or VarGrad variance reduction

---

## Common Pitfalls

- **Local ≠ Global:** A local explanation tells you about *this* prediction, not the model in general. Averaging local explanations can give misleading global views.
- **Attribution ≠ Causation:** High attribution for a feature means the model uses it, not that the feature causally affects the outcome in the real world.
- **Method shopping:** Running multiple methods and reporting only the one with the cleanest-looking output introduces selection bias in your explanation.
- **Ignoring baselines:** For gradient-based methods, the baseline is a model parameter as important as the architecture.

---

## Connections

- **Builds on:** Guide 01 (why interpretability matters), neural network fundamentals
- **Leads to:** Guide 03 (Captum library), Module 01 (gradient methods in depth)
- **Related to:** Explainability in fairness literature, Model documentation standards

---


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Key Insight" and why it matters in practice.

2. Given a real-world scenario involving taxonomy of interpretability methods, what would be your first three steps to apply the techniques from this guide?

</div>

## Further Reading

- Sundararajan et al. (2017). Axiomatic Attribution for Deep Networks. *ICML 2017* — The foundational paper for Integrated Gradients and the sensitivity/implementation invariance axioms.
- Doshi-Velez & Kim (2017). Towards a Rigorous Science of Interpretable Machine Learning. *arXiv* — Framework for evaluating interpretability methods.
- Samek et al. (2019). Explainable AI: Interpreting, Explaining and Visualizing Deep Learning. *Springer* — Comprehensive textbook.
- Rudin (2019). Stop Explaining Black Box Machine Learning Models for High Stakes Decisions. *Nature Machine Intelligence* — Argues for intrinsic interpretability in high-stakes settings.

---

## Cross-References

<a class="link-card" href="../notebooks/01_environment_setup.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
