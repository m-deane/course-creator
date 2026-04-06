# TCAV: Testing with Concept Activation Vectors

> **Reading time:** ~7 min | **Module:** 6 — Concept & Example Explanations | **Prerequisites:** Module 3 Layer Attribution


## Learning Objectives

By the end of this guide, you will be able to:
1. Explain the motivation for concept-based explanations vs. feature attribution
2. Describe how Concept Activation Vectors (CAVs) encode human-interpretable concepts
3. Implement TCAV in Captum to test whether a model uses a specific concept
4. Interpret TCAV sensitivity scores and statistical significance
5. Design meaningful concept datasets for TCAV experiments


<div class="callout-key">

<strong>Key Concept Summary:</strong> This guide covers the core concepts of tcav: testing with concept activation vectors.

</div>

---

## 1. Why Concept-Based Explanations?

Feature attribution methods (saliency, IG, SHAP) explain predictions in terms of input pixels or features. This works for understanding individual predictions but fails to answer higher-level questions:

- "Does this tumor classifier rely on texture or shape?"
- "Does this skin lesion model use a 'blue-white veil' concept like dermatologists do?"
- "Is the bird classifier using the correct features (beak shape, wing pattern) rather than spurious ones (image background)?"

TCAV (Kim et al., 2018) answers these questions by testing whether the model's internal representations respond to human-defined concepts.

---

## 2. The Core TCAV Idea

TCAV asks: **"How sensitive is the model's prediction of class C to a concept K?"**
<div class="callout-insight">

<strong>Insight:</strong> TCAV asks: **"How sensitive is the model's prediction of class C to a concept K?"**

</div>


### Step 1: Define a Concept

A concept is defined as a set of example images/inputs that exemplify the concept:
- Concept "striped": images of striped textures
- Concept "curved": images of curved objects
- Concept "furry": images of furry animals

### Step 2: Fit a Concept Activation Vector

For each concept, a linear classifier (CAV) is trained to distinguish concept examples from random non-concept examples in a model's intermediate layer activations:

$$\text{CAV}_{K,l} = \text{LinearProbe}(\{f_l(x) : x \in \text{concept}_K\}, \{f_l(x) : x \in \text{random}\})$$

where $f_l(x)$ is the activation of layer $l$ on input $x$.

### Step 3: Compute TCAV Score

The TCAV score measures the fraction of inputs in class $C$ whose activation gradient in the direction of the CAV is positive:

$$\text{TCAV}_{K,C,l} = \frac{|\{x \in C : \nabla_{f_l} F_C(x) \cdot v_{K,l} > 0\}|}{|C|}$$

where:
- $F_C(x)$ = model's predicted probability for class $C$
- $\nabla_{f_l} F_C(x)$ = gradient of $F_C$ with respect to layer $l$ activations
- $v_{K,l}$ = the learned CAV direction

A TCAV score of 0.8 means 80% of class $C$ images have activations that increase when perturbed in the concept direction — the model is using concept $K$ for class $C$.

---

## 3. The Directional Derivative

The key computation in TCAV is the **directional derivative** of the model output with respect to the concept direction:

$$S_{K,C,l}(x) = \nabla_{f_l} F_C(x) \cdot v_{K,l}$$

**Positive $S_{K,C,l}$:** The model's prediction of class $C$ increases when we move activations in the concept direction. The model uses this concept positively for this class.

**Negative $S_{K,C,l}$:** The concept is anti-correlated with the model's prediction for class $C$.

The TCAV score aggregates this over the full class:

$$\text{TCAV}_{K,C,l} = \frac{|\{x \in C : S_{K,C,l}(x) > 0\}|}{|C|}$$

---

## 4. Statistical Testing: Is the CAV Meaningful?

A critical issue: **random concept datasets can also produce high TCAV scores by chance**.

TCAV uses a two-sided hypothesis test:
- **Null hypothesis**: The concept is random (TCAV ≈ 0.5 from random CAVs)
- **Test**: Compare TCAV score against distribution from multiple random CAV runs

The Captum implementation trains CAVs multiple times with different random concept splits and tests for statistical significance, filtering out concepts that could be explained by chance.

---

## 5. Concept Dataset Design

The quality of TCAV results depends heavily on the concept dataset.

### Good Concept Datasets
- Clear, unambiguous examples of the concept
- Diverse examples (different positions, scales, backgrounds)
- Minimum 20-50 examples per concept
- Concept images from the same distribution as the test class

### Common Pitfalls
- Too few examples (< 15): noisy CAV, unreliable results
- Confounded concepts: "striped" images that also tend to be "colorful"
- Concept images from a different distribution than the model's training data

### Example Concept Sources
- **ImageNet:** Use specific synset subsets as concept examples
- **CIFAR-10:** Use class subsets as concepts (e.g., "vehicle" = car+truck)
- **Custom datasets:** Curate images manually for abstract concepts
- **Synthetic:** Generate concept-specific textures using tools like PIL

---

## 6. Captum TCAV Implementation


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from captum.concept import TCAV, Concept

# Step 1: Define concepts as Captum Concept objects
def make_concept(concept_images: torch.Tensor, concept_id: int, name: str) -> Concept:
    """Wrap concept images in a Captum Concept object."""
    dataset = TensorDataset(concept_images)
    loader = DataLoader(dataset, batch_size=32)
    return Concept(id=concept_id, name=name, data_iter=loader)

# Step 2: Instantiate TCAV with model and layer
tcav = TCAV(
    model=model,
    layers=["layer4"],  # ResNet layer 4 (penultimate)
    model_id="resnet18_imagenet",
    save_path="./tcav_results/",
)

# Step 3: Run TCAV experiment
experimental_sets = [
    [concept_striped, random_concept_0],
    [concept_striped, random_concept_1],
    [concept_striped, random_concept_2],  # multiple random for significance testing
    [concept_dotted, random_concept_0],
    [concept_curved, random_concept_0],
]

# Class to test (e.g., zebra = class 340 in ImageNet)
target_class = torch.tensor([340])

scores = tcav.interpret(
    inputs=class_images,
    experimental_sets=experimental_sets,
    target=target_class,
    n_steps=5,
)
```

</div>

</div>

---

## 7. Interpreting TCAV Results

### Reading the Scores


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# scores is a dict: layer -> concept_name -> TCAV score
for layer, concept_scores in scores.items():
    for concept_name, score in concept_scores.items():
        print(f"Layer: {layer}, Concept: {concept_name}, TCAV: {score:.3f}")
```

</div>

</div>

### Example Output for a Zebra Classifier

```

Layer: layer4, Concept: striped,  TCAV: 0.847  ← high! model uses stripes
Layer: layer4, Concept: dotted,   TCAV: 0.523  ← ~random, not using dots
Layer: layer4, Concept: curved,   TCAV: 0.612  ← moderate use of curves
Layer: layer4, Concept: 4-legged, TCAV: 0.731  ← model uses leg structure
```

**Score of 0.847:** 84.7% of zebra images have activations that increase toward the "striped" concept direction — strong evidence the model uses stripes for zebra classification.

**Score ~0.5:** The concept is not systematically used (could be random).

---

## 8. Multi-Layer TCAV Analysis

Running TCAV on multiple layers reveals at which depth the concept is encoded:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
tcav = TCAV(
    model=model,
    layers=["layer1", "layer2", "layer3", "layer4"],
    model_id="resnet18",
    save_path="./tcav_multilayer/",
)
```



Typical findings:
- **Early layers** (layer1, layer2): Encode low-level concepts (edges, colors, textures)
- **Late layers** (layer3, layer4): Encode high-level concepts (object parts, shapes)

Plotting TCAV scores by layer shows where each concept becomes relevant to the prediction.

---

## 9. TCAV Limitations

### Requires Labeled Concept Sets
TCAV cannot discover concepts automatically — you must define what to test. This requires domain knowledge and careful curation.

### Linear Concept Assumption
CAVs use linear probes. If a concept is encoded non-linearly in a layer's representation, TCAV may report low sensitivity even if the model uses the concept.

### Concept Ambiguity
Concepts like "texture" are multi-dimensional. A single CAV captures only one direction in the activation space, potentially missing important aspects of the concept.

### Spurious Correlations in Concept Datasets
If your "striped" images happen to all be grayscale, the CAV may encode "grayscale" rather than "stripes."

---

## 10. TCAV vs. Attribution Methods

| Aspect | TCAV | Attribution (IG/SHAP) |
|--------|------|---------------------|
| Explanation type | Concept-level (global) | Feature-level (local) |
| Granularity | Which concepts does the model use? | Which pixels drove this prediction? |
| Human interpretability | High (concepts = words) | Lower (pixel heatmaps) |
| Requires concept labels | Yes | No |
| Model access | Layer activations + gradients | Gradients only |
| Suitable for | Bias auditing, concept testing | Individual predictions |

TCAV and attribution methods are **complementary**:
- Use TCAV to test if a model uses a known problematic concept (e.g., "skin color" for a medical model)
- Use attribution to explain individual predictions

---


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Why Concept-Based Explanations?" and why it matters in practice.

2. Given a real-world scenario involving tcav: testing with concept activation vectors, what would be your first three steps to apply the techniques from this guide?


## Summary

| Step | Description | Key Formula |
|------|-------------|-------------|
| Define concept | Curate positive examples + random negatives | — |
| Train CAV | Linear probe on layer activations | $v_{K,l} = \text{LinearProbe}(f_l(\text{concept}), f_l(\text{random}))$ |
| Compute directional derivative | Gradient in CAV direction | $S_{K,C,l}(x) = \nabla_{f_l} F_C(x) \cdot v_{K,l}$ |
| TCAV score | Fraction with positive directional derivative | $\text{TCAV} = \|S > 0\| / \|C\|$ |
| Significance test | Compare to random CAV distribution | Two-sided test, $p < 0.05$ |

---

## Further Reading

- Kim, B., et al. (2018). Interpretability beyond classification uncertainty: Quantitative testing with concept activation vectors (TCAV). *ICML*.
- Ghassemi, M., et al. (2021). The false hope of current approaches to explainable artificial intelligence in health care. *The Lancet Digital Health*.
- Captum TCAV documentation: https://captum.ai/api/concept.html

---

## Cross-References

<a class="link-card" href="../notebooks/01_tcav_texture_shape.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
