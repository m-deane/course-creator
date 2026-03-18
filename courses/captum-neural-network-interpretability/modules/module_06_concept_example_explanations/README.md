# Module 06 — Concept-Based & Example-Based Explanations

## Overview

This module covers two complementary explanation paradigms that go beyond pixel-level attribution:

1. **Concept-based explanations (TCAV):** Test whether a model uses human-interpretable concepts (stripes, texture, curved shapes) for a given prediction class
2. **Example-based explanations (TracIn, SimilarityInfluence):** Find training examples that most influenced a prediction

## Prerequisites

- Module 01-04: Attribution methods basics
- Module 05: SHAP and gradient methods
- Familiarity with linear probes and cosine similarity

## Module Contents

### Guides
| File | Topic |
|------|-------|
| `guides/01_tcav_guide.md` | TCAV theory, CAV training, directional derivatives |
| `guides/01_tcav_slides.md` | Companion deck (15 slides) |
| `guides/02_influence_functions_guide.md` | TracIn, TracInCPFast, SimilarityInfluence |
| `guides/02_influence_functions_slides.md` | Companion deck (16 slides) |

### Notebooks
| File | Topic | Time |
|------|-------|------|
| `notebooks/01_tcav_texture_shape.ipynb` | TCAV on ResNet-18: texture vs. shape bias | 15 min |
| `notebooks/02_tracin_influential_examples.ipynb` | TracIn proponents, opponents, self-influence | 15 min |
| `notebooks/03_similarity_influence.ipynb` | SimilarityInfluence: representation-space kNN | 15 min |

### Exercises
| File | Topic |
|------|-------|
| `exercises/01_concept_self_check.py` | CAV training, directional derivatives, interpretation |

## Key Concepts

### TCAV
- **CAV:** Linear probe trained to separate concept activations from random in a layer
- **Directional derivative:** $S_{K,C,l}(x) = \nabla_{f_l} F_C(x) \cdot v_{K,l}$
- **TCAV score:** Fraction of class $C$ inputs with positive directional derivative
- **Significance:** Test against distribution from multiple random concept sets

### TracIn
- Traces gradient alignment across training checkpoints
- **Proponents:** High TracIn score = training on this example helped with the test prediction
- **Self-influence:** Measures $\|\nabla L(z)\|^2$ throughout training; high values = mislabeled/atypical

### SimilarityInfluence
- Finds training examples nearest to test in representation space
- Requires only the final model (no checkpoints)
- Validates that learned features encode semantic meaning

## Method Selection

| Question | Method |
|----------|--------|
| "Does model use concept X for class Y?" | TCAV |
| "What training examples drove this prediction?" | TracIn / TracInCPFast |
| "What training examples look like this to the model?" | SimilarityInfluence |
| "Are there mislabeled training examples?" | TracIn self-influence |

## Running the Exercises

```bash
python modules/module_06_concept_example_explanations/exercises/01_concept_self_check.py
```

## Next Module

**Module 07:** NLP and Transformer interpretability — token-level attributions, attention vs. IG comparison.
