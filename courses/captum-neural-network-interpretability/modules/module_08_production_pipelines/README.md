# Module 08 — Production Interpretability Pipelines

## Overview

This module bridges the gap between running Captum attribution methods in a notebook and deploying them in a production system. Key topics:
- Captum Insights: interactive web-based attribution visualization for development
- FastAPI interpretability service: RESTful API for serving attributions programmatically
- Model debugging: using attributions to detect spurious correlations and data artifacts
- Attribution method selection: a comprehensive decision framework for any model and use case

## Prerequisites

- Modules 01-07: all attribution methods (IG, GradCAM, SHAP, TCAV, LayerIG)
- Basic familiarity with REST APIs (HTTP verbs, JSON, status codes)
- Optional: Docker, FastAPI, uvicorn for the API guide

## Module Contents

### Guides
| File | Topic |
|------|-------|
| `guides/01_captum_insights_guide.md` | AttributionVisualizer, ImageFeature, Flask server, Jupyter widget |
| `guides/01_captum_insights_slides.md` | Companion deck (16 slides) |
| `guides/02_interpretability_api_guide.md` | FastAPI + Captum service: model registry, caching, batch endpoints |
| `guides/02_interpretability_api_slides.md` | Companion deck (15 slides) |
| `guides/03_model_debugging_guide.md` | Spurious correlations, border artifacts, regulatory reporting, drift monitoring |
| `guides/03_model_debugging_slides.md` | Companion deck (13 slides) |
| `guides/04_decision_flowchart_guide.md` | Which method for which architecture and use case — comprehensive framework |
| `guides/04_decision_flowchart_slides.md` | Companion deck (15 slides) |

### Notebooks
| File | Topic | Time |
|------|-------|------|
| `notebooks/01_captum_insights_demo.ipynb` | Insights setup, synthetic dataset, manual IG verification | 15 min |
| `notebooks/02_batch_attribution.ipynb` | Batch processing, caching, latency benchmark, audit reports | 15 min |

### Exercises
| File | Topic |
|------|-------|
| `exercises/01_production_self_check.py` | Insights config, cache correctness, delta threshold, method latency, baseline sensitivity |

## Key Concepts

### Captum Insights
- `AttributionVisualizer(models, score_func, classes, features, dataset, num_examples)` — central object
- `ImageFeature(name, baseline_transforms, input_transforms)` — preprocessor + renderer
- `Batch(inputs, labels)` — single-example data container for Insights
- `visualizer.serve(port=5001)` — Flask dev server; `CaptumInsightsWidget` for Jupyter

### Interpretability API
- **Model registry:** thread-safe dict of `RegisteredModel` objects loaded at startup
- **Caching:** MD5 hash of (input tensor, method, target) as cache key; LRU eviction
- **Batch endpoint:** `asyncio.gather` with semaphore to limit concurrent attributions
- **Pydantic validation:** Literal types for method names, validator for n_steps range

### Model Debugging
- **Spurious correlation check:** measure `background_attribution_fraction` with object mask
- **Border artifact check:** measure fraction of attribution in outermost 10% of image
- **Dual attribution:** run IG toward both true class and predicted class for misclassifications
- **Regulatory reports:** `AttributionReport` dataclass with `convergence_delta` completeness check
- **Drift monitoring:** Wasserstein distance between reference and current attribution distributions

### Method Selection Framework
```
Architecture → {CNN, ViT, Transformer, Tabular}
    × Purpose → {Debug, Explain, Compliance, Real-time, Research}
    × Latency → {<10ms, <200ms, <500ms, >500ms}
```

| Scenario | Method |
|----------|--------|
| CNN + real-time (<10ms) | Saliency or GradCAM |
| CNN + compliance | IG, n_steps=200, check delta<0.05 |
| Text + token attribution | LayerIntegratedGradients on embedding layer |
| Tabular + Shapley guarantees | DeepLIFTSHAP with training set background |
| Tree model (non-differentiable) | KernelSHAP only |
| Development debugging | Saliency scan → IG on suspicious examples |

## Running the Exercises

```bash
python modules/module_08_production_pipelines/exercises/01_production_self_check.py
```

## Course Completion

This is the final module of the Captum Neural Network Interpretability course. After completing Modules 01-08, you have covered:

| Module | Methods |
|--------|---------|
| 01: Foundations | Saliency, Gradient×Input, Guided Backprop |
| 02: Integrated Gradients | IG, convergence delta, baselines |
| 03: Layer & Neuron | LayerIG, LayerConductance, NeuronIG |
| 04: GradCAM & Perturbation | GradCAM, Occlusion, feature ablation |
| 05: SHAP Methods | KernelSHAP, GradientSHAP, DeepLIFT |
| 06: Concept & Example | TCAV, TracIn, SimilarityInfluence |
| 07: NLP & Transformers | LayerIG for text, attention vs. attribution |
| 08: Production Pipelines | Insights, API, debugging, decision framework |

See the `projects/` directory for portfolio projects applying these tools to real-world CV and NLP interpretability audits.
