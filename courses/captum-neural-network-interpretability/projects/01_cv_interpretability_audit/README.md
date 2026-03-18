# Project 01: Computer Vision Interpretability Audit

## Overview

Conduct a comprehensive interpretability audit of a pretrained image classification model. Apply multiple Captum attribution methods to systematically characterize what the model has learned, detect potential shortcut correlations, and produce a structured audit report suitable for a model card.

This project integrates every technique from Modules 01-08 into a single end-to-end workflow.

---

## Learning Outcomes

By completing this project you will:
- Build a systematic attribution analysis pipeline from scratch
- Apply and compare IG, GradCAM, GradientSHAP, and Saliency on the same model
- Detect spurious correlations using mean attribution maps per class
- Produce a model audit report with quantitative findings
- Present attribution evidence to a non-technical audience

---

## Task Specification

### Dataset and Model

**Option A (recommended):** ResNet-50 pretrained on ImageNet-1K, evaluated on a subset of ImageNet validation images (download via Hugging Face datasets or torchvision).

**Option B:** Fine-tune a ResNet-18 on a domain-specific dataset of your choice (e.g., chest X-rays, satellite imagery, food-101) and audit the fine-tuned model.

### Required Deliverables

#### 1. Method Comparison Notebook

File: `method_comparison.ipynb`

For 5 representative images per class (minimum 5 classes, 25 total images):
- Compute Saliency, IG, and GradCAM attributions
- Visualize with three-panel plots (input / heatmap / overlay)
- Compute Spearman rank correlation between all method pairs
- Identify and annotate 3 examples where methods strongly disagree

#### 2. Per-Class Attribution Analysis

File: `per_class_analysis.ipynb`

- Compute mean attribution map for each of 10 classes
- Measure border attribution fraction per class (flag if >20%)
- Measure attribution mass inside vs. outside a rough bounding box (or segmentation mask if available)
- Produce a summary table: class name, n_examples, mean_border_fraction, diagnosis (PASS/WARN)

#### 3. Spurious Correlation Investigation

File: `spurious_correlation_investigation.ipynb`

Choose one class that the per-class analysis flagged (or shows visually suspicious attribution). Investigate:
- What background/contextual features does the model attribute to?
- Run a counterfactual test: replace the background with a uniform color or random noise. Does accuracy drop?
- Propose a data augmentation strategy to reduce the spurious correlation

#### 4. Audit Report

File: `audit_report.json`

Using the `ModelAuditRunner` from `templates/model_audit_template.py`, generate a structured JSON report covering:
- Per-class attribution statistics
- Method agreement (Spearman r matrix)
- IG convergence delta distribution (fraction with |δ| < 0.05)
- Overall PASS/FAIL verdict
- Findings and recommendations sections

#### 5. Executive Summary

File: `executive_summary.md`

A 1-page non-technical summary explaining:
- What the model appears to rely on for its predictions
- Any detected shortcuts or potential failure modes
- Confidence in the model's deployment readiness
- Recommended mitigations for any issues found

---

## Technical Requirements

### Attribution Configuration

```python
from captum.attr import IntegratedGradients, LayerGradCam, Saliency, GradientShap

model.eval()
ig = IntegratedGradients(lambda x: model(x))

# Compliance-grade IG
attrs, delta = ig.attribute(
    inputs, torch.zeros_like(inputs),
    target=pred_class, n_steps=100,
    return_convergence_delta=True,
)
assert abs(delta.item()) < 0.05, "Completeness check failed — increase n_steps"
```

### Evaluation Metrics

| Metric | Definition | Threshold |
|--------|-----------|-----------|
| Border attribution fraction | % of total |attr| in outermost 10% of image | WARN if >20% |
| Method agreement (Spearman r) | IG vs. Saliency, IG vs. GradCAM | Report all pairs |
| Convergence delta | \|sum(attrs) - (f(x) - f(baseline))\| | < 0.05 for compliance |
| Background attribution ratio | % of |attr| outside known object region | WARN if >50% |

---

## Suggested Architecture

```
01_cv_interpretability_audit/
├── method_comparison.ipynb
├── per_class_analysis.ipynb
├── spurious_correlation_investigation.ipynb
├── audit_report.json
├── executive_summary.md
├── figures/
│   ├── method_comparison_gallery.png
│   ├── per_class_mean_attribution.png
│   ├── method_agreement_matrix.png
│   └── spurious_correlation_counterfactual.png
└── data/
    └── sample_images/   (or a script to download them)
```

---

## Evaluation Criteria

| Component | What strong work demonstrates |
|-----------|-------------------------------|
| Method comparison | Correct API usage, visual clarity, quantitative correlation analysis |
| Per-class analysis | Systematic coverage, detection of at least one suspicious class |
| Spurious correlation | Counterfactual experiment with clear result, reasoned interpretation |
| Audit report | Valid JSON, all required fields, honest PASS/FAIL verdict |
| Executive summary | Clear writing, no jargon, actionable recommendations |

---

## Getting Started

```python
# Quick start with ResNet-18 on CIFAR-10
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import CIFAR10
import torchvision.transforms as T

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
transform = T.Compose([T.Resize(224), T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
dataset = CIFAR10(root='/tmp/cifar10', train=False, download=True, transform=transform)

# Use templates/image_attribution_template.py
from templates.image_attribution_template import ImageAttributionPipeline
pipeline = ImageAttributionPipeline(model, gradcam_layer=model.layer4[-1])

# Use templates/model_audit_template.py
from templates.model_audit_template import ModelAuditRunner
from torch.utils.data import DataLoader
runner = ModelAuditRunner(model, DataLoader(dataset, batch_size=4), model_id='resnet18_cifar10')
report = runner.run_full_audit()
runner.save_report('audit_report.json')
```

---

## Extensions

- Apply TCAV (Module 06) to test for texture vs. shape bias
- Use TracIn to find the most influential training images for a selected test case
- Extend the audit to a second model and compare the two audit reports
- Deploy the model as a FastAPI service and add attribution endpoints
