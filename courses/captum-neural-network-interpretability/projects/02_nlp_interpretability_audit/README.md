# Project 02: NLP Interpretability Audit

## Overview

Conduct a comprehensive interpretability audit of a pretrained text classification model. Apply token attribution, layer analysis, and attention comparison to characterize the model's reasoning, identify linguistic patterns it relies on, and produce evidence for or against deployment.

This project applies Modules 05-08 to NLP, with a focus on LayerIntegratedGradients, LayerConductance, and attention vs. attribution comparison.

---

## Learning Outcomes

By completing this project you will:
- Apply LayerIntegratedGradients to a real classification task
- Identify which BERT encoder layers are most important for the task
- Empirically test the "attention is not explanation" claim on your data
- Detect annotation artifacts and dataset biases via attribution
- Produce a token-level attribution analysis report

---

## Task Specification

### Model and Dataset

**Option A (recommended):** `textattack/bert-base-uncased-SST-2` on SST-2 or IMDB sentiment data. Use HuggingFace `datasets` library.

**Option B:** Any HuggingFace sequence classification model of your choice. Good alternatives:
- `cross-encoder/nli-distilroberta-base` on SNLI/MultiNLI
- `distilbert-base-uncased-finetuned-sst-2-english` on your own collected sentences
- A domain-specific classifier (legal, medical, news)

### Required Deliverables

#### 1. Token Attribution Analysis Notebook

File: `token_attribution_analysis.ipynb`

For 20 representative sentences (10 positive, 10 negative for sentiment; or 10 per class):
- Compute LayerIntegratedGradients attributions with PAD baseline
- Visualize as colored token spans (green=positive, red=negative)
- Aggregate subword tokens to word level
- Identify and annotate the top-3 most influential words per sentence
- Check convergence delta (|δ| < 0.05 for all examples)

#### 2. Attention vs. Attribution Comparison

File: `attention_vs_attribution.ipynb`

For 15 carefully selected sentences including:
- Clear positive/negative examples (expected: method agreement)
- Negation cases ("not bad", "never boring", "far from perfect")
- Adversarial examples (insert unrelated strong sentiment words)

Compute:
- Last-layer mean attention from CLS token
- Attention rollout
- LayerIG token attributions

Report:
- Spearman r between |attention| and |IG| for each sentence
- Cases where methods strongly disagree (r < 0.3)
- Do negation words receive positive attribution from IG but low attention?

#### 3. Layer Importance Profile

File: `layer_importance_profile.ipynb`

- Compute LayerConductance for all 12 encoder layers on 20 examples
- Plot mean layer importance profile for positive and negative classes separately
- Identify which layers dominate (expected: 9-11 for SST-2 sentiment)
- Compare layer profile for easy (high-confidence) vs. hard (low-confidence) examples
- Optional: compare profile for simple sentences vs. complex/negation sentences

#### 4. Dataset Bias Investigation

File: `dataset_bias_investigation.ipynb`

Test whether the model has learned annotation artifacts or spurious patterns:
- **Hypothesis-only test:** For NLI models, does the model classify correctly from the hypothesis alone (without the premise)?
- **Length bias:** Do longer sentences receive higher attribution mass on late tokens?
- **Lexical shortcuts:** Pick 5 high-attribution words from your analysis. Does replacing them change the prediction even when semantics are preserved?

Report findings as a structured table: artifact, evidence, impact, recommended fix.

#### 5. Attribution Report

File: `attribution_report.json`

Generate a JSON report for 10 representative examples containing:
```json
{
  "example_id": "...",
  "text": "...",
  "prediction": {"class": "POSITIVE", "confidence": 0.97},
  "attribution": {
    "method": "layer_integrated_gradients",
    "n_steps": 100,
    "convergence_delta": 0.0003,
    "top_words": [
      {"word": "brilliant", "attribution": 0.42, "rank": 1},
      {"word": "moving", "attribution": 0.31, "rank": 2}
    ]
  }
}
```

#### 6. Executive Summary

File: `executive_summary.md`

A 1-page summary covering:
- What linguistic features the model relies on most
- Whether attention and attribution agree (with specific examples)
- Evidence of any annotation artifacts or lexical shortcuts
- Layer analysis findings (which depth matters for this task)
- Deployment recommendation with justification

---

## Technical Requirements

### Core Attribution Setup

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from captum.attr import LayerIntegratedGradients, LayerConductance

model_name = "textattack/bert-base-uncased-SST-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

def forward_func(input_ids, attention_mask=None, token_type_ids=None):
    return model(input_ids=input_ids, attention_mask=attention_mask,
                 token_type_ids=token_type_ids).logits

lig = LayerIntegratedGradients(forward_func, model.bert.embeddings)
```

### Subword Aggregation (required)

```python
def aggregate_subwords(tokens, attrs, skip_special=True):
    special = {'[CLS]', '[SEP]', '[PAD]'}
    words, word_attrs = [], []
    cur_word, cur_attr = None, 0.0
    for tok, attr in zip(tokens, attrs):
        if skip_special and tok in special: continue
        if tok.startswith('##'):
            cur_word = (cur_word or '') + tok[2:]
            cur_attr += attr
        else:
            if cur_word: words.append(cur_word); word_attrs.append(cur_attr)
            cur_word, cur_attr = tok, attr
    if cur_word: words.append(cur_word); word_attrs.append(cur_attr)
    return words, np.array(word_attrs)
```

### Convergence Check (required for report)

```python
# Every example in the report must pass this check
attrs, delta = lig.attribute(
    input_ids, baseline_ids,
    additional_forward_args=(attention_mask, token_type_ids),
    target=pred_class, n_steps=100,
    return_convergence_delta=True,
)
assert abs(delta.item()) < 0.05, f"delta={delta.item():.5f} — increase n_steps"
```

---

## Suggested Architecture

```
02_nlp_interpretability_audit/
├── token_attribution_analysis.ipynb
├── attention_vs_attribution.ipynb
├── layer_importance_profile.ipynb
├── dataset_bias_investigation.ipynb
├── attribution_report.json
├── executive_summary.md
└── figures/
    ├── token_attribution_gallery.png
    ├── attention_vs_ig_comparison.png
    ├── layer_importance_pos_neg.png
    └── token_layer_heatmap.png
```

---

## Evaluation Criteria

| Component | What strong work demonstrates |
|-----------|-------------------------------|
| Token attribution | Correct LayerIG setup, subword merging, delta verification |
| Attention vs. attribution | Quantitative Spearman r, clear negation case study |
| Layer profile | Correct LayerConductance loop, comparison across sentence types |
| Dataset bias | Creative hypothesis testing, honest reporting of negative results |
| Attribution report | Valid JSON, delta < 0.05 for all entries |
| Executive summary | Clear, evidence-based, actionable |

---

## Getting Started

```python
# Quick start using the text attribution template
from templates.text_attribution_template import TextAttributionPipeline

pipeline = TextAttributionPipeline(
    model_name="textattack/bert-base-uncased-SST-2",
    labels=["NEGATIVE", "POSITIVE"],
)

# Single attribution
result = pipeline.attribute("This film was absolutely brilliant.", n_steps=100)
pipeline.print_colored(result)

# Layer profile
layer_result = pipeline.attribute_all_layers("A masterpiece of modern cinema.")
print("Top layers:", sorted(range(12), key=lambda i: -layer_result['layer_scores'][i])[:3])
```

---

## Extensions

- Apply the full pipeline to a second language (using multilingual-BERT)
- Compare sentence-level confidence calibration with token-level attribution quality — do high-confidence predictions also have clearer token attributions?
- Use SimilarityInfluence (Module 06) to find training examples that are most similar to your test cases in representation space
- Build an interactive demo using Captum Insights (Module 08) and present it
