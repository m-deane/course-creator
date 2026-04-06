# Token-Level Attribution for NLP Models

> **Reading time:** ~8 min | **Module:** 7 — NLP & Transformers | **Prerequisites:** Module 2 Integrated Gradients


## Learning Objectives

By the end of this guide, you will be able to:
1. Explain why NLP models require special handling for attribution (discrete inputs, embedding layers)
2. Apply Integrated Gradients to token embeddings and aggregate to token-level scores
3. Define and use reference tokens (pad, mask, uniform) as attribution baselines
4. Visualize token attributions with colored text (green = positive, red = negative)
5. Implement token attribution for any HuggingFace sequence classification model


<div class="callout-key">

<strong>Key Concept Summary:</strong> This guide covers the core concepts of token-level attribution for nlp models.

</div>

---

## 1. The NLP Attribution Challenge

Neural network attribution methods are defined on continuous input spaces. Text inputs are discrete (tokens from a vocabulary of 30K+ tokens), which creates two challenges:
<div class="callout-key">

<strong>Key Point:</strong> Neural network attribution methods are defined on continuous input spaces. Text inputs are discrete (tokens from a vocabulary of 30K+ tokens), which creates two challenges:

</div>


### Challenge 1: Discrete Inputs
Gradient methods require the input to be differentiable. Text token IDs are integers — they have no meaningful gradient.

**Solution:** Differentiate with respect to the continuous **token embeddings**, not the token IDs.

$$\phi_i = \text{AttributionOf}(e_i) \quad \text{where } e_i = \text{Embed}(t_i)$$

Then aggregate the embedding-space attribution to a scalar per token by summing or taking L2 norm across the embedding dimension.

### Challenge 2: What is the "Absent" State?

For images, the baseline is a black or neutral image. For tabular data, it's the feature mean. For text, what does it mean for a word to be "absent"?

Common choices:
- **Padding token** `[PAD]`: Most natural "absent" token for BERT-style models
- **Mask token** `[MASK]`: Used in masked language modeling
- **Uniform embedding**: Average embedding of all tokens in the vocabulary
- **Zero embedding**: Numerically simplest but may be outside the distribution

The choice significantly affects attributions and should be motivated by the model's pretraining.

---

## 2. Integrated Gradients for Token Embeddings

For a transformer model with embedding layer $E$, the IG attribution for token $i$ is:

$$\text{IG}_i = (e_i - e'_i) \cdot \int_0^1 \frac{\partial F(E'(\alpha))}{\partial e_i} d\alpha$$

where:
- $e_i = E(t_i)$ = embedding of token $i$
- $e'_i = E(t'_i)$ = embedding of the baseline token (e.g., `[PAD]`)
- $F(\cdot)$ = model output (logit or probability for target class)
- $E'(\alpha) = e' + \alpha(e - e')$ = interpolated embedding

The integral gives a vector of shape `(embedding_dim,)` per token. To get a scalar attribution per token, sum across the embedding dimension:

$$\phi_i = \sum_j \text{IG}_{i,j}$$

---

## 3. Captum LayerIntegratedGradients

Captum's `LayerIntegratedGradients` applies IG at a specific layer — the embedding layer:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
from captum.attr import LayerIntegratedGradients

# BERT sentiment classifier
model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2")
tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")

# Forward function wrapping the model
def forward_func(input_ids, attention_mask=None, token_type_ids=None):
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )
    return output.logits

# LayerIG attributes w.r.t. the embedding layer
lig = LayerIntegratedGradients(
    forward_func=forward_func,
    layer=model.bert.embeddings,  # target the embedding layer
)
```

</div>

</div>

---

## 4. Baseline Construction


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def create_baselines(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """Create PAD-token baseline with same shape as input_ids."""
    # All positions become [PAD] token
    pad_id = tokenizer.pad_token_id
    return torch.full_like(input_ids, pad_id)


def create_mask_baselines(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """Use [MASK] for content tokens, keep [CLS] and [SEP] unchanged."""
    mask_id = tokenizer.mask_token_id
    cls_id  = tokenizer.cls_token_id
    sep_id  = tokenizer.sep_token_id

    baselines = input_ids.clone()
    # Replace all non-special tokens with [MASK]
    special_ids = {cls_id, sep_id, tokenizer.pad_token_id}
    for i in range(baselines.shape[1]):
        if baselines[0, i].item() not in special_ids:
            baselines[0, i] = mask_id
    return baselines
```

</div>

</div>

---

## 5. Computing Token Attributions End-to-End


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from captum.attr import LayerIntegratedGradients

# Load model
model_name = "textattack/bert-base-uncased-SST-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Input text
text = "The movie was surprisingly engaging with brilliant performances."
inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)

input_ids      = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
token_type_ids = inputs.get("token_type_ids", None)

# Baseline: all PAD tokens
baseline_ids = create_baselines(input_ids, tokenizer)

# Forward function
def forward_func(input_ids, attention_mask=None, token_type_ids=None):
    out = model(input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)
    return out.logits

# Compute prediction
with torch.no_grad():
    logits = forward_func(input_ids, attention_mask, token_type_ids)
    predicted_class = logits.argmax(dim=1).item()
    labels = ["NEGATIVE", "POSITIVE"]
    print(f"Prediction: {labels[predicted_class]}")

# LayerIG attribution
lig = LayerIntegratedGradients(forward_func, model.bert.embeddings)

attributions, delta = lig.attribute(
    inputs=input_ids,
    baselines=baseline_ids,
    additional_forward_args=(attention_mask, token_type_ids),
    target=predicted_class,
    n_steps=50,
    return_convergence_delta=True,
)

# attributions shape: (1, seq_len, hidden_size)

# Aggregate to token-level: sum over embedding dim
token_attrs = attributions.sum(dim=-1).squeeze(0)  # shape: (seq_len,)
```

</div>

</div>

---

## 6. Token Attribution Visualization

The standard visualization uses colored text: green for positive attributions, red for negative.
<div class="callout-warning">

<strong>Warning:</strong> The standard visualization uses colored text: green for positive attributions, red for negative.

</div>


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def visualize_token_attributions(tokens: list, attributions: np.ndarray,
                                  title: str = "Token Attributions") -> plt.Figure:
    """Visualize token attributions with colored text."""
    attrs = attributions.copy()
    # Normalize to [-1, 1] for coloring
    max_abs = np.abs(attrs).max()
    if max_abs > 0:
        attrs_norm = attrs / max_abs
    else:
        attrs_norm = attrs

    fig, ax = plt.subplots(figsize=(max(12, len(tokens) * 0.8), 2.5))
    ax.axis('off')

    x_pos = 0.0
    token_width = 1.0 / max(len(tokens), 1)

    for i, (tok, attr_norm) in enumerate(zip(tokens, attrs_norm)):
        # Color: green (positive) to white (neutral) to red (negative)
        if attr_norm > 0:
            r, g, b = 1 - attr_norm * 0.5, 1.0, 1 - attr_norm * 0.5  # green
        else:
            r, g, b = 1.0, 1 + attr_norm * 0.5, 1 + attr_norm * 0.5  # red (negative)

        color = (np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1))
        x_center = (i + 0.5) / len(tokens)

        # Draw colored rectangle
        rect = plt.Rectangle(
            (x_center - token_width*0.45, 0.2),
            token_width * 0.9, 0.6,
            facecolor=color, edgecolor='lightgray', linewidth=0.5
        )
        ax.add_patch(rect)

        # Token text
        ax.text(x_center, 0.5, tok, ha='center', va='center',
                fontsize=max(7, min(11, 80 // len(tokens))),
                color='black')

        # Attribution value below
        ax.text(x_center, 0.1, f"{attributions[i]:.3f}",
                ha='center', va='center', fontsize=6, color='gray')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    return fig
```

</div>

</div>

---

## 7. Special Tokens in Attribution

BERT-style models add special tokens: `[CLS]` (start), `[SEP]` (end/separator), `[PAD]` (padding).

**What do attributions on special tokens mean?**

- **[CLS]:** In BERT, the classification token aggregates global sequence information. High attribution on [CLS] means the model's decision depends heavily on the aggregate representation, not specific words.
- **[SEP]:** Separates segments. High attribution can indicate the model is sensitive to the boundary between question and answer (in QA tasks).
- **[PAD]:** Should have zero attribution (padding doesn't convey information). High [PAD] attribution signals a problem — usually due to poor baseline choice.

Best practice: Exclude or minimize special token attributions when communicating with non-technical stakeholders.

---

## 8. Aggregation Methods

After computing per-embedding-dimension attributions, several aggregation methods give different insights:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Method 1: Sum across embedding dim (signed, preserves direction)
token_attrs_sum = attributions.sum(dim=-1)

# Method 2: L2 norm (always positive, magnitude only)
token_attrs_l2 = attributions.norm(dim=-1)

# Method 3: Mean across embedding dim
token_attrs_mean = attributions.mean(dim=-1)
```



**Sum:** Best for understanding direction (positive = toward target class, negative = away from target class).

**L2 norm:** Best for understanding magnitude without direction (e.g., how important is this token regardless of direction?).

For visualization, sum is most common; for ranking token importance, L2 or abs sum is used.

---

## 9. Reference Tokens: Comparison

| Reference Token | When to Use | Advantages | Disadvantages |
|-----------------|-------------|------------|---------------|
| `[PAD]` | BERT, RoBERTa | Model-appropriate absent state | PAD may have non-zero embedding |
| `[MASK]` | BERT (bidirectional) | Semantically "unknown" | Changes model behavior |
| Uniform | Any model | Distribution-neutral | Out of distribution |
| Zero | Any model | Numerically clean | Likely not in training distribution |

**Recommendation:** Use `[PAD]` for BERT-family models. Verify that baseline produces near-zero output for neutral sentences.

---

## 10. Limitations of Token Attribution

### Long-Range Dependencies
Transformers handle long-range dependencies. Attribution at the token level may not reveal that two distant tokens interact to drive the prediction.

### Subword Tokenization
BERT uses WordPiece tokenization, splitting words into subwords. "Unhappy" might become ["un", "##happy"]. Attribution should be aggregated back to the word level for readability.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
def aggregate_subword_attributions(tokens, token_attrs, tokenizer):
    """Merge subword attributions back to word level."""
    word_attrs = []
    word_tokens = []
    current_word_attr = 0.0
    current_word = ""

    for tok, attr in zip(tokens, token_attrs):
        if tok.startswith("##"):
            current_word_attr += attr
            current_word += tok[2:]
        else:
            if current_word:
                word_attrs.append(current_word_attr)
                word_tokens.append(current_word)
            current_word_attr = attr
            current_word = tok

    if current_word:
        word_attrs.append(current_word_attr)
        word_tokens.append(current_word)

    return word_tokens, word_attrs
```



---


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "The NLP Attribution Challenge" and why it matters in practice.

2. Given a real-world scenario involving token-level attribution for nlp models, what would be your first three steps to apply the techniques from this guide?


## Summary

| Concept | Implementation |
|---------|---------------|
| Attribution target | Token embeddings (not token IDs) |
| Captum class | `LayerIntegratedGradients` with embedding layer |
| Aggregation | `attributions.sum(dim=-1)` or `.norm(dim=-1)` |
| Baseline | `[PAD]` for BERT-family models |
| Visualization | Colored text: green=positive, red=negative |
| Special tokens | Include in computation but optionally exclude from display |

---

## Further Reading

- Sundararajan, M., et al. (2017). Axiomatic attribution for deep networks. *ICML*.
- DeYoung, J., et al. (2020). ERASER: A benchmark to evaluate rationalized NLP models. *ACL*.
- Bastings, J., & Filippova, K. (2020). The elephant in the interpretability room: Why use attention as explanation when we have saliency methods? *BlackboxNLP workshop*.
- Captum LayerIntegratedGradients: https://captum.ai/api/layer.html#layerintegratedgradients

---

## Cross-References

<a class="link-card" href="../notebooks/01_bert_sentiment_ig.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
