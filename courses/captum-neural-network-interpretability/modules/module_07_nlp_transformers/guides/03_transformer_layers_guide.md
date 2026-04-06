# Layer-Wise Attribution in Transformers

> **Reading time:** ~7 min | **Module:** 7 — NLP & Transformers | **Prerequisites:** Module 2 Integrated Gradients


## Learning Objectives

By the end of this guide, you will be able to:
1. Apply `LayerIntegratedGradients` across all BERT encoder layers to measure layer importance
2. Interpret layer-by-layer attribution patterns (early = syntax, late = semantics)
3. Use `LayerConductance` to measure information flow at each layer
4. Analyze how different task types affect which layers carry attribution
5. Extend layer attribution to GPT-2 and other architectures


<div class="callout-key">
<strong>Key Concept Summary:</strong> This guide covers the core concepts of layer-wise attribution in transformers.
</div>

---

## 1. Why Analyze Individual Layers?

Token attribution at the embedding level tells you *which words* mattered. Layer attribution answers a complementary question: *at which processing depth* did the relevant information emerge?

This matters for:
- **Model compression:** If only the last 3 layers contribute meaningfully, earlier layers could be pruned
- **Transfer learning:** Knowing which layers encode task-relevant features guides layer freezing strategies
- **Understanding BERT:** Research shows different layers specialize in different linguistic phenomena

---

## 2. BERT Layer Roles (From Literature)

Based on probing studies (Tenney et al., 2019; Jawahar et al., 2019):

| Layers | Encoded information |
|--------|-------------------|
| 0-3 | Part-of-speech, surface features |
| 4-6 | Syntax, constituency parsing |
| 7-9 | Semantic roles, long-range dependencies |
| 10-12 | Task-specific semantics, classification signal |

For sentiment classification, layers 10-12 carry most attribution. For syntax-heavy tasks (parsing, NER), middle layers carry more.

---

## 3. LayerIntegratedGradients Across All Layers

Apply IG at each encoder layer independently:
<div class="callout-warning">
<strong>Warning:</strong> Apply IG at each encoder layer independently:
</div>


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
from captum.attr import LayerIntegratedGradients

def compute_layer_attributions(
    model, tokenizer, text: str, target_class: int, n_steps: int = 50
) -> dict:
    """Compute LayerIG attributions for each BERT encoder layer."""
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    input_ids      = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_type_ids = inputs.get("token_type_ids")

    baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)

    def forward_func(input_ids, attention_mask=None, token_type_ids=None):
        return model(input_ids, attention_mask=attention_mask,
                     token_type_ids=token_type_ids).logits

    layer_attributions = {}

    # Embedding layer
    lig_emb = LayerIntegratedGradients(forward_func, model.bert.embeddings)
    attrs, delta = lig_emb.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask, token_type_ids),
        target=target_class,
        n_steps=n_steps,
        return_convergence_delta=True,
    )
    layer_attributions['embeddings'] = {
        'attrs': attrs.sum(dim=-1).squeeze(0).detach(),
        'delta': delta.item(),
    }

    # Each encoder layer
    for layer_idx, encoder_layer in enumerate(model.bert.encoder.layer):
        lig = LayerIntegratedGradients(forward_func, encoder_layer)
        attrs, delta = lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask, token_type_ids),
            target=target_class,
            n_steps=n_steps,
            return_convergence_delta=True,
        )
        # Sum over hidden dim AND over tokens → scalar per layer
        layer_importance = attrs.sum(dim=-1).squeeze(0).abs().sum().item()
        layer_attributions[f'layer_{layer_idx}'] = {
            'attrs': attrs.sum(dim=-1).squeeze(0).detach(),  # (seq_len,)
            'importance': layer_importance,
            'delta': delta.item(),
        }

    return layer_attributions
```

</div>
</div>

---

## 4. LayerConductance: A Cleaner Approach

`LayerConductance` measures the contribution of each layer as a whole (summed across all units in the layer):

$$C_l = \int_0^1 \frac{\partial F(x'+ \alpha(x-x'))}{\partial h_l(\alpha)} \cdot \frac{dh_l(\alpha)}{d\alpha} d\alpha$$

This is the chain-rule application of IG at each layer — it measures how much the layer's activation contributes to the final output difference.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
from captum.attr import LayerConductance

def layer_conductance_all_layers(model, tokenizer, text, target_class):
    """Compute LayerConductance for each BERT encoder layer."""
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    input_ids      = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    baseline_ids   = torch.full_like(input_ids, tokenizer.pad_token_id)

    def forward_func(input_ids, attention_mask=None):
        return model(input_ids, attention_mask=attention_mask).logits

    layer_scores = []
    for layer_idx, encoder_layer in enumerate(model.bert.encoder.layer):
        lc = LayerConductance(forward_func, encoder_layer)
        conductance = lc.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            target=target_class,
            n_steps=20,
        )
        # Total conductance magnitude for this layer
        score = conductance.abs().sum().item()
        layer_scores.append(score)

    return layer_scores
```

</div>
</div>

---

## 5. Visualizing Layer Importance

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_layer_importance(layer_scores: list, title: str):
    """Plot bar chart of layer importance scores."""
    n_layers = len(layer_scores)
    layers = [f"L{i}" for i in range(n_layers)]

    fig, ax = plt.subplots(figsize=(10, 4))

    # Color by importance (gradient from light to dark blue)
    max_score = max(layer_scores)
    colors = [plt.cm.Blues(0.3 + 0.7 * s / max_score) for s in layer_scores]

    bars = ax.bar(layers, layer_scores, color=colors, edgecolor='white')

    # Annotate top layers
    top3_idx = np.argsort(layer_scores)[-3:]
    for idx in top3_idx:
        ax.bar(layers[idx], layer_scores[idx], color='#d73027', edgecolor='white')
        ax.text(idx, layer_scores[idx] * 1.02, f"#{idx}", ha='center', fontsize=8)

    ax.set_xlabel("BERT Encoder Layer")
    ax.set_ylabel("Attribution Magnitude (LayerConductance)")
    ax.set_title(title, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig
```

</div>
</div>

---

## 6. Cross-Task Layer Importance Patterns

Different tasks show different layer importance patterns:

### Sentiment Analysis (SST-2)
```
Layer importance profile:
L0  L1  L2  L3  L4  L5  L6  L7  L8  L9 L10 L11
 ░   ░   ░   ░   ░   ░   ░   ▒   ▒   ▓  ▓▓  ▓▓▓

→ Last 3 layers dominate: semantic understanding
```

### Named Entity Recognition (NER)
```
Layer importance profile:
L0  L1  L2  L3  L4  L5  L6  L7  L8  L9 L10 L11
 ░   ▒   ▓   ▓▓  ▓   ▒   ░   ░   ░   ░   ░   ░

→ Middle layers dominate: syntactic/contextual features for NER
```

### Question Answering (SQuAD)
```
Layer importance profile:
L0  L1  L2  L3  L4  L5  L6  L7  L8  L9 L10 L11
 ░   ░   ░   ░   ▒   ▒   ▓   ▓▓  ▓▓  ▓  ░   ░

→ Layers 5-8: cross-sentence attention and semantic matching
```

---

## 7. Layer Attribution Heatmap: Tokens × Layers

A comprehensive visualization shows how token importance varies across layers:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
def plot_token_layer_heatmap(
    layer_token_attrs: dict,
    tokens: list,
    title: str = "Token Attribution by Layer"
):
    """Heatmap: rows=layers, columns=tokens, values=attribution."""
    layer_names = sorted([k for k in layer_token_attrs.keys() if k.startswith("layer")])
    n_layers = len(layer_names)
    n_tokens = len(tokens)

    matrix = np.zeros((n_layers, n_tokens))
    for i, layer_name in enumerate(layer_names):
        attrs = layer_token_attrs[layer_name]['attrs'].numpy()
        matrix[i, :len(attrs)] = attrs[:n_tokens]

    fig, ax = plt.subplots(figsize=(max(12, n_tokens * 0.6), n_layers * 0.5 + 2))
    vmax = np.abs(matrix).max()
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')

    ax.set_xticks(range(n_tokens))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"Layer {i}" for i in range(n_layers)], fontsize=9)
    ax.set_xlabel("Token")
    ax.set_ylabel("BERT Layer")
    ax.set_title(title, fontweight='bold')
    plt.colorbar(im, ax=ax, label="Attribution (red=positive, blue=negative)")
    plt.tight_layout()
    return fig
```

</div>
</div>

---

## 8. GPT-2 Layer Attribution

GPT-2 uses causal (left-to-right) attention, unlike BERT's bidirectional attention. This changes the layer attribution pattern:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

gpt2_model     = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model.eval()

# For GPT-2, attribute to the next-token logit
def forward_gpt2(input_ids, attention_mask=None):
    outputs = gpt2_model(input_ids, attention_mask=attention_mask)
    # Use last-token logit for attribution
    return outputs.logits[:, -1, :]

# LayerIG on GPT-2's transformer blocks
for block_idx, block in enumerate(gpt2_model.transformer.h):
    lig = LayerIntegratedGradients(forward_gpt2, block)
    # ... same as BERT
```

</div>
</div>

GPT-2's layer attribution pattern differs: since attention is causal, early layers process local context while later layers integrate longer-range dependencies.

---

## 9. Probing Layer Representations

Layer attribution via IG tells you which layers are computationally involved. A complementary approach is **probing**: train linear classifiers to predict linguistic labels from each layer's representation.

This is related but different:
- **Probing:** What information is *encoded* in each layer? (representation analysis)
- **Layer IG:** What information is *used* for the prediction? (computation analysis)

A layer can encode rich linguistic information (high probing accuracy) but not use it for a specific task (low layer attribution).

---

## 10. Practical Tips for Layer Attribution

### Memory Efficiency
Computing LayerIG for all 12 BERT layers plus the embedding layer requires 13 separate backward passes. For production analysis:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
# Process in batches and free gradients
for layer_idx, encoder_layer in enumerate(model.bert.encoder.layer):
    with torch.no_grad():
        # Pre-compute for efficiency
        pass
    lig = LayerIntegratedGradients(forward_func, encoder_layer)
    attrs = lig.attribute(...)
    # Store summary statistics only, discard full tensors
    layer_scores.append(attrs.abs().sum().item())
    del attrs  # free memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
```

</div>
</div>

### Normalization
When comparing layer importance across texts of different length, normalize by sequence length or use the mean attribution per token rather than the sum.

---


---

## Practice Questions

<div class="callout-info">
<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Why Analyze Individual Layers?" and why it matters in practice.

2. Given a real-world scenario involving layer-wise attribution in transformers, what would be your first three steps to apply the techniques from this guide?
</div>

## Summary

| Method | What it measures | Output |
|--------|-----------------|--------|
| `LayerIG` at layer $l$ | Attribution through layer $l$'s activations | Per-token scores at layer $l$ |
| `LayerConductance` | Layer's total contribution to output change | Scalar per layer |
| Stacking across layers | Layer importance profile | Vector of length n_layers |
| Heatmap | Full picture | Matrix (layers × tokens) |

**BERT layer patterns by task:**
- Classification: last 3 layers dominate
- NER/Syntax: middle layers (4-6) dominate
- QA: layers 5-9 dominate

---

## Further Reading

- Tenney, I., et al. (2019). BERT rediscovers the classical NLP pipeline. *ACL*.
- Jawahar, G., et al. (2019). What does BERT learn about the structure of language? *ACL*.
- Michel, P., et al. (2019). Are sixteen heads really better than one? *NeurIPS*.
- Voita, E., et al. (2019). Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned. *ACL*.

---

## Cross-References

<a class="link-card" href="../notebooks/01_bert_sentiment_ig.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
