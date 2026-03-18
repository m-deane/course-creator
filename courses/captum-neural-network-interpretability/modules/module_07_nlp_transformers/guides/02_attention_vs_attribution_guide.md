# Attention Weights vs. Integrated Gradients: Why They Disagree

## Learning Objectives

By the end of this guide, you will be able to:
1. Explain what attention weights measure and why they are not attribution scores
2. Identify the formal properties that attention weights fail to satisfy
3. Empirically compare attention vs. IG attributions on the same BERT model
4. Apply rollout and gradient-weighted attention as improved attention-based explanations
5. Make principled decisions about when (not) to use attention as explanation

---

## 1. The "Attention is Explanation" Assumption

When Transformers became dominant in NLP, attention weights were widely used as explanations: "the model attended to word X, therefore word X was important for the prediction."

This interpretation is intuitive: attention weights show where the model "looks." But it is **not generally correct**.

Jain & Wallace (2019) showed that attention weights often disagree with input gradient scores, and that in many cases, models perform equally well when attention is randomized or permuted — undermining the claim that attention explains model decisions.

---

## 2. What Attention Weights Actually Measure

In a self-attention layer, for token $i$ attending to token $j$:

$$\alpha_{ij} = \text{softmax}\left(\frac{Q_i K_j^\top}{\sqrt{d_k}}\right)$$

$$\text{output}_i = \sum_j \alpha_{ij} V_j$$

Attention weight $\alpha_{ij}$ measures **how much of value vector $V_j$ is mixed into the representation of token $i$**. It is a routing weight in the information flow, not an attribution to the input.

### What attention does NOT measure:
1. Whether token $j$'s presence changed the output (vs. its absence)
2. The marginal contribution of token $j$ to the final prediction
3. Whether a different value of token $j$ would change the prediction

---

## 3. The Formal Argument

Attribution methods must satisfy the efficiency axiom:

$$\sum_i \phi_i = f(x) - f(x')$$

Attention weights satisfy no analogous conservation law. They are normalized to sum to 1 by softmax, but this normalization is unrelated to how much each token contributes to the final prediction.

**Counterexample:** Suppose attention weight $\alpha_{ij} = 0.9$ (token $i$ attends heavily to token $j$), but value vector $V_j$ is nearly constant regardless of the input token — then changing token $j$ has little effect on the output, despite high attention weight.

---

## 4. Multi-Head Attention Complicates Aggregation

BERT-base has 12 heads per layer and 12 layers = **144 attention matrices**.

How do you aggregate to get a single "attention score" per token?

**Common approaches (all arbitrary):**
1. Average across heads: $\bar{\alpha}_j = \frac{1}{H}\sum_h \alpha_{ij}^{(h)}$
2. Maximum across heads
3. Attention rollout (Abnar & Zuidema, 2020): propagate attention through layers
4. Gradient-weighted attention: $\text{GradAttn}_{ij} = \alpha_{ij} \cdot \nabla_{\alpha_{ij}} F$

None of these have the theoretical grounding of Integrated Gradients.

---

## 5. Attention Rollout

Attention rollout (Abnar & Zuidema, 2020) improves over raw attention by propagating attention weights through all layers:

$$\tilde{A}^l = A^l \tilde{A}^{l-1} \quad \text{with} \quad \tilde{A}^0 = I$$

where $A^l = \frac{1}{H}\sum_h \alpha^{l,h}$ is the mean-head attention at layer $l$.

This accounts for the fact that token representations at layer $l$ are mixtures of inputs from all previous layers.

```python
def attention_rollout(attentions_list: list[torch.Tensor]) -> torch.Tensor:
    """
    Args:
        attentions_list: list of (1, n_heads, seq_len, seq_len) attention matrices,
                         one per layer
    Returns:
        rollout: (seq_len, seq_len) — how much each token's final representation
                 came from each input token
    """
    result = torch.eye(attentions_list[0].shape[-1])
    for attn in attentions_list:
        # Average over heads
        attn_mean = attn.mean(dim=1).squeeze(0)  # (seq_len, seq_len)
        # Add residual (identity) to account for skip connections
        attn_with_residual = 0.5 * attn_mean + 0.5 * torch.eye(attn_mean.shape[0])
        # Normalize
        attn_normalized = attn_with_residual / attn_with_residual.sum(dim=-1, keepdim=True)
        result = attn_normalized @ result
    return result
```

---

## 6. Gradient-Weighted Attention

Serrano & Smith (2019) proposed multiplying attention weights by the gradient of the output with respect to attention:

$$\text{GradAttn}_{ij} = \alpha_{ij} \cdot \left|\frac{\partial F}{\partial \alpha_{ij}}\right|$$

This is analogous to Saliency × Input for regular inputs and has better empirical correlation with gold-standard rationale annotations than raw attention.

```python
from captum.attr import LayerGradientXActivation

# Gradient × Attention attribution
model.eval()
for attn_layer in model.bert.encoder.layer:
    lga = LayerGradientXActivation(
        forward_func,
        attn_layer.attention.self,  # attention module
    )
    grad_attn = lga.attribute(
        inputs=input_ids,
        additional_forward_args=(attention_mask, token_type_ids),
        target=pred_class,
    )
```

---

## 7. Empirical Comparison: When They Agree vs. Disagree

On most well-functioning models, attention and IG broadly agree on the top tokens. However, they disagree in important cases:

### Case 1: Negation
```
Text: "The movie was NOT boring at all."
IG:   "NOT" has high negative attribution (flips sentiment)
Attn: "boring" has high attention (attended to most)
```

Attention focuses on the sentiment word; IG correctly identifies the negation modifier.

### Case 2: Adversarial Inputs
```
Text: "movie great terrible film"  (adversarial)
IG:   attributes heavily to "terrible" (correct)
Attn: distributes attention broadly across all words
```

### Case 3: Long Documents
In long documents, attention often diffuses — no single token gets high weight. IG maintains discriminative signal via the path integral.

---

## 8. Formal Comparison: Which Properties Do They Satisfy?

| Property | IG/SHAP | Gradient×Input | Raw Attention | Attn Rollout | Grad-Attn |
|----------|---------|----------------|---------------|--------------|-----------|
| Efficiency axiom | ✓ | ✗ | ✗ | ✗ | ✗ |
| Sensitivity | ✓ | ✓ | ✗ | Partial | Partial |
| Completeness | ✓ | ✗ | ✗ | ✗ | ✗ |
| Implementation invariance | ✓ | ✗ | ✗ | ✗ | ✗ |
| Handles negation | Good | Good | Poor | Moderate | Moderate |

**Recommendation:** Use attention visualization for debugging and exploratory analysis, but use IG or SHAP for explanations that need theoretical guarantees.

---

## 9. The Case FOR Using Attention

Despite the formal limitations, attention weights provide value:

1. **Cheap:** No additional forward/backward passes required
2. **Available:** Attention matrices are output by most Transformer APIs
3. **Interactive:** Can visualize attention patterns for all head/layer combinations
4. **Debugging:** Attention heads often specialize (coreference, syntax, position)

For **exploratory analysis** — understanding how a model processes text, not explaining predictions to stakeholders — attention visualization is a valuable tool.

**Rule of thumb:**
- Stakeholder explanation → Use IG/SHAP
- Model debugging → Attention + IG are complementary
- Research: Always compare attention to gradient-based methods

---

## 10. Attention Head Specialization

While individual heads don't explain predictions, analyzing which heads specialize in different linguistic patterns is informative for understanding BERT's representations:

```python
# Extract all attention matrices
outputs = model(input_ids, attention_mask=attention_mask,
                output_attentions=True)
all_attentions = outputs.attentions  # list of (1, 12, seq_len, seq_len)

# Visualize head 0 in layer 0
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 8))
attn_matrix = all_attentions[0][0, 0, :, :].detach().numpy()  # layer 0, head 0
im = ax.imshow(attn_matrix, cmap='Blues')
ax.set_xticks(range(len(tokens)))
ax.set_yticks(range(len(tokens)))
ax.set_xticklabels(tokens, rotation=90)
ax.set_yticklabels(tokens)
plt.colorbar(im)
plt.title("BERT Layer 0, Head 0 Attention Pattern")
```

Some heads specialize in: next-token attention, previous-token attention, [CLS]-to-all, coreference, syntactic dependencies.

---

## Summary

| Method | Theoretical basis | Practical speed | Use case |
|--------|-------------------|-----------------|----------|
| Raw attention | None | Very fast | Exploration only |
| Attention rollout | Propagation | Fast | Better exploration |
| Gradient × Attention | Gradient signal | Moderate | Improved attention |
| Integrated Gradients | Axiomatic | Moderate | Stakeholder explanation |
| SHAP | Game theory | Slow | High-stakes decisions |

**Core message:** Attention is not attribution. Use IG or SHAP when explanations need to satisfy formal properties.

---

## Further Reading

- Jain, S., & Wallace, B. C. (2019). Attention is not explanation. *NAACL*.
- Wiegreffe, S., & Pinter, Y. (2019). Attention is not not explanation. *EMNLP*.
- Abnar, S., & Zuidema, W. (2020). Quantifying attention flow in transformers. *ACL*.
- Bastings, J., & Filippova, K. (2020). The elephant in the interpretability room. *BlackboxNLP*.
