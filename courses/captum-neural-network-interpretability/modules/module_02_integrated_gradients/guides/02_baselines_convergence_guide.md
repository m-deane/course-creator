# Baselines, Convergence, and NoiseTunnel

> **Reading time:** ~8 min | **Module:** 2 — Integrated Gradients | **Prerequisites:** Module 1 Gradient Methods


## In Brief

The baseline is the most consequential practical decision when applying Integrated Gradients. It defines the reference point and shapes all attributions. This guide covers empirical baseline selection, convergence diagnostics, and variance reduction with NoiseTunnel.

## Key Insight

The baseline is not a hyperparameter to tune for best-looking results — it defines the question you are asking. Choosing the right baseline means choosing the right null hypothesis: "what does the model predict when input feature $i$ carries *this specific kind* of no information?"


<div class="callout-key">

<strong>Key Concept Summary:</strong> The baseline is the most consequential practical decision when applying Integrated Gradients.

</div>

---

## 1. Baseline Interpretation

The baseline $x'$ represents "the absence of the feature being attributed." All attributions are measured relative to this reference.

The attribution $\text{IG}_i(x)$ answers:
> "Compared to the prediction at baseline $x'$, how much did feature $i$ contribute to moving the prediction to $f(x)$?"

Different baselines give different but equally valid answers to different questions.

---

## 2. Standard Baseline Choices

### Zero Baseline (Black Image)
<div class="callout-warning">

<strong>Warning:</strong> baseline = torch.zeros_like(input_tensor)

</div>


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
baseline = torch.zeros_like(input_tensor)
```

</div>

</div>

**Interpretation:** "Compared to a completely black image."

**When to use:** Standard for image classification when you want pixel-level attribution. Simple, fast, reproducible.

**Limitations:** A black image may not be in the distribution of natural images. For models that have never seen all-black images during training, the prediction at the baseline is undefined.

**Good for:** Pixel-by-pixel importance (each pixel's contribution above the black background).

### Blurred Image Baseline


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
from torchvision.transforms.functional import gaussian_blur

def create_blurred_baseline(input_tensor, kernel_size=41, sigma=15.0):
    """Create a heavily blurred version of the input as baseline."""
    return gaussian_blur(input_tensor, kernel_size=[kernel_size, kernel_size],
                         sigma=[sigma, sigma])
```

</div>

</div>

**Interpretation:** "Compared to the same image without local detail (only low-frequency global structure)."

**When to use:** When you want to identify which *local* patterns (edges, textures, shapes) drive the prediction, relative to the overall gist of the image.

**Limitations:** Baseline is input-dependent — cannot be precomputed and stored.

**Good for:** Understanding which fine-grained local features distinguish the prediction.

### Random Noise Baseline (Averaged)


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def random_noise_baseline(input_tensor, n_samples=50):
    """Average of multiple random noise baselines."""
    baselines = []
    for _ in range(n_samples):
        noise = torch.randn_like(input_tensor) * 0.1
        baselines.append(noise)
    return torch.stack(baselines).mean(0)
```

</div>

</div>

**Interpretation:** Averaged over many "completely random" references.

**When to use:** When you want attributions that are less dependent on any specific baseline choice.

**Limitations:** Computationally expensive; averaging over baselines is itself a form of Expected Gradients (SHAP).

### Training Set Mean Baseline (Tabular)


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
baseline_mean = torch.tensor(X_train.mean(axis=0, keepdims=True))
```

</div>

</div>

**Interpretation:** "Compared to a typical input from the training distribution."

**When to use:** Tabular data where a "zero" input is often out of distribution. The mean represents a realistic reference.

**Good for:** Tabular attribution where zero values may not be meaningful.

### Domain-Specific Baselines

**Text (token-level attribution):**

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Use [MASK] token embedding as baseline

# This represents "no token" in transformer models
baseline_ids = tokenizer(
    "[MASK] " * seq_len,
    return_tensors="pt"
)["input_ids"]
```

</div>

</div>

**Tabular — adversarial baseline:**

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Use the mean of the *opposite* class as baseline

# For a fraud detection model: use mean legitimate transaction as baseline
baseline_fraud = X_train[y_train == 0].mean(axis=0)  # legitimate average
```

</div>

</div>

---

## 3. Empirical Baseline Comparison

The right way to choose a baseline: run attribution with multiple baselines and check:
<div class="callout-key">

<strong>Key Point:</strong> The right way to choose a baseline: run attribution with multiple baselines and check:

</div>


1. **Qualitative:** Do the high-attribution regions make semantic sense?
2. **Quantitative:** How much does the attribution ranking change across baselines?
3. **Completeness check:** Does the convergence delta remain small?


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
from captum.attr import IntegratedGradients
from torchvision.transforms.functional import gaussian_blur

def compare_baselines(model, input_tensor, target_class, n_steps=50):
    """
    Compute IG attributions with multiple baseline choices.
    Returns dict {baseline_name: attribution_tensor}
    """
    ig = IntegratedGradients(model)

    # Define baselines
    baselines = {
        'zero (black)': torch.zeros_like(input_tensor),
        'blurred':       gaussian_blur(input_tensor, [41, 41], [15.0, 15.0]),
        'random (avg)':  torch.stack([
            torch.randn_like(input_tensor) * 0.1 for _ in range(20)
        ]).mean(0),
    }

    results = {}
    for name, bl in baselines.items():
        inp = input_tensor.clone().requires_grad_(True)
        attr, delta = ig.attribute(
            inp, baselines=bl, target=target_class,
            n_steps=n_steps, return_convergence_delta=True
        )
        results[name] = {
            'attr': attr.detach(),
            'delta': delta.item()
        }
        print(f"{name:<20}: delta = {delta.item():.5f}")

    return results
```

</div>

</div>

### Measuring Baseline Agreement

If attributions are robust to baseline choice, the rankings should agree:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import scipy.stats

def baseline_rank_correlation(results):
    """Measure Spearman rank correlation between baseline attributions."""
    names = list(results.keys())
    attrs = {n: results[n]['attr'].squeeze().abs().numpy().flatten()
             for n in names}

    print("Rank correlation matrix (higher = more agreement):")
    for n1 in names:
        for n2 in names:
            rho, _ = scipy.stats.spearmanr(attrs[n1], attrs[n2])
            print(f"  {n1} vs {n2}: rho = {rho:.3f}")
```

</div>

</div>

---

## 4. Convergence Diagnostics

### The Convergence Delta
<div class="callout-insight">

<strong>Insight:</strong> The convergence delta $\delta$ measures the numerical integration error:

</div>


The convergence delta $\delta$ measures the numerical integration error:

$$\delta = \left| \sum_i \text{IG}_i^{\text{approx}}(x) - (f(x) - f(x')) \right|$$

In Captum:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
attr, delta = ig.attribute(
    inputs, baselines=baseline, target=class_idx,
    n_steps=50, return_convergence_delta=True
)
print(f"Convergence delta: {delta.item():.5f}")
```

</div>

</div>

### What Delta Values Mean

| Delta | Interpretation | Action |
|-------|----------------|--------|
| < 0.01 | Excellent | No action needed |
| 0.01 – 0.05 | Good | Acceptable for most uses |
| 0.05 – 0.10 | Marginal | Increase n_steps to 100 |
| > 0.10 | Poor | Increase n_steps to 300+ |

### Why Delta Is Large

Large convergence delta usually means:
1. Too few steps (`n_steps` too small)
2. Highly non-linear model in the integration region
3. Baseline very far from the input (long path, many nonlinearities)
4. Numerical issues with gradient computation

### Fixing Large Delta


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Start with n_steps=50, increase if delta > 0.05
for n_steps in [50, 100, 200, 300, 500]:
    attr, delta = ig.attribute(
        inputs, baselines=baseline, target=class_idx,
        n_steps=n_steps, return_convergence_delta=True
    )
    print(f"n_steps={n_steps}: delta={delta.item():.5f}")
    if abs(delta.item()) < 0.05:
        print(f"Converged at n_steps={n_steps}")
        break
```

</div>


---

## 5. NoiseTunnel on Integrated Gradients

Applying SmoothGrad to IG produces smooth, high-quality attributions:
<div class="callout-warning">

<strong>Warning:</strong> Applying SmoothGrad to IG produces smooth, high-quality attributions:



<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
from captum.attr import IntegratedGradients, NoiseTunnel

ig = IntegratedGradients(model)
nt = NoiseTunnel(ig)

# SmoothGrad-IG: average IG over noisy inputs
smooth_ig_attr = nt.attribute(
    input_tensor,
    nt_type='smoothgrad',
    nt_samples=10,          # 10 noise samples (× n_steps = 10×50=500 passes)
    stdevs=0.05,            # Noise std
    baselines=baseline,
    target=class_idx,
    n_steps=50
)
```



**Cost:** `nt_samples × n_steps` total passes. With `nt_samples=10, n_steps=50`: 500 passes.

### When to Use NoiseTunnel on IG

- When IG maps appear noisy despite high n_steps
- When producing publication-quality visualizations
- When input images have high-frequency texture that creates gradient noise
- **Not** for real-time applications (expensive)

### IG vs SmoothGrad-IG vs SmoothGrad-Saliency

| Method | Quality | Cost | Satisfies axioms? |
|--------|---------|------|-------------------|
| Saliency | Noisy | 1× | No |
| SmoothGrad-Saliency | Clean | 20× | No |
| IG (50 steps) | Good | 50× | Yes |
| SmoothGrad-IG (10×50) | Very clean | 500× | Yes (approx) |

For most practical purposes, IG at 50 steps is the best balance.

---

## 6. Layer Integrated Gradients for Text

For transformer models, token-level attribution requires `LayerIntegratedGradients`:
<div class="callout-key">

<strong>Key Point:</strong> For transformer models, token-level attribution requires `LayerIntegratedGradients`:



<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
from captum.attr import LayerIntegratedGradients

# Model and embedding layer
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
embedding_layer = model.distilbert.embeddings

lig = LayerIntegratedGradients(model, embedding_layer)

def attribute_text(text, tokenizer, lig_method, target_class=1):
    """
    Compute token-level IG attributions for text input.

    Returns:
    - tokens: list of token strings
    - attributions: per-token importance scores
    """
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Baseline: [PAD] token IDs
    baseline_ids = torch.zeros_like(input_ids)

    def forward_func(inp_ids, att_mask):
        return model(input_ids=inp_ids, attention_mask=att_mask).logits

    attributions, delta = lig_method.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        target=target_class,
        n_steps=50,
        return_convergence_delta=True
    )

    # Sum across embedding dimension for per-token importance
    token_attr = attributions.sum(dim=-1).squeeze(0).detach()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    return tokens, token_attr
```



### Visualizing Token Attributions


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def visualize_token_attribution(tokens, attributions, title=""):
    """
    Visualize token attributions as colored text.
    Positive = green (supports positive class), negative = red.
    """
    attr_norm = attributions / (attributions.abs().max() + 1e-8)

    fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.8), 1.5))
    ax.set_xlim(0, len(tokens))
    ax.set_ylim(0, 1)
    ax.axis('off')

    x_pos = 0
    for token, score in zip(tokens, attr_norm.numpy()):
        color = (0.2, 0.8, 0.2, min(abs(score) + 0.1, 1.0)) if score > 0 \
                else (0.8, 0.2, 0.2, min(abs(score) + 0.1, 1.0))
        ax.text(
            x_pos + 0.4, 0.5, token,
            ha='center', va='center',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=color, edgecolor='none', alpha=0.7)
        )
        x_pos += 0.8

    plt.title(title)
    plt.tight_layout()
    plt.show()
```



---

## Common Pitfalls

- **Using a meaningful baseline "for convenience":** A baseline that already contains your expected signal (e.g., using a low-quality image as baseline for quality classification) will inflate attribution of features unique to the higher-quality image.
- **Not checking convergence:** Always use `return_convergence_delta=True` during development.
- **Confusing high delta with wrong attributions:** High delta means the numerical approximation is poor — increase n_steps before trusting the attributions.
- **NoiseTunnel with return_convergence_delta:** NoiseTunnel currently does not return convergence delta — validate the underlying IG without NoiseTunnel first, then add NoiseTunnel.

---

## Connections

- **Builds on:** Guide 01 (IG theory and completeness property)
- **Leads to:** Module 02 notebooks (empirical baseline comparison, text attribution)
- **Related to:** Expected Gradients (SHAP's baseline averaging approach), Guided IG (path choice)

---


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Key Insight" and why it matters in practice.

2. Given a real-world scenario involving baselines, convergence, and noisetunnel, what would be your first three steps to apply the techniques from this guide?


## Further Reading

- Sturmfels et al. (2020). Visualizing the Impact of Feature Attribution Baselines. *Distill* — Visual exploration of how baseline choice affects attributions.
- Smilkov et al. (2017). SmoothGrad. *arXiv* — Original SmoothGrad paper.
- Fong & Vedaldi (2017). Interpretable Explanations of Black Boxes by Meaningful Perturbation. *ICCV* — Blurred baseline and meaningful perturbation.

---

## Cross-References

<a class="link-card" href="../notebooks/01_ig_image_classification.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
