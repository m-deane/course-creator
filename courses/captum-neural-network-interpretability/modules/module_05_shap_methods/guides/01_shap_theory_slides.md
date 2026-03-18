---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# SHAP Theory
## Shapley Values from Cooperative Game Theory

Module 05 — SHAP and KernelSHAP in Captum

<!-- Speaker notes: Welcome to the SHAP theory module. SHAP is arguably the most theoretically grounded attribution method in interpretability. It derives directly from cooperative game theory, a branch of mathematics developed in the 1950s by Lloyd Shapley. Today we'll build up from first principles to the KernelSHAP approximation used in Captum. -->

---

## The Fair Division Problem

**Scenario:** Three players collaborate to earn $100. How do we split it fairly?

```
Player A alone:   earns $30
Player B alone:   earns $40
Players A+B:      earn $90
Player C alone:   earns $0
All three:        earn $100
```

**Key insight:** Value comes from *collaboration* — but we need a principled way to credit individual contributions.

> In ML: features are players, the prediction is the value to distribute.

<!-- Speaker notes: Start with an intuition before the math. The fair division problem is ancient — how do you credit contributions when value emerges from collaboration? This is exactly what SHAP solves for model predictions. Feature A alone might not be predictive, but combined with feature B, they become powerful. SHAP allocates credit for that joint contribution fairly. -->

---

## From Game Theory to Machine Learning

<div class="columns">

**Cooperative Game**
- Players: $\{1, 2, \ldots, d\}$
- Coalition: subset $S \subseteq \{1,\ldots,d\}$
- Value function: $v: 2^{\{1,\ldots,d\}} \to \mathbb{R}$

**ML Attribution**
- Players: features $x_1, \ldots, x_d$
- Coalition: features used for prediction
- Value function:
$$v(S) = \mathbb{E}_{X_{\bar{S}}}[f(X) \mid X_S = x_S]$$

</div>

The value function **averages over missing features** — if a feature isn't in the coalition, we marginalize it out using the data distribution.

<!-- Speaker notes: The translation from game theory to ML is elegant. Features are players. A coalition is a subset of features we reveal to the model. The value is the model's expected output given only those features. The expectation over missing features is crucial — it means we're computing what the model would predict if it only had access to the coalition features, averaging over all possible values of the rest. -->

---

## The Shapley Value Formula

$$\boxed{\phi_i(v) = \sum_{S \subseteq \{1,\ldots,d\} \setminus \{i\}} \frac{|S|!\,(d - |S| - 1)!}{d!} \left[v(S \cup \{i\}) - v(S)\right]}$$

**Decomposed:**

| Term | Meaning |
|------|---------|
| $v(S \cup \{i\}) - v(S)$ | Marginal contribution of feature $i$ to coalition $S$ |
| $\frac{\|S\|!(d-\|S\|-1)!}{d!}$ | Fraction of orderings where $S$ precedes $i$ |
| $\sum_S$ | Average over all possible coalitions |

**Intuition:** Shapley value = average marginal contribution across all orderings.

<!-- Speaker notes: This formula looks dense but the intuition is simple. For each possible coalition S that doesn't include feature i, we compute: what does adding feature i to that coalition change? Then we average these marginal contributions, weighted by how often each coalition appears in a random ordering. The weighting term ensures we average uniformly over all n! orderings. -->

---

## Visualizing Marginal Contributions

For $d=3$ features, all 6 orderings:

```
A → B → C:  φ_B = v({A,B}) - v({A})
A → C → B:  φ_B = v({A,B,C}) - v({A,C})
B → A → C:  φ_B = v({B}) - v({})
B → C → A:  φ_B = v({B}) - v({})
C → A → B:  φ_B = v({A,B,C}) - v({A,C})
C → B → A:  φ_B = v({B,C}) - v({C})
```

$$\phi_B = \frac{1}{6}\left[(v(\{A,B\})-v(\{A\})) + 2(v(\{A,B,C\})-v(\{A,C\})) + 2(v(\{B\})-v(\{\})) + (v(\{B,C\})-v(\{C\}))\right]$$

<!-- Speaker notes: This slide makes the averaging concrete. With 3 features, there are 6 orderings. For each ordering, we look at the marginal contribution of B when it enters — what did B add to whoever came before it? Average these 6 marginal contributions and you have the Shapley value for B. With more features, this becomes 2^d coalitions to evaluate. -->

---

## The Four Axioms

Shapley values are the **unique** attribution satisfying all four:

<div class="columns">

**Efficiency**
$$\sum_{i=1}^d \phi_i = f(x) - \mathbb{E}[f(X)]$$
Attributions sum to total prediction change.

**Symmetry**
Equal contributors get equal credit:
$$v(S\cup\{i\}) = v(S\cup\{j\}) \Rightarrow \phi_i = \phi_j$$

</div>

<div class="columns">

**Dummy**
Zero contributors get zero credit:
$$v(S\cup\{i\}) = v(S)\ \forall S \Rightarrow \phi_i = 0$$

**Linearity**
$$\phi_i(v + w) = \phi_i(v) + \phi_i(w)$$
Attributions compose across models.

</div>

<!-- Speaker notes: These four axioms are what make SHAP theoretically compelling. Efficiency means attributions are conservative — nothing is lost or double-counted. The total attribution equals exactly the prediction minus the expected prediction. Symmetry means the method is unbiased. Dummy means truly irrelevant features don't get credit. Linearity means we can decompose attribution across ensemble members. Shapley proved these axioms uniquely determine the Shapley value — there is no other method satisfying all four. -->

---

## The Computational Challenge

$$\text{Coalitions: } 2^d \quad \text{For } d=20: \quad 2^{20} = 1{,}048{,}576$$

For each coalition, we need:
$$v(S) = \frac{1}{n_{bg}} \sum_{k=1}^{n_{bg}} f(x_S, x^{(k)}_{\bar{S}})$$

**Total evaluations:** $2^d \times n_{bg}$

| Features | Background | Model calls |
|----------|-----------|-------------|
| 10 | 100 | 102,400 |
| 20 | 100 | 104,857,600 |
| 100 | 100 | $10^{32}$ (impossible) |

**Solution:** KernelSHAP — approximate via weighted regression.

<!-- Speaker notes: This is the curse of dimensionality hitting SHAP. Exact computation is only feasible for small feature counts. For real models with hundreds of features, we need an approximation. KernelSHAP samples a subset of coalitions and fits a weighted linear regression — and crucially, Lundberg and Lee proved this gives exact Shapley values in expectation. -->

---

## KernelSHAP: The Key Insight

**Claim:** Shapley values minimize this weighted regression:

$$\mathcal{L}(f, g, \pi_x) = \sum_{z' \in \{0,1\}^d} \pi_x(z') \left[f(h_x(z')) - g(z')\right]^2$$

where $g(z') = \phi_0 + \sum_i \phi_i z'_i$ and:

$$\boxed{\pi_x(z') = \frac{d - 1}{\dbinom{d}{|z'|}\,|z'|\,(d - |z'|)}}$$

The **SHAP kernel** assigns higher weight to very small and very large coalitions.

<!-- Speaker notes: This is the mathematical heart of KernelSHAP. We're looking for a linear function g that best explains the model f. The SHAP kernel pi_x determines how much each coalition matters. The kernel puts high weight on coalitions of size 0, 1, d-1, and d — these are the coalitions that most constrain the Shapley values. In practice, we sample coalitions according to this distribution rather than enumerating all 2^d. -->

---

## SHAP Kernel Weight Distribution

```python
import numpy as np
import matplotlib.pyplot as plt

d = 10
sizes = np.arange(1, d)
weights = [(d-1) / (comb(d, s) * s * (d-s)) for s in sizes]

plt.bar(sizes, weights)
plt.xlabel("Coalition size |z'|")
plt.ylabel("SHAP kernel weight π(z')")
plt.title("KernelSHAP concentrates on extreme coalitions")
```

```
Coalition size:  1    2    3   ...   7    8    9
Kernel weight:  HIGH low  low  ...  low  low  HIGH
```

Small coalitions (few features included) and large coalitions (most features included) get the highest weight.

<!-- Speaker notes: The SHAP kernel shape is important to understand. The method focuses computational effort on coalitions near size 1 and near size d-1. Why? Because these coalitions give the most information about individual feature contributions. A coalition of size 1 tells us the direct effect of a single feature; a coalition of size d-1 tells us what happens when we remove one feature. Mid-sized coalitions contain higher-order interaction information but are less diagnostic for Shapley values. -->

---

## KernelSHAP Algorithm

```mermaid
flowchart LR
    A[Sample M coalitions\nz' ~ π_x] --> B[Map z' to input\nhx: {0,1}^d → R^d]
    B --> C[Evaluate model\nf for each masked input]
    C --> D[Weighted linear\nregression with SHAP kernel]
    D --> E[φ₁, φ₂, ..., φd\nShapley values]
```

**Step 2 in detail:** If feature $i$ is absent ($z'_i = 0$), replace $x_i$ with a random sample from the background dataset.

This simulates "not knowing" the feature's value.

<!-- Speaker notes: The algorithm is straightforward once you understand the key mapping. We sample coalition masks, replace absent features with background samples (this is the marginalization), evaluate the model on these perturbed inputs, then fit a weighted regression. The regression coefficients are our Shapley value estimates. More samples M gives better estimates at higher computational cost. Captum's KernelShap implementation handles all of this. -->

---

## Missing Feature Simulation

**How does KernelSHAP simulate "feature absent"?**

```python
# Background dataset: representative samples from training data
background = X_train[np.random.choice(len(X_train), 100)]

# For coalition S={1,3}, features 2,4,5,... are "absent"
# Replace absent features with random background samples:
masked_input = x.copy()
for absent_feature in absent_features:
    bg_sample = background[np.random.randint(len(background))]
    masked_input[absent_feature] = bg_sample[absent_feature]
```

**Background dataset choices:**
- 50-200 samples from training distribution
- K-means cluster centers (compact, representative)
- Never use test data (leakage)

<!-- Speaker notes: The background dataset is a crucial design choice. When we remove a feature from a coalition, we need to replace it with something. KernelSHAP uses random samples from the background dataset. This marginalizes over the feature's distribution, which is theoretically correct. The background should represent the training distribution — if it doesn't, your Shapley values will be biased. K-means centers are a common choice that captures distribution shape with few samples. -->

---

## Captum KernelSHAP: Code

```python
import torch
from captum.attr import KernelShap

model.eval()
kernel_shap = KernelShap(model)

# Background: mean of training data or representative samples
baseline = X_train.mean(dim=0, keepdim=True)

# Compute SHAP values for one sample
attributions = kernel_shap.attribute(
    inputs=x_test,          # shape: (1, num_features)
    baselines=baseline,     # shape: (1, num_features)
    n_samples=500,          # coalition samples (more = more accurate)
    perturbations_per_eval=64,  # batch size
    target=predicted_class,
)
# attributions[i] = φ_i (Shapley value for feature i)
```

**Output:** Tensor of same shape as input, values are $\phi_i$ for each feature.

<!-- Speaker notes: The Captum API is clean. You wrap your model with KernelShap, provide an input and baseline, and call attribute. The n_samples parameter is the number of coalition samples — 200-500 is reasonable for tabular data. perturbations_per_eval batches the model evaluations for GPU efficiency. The output is a tensor of the same shape as the input, where each value is the Shapley value for that feature or pixel. -->

---

## SHAP Value Interpretation

$$f(x) = \underbrace{\mathbb{E}[f(X)]}_{\text{baseline}} + \underbrace{\phi_1 + \phi_2 + \cdots + \phi_d}_{\text{feature contributions}}$$

```
Prediction: 0.82  (probability of default)
Baseline:   0.35  (population average)
────────────────────────────────────────
income:     -0.18  ← pushes DOWN (good income)
debt_ratio: +0.31  ← pushes UP  (high debt)
age:        -0.09  ← pushes DOWN (older = less risk)
employment: +0.43  ← pushes UP  (unemployed)
────────────────────────────────────────
Sum:        +0.47  ✓ matches 0.82 - 0.35
```

<!-- Speaker notes: This is what SHAP values look like in practice. The baseline is the model's average prediction. Each feature's Shapley value tells you how much that feature pushes the prediction up or down relative to baseline. In this credit risk example, high debt ratio adds 0.31 and unemployment adds 0.43, while good income subtracts 0.18. These sum to 0.47, which equals the prediction minus baseline. The efficiency axiom guarantees this accounting. -->

---

## SHAP vs. Other Attribution Methods

| Property | SHAP | IG | Saliency | Occlusion |
|----------|------|-----|---------|----------|
| Efficiency axiom | ✓ | ✓ | ✗ | ✗ |
| Symmetry axiom | ✓ | ✗ | ✗ | ✗ |
| Dummy axiom | ✓ | ✓ | ✗ | ✓ |
| Model-agnostic | ✓ | ✗ | ✗ | ✓ |
| Computational cost | High | Medium | Low | Medium |
| Handles interactions | ✓ | Partial | ✗ | Partial |

**KernelSHAP** satisfies all four axioms for any black-box model.

<!-- Speaker notes: This comparison shows why SHAP is considered the gold standard theoretically. It's the only method satisfying all four axioms AND being model-agnostic. Integrated Gradients satisfies efficiency and dummy but not symmetry. Saliency satisfies none of the fairness axioms. The cost is computational — SHAP is slower than gradient methods. For neural networks, GradientSHAP and DeepLIFTSHAP provide faster approximations that sacrifice some theoretical guarantees. -->

---

## SHAP as a Unifying Framework

**Lundberg & Lee (2017) showed many methods are SHAP variants:**

```
KernelSHAP with SHAP kernel  →  Exact Shapley values
KernelSHAP with LIME kernel  →  LIME (approximate, biased)
SHAP + gradient expectations →  GradientSHAP ≈ Integrated Gradients
SHAP + DeepLIFT propagation  →  DeepLIFT SHAP
SHAP + tree structure        →  TreeSHAP (exact, fast for trees)
```

SHAP provides **principled unification** of the attribution landscape.

<!-- Speaker notes: One of SHAP's most important contributions is showing that existing methods are approximate or special cases of SHAP. LIME, which was very popular, turns out to be KernelSHAP with the wrong kernel — it doesn't satisfy the efficiency or symmetry axioms. GradientSHAP uses the expected gradient over baseline samples, which approximates Shapley values for smooth models. This unification helps practitioners choose the right method for their situation. -->

---

## Summary

<div class="columns">

**Key Formulas**

Shapley value:
$$\phi_i = \sum_S \frac{|S|!(d-|S|-1)!}{d!}[v(S\cup\{i\}) - v(S)]$$

Efficiency:
$$\sum_i \phi_i = f(x) - \mathbb{E}[f(X)]$$

SHAP kernel:
$$\pi_x(z') = \frac{d-1}{\binom{d}{|z'|}|z'|(d-|z'|)}$$

**Key Takeaways**

- Shapley values uniquely satisfy 4 axioms
- KernelSHAP approximates via weighted regression
- Background dataset simulates missing features
- Captum: `KernelShap(model).attribute(...)`
- Tradeoff: theory-optimal but computationally heavy

</div>

<!-- Speaker notes: To summarize: Shapley values from cooperative game theory give us the theoretically optimal way to attribute a model's prediction to its features. The four axioms — efficiency, symmetry, dummy, linearity — uniquely determine the Shapley value. KernelSHAP makes this tractable by sampling coalitions and fitting a weighted regression. In Captum, this is wrapped in the clean KernelShap API. Next we'll look at gradient-based SHAP methods that are faster for neural networks. -->

---

<!-- _class: lead -->

## Next: GradientSHAP and DeepLIFT SHAP

Faster SHAP variants that exploit neural network structure

**Guide 02:** `02_gradient_deep_shap_guide.md`

<!-- Speaker notes: Now that we understand the theoretical foundation, the next guide covers GradientSHAP and DeepLIFT SHAP — methods that achieve faster computation by leveraging the differentiable structure of neural networks. These are typically preferred for deep networks over KernelSHAP. -->
