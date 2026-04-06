# Influence Functions, TracIn, and TracInCPFast

> **Reading time:** ~8 min | **Module:** 6 — Concept & Example Explanations | **Prerequisites:** Module 3 Layer Attribution


## Learning Objectives

By the end of this guide, you will be able to:
1. Explain the influence function approximation for measuring training data influence
2. Describe TracIn's checkpointing-based approach and why it improves over influence functions
3. Implement TracInCPFast in Captum to find proponents and opponents for a test prediction
4. Visualize and interpret influential training examples
5. Apply influence methods for data debugging and mislabeled example detection


<div class="callout-key">

<strong>Key Concept Summary:</strong> This guide covers the core concepts of influence functions, tracin, and tracincpfast.

</div>

---

## 1. The Core Question: Example-Based Explanations

Attribution methods explain predictions in terms of input features. But a complementary question is: **which training examples most influenced this prediction?**

This matters because:
- **Mislabeled data detection:** If the most influential training example for a "cat" prediction is actually labeled "dog," the model may be confused
- **Spurious correlation discovery:** If all influential examples for a "wolf" prediction are in snowy settings, the model may be predicting snow, not wolf
- **Data quality auditing:** Understanding which training examples drive model behavior
- **Counterfactual reasoning:** "If I had labeled this training example differently, how would the prediction change?"

---

## 2. Classical Influence Functions

### Setup

Let $\mathcal{D} = \{(x_1, y_1), \ldots, (x_n, y_n)\}$ be the training set and $\hat{\theta}$ be the trained parameters minimizing:

$$\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^n \ell(x_i, y_i, \theta)$$

The **influence of training example $(x_i, y_i)$ on the prediction $f(x_{test}, \hat{\theta})$** is defined as the change in test loss if that training example were removed:

$$\mathcal{I}(x_i, x_{test}) = -\frac{1}{n} \nabla_\theta \ell(x_{test}, \hat{\theta})^\top H_{\hat{\theta}}^{-1} \nabla_\theta \ell(x_i, y_i, \hat{\theta})$$

where $H_{\hat{\theta}} = \frac{1}{n} \sum_i \nabla_\theta^2 \ell(x_i, y_i, \hat{\theta})$ is the Hessian matrix.

### Derivation Intuition

If we upweight training example $i$ by $\epsilon$, the optimal parameters shift by:

$$\Delta\hat{\theta} \approx -\frac{1}{n} H_{\hat{\theta}}^{-1} \nabla_\theta \ell(x_i, y_i, \hat{\theta}) \cdot \epsilon$$

The effect on the test loss is then:

$$\Delta \mathcal{L}_{test} \approx \nabla_\theta \ell(x_{test})^\top \Delta\hat{\theta}$$

Combining gives the influence formula above.

---

## 3. The Hessian Problem

Computing $H_{\hat{\theta}}^{-1}$ for deep neural networks is computationally infeasible:
- A ResNet-50 has ~25 million parameters
- The Hessian is a 25M × 25M matrix — impossible to store or invert

Classical influence functions require expensive approximations (LiSSA, Conjugate Gradient) that can be unstable for non-convex models.

This limitation motivated TracIn.

---

## 4. TracIn: Training Data Influence via Checkpoints

TracIn (Pruthi et al., 2020) takes a completely different approach: instead of inverting the Hessian, it estimates influence by tracing the loss trajectories during training.

### The TracIn Formula

$$\text{TracIn}(z, z_{test}) = \sum_{t \in \text{checkpoints}} \nabla_\theta \ell(z_{test}, \theta_t) \cdot \nabla_\theta \ell(z, \theta_t) \cdot \eta_t$$

where:
- $z = (x, y)$ is a training example
- $z_{test}$ is the test example
- $\theta_t$ is the model checkpoint at step $t$
- $\eta_t$ is the learning rate at step $t$
- $\cdot$ is the dot product of gradient vectors

### Interpretation

$\text{TracIn}(z, z_{test}) > 0$ means training on example $z$ **increased** the loss on $z_{test}$ — example $z$ is a **proponent** that reinforces the model's behavior on $z_{test}$.

$\text{TracIn}(z, z_{test}) < 0$ means training on $z$ **decreased** the loss on $z_{test}$ — example $z$ is an **opponent** that pushes against the prediction.

---

## 5. TracIn Algorithm

```

For each training checkpoint θ_t:
    1. Compute gradient of test loss: g_test = ∇_θ L(z_test, θ_t)
    2. Compute gradient of training loss: g_train = ∇_θ L(z, θ_t)
    3. Contribution: η_t * dot(g_test, g_train)

TracIn score = sum over all checkpoints
```

The key insight: gradient alignment across checkpoints is a proxy for training data influence. If a training example's gradient consistently aligns with the test gradient, it has been "helping" the model on that test example throughout training.

---

## 6. TracInCPFast: Efficient Approximation

Computing full gradient dot products is expensive ($O(n \cdot |\theta|)$ per test example). TracInCPFast approximates by using only the **last layer gradients**:

$$\text{TracInCPFast}(z, z_{test}) \approx \sum_t \nabla_{\theta_{last}} \ell(z_{test}, \theta_t) \cdot \nabla_{\theta_{last}} \ell(z, \theta_t) \cdot \eta_t$$

This is practical because:
- Last layer gradients are small (just the linear head)
- The approximation works well in practice (high rank correlation with full TracIn)
- Linear time in training set size

---

## 7. Captum TracIn Implementation

### Setup: Saving Checkpoints
<div class="callout-key">

<strong>Key Point:</strong> TracIn requires model checkpoints saved during training:

</div>


TracIn requires model checkpoints saved during training:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import torch
import torch.nn as nn
from torch.optim import SGD

model = MyModel()
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

checkpoint_paths = []
for epoch in range(50):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch[0]), batch[1])
        loss.backward()
        optimizer.step()

    # Save checkpoint every 10 epochs
    if epoch % 10 == 0:
        path = f"checkpoints/model_epoch_{epoch}.pt"
        torch.save(model.state_dict(), path)
        checkpoint_paths.append((path, 0.01))  # (path, learning_rate)
```

</div>

</div>

### Running TracInCPFast


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
from captum.influence import TracInCPFast

tracin = TracInCPFast(
    model=model,
    train_dataset=train_dataset,
    checkpoints=checkpoint_paths,   # list of (path, lr) tuples
    checkpoints_load_func=lambda m, p: m.load_state_dict(torch.load(p)),
    loss_fn=nn.CrossEntropyLoss(reduction='none'),
    batch_size=64,
    vectorize=False,
)

# Find most influential training examples for a test input
test_input, test_target = test_dataset[0]
test_input = test_input.unsqueeze(0)
test_target = torch.tensor([test_target])

# Top-k proponents (highest influence)
proponents = tracin.influence(
    inputs=(test_input, test_target),
    top_k=5,
    additional_forward_args=None,
    unpack_inputs=True,
)

# proponents: (influences, indices) for most helpful training examples
```

</div>

</div>

---

## 8. SimilarityInfluence: Representation Similarity

Captum also provides `SimilarityInfluence`, which finds training examples most similar to a test input in the model's learned representation space:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
from captum.influence import SimilarityInfluence

sim_influence = SimilarityInfluence(
    module=model,
    layers=["layer4"],      # which layer's representation to use
    influence_src_dataset=train_dataset,
    activation_dir="./activations/",   # cache directory
    model_id="resnet18",
    batch_size=64,
    load_from_disk=True,    # use cached activations if available
)

# Find most similar training examples
top_similar = sim_influence.influence(
    inputs=test_input,
    top_k=5,
)
```

</div>

</div>

`SimilarityInfluence` is a simpler, faster alternative when you care about similarity rather than gradient-based influence.

---

## 9. Proponents vs. Opponents

The most useful TracIn analysis finds:

**Top proponents:** Training examples that most contributed to the current prediction
- If a test image is misclassified, proponents reveal what drove the error
- If proponents are mislabeled, that explains the mistake

**Top opponents:** Training examples whose removal would most increase test confidence
- Opponents push against the prediction
- May reveal contradictory information in the training set


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Get both proponents and opponents
influences, train_indices = tracin.influence(
    inputs=(test_input, test_target),
    top_k=5,
    proponents=True,   # True for proponents, False for opponents
)

# Visualize
fig, axes = plt.subplots(1, 6, figsize=(18, 3))
axes[0].imshow(test_image_np)
axes[0].set_title('Test Image', fontweight='bold')
for i, (idx, inf_val) in enumerate(zip(train_indices[0], influences[0])):
    train_img, train_label = train_dataset[idx.item()]
    axes[i+1].imshow(to_numpy(train_img))
    axes[i+1].set_title(f'Proponent {i+1}\nInf={inf_val:.3f}\nLabel={label_names[train_label]}')
```

</div>


---

## 10. Mislabeled Example Detection

TracIn can identify mislabeled training examples by finding examples where:
1. The model predicts a different class than the label
2. The TracIn self-influence is low (the example is an outlier relative to its class)

### Self-Influence

**Self-influence** measures how much each training example influenced its own prediction during training:

$$\text{SelfInfluence}(z) = \sum_t \|\nabla_\theta \ell(z, \theta_t)\|^2 \cdot \eta_t$$

High self-influence examples are "difficult" — the model had high loss on them throughout training. These are often:
- Mislabeled examples
- Atypical examples (genuine but unusual)
- Ambiguous examples near the decision boundary


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python

# Compute self-influence for all training examples
self_influences = tracin.self_influence(
    inputs=train_loader,
    show_progress=True,
)

# Visualize high self-influence examples (likely mislabeled or unusual)
high_si_indices = self_influences.argsort(descending=True)[:20]
```



---

## 11. Limitations

### Checkpoint Requirements
TracIn requires checkpoints saved during training. If you only have the final model, you cannot run TracIn. Plan for checkpoint saving at training time.

### Computational Cost
Even TracInCPFast scales as $O(|\mathcal{D}| \times |\text{checkpoints}|)$. For training sets of 100K+ examples, this can take significant compute.

### Gradient Approximation
TracInCPFast uses only last-layer gradients. For models where the informative learning happens in earlier layers (common in fine-tuning), this may be a poor approximation.

### Influence Interpretation
High TracIn influence is not the same as causation. Example $z$ having high influence on $z_{test}$ means their gradients aligned during training — not necessarily that $z$ caused the prediction on $z_{test}$.

---

## 12. Choosing Between Methods

| Method | Best for | Computation | Requires |
|--------|----------|-------------|---------|
| TracIn | Finding true training data influence | High | Checkpoints |
| TracInCPFast | Large-scale influence analysis | Medium | Checkpoints |
| SimilarityInfluence | Finding similar training examples | Low | One model |
| Classical IF | Theoretically exact influence | Very high | Hessian approximation |

For most practical use cases, **TracInCPFast** (when checkpoints exist) or **SimilarityInfluence** (when only the final model is available) are the recommended choices.

---


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "The Core Question: Example-Based Explanations" and why it matters in practice.

2. Given a real-world scenario involving influence functions, tracin, and tracincpfast, what would be your first three steps to apply the techniques from this guide?




## Summary

| Concept | Description |
|---------|-------------|
| TracIn | Traces gradient alignment across checkpoints |
| TracInCPFast | Last-layer gradient approximation of TracIn |
| Proponent | Training example that increased test loss (reinforces prediction) |
| Opponent | Training example that decreased test loss (contradicts prediction) |
| Self-influence | How much an example influenced its own training |
| SimilarityInfluence | Representation-space nearest neighbor approach |

---

## Further Reading

- Koh, P.W., & Liang, P. (2017). Understanding black-box predictions via influence functions. *ICML*.
- Pruthi, G., et al. (2020). Estimating training data influence by tracing gradient descent. *NeurIPS*.
- Feldman, V., & Zhang, C. (2020). What neural networks memorize and why: Discovering the long tail via influence estimation. *NeurIPS*.
- Captum TracIn documentation: https://captum.ai/api/influence.html

---

## Cross-References

<a class="link-card" href="../notebooks/01_tcav_texture_shape.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
