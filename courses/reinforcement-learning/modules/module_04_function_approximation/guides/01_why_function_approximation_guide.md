# Why Function Approximation?

> **Reading time:** ~13 min | **Module:** 4 — Function Approximation | **Prerequisites:** Module 3

## In Brief

Tabular methods store a value for every state (or state-action pair). When the state space is large or continuous, that table becomes impossibly big. Function approximation replaces the table with a parameterized function $\hat{v}(s, \mathbf{w}) \approx V^\pi(s)$ that generalizes across states, making RL tractable in the real world.

<div class="callout-key">

<strong>Key Concept:</strong> Tabular methods store a value for every state (or state-action pair). When the state space is large or continuous, that table becomes impossibly big.

</div>


## Key Insight

A table is just a lookup function with one parameter per entry. Function approximation is the same idea with far fewer parameters — it trades exact representation of each state for the ability to generalize from states seen during training to states never visited.

---


<div class="callout-key">

<strong>Key Point:</strong> A table is just a lookup function with one parameter per entry.

</div>

## 1. The Curse of Dimensionality

### What It Is

<div class="callout-key">

<strong>Key Point:</strong> ### What It Is

The state space of a tabular RL agent grows exponentially with the number of state dimensions.

</div>


The state space of a tabular RL agent grows exponentially with the number of state dimensions. For $d$ binary features, the table needs $2^d$ entries.

```
1 dimension  (4 positions):         4 states
2 dimensions (4 x 4 grid):         16 states
3 dimensions (4 x 4 x 4 cube):     64 states
d dimensions:                      4^d states
```

### Why It Kills Tabular RL

A real-world robot arm has joint angles, velocities, and torques — each continuous. Discretizing each angle to 100 bins and tracking 10 joints gives $100^{10} = 10^{20}$ states. Storing a Q-value per entry at 8 bytes requires $8 \times 10^{20}$ bytes — roughly $8 \times 10^8$ terabytes. No computer has that.

Even ignoring memory, the agent would need to visit each state enough times to form reliable estimates. With $10^{20}$ states, a million steps of experience covers a negligible fraction.

### Formal Statement

Let $|\mathcal{S}|$ denote the number of states. Tabular storage is $\mathcal{O}(|\mathcal{S}|)$. If the state is a vector $\mathbf{s} \in \mathbb{R}^d$ discretized at resolution $m$, then $|\mathcal{S}| = m^d$. Storage grows exponentially in $d$.

---

## 2. Generalization: Sharing Knowledge Across Similar States

Tabular methods treat every state as unrelated to every other. An agent that visits state $s_1$ and learns its value gains no information about $s_2$, even if $s_1$ and $s_2$ are nearly identical.

<div class="callout-info">

<strong>Info:</strong> Tabular methods treat every state as unrelated to every other.

</div>


```
Tabular:                     Function Approximation:

s_1 → V(s_1) = 7.2          s_1  ─┐
s_2 → V(s_2) = ?    (no info)     │→  f(s, w) ──→ 7.1   (generalized)
s_3 → V(s_3) = 7.0          s_2  ─┘
```

With function approximation, updating the parameters after visiting $s_1$ changes the predictions for all similar states. This is the central feature — not a side effect.

### The Generalization Assumption

Generalization is beneficial only if similar states truly have similar values. This assumption holds in most physical and game environments (nearby board positions tend to have similar prospects) but can fail in adversarial or highly stochastic settings.

---

## 3. Function Approximation: The Core Idea

### Formal Definition

Instead of a table $V : \mathcal{S} \to \mathbb{R}$, we define a parameterized approximation:

$$\hat{v}(s, \mathbf{w}) \approx V^\pi(s)$$

where $\mathbf{w} \in \mathbb{R}^d$ is a weight vector with far fewer components than $|\mathcal{S}|$.

The same idea applies to action-value functions:

$$\hat{q}(s, a, \mathbf{w}) \approx Q^\pi(s, a)$$

### Differentiability Requirement

For gradient-based learning (the standard approach in RL), we require $\hat{v}(s, \mathbf{w})$ to be differentiable with respect to $\mathbf{w}$:

$$\nabla_{\mathbf{w}} \hat{v}(s, \mathbf{w}) = \left[ \frac{\partial \hat{v}}{\partial w_1}, \frac{\partial \hat{v}}{\partial w_2}, \ldots, \frac{\partial \hat{v}}{\partial w_d} \right]^T$$

This gradient vector tells us how each weight affects the prediction for state $s$.

### Stochastic Gradient Descent (SGD) on VE

We minimize the **Mean Squared Value Error** over the state distribution $\mu(s)$:

$$\overline{VE}(\mathbf{w}) = \sum_{s \in \mathcal{S}} \mu(s) \left[ V^\pi(s) - \hat{v}(s, \mathbf{w}) \right]^2$$

The SGD update for a single sampled state $S_t$ is:

$$\mathbf{w} \leftarrow \mathbf{w} + \alpha \left[ V^\pi(S_t) - \hat{v}(S_t, \mathbf{w}) \right] \nabla_{\mathbf{w}} \hat{v}(S_t, \mathbf{w})$$

In practice we never know $V^\pi(S_t)$, so we substitute a target (Monte Carlo return or TD target) — covered in Guide 02.

---

## 4. Types of Function Approximators

### 4.1 Linear Approximation

$$\hat{v}(s, \mathbf{w}) = \mathbf{w}^T \mathbf{x}(s) = \sum_{i=1}^{d} w_i \, x_i(s)$$

where $\mathbf{x}(s) \in \mathbb{R}^d$ is a hand-crafted or learned **feature vector**. Linear in the weights, not necessarily in the raw state.

**Properties:** Convergence guarantees under on-policy training. Computationally cheap. Interpretable. Requires good feature engineering.

### 4.2 Polynomial Basis

Extend a scalar state $s$ to polynomial features:

$$\mathbf{x}(s) = [1, s, s^2, s^3, \ldots, s^n]^T$$

For multi-dimensional states, cross terms are included (e.g., $x_1 x_2$). The number of features grows polynomially in the degree.

### 4.3 Fourier Basis

Represent the value function as a sum of sinusoids. For a state normalized to $[0,1]^d$:

$$x_i(s) = \cos\left(\pi \, \mathbf{c}_i^T \mathbf{s}\right), \quad \mathbf{c}_i \in \mathbb{Z}^d$$

Fourier features have well-understood approximation error bounds and no cross-feature interference (the basis is orthogonal), which enables per-feature learning rates. Sutton & Barto Chapter 9.5.2.

### 4.4 Tile Coding

Overlay multiple offset grids (tilings) on the state space. Each tiling activates exactly one tile per state. The feature vector is the union of active tiles — a binary vector with exactly one "1" per tiling.

```
State space (1D, s ∈ [0, 1]):

Tiling 1: |--T1--|--T2--|--T3--|--T4--|
Tiling 2:    |--T1--|--T2--|--T3--|--T4--|   (shifted by half a tile)
Tiling 3:      |--T1--|--T2--|--T3--|--T4--|  (shifted by a quarter tile)

For s = 0.35:
  Tiling 1: T2 active
  Tiling 2: T1 active
  Tiling 3: T2 active
  Feature vector: [0,1,0,0, 1,0,0,0, 0,1,0,0, ...]  (sparse, binary)
```

Properties: Sparse and binary features make updates $\mathcal{O}(n_{\text{tilings}})$ instead of $\mathcal{O}(d)$. Generalizes within tiles, not across tile boundaries. Covered in depth in Guide 02.

### 4.5 Radial Basis Functions (RBFs)

Place Gaussian "bumps" at $n$ centers $\mathbf{c}_i$ across the state space:

$$x_i(s) = \exp\left( -\frac{\|s - \mathbf{c}_i\|^2}{2\sigma_i^2} \right)$$

Each feature measures distance from state $s$ to center $i$. Smooth generalization — states close to center $i$ activate feature $i$ strongly.

### 4.6 Neural Networks

Replace the fixed feature map $\mathbf{x}(s)$ with a learned representation:

$$\hat{v}(s, \mathbf{w}) = f_{\text{NN}}(s; \mathbf{w})$$

No manual feature engineering. Can represent arbitrary functions. No convergence guarantee under bootstrapping (the deadly triad — Guide 03). Requires careful engineering (replay buffers, target networks) to be stable.

---

## 5. Feature Engineering for RL

### Why Features Matter

For linear approximators, the feature vector $\mathbf{x}(s)$ determines what information the agent can represent. Raw state variables (pixel values, joint angles) are often poor features. Good features capture the structure of the value function.

### Feature Design Principles

**Relevance:** Include state information that predicts future reward.

**Scale normalization:** Normalize continuous inputs to $[-1, 1]$ or $[0, 1]$ before feeding into features. Large-magnitude inputs dominate weight updates.

**Basis completeness:** The feature space should be rich enough to represent the true value function approximately. More features = better representational capacity but slower learning.

**Sparsity:** Sparse features (many zeros per state) allow $\mathcal{O}(k)$ updates where $k$ is the number of non-zero features, regardless of the total feature dimension.

### Feature Engineering Process


The following implementation builds on the approach above:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np

def normalize_state(s, low, high):
    """Normalize state vector to [0, 1]."""
    return (s - low) / (high - low)

def polynomial_features(s, degree=3):
    """Polynomial feature expansion for scalar state."""
    return np.array([s**k for k in range(degree + 1)])

def rbf_features(s, centers, sigma=1.0):
    """Radial basis function features."""
    diffs = s - centers          # shape: (n_centers,)
    return np.exp(-diffs**2 / (2 * sigma**2))

# Example: CartPole-style state [x, x_dot, theta, theta_dot]
def cartpole_features(state):
    x, x_dot, theta, theta_dot = state
    # Normalize to [-1, 1] range
    x_n         = x / 2.4
    x_dot_n     = x_dot / 3.0
    theta_n     = theta / 0.2095
    theta_dot_n = theta_dot / 3.0
    # Include cross-terms that interact physically
    return np.array([
        1.0,             # bias
        x_n,
        x_dot_n,
        theta_n,
        theta_dot_n,
        x_n * theta_n,          # pole position x cart offset
        x_dot_n * theta_dot_n   # velocity correlation
    ])
```

</div>
</div>

---

## 6. Comparison: Tabular vs Function Approximation

| Property | Tabular | Linear FA | Neural Network FA |
|---|---|---|---|
| State space requirement | Finite, small | Any (with features) | Any |
| Memory | $\mathcal{O}(|\mathcal{S}|)$ | $\mathcal{O}(d_{\text{features}})$ | $\mathcal{O}(|\mathbf{w}|)$ |
| Generalization | None | Via feature similarity | Via learned representation |
| Convergence guarantee | Yes (tabular TD) | Yes (on-policy, linear) | No (off-policy / nonlinear) |
| Feature engineering | Not needed | Required | Optional (raw inputs OK) |
| Interpretability | Full (read the table) | Partial (weight magnitudes) | Low (black box) |
| Sample efficiency | Low (no generalization) | Medium | High (if enough data) |
| Best for | Small, discrete MDPs | Moderate-scale, structured | Large-scale, image/raw input |

---


<div class="compare">
<div class="compare-card">
<div class="header before">6. Comparison: Tabular</div>
<div class="body">

See detailed comparison in the table above.

</div>
</div>
<div class="compare-card">
<div class="header after">Function Approximation</div>
<div class="body">

See detailed comparison in the table above.

</div>
</div>
</div>

## 7. Visual: From Table to Function

```
Tabular (4-state MDP):

  State | Value
  ------+------
    s1  |  2.1
    s2  |  4.7
    s3  |  1.3
    s4  |  3.8

         ↕  Replace with

Linear FA (infinite-state MDP):

  V(s) ≈ w0 + w1·x1(s) + w2·x2(s)
         ↑
   3 weights → approximate any state
   Generalizes across the whole space
```

The following diagram illustrates how different approaches generalize across the state space:

```
Generalization region of each approach:

Tabular:       |■| |■| |■| |■|   (each state isolated)

Tile coding:   |███|███|███|███|  (states in same tile share)
               |  ███|███|███|  |  (across tilings)

RBF:          .·:·.  .·:·.       (smooth, overlapping bumps)

Neural net:   ~~~~~~~~~~~~        (arbitrary smooth function)
```

---

## Common Pitfalls

<div class="callout-danger">

<strong>Danger:</strong> The pitfalls below are the most common mistakes practitioners make. Each one can silently degrade your results without obvious errors.

</div>

**Pitfall 1: Forgetting to normalize features.**
Features with very different scales (e.g., position in meters vs angle in radians) cause one feature to dominate gradient updates. Always normalize inputs before computing feature vectors.

<div class="callout-warning">

<strong>Warning:</strong> **Pitfall 1: Forgetting to normalize features.**
Features with very different scales (e.g., position in meters vs angle in radians) cause one feature to dominate gradient updates.

</div>

**Pitfall 2: Using tabular Q-learning with a continuous state space by discretizing coarsely.**
Coarse discretization loses information that the agent needs to distinguish similar states with different optimal actions. Tile coding or RBFs handle the same problem more gracefully.

**Pitfall 3: Expecting convergence from neural network FA off-policy.**
The convergence guarantees for linear TD(0) do not extend to nonlinear approximators trained off-policy. Divergence is common without stabilization techniques (replay, target networks). See Guide 03.

**Pitfall 4: Too few features for the task.**
If the feature space cannot represent the true value function, no amount of training will produce a good policy. Diagnostic: plot $\hat{v}(s, \mathbf{w})$ against true values across many states and check for systematic bias.

**Pitfall 5: Conflating approximation error and learning error.**
Even with perfect weights, $\hat{v}(s, \mathbf{w})$ approximates $V^\pi(s)$ with error (approximation error). The learning algorithm also introduces error relative to the best approximation (learning error). Both exist simultaneously and must be addressed separately.

---

## Connections


<div class="callout-info">

<strong>Info:</strong> This section maps how this guide connects to the broader course. Use these links to navigate related material.

</div>

- **Builds on:** Module 03 (tabular TD methods), Module 00 (MDP notation and $V^\pi$, $Q^\pi$ definitions)
- **Leads to:** Guide 02 (linear methods and semi-gradient TD), Guide 03 (deadly triad), Module 05 (DQN with neural network FA)
- **Related to:** Supervised learning (FA uses the same regression machinery, but with non-stationary targets)

---

## Practice Problems

1. **Dimensionality check.** A drone is controlled by 6 continuous state variables (x, y, z, roll, pitch, yaw), each discretized to 50 bins. How many table entries are needed? How many weights would a degree-2 polynomial FA need (include all cross terms)?

2. **Feature design.** Design a feature vector for a mountain car problem (state: position $\in [-1.2, 0.6]$, velocity $\in [-0.07, 0.07]$) using tile coding with 4 tilings and 8 tiles per dimension. How many total features does this produce?

3. **Generalization reasoning.** Explain why a tabular agent and a linear FA agent trained on the same environment for the same number of steps might differ in their value estimates for unvisited states.

---

## Further Reading

- Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, 2nd ed., Chapter 9 — the authoritative source for this module's notation and results
- Tsitsiklis & Van Roy (1997), "An analysis of temporal-difference learning with function approximation" — proves convergence of on-policy linear TD(0) to the TD fixed point
- Mnih et al. (2015), "Human-level control through deep reinforcement learning" (DQN paper) — the landmark paper showing neural FA works with stabilization tricks


---

## Cross-References

<a class="link-card" href="./01_why_function_approximation_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Interactive slide deck covering the key concepts with visual examples.</div>
</a>

<a class="link-card" href="../notebooks/01_tile_coding.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises and real data.</div>
</a>
