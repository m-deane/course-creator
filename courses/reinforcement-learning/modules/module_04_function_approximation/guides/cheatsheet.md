# Module 04 Cheatsheet — Function Approximation

> **Reading time:** ~8 min | **Module:** 4 — Function Approximation | **Prerequisites:** Module 3

## Core Notation

| Symbol | Meaning |
|---|---|
| $s \in \mathcal{S}$ | State (possibly continuous) |
| $\mathbf{w} \in \mathbb{R}^d$ | Weight vector (learned parameters) |
| $\mathbf{x}(s) \in \mathbb{R}^d$ | Feature vector for state $s$ (fixed by designer) |
| $\hat{v}(s, \mathbf{w})$ | Approximate state value |
| $\hat{q}(s, a, \mathbf{w})$ | Approximate action value |
| $V^\pi(s)$ | True state value under policy $\pi$ |
| $\delta_t$ | TD error at timestep $t$ |
| $\mu(s)$ | On-policy state distribution (visit frequency) |
| $\overline{VE}(\mathbf{w})$ | Mean squared value error |

---

## Linear Function Approximation

$$\hat{v}(s, \mathbf{w}) = \mathbf{w}^T \mathbf{x}(s) = \sum_{i=1}^{d} w_i \, x_i(s)$$

<div class="callout-insight">

<strong>Insight:</strong> $$\hat{v}(s, \mathbf{w}) = \mathbf{w}^T \mathbf{x}(s) = \sum_{i=1}^{d} w_i \, x_i(s)$$

**Gradient** (constant — does not depend on $\mathbf{w}$):

$$\nabla_{\mathbf{w}} \hat{v}(s, \mathbf{w}) = \math...

</div>


**Gradient** (constant — does not depend on $\mathbf{w}$):

$$\nabla_{\mathbf{w}} \hat{v}(s, \mathbf{w}) = \mathbf{x}(s)$$

**Value error objective:**

$$\overline{VE}(\mathbf{w}) = \sum_{s \in \mathcal{S}} \mu(s) \left[ V^\pi(s) - \mathbf{w}^T \mathbf{x}(s) \right]^2$$

---

## Semi-Gradient TD(0) Update

For transition $(S_t, R_{t+1}, S_{t+1})$:

<div class="callout-key">

<strong>Key Point:</strong> For transition $(S_t, R_{t+1}, S_{t+1})$:

**Step 1 — TD error:**

$$\delta_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}) - \hat{v}(S_t, \mathbf{w})$$

(Use $\hat{v}(S_{t+1}, \mathbf{w}) = 0$ if $...

</div>


**Step 1 — TD error:**

$$\delta_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}) - \hat{v}(S_t, \mathbf{w})$$

(Use $\hat{v}(S_{t+1}, \mathbf{w}) = 0$ if $S_{t+1}$ is terminal.)

**Step 2 — Weight update:**

$$\mathbf{w} \leftarrow \mathbf{w} + \alpha \, \delta_t \, \nabla_{\mathbf{w}} \hat{v}(S_t, \mathbf{w})$$

**For linear FA** (substitute gradient = feature vector):

$$\boxed{\mathbf{w} \leftarrow \mathbf{w} + \alpha \, \delta_t \, \mathbf{x}(S_t)}$$

**"Semi-gradient"** = treat TD target as a fixed constant; do not differentiate through $\hat{v}(S_{t+1}, \mathbf{w})$.

**Convergence:** Guaranteed on-policy + linear FA. Not guaranteed off-policy or with nonlinear FA.

---

## Tile Coding Setup Steps

1. **Choose parameters:**
   - $n$ = number of tilings (typically 4, 8, or 16)
   - $m$ = tiles per dimension (typically 4, 8, 16)
   - State bounds: $[s_{\text{low}}, s_{\text{high}}]$ per dimension

<div class="callout-info">

<strong>Info:</strong> **Choose parameters:**
   - $n$ = number of tilings (typically 4, 8, or 16)
   - $m$ = tiles per dimension (typically 4, 8, 16)
   - State bounds: $[s_{\text{low}}, s_{\text{high}}]$ per dimension

2.

</div>


2. **Compute scale factor** (converts state range to tile indices):
   $$\text{scale} = m \,/\, (s_{\text{high}} - s_{\text{low}})$$

3. **For each tiling** $i = 0, 1, \ldots, n-1$:
   - Offset = $i \,/\, n$ (fraction of a tile width)
   - Scaled state = $(s - s_{\text{low}}) \times \text{scale}$
   - Tile index = $\lfloor \text{scaled state} + \text{offset} \rfloor$ (integer, per dimension)
   - Clip to $[0, m-1]$
   - Flatten to 1D index; set feature = 1

4. **Feature vector size:** $n \times m^k$ (total), exactly $n$ non-zero entries.

5. **Learning rate:** $\alpha = \alpha_0 / n$, with $\alpha_0 \in [0.1, 0.5]$.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
class TileCoder:
    def __init__(self, n_tilings, tiles_per_dim, state_low, state_high):
        self.n_tilings = n_tilings
        self.tiles_per_dim = np.asarray(tiles_per_dim)
        self.scale = self.tiles_per_dim / (np.asarray(state_high) - np.asarray(state_low))
        self.low = np.asarray(state_low)
        self.n_tiles_per_tiling = int(np.prod(self.tiles_per_dim))
        self.n_features = n_tilings * self.n_tiles_per_tiling

    def encode(self, state):
        features = np.zeros(self.n_features)
        scaled = (np.asarray(state) - self.low) * self.scale
        for i in range(self.n_tilings):
            idx = np.floor(scaled + i / self.n_tilings).astype(int)
            idx = np.clip(idx, 0, self.tiles_per_dim - 1)
            flat = int(np.ravel_multi_index(idx, self.tiles_per_dim))
            features[i * self.n_tiles_per_tiling + flat] = 1.0
        return features
```

</div>
</div>

---

## Deadly Triad Conditions

| Condition | Description | Example |
|---|---|---|
| **Function Approximation** | $|\mathbf{w}| \ll |\mathcal{S}|$; weights shared across states | Tile coding, neural network |
| **Bootstrapping** | Target depends on current $\mathbf{w}$: $R + \gamma \hat{v}(S', \mathbf{w})$ | TD(0), Q-learning, DQN |
| **Off-Policy Training** | Behavior policy $b \neq$ target policy $\pi$ | Q-learning, experience replay |

<div class="callout-warning">

<strong>Warning:</strong> All three → potential divergence.

</div>


**The rule:** Any two → safe. All three → potential divergence.

### Pairwise Safety

| Combination | Safe? | Why |
|---|---|---|
| FA + bootstrapping + on-policy | Yes | $\mathbf{A}$ is positive definite (on-policy guarantees this) |
| FA + Monte Carlo + off-policy | Yes | No bootstrap feedback; IS corrects distribution |
| Tabular + bootstrapping + off-policy | Yes | No shared weights; states update independently |
| **FA + bootstrapping + off-policy** | **No** | All three interact; $\mathbf{A}$ may not be positive definite |

### Baird's Counterexample Summary

- 7-state MDP, linear FA, all rewards = 0
- True optimal: $\mathbf{w} = \mathbf{0}$
- Semi-gradient TD off-policy: weights diverge to $\infty$
- Proof that the triad causes divergence even in the simplest case

---

## Feature Types Comparison

| Feature Type | Formula | Dense/Sparse | Smooth | Scales to High $d$? | Best Use Case |
|---|---|---|---|---|---|
| **Polynomial** | $\prod_j s_j^{c_{ij}}$ | Dense | Yes | No ($d!$ features) | Low-dim, smooth $V^\pi$ |
| **Fourier basis** | $\cos(\pi \mathbf{c}^T \mathbf{s})$ | Dense | Yes | No ($(n+1)^k$ features) | Bounded state, periodic structure |
| **Tile coding** | Binary tiles, $n$ tilings | Sparse ($n$ active) | No (piecewise) | Moderately ($n \cdot m^k$) | Default for 1D–4D continuous state |
| **RBF** | $\exp(-\|s-c_i\|^2/2\sigma^2)$ | Dense | Yes | No (one center per feature) | Smooth, locally-structured $V^\pi$ |
| **Neural network** | Learned $f(s; \mathbf{w})$ | Dense | Yes (smooth) | Yes | High-dim, raw input (pixels) |

---

## When to Use Tabular vs Linear FA vs Neural Network FA

### Use Tabular RL When

<div class="callout-insight">

<strong>Insight:</strong> ### Use Tabular RL When

- State space is small and discrete (less than ~$10^6$ states)
- You need convergence guarantees
- Debugging a conceptual algorithm (tabular = transparent)
- Environment: Grid...



- State space is small and discrete (less than ~$10^6$ states)
- You need convergence guarantees
- Debugging a conceptual algorithm (tabular = transparent)
- Environment: GridWorld, simple card games, small MDPs

### Use Linear FA When

- State space is continuous but low-dimensional (1D–6D)
- You can design meaningful features (tile coding is the default)
- You need convergence guarantees (use on-policy SARSA, not Q-learning)
- Training data is limited (linear is more sample-efficient than neural)
- Interpretability matters (weight magnitudes indicate feature importance)

### Use Neural Network FA When

- State space is high-dimensional or raw (images, sensor arrays)
- Feature engineering is infeasible or domain knowledge is limited
- You have access to large amounts of training data
- You are willing to use DQN stabilization tricks (replay + target network)
- Convergence is not guaranteed but empirical performance is acceptable

### Decision Tree

```
Is the state space small and discrete?
  YES → Tabular Q-learning or TD
  NO →
    Can you design good features? AND is the state low-dimensional?
      YES → Linear FA (tile coding default, SARSA for convergence)
      NO →
        Is on-policy training acceptable?
          YES → Neural FA + policy gradient (PPO, A2C)
          NO  → Neural FA + DQN (replay + target network, no guarantee)
```

---

## Quick Reference: Algorithms and Their Triad Status

| Algorithm | FA | Bootstrap | Off-Policy | Converges? |
|---|---|---|---|---|
| Tabular TD(0) | No | Yes | No | Yes |
| Tabular Q-learning | No | Yes | Yes | Yes |
| Linear semi-gradient TD(0) | Linear | Yes | No (on-policy) | Yes |
| Linear SARSA | Linear | Yes | No (on-policy) | Near-optimal |
| Linear Q-learning | Linear | Yes | Yes | No guarantee |
| DQN | Neural | Yes | Yes (replay) | No guarantee |
| REINFORCE | Neural | No (MC) | No | Yes |
| PPO | Neural | Yes | Near on-policy | Near-optimal |

---

## Common Bugs Checklist

- [ ] Zeroed bootstrap value at terminal state: `v_next = 0.0 if done else predict(next_state)`
- [ ] Learning rate scaled by number of tilings: `alpha = alpha_0 / n_tilings`
- [ ] State normalized before computing features (polynomial, Fourier, RBF)
- [ ] Feature vector initialized fresh for each state (not accumulated across steps)
- [ ] Off-policy control uses SARSA (not Q-learning) if convergence is required
- [ ] Target network updated on a fixed schedule (not every step)
- [ ] Replay buffer large enough that recent episodes are a small fraction (capacity $\geq$ 10,000)

---

## Notation Reference (Module 04 vs Sutton & Barto Chapter 9)

| This module | S&B notation | Meaning |
|---|---|---|
| $\hat{v}(s, \mathbf{w})$ | $\hat{v}(s, \mathbf{w})$ | Approximate value |
| $\mathbf{x}(s)$ | $\mathbf{x}(s)$ | Feature vector |
| $\delta_t$ | $\delta_t$ | TD error |
| $\overline{VE}(\mathbf{w})$ | $\overline{VE}(\mathbf{w})$ | Mean squared value error |
| $\mathbf{w}_{TD}$ | $\mathbf{w}_{TD}$ | TD fixed point |
| $d^\pi(s)$ | $\mu(s)$ | On-policy state distribution |

All notation is consistent with S&B 2nd edition Chapters 9–11.
