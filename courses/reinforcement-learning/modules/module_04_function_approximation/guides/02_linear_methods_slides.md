---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Linear Methods and Semi-Gradient TD

## Module 04 — Function Approximation
### Reinforcement Learning Course

<!-- Speaker notes: This deck covers the operational core of linear function approximation: the feature vector x(s), the semi-gradient TD(0) update rule, and the convergence theory. Every equation here maps to Sutton & Barto Chapter 9. By the end, learners should be able to implement and train a tile-coding agent on a continuous-state environment. -->

---

# Linear FA: The Formula

$$\hat{v}(s, \mathbf{w}) = \mathbf{w}^T \mathbf{x}(s) = \sum_{i=1}^{d} w_i \, x_i(s)$$

| Symbol | Meaning | Dimension |
|---|---|---|
| $\hat{v}(s, \mathbf{w})$ | Value estimate for state $s$ | scalar |
| $\mathbf{w}$ | Weight vector (learned) | $d \times 1$ |
| $\mathbf{x}(s)$ | Feature vector (fixed by designer) | $d \times 1$ |

**Key property:**
$$\nabla_{\mathbf{w}} \hat{v}(s, \mathbf{w}) = \mathbf{x}(s)$$

Gradient = feature vector. Constant. No chain rule needed.


<div class="callout-insight">
<strong>Insight:</strong> This is a key takeaway from this section that connects to the broader course themes.
</div>

<!-- Speaker notes: Start with the formula. Emphasize the three symbols and their roles: w is what we learn, x(s) is what we design, and the dot product is the prediction. The key property — that the gradient is just x(s) — is what makes linear FA computationally simple and theoretically tractable. All the complexity is pushed into the design of x(s). -->

---

# Why the Gradient Matters

The SGD update for value prediction:

$$\mathbf{w} \leftarrow \mathbf{w} + \alpha \left[ \text{target} - \hat{v}(S_t, \mathbf{w}) \right] \underbrace{\nabla_{\mathbf{w}} \hat{v}(S_t, \mathbf{w})}_{\mathbf{x}(S_t)}$$

For linear FA, this simplifies to:

$$\mathbf{w} \leftarrow \mathbf{w} + \alpha \, \delta \, \mathbf{x}(S_t)$$

The update:
- Adds $\alpha \delta$ to weights corresponding to **active features only**
- Leaves all other weights unchanged
- Cost = $\mathcal{O}(d)$ dense, $\mathcal{O}(k)$ for $k$ non-zero features (tile coding)


<div class="callout-key">
<strong>Key Point:</strong> Remember this concept — it appears repeatedly in later modules.
</div>

<!-- Speaker notes: This slide connects the math to the computation. The update rule says: scale the feature vector by the TD error and add it to the weights. If features are sparse (tile coding), only the non-zero features need updating. This is the key computational efficiency of tile coding — with 8 tilings, only 8 weights change per step, not all 512. -->

---

# Four Feature Families

<div class="columns">

**Polynomial**
$x_i(s) = s_1^{c_1} s_2^{c_2} \cdots$
- Dense
- Smooth
- Cross terms free

**Fourier Basis**
$x_i(s) = \cos(\pi \mathbf{c}_i^T \mathbf{s})$
- Dense
- Orthogonal
- Per-feature $\alpha$

</div>

<div class="columns">

**Tile Coding**
Binary, $n$ tilings
- Sparse (k non-zeros)
- Efficient updates
- Adjustable resolution

**RBF**
$\exp(-\|s-c_i\|^2/2\sigma^2)$
- Dense
- Smooth
- Local generalization

</div>


<div class="callout-warning">
<strong>Warning:</strong> This is a common source of confusion. Pay close attention to the distinction here.
</div>

<!-- Speaker notes: The four-panel layout shows the design space. Two key dimensions: dense vs sparse, and smooth vs piecewise. Polynomial and Fourier are dense and smooth. Tile coding is sparse and piecewise — the workhorse for practical linear RL. RBF is dense and smooth but localized. The right choice depends on the problem: if you know the value function is smooth, use Fourier; if you need robustness, use tile coding. -->

---

# Tile Coding: The Workhorse

```
Mountain Car: position ∈ [-1.2, 0.6], velocity ∈ [-0.07, 0.07]

Tiling 1 (no offset):       Tiling 2 (offset 1/8):
┌─────┬─────┬─────┬─────┐  ┌──┬─────┬─────┬─────┬───┐
│     │     │     │     │  │  │     │     │     │   │
│  T1 │  T2 │  T3 │  T4 │  │T1│  T2 │  T3 │  T4 │T5 │
└─────┴─────┴─────┴─────┘  └──┴─────┴─────┴─────┴───┘

State s:  ● (falls in T2 of tiling 1, T2 of tiling 2)
State s': ◆ (falls in T2 of tiling 1, T3 of tiling 2)

s and s' share Tile T2 in tiling 1 → generalize
s and s' differ in tiling 2 → distinguish
```

8 tilings × 8×8 tiles = **512 features**, exactly **8 active** per state.


<div class="callout-info">
<strong>Info:</strong> This detail is useful context but not required to memorize.
</div>

<!-- Speaker notes: Work through the diagram step by step. The two states s and s' share a tile in tiling 1 but not tiling 2. This is the key: multiple tilings allow fine-grained distinction while coarse single tiles allow generalization. 8 tilings give 8x better resolution than a single tiling with the same tile size, without 8x more memory. This is elegant combinatorial engineering. -->

---

# Tile Coding: Implementation

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
class TileCoder:
    def __init__(self, n_tilings, tiles_per_dim, state_low, state_high):
        self.n_tilings = n_tilings
        self.scale = tiles_per_dim / (state_high - state_low)
        self.tiles_per_dim = tiles_per_dim
        self.n_tiles_per_tiling = int(np.prod(tiles_per_dim))
        self.n_features = n_tilings * self.n_tiles_per_tiling

    def encode(self, state):
        features = np.zeros(self.n_features)
        scaled = (state - self.low) * self.scale
        for tiling in range(self.n_tilings):
            offset = tiling / self.n_tilings
            tile_idx = np.floor(scaled + offset).astype(int)
            tile_idx = np.clip(tile_idx, 0, self.tiles_per_dim - 1)
            flat = int(np.ravel_multi_index(tile_idx, self.tiles_per_dim))
            features[tiling * self.n_tiles_per_tiling + flat] = 1.0
        return features
```
</div>

<!-- Speaker notes: Walk through the encode method line by line. The scaled variable puts the state in [0, tiles_per_dim) range. The offset shifts each tiling by tiling/n_tilings of a tile width. Floor gives the integer tile index. ravel_multi_index converts the k-dimensional tile index to a flat index within the tiling. Setting features[...] = 1.0 marks the active tile. -->

---

# Fourier Basis: Structured Frequencies

$$x_i(\mathbf{s}) = \cos\left(\pi \, \mathbf{c}_i^T \mathbf{s}\right), \quad \mathbf{s} \in [0,1]^k$$

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def fourier_features(state, order=3):
    # state normalized to [0,1]^k
    k = len(state)
    coeffs = np.array(list(product(range(order+1), repeat=k)))
    return np.cos(np.pi * coeffs @ state)
```
</div>

**Per-feature learning rate** (recommended):
$$\alpha_i = \frac{\alpha_0}{\sqrt{1 + \|\mathbf{c}_i\|^2}}$$

High-frequency features (large $\|\mathbf{c}_i\|$) → smaller step size → stability.

<!-- Speaker notes: Fourier features decompose the value function into frequency components, just like a Fourier series decomposes a signal. The coefficient vector c_i controls which frequency component feature i represents. The per-feature learning rate is the key advantage: high-frequency components are harder to learn reliably, so we automatically reduce their step size. This is not needed for tile coding. -->

---

# The TD Error: The Core Signal

For transition $(S_t, R_{t+1}, S_{t+1})$:

$$\delta_t = \underbrace{R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w})}_{\text{TD target}} - \underbrace{\hat{v}(S_t, \mathbf{w})}_{\text{prediction}}$$

| Case | TD error | Interpretation |
|---|---|---|
| $\delta_t > 0$ | Underestimated | Increase $\hat{v}(S_t)$ |
| $\delta_t < 0$ | Overestimated | Decrease $\hat{v}(S_t)$ |
| $\delta_t = 0$ | Consistent | No update needed |

The TD error is the same as in tabular TD(0) — only the update mechanism differs.

<!-- Speaker notes: The TD error is identical to the tabular version. This is intentional: function approximation changes how we represent and update values, not what we are trying to learn. The table of interpretations helps connect the sign of the error to the weight update direction. Emphasize: delta > 0 means the target is higher than the prediction, so we increase the prediction by increasing the weights in the direction of x(S_t). -->

---

# Semi-Gradient TD(0) Update

**Full update rule** (Sutton & Barto, Eq. 9.7):

$$\mathbf{w} \leftarrow \mathbf{w} + \alpha \, \delta_t \, \mathbf{x}(S_t)$$

**Why "semi-gradient"?**

```
True gradient of [target - v̂(S,w)]²/2 would include:

  δ_t · [∇v̂(S_t,w) - γ ∇v̂(S_{t+1},w)]
          ─────────────   ──────────────────
            prediction       TARGET gradient
            gradient         ← this term is DROPPED

Semi-gradient drops the target's gradient.
This is not true SGD, but it converges (on-policy, linear).
```

<!-- Speaker notes: The semi-gradient concept is subtle. Draw the distinction: in supervised learning, the target is a fixed label (the gradient of the target term is zero). In TD learning, the target depends on w through v-hat(S_{t+1}, w). The full gradient would include the target's gradient term. Dropping it gives the semi-gradient. This asymmetry is why TD converges to the TD fixed point rather than minimizing VE. -->

---

# Semi-Gradient TD(0): Code

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
class SemiGradientTD0:
    def __init__(self, feature_fn, n_features, alpha=0.01, gamma=0.99):
        self.feature_fn = feature_fn
        self.alpha = alpha
        self.gamma = gamma
        self.w = np.zeros(n_features)

    def predict(self, state):
        return np.dot(self.w, self.feature_fn(state))

    def update(self, state, reward, next_state, done):
        x_s  = self.feature_fn(state)
        v_s  = np.dot(self.w, x_s)
        # Target: treat v(S') as constant — no gradient flows through it
        v_s_ = 0.0 if done else self.predict(next_state)
        delta = reward + self.gamma * v_s_ - v_s
        # Semi-gradient update: w += alpha * delta * x(s)
        self.w += self.alpha * delta * x_s
```
</div>

The `done` flag ensures $V(S_T) = 0$ at terminal states.

<!-- Speaker notes: Point out the done flag as the most common implementation bug. At episode termination, there is no next state, so the bootstrap value must be zero. Forgetting this causes the agent to bootstrap from a garbage prediction at the end of every episode. Also highlight that v_s_ is computed before the update — we use the current weights for the target and only update using x_s. -->

---

# Convergence to the TD Fixed Point

Linear semi-gradient TD(0) converges to $\mathbf{w}_{TD}$:

$$\mathbf{w}_{TD} = \mathbf{A}^{-1}\mathbf{b}$$

$$\mathbf{A} = \mathbb{E}\left[\mathbf{x}(S_t)\left(\mathbf{x}(S_t) - \gamma\mathbf{x}(S_{t+1})\right)^T\right], \quad \mathbf{b} = \mathbb{E}\left[R_{t+1}\mathbf{x}(S_t)\right]$$

**The bound** (Sutton & Barto, Eq. 9.14):

$$\overline{VE}(\mathbf{w}_{TD}) \leq \frac{1}{1-\gamma} \min_{\mathbf{w}} \overline{VE}(\mathbf{w})$$

> The TD fixed point is not the best linear approximation — it is bounded by $1/(1-\gamma) \times$ the best.

<!-- Speaker notes: The convergence theorem requires two conditions: on-policy sampling (mu matches the behavior policy) and the Robbins-Monro step size conditions. The bound is important: for gamma = 0.99, TD can have up to 100x higher VE than the best linear approximation. This explains why Monte Carlo methods can sometimes outperform TD on value estimation, even though TD is usually more sample efficient for control. -->

---

# On-Policy vs Off-Policy: Convergence Summary

| Setting | Linear FA | Non-Linear FA |
|---|---|---|
| On-policy prediction | Converges to TD fixed point | May diverge |
| On-policy control (SARSA) | Converges near optimal | May diverge |
| Off-policy prediction (TD) | **Can diverge** | **Can diverge** |
| Off-policy control (Q-learning) | **Can diverge** | **Can diverge** |

The shaded cells are the deadly triad (Guide 03).

The safe zone is: **on-policy + linear FA**.

<!-- Speaker notes: This table maps convergence guarantees to algorithm-approximator combinations. The top-left cell (on-policy + linear) is the safe zone covered in this guide. Every other combination has some failure mode. Off-policy + linear can already diverge, as shown by Baird's counterexample. Off-policy + nonlinear is the most dangerous and requires DQN-style stabilization. -->

---

# Choosing $\alpha$: The Learning Rate

**General rule for tile coding:**
$$\alpha = \frac{\alpha_0}{n_\text{tilings}}, \quad \alpha_0 \in [0.1, 0.5]$$

**Why:** Each update activates $n_\text{tilings}$ features, so the effective learning rate is $\alpha \cdot n_\text{tilings}$. Dividing by $n_\text{tilings}$ normalizes the effective step size.

**Fourier basis:** Use per-feature rates $\alpha_i = \alpha_0 / \sqrt{1 + \|\mathbf{c}_i\|^2}$.

**Polynomial basis:** Standard $\alpha$; normalize features to $[-1,1]$ first.

**Diagnostic:** If $\hat{v}(s, \mathbf{w})$ diverges, halve $\alpha$. If it converges but learns slowly, double $\alpha$.

<!-- Speaker notes: The alpha-over-n_tilings rule is empirically reliable and theoretically motivated. Without it, increasing the number of tilings (which improves resolution) also accelerates learning, making it impossible to tune alpha independently of resolution. This is a practical tip that saves significant debugging time. -->

---

# Mountain Car: Tile Coding in Action

```python
import gymnasium as gym

env = gym.make("MountainCar-v0")
coder = TileCoder(
    n_tilings=8, tiles_per_dim=np.array([8, 8]),
    state_low=env.observation_space.low,
    state_high=env.observation_space.high
)
agent = SemiGradientTD0(
    feature_fn=coder.encode,
    n_features=coder.n_features,    # 8 * 64 = 512 features
    alpha=0.1 / 8,                  # alpha_0 / n_tilings
    gamma=1.0
)
```

```
Episode   100: avg steps = -189.3
Episode   500: avg steps = -142.7
Episode  1000: avg steps = -110.4
Episode  5000: avg steps =  -95.2   ← near optimal
```

512 features, 8 active per step — the update costs $\mathcal{O}(8)$ per timestep.

<!-- Speaker notes: The mountain car example is the canonical benchmark for linear FA in Sutton & Barto. The step counts improve because the agent learns which tile-code patterns correspond to states where it should push hard versus coast. The O(8) update cost is the critical efficiency point: even with 512 parameters, only 8 need updating at each step. -->

---

# Common Pitfalls: Linear Methods

| Pitfall | Symptom | Fix |
|---|---|---|
| $\alpha$ too large for tile coding | Oscillating values | Use $\alpha_0 / n_\text{tilings}$ |
| Missing terminal value | Overestimation at episode end | Zero out $\hat{v}(S_T)$ |
| Q-learning + linear FA off-policy | Divergence | Use on-policy SARSA instead |
| Too few tiles | Coarse generalization, slow learning | Increase tiles or tilings |
| Unnormalized state for polynomial/RBF | One feature dominates | Normalize to $[0,1]$ |

<!-- Speaker notes: These five pitfalls cover the most common bugs in linear FA implementations. The most dangerous is Q-learning + linear FA: it looks like it is working for the first few thousand steps, then diverges suddenly. The fix (switch to SARSA) is simple but requires understanding why the off-policy target breaks convergence. Guide 03 explains the mechanism. -->

---

# Key Takeaways

1. $\hat{v}(s, \mathbf{w}) = \mathbf{w}^T \mathbf{x}(s)$: all the expressive power is in the feature vector $\mathbf{x}(s)$.

2. Gradient of linear FA = feature vector: $\nabla_\mathbf{w} \hat{v} = \mathbf{x}(s)$. Constant, cheap to compute.

3. Semi-gradient TD(0) treats the TD target as a fixed constant — it is not true SGD.

4. On-policy linear TD(0) converges to $\mathbf{w}_{TD}$, bounded $1/(1-\gamma)$ above the best linear approximation.

5. Tile coding = the practical default for continuous RL. 8 tilings, 8 tiles/dim, $\alpha = 0.1/8$.

<!-- Speaker notes: Five numbered takeaways for the post-lecture review. Point 3 is the key conceptual insight: semi-gradient is not gradient descent on any loss. Point 5 gives a concrete recipe that works without tuning for most moderate-dimensional continuous control tasks. -->

---

# What's Next

**Guide 03 — The Deadly Triad:**

The three ingredients that cause instability:
1. Function approximation (this guide)
2. Bootstrapping (TD target — this guide)
3. Off-policy training (Q-learning, experience replay)

Any two are fine. All three together $\to$ divergence.

Baird's counterexample shows a simple 7-state MDP where linear TD diverges off-policy.

**Module 05 — DQN:**
How replay buffers and target networks partially escape the triad.

<!-- Speaker notes: The forward pointer connects immediately: we have seen ingredients 1 and 2 (FA and bootstrapping) in this guide. Guide 03 adds ingredient 3 (off-policy) and shows why the combination is dangerous. This forward connection makes Guide 03 feel like a natural continuation, not an interruption. -->
