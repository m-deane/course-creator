# Linear Methods and Semi-Gradient TD

> **Reading time:** ~19 min | **Module:** 4 — Function Approximation | **Prerequisites:** Module 3

## In Brief

Linear function approximation represents $\hat{v}(s, \mathbf{w}) = \mathbf{w}^T \mathbf{x}(s)$ — a dot product between a weight vector and a hand-crafted feature vector. Combined with semi-gradient TD(0), this produces an efficient, convergent on-policy learning algorithm with well-understood theoretical properties.

<div class="callout-key">

<strong>Key Concept:</strong> Linear function approximation represents $\hat{v}(s, \mathbf{w}) = \mathbf{w}^T \mathbf{x}(s)$ — a dot product between a weight vector and a hand-crafted feature vector. Combined with semi-gradient TD(0), this produces an efficient, convergent on-policy learning algorithm with well-understood theoretical properties.

</div>


## Key Insight

"Semi-gradient" means we treat the TD target as a fixed label and differentiate only through the prediction, not the target. This asymmetry is what makes the algorithm practical — and what breaks the clean convergence story of supervised learning.

---


<div class="callout-key">

<strong>Key Point:</strong> "Semi-gradient" means we treat the TD target as a fixed label and differentiate only through the prediction, not the target.

</div>

## 1. Linear Function Approximation

### Formal Definition

<div class="callout-key">

<strong>Key Point:</strong> ### Formal Definition

$$\hat{v}(s, \mathbf{w}) = \mathbf{w}^T \mathbf{x}(s) = \sum_{i=1}^{d} w_i \, x_i(s)$$

where:
- $\mathbf{w} \in \mathbb{R}^d$ — the weight vector (the parameters we learn)
- $\...

</div>


$$\hat{v}(s, \mathbf{w}) = \mathbf{w}^T \mathbf{x}(s) = \sum_{i=1}^{d} w_i \, x_i(s)$$

where:
- $\mathbf{w} \in \mathbb{R}^d$ — the weight vector (the parameters we learn)
- $\mathbf{x}(s) \in \mathbb{R}^d$ — the feature vector for state $s$ (fixed, chosen by designer)
- $d$ — the number of features, typically $d \ll |\mathcal{S}|$

### Gradient

Since $\hat{v}(s, \mathbf{w})$ is linear in $\mathbf{w}$, its gradient with respect to $\mathbf{w}$ is simply the feature vector:

$$\nabla_{\mathbf{w}} \hat{v}(s, \mathbf{w}) = \mathbf{x}(s)$$

This is the key property of linear FA. The gradient does not depend on $\mathbf{w}$ or the current prediction — it is just the feature vector evaluated at $s$. This makes the update computationally cheap and enables convergence analysis.

### Linear FA for Action Values

For control, approximate $Q^\pi(s, a)$:

$$\hat{q}(s, a, \mathbf{w}) = \mathbf{w}^T \mathbf{x}(s, a)$$

Here the feature vector $\mathbf{x}(s, a)$ encodes both state and action. For discrete actions with $|\mathcal{A}|$ choices, a common approach uses separate weights per action:

$$\hat{q}(s, a, \mathbf{w}) = \mathbf{w}_a^T \mathbf{x}(s), \quad \mathbf{w} = [\mathbf{w}_1, \ldots, \mathbf{w}_{|\mathcal{A}|}]$$

---

## 2. Feature Vectors in Depth

### 2.1 Polynomial Basis

For a $k$-dimensional state $\mathbf{s} = (s_1, \ldots, s_k)$, the order-$n$ polynomial features are all monomials up to degree $n$:

$$x_i(\mathbf{s}) = \prod_{j=1}^{k} s_j^{c_{ij}}, \quad \sum_{j=1}^{k} c_{ij} \leq n, \quad c_{ij} \in \{0, 1, \ldots, n\}$$

For $k=2$ dimensions and $n=2$:

$$\mathbf{x}(\mathbf{s}) = [1, s_1, s_2, s_1^2, s_1 s_2, s_2^2]^T$$

Total features: $\binom{k+n}{n}$. For $k=4$ (CartPole) and $n=3$: $\binom{7}{3} = 35$ features.


<span class="filename">example.py</span>
</div>
The following implementation builds on the approach above:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np
from itertools import product as iproduct

def polynomial_features(state, degree=2):
    """
    Polynomial feature expansion.

    Parameters
    ----------
    state   : array-like of shape (k,)
    degree  : maximum polynomial degree

    Returns
    -------
    features : ndarray of shape (n_features,)
    """
    state = np.asarray(state, dtype=float)
    k = len(state)
    features = []
    # All exponent combinations with sum <= degree
    for exponents in iproduct(range(degree + 1), repeat=k):
        if sum(exponents) <= degree:
            features.append(np.prod(state ** np.array(exponents)))
    return np.array(features)

# Example
s = np.array([0.5, -0.3])
x = polynomial_features(s, degree=2)
print(f"Features: {x}")  # [1, 0.5, -0.3, 0.25, -0.15, 0.09]
```

</div>
</div>

### 2.2 Fourier Basis

For a state $\mathbf{s}$ normalized to $[0, 1]^k$, the order-$n$ Fourier features are:

$$x_i(\mathbf{s}) = \cos(\pi \, \mathbf{c}_i^T \mathbf{s}), \quad \mathbf{c}_i \in \{0, 1, \ldots, n\}^k$$

The coefficient vectors $\mathbf{c}_i$ capture different frequency components. The total number of features is $(n+1)^k$.

**Key advantage:** Fourier features have a natural per-feature learning rate:

$$\alpha_i = \alpha \Big/ \sqrt{1 + \|\mathbf{c}_i\|^2}$$

Higher-frequency features get smaller learning rates, which improves stability.


<span class="filename">example.py</span>
</div>
The following implementation builds on the approach above:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def fourier_features(state, order=3, state_low=None, state_high=None):
    """
    Fourier basis features for a state in [0,1]^k after normalization.

    Parameters
    ----------
    state      : array-like of shape (k,)
    order      : highest Fourier coefficient
    state_low  : lower bounds for normalization (array of shape (k,))
    state_high : upper bounds for normalization (array of shape (k,))

    Returns
    -------
    features : ndarray of shape ((order+1)^k,)
    """
    state = np.asarray(state, dtype=float)
    if state_low is not None:
        state = (state - state_low) / (state_high - state_low)
    k = len(state)
    coeffs = np.array(list(iproduct(range(order + 1), repeat=k)))
    # cos(pi * c^T * s) for each coefficient vector c
    return np.cos(np.pi * coeffs @ state)

# Per-feature learning rates for Fourier basis
def fourier_alphas(coeffs, base_alpha):
    """Scale learning rates by coefficient norm."""
    norms = np.linalg.norm(coeffs, axis=1)
    return np.where(norms == 0, base_alpha, base_alpha / norms)
```

</div>
</div>

### 2.3 Tile Coding (Deep Dive)

Tile coding is the dominant linear FA technique for continuous RL. The idea: overlay $n$ grids (tilings) on the state space, each offset by a fraction of the tile width. Each tiling activates exactly one tile per state. The feature vector concatenates the active tiles across all tilings — sparse binary with exactly $n$ ones in $n \times m^k$ dimensions.

#### Construction

For a $k$-dimensional state with $m$ tiles per dimension and $n$ tilings:

1. Total tiles per tiling: $m^k$
2. Total features: $n \times m^k$
3. Non-zero features per state: exactly $n$ (one per tiling)
4. Offset for tiling $i$: shift by $(i/n) \times (\text{tile width})$ in each dimension


<span class="filename">example.py</span>
</div>
The following implementation builds on the approach above:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
class TileCoder:
    """
    Tile coding for continuous state spaces.

    Follows the tilecoding.py interface from Sutton's website,
    implemented from scratch for clarity.
    """

    def __init__(self, n_tilings, tiles_per_dim, state_low, state_high):
        """
        Parameters
        ----------
        n_tilings     : int, number of overlapping grids
        tiles_per_dim : int or list, tiles along each dimension
        state_low     : array, lower bound of each state dimension
        state_high    : array, upper bound of each state dimension
        """
        self.n_tilings = n_tilings
        state_low  = np.asarray(state_low,  dtype=float)
        state_high = np.asarray(state_high, dtype=float)
        self.k = len(state_low)

        if isinstance(tiles_per_dim, int):
            tiles_per_dim = [tiles_per_dim] * self.k
        self.tiles_per_dim = np.array(tiles_per_dim)

        # Scale: map state range to [0, tiles_per_dim) per dimension
        self.scale = self.tiles_per_dim / (state_high - state_low)
        self.low   = state_low

        # Total features = n_tilings * product(tiles_per_dim)
        self.n_tiles_per_tiling = int(np.prod(self.tiles_per_dim))
        self.n_features = n_tilings * self.n_tiles_per_tiling

    def encode(self, state):
        """
        Return sparse binary feature vector for state.

        Returns
        -------
        features : ndarray of shape (n_features,), dtype float
        """
        state = np.asarray(state, dtype=float)
        scaled = (state - self.low) * self.scale  # in [0, tiles_per_dim)

        features = np.zeros(self.n_features)

        for tiling in range(self.n_tilings):
            # Shift state by (tiling / n_tilings) in each dimension
            offset = tiling / self.n_tilings
            # Integer tile index along each dimension
            tile_idx = np.floor(scaled + offset).astype(int)
            # Clip to valid range
            tile_idx = np.clip(tile_idx, 0, self.tiles_per_dim - 1)

            # Convert multi-dimensional tile index to flat index
            flat_idx = int(np.ravel_multi_index(tile_idx, self.tiles_per_dim))

            # Set the active tile for this tiling
            feature_idx = tiling * self.n_tiles_per_tiling + flat_idx
            features[feature_idx] = 1.0

        return features

    def predict(self, state, weights):
        """Linear prediction: w^T x(s)."""
        return np.dot(weights, self.encode(state))

# Example: Mountain Car
# State: position in [-1.2, 0.6], velocity in [-0.07, 0.07]
coder = TileCoder(
    n_tilings=8,
    tiles_per_dim=8,
    state_low=[-1.2, -0.07],
    state_high=[0.6, 0.07]
)

state = np.array([-0.5, 0.02])
x = coder.encode(state)
print(f"Feature vector: {x.shape} features, {int(x.sum())} active")
# Output: (512,) features, 8 active
```

</div>
</div>

#### Why Multiple Tilings?

A single tiling provides resolution equal to one tile width. With $n$ tilings each offset by $1/n$ of the tile width, the effective resolution increases $n$-fold while each tiling remains coarse. The offset ensures that states close together do not activate the same tile in all tilings — the overlap pattern differentiates them.

```
Single tiling (4 tiles):   Resolution = tile_width / 1
Double tiling (4 tiles each, offset by 0.5): Resolution = tile_width / 2
n tilings: Resolution = tile_width / n
```

### 2.4 Radial Basis Functions


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def rbf_features(state, centers, sigma=1.0):
    """
    Radial basis function features.

    Parameters
    ----------
    state   : array-like of shape (k,)
    centers : array-like of shape (n_centers, k)
    sigma   : bandwidth (scalar or array of shape (n_centers,))

    Returns
    -------
    features : ndarray of shape (n_centers,)
    """
    state   = np.asarray(state,   dtype=float)
    centers = np.asarray(centers, dtype=float)
    dists_sq = np.sum((centers - state) ** 2, axis=1)
    return np.exp(-dists_sq / (2 * sigma**2))
```

</div>
</div>

---

## 3. Semi-Gradient TD(0)

### Why "Semi-Gradient"?

The true SGD update for supervised regression uses the gradient of the loss:

$$\mathbf{w} \leftarrow \mathbf{w} + \alpha \left[ y - \hat{v}(S_t, \mathbf{w}) \right] \nabla_{\mathbf{w}} \hat{v}(S_t, \mathbf{w})$$

In RL, the "label" is the TD target $R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w})$. If we differentiate through the target (treating $\mathbf{w}$ inside the target as a variable), we get the full gradient. If we treat the target as a fixed constant (stop the gradient), we get the **semi-gradient**.

**Full gradient:**

$$\nabla_{\mathbf{w}} \left[ R + \gamma \hat{v}(S', \mathbf{w}) - \hat{v}(S, \mathbf{w}) \right]^2 / 2 = \left[ \delta_t \right] \left[ \nabla_{\mathbf{w}} \hat{v}(S, \mathbf{w}) - \gamma \nabla_{\mathbf{w}} \hat{v}(S', \mathbf{w}) \right]$$

**Semi-gradient (standard):**

$$\nabla_{\mathbf{w}} \left[ \text{target} - \hat{v}(S, \mathbf{w}) \right]^2 / 2 \approx \left[ \delta_t \right] \nabla_{\mathbf{w}} \hat{v}(S, \mathbf{w})$$

The target's gradient term $\gamma \nabla_{\mathbf{w}} \hat{v}(S', \mathbf{w})$ is dropped. This is not a true gradient descent step, but it has better convergence properties in practice and converges to the TD fixed point for linear approximators.

### The Semi-Gradient TD(0) Update

For a transition $(S_t, R_{t+1}, S_{t+1})$:

**TD error (scalar):**
$$\delta_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}) - \hat{v}(S_t, \mathbf{w})$$

**Weight update (vector):**
$$\mathbf{w} \leftarrow \mathbf{w} + \alpha \, \delta_t \, \nabla_{\mathbf{w}} \hat{v}(S_t, \mathbf{w})$$

For linear FA where $\nabla_{\mathbf{w}} \hat{v}(S_t, \mathbf{w}) = \mathbf{x}(S_t)$:

$$\boxed{\mathbf{w} \leftarrow \mathbf{w} + \alpha \, \delta_t \, \mathbf{x}(S_t)}$$

This is Sutton & Barto Equation 9.7.

### Implementation

```python
class SemiGradientTD0:
    """
    Semi-gradient TD(0) for on-policy prediction with linear FA.

    Learns V^pi(s) ≈ w^T x(s) under policy pi.
    """

    def __init__(self, feature_fn, n_features, alpha=0.01, gamma=0.99):
        """
        Parameters
        ----------
        feature_fn  : callable, s -> x(s), returns ndarray of shape (n_features,)
        n_features  : int, dimension of feature vector
        alpha       : float, learning rate
        gamma       : float, discount factor
        """
        self.feature_fn = feature_fn
        self.alpha = alpha
        self.gamma = gamma
        self.w = np.zeros(n_features)

    def predict(self, state):
        """Value estimate: w^T x(s)."""
        return np.dot(self.w, self.feature_fn(state))

    def update(self, state, reward, next_state, done):
        """
        One semi-gradient TD(0) update.

        Parameters
        ----------
        state      : current state
        reward     : R_{t+1}
        next_state : S_{t+1}
        done       : bool, True if terminal
        """
        x_s  = self.feature_fn(state)
        v_s  = np.dot(self.w, x_s)
        v_s_ = 0.0 if done else self.predict(next_state)

        # TD error
        delta = reward + self.gamma * v_s_ - v_s

        # Semi-gradient update: only differentiate through prediction, not target
        self.w += self.alpha * delta * x_s   # gradient = x(s) for linear FA


def run_td0_episode(env, agent, policy):
    """
    Run one episode of semi-gradient TD(0) prediction.

    Parameters
    ----------
    env    : Gymnasium environment
    agent  : SemiGradientTD0 instance
    policy : callable, state -> action

    Returns
    -------
    total_reward : float
    """
    state, _ = env.reset()
    total_reward = 0.0

    while True:
        action = policy(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.update(state, reward, next_state, done)

        state = next_state
        total_reward += reward

        if done:
            break

    return total_reward
```

### Semi-Gradient TD(0) with Tile Coding: Complete Example

```python
import gymnasium as gym

# Environment
env = gym.make("MountainCar-v0")

# Feature representation
coder = TileCoder(
    n_tilings=8,
    tiles_per_dim=8,
    state_low=env.observation_space.low,
    state_high=env.observation_space.high
)

# Agent
agent = SemiGradientTD0(
    feature_fn=coder.encode,
    n_features=coder.n_features,
    alpha=0.1 / coder.n_tilings,   # scale alpha by n_tilings (standard practice)
    gamma=1.0
)

# Fixed policy (e.g., push in direction of velocity)
def velocity_policy(state):
    velocity = state[1]
    return 2 if velocity >= 0 else 0   # push right if moving right

# Train
n_episodes = 500
rewards = []
for ep in range(n_episodes):
    r = run_td0_episode(env, agent, velocity_policy)
    rewards.append(r)
    if (ep + 1) % 100 == 0:
        print(f"Episode {ep+1}: avg reward = {np.mean(rewards[-100:]):.1f}")

env.close()
```

---

## 4. Convergence of Linear TD(0)

### The TD Fixed Point

Semi-gradient TD(0) does not minimize VE directly. Instead, it converges to the **TD fixed point** $\mathbf{w}_{TD}$:

$$\mathbf{w}_{TD} = \mathbf{A}^{-1} \mathbf{b}$$

where:
- $\mathbf{A} = \mathbb{E}[\mathbf{x}(S_t)(\mathbf{x}(S_t) - \gamma \mathbf{x}(S_{t+1}))^T]$
- $\mathbf{b} = \mathbb{E}[R_{t+1} \, \mathbf{x}(S_t)]$

This is a linear system — hence the closed-form solution.

### Convergence Theorem (Tsitsiklis & Van Roy, 1997)

Under on-policy sampling and with a linear approximator, semi-gradient TD(0) converges with probability 1 to $\mathbf{w}_{TD}$ provided:

1. The step size $\alpha_t$ satisfies the Robbins-Monro conditions: $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$
2. The policy $\pi$ induces an ergodic Markov chain (all states visited infinitely often)

### VE at the TD Fixed Point

The VE at $\mathbf{w}_{TD}$ is bounded relative to the minimum VE:

$$\overline{VE}(\mathbf{w}_{TD}) \leq \frac{1}{1 - \gamma} \min_{\mathbf{w}} \overline{VE}(\mathbf{w})$$

The extra factor $1/(1-\gamma)$ means TD is suboptimal compared to Monte Carlo gradient methods. For $\gamma = 0.99$, the TD fixed point can have up to $100\times$ higher VE than the best linear approximation. In practice, TD still wins on sample efficiency because it uses bootstrapping.

### Why Off-Policy Breaks Convergence

The matrix $\mathbf{A}$ above uses the on-policy distribution $\mu(s)$ to weight states. Under off-policy training (e.g., Q-learning with a different behavior policy), $\mu(s)$ changes, and $\mathbf{A}$ may no longer be positive definite — the fixed point equation has no solution or yields unstable dynamics. This is one component of the deadly triad (Guide 03).

---

## 5. Visual: Semi-Gradient TD(0) Flow

```
Episode step:
    (S_t, A_t, R_{t+1}, S_{t+1})
          │
          ▼
    Compute features:
    x_t = x(S_t)                   [d-vector]
    x_t' = x(S_{t+1})             [d-vector]
          │
          ▼
    Compute predictions:
    v_t  = w^T x_t                 [scalar]
    v_t' = w^T x_t'               [scalar]  ← STOP GRADIENT HERE
          │
          ▼
    TD error:
    δ_t = R_{t+1} + γ v_t' - v_t  [scalar]
          │
          ▼
    Update weights:
    w ← w + α δ_t x_t             [d-vector update]
```

The "stop gradient" annotation marks the semi-gradient point. The target $v_t' = \mathbf{w}^T \mathbf{x}(S_{t+1})$ is computed using current $\mathbf{w}$ but treated as a constant during the update.

---

## 6. Comparing Feature Representations

| Feature Type | Computation | Storage | Generalization | Best For |
|---|---|---|---|---|
| Polynomial | $\mathcal{O}(d)$ — dense | Dense | Smooth, global | Low-dim, smooth value fn |
| Fourier | $\mathcal{O}(d)$ — dense | Dense | Frequency-based | Bounded state, periodic tasks |
| Tile coding | $\mathcal{O}(n_{\text{tilings}})$ — sparse | Sparse | Within-tile | General continuous state |
| RBF | $\mathcal{O}(n_{\text{centers}})$ — dense | Dense | Smooth, local | Smooth, low-dim state |

**Rule of thumb:** Use tile coding as the default for continuous 1D–4D state spaces. Add polynomial or Fourier features when the value function has known global structure.

---

## Common Pitfalls

<div class="callout-danger">

<strong>Danger:</strong> The pitfalls below are the most common mistakes practitioners make. Each one can silently degrade your results without obvious errors.

</div>

**Pitfall 1: Setting $\alpha$ too large for tile coding.**
The standard rule is $\alpha = \alpha_0 / n_{\text{tilings}}$ where $\alpha_0 \in [0.1, 0.5]$. Tile coding activates $n_{\text{tilings}}$ features simultaneously, so the effective step size scales with $n_{\text{tilings}}$. Dividing by $n_{\text{tilings}}$ keeps the effective step constant regardless of the number of tilings.

<div class="callout-warning">

<strong>Warning:</strong> **Pitfall 1: Setting $\alpha$ too large for tile coding.**
The standard rule is $\alpha = \alpha_0 / n_{\text{tilings}}$ where $\alpha_0 \in [0.1, 0.5]$.

</div>

**Pitfall 2: Forgetting the terminal-state treatment.**
At terminal states, $V(S_{T}) = 0$ by definition (no future rewards). The TD target for the last transition is simply $R_T$, not $R_T + \gamma \hat{v}(S_T, \mathbf{w})$. Use `done` flag to zero out the bootstrap value.

**Pitfall 3: Using semi-gradient TD for off-policy control (Q-learning + FA).**
Naive Q-learning with linear FA (semi-gradient Q-learning) lacks convergence guarantees when the behavior policy differs from the greedy policy. It can diverge. Use on-policy SARSA + FA for safe convergent control, or DQN techniques for off-policy.

**Pitfall 4: Expecting tile coding to scale to high-dimensional states.**
Tile coding with $d$ dimensions, $m$ tiles, and $n$ tilings has $n \cdot m^d$ features. For $d=6$, $m=8$, $n=8$: $8 \cdot 8^6 = 134$ million features. The sparsity saves update cost but memory becomes prohibitive. Hash-based tile coding (Sutton's tilecoding.py) addresses this via hashing to a fixed-size array.

**Pitfall 5: Confusing the semi-gradient with true gradient.**
Semi-gradient TD does not follow the gradient of any objective function. This means it does not guarantee descent on VE. It converges to a different point (the TD fixed point) that may have higher VE than the best possible linear approximation.

---

## Connections


<div class="callout-info">

<strong>Info:</strong> This section maps how this guide connects to the broader course. Use these links to navigate related material.

</div>

- **Builds on:** Guide 01 (function approximation motivation and feature vectors), Module 03 (tabular TD(0) — linear methods are the same algorithm with generalization added)
- **Leads to:** Guide 03 (the deadly triad — off-policy + FA breaks convergence), Module 05 (DQN uses neural networks in place of linear FA with the same semi-gradient idea)
- **Related to:** Supervised regression (same SGD form, but RL targets are non-stationary), temporal difference learning (tabular TD(0) is a special case with tabular features)

---

## Practice Problems

1. **Feature count.** A mountain car has 2 state dimensions. Using tile coding with 8 tilings and 8 tiles per dimension, how many total features? How many are non-zero for any given state? What is the memory footprint at 4 bytes per float?

2. **Update derivation.** Starting from the VE objective $\overline{VE}(\mathbf{w}) = \mathbb{E}[(V^\pi(S) - \mathbf{w}^T \mathbf{x}(S))^2]$, derive the SGD update. Identify precisely where the TD target replaces $V^\pi(S)$ and where the gradient is truncated.

3. **Convergence check.** Implement semi-gradient SARSA (on-policy control) with tile coding on MountainCar-v0. Plot the learning curve. Then switch to Q-learning (off-policy target). Do both converge? How does the behavior differ?

4. **Learning rate sensitivity.** For tile coding with $n = 1, 2, 4, 8$ tilings, train for 1000 episodes on a fixed environment and plot VE vs episode for $\alpha = 0.5, 0.5/n$. Verify that $\alpha / n$ produces stable learning while $\alpha = 0.5$ diverges for large $n$.

---

## Further Reading

- Sutton & Barto (2018), Chapter 9.4–9.6 — linear semi-gradient methods, tile coding, and Fourier basis (the primary reference for this guide)
- Sutton & Barto (2018), Chapter 9.4 — the TD fixed point and the $1/(1-\gamma)$ bound (Equation 9.14)
- Tsitsiklis & Van Roy (1997), "An analysis of temporal-difference learning with function approximation" — formal convergence proof for on-policy linear TD(0)
- Sutton's tilecoding.py — reference implementation of hash-based tile coding: `http://incompleteideas.net/tiles/tiles3.html`


---

## Cross-References

<a class="link-card" href="./02_linear_methods_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Interactive slide deck covering the key concepts with visual examples.</div>
</a>

<a class="link-card" href="../notebooks/01_tile_coding.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises and real data.</div>
</a>
