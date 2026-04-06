# Forward-Backward Algorithm

> **Reading time:** ~5 min | **Module:** Module 2: Algorithms | **Prerequisites:** Modules 0-1

<div class="callout-key">

**Key Concept Summary:** Given an HMM $\lambda$ and observations $O = o_1, ..., o_T$, compute $P(O | \lambda)$.

</div>

## The Evaluation Problem

Given an HMM $\lambda$ and observations $O = o_1, ..., o_T$, compute $P(O | \lambda)$.

**Naive approach**: Sum over all $K^T$ state sequences - exponential complexity!

**Forward algorithm**: Dynamic programming in $O(T \cdot K^2)$ time.

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## Forward Algorithm

### Forward Variables

Define $\alpha_t(i)$ as the probability of observing $o_1, ..., o_t$ AND being in state $i$ at time $t$:

$$\alpha_t(i) = P(o_1, o_2, ..., o_t, q_t = s_i | \lambda)$$

### Recursion

**Initialization** ($t = 1$):
$$\alpha_1(i) = \pi_i \cdot b_i(o_1)$$

**Induction** ($t = 2, ..., T$):
$$\alpha_t(j) = \left[ \sum_{i=1}^{K} \alpha_{t-1}(i) \cdot a_{ij} \right] \cdot b_j(o_t)$$

**Termination**:
$$P(O | \lambda) = \sum_{i=1}^{K} \alpha_T(i)$$

### Implementation


<span class="filename">forward.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np
from typing import Tuple

def forward(
    observations: np.ndarray,
    pi: np.ndarray,
    A: np.ndarray,
    B: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Forward algorithm for HMM.

    Args:
        observations: Sequence of observation indices (T,)
        pi: Initial distribution (K,)
        A: Transition matrix (K, K)
        B: Emission matrix (K, M)

    Returns:
        alpha: Forward variables (T, K)
        log_likelihood: Log P(O|λ)
    """
    T = len(observations)
    K = len(pi)

    # Initialize alpha matrix
    alpha = np.zeros((T, K))

    # Initialization (t=0)
    alpha[0] = pi * B[:, observations[0]]

    # Induction
    for t in range(1, T):
        for j in range(K):
            alpha[t, j] = np.sum(alpha[t-1] * A[:, j]) * B[j, observations[t]]

    # Total probability
    likelihood = np.sum(alpha[-1])

    return alpha, np.log(likelihood)

# Example
pi = np.array([0.6, 0.4])
A = np.array([[0.7, 0.3], [0.4, 0.6]])
B = np.array([[0.6, 0.3, 0.1], [0.1, 0.4, 0.5]])

observations = np.array([0, 1, 2, 1, 0])  # Walk, Shop, Clean, Shop, Walk

alpha, log_lik = forward(observations, pi, A, B)
print(f"Log-likelihood: {log_lik:.4f}")
print(f"Forward variables:\n{alpha}")
```

</div>
</div>

### Scaled Forward Algorithm

For long sequences, $\alpha_t(i)$ underflows. Use scaling factors:


<span class="filename">forward_scaled.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def forward_scaled(
    observations: np.ndarray,
    pi: np.ndarray,
    A: np.ndarray,
    B: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Scaled forward algorithm to prevent underflow.
    """
    T = len(observations)
    K = len(pi)

    alpha = np.zeros((T, K))
    scaling = np.zeros(T)

    # Initialization
    alpha[0] = pi * B[:, observations[0]]
    scaling[0] = np.sum(alpha[0])
    alpha[0] /= scaling[0]

    # Induction
    for t in range(1, T):
        for j in range(K):
            alpha[t, j] = np.sum(alpha[t-1] * A[:, j]) * B[j, observations[t]]
        scaling[t] = np.sum(alpha[t])
        alpha[t] /= scaling[t]

    # Log-likelihood from scaling factors
    log_likelihood = np.sum(np.log(scaling))

    return alpha, scaling, log_likelihood
```

</div>
</div>

## Backward Algorithm

### Backward Variables

Define $\beta_t(i)$ as the probability of observing $o_{t+1}, ..., o_T$ given state $i$ at time $t$:

$$\beta_t(i) = P(o_{t+1}, ..., o_T | q_t = s_i, \lambda)$$

### Recursion

**Initialization** ($t = T$):
$$\beta_T(i) = 1$$

**Induction** ($t = T-1, ..., 1$):
$$\beta_t(i) = \sum_{j=1}^{K} a_{ij} \cdot b_j(o_{t+1}) \cdot \beta_{t+1}(j)$$

### Implementation


<span class="filename">backward.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def backward(
    observations: np.ndarray,
    A: np.ndarray,
    B: np.ndarray
) -> np.ndarray:
    """
    Backward algorithm for HMM.
    """
    T = len(observations)
    K = A.shape[0]

    beta = np.zeros((T, K))

    # Initialization
    beta[-1] = 1

    # Induction (backwards)
    for t in range(T-2, -1, -1):
        for i in range(K):
            beta[t, i] = np.sum(
                A[i, :] * B[:, observations[t+1]] * beta[t+1]
            )

    return beta

def backward_scaled(
    observations: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    scaling: np.ndarray
) -> np.ndarray:
    """
    Scaled backward algorithm (using forward scaling factors).
    """
    T = len(observations)
    K = A.shape[0]

    beta = np.zeros((T, K))
    beta[-1] = 1 / scaling[-1]

    for t in range(T-2, -1, -1):
        for i in range(K):
            beta[t, i] = np.sum(
                A[i, :] * B[:, observations[t+1]] * beta[t+1]
            )
        beta[t] /= scaling[t]

    return beta
```

</div>
</div>

## State Probabilities

### Posterior State Probability

$\gamma_t(i)$ = probability of being in state $i$ at time $t$ given all observations:

$$\gamma_t(i) = P(q_t = s_i | O, \lambda) = \frac{\alpha_t(i) \cdot \beta_t(i)}{P(O | \lambda)}$$

### Transition Probability

$\xi_t(i, j)$ = probability of transition from $i$ to $j$ at time $t$:

$$\xi_t(i, j) = P(q_t = s_i, q_{t+1} = s_j | O, \lambda)$$

$$= \frac{\alpha_t(i) \cdot a_{ij} \cdot b_j(o_{t+1}) \cdot \beta_{t+1}(j)}{P(O | \lambda)}$$

### Implementation


<span class="filename">compute_posteriors.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def compute_posteriors(
    observations: np.ndarray,
    pi: np.ndarray,
    A: np.ndarray,
    B: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gamma and xi using forward-backward.
    """
    T = len(observations)
    K = len(pi)

    # Forward-backward
    alpha, scaling, _ = forward_scaled(observations, pi, A, B)
    beta = backward_scaled(observations, A, B, scaling)

    # Gamma: P(q_t = i | O)
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)

    # Xi: P(q_t = i, q_{t+1} = j | O)
    xi = np.zeros((T-1, K, K))

    for t in range(T-1):
        for i in range(K):
            for j in range(K):
                xi[t, i, j] = (
                    alpha[t, i] *
                    A[i, j] *
                    B[j, observations[t+1]] *
                    beta[t+1, j]
                )
        xi[t] /= xi[t].sum()

    return gamma, xi

# Compute posteriors
gamma, xi = compute_posteriors(observations, pi, A, B)

print("State probabilities at each time step:")
for t, g in enumerate(gamma):
    print(f"  t={t}: P(Sunny)={g[0]:.3f}, P(Rainy)={g[1]:.3f}")
```

</div>
</div>

## Relationship Between Variables

```
Forward (α):   Past → Present
               Computes P(o₁...oₜ, qₜ=i)

Backward (β):  Future ← Present
               Computes P(oₜ₊₁...oT | qₜ=i)

Combined:      P(qₜ=i | O) ∝ α(i) × β(i)
               Full posterior using all observations
```

## Computational Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Forward | O(T × K²) | O(T × K) |
| Backward | O(T × K²) | O(T × K) |
| Gamma | O(T × K) | O(T × K) |
| Xi | O(T × K²) | O(T × K²) |

## Vectorized Implementation


<span class="filename">forward_vectorized.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def forward_vectorized(
    observations: np.ndarray,
    pi: np.ndarray,
    A: np.ndarray,
    B: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Vectorized forward algorithm.
    """
    T = len(observations)
    K = len(pi)

    log_alpha = np.zeros((T, K))

    # Initialization (log domain)
    log_alpha[0] = np.log(pi) + np.log(B[:, observations[0]])

    # Induction
    for t in range(1, T):
        # Log-sum-exp for numerical stability
        for j in range(K):
            log_alpha[t, j] = (
                np.logaddexp.reduce(log_alpha[t-1] + np.log(A[:, j])) +
                np.log(B[j, observations[t]])
            )

    # Total log-likelihood
    log_likelihood = np.logaddexp.reduce(log_alpha[-1])

    return np.exp(log_alpha), log_likelihood
```

</div>
</div>

<div class="callout-insight">

**Insight:** Understanding forward-backward algorithm is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Key Takeaways

1. **Forward algorithm** computes P(O|λ) efficiently via dynamic programming

2. **Backward algorithm** computes future observation probability given current state

3. **Combined** they give posterior state probabilities (gamma, xi)

4. **Scaling/log-domain** essential for numerical stability

5. **O(T × K²)** complexity - linear in sequence length

---

## Conceptual Practice Questions

1. Walk through the forward algorithm step by step. What is the computational advantage over brute force?

2. Why do we need the backward algorithm in addition to the forward algorithm?

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="./01_forward_backward_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_forward_backward_impl.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./02_viterbi_algorithm.md">
  <div class="link-card-title">02 Viterbi Algorithm</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_baum_welch.md">
  <div class="link-card-title">03 Baum Welch</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

