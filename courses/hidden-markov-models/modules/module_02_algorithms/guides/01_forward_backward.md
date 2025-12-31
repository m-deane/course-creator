# Forward-Backward Algorithm

## The Evaluation Problem

Given an HMM $\lambda$ and observations $O = o_1, ..., o_T$, compute $P(O | \lambda)$.

**Naive approach**: Sum over all $K^T$ state sequences - exponential complexity!

**Forward algorithm**: Dynamic programming in $O(T \cdot K^2)$ time.

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

### Scaled Forward Algorithm

For long sequences, $\alpha_t(i)$ underflows. Use scaling factors:

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

## State Probabilities

### Posterior State Probability

$\gamma_t(i)$ = probability of being in state $i$ at time $t$ given all observations:

$$\gamma_t(i) = P(q_t = s_i | O, \lambda) = \frac{\alpha_t(i) \cdot \beta_t(i)}{P(O | \lambda)}$$

### Transition Probability

$\xi_t(i, j)$ = probability of transition from $i$ to $j$ at time $t$:

$$\xi_t(i, j) = P(q_t = s_i, q_{t+1} = s_j | O, \lambda)$$

$$= \frac{\alpha_t(i) \cdot a_{ij} \cdot b_j(o_{t+1}) \cdot \beta_{t+1}(j)}{P(O | \lambda)}$$

### Implementation

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

## Key Takeaways

1. **Forward algorithm** computes P(O|λ) efficiently via dynamic programming

2. **Backward algorithm** computes future observation probability given current state

3. **Combined** they give posterior state probabilities (gamma, xi)

4. **Scaling/log-domain** essential for numerical stability

5. **O(T × K²)** complexity - linear in sequence length
