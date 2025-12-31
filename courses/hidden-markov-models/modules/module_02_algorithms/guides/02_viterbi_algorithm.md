# The Viterbi Algorithm: Finding the Most Likely State Sequence

## Introduction

The Viterbi algorithm solves the **decoding problem**: Given observations $O = (o_1, ..., o_T)$ and model $\lambda$, find the most likely state sequence:

$$S^* = \arg\max_S P(S | O, \lambda)$$

This is different from computing marginal probabilities at each time step.

## The Key Insight

### Why Not Just Take Marginal Maxima?

Taking the most likely state at each time step independently can give an **impossible** sequence (zero probability transitions).

```python
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_viterbi_vs_marginal():
    """
    Show why marginal maximization can fail.
    """
    # Simple 2-state HMM
    # State 0 always transitions to state 1
    # State 1 always transitions to state 0
    A = np.array([
        [0.0, 1.0],
        [1.0, 0.0]
    ])

    # Emissions
    B = np.array([
        [0.9, 0.1],  # State 0 usually emits symbol 0
        [0.2, 0.8]   # State 1 usually emits symbol 1
    ])

    pi = np.array([1.0, 0.0])  # Start in state 0

    # Observations
    observations = [0, 1, 0, 1]  # Alternating

    print("Transition matrix (deterministic alternation):")
    print(A)
    print("\nObservations:", observations)

    # Marginal probabilities
    # With this transition matrix, states MUST alternate
    # So marginal at each step is:
    # t=0: State 0 (started there)
    # t=1: State 1 (must transition from 0)
    # t=2: State 0 (must transition from 1)
    # etc.

    marginal_sequence = [0, 1, 0, 1]

    # Any other sequence is IMPOSSIBLE (probability 0)
    print("\nMarginal maximization gives:", marginal_sequence)
    print("Viterbi would give the same (only valid sequence)")

    # Now consider a case where they differ
    A2 = np.array([
        [0.7, 0.3],
        [0.4, 0.6]
    ])

    B2 = np.array([
        [0.9, 0.1],
        [0.4, 0.6]
    ])

    obs2 = [0, 0, 1, 0]

    print("\n" + "=" * 60)
    print("Example where marginal and Viterbi can differ:")
    print("=" * 60)

    # This requires actual computation - shown in next section

demonstrate_viterbi_vs_marginal()
```

## The Viterbi Recursion

### Algorithm Overview

Define:
$$\delta_t(i) = \max_{s_1,...,s_{t-1}} P(s_1,...,s_{t-1}, s_t=i, o_1,...,o_t | \lambda)$$

This is the probability of the most likely path ending in state $i$ at time $t$.

### Recursion

**Initialization:**
$$\delta_1(i) = \pi_i b_i(o_1)$$

**Recursion:**
$$\delta_t(j) = \max_i [\delta_{t-1}(i) \cdot a_{ij}] \cdot b_j(o_t)$$

**Backtracking:**
$$\psi_t(j) = \arg\max_i [\delta_{t-1}(i) \cdot a_{ij}]$$

### Implementation

```python
def viterbi_algorithm(observations, pi, A, B):
    """
    Viterbi algorithm for discrete HMM.

    Parameters:
    -----------
    observations : list
        Sequence of observed symbols (integers)
    pi : array
        Initial state distribution
    A : array
        Transition matrix (K x K)
    B : array
        Emission matrix (K x M)

    Returns:
    --------
    best_path : list
        Most likely state sequence
    best_prob : float
        Probability of best path
    delta : array
        Viterbi probabilities (T x K)
    psi : array
        Backtrack pointers (T x K)
    """
    T = len(observations)
    K = len(pi)

    # Initialize
    delta = np.zeros((T, K))
    psi = np.zeros((T, K), dtype=int)

    # t = 0
    delta[0] = pi * B[:, observations[0]]
    psi[0] = 0  # No previous state

    # Forward pass
    for t in range(1, T):
        for j in range(K):
            # Find best previous state
            probs = delta[t-1] * A[:, j]
            psi[t, j] = np.argmax(probs)
            delta[t, j] = probs[psi[t, j]] * B[j, observations[t]]

    # Backtrack
    best_path = [0] * T
    best_path[T-1] = np.argmax(delta[T-1])
    best_prob = delta[T-1, best_path[T-1]]

    for t in range(T-2, -1, -1):
        best_path[t] = psi[t+1, best_path[t+1]]

    return best_path, best_prob, delta, psi

# Example
pi = np.array([0.6, 0.4])
A = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])
B = np.array([
    [0.5, 0.5],
    [0.1, 0.9]
])

observations = [0, 0, 1, 1, 0]

best_path, best_prob, delta, psi = viterbi_algorithm(observations, pi, A, B)

print("Viterbi Algorithm Results:")
print("=" * 60)
print(f"Observations: {observations}")
print(f"Best path:    {best_path}")
print(f"Best prob:    {best_prob:.6e}")
print("\nDelta (Viterbi probabilities):")
print(delta)
print("\nPsi (backtrack pointers):")
print(psi)
```

## Log-Space Implementation

For numerical stability with long sequences:

```python
def viterbi_log(observations, log_pi, log_A, log_B):
    """
    Viterbi algorithm in log-space for numerical stability.
    """
    T = len(observations)
    K = len(log_pi)

    # Initialize in log space
    log_delta = np.zeros((T, K))
    psi = np.zeros((T, K), dtype=int)

    # t = 0
    log_delta[0] = log_pi + log_B[:, observations[0]]

    # Forward pass
    for t in range(1, T):
        for j in range(K):
            probs = log_delta[t-1] + log_A[:, j]
            psi[t, j] = np.argmax(probs)
            log_delta[t, j] = probs[psi[t, j]] + log_B[j, observations[t]]

    # Backtrack
    best_path = [0] * T
    best_path[T-1] = np.argmax(log_delta[T-1])
    log_prob = log_delta[T-1, best_path[T-1]]

    for t in range(T-2, -1, -1):
        best_path[t] = psi[t+1, best_path[t+1]]

    return best_path, log_prob, log_delta, psi

# Convert to log space
log_pi = np.log(pi + 1e-10)
log_A = np.log(A + 1e-10)
log_B = np.log(B + 1e-10)

best_path_log, log_prob, log_delta, _ = viterbi_log(
    observations, log_pi, log_A, log_B
)

print("\nLog-space Viterbi Results:")
print(f"Best path: {best_path_log}")
print(f"Log prob:  {log_prob:.4f}")
print(f"Prob:      {np.exp(log_prob):.6e}")
```

## Gaussian HMM Viterbi

```python
from scipy import stats

def viterbi_gaussian(observations, pi, A, means, covars):
    """
    Viterbi algorithm for Gaussian HMM.
    """
    T = len(observations)
    K = len(pi)

    # Compute emission log-probabilities
    log_B = np.zeros((T, K))
    for t in range(T):
        for k in range(K):
            log_B[t, k] = stats.norm.logpdf(
                observations[t],
                loc=means[k],
                scale=np.sqrt(covars[k])
            )

    # Log space Viterbi
    log_delta = np.zeros((T, K))
    psi = np.zeros((T, K), dtype=int)

    log_pi = np.log(pi + 1e-10)
    log_A = np.log(A + 1e-10)

    # t = 0
    log_delta[0] = log_pi + log_B[0]

    # Forward pass
    for t in range(1, T):
        for j in range(K):
            probs = log_delta[t-1] + log_A[:, j]
            psi[t, j] = np.argmax(probs)
            log_delta[t, j] = probs[psi[t, j]] + log_B[t, j]

    # Backtrack
    best_path = [0] * T
    best_path[T-1] = np.argmax(log_delta[T-1])

    for t in range(T-2, -1, -1):
        best_path[t] = psi[t+1, best_path[t+1]]

    return best_path, log_delta

# Example: Market regimes
np.random.seed(42)

# True parameters
true_pi = np.array([0.5, 0.5])
true_A = np.array([
    [0.9, 0.1],
    [0.1, 0.9]
])
true_means = np.array([0.05, -0.03])  # Bull vs Bear returns
true_vars = np.array([0.01, 0.04])    # Low vs High volatility

# Generate data
T = 100
true_states = [0]
observations = []

for t in range(T):
    # State transition
    if t > 0:
        true_states.append(
            np.random.choice(2, p=true_A[true_states[-1]])
        )
    # Emission
    obs = np.random.normal(
        true_means[true_states[-1]],
        np.sqrt(true_vars[true_states[-1]])
    )
    observations.append(obs)

# Decode with Viterbi
decoded_path, log_delta = viterbi_gaussian(
    observations, true_pi, true_A, true_means, true_vars
)

# Visualize
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Observations
ax1 = axes[0]
ax1.plot(observations, 'k-', alpha=0.7)
ax1.set_ylabel('Return')
ax1.set_title('Observations (Returns)')
ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)

# True states
ax2 = axes[1]
ax2.fill_between(range(T), true_states, alpha=0.5, step='mid')
ax2.set_ylabel('State')
ax2.set_title('True States')
ax2.set_yticks([0, 1])
ax2.set_yticklabels(['Bull', 'Bear'])

# Decoded states
ax3 = axes[2]
ax3.fill_between(range(T), decoded_path, alpha=0.5, step='mid', color='orange')
ax3.set_ylabel('State')
ax3.set_title('Viterbi Decoded States')
ax3.set_yticks([0, 1])
ax3.set_yticklabels(['Bull', 'Bear'])
ax3.set_xlabel('Time')

plt.tight_layout()
plt.show()

# Accuracy
accuracy = np.mean(np.array(decoded_path) == np.array(true_states))
print(f"\nDecoding accuracy: {accuracy:.1%}")
```

## Viterbi vs Forward-Backward

```python
def compare_viterbi_forward_backward(observations, pi, A, B):
    """
    Compare Viterbi (most likely path) vs marginal maximization.
    """
    T = len(observations)
    K = len(pi)

    # Forward-backward (simplified)
    # Forward
    alpha = np.zeros((T, K))
    alpha[0] = pi * B[:, observations[0]]
    for t in range(1, T):
        alpha[t] = (alpha[t-1] @ A) * B[:, observations[t]]

    # Backward
    beta = np.zeros((T, K))
    beta[T-1] = 1
    for t in range(T-2, -1, -1):
        beta[t] = A @ (B[:, observations[t+1]] * beta[t+1])

    # Marginal (gamma)
    gamma = alpha * beta
    gamma = gamma / gamma.sum(axis=1, keepdims=True)

    # Marginal maximization
    marginal_path = np.argmax(gamma, axis=1).tolist()

    # Viterbi
    viterbi_path, _, _, _ = viterbi_algorithm(observations, pi, A, B)

    # Compare
    print("Viterbi vs Marginal Maximization:")
    print("=" * 60)
    print(f"Observations:    {observations}")
    print(f"Viterbi path:    {viterbi_path}")
    print(f"Marginal path:   {marginal_path}")
    print(f"Same: {viterbi_path == marginal_path}")

    # Probability of each path
    def path_probability(path):
        prob = pi[path[0]] * B[path[0], observations[0]]
        for t in range(1, T):
            prob *= A[path[t-1], path[t]] * B[path[t], observations[t]]
        return prob

    print(f"\nP(Viterbi path):  {path_probability(viterbi_path):.6e}")
    print(f"P(Marginal path): {path_probability(marginal_path):.6e}")

    return gamma, viterbi_path, marginal_path

# Example where they might differ
pi = np.array([0.5, 0.5])
A = np.array([
    [0.9, 0.1],
    [0.1, 0.9]
])
B = np.array([
    [0.6, 0.4],
    [0.4, 0.6]
])

observations = [0, 1, 0, 1, 0]

gamma, viterbi, marginal = compare_viterbi_forward_backward(
    observations, pi, A, B
)
```

## Applications in Finance

```python
def regime_detection_with_viterbi():
    """
    Use Viterbi for market regime detection.
    """
    # Simulate market data
    np.random.seed(123)
    T = 500

    # True regimes
    true_regimes = []
    current = 0

    for _ in range(T):
        true_regimes.append(current)
        if current == 0:  # Bull
            current = 1 if np.random.random() < 0.02 else 0
        else:  # Bear
            current = 0 if np.random.random() < 0.05 else 1

    # Generate returns
    bull_params = (0.0004, 0.01)   # Mean, std (daily)
    bear_params = (-0.001, 0.025)

    returns = []
    for regime in true_regimes:
        params = bull_params if regime == 0 else bear_params
        returns.append(np.random.normal(params[0], params[1]))

    # Decode
    pi = np.array([0.5, 0.5])
    A = np.array([[0.98, 0.02], [0.05, 0.95]])

    decoded, _ = viterbi_gaussian(
        returns, pi, A,
        means=np.array([0.0004, -0.001]),
        covars=np.array([0.01**2, 0.025**2])
    )

    # Visualize
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Cumulative returns
    ax1 = axes[0]
    cumret = np.cumsum(returns)
    ax1.plot(cumret, 'k-', alpha=0.8)
    ax1.set_ylabel('Cumulative Return')
    ax1.set_title('Simulated Market Returns')

    # Color by true regime
    for t in range(T-1):
        color = 'green' if true_regimes[t] == 0 else 'red'
        ax1.axvspan(t, t+1, alpha=0.1, color=color)

    # True vs decoded
    ax2 = axes[1]
    ax2.fill_between(range(T), true_regimes, alpha=0.5, label='True', step='mid')
    ax2.set_ylabel('True Regime')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Bull', 'Bear'])

    ax3 = axes[2]
    ax3.fill_between(range(T), decoded, alpha=0.5, color='orange',
                     label='Decoded', step='mid')
    ax3.set_ylabel('Decoded Regime')
    ax3.set_xlabel('Day')
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Bull', 'Bear'])

    plt.tight_layout()
    plt.show()

    accuracy = np.mean(np.array(decoded) == np.array(true_regimes))
    print(f"\nRegime detection accuracy: {accuracy:.1%}")

regime_detection_with_viterbi()
```

## Key Takeaways

1. **Viterbi finds the most likely state sequence** - not marginal maxima

2. **Dynamic programming** makes it computationally efficient: $O(TK^2)$

3. **Log-space computation** prevents numerical underflow

4. **Backtracking** recovers the optimal path after forward pass

5. **Different from marginal probabilities** - Viterbi ensures valid transitions

6. **Essential for regime detection** in financial applications
