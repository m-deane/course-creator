# Markov Chains Foundations

> **Reading time:** ~6 min | **Module:** Module 0: Foundations | **Prerequisites:** Basic linear algebra, Python

<div class="callout-key">

**Key Concept Summary:** A Markov chain is a stochastic process where future states depend only on the current state—not the past.

</div>

## What is a Markov Chain?

A Markov chain is a stochastic process where future states depend only on the current state—not the past.

### The Markov Property

$$P(X_{t+1} | X_t, X_{t-1}, ..., X_1) = P(X_{t+1} | X_t)$$

<div class="callout-insight">

**Insight:** "The future is independent of the past, given the present." This memorylessness property is what makes Markov chains tractable — you only need to track the current state, not the entire history.

</div>
<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## Formal Definition

A discrete-time Markov chain consists of:

<div class="flow">
<div class="flow-step mint">1. State space</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">2. Transition matrix</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">3. Initial distribution</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Ergodicity</div>
</div>


1. **State space** $S = \{s_1, s_2, ..., s_K\}$
2. **Transition matrix** $A$ where $a_{ij} = P(X_{t+1} = s_j | X_t = s_i)$
3. **Initial distribution** $\pi$ where $\pi_i = P(X_1 = s_i)$

### Transition Matrix Properties

$$A = \begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1K} \\
a_{21} & a_{22} & \cdots & a_{2K} \\
\vdots & \vdots & \ddots & \vdots \\
a_{K1} & a_{K2} & \cdots & a_{KK}
\end{pmatrix}$$

Constraints:
- $a_{ij} \geq 0$ (non-negative)
- $\sum_{j=1}^K a_{ij} = 1$ (rows sum to 1)

## How to Build a Markov Chain in Python

### Basic Markov Chain Class

```python
import numpy as np
from typing import List, Optional

class MarkovChain:
    """Discrete-time Markov chain."""

    def __init__(
        self,
        transition_matrix: np.ndarray,
        state_names: Optional[List[str]] = None
    ):
        self.A = np.array(transition_matrix)
        self.n_states = self.A.shape[0]
        self.state_names = state_names or [f"S{i}" for i in range(self.n_states)]

        # Validate
        assert self.A.shape[0] == self.A.shape[1], "Matrix must be square"
        assert np.allclose(self.A.sum(axis=1), 1), "Rows must sum to 1"
        assert np.all(self.A >= 0), "Probabilities must be non-negative"

    def step(self, current_state: int) -> int:
        """Sample next state given current state."""
        return np.random.choice(self.n_states, p=self.A[current_state])

    def simulate(
        self,
        n_steps: int,
        initial_state: Optional[int] = None,
        initial_dist: Optional[np.ndarray] = None
    ) -> List[int]:
        """Simulate a sequence of states."""
        if initial_state is None:
            if initial_dist is None:
                initial_dist = np.ones(self.n_states) / self.n_states
            initial_state = np.random.choice(self.n_states, p=initial_dist)

        states = [initial_state]
        for _ in range(n_steps - 1):
            states.append(self.step(states[-1]))

        return states

    def n_step_transition(self, n: int) -> np.ndarray:
        """Compute n-step transition matrix A^n."""
        return np.linalg.matrix_power(self.A, n)

    def stationary_distribution(self) -> np.ndarray:
        """Compute stationary distribution (left eigenvector for eigenvalue 1)."""
        eigenvalues, eigenvectors = np.linalg.eig(self.A.T)

        # Find eigenvector for eigenvalue ≈ 1
        idx = np.argmin(np.abs(eigenvalues - 1))
        stationary = np.real(eigenvectors[:, idx])

        # Normalize to probability distribution
        return stationary / stationary.sum()

# Example: Market regime model
transition_matrix = np.array([
    [0.95, 0.05],  # Bull → Bull: 95%, Bull → Bear: 5%
    [0.15, 0.85]   # Bear → Bull: 15%, Bear → Bear: 85%
])

mc = MarkovChain(transition_matrix, state_names=["Bull", "Bear"])

# Simulate 100 steps
states = mc.simulate(100, initial_state=0)
print(f"Simulated states: {states[:20]}...")

# Stationary distribution
pi = mc.stationary_distribution()
print(f"Stationary distribution: Bull={pi[0]:.2%}, Bear={pi[1]:.2%}")
```

## Key Properties

### Stationary Distribution

The stationary distribution $\pi^*$ satisfies:

$$\pi^* = \pi^* A$$

or equivalently, $\pi^*$ is the left eigenvector of $A$ for eigenvalue 1.

```python
def verify_stationary(mc: MarkovChain):
    """Verify stationary distribution properties."""
    pi = mc.stationary_distribution()

    # Check pi * A = pi
    pi_next = pi @ mc.A
    print(f"π: {pi}")
    print(f"πA: {pi_next}")
    print(f"Converged: {np.allclose(pi, pi_next)}")

    # Long-run simulation check
    long_simulation = mc.simulate(10000, initial_state=0)
    empirical_dist = np.bincount(long_simulation, minlength=mc.n_states) / len(long_simulation)
    print(f"Empirical: {empirical_dist}")
    print(f"Theoretical: {pi}")
```

### Ergodicity

A Markov chain is **ergodic** if it is:
1. **Irreducible**: Every state reachable from every other state
2. **Aperiodic**: No regular cycling patterns

Ergodic chains have a unique stationary distribution.

```python
def check_irreducibility(A: np.ndarray) -> bool:
    """Check if chain is irreducible using reachability."""
    n = A.shape[0]
    # Sum of powers up to n-1 shows all reachable states
    reachability = sum(np.linalg.matrix_power(A > 0, k) for k in range(n))
    return np.all(reachability > 0)

def check_aperiodicity(A: np.ndarray) -> bool:
    """Check if chain is aperiodic."""
    # If any self-transition, chain is aperiodic
    if np.any(np.diag(A) > 0):
        return True

    # Otherwise, need to check GCD of cycle lengths
    # Simplified: check if powers eventually have all positive entries
    An = A.copy()
    for _ in range(A.shape[0] * 2):
        An = An @ A
        if np.all(An > 0):
            return True
    return False
```

### Convergence Rate

The rate of convergence to the stationary distribution depends on the **second largest eigenvalue**:

$$\|\pi_t - \pi^*\| \leq C \cdot |\lambda_2|^t$$


<span class="filename">convergence_rate.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def convergence_rate(mc: MarkovChain) -> float:
    """Compute convergence rate (second largest eigenvalue magnitude)."""
    eigenvalues = np.linalg.eigvals(mc.A)
    eigenvalues_sorted = sorted(np.abs(eigenvalues), reverse=True)
    return eigenvalues_sorted[1]

rate = convergence_rate(mc)
print(f"Convergence rate: {rate:.4f}")
print(f"Half-life: {np.log(2) / np.log(1/rate):.1f} steps")
```

</div>
</div>

## Expected Hitting Times

The expected number of steps to reach state $j$ starting from state $i$:


<span class="filename">expected_hitting_time.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def expected_hitting_time(mc: MarkovChain, target_state: int) -> np.ndarray:
    """
    Compute expected hitting times to target state from all other states.
    """
    n = mc.n_states
    j = target_state

    # System: h_i = 1 + sum_{k != j} A_{ik} * h_k for i != j
    # h_j = 0

    # Remove target state from system
    mask = np.ones(n, dtype=bool)
    mask[j] = False

    A_reduced = mc.A[np.ix_(mask, mask)]
    b = np.ones(n - 1)

    # Solve (I - A_reduced) h = 1
    hitting_times = np.linalg.solve(np.eye(n-1) - A_reduced, b)

    # Insert 0 for target state
    result = np.zeros(n)
    result[mask] = hitting_times

    return result

# Expected time to reach Bear state from Bull
hitting = expected_hitting_time(mc, target_state=1)
print(f"Expected steps to Bear from Bull: {hitting[0]:.1f}")
```

</div>
</div>

## Estimating Transition Probabilities

### Maximum Likelihood Estimation


<span class="filename">estimate_transitions.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def estimate_transitions(states: List[int], n_states: int) -> np.ndarray:
    """
    Estimate transition matrix from observed state sequence.
    """
    # Count transitions
    counts = np.zeros((n_states, n_states))

    for i in range(len(states) - 1):
        counts[states[i], states[i+1]] += 1

    # Normalize rows
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero

    return counts / row_sums

# Estimate from simulation
estimated_A = estimate_transitions(states, mc.n_states)
print("Estimated transition matrix:")
print(estimated_A)
print("\nTrue transition matrix:")
print(mc.A)
```

</div>
</div>

### Bayesian Estimation


<span class="filename">bayesian_transitions.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def bayesian_transitions(
    states: List[int],
    n_states: int,
    prior_alpha: float = 1.0
) -> tuple:
    """
    Bayesian estimation with Dirichlet prior.
    Returns posterior mean and credible intervals.
    """
    from scipy import stats

    counts = np.zeros((n_states, n_states))
    for i in range(len(states) - 1):
        counts[states[i], states[i+1]] += 1

    # Posterior parameters (Dirichlet)
    posterior_alpha = counts + prior_alpha

    # Posterior mean
    mean = posterior_alpha / posterior_alpha.sum(axis=1, keepdims=True)

    # Credible intervals (using Beta marginals)
    ci_lower = np.zeros_like(mean)
    ci_upper = np.zeros_like(mean)

    for i in range(n_states):
        for j in range(n_states):
            alpha = posterior_alpha[i, j]
            beta = posterior_alpha[i].sum() - alpha
            ci_lower[i, j] = stats.beta.ppf(0.025, alpha, beta)
            ci_upper[i, j] = stats.beta.ppf(0.975, alpha, beta)

    return mean, ci_lower, ci_upper
```

</div>
</div>

## Visualization


<span class="filename">plot_markov_chain.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import matplotlib.pyplot as plt

def plot_markov_chain(mc: MarkovChain, simulation: List[int] = None):
    """Visualize Markov chain and simulation."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Transition matrix heatmap
    ax1 = axes[0]
    im = ax1.imshow(mc.A, cmap='Blues', vmin=0, vmax=1)
    ax1.set_xticks(range(mc.n_states))
    ax1.set_yticks(range(mc.n_states))
    ax1.set_xticklabels(mc.state_names)
    ax1.set_yticklabels(mc.state_names)
    ax1.set_xlabel('To State')
    ax1.set_ylabel('From State')
    ax1.set_title('Transition Matrix')

    # Add text annotations
    for i in range(mc.n_states):
        for j in range(mc.n_states):
            ax1.text(j, i, f'{mc.A[i,j]:.2f}', ha='center', va='center')

    plt.colorbar(im, ax=ax1)

    # Simulation trajectory
    if simulation:
        ax2 = axes[1]
        ax2.step(range(len(simulation)), simulation, where='mid')
        ax2.set_yticks(range(mc.n_states))
        ax2.set_yticklabels(mc.state_names)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('State')
        ax2.set_title('Simulated Trajectory')

    plt.tight_layout()
    plt.show()

plot_markov_chain(mc, states[:100])
```

</div>
</div>

<div class="callout-insight">

**Insight:** Understanding markov chains foundations is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Key Takeaways

1. **Markov property** enables tractable analysis of sequential data

2. **Transition matrix** encodes all dynamics in a compact form

3. **Stationary distribution** describes long-run behavior

4. **Ergodicity** ensures convergence to unique equilibrium

5. **MLE estimation** is straightforward from observed sequences

---

## Conceptual Practice Questions

1. Explain the Markov property in your own words. Why does it make sequential modeling tractable?

2. If a transition matrix has rows [0.9, 0.1] and [0.3, 0.7], what is the expected duration of each state?

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="./01_markov_chains_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_markov_chain_basics.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./02_probability_review.md">
  <div class="link-card-title">02 Probability Review</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_transition_matrices.md">
  <div class="link-card-title">02 Transition Matrices</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

