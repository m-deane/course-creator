# Markov Chains: Foundations

## In Brief

A Markov chain is a stochastic process where the probability of transitioning to any future state depends only on the current state, not on how the process arrived there. This "memoryless" property is the Markov property, and it is the mathematical foundation upon which all Hidden Markov Model algorithms rest.

## Key Insight

The Markov property converts an otherwise intractable sequential modeling problem into a tractable one. Without it, predicting the next state would require tracking the entire history of the process — exponential complexity. With it, the current state is a sufficient statistic for the future, reducing the problem to matrix operations.

In financial modeling, the Markov property formalizes the intuition that "the current market regime contains all relevant information about future regime transitions." Past regime history beyond the current state adds no predictive power.

## Formal Definition

A discrete-time Markov chain is a sequence of random variables $X_1, X_2, X_3, \ldots$ taking values in a state space $S = \{s_1, s_2, \ldots, s_K\}$ that satisfies the **Markov property**:

$$P(X_{t+1} = s_j \mid X_t = s_i, X_{t-1} = s_{i-1}, \ldots, X_1 = s_{i_1}) = P(X_{t+1} = s_j \mid X_t = s_i)$$

A homogeneous (time-invariant) Markov chain is fully specified by three components:

1. **State space** $S = \{s_1, \ldots, s_K\}$
2. **Transition matrix** $A \in \mathbb{R}^{K \times K}$ where $a_{ij} = P(X_{t+1} = s_j \mid X_t = s_i)$
3. **Initial distribution** $\pi$ where $\pi_i = P(X_1 = s_i)$

The transition matrix satisfies two constraints:
- **Non-negativity**: $a_{ij} \geq 0$ for all $i, j$
- **Row-stochastic**: $\sum_{j=1}^K a_{ij} = 1$ for all $i$

### Stationary Distribution

A probability vector $\pi^*$ is a **stationary distribution** if $\pi^* = \pi^* A$. It is the left eigenvector of $A$ corresponding to eigenvalue 1.

### Ergodicity

A Markov chain is **ergodic** if it is:
- **Irreducible**: every state is reachable from every other state
- **Aperiodic**: the chain does not cycle through states deterministically

Ergodic chains have a unique stationary distribution and converge to it from any starting distribution.

## Intuitive Explanation

Imagine a weather system that switches between two states: Sunny and Rainy. Each morning the weather changes or stays the same according to fixed probabilities. The Markov property says that today's weather is all you need to predict tomorrow — you don't need to know last week's weather history.

The transition matrix encodes all possible one-step transitions. Row $i$ tells you the distribution over tomorrow's states given that today's state is $i$. The diagonal entries $a_{ii}$ are self-transition probabilities — how "sticky" each state is.

**Expected duration in a state**: If $a_{ii}$ is the self-transition probability, the expected number of consecutive steps spent in state $i$ follows a geometric distribution with mean $\frac{1}{1 - a_{ii}}$. A bull market with $a_{bull,bull} = 0.95$ lasts on average $\frac{1}{0.05} = 20$ periods.

**Convergence**: Starting from any initial distribution, repeated multiplication by $A$ drives the distribution toward $\pi^*$. The speed of convergence is controlled by the second-largest eigenvalue: the closer it is to 1, the slower the convergence.

## Code Implementation

```python
import numpy as np
from typing import List, Optional
from scipy import stats


class MarkovChain:
    """Discrete-time homogeneous Markov chain."""

    def __init__(
        self,
        transition_matrix: np.ndarray,
        state_names: Optional[List[str]] = None,
    ):
        self.A = np.array(transition_matrix, dtype=float)
        self.n_states = self.A.shape[0]
        self.state_names = state_names or [f"S{i}" for i in range(self.n_states)]

        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("Transition matrix must be square")
        if not np.allclose(self.A.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("Each row must sum to 1")
        if np.any(self.A < 0):
            raise ValueError("All transition probabilities must be non-negative")

    def simulate(
        self,
        n_steps: int,
        initial_state: Optional[int] = None,
        initial_dist: Optional[np.ndarray] = None,
    ) -> List[int]:
        """Simulate a state sequence from the chain."""
        if initial_state is None:
            if initial_dist is None:
                initial_dist = np.ones(self.n_states) / self.n_states
            initial_state = np.random.choice(self.n_states, p=initial_dist)

        states = [initial_state]
        for _ in range(n_steps - 1):
            current = states[-1]
            next_state = np.random.choice(self.n_states, p=self.A[current])
            states.append(next_state)
        return states

    def stationary_distribution(self) -> np.ndarray:
        """Compute stationary distribution via eigenvalue decomposition."""
        eigenvalues, eigenvectors = np.linalg.eig(self.A.T)
        # Find eigenvector for eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.real(eigenvectors[:, idx])
        return pi / pi.sum()

    def convergence_rate(self) -> float:
        """Return the second-largest eigenvalue magnitude (mixing rate)."""
        eigenvalues = np.sort(np.abs(np.linalg.eigvals(self.A)))[::-1]
        return float(eigenvalues[1])

    def expected_state_duration(self) -> np.ndarray:
        """Expected consecutive steps in each state: 1 / (1 - a_ii)."""
        diag = np.diag(self.A)
        return 1.0 / (1.0 - diag + 1e-12)

    def n_step_matrix(self, n: int) -> np.ndarray:
        """Transition matrix for n steps via matrix power."""
        return np.linalg.matrix_power(self.A, n)

    def is_irreducible(self) -> bool:
        """Check irreducibility: all states reachable from all states."""
        n = self.n_states
        # Sum of powers A^1 + ... + A^n covers all paths up to n steps
        reachable = sum(np.linalg.matrix_power(self.A > 0, k) for k in range(1, n + 1))
        return bool(np.all(reachable > 0))

    def is_aperiodic(self) -> bool:
        """Check aperiodicity: any self-transition guarantees aperiodicity."""
        return bool(np.any(np.diag(self.A) > 0))


def estimate_transition_matrix(states: List[int], n_states: int) -> np.ndarray:
    """MLE estimate of transition matrix from observed state sequence."""
    counts = np.zeros((n_states, n_states))
    for i in range(len(states) - 1):
        counts[states[i], states[i + 1]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    # Avoid division by zero for states never visited as source
    row_sums[row_sums == 0] = 1.0
    return counts / row_sums


def bayesian_transition_estimate(
    states: List[int], n_states: int, prior_alpha: float = 1.0
):
    """Bayesian estimate with symmetric Dirichlet prior.

    Returns posterior mean and 95% credible intervals.
    """
    counts = np.zeros((n_states, n_states))
    for i in range(len(states) - 1):
        counts[states[i], states[i + 1]] += 1

    posterior_alpha = counts + prior_alpha
    mean = posterior_alpha / posterior_alpha.sum(axis=1, keepdims=True)

    ci_lower = np.zeros_like(mean)
    ci_upper = np.zeros_like(mean)
    for i in range(n_states):
        alpha_row_sum = posterior_alpha[i].sum()
        for j in range(n_states):
            alpha = posterior_alpha[i, j]
            beta = alpha_row_sum - alpha
            ci_lower[i, j] = stats.beta.ppf(0.025, alpha, beta)
            ci_upper[i, j] = stats.beta.ppf(0.975, alpha, beta)

    return mean, ci_lower, ci_upper


# --- Example: Bull/Bear market regime model ---

if __name__ == "__main__":
    # Two-state market regime model
    A = np.array([
        [0.95, 0.05],   # Bull -> Bull: 95%, Bull -> Bear: 5%
        [0.15, 0.85],   # Bear -> Bull: 15%, Bear -> Bear: 85%
    ])

    mc = MarkovChain(A, state_names=["Bull", "Bear"])

    # Stationary distribution
    pi = mc.stationary_distribution()
    print(f"Stationary distribution: Bull={pi[0]:.2%}, Bear={pi[1]:.2%}")
    # Bull = 75%, Bear = 25%

    # Expected durations
    durations = mc.expected_state_duration()
    print(f"Expected duration — Bull: {durations[0]:.1f}, Bear: {durations[1]:.1f}")
    # Bull: 20 periods, Bear: 6.7 periods

    # Convergence rate
    rate = mc.convergence_rate()
    half_life = np.log(2) / np.log(1 / rate)
    print(f"Second eigenvalue: {rate:.4f}, half-life: {half_life:.1f} steps")

    # Simulate 500 steps
    np.random.seed(42)
    path = mc.simulate(500, initial_state=0)
    print(f"Fraction of time in Bull: {path.count(0) / 500:.2%}")

    # Estimate from simulated data
    A_hat = estimate_transition_matrix(path, n_states=2)
    print(f"Estimated A:\n{A_hat.round(3)}")
```

## Common Pitfalls

**Confusing row-stochastic and column-stochastic conventions.** Some textbooks define $a_{ij} = P(X_t = s_i \mid X_{t+1} = s_j)$ (column convention), which transposes the matrix. Always verify: do the rows sum to 1? If the columns sum to 1, you have a column-stochastic matrix. NumPy's matrix multiply `pi @ A` assumes rows sum to 1.

**Ignoring numerical precision in row-sum validation.** Floating-point arithmetic means `A.sum(axis=1)` rarely equals exactly 1.0. Use `np.allclose(A.sum(axis=1), 1.0, atol=1e-8)` instead of exact equality.

**Assuming irreducibility without checking.** A Markov chain with absorbing states (a state with $a_{ii} = 1$) is not irreducible and has no unique stationary distribution. Always call `mc.is_irreducible()` before computing the stationary distribution.

**Underestimating estimation variance with short sequences.** MLE counts transitions: if a state is visited only 5 times, the estimated row has high variance. Use Bayesian estimation with a Dirichlet prior when transition counts per row are below 50.

**Conflating the stationary distribution with the initial distribution.** The initial distribution $\pi_0$ is where the chain starts. The stationary distribution $\pi^*$ is where it ends up in the long run. They are equal only if the chain starts at stationarity.

## Connections

**Forward to HMMs**: The Markov chain is the "hidden" layer in HMMs. The transition matrix $A$ learned by Baum-Welch is exactly the Markov chain transition matrix, but for a chain we cannot observe directly.

**Stationary distribution in HMMs**: In a trained HMM, the stationary distribution gives the long-run fraction of time spent in each regime. For financial applications it is the unconditional probability of each market regime.

**Chapman-Kolmogorov equation**: $P(X_{t+n} = j \mid X_t = i) = [A^n]_{ij}$. This is used directly in the Forward algorithm and Viterbi algorithm, which maintain probability distributions over states and propagate them forward through time.

**Ergodicity and Baum-Welch**: Baum-Welch is well-posed only if the underlying Markov chain is ergodic. Non-ergodic chains can have multiple stationary distributions, making parameter estimation ambiguous.

## Practice Problems

**Problem 1.** Given the transition matrix:
$$A = \begin{pmatrix} 0.8 & 0.2 \\ 0.3 & 0.7 \end{pmatrix}$$

Compute the stationary distribution analytically using $\pi^* = \pi^* A$ and the normalization $\pi_1^* + \pi_2^* = 1$. Verify your answer using the eigenvalue method.

*Answer*: Solve $0.8\pi_1 + 0.3\pi_2 = \pi_1$ and $\pi_1 + \pi_2 = 1$. This gives $\pi_1 = 0.6$, $\pi_2 = 0.4$.

**Problem 2.** A market has three regimes: Bull (0), Neutral (1), Bear (2) with transition matrix:
$$A = \begin{pmatrix} 0.90 & 0.08 & 0.02 \\ 0.10 & 0.80 & 0.10 \\ 0.02 & 0.08 & 0.90 \end{pmatrix}$$

(a) Compute the expected duration in each regime.
(b) Compute the 10-step transition matrix $A^{10}$.
(c) If today the market is in the Neutral regime, what is the probability it is in the Bull regime in 30 days?

**Problem 3.** Write a function `is_ergodic(A)` that checks both irreducibility and aperiodicity. Test it on the three-regime matrix above and on an absorbing chain where state 2 is absorbing ($a_{22} = 1$).

**Problem 4.** Simulate 1000 steps from the three-regime chain above starting from a uniform distribution. Compute the empirical fraction of time in each state and compare it to the theoretical stationary distribution. How many steps does it take for the empirical distribution to be within 1% of the stationary distribution?

**Problem 5.** Implement Bayesian estimation with a Dirichlet prior. Using the simulated sequence from Problem 4, estimate the transition matrix and compare the posterior mean, MLE, and true matrix. Compute 95% credible intervals for each entry. Which entries have the widest credible intervals and why?

## Further Reading

- **Norris, J.R. (1998)** — *Markov Chains*. Cambridge University Press. The standard rigorous text; Chapter 1 covers all properties discussed here.
- **Grinstead & Snell (2006)** — *Introduction to Probability*. Free online. Chapter 11 on Markov chains uses intuitive examples.
- **hmmlearn documentation** — The `MarkovChain` building blocks directly inform the transition matrix component of `hmmlearn.hmm.GaussianHMM`.
- **Hamilton (1994)** — *Time Series Analysis*, Chapter 22. The original financial application of Markov-switching models that motivated much of the HMM work in finance.
