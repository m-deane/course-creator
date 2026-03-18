# Transition Matrices and State Dynamics

## In Brief

The transition matrix is the complete mathematical description of how a Markov chain moves between states. Every structural property of the chain — how long it stays in each state, where it ends up in the long run, how quickly it forgets its starting point — is encoded in this single $K \times K$ matrix. In HMMs, the transition matrix is the $A$ parameter that Baum-Welch estimates and that the Forward and Viterbi algorithms propagate forward through time.

## Key Insight

The Chapman-Kolmogorov equation $P(X_{t+n} = j \mid X_t = i) = [A^n]_{ij}$ means that all multi-step transition probabilities are obtained by matrix exponentiation. This is computationally efficient and mathematically elegant: the entire long-run behavior of the chain is encoded in the eigenvectors of $A$. The stationary distribution is the left eigenvector for eigenvalue 1; the convergence rate is determined by the second-largest eigenvalue.

## Formal Definition

For a Markov chain with $K$ states, the **transition matrix** $A \in \mathbb{R}^{K \times K}$ has entries:
$$a_{ij} = P(X_{t+1} = j \mid X_t = i), \quad i, j \in \{1, \ldots, K\}$$

**Constraints:**
- Non-negativity: $a_{ij} \geq 0$
- Row-stochastic: $\sum_{j=1}^{K} a_{ij} = 1$ for all $i$

### Chapman-Kolmogorov Equation

The $n$-step transition probability is:
$$P(X_{t+n} = j \mid X_t = i) = [A^n]_{ij}$$

### Stationary Distribution

$\pi^*$ is a stationary distribution if $\pi^* A = \pi^*$, i.e., $\pi^*$ is the left eigenvector of $A$ for eigenvalue 1.

### Expected Hitting Time

The expected number of steps to reach state $j$ from state $i$ satisfies:
$$h_i = 1 + \sum_{k \neq j} a_{ik} h_k, \quad h_j = 0$$

This is a linear system: $(I - A_{\text{reduced}}) h = \mathbf{1}$ where $A_{\text{reduced}}$ is $A$ with the target state's row and column removed.

### State Classification

- **Transient**: state may not be revisited
- **Recurrent**: state is always eventually revisited
- **Absorbing**: $a_{ii} = 1$ (once entered, never left)

## Intuitive Explanation

Think of the transition matrix as a map that tells each state what to do at the next time step. Row $i$ is the instruction sheet for state $i$: entry $a_{ij}$ is the probability of moving to state $j$.

**Persistence vs. switching**: High diagonal values (close to 1) mean the chain is "sticky" — it tends to stay in the same state. A financial regime model with $a_{bull,bull} = 0.95$ says that once the market enters a bull regime, it stays there 95% of the time, producing an expected duration of 20 periods. Low diagonal values mean rapid switching.

**Long-run behavior**: Multiply any initial distribution $\pi_0$ by $A$ repeatedly. The distribution converges to $\pi^*$ regardless of the starting point (for ergodic chains). This is the statistical equilibrium: the market spends $\pi^*_{\text{bull}}$ fraction of time in bull and $\pi^*_{\text{bear}}$ fraction in bear, unconditionally.

**Matrix power interpretation**: $[A^{30}]_{ij}$ is the probability of being in state $j$ exactly 30 steps after starting in state $i$. As $n \to \infty$, all rows of $A^n$ converge to $\pi^*$: the starting state is eventually forgotten.

## Code Implementation

```python
import numpy as np
from scipy import stats
from typing import List, Optional, Tuple


def create_transition_matrix(n_states: int, style: str = "persistent",
                              persistence: float = 0.9) -> np.ndarray:
    """Create a transition matrix with a specified structural style.

    Styles:
    - 'persistent': high diagonal (sticky regimes), finance default
    - 'uniform': all transitions equally likely
    - 'random': randomly normalized matrix
    - 'cyclic': states rotate in order with high probability
    - 'absorbing': last state is absorbing (terminal)
    """
    if style == "persistent":
        A = np.full((n_states, n_states), (1 - persistence) / (n_states - 1))
        np.fill_diagonal(A, persistence)
    elif style == "uniform":
        A = np.ones((n_states, n_states)) / n_states
    elif style == "random":
        A = np.random.rand(n_states, n_states)
        A = A / A.sum(axis=1, keepdims=True)
    elif style == "cyclic":
        A = np.zeros((n_states, n_states))
        for i in range(n_states):
            A[i, (i + 1) % n_states] = 1 - persistence
            A[i, i] = persistence
    elif style == "absorbing":
        A = np.random.rand(n_states, n_states)
        A = A / A.sum(axis=1, keepdims=True)
        A[-1, :] = 0.0
        A[-1, -1] = 1.0
    else:
        raise ValueError(f"Unknown style: {style}")
    return A


def stationary_distribution(A: np.ndarray, method: str = "eigenvalue") -> np.ndarray:
    """Compute the stationary distribution by three methods.

    Methods: 'eigenvalue', 'power_iteration', 'linear_system'
    """
    n = A.shape[0]
    if method == "eigenvalue":
        eigenvalues, eigenvectors = np.linalg.eig(A.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.real(eigenvectors[:, idx])
        return pi / pi.sum()

    elif method == "power_iteration":
        pi = np.ones(n) / n
        for _ in range(10_000):
            pi_new = pi @ A
            if np.allclose(pi, pi_new, atol=1e-12):
                break
            pi = pi_new
        return pi

    elif method == "linear_system":
        # Solve (A^T - I) pi = 0 with normalization constraint
        mat = A.T - np.eye(n)
        # Replace last equation with normalization: sum(pi) = 1
        mat[-1, :] = 1.0
        rhs = np.zeros(n)
        rhs[-1] = 1.0
        pi = np.linalg.solve(mat, rhs)
        return pi / pi.sum()

    raise ValueError(f"Unknown method: {method}")


def expected_hitting_times(A: np.ndarray, target_state: int) -> np.ndarray:
    """Expected steps to reach target_state from each other state.

    Solves (I - A_reduced) h = 1 where A_reduced removes the target row/col.
    """
    n = A.shape[0]
    other_states = [i for i in range(n) if i != target_state]
    A_reduced = A[np.ix_(other_states, other_states)]
    b = np.ones(len(other_states))
    h_reduced = np.linalg.solve(np.eye(len(other_states)) - A_reduced, b)

    h = np.zeros(n)
    for idx, state in enumerate(other_states):
        h[state] = h_reduced[idx]
    # h[target_state] = 0 by definition
    return h


def communicating_classes(A: np.ndarray) -> List[List[int]]:
    """Find communicating classes (strongly connected components).

    Uses reachability: states i and j communicate if j is reachable from i
    AND i is reachable from j.
    """
    n = A.shape[0]
    max_steps = n

    # Compute reachability matrix
    reach = np.zeros((n, n), dtype=bool)
    A_power = np.eye(n)
    for _ in range(max_steps):
        A_power = A_power @ A
        reach |= A_power > 1e-10

    # States i and j communicate if each can reach the other
    communicate = reach & reach.T

    # Group into classes
    visited = [False] * n
    classes = []
    for i in range(n):
        if not visited[i]:
            cls = [j for j in range(n) if communicate[i, j]]
            for j in cls:
                visited[j] = True
            classes.append(sorted(cls))
    return classes


def convergence_rate(A: np.ndarray) -> Tuple[float, float]:
    """Return second-largest eigenvalue and implied mixing half-life."""
    eigenvalues = np.sort(np.abs(np.linalg.eigvals(A)))[::-1]
    lambda2 = eigenvalues[1]
    if lambda2 >= 1.0:
        half_life = float("inf")
    else:
        half_life = np.log(2) / np.log(1 / lambda2)
    return float(lambda2), half_life


def verify_stationary(A: np.ndarray, pi: np.ndarray, tol: float = 1e-8) -> bool:
    """Check that pi A = pi (stationary property)."""
    residual = np.max(np.abs(pi @ A - pi))
    return bool(residual < tol)


# --- Market regime example ---

if __name__ == "__main__":
    # Three-state market regime transition matrix
    A = np.array([
        [0.90, 0.08, 0.02],  # Bull -> Bull, Neutral, Bear
        [0.10, 0.80, 0.10],  # Neutral -> Bull, Neutral, Bear
        [0.02, 0.08, 0.90],  # Bear -> Bull, Neutral, Bear
    ])

    print("=== Market Regime Transition Matrix ===")
    print(A)

    # Stationary distribution — three methods should agree
    for method in ["eigenvalue", "power_iteration", "linear_system"]:
        pi = stationary_distribution(A, method=method)
        ok = verify_stationary(A, pi)
        print(f"{method}: pi={pi.round(4)}, verified={ok}")

    # Expected durations
    diag = np.diag(A)
    durations = 1.0 / (1.0 - diag)
    regime_names = ["Bull", "Neutral", "Bear"]
    for name, d in zip(regime_names, durations):
        print(f"Expected duration in {name}: {d:.1f} periods")

    # Expected hitting times to Bear regime
    h = expected_hitting_times(A, target_state=2)
    for name, ht in zip(regime_names, h):
        if ht > 0:
            print(f"Expected steps from {name} to Bear: {ht:.1f}")

    # Convergence rate
    lambda2, half_life = convergence_rate(A)
    print(f"\nSecond eigenvalue: {lambda2:.4f}")
    print(f"Mixing half-life: {half_life:.1f} steps")

    # 10-step transition matrix
    A10 = np.linalg.matrix_power(A, 10)
    print("\n10-step transitions from Bull state:")
    for name, prob in zip(regime_names, A10[0]):
        print(f"  P(Bull -> {name} in 10 steps) = {prob:.3f}")
```

## Common Pitfalls

**Using column-stochastic convention by mistake.** Some implementations (especially from physics or MATLAB backgrounds) define $A_{ij}$ as the probability from $j$ to $i$ (columns sum to 1). In the row-stochastic convention used by hmmlearn and this course, $A_{ij}$ is from $i$ to $j$ (rows sum to 1), and updating a distribution uses `pi @ A` not `A @ pi`.

**Forgetting the row-stochasticity check after manual construction.** When building a custom transition matrix (e.g., for a sticky HMM), always verify `np.allclose(A.sum(axis=1), 1.0)` after any modification. Rows that don't sum to 1 will silently corrupt forward pass probabilities.

**Computing the stationary distribution when the chain is not ergodic.** If the chain has absorbing states, the eigenvalue method still returns a vector, but it corresponds to the absorbing state having all the weight. Check ergodicity first with `communicating_classes(A)`.

**Using `np.linalg.matrix_power` for large n.** Matrix power is stable but can be slow for large matrices and large $n$. For long-horizon predictions, compute the stationary distribution instead — for sufficiently large $n$, $A^n$ converges to a matrix where every row equals $\pi^*$.

**Misinterpreting the self-transition probability as the persistence probability.** The expected duration in state $i$ is $1/(1 - a_{ii})$, not $a_{ii}$. A self-transition of 0.95 gives duration 20, not 0.95.

## Connections

**HMM transition matrix**: In an HMM, $A$ is exactly the Markov chain transition matrix for the hidden state layer. The Forward algorithm multiplies the current alpha vector by $A$ to propagate state probability distributions forward through time: `alpha[t] = (alpha[t-1] @ A) * emission_probs[t]`.

**Baum-Welch M-step for $A$**: The M-step updates $A$ using expected transition counts: $\hat{a}_{ij} = \frac{\sum_t \xi_t(i,j)}{\sum_t \gamma_t(i)}$. The result is always row-stochastic by construction.

**Viterbi algorithm**: The Viterbi forward pass uses `max` instead of `sum` when propagating through $A$: `delta[t,j] = max_i(delta[t-1,i] * A[i,j]) * emission_probs[t,j]`. The argmax defines the backtracking pointer.

**Sticky HMM (Module 05)**: The stickiness modification $a_{ii}^* = \kappa + (1-\kappa) a_{ii}$ directly alters the diagonal of $A$, increasing expected regime duration. All other properties (stationary distribution, convergence rate) can be recomputed from the modified matrix.

## Practice Problems

**Problem 1.** Given:
$$A = \begin{pmatrix} 0.7 & 0.3 \\ 0.4 & 0.6 \end{pmatrix}$$

(a) Compute $A^2$, $A^5$, $A^{20}$ by matrix multiplication.
(b) What do all rows of $A^{20}$ converge to? Verify against the stationary distribution.
(c) Compute the mixing half-life and interpret it.

**Problem 2.** Construct a three-state persistent transition matrix where:
- Bull: 90% self-transition, 8% to Neutral, 2% to Bear
- Neutral: 10% to Bull, 80% self-transition, 10% to Bear
- Bear: 2% to Bull, 8% to Neutral, 90% self-transition

(a) Compute the stationary distribution using all three methods (eigenvalue, power iteration, linear system) and verify they agree.
(b) If today the market is in the Neutral regime, what is the probability it is in the Bull regime in 30 trading days?
(c) From the Bear regime, what is the expected number of days until the market first enters the Bull regime?

**Problem 3.** The following matrix represents a chain with an absorbing state:
$$A = \begin{pmatrix} 0.5 & 0.3 & 0.2 \\ 0.2 & 0.5 & 0.3 \\ 0.0 & 0.0 & 1.0 \end{pmatrix}$$

(a) Identify the absorbing state.
(b) What happens when you run `communicating_classes(A)`?
(c) Why does the standard stationary distribution computation fail here?
(d) Compute the expected hitting time to the absorbing state from each non-absorbing state.

**Problem 4.** Implement a function `create_block_transition_matrix(group_probs, within_group_A, between_group_prob)` that creates a transition matrix for $n$ groups. Within each group, transitions follow `within_group_A` (scaled). Between groups, all states have equal probability of transitioning to any state in any other group (scaled by `between_group_prob`). Test that your matrix is row-stochastic.

**Problem 5.** Given 500 simulated steps from the three-state persistent chain (Bull/Neutral/Bear), estimate the transition matrix using MLE. Compare the estimated matrix to the true matrix entry by entry. Which transitions have the highest estimation error? Is this related to how often each state is visited?

## Further Reading

- **Levin, Peres, Wilmer (2017)** — *Markov Chains and Mixing Times*. The definitive text on convergence rates, mixing times, and the spectral gap.
- **Hamilton (1989)** — "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica*, 57(2). The seminal paper introducing Markov-switching models in econometrics.
- **Norris (1998)** — *Markov Chains*, Cambridge. Chapters 1–3 cover hitting times, communicating classes, and stationary distributions rigorously.
- **hmmlearn source** — `hmmlearn/hmm.py`. The `_do_mstep` method shows exactly how the M-step updates the transition matrix from accumulated gamma and xi statistics.
