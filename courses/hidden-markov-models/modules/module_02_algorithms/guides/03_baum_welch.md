# Baum-Welch Algorithm: Learning HMM Parameters

> **Reading time:** ~14 min | **Module:** Module 2: Algorithms | **Prerequisites:** Modules 0-1

<div class="callout-key">

**Key Concept Summary:** The Baum-Welch algorithm learns HMM parameters (transition probabilities, emission distributions, initial state distribution) from observed data using the Expectation-Maximization (EM) framework. It iteratively improves parameter estimates to maximize the likelihood of observed sequences.

</div>

## In Brief

The Baum-Welch algorithm learns HMM parameters (transition probabilities, emission distributions, initial state distribution) from observed data using the Expectation-Maximization (EM) framework. It iteratively improves parameter estimates to maximize the likelihood of observed sequences.

<div class="callout-insight">

**Insight:** We can't directly count transitions between hidden states (they're hidden!), but we can compute expected counts using current parameter estimates. EM alternates between computing these expected counts (E-step) and updating parameters to maximize likelihood (M-step).

</div>
<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## Intuitive Explanation

## Formal Definition

### The Learning Problem

**Given:**
- Observation sequence $O = (o_1, o_2, ..., o_T)$
- Model structure (number of states $K$)

**Find:**
- Parameters $\lambda^* = (\pi^*, A^*, B^*)$ that maximize $P(O|\lambda)$

$$\lambda^* = \argmax_{\lambda} P(O|\lambda)$$

### EM Framework for HMMs

Baum-Welch is an application of the Expectation-Maximization algorithm:

**E-Step:** Compute expected sufficient statistics given current parameters $\lambda^{(n)}$

**M-Step:** Update parameters to maximize expected log-likelihood

**Iterate** until convergence: $|\log P(O|\lambda^{(n+1)}) - \log P(O|\lambda^{(n)})| < \epsilon$

### Key Quantities

#### Forward Variable

$$\alpha_t(i) = P(o_1, ..., o_t, q_t = s_i | \lambda)$$

Probability of partial observation sequence up to time $t$ and being in state $i$ at time $t$.

#### Backward Variable

$$\beta_t(i) = P(o_{t+1}, ..., o_T | q_t = s_i, \lambda)$$

Probability of partial observation sequence from $t+1$ to end, given state $i$ at time $t$.

#### State Occupation Probability

$$\gamma_t(i) = P(q_t = s_i | O, \lambda) = \frac{\alpha_t(i) \beta_t(i)}{\sum_{j=1}^K \alpha_t(j) \beta_t(j)}$$

Probability of being in state $i$ at time $t$ given the full observation sequence.

#### Transition Probability

$$\xi_t(i,j) = P(q_t = s_i, q_{t+1} = s_j | O, \lambda) = \frac{\alpha_t(i) a_{ij} b_j(o_{t+1}) \beta_{t+1}(j)}{\sum_{i'=1}^K \sum_{j'=1}^K \alpha_t(i') a_{i'j'} b_{j'}(o_{t+1}) \beta_{t+1}(j')}$$

Probability of transition from state $i$ to state $j$ at time $t$ given full observation sequence.

### Parameter Updates

**Initial State Distribution:**
$$\hat{\pi}_i = \gamma_1(i)$$

Expected probability of starting in state $i$.

**Transition Matrix:**
$$\hat{a}_{ij} = \frac{\sum_{t=1}^{T-1} \xi_t(i,j)}{\sum_{t=1}^{T-1} \gamma_t(i)}$$

Expected number of transitions from $i$ to $j$ divided by expected number of times in state $i$.

**Emission Probabilities (Discrete):**
$$\hat{b}_i(v_k) = \frac{\sum_{t=1, o_t=v_k}^{T} \gamma_t(i)}{\sum_{t=1}^{T} \gamma_t(i)}$$

Expected number of times in state $i$ observing symbol $v_k$ divided by expected time in state $i$.

**Emission Parameters (Gaussian):**

Mean:
$$\hat{\mu}_i = \frac{\sum_{t=1}^{T} \gamma_t(i) \cdot o_t}{\sum_{t=1}^{T} \gamma_t(i)}$$

Variance:
$$\hat{\sigma}_i^2 = \frac{\sum_{t=1}^{T} \gamma_t(i) \cdot (o_t - \hat{\mu}_i)^2}{\sum_{t=1}^{T} \gamma_t(i)}$$

### The Learning Challenge

Imagine you're trying to learn market regime parameters but can't see the regimes directly:

```
Hidden States:  ???  →  ???  →  ???  →  ???
                 ↓      ↓      ↓      ↓
Observations:   +2%    -1%    +3%    -2%
```

**The Problem:**
- To count "Bull → Bear" transitions, you need to know when you're in Bull vs Bear (but you don't!)
- To estimate mean return in Bull state, you need to know which observations came from Bull (but you don't!)

**The Solution:**
- Use current parameter guesses to compute *expected* counts
- Update parameters based on these expected counts
- Repeat until parameters stabilize

### The EM Intuition

**E-Step:** "If the parameters were as I currently estimate, how much time would we expect to spend in each state, and how many transitions would we expect?"

**M-Step:** "Given these expected counts, what parameters would maximize likelihood?"

**Example:**

Iteration 1:
- Guess: Bull state has mean return = 1%, Bear state has mean = -1%
- E-step: Compute probabilities—when returns are positive, probably in Bull state
- M-step: Recompute mean of Bull using weighted average (weights = probability of Bull at each time)
- New estimate: Bull mean = 1.2%, Bear mean = -0.8%

Iteration 2:
- Use new parameters to recompute state probabilities
- Update parameters again
- ...continue until convergence

### Why It Works

The EM algorithm has a beautiful guarantee:

$$\log P(O | \lambda^{(n+1)}) \geq \log P(O | \lambda^{(n)})$$

Each iteration increases (or keeps constant) the likelihood. We climb uphill toward a local maximum.

**Caveat:** Only guaranteed to find *local* maximum, not necessarily global. Solution: Run with multiple random initializations.

## Code Implementation

### Complete Baum-Welch Implementation

```python
import numpy as np
from typing import Tuple, List

class BaumWelchHMM:
    """
    Hidden Markov Model with Baum-Welch (EM) training.
    """

    def __init__(self, n_states: int, n_observations: int = None):
        """
        Args:
            n_states: Number of hidden states
            n_observations: Number of discrete observation symbols (None for continuous)
        """
        self.K = n_states
        self.M = n_observations

        # Initialize parameters randomly
        self.pi = np.ones(self.K) / self.K
        self.A = np.random.rand(self.K, self.K)
        self.A = self.A / self.A.sum(axis=1, keepdims=True)

        if n_observations is not None:
            # Discrete emissions
            self.B = np.random.rand(self.K, self.M)
            self.B = self.B / self.B.sum(axis=1, keepdims=True)
            self.continuous = False
        else:
            # Gaussian emissions
            self.means = np.random.randn(self.K)
            self.stds = np.ones(self.K)
            self.continuous = True

    def forward(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Forward algorithm.

        Returns:
            alpha: Forward probabilities (T x K)
            log_likelihood: log P(O|λ)
        """
        T = len(observations)
        alpha = np.zeros((T, self.K))

        # Initialization
        for i in range(self.K):
            alpha[0, i] = self.pi[i] * self._emission_prob(observations[0], i)

        # Recursion
        for t in range(1, T):
            for j in range(self.K):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * \
                              self._emission_prob(observations[t], j)

        # Termination
        log_likelihood = np.log(np.sum(alpha[-1]) + 1e-10)

        return alpha, log_likelihood

    def backward(self, observations: np.ndarray) -> np.ndarray:
        """
        Backward algorithm.

        Returns:
            beta: Backward probabilities (T x K)
        """
        T = len(observations)
        beta = np.zeros((T, self.K))

        # Initialization
        beta[-1, :] = 1.0

        # Recursion (backward)
        for t in range(T-2, -1, -1):
            for i in range(self.K):
                beta[t, i] = np.sum(
                    self.A[i, :] *
                    np.array([self._emission_prob(observations[t+1], j) for j in range(self.K)]) *
                    beta[t+1, :]
                )

        return beta

    def _emission_prob(self, observation, state: int) -> float:
        """
        Compute emission probability b_state(observation).
        """
        if self.continuous:
            # Gaussian emission
            return (1.0 / (self.stds[state] * np.sqrt(2*np.pi))) * \
                   np.exp(-0.5 * ((observation - self.means[state]) / self.stds[state])**2)
        else:
            # Discrete emission
            return self.B[state, observation]

    def e_step(
        self,
        observations: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        E-step: Compute expected sufficient statistics.

        Returns:
            gamma: State occupation probabilities (T x K)
            xi: State transition probabilities (T-1 x K x K)
        """
        T = len(observations)

        # Compute gamma_t(i) = P(q_t = i | O, λ)
        gamma = alpha * beta
        gamma = gamma / (gamma.sum(axis=1, keepdims=True) + 1e-10)

        # Compute xi_t(i,j) = P(q_t = i, q_{t+1} = j | O, λ)
        xi = np.zeros((T-1, self.K, self.K))

        for t in range(T-1):
            denominator = 0.0
            for i in range(self.K):
                for j in range(self.K):
                    xi[t, i, j] = alpha[t, i] * \
                                  self.A[i, j] * \
                                  self._emission_prob(observations[t+1], j) * \
                                  beta[t+1, j]
                    denominator += xi[t, i, j]

            xi[t] = xi[t] / (denominator + 1e-10)

        return gamma, xi

    def m_step(
        self,
        observations: np.ndarray,
        gamma: np.ndarray,
        xi: np.ndarray
    ):
        """
        M-step: Update parameters to maximize expected log-likelihood.
        """
        T = len(observations)

        # Update initial state distribution
        self.pi = gamma[0]

        # Update transition matrix
        for i in range(self.K):
            for j in range(self.K):
                numerator = np.sum(xi[:, i, j])
                denominator = np.sum(gamma[:-1, i]) + 1e-10
                self.A[i, j] = numerator / denominator

        # Normalize rows
        self.A = self.A / (self.A.sum(axis=1, keepdims=True) + 1e-10)

        # Update emission parameters
        if self.continuous:
            # Update Gaussian parameters
            for i in range(self.K):
                # Weighted mean
                self.means[i] = np.sum(gamma[:, i] * observations) / \
                                (np.sum(gamma[:, i]) + 1e-10)

                # Weighted variance
                diff = observations - self.means[i]
                self.stds[i] = np.sqrt(
                    np.sum(gamma[:, i] * diff**2) / (np.sum(gamma[:, i]) + 1e-10)
                )

        else:
            # Update discrete emission probabilities
            for i in range(self.K):
                for k in range(self.M):
                    numerator = np.sum(gamma[observations == k, i])
                    denominator = np.sum(gamma[:, i]) + 1e-10
                    self.B[i, k] = numerator / denominator

            # Normalize
            self.B = self.B / (self.B.sum(axis=1, keepdims=True) + 1e-10)

    def fit(
        self,
        observations: np.ndarray,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
        verbose: bool = True
    ) -> List[float]:
        """
        Fit HMM parameters using Baum-Welch algorithm.

        Args:
            observations: Observation sequence (T,)
            max_iterations: Maximum EM iterations
            tolerance: Convergence threshold for log-likelihood
            verbose: Print progress

        Returns:
            log_likelihoods: Log-likelihood at each iteration
        """
        log_likelihoods = []

        for iteration in range(max_iterations):
            # E-step: Forward-Backward
            alpha, log_likelihood = self.forward(observations)
            beta = self.backward(observations)
            gamma, xi = self.e_step(observations, alpha, beta)

            # M-step: Update parameters
            self.m_step(observations, gamma, xi)

            log_likelihoods.append(log_likelihood)

            if verbose and (iteration % 10 == 0 or iteration == max_iterations - 1):
                print(f"Iteration {iteration:3d}: log P(O|λ) = {log_likelihood:.4f}")

            # Check convergence
            if iteration > 0:
                improvement = log_likelihoods[-1] - log_likelihoods[-2]
                if improvement < tolerance:
                    if verbose:
                        print(f"Converged after {iteration+1} iterations")
                    break

        return log_likelihoods

    def predict_states(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict most likely state sequence using Viterbi algorithm.
        """
        T = len(observations)
        delta = np.zeros((T, self.K))
        psi = np.zeros((T, self.K), dtype=int)

        # Initialization
        for i in range(self.K):
            delta[0, i] = self.pi[i] * self._emission_prob(observations[0], i)

        # Recursion
        for t in range(1, T):
            for j in range(self.K):
                probs = delta[t-1] * self.A[:, j]
                psi[t, j] = np.argmax(probs)
                delta[t, j] = np.max(probs) * self._emission_prob(observations[t], j)

        # Backtracking
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])

        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]

        return states


# Example: Learn market regime parameters from data
np.random.seed(42)

# Generate synthetic data from known HMM
true_hmm = BaumWelchHMM(n_states=2, n_observations=None)
true_hmm.means = np.array([0.001, -0.002])  # Bull, Bear
true_hmm.stds = np.array([0.01, 0.02])
true_hmm.A = np.array([[0.95, 0.05],
                       [0.10, 0.90]])
true_hmm.pi = np.array([0.7, 0.3])

# Simulate observations
def simulate_hmm(hmm, T):
    states = [np.random.choice(hmm.K, p=hmm.pi)]
    observations = [np.random.normal(hmm.means[states[0]], hmm.stds[states[0]])]

    for _ in range(T-1):
        states.append(np.random.choice(hmm.K, p=hmm.A[states[-1]]))
        observations.append(np.random.normal(hmm.means[states[-1]], hmm.stds[states[-1]]))

    return np.array(states), np.array(observations)

true_states, observations = simulate_hmm(true_hmm, 500)

print("=" * 60)
print("TRUE PARAMETERS:")
print("=" * 60)
print(f"Means: {true_hmm.means}")
print(f"Stds: {true_hmm.stds}")
print(f"Transition matrix:\n{true_hmm.A}")
print(f"Initial distribution: {true_hmm.pi}")

# Learn parameters using Baum-Welch
print("\n" + "=" * 60)
print("LEARNING PARAMETERS WITH BAUM-WELCH:")
print("=" * 60)

learned_hmm = BaumWelchHMM(n_states=2, n_observations=None)
log_likelihoods = learned_hmm.fit(observations, max_iterations=100, verbose=True)

print("\n" + "=" * 60)
print("LEARNED PARAMETERS:")
print("=" * 60)
print(f"Means: {learned_hmm.means}")
print(f"Stds: {learned_hmm.stds}")
print(f"Transition matrix:\n{learned_hmm.A}")
print(f"Initial distribution: {learned_hmm.pi}")

# Predict states
predicted_states = learned_hmm.predict_states(observations)
accuracy = np.mean(predicted_states == true_states)
print(f"\nState prediction accuracy: {accuracy:.2%}")
```

### Visualization


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Plot 1: Log-likelihood convergence
ax1 = axes[0]
ax1.plot(log_likelihoods, 'o-', linewidth=2)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Log P(O|λ)')
ax1.set_title('Baum-Welch Convergence')
ax1.grid(True, alpha=0.3)

# Plot 2: True vs predicted states
ax2 = axes[1]
ax2.plot(true_states, label='True States', alpha=0.7, linewidth=2)
ax2.plot(predicted_states, label='Predicted States', alpha=0.7, linewidth=2, linestyle='--')
ax2.set_xlabel('Time')
ax2.set_ylabel('State')
ax2.set_title('State Sequence: True vs Predicted')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Observations with predicted states
ax3 = axes[2]
for state in range(2):
    mask = predicted_states == state
    ax3.scatter(np.where(mask)[0], observations[mask],
               label=f'Predicted State {state}', alpha=0.5, s=10)
ax3.set_xlabel('Time')
ax3.set_ylabel('Observation (Return)')
ax3.set_title('Observations Colored by Predicted State')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.axhline(0, color='black', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()
```

</div>
</div>

## Common Pitfalls

1. **Local Maxima**
   - EM only guarantees convergence to local maximum
   - Solution: Run with multiple random initializations
   - Select model with highest final likelihood


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">fit_with_multiple_initializations.py</span>
</div>

```python
def fit_with_multiple_initializations(observations, n_states, n_inits=10):
    """Run Baum-Welch with multiple random starts."""
    best_hmm = None
    best_likelihood = -np.inf

    for i in range(n_inits):
        hmm = BaumWelchHMM(n_states, n_observations=None)
        log_liks = hmm.fit(observations, verbose=False)

        if log_liks[-1] > best_likelihood:
            best_likelihood = log_liks[-1]
            best_hmm = hmm

        print(f"Init {i+1}/{n_inits}: final log-likelihood = {log_liks[-1]:.4f}")

    return best_hmm
```

</div>
</div>

2. **Numerical Underflow**
   - Forward/backward probabilities become very small
   - Multiplying many small numbers → underflow to zero
   - Solution: Use log-space computation or scaling


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">forward_scaled.py</span>
</div>

```python
def forward_scaled(self, observations):
    """Forward algorithm with scaling to prevent underflow."""
    T = len(observations)
    alpha = np.zeros((T, self.K))
    scales = np.zeros(T)

    # Initialization
    alpha[0] = self.pi * np.array([self._emission_prob(observations[0], i) for i in range(self.K)])
    scales[0] = np.sum(alpha[0])
    alpha[0] = alpha[0] / scales[0]

    # Recursion with scaling
    for t in range(1, T):
        alpha[t] = np.array([
            np.sum(alpha[t-1] * self.A[:, j]) * self._emission_prob(observations[t], j)
            for j in range(self.K)
        ])
        scales[t] = np.sum(alpha[t])
        alpha[t] = alpha[t] / scales[t]

    log_likelihood = np.sum(np.log(scales))
    return alpha, scales, log_likelihood
```

</div>
</div>

3. **Overfitting with Too Many States**
   - More states always increase training likelihood
   - But may not generalize to new data
   - Solution: Use BIC/AIC for model selection


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">compute_bic.py</span>
</div>

```python
def compute_bic(log_likelihood, n_params, n_observations):
    """
    Bayesian Information Criterion: lower is better.

    BIC = -2 * log L + k * log(n)
    where k = number of parameters, n = number of observations
    """
    return -2 * log_likelihood + n_params * np.log(n_observations)

def select_num_states(observations, max_states=5):
    """Select number of states using BIC."""
    bics = []

    for K in range(2, max_states + 1):
        hmm = BaumWelchHMM(K, n_observations=None)
        log_liks = hmm.fit(observations, verbose=False)

        # Count parameters
        n_params = K - 1  # pi (K-1 free params)
        n_params += K * (K - 1)  # A (K rows, K-1 free per row)
        n_params += 2 * K  # Gaussian: mean and std for each state

        bic = compute_bic(log_liks[-1], n_params, len(observations))
        bics.append(bic)

        print(f"K={K}: BIC={bic:.2f}")

    best_K = np.argmin(bics) + 2
    print(f"\nBest number of states: {best_K}")
    return best_K
```

</div>
</div>

4. **Poor Initialization**
   - Random initialization may start far from good solution
   - Better: K-means clustering of observations for initial state assignment


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">initialize_with_kmeans.py</span>

```python
from sklearn.cluster import KMeans

def initialize_with_kmeans(observations, n_states):
    """Initialize HMM parameters using K-means clustering."""
    hmm = BaumWelchHMM(n_states, n_observations=None)

    # Cluster observations
    kmeans = KMeans(n_clusters=n_states, random_state=42)
    labels = kmeans.fit_predict(observations.reshape(-1, 1))

    # Initialize means and stds from clusters
    for i in range(n_states):
        cluster_obs = observations[labels == i]
        if len(cluster_obs) > 0:
            hmm.means[i] = np.mean(cluster_obs)
            hmm.stds[i] = np.std(cluster_obs)

    # Initialize transition matrix from cluster sequence
    trans_counts = np.zeros((n_states, n_states))
    for t in range(len(labels) - 1):
        trans_counts[labels[t], labels[t+1]] += 1

    hmm.A = trans_counts / (trans_counts.sum(axis=1, keepdims=True) + 1e-10)

    # Initialize pi from first observation cluster
    pi_counts = np.bincount(labels[:min(100, len(labels))], minlength=n_states)
    hmm.pi = pi_counts / pi_counts.sum()

    return hmm
```


5. **Ignoring Convergence Criteria**
   - Stopping too early: Parameters haven't stabilized
   - Stopping too late: Wasting computation
   - Monitor both likelihood improvement and parameter changes

## Connections

### Builds on:
- **Forward algorithm**: Computes α for E-step
- **Backward algorithm**: Computes β for E-step
- **EM algorithm**: General framework for latent variable models

### Leads to:
- **Model selection**: BIC/AIC for choosing number of states
- **Online learning**: Updating parameters as new data arrives
- **Structured HMMs**: Constraining transition/emission patterns

### Related to:
- **K-means**: Hard clustering (HMM is soft clustering over time)
- **Mixture models**: HMM without temporal structure
- **Kalman filter EM**: Continuous state-space version

## Practice Problems

1. **Parameter Counting**

   For a K-state HMM with Gaussian emissions, how many free parameters are there?

   <details>
   <summary>Solution</summary>

   - Initial distribution π: K-1 free parameters (must sum to 1)
   - Transition matrix A: K×(K-1) free parameters (each row sums to 1)
   - Gaussian means: K parameters
   - Gaussian standard deviations: K parameters

   Total: (K-1) + K(K-1) + K + K = K² + K - 1

   For K=2: 2² + 2 - 1 = 5 parameters
   For K=3: 3² + 3 - 1 = 11 parameters
   </details>

2. **EM Guarantee**

   True or False: The Baum-Welch algorithm is guaranteed to find the global maximum likelihood parameters.

   <details>
   <summary>Answer</summary>

   **False.** Baum-Welch (EM) only guarantees convergence to a *local* maximum. The likelihood surface for HMMs is typically non-convex with multiple local maxima.

   This is why we:
   - Run with multiple random initializations
   - Use smart initialization (e.g., K-means)
   - Select the run with highest final likelihood
   </details>

3. **Convergence**

   You run Baum-Welch for 100 iterations. The log-likelihood increases from -1000 to -500. Should you keep running?

   <details>
   <summary>Answer</summary>

   Not necessarily. Check:
   - Is improvement still significant? (e.g., last 10 iterations improve < 0.001)
   - Have parameters stabilized? (e.g., ||A_new - A_old|| < threshold)

   If yes to both, you've likely converged. If likelihood still improving substantially, continue.

   Plot log-likelihood vs iteration—look for plateau.
   </details>

4. **Implementation**

   In the M-step, why do we divide by Σₜ γₜ(i) when updating parameters?

   <details>
   <summary>Answer</summary>

   γₜ(i) = P(state i at time t | observations)

   Σₜ γₜ(i) = expected total time spent in state i

   For mean:
   - Numerator: Σₜ γₜ(i)·oₜ = weighted sum of observations (weighted by probability of being in state i)
   - Denominator: Σₜ γₜ(i) = total weight

   Result: Weighted average of observations assigned to state i

   This is the maximum likelihood estimate when we have "soft" assignments (probabilities) rather than hard assignments.
   </details>

<div class="callout-insight">

**Insight:** Understanding baum-welch algorithm is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.


## Further Reading

### Foundational Papers

1. **Baum et al. (1970)** - "A Maximization Technique Occurring in the Statistical Analysis of Probabilistic Functions of Markov Chains"
   - Original Baum-Welch paper
   - Proves convergence properties
   - Annals of Mathematical Statistics

2. **Dempster, Laird, & Rubin (1977)** - "Maximum Likelihood from Incomplete Data via the EM Algorithm"
   - General EM framework
   - Shows how Baum-Welch is a special case
   - Journal of the Royal Statistical Society

### Textbooks

3. **Bishop (2006)** - "Pattern Recognition and Machine Learning"
   - Chapter 13: Sequential Data
   - Section 13.2: HMMs with EM
   - Clear derivations

4. **Murphy (2012)** - "Machine Learning: A Probabilistic Perspective"
   - Chapter 17.4: Learning HMMs
   - Algorithm details and code
   - Extensions and variants

### Applied Guides

5. **Rabiner (1989)** - "A Tutorial on Hidden Markov Models"
   - Section IV: Learning (Baum-Welch)
   - Worked examples
   - Implementation tips

6. **Bilmes (1998)** - "A Gentle Tutorial of the EM Algorithm"
   - Intuitive EM explanation
   - HMM as example
   - UC Berkeley Technical Report

**Key Takeaway:** Baum-Welch solves the chicken-and-egg problem of HMM learning: we need states to estimate parameters, and parameters to infer states. By alternating between soft state inference (E-step) and parameter updates (M-step), we iteratively improve our model until convergence to a local optimum.

---

## Conceptual Practice Questions

1. What are the three parameter matrices in a Gaussian HMM and what does each control?

2. Why is the number of hidden states a hyperparameter rather than a learned parameter?

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.


---

## Cross-References

<a class="link-card" href="./03_baum_welch_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_forward_backward_impl.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_forward_backward.md">
  <div class="link-card-title">01 Forward Backward</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_viterbi_algorithm.md">
  <div class="link-card-title">02 Viterbi Algorithm</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

