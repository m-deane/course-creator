# EM for Gaussian HMMs: Continuous Observations

## In Brief

Gaussian Hidden Markov Models extend HMMs to continuous observations by using Gaussian (normal) emission distributions. The EM algorithm adapts naturally: the E-step computes state occupation probabilities, and the M-step updates means and covariances using weighted maximum likelihood estimation.

> 💡 **Key Insight:** For continuous observations (like financial returns), we can't count "how many times we saw value X in state i" because each observation is unique. Instead, we use probability densities and compute weighted statistics where weights are the probabilities of being in each state.

## Formal Definition

### Gaussian HMM Model

**States:** $S = \{s_1, ..., s_K\}$

**Observations:** $o_t \in \mathbb{R}^D$ (continuous, D-dimensional)

**Emission Model:** Each state $i$ has a Gaussian distribution:

$$b_i(o_t) = \mathcal{N}(o_t | \mu_i, \Sigma_i) = \frac{1}{(2\pi)^{D/2} |\Sigma_i|^{1/2}} \exp\left(-\frac{1}{2}(o_t - \mu_i)^T \Sigma_i^{-1} (o_t - \mu_i)\right)$$

**Parameters:**
- $\pi \in \mathbb{R}^K$: Initial state distribution
- $A \in \mathbb{R}^{K \times K}$: Transition matrix
- $\mu_i \in \mathbb{R}^D$: Mean vector for state $i$
- $\Sigma_i \in \mathbb{R}^{D \times D}$: Covariance matrix for state $i$

### Univariate Gaussian Case

For 1D observations (e.g., returns):

$$b_i(o_t) = \frac{1}{\sqrt{2\pi\sigma_i^2}} \exp\left(-\frac{(o_t - \mu_i)^2}{2\sigma_i^2}\right)$$

Parameters: $\mu_i \in \mathbb{R}$, $\sigma_i^2 \in \mathbb{R}^+$

### EM Algorithm for Gaussian HMMs

**E-Step:** Compute state occupation probabilities (same as discrete HMMs)

$$\gamma_t(i) = P(q_t = s_i | O, \lambda) = \frac{\alpha_t(i) \beta_t(i)}{\sum_j \alpha_t(j) \beta_t(j)}$$

$$\xi_t(i,j) = P(q_t = s_i, q_{t+1} = s_j | O, \lambda)$$

**M-Step:** Update Gaussian parameters

**Mean Update (Univariate):**
$$\hat{\mu}_i = \frac{\sum_{t=1}^T \gamma_t(i) \cdot o_t}{\sum_{t=1}^T \gamma_t(i)}$$

Weighted average of observations, weighted by probability of being in state $i$.

**Variance Update (Univariate):**
$$\hat{\sigma}_i^2 = \frac{\sum_{t=1}^T \gamma_t(i) \cdot (o_t - \hat{\mu}_i)^2}{\sum_{t=1}^T \gamma_t(i)}$$

Weighted variance around the updated mean.

**Multivariate Updates:**

Mean vector:
$$\hat{\mu}_i = \frac{\sum_{t=1}^T \gamma_t(i) \cdot o_t}{\sum_{t=1}^T \gamma_t(i)}$$

Covariance matrix:
$$\hat{\Sigma}_i = \frac{\sum_{t=1}^T \gamma_t(i) \cdot (o_t - \hat{\mu}_i)(o_t - \hat{\mu}_i)^T}{\sum_{t=1}^T \gamma_t(i)}$$

## Intuitive Explanation

### The Learning Process

Imagine you have daily stock returns and want to learn two market regimes (bull/bear):

**Initial Guess:**
- Bull: mean = +0.1%, std = 1%
- Bear: mean = -0.1%, std = 2%

**E-Step (Soft Clustering):**
For each day's return, compute:
- "How likely was this return if we were in Bull state?"
- "How likely if we were in Bear state?"
- Use these likelihoods + transition dynamics → probability of being in each state

**Example:**
```
Day 1: return = +2.5%
  P(Bull | return=+2.5%, history) = 0.85
  P(Bear | return=+2.5%, history) = 0.15

Day 2: return = -3.0%
  P(Bull | return=-3.0%, history) = 0.10
  P(Bear | return=-3.0%, history) = 0.90
```

**M-Step (Update Statistics):**
Recompute Bull regime mean:
```
Bull mean = (0.85 × 2.5% + 0.10 × (-3.0%) + ...) / (0.85 + 0.10 + ...)
          = weighted average of all returns, weighted by P(Bull)
```

Similarly for Bear mean, and for both variances.

### Why This Works

**Key Insight:** We're doing weighted maximum likelihood estimation, where weights are state occupation probabilities.

If we knew states exactly:
```
Bull returns: [+2.5%, +1.2%, +0.8%, ...]
Mean = average of bull returns
Variance = variance of bull returns
```

Since we don't know states, we weight each observation:
```
All returns: [+2.5%, -3.0%, +1.2%, ...]
Weights (Bull): [0.85, 0.10, 0.90, ...]

Weighted mean = Σ weights[i] × returns[i] / Σ weights[i]
```

This is the maximum likelihood estimate given soft assignments!

## Code Implementation

### Complete Gaussian HMM with EM

```python
import numpy as np
from scipy import stats
from typing import Tuple, Optional
import matplotlib.pyplot as plt

class GaussianHMM:
    """
    Gaussian Hidden Markov Model with EM training.
    """

    def __init__(
        self,
        n_states: int,
        n_features: int = 1,
        covariance_type: str = 'full'
    ):
        """
        Args:
            n_states: Number of hidden states
            n_features: Dimension of observations
            covariance_type: 'full', 'diag', or 'spherical'
        """
        self.K = n_states
        self.D = n_features
        self.covariance_type = covariance_type

        # Initialize parameters
        self.pi = np.ones(self.K) / self.K

        self.A = np.random.rand(self.K, self.K)
        self.A = self.A / self.A.sum(axis=1, keepdims=True)

        self.means = np.random.randn(self.K, self.D)

        if covariance_type == 'full':
            self.covars = np.array([np.eye(self.D) for _ in range(self.K)])
        elif covariance_type == 'diag':
            self.covars = np.ones((self.K, self.D))
        elif covariance_type == 'spherical':
            self.covars = np.ones(self.K)

    def _emission_prob(self, observation: np.ndarray, state: int) -> float:
        """
        Compute Gaussian emission probability.

        Args:
            observation: Shape (D,)
            state: State index

        Returns:
            Probability density
        """
        if self.covariance_type == 'full':
            return stats.multivariate_normal.pdf(
                observation,
                mean=self.means[state],
                cov=self.covars[state]
            )
        elif self.covariance_type == 'diag':
            return stats.multivariate_normal.pdf(
                observation,
                mean=self.means[state],
                cov=np.diag(self.covars[state])
            )
        elif self.covariance_type == 'spherical':
            return stats.multivariate_normal.pdf(
                observation,
                mean=self.means[state],
                cov=self.covars[state] * np.eye(self.D)
            )

    def forward(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Forward algorithm with scaling.

        Args:
            observations: Shape (T, D)

        Returns:
            alpha: Scaled forward probabilities (T, K)
            log_likelihood: log P(O|λ)
        """
        T = len(observations)
        alpha = np.zeros((T, self.K))
        scales = np.zeros(T)

        # Initialization
        for i in range(self.K):
            alpha[0, i] = self.pi[i] * self._emission_prob(observations[0], i)

        scales[0] = np.sum(alpha[0])
        alpha[0] = alpha[0] / (scales[0] + 1e-10)

        # Recursion
        for t in range(1, T):
            for j in range(self.K):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * \
                              self._emission_prob(observations[t], j)

            scales[t] = np.sum(alpha[t])
            alpha[t] = alpha[t] / (scales[t] + 1e-10)

        log_likelihood = np.sum(np.log(scales + 1e-10))

        return alpha, log_likelihood

    def backward(self, observations: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """
        Backward algorithm with scaling.

        Args:
            observations: Shape (T, D)
            scales: Scaling factors from forward algorithm

        Returns:
            beta: Scaled backward probabilities (T, K)
        """
        T = len(observations)
        beta = np.zeros((T, self.K))

        # Initialization
        beta[-1, :] = 1.0

        # Recursion
        for t in range(T-2, -1, -1):
            for i in range(self.K):
                beta[t, i] = np.sum(
                    self.A[i, :] *
                    np.array([self._emission_prob(observations[t+1], j) for j in range(self.K)]) *
                    beta[t+1, :]
                )

            # Scale with same factor as forward
            beta[t] = beta[t] / (scales[t+1] + 1e-10)

        return beta

    def e_step(
        self,
        observations: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        E-step: Compute state occupation and transition probabilities.

        Args:
            observations: Shape (T, D)
            alpha: Forward probabilities (T, K)
            beta: Backward probabilities (T, K)

        Returns:
            gamma: State probabilities (T, K)
            xi: Transition probabilities (T-1, K, K)
        """
        T = len(observations)

        # Gamma: P(state at time t | all observations)
        gamma = alpha * beta
        gamma = gamma / (gamma.sum(axis=1, keepdims=True) + 1e-10)

        # Xi: P(state i at t, state j at t+1 | all observations)
        xi = np.zeros((T-1, self.K, self.K))

        for t in range(T-1):
            for i in range(self.K):
                for j in range(self.K):
                    xi[t, i, j] = alpha[t, i] * \
                                  self.A[i, j] * \
                                  self._emission_prob(observations[t+1], j) * \
                                  beta[t+1, j]

            xi[t] = xi[t] / (xi[t].sum() + 1e-10)

        return gamma, xi

    def m_step(
        self,
        observations: np.ndarray,
        gamma: np.ndarray,
        xi: np.ndarray
    ):
        """
        M-step: Update Gaussian HMM parameters.

        Args:
            observations: Shape (T, D)
            gamma: State probabilities (T, K)
            xi: Transition probabilities (T-1, K, K)
        """
        T = len(observations)

        # Update initial state distribution
        self.pi = gamma[0]

        # Update transition matrix
        for i in range(self.K):
            for j in range(self.K):
                self.A[i, j] = np.sum(xi[:, i, j]) / (np.sum(gamma[:-1, i]) + 1e-10)

        # Normalize
        self.A = self.A / (self.A.sum(axis=1, keepdims=True) + 1e-10)

        # Update Gaussian parameters
        for i in range(self.K):
            # Weighted mean
            weights = gamma[:, i]
            self.means[i] = np.sum(weights[:, np.newaxis] * observations, axis=0) / \
                           (np.sum(weights) + 1e-10)

            # Weighted covariance
            diff = observations - self.means[i]

            if self.covariance_type == 'full':
                self.covars[i] = np.sum(
                    weights[:, np.newaxis, np.newaxis] * (diff[:, :, np.newaxis] @ diff[:, np.newaxis, :]),
                    axis=0
                ) / (np.sum(weights) + 1e-10)

                # Ensure positive definite
                self.covars[i] += 1e-6 * np.eye(self.D)

            elif self.covariance_type == 'diag':
                self.covars[i] = np.sum(weights[:, np.newaxis] * diff**2, axis=0) / \
                                (np.sum(weights) + 1e-10) + 1e-6

            elif self.covariance_type == 'spherical':
                self.covars[i] = np.sum(weights[:, np.newaxis] * diff**2) / \
                                (np.sum(weights) * self.D + 1e-10) + 1e-6

    def fit(
        self,
        observations: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-4,
        verbose: bool = True
    ) -> list:
        """
        Fit Gaussian HMM using EM algorithm.

        Args:
            observations: Shape (T, D) or (T,) for univariate
            max_iter: Maximum iterations
            tol: Convergence tolerance
            verbose: Print progress

        Returns:
            log_likelihoods: Log-likelihood history
        """
        # Ensure observations are 2D
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)

        # Get scaling factors for backward
        _, scales_init = self.forward(observations)

        log_likelihoods = []

        for iteration in range(max_iter):
            # E-step
            alpha, log_likelihood = self.forward(observations)
            _, scales = self.forward(observations)
            beta = self.backward(observations, scales)
            gamma, xi = self.e_step(observations, alpha, beta)

            # M-step
            self.m_step(observations, gamma, xi)

            log_likelihoods.append(log_likelihood)

            if verbose and (iteration % 10 == 0 or iteration == max_iter - 1):
                print(f"Iteration {iteration:3d}: log P(O|λ) = {log_likelihood:.4f}")

            # Check convergence
            if iteration > 0:
                if abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
                    if verbose:
                        print(f"Converged after {iteration + 1} iterations")
                    break

        return log_likelihoods

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict most likely state sequence (Viterbi).

        Args:
            observations: Shape (T, D) or (T,)

        Returns:
            states: Most likely state sequence (T,)
        """
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)

        T = len(observations)
        delta = np.zeros((T, self.K))
        psi = np.zeros((T, self.K), dtype=int)

        # Initialization
        for i in range(self.K):
            delta[0, i] = np.log(self.pi[i] + 1e-10) + \
                         np.log(self._emission_prob(observations[0], i) + 1e-10)

        # Recursion
        for t in range(1, T):
            for j in range(self.K):
                probs = delta[t-1] + np.log(self.A[:, j] + 1e-10)
                psi[t, j] = np.argmax(probs)
                delta[t, j] = np.max(probs) + \
                             np.log(self._emission_prob(observations[t], j) + 1e-10)

        # Backtracking
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])

        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]

        return states


# Example: Market regime detection
np.random.seed(42)

# Generate synthetic market data
def generate_market_data(T=1000):
    """Generate returns from 2-regime model."""
    # True parameters
    states = [0]  # Start in bull
    returns = []

    true_A = np.array([[0.98, 0.02],   # Bull persistent
                       [0.05, 0.95]])   # Bear persistent

    true_means = np.array([0.0005, -0.0003])  # Bull: +0.05%/day, Bear: -0.03%/day
    true_stds = np.array([0.01, 0.015])        # Bull: 1%, Bear: 1.5% vol

    for t in range(T):
        # Generate return from current state
        returns.append(np.random.normal(true_means[states[-1]], true_stds[states[-1]]))

        # Transition to next state
        if t < T - 1:
            next_state = np.random.choice(2, p=true_A[states[-1]])
            states.append(next_state)

    return np.array(returns), np.array(states)

returns, true_states = generate_market_data(1000)

print("=" * 70)
print("MARKET REGIME DETECTION WITH GAUSSIAN HMM")
print("=" * 70)

# Fit Gaussian HMM
model = GaussianHMM(n_states=2, n_features=1, covariance_type='spherical')
log_liks = model.fit(returns, max_iter=100, verbose=True)

# Predict states
predicted_states = model.predict(returns)

# Identify which state is bull/bear by mean
bull_state = np.argmax(model.means.flatten())
bear_state = 1 - bull_state

print("\n" + "=" * 70)
print("LEARNED PARAMETERS:")
print("=" * 70)
print(f"Bull State (State {bull_state}):")
print(f"  Mean: {model.means[bull_state, 0]:.6f} ({model.means[bull_state, 0]*252:.2%} annualized)")
print(f"  Std:  {np.sqrt(model.covars[bull_state]):.6f} ({np.sqrt(model.covars[bull_state]*252):.2%} annualized)")

print(f"\nBear State (State {bear_state}):")
print(f"  Mean: {model.means[bear_state, 0]:.6f} ({model.means[bear_state, 0]*252:.2%} annualized)")
print(f"  Std:  {np.sqrt(model.covars[bear_state]):.6f} ({np.sqrt(model.covars[bear_state]*252):.2%} annualized)")

print(f"\nTransition Matrix:")
print(model.A)

# Compute accuracy
# Map predicted states to true states (account for label switching)
if np.mean(predicted_states == true_states) < 0.5:
    predicted_states = 1 - predicted_states

accuracy = np.mean(predicted_states == true_states)
print(f"\nState Prediction Accuracy: {accuracy:.2%}")

# Visualization
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Plot 1: Returns with true states
ax1 = axes[0]
for state in [0, 1]:
    mask = true_states == state
    ax1.scatter(np.where(mask)[0], returns[mask],
               label=f'True State {state}', alpha=0.5, s=5)
ax1.set_ylabel('Return')
ax1.set_title('Returns Colored by True States')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Returns with predicted states
ax2 = axes[1]
for state in [0, 1]:
    mask = predicted_states == state
    ax2.scatter(np.where(mask)[0], returns[mask],
               label=f'Predicted State {state}', alpha=0.5, s=5)
ax2.set_ylabel('Return')
ax2.set_title('Returns Colored by Predicted States')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Log-likelihood convergence
ax3 = axes[2]
ax3.plot(log_liks, 'o-', linewidth=2)
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Log-Likelihood')
ax3.set_title('EM Convergence')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Common Pitfalls

1. **Singular Covariance Matrices**
   - Problem: Covariance becomes singular (non-invertible)
   - Causes: Insufficient data for state, numerical issues
   - Solution: Add small regularization to diagonal

```python
# In M-step
self.covars[i] += 1e-6 * np.eye(self.D)  # Regularization
```

2. **Covariance Type Selection**
   - **Full**: Most flexible, but K×D×D parameters (can overfit)
   - **Diagonal**: Assumes features uncorrelated (often reasonable)
   - **Spherical**: Single variance per state (most constrained)

```python
# Start with diagonal, use full only if needed
model = GaussianHMM(n_states=2, covariance_type='diag')
```

3. **Label Switching**
   - EM doesn't preserve state labels across runs
   - State 0 in one run might be State 1 in another
   - Solution: Identify states by learned parameters (e.g., highest mean = bull)

```python
def identify_states(model):
    """Identify bull/bear by mean."""
    bull_idx = np.argmax(model.means)
    bear_idx = np.argmin(model.means)
    return bull_idx, bear_idx
```

4. **Numerical Underflow**
   - Forward/backward probabilities become very small
   - Solution: Use scaling (shown in code) or log-space

5. **Initialization Matters**
   - Random initialization can lead to poor local optima
   - Better: K-means initialization

```python
from sklearn.cluster import KMeans

def initialize_from_kmeans(model, observations):
    """Initialize Gaussian parameters from K-means."""
    kmeans = KMeans(n_clusters=model.K, random_state=42)
    labels = kmeans.fit_predict(observations.reshape(-1, 1))

    for i in range(model.K):
        mask = labels == i
        if mask.sum() > 0:
            model.means[i] = observations[mask].mean()
            if model.covariance_type == 'spherical':
                model.covars[i] = observations[mask].var()

    return model
```

## Connections

### Builds on:
- **Discrete HMM EM**: Same E-step, different M-step
- **Gaussian distributions**: MLE for mean and covariance
- **Weighted statistics**: Soft clustering

### Leads to:
- **Multivariate Gaussian HMMs**: Multiple features
- **Mixture of Gaussians**: HMM without temporal structure
- **Gaussian mixture regression**: State-dependent regression

### Related to:
- **K-means**: Hard clustering (Gaussian HMM is soft clustering over time)
- **GMM**: Mixture model without transitions
- **Kalman filter**: Continuous state space version

## Practice Problems

1. **Parameter Interpretation**

   A 2-state Gaussian HMM for daily returns has:
   - State 0: μ = 0.001, σ = 0.015
   - State 1: μ = -0.002, σ = 0.025

   Interpret these states financially.

   <details>
   <summary>Answer</summary>

   State 0 (Likely Bull):
   - Mean: +0.1% daily = ~25% annualized (0.001 × 252)
   - Vol: 1.5% daily = ~24% annualized (0.015 × √252)
   - Positive drift, moderate volatility

   State 1 (Likely Bear):
   - Mean: -0.2% daily = ~-50% annualized
   - Vol: 2.5% daily = ~40% annualized
   - Negative drift, high volatility
   - Classic bear market characteristics
   </details>

2. **Weighted Average**

   You have 3 observations with state probabilities:
   ```
   obs = [0.05, 0.02, -0.01]
   P(state=0) = [0.9, 0.3, 0.1]
   ```

   Compute the updated mean for state 0.

   <details>
   <summary>Solution</summary>

   μ₀ = Σ γₜ(0) · oₜ / Σ γₜ(0)
      = (0.9×0.05 + 0.3×0.02 + 0.1×(-0.01)) / (0.9 + 0.3 + 0.1)
      = (0.045 + 0.006 - 0.001) / 1.3
      = 0.050 / 1.3
      = 0.0385
   </details>

3. **Covariance Types**

   For a 2-state, 3-feature Gaussian HMM, how many covariance parameters with each type?

   <details>
   <summary>Answer</summary>

   - **Full**: 2 states × 3×3 covariance matrix = 2 × 9 = 18 parameters
     (but covariance is symmetric, so actually 2 × 6 = 12 unique values)

   - **Diagonal**: 2 states × 3 variances = 6 parameters

   - **Spherical**: 2 states × 1 variance = 2 parameters
   </details>

4. **Design Decision**

   You have daily stock returns for regime detection. Should you use multivariate Gaussian HMM with returns from multiple stocks, or separate univariate HMMs?

   <details>
   <summary>Answer</summary>

   **Multivariate is better if:**
   - Stocks share common regimes (market-wide bull/bear)
   - Want to capture correlation structure
   - Have enough data for full covariance estimation

   **Separate univariate if:**
   - Stocks have independent regimes
   - Limited data (avoid overfitting)
   - Interpretability more important

   For market regimes, multivariate with diagonal covariance is often a good compromise.
   </details>

## Further Reading

### Key Papers

1. **Poritz (1988)** - "Hidden Markov Models: A Guided Tour"
   - Continuous observation HMMs
   - Gaussian emission derivations

2. **Juang & Rabiner (1991)** - "Hidden Markov Models for Speech Recognition"
   - Gaussian mixture emissions
   - Practical implementation details

### Financial Applications

3. **Hamilton (1989)** - "A New Approach to the Economic Analysis of Nonstationary Time Series"
   - Regime-switching with Gaussian emissions
   - Economic applications

4. **Guidolin & Timmermann (2008)** - "International Asset Allocation under Regime Switching"
   - Multivariate Gaussian HMMs for portfolios
   - Journal of Financial Economics

### Implementation Guides

5. **hmmlearn Documentation**
   - GaussianHMM class implementation
   - https://hmmlearn.readthedocs.io/

6. **Murphy (2012)** - "Machine Learning: A Probabilistic Perspective"
   - Section 17.4.2: Gaussian HMMs
   - Complete code examples

---

**Key Takeaway:** Gaussian HMMs extend the EM framework to continuous observations by using probability densities instead of discrete probabilities. The M-step becomes weighted maximum likelihood estimation for Gaussian parameters, where weights are the state occupation probabilities from the E-step.
