# HMM Parameter Estimation: The Baum-Welch Algorithm

> **Reading time:** ~7 min | **Module:** Module 5: Extensions | **Prerequisites:** Modules 0-4

<div class="callout-key">

**Key Concept Summary:** Given observations, how do we learn the HMM parameters $\lambda = (\pi, A, B)$?

</div>

## Introduction

Given observations, how do we learn the HMM parameters $\lambda = (\pi, A, B)$?

The **Baum-Welch algorithm** (Expectation-Maximization for HMMs) iteratively improves parameters to maximize likelihood.

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## The Learning Problem

### Objective

Find parameters that maximize:

$$\lambda^* = \arg\max_\lambda P(O | \lambda)$$

This is non-convex, so we use iterative optimization.

## EM Framework for HMMs

### E-Step: Compute Expected Statistics

Given current parameters, compute:
- $\gamma_t(i) = P(S_t = i | O, \lambda)$: Probability of being in state $i$ at time $t$
- $\xi_t(i,j) = P(S_t = i, S_{t+1} = j | O, \lambda)$: Transition probability at time $t$

### M-Step: Update Parameters

Update parameters using expected statistics.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">baumwelchtrainer.py</span>
</div>

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class BaumWelchTrainer:
    """
    Baum-Welch algorithm for Gaussian HMM.
    """

    def __init__(self, n_states, n_features=1):
        self.n_states = n_states
        self.n_features = n_features

        # Parameters
        self.pi = None
        self.A = None
        self.means = None
        self.covars = None

    def initialize(self, observations, method='kmeans'):
        """
        Initialize parameters.
        """
        n_samples = len(observations)
        K = self.n_states

        # Initial distribution
        self.pi = np.ones(K) / K

        # Transition matrix (slightly persistent)
        self.A = np.ones((K, K)) * 0.1 / (K - 1)
        np.fill_diagonal(self.A, 0.9)

        # Emission parameters
        if method == 'kmeans':
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
            labels = kmeans.fit_predict(observations.reshape(-1, 1))
            self.means = kmeans.cluster_centers_.flatten()

            self.covars = np.array([
                np.var(observations[labels == k]) + 1e-6
                for k in range(K)
            ])
        else:
            # Random initialization
            self.means = np.percentile(observations,
                                       np.linspace(10, 90, K))
            self.covars = np.ones(K) * np.var(observations)

        return self

    def forward(self, observations):
        """
        Forward algorithm (alpha).
        """
        T = len(observations)
        K = self.n_states

        alpha = np.zeros((T, K))
        scale = np.zeros(T)

        # t = 0
        for k in range(K):
            alpha[0, k] = self.pi[k] * self._emission_prob(observations[0], k)

        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0]

        # Forward pass
        for t in range(1, T):
            for k in range(K):
                alpha[t, k] = np.sum(alpha[t-1] * self.A[:, k]) * \
                              self._emission_prob(observations[t], k)

            scale[t] = alpha[t].sum()
            if scale[t] > 0:
                alpha[t] /= scale[t]

        return alpha, scale

    def backward(self, observations, scale):
        """
        Backward algorithm (beta).
        """
        T = len(observations)
        K = self.n_states

        beta = np.zeros((T, K))
        beta[T-1] = 1  # Initialize

        for t in range(T-2, -1, -1):
            for k in range(K):
                beta[t, k] = np.sum(
                    self.A[k] *
                    np.array([self._emission_prob(observations[t+1], j)
                              for j in range(K)]) *
                    beta[t+1]
                )

            if scale[t+1] > 0:
                beta[t] /= scale[t+1]

        return beta

    def _emission_prob(self, obs, state):
        """
        Emission probability for univariate Gaussian.
        """
        return stats.norm.pdf(obs, self.means[state],
                              np.sqrt(self.covars[state]))

    def compute_gamma_xi(self, observations, alpha, beta, scale):
        """
        Compute gamma and xi.
        """
        T = len(observations)
        K = self.n_states

        # Gamma: P(S_t = k | O)
        gamma = alpha * beta
        gamma = gamma / gamma.sum(axis=1, keepdims=True)

        # Xi: P(S_t = i, S_{t+1} = j | O)
        xi = np.zeros((T-1, K, K))

        for t in range(T-1):
            for i in range(K):
                for j in range(K):
                    xi[t, i, j] = (
                        alpha[t, i] *
                        self.A[i, j] *
                        self._emission_prob(observations[t+1], j) *
                        beta[t+1, j]
                    )

            # Normalize
            xi_sum = xi[t].sum()
            if xi_sum > 0:
                xi[t] /= xi_sum

        return gamma, xi

    def m_step(self, observations, gamma, xi):
        """
        M-step: Update parameters.
        """
        T = len(observations)
        K = self.n_states

        # Update initial distribution
        self.pi = gamma[0] / gamma[0].sum()

        # Update transition matrix
        for i in range(K):
            for j in range(K):
                self.A[i, j] = xi[:, i, j].sum() / gamma[:-1, i].sum()

        # Normalize rows
        self.A = self.A / self.A.sum(axis=1, keepdims=True)

        # Update emission parameters
        for k in range(K):
            # Mean
            self.means[k] = np.sum(gamma[:, k] * observations) / gamma[:, k].sum()

            # Variance
            diff = observations - self.means[k]
            self.covars[k] = np.sum(gamma[:, k] * diff**2) / gamma[:, k].sum()
            self.covars[k] = max(self.covars[k], 1e-6)  # Prevent collapse

    def log_likelihood(self, scale):
        """
        Compute log-likelihood from scaling factors.
        """
        return np.sum(np.log(scale + 1e-10))

    def fit(self, observations, n_iter=100, tol=1e-4, verbose=True):
        """
        Fit model using Baum-Welch.
        """
        self.initialize(observations)

        log_liks = []

        for iteration in range(n_iter):
            # E-step
            alpha, scale = self.forward(observations)
            beta = self.backward(observations, scale)
            gamma, xi = self.compute_gamma_xi(observations, alpha, beta, scale)

            # M-step
            self.m_step(observations, gamma, xi)

            # Log-likelihood
            ll = self.log_likelihood(scale)
            log_liks.append(ll)

            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: Log-likelihood = {ll:.4f}")

            # Check convergence
            if iteration > 0 and abs(log_liks[-1] - log_liks[-2]) < tol:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break

        return log_liks

    def predict(self, observations):
        """
        Predict most likely states (Viterbi).
        """
        T = len(observations)
        K = self.n_states

        # Log probabilities
        log_delta = np.zeros((T, K))
        psi = np.zeros((T, K), dtype=int)

        # Initialize
        for k in range(K):
            log_delta[0, k] = np.log(self.pi[k] + 1e-10) + \
                              np.log(self._emission_prob(observations[0], k) + 1e-10)

        # Forward
        for t in range(1, T):
            for k in range(K):
                trans_probs = log_delta[t-1] + np.log(self.A[:, k] + 1e-10)
                psi[t, k] = np.argmax(trans_probs)
                log_delta[t, k] = trans_probs[psi[t, k]] + \
                                  np.log(self._emission_prob(observations[t], k) + 1e-10)

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[T-1] = np.argmax(log_delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]

        return states


# Example: Learn from simulated data
np.random.seed(42)

# True parameters
true_pi = np.array([0.6, 0.4])
true_A = np.array([[0.9, 0.1], [0.15, 0.85]])
true_means = np.array([-1.0, 2.0])
true_vars = np.array([0.5, 1.0])

# Generate data
T = 500
true_states = [np.random.choice(2, p=true_pi)]
observations = []

for t in range(T):
    obs = np.random.normal(true_means[true_states[-1]],
                           np.sqrt(true_vars[true_states[-1]]))
    observations.append(obs)

    if t < T - 1:
        true_states.append(np.random.choice(2, p=true_A[true_states[-1]]))

observations = np.array(observations)
true_states = np.array(true_states)

# Fit model
trainer = BaumWelchTrainer(n_states=2)
log_liks = trainer.fit(observations, n_iter=100, verbose=True)

print("\n" + "=" * 60)
print("LEARNED PARAMETERS:")
print("=" * 60)

print("\nInitial distribution:")
print(f"  True:    {true_pi}")
print(f"  Learned: {trainer.pi.round(4)}")

print("\nTransition matrix:")
print(f"  True:\n{true_A}")
print(f"  Learned:\n{trainer.A.round(4)}")

print("\nEmission means:")
print(f"  True:    {true_means}")
print(f"  Learned: {trainer.means.round(4)}")

print("\nEmission variances:")
print(f"  True:    {true_vars}")
print(f"  Learned: {trainer.covars.round(4)}")

# Decode states
decoded = trainer.predict(observations)
accuracy = np.mean(decoded == true_states)
print(f"\nState decoding accuracy: {accuracy:.1%}")

# Visualize
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Log-likelihood convergence
ax1 = axes[0]
ax1.plot(log_liks, 'b-', linewidth=2)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Log-Likelihood')
ax1.set_title('Baum-Welch Convergence')
ax1.grid(True, alpha=0.3)

# Observations with states
ax2 = axes[1]
ax2.plot(observations, 'k-', alpha=0.5, linewidth=0.5)
for t in range(T-1):
    ax2.axvspan(t, t+1, alpha=0.2,
                color='blue' if true_states[t] == 0 else 'red')
ax2.set_ylabel('Observation')
ax2.set_title('Observations (True States: Blue=0, Red=1)')

# Decoded vs true
ax3 = axes[2]
ax3.step(range(T), true_states, 'b-', label='True', alpha=0.7, where='mid')
ax3.step(range(T), decoded, 'r--', label='Decoded', alpha=0.7, where='mid')
ax3.set_xlabel('Time')
ax3.set_ylabel('State')
ax3.set_title('True vs Decoded States')
ax3.legend()

plt.tight_layout()
plt.show()
```

</div>
</div>

## Multiple Random Restarts


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">fit_with_restarts.py</span>
</div>

```python
def fit_with_restarts(observations, n_states, n_restarts=10):
    """
    Fit HMM with multiple random initializations.
    """
    best_model = None
    best_ll = -np.inf

    for restart in range(n_restarts):
        np.random.seed(restart * 100)

        trainer = BaumWelchTrainer(n_states=n_states)
        trainer.initialize(observations, method='random' if restart > 0 else 'kmeans')

        try:
            log_liks = trainer.fit(observations, n_iter=50, verbose=False)
            final_ll = log_liks[-1]

            if final_ll > best_ll:
                best_ll = final_ll
                best_model = trainer
                best_log_liks = log_liks

            print(f"Restart {restart+1}: Final LL = {final_ll:.4f}")

        except Exception as e:
            print(f"Restart {restart+1}: Failed - {e}")

    print(f"\nBest log-likelihood: {best_ll:.4f}")

    return best_model, best_log_liks

best_model, best_ll = fit_with_restarts(observations, n_states=2, n_restarts=5)
```

</div>
</div>

## Practical Considerations

### Avoiding Local Minima


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">discuss_training_issues.py</span>
</div>

```python
def discuss_training_issues():
    """
    Common training issues and solutions.
    """
    issues = """
BAUM-WELCH TRAINING ISSUES:
================================================================================

1. LOCAL MINIMA
   - EM only guarantees local convergence
   - Solution: Multiple random restarts
   - Solution: Use good initialization (k-means)

2. NUMERICAL UNDERFLOW
   - Product of many small probabilities
   - Solution: Work in log-space
   - Solution: Use scaling factors (as shown)

3. DEGENERATE SOLUTIONS
   - Variance collapses to zero
   - Solution: Add minimum variance constraint
   - Solution: Regularization (add small constant)

4. OVERFITTING
   - Too many states for data
   - Solution: Use BIC/AIC for model selection
   - Solution: Cross-validation

5. SLOW CONVERGENCE
   - EM can be slow near optimum
   - Solution: Use convergence tolerance
   - Solution: Consider gradient-based methods

6. LABEL SWITCHING
   - State labels can swap between runs
   - Solution: Order states by parameter (e.g., mean)
   - Not a problem for likelihood

PRACTICAL TIPS:
- Start with k-means for mean initialization
- Use 5-10 random restarts
- Monitor log-likelihood for convergence
- Validate on held-out data
    """
    print(issues)

discuss_training_issues()
```

</div>
</div>

### Using hmmlearn


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">train_with_hmmlearn.py</span>
</div>

```python
from hmmlearn import hmm

def train_with_hmmlearn(observations, n_states=2):
    """
    Train using hmmlearn (recommended for production).
    """
    observations = observations.reshape(-1, 1)

    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type='full',
        n_iter=200,
        random_state=42,
        init_params='stmc',  # Initialize all parameters
        params='stmc'  # Train all parameters
    )

    model.fit(observations)

    print("hmmlearn Results:")
    print("=" * 60)
    print(f"Log-likelihood: {model.score(observations):.4f}")
    print(f"Converged: {model.monitor_.converged}")
    print(f"Iterations: {model.monitor_.iter}")
    print(f"\nMeans: {model.means_.flatten()}")
    print(f"Variances: {model.covars_.flatten()}")
    print(f"Transition matrix:\n{model.transmat_.round(4)}")

    return model

hmmlearn_model = train_with_hmmlearn(observations)
```


<div class="callout-insight">

**Insight:** Understanding hmm parameter estimation is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.


## Key Takeaways

1. **Baum-Welch is EM for HMMs** - E-step computes state probabilities, M-step updates parameters

2. **Forward-backward** computes the necessary statistics efficiently

3. **Multiple restarts** help avoid local minima

4. **Scaling/log-space** prevents numerical underflow

5. **hmmlearn** provides robust, production-ready implementation

6. **Initialization matters** - use k-means or domain knowledge

---

## Conceptual Practice Questions

1. What are the three parameter matrices in a Gaussian HMM and what does each control?

2. Why is the number of hidden states a hyperparameter rather than a learned parameter?

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.


---

## Cross-References

<a class="link-card" href="./02_parameter_estimation_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_hhmm_implementation.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_advanced_hmms.md">
  <div class="link-card-title">01 Advanced Hmms</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

