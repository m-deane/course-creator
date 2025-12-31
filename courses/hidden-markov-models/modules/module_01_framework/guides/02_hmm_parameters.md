# HMM Parameters: The Complete Picture

## The Three Parameter Sets

A Hidden Markov Model is fully specified by three parameter sets:

1. **Initial distribution** $\pi$: Starting state probabilities
2. **Transition matrix** $A$: State-to-state transitions
3. **Emission parameters** $B$: Observation probabilities given state

Together: $\lambda = (\pi, A, B)$

## Initial State Distribution

### Definition

$\pi_i = P(S_1 = i)$ for $i = 1, ..., K$

Constraints:
- $\pi_i \geq 0$ for all $i$
- $\sum_i \pi_i = 1$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class HMMParameters:
    """
    Complete HMM parameter specification.
    """

    def __init__(self, n_states):
        self.n_states = n_states
        self.pi = None  # Initial distribution
        self.A = None   # Transition matrix
        self.B = None   # Emission parameters (varies by type)

    def set_initial_distribution(self, pi=None, style='uniform'):
        """
        Set the initial state distribution.
        """
        if pi is not None:
            self.pi = np.array(pi)
        elif style == 'uniform':
            self.pi = np.ones(self.n_states) / self.n_states
        elif style == 'first_state':
            self.pi = np.zeros(self.n_states)
            self.pi[0] = 1.0
        elif style == 'random':
            self.pi = np.random.dirichlet(np.ones(self.n_states))

        assert np.isclose(self.pi.sum(), 1.0), "π must sum to 1"
        assert all(self.pi >= 0), "π must be non-negative"

        return self

    def set_transition_matrix(self, A=None, style='persistent', persistence=0.9):
        """
        Set the transition matrix.
        """
        if A is not None:
            self.A = np.array(A)
        elif style == 'persistent':
            self.A = np.eye(self.n_states) * persistence
            off_diag = (1 - persistence) / (self.n_states - 1)
            self.A[self.A == 0] = off_diag
        elif style == 'uniform':
            self.A = np.ones((self.n_states, self.n_states)) / self.n_states
        elif style == 'random':
            self.A = np.random.dirichlet(np.ones(self.n_states), self.n_states)

        # Verify properties
        assert np.allclose(self.A.sum(axis=1), 1.0), "Rows must sum to 1"
        assert np.all(self.A >= 0), "Elements must be non-negative"

        return self

# Example
np.random.seed(42)
hmm = HMMParameters(n_states=3)
hmm.set_initial_distribution(style='uniform')
hmm.set_transition_matrix(style='persistent', persistence=0.85)

print("HMM Parameters (Partial):")
print("=" * 60)
print(f"\nInitial distribution π:")
print(f"  {hmm.pi}")
print(f"\nTransition matrix A:")
print(hmm.A.round(3))
```

## Transition Matrix Details

### Ergodicity and Stationarity

```python
def analyze_transition_matrix(A):
    """
    Analyze properties of transition matrix.
    """
    print("Transition Matrix Analysis:")
    print("=" * 60)

    n_states = A.shape[0]

    # 1. Check row stochasticity
    row_sums = A.sum(axis=1)
    print(f"\n1. Row stochastic: {np.allclose(row_sums, 1.0)}")

    # 2. Compute stationary distribution
    eigenvalues, eigenvectors = np.linalg.eig(A.T)
    idx = np.argmin(np.abs(eigenvalues - 1))
    pi_stationary = np.real(eigenvectors[:, idx])
    pi_stationary = pi_stationary / pi_stationary.sum()

    print(f"\n2. Stationary distribution:")
    print(f"   π = {pi_stationary.round(4)}")

    # 3. Expected state duration
    print(f"\n3. Expected state durations:")
    for i in range(n_states):
        expected_duration = 1 / (1 - A[i, i]) if A[i, i] < 1 else float('inf')
        print(f"   State {i}: {expected_duration:.2f} periods")

    # 4. Check irreducibility (all states reachable)
    A_power = A.copy()
    reach = A > 0
    for _ in range(n_states):
        A_power = A_power @ A
        reach = reach | (A_power > 1e-10)

    irreducible = np.all(reach)
    print(f"\n4. Irreducible: {irreducible}")

    # 5. Check aperiodicity
    diag_positive = np.any(np.diag(A) > 0)
    print(f"\n5. Aperiodic (diagonal > 0): {diag_positive}")

    # 6. Ergodic = irreducible + aperiodic
    ergodic = irreducible and diag_positive
    print(f"\n6. Ergodic: {ergodic}")

    return {
        'stationary': pi_stationary,
        'irreducible': irreducible,
        'aperiodic': diag_positive,
        'ergodic': ergodic
    }

analyze_transition_matrix(hmm.A)
```

## Emission Parameters

### Discrete Emissions

For discrete observations $o \in \{1, ..., M\}$:

$$b_i(o) = P(O_t = o | S_t = i)$$

```python
class DiscreteHMM(HMMParameters):
    """
    HMM with discrete emissions.
    """

    def __init__(self, n_states, n_symbols):
        super().__init__(n_states)
        self.n_symbols = n_symbols

    def set_emission_matrix(self, B=None, style='random'):
        """
        Set discrete emission probabilities.
        """
        if B is not None:
            self.B = np.array(B)
        elif style == 'random':
            self.B = np.random.dirichlet(
                np.ones(self.n_symbols), self.n_states
            )
        elif style == 'identity':
            # Each state emits its corresponding symbol with high prob
            self.B = np.eye(self.n_states, self.n_symbols) * 0.8
            self.B += 0.2 / self.n_symbols

        # B[i,j] = P(emit symbol j | state i)
        assert self.B.shape == (self.n_states, self.n_symbols)
        assert np.allclose(self.B.sum(axis=1), 1.0)

        return self

    def emission_prob(self, state, symbol):
        """Get emission probability."""
        return self.B[state, symbol]

# Example
discrete_hmm = DiscreteHMM(n_states=2, n_symbols=4)
discrete_hmm.set_initial_distribution(style='uniform')
discrete_hmm.set_transition_matrix(style='persistent', persistence=0.9)
discrete_hmm.set_emission_matrix(style='random')

print("\nDiscrete HMM Emission Matrix B:")
print("(Rows = states, Columns = symbols)")
print(discrete_hmm.B.round(3))
```

### Gaussian Emissions

For continuous observations:

$$b_i(o) = \mathcal{N}(o; \mu_i, \sigma_i^2)$$

```python
class GaussianHMM(HMMParameters):
    """
    HMM with Gaussian emissions.
    """

    def __init__(self, n_states, n_features=1):
        super().__init__(n_states)
        self.n_features = n_features
        self.means = None
        self.covars = None

    def set_emission_params(self, means=None, covars=None, style='separated'):
        """
        Set Gaussian emission parameters.
        """
        if means is not None:
            self.means = np.array(means)
        elif style == 'separated':
            # States have well-separated means
            self.means = np.linspace(-2, 2, self.n_states).reshape(-1, 1)
            if self.n_features > 1:
                self.means = np.hstack([
                    self.means,
                    np.random.randn(self.n_states, self.n_features - 1) * 0.5
                ])

        if covars is not None:
            self.covars = np.array(covars)
        elif style == 'separated':
            self.covars = np.array([
                np.eye(self.n_features) * (0.3 + 0.2 * i)
                for i in range(self.n_states)
            ])

        return self

    def emission_prob(self, state, observation):
        """
        Compute emission probability (density) for Gaussian.
        """
        return stats.multivariate_normal.pdf(
            observation,
            mean=self.means[state].flatten(),
            cov=self.covars[state]
        )

    def visualize_emissions(self):
        """
        Visualize Gaussian emission distributions.
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.linspace(-5, 5, 500)

        for state in range(self.n_states):
            mean = self.means[state, 0]
            std = np.sqrt(self.covars[state, 0, 0])
            y = stats.norm.pdf(x, mean, std)
            ax.fill_between(x, y, alpha=0.3, label=f'State {state}')
            ax.plot(x, y, linewidth=2)

        ax.set_xlabel('Observation Value')
        ax.set_ylabel('Density')
        ax.set_title('Gaussian Emission Distributions by State')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

# Example
gauss_hmm = GaussianHMM(n_states=3, n_features=1)
gauss_hmm.set_initial_distribution(pi=[0.6, 0.3, 0.1])
gauss_hmm.set_transition_matrix(A=[
    [0.8, 0.15, 0.05],
    [0.1, 0.8, 0.1],
    [0.05, 0.15, 0.8]
])
gauss_hmm.set_emission_params(
    means=[[-1.5], [0.5], [2.0]],
    covars=[[[0.3]], [[0.5]], [[0.8]]]
)

print("\nGaussian HMM Parameters:")
print("=" * 60)
print(f"Means: {gauss_hmm.means.flatten()}")
print(f"Variances: {[c[0,0] for c in gauss_hmm.covars]}")

gauss_hmm.visualize_emissions()
```

## Parameter Constraints

### Ensuring Valid Parameters

```python
def validate_hmm_parameters(pi, A, B=None, means=None, covars=None):
    """
    Validate HMM parameters.
    """
    errors = []
    warnings = []

    n_states = len(pi)

    # Check π
    if not np.isclose(pi.sum(), 1.0):
        errors.append(f"π sums to {pi.sum()}, not 1.0")
    if np.any(pi < 0):
        errors.append("π contains negative values")
    if np.any(pi == 0):
        warnings.append("π contains zeros - some states never start")

    # Check A
    if A.shape != (n_states, n_states):
        errors.append(f"A shape {A.shape} doesn't match n_states={n_states}")
    if not np.allclose(A.sum(axis=1), 1.0):
        errors.append("A rows don't sum to 1")
    if np.any(A < 0):
        errors.append("A contains negative values")
    if np.any(A == 0):
        warnings.append("A contains zeros - some transitions impossible")

    # Check B (discrete)
    if B is not None:
        if B.shape[0] != n_states:
            errors.append(f"B has {B.shape[0]} rows, expected {n_states}")
        if not np.allclose(B.sum(axis=1), 1.0):
            errors.append("B rows don't sum to 1")
        if np.any(B < 0):
            errors.append("B contains negative values")

    # Check Gaussian params
    if means is not None:
        if len(means) != n_states:
            errors.append(f"means has {len(means)} entries, expected {n_states}")

    if covars is not None:
        for i, cov in enumerate(covars):
            # Check positive definite
            try:
                np.linalg.cholesky(cov)
            except:
                errors.append(f"Covariance {i} is not positive definite")

    # Report
    print("Parameter Validation:")
    print("=" * 60)

    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"  ❌ {e}")
    else:
        print("\n✓ All parameters valid")

    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  ⚠ {w}")

    return len(errors) == 0

# Test validation
validate_hmm_parameters(
    pi=gauss_hmm.pi,
    A=gauss_hmm.A,
    means=gauss_hmm.means,
    covars=gauss_hmm.covars
)
```

## Parameter Interpretation

### Financial Market Example

```python
def create_market_regime_hmm():
    """
    Create HMM for market regime detection.
    """
    # 3 states: Bull, Neutral, Bear

    hmm = GaussianHMM(n_states=3, n_features=1)

    # Initial: Market starts neutral
    hmm.set_initial_distribution(pi=[0.2, 0.6, 0.2])

    # Transitions: Regimes are persistent
    hmm.set_transition_matrix(A=[
        [0.90, 0.08, 0.02],  # Bull: stays bull, sometimes neutral
        [0.10, 0.80, 0.10],  # Neutral: can go either way
        [0.02, 0.08, 0.90]   # Bear: stays bear, sometimes neutral
    ])

    # Emissions: Returns distribution by regime
    hmm.set_emission_params(
        means=[[0.10], [0.02], [-0.08]],  # Annualized mean returns
        covars=[[[0.12]], [[0.18]], [[0.25]]]  # Annualized volatility squared
    )

    return hmm

market_hmm = create_market_regime_hmm()

print("Market Regime HMM:")
print("=" * 60)

state_names = ['Bull', 'Neutral', 'Bear']

print("\nInitial probabilities:")
for i, name in enumerate(state_names):
    print(f"  P(start in {name}): {market_hmm.pi[i]:.2f}")

print("\nTransition probabilities:")
for i, from_name in enumerate(state_names):
    print(f"  From {from_name}:")
    for j, to_name in enumerate(state_names):
        print(f"    → {to_name}: {market_hmm.A[i,j]:.2f}")

print("\nEmission parameters (daily returns):")
for i, name in enumerate(state_names):
    daily_mean = market_hmm.means[i, 0] / 252
    daily_vol = np.sqrt(market_hmm.covars[i, 0, 0] / 252)
    print(f"  {name}: μ={daily_mean*100:.3f}%, σ={daily_vol*100:.2f}%")

# Expected regime duration
print("\nExpected regime duration:")
for i, name in enumerate(state_names):
    duration = 1 / (1 - market_hmm.A[i, i])
    print(f"  {name}: {duration:.1f} days")

market_hmm.visualize_emissions()
```

## Key Takeaways

1. **Three parameter sets** fully specify an HMM: $\lambda = (\pi, A, B)$

2. **Initial distribution** $\pi$ determines starting state probabilities

3. **Transition matrix** $A$ controls regime dynamics and persistence

4. **Emission parameters** $B$ link hidden states to observations

5. **Parameter interpretation** is domain-specific - design with application in mind

6. **Validation** ensures mathematical consistency of parameters
