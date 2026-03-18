# Probability Review for Hidden Markov Models

## In Brief

Hidden Markov Models are fundamentally Bayesian inference machines: they use observed data to update beliefs about unobservable hidden states. The five probability concepts that make this inference work are conditional probability, Bayes' theorem, the law of total probability, the chain rule, and conditional independence. Every HMM algorithm reduces to these foundations.

## Key Insight

The central challenge of HMMs is inverting a conditional relationship. We know how hidden states generate observations — the emission model $P(o_t \mid q_t)$ — but we want to infer what states generated a given observation sequence, $P(q_t \mid O)$. Bayes' theorem performs this inversion. Understanding this inversion, and how the Forward-Backward algorithm implements it efficiently across a whole sequence, is the key to understanding HMMs.

## Formal Definition

### Conditional Probability

For events $A$ and $B$ with $P(B) > 0$:
$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

### Bayes' Theorem

$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

In HMM context: $P(\text{state} \mid \text{observations}) = \frac{P(\text{observations} \mid \text{state}) \cdot P(\text{state})}{P(\text{observations})}$

### Law of Total Probability

For a partition $\{B_1, \ldots, B_n\}$ of the sample space:
$$P(A) = \sum_{i=1}^{n} P(A \mid B_i) \cdot P(B_i)$$

This is the formula the Forward algorithm applies at every time step: marginalizing over all possible hidden states.

### Chain Rule

$$P(A_1, A_2, \ldots, A_n) = P(A_1) \cdot \prod_{t=2}^{n} P(A_t \mid A_1, \ldots, A_{t-1})$$

Combined with the Markov property: $P(q_1, \ldots, q_T) = P(q_1) \prod_{t=2}^{T} P(q_t \mid q_{t-1})$

### Conditional Independence

$A$ and $B$ are conditionally independent given $C$ if:
$$P(A, B \mid C) = P(A \mid C) \cdot P(B \mid C)$$

In HMMs: the observation $o_t$ and all future/past observations are conditionally independent given the current state $q_t$. This structural assumption makes dynamic programming tractable.

## Intuitive Explanation

### The Regime Inference Analogy

Suppose we believe the market is in a Bull regime with probability 60% and Bear with probability 40%. We observe a positive daily return (+1.5%). How should we update our beliefs?

- $P(\text{Bull}) = 0.6$, $P(\text{Bear}) = 0.4$ — the **prior**
- $P(+1.5\% \mid \text{Bull}) = 0.7$, $P(+1.5\% \mid \text{Bear}) = 0.3$ — the **likelihood** from our emission model

First, the **marginal probability** of observing +1.5% (law of total probability):
$$P(+1.5\%) = 0.7 \times 0.6 + 0.3 \times 0.4 = 0.54$$

Then, Bayes' theorem gives the **posterior**:
$$P(\text{Bull} \mid +1.5\%) = \frac{0.7 \times 0.6}{0.54} \approx 77.8\%$$

The positive return increased our confidence in the Bull regime from 60% to 78%. This single computation is exactly what the Forward algorithm performs recursively across an entire observation sequence.

### Why Conditional Independence Is Critical

The observation $o_t$ at time $t$ depends only on the hidden state $q_t$ at that same time. Once $q_t$ is known, $o_t$ carries no additional information about past states $q_1, \ldots, q_{t-1}$ or past observations $o_1, \ldots, o_{t-1}$.

Without this assumption, the joint probability $P(O \mid q_1, \ldots, q_T)$ would require tracking all pairwise dependencies among observations — intractable. With it:
$$P(O \mid Q) = \prod_{t=1}^{T} P(o_t \mid q_t)$$

## Code Implementation

```python
import numpy as np
from scipy import stats


def conditional_probability(joint_prob: float, condition_prob: float) -> float:
    """P(A|B) = P(A,B) / P(B)"""
    if condition_prob <= 0:
        raise ValueError("Conditioning event has zero probability")
    return joint_prob / condition_prob


def bayes_update(likelihood: float, prior: float, marginal: float) -> float:
    """P(H|E) = P(E|H) * P(H) / P(E)"""
    if marginal <= 0:
        raise ValueError("Marginal probability must be positive")
    return likelihood * prior / marginal


def total_probability(likelihoods: np.ndarray, priors: np.ndarray) -> float:
    """P(E) = sum_i P(E|H_i) * P(H_i)"""
    return float(np.dot(likelihoods, priors))


class RegimeBeliefUpdater:
    """Illustrates Bayesian updating for a single observation."""

    def __init__(self, n_states: int):
        self.n_states = n_states
        self.prior = np.ones(n_states) / n_states  # Uniform prior

    def update(self, likelihoods: np.ndarray) -> np.ndarray:
        """Update beliefs given emission likelihoods P(obs | state_i).

        This is the single-step version of what the Forward algorithm does.
        """
        joint = likelihoods * self.prior          # P(obs, state_i)
        marginal = joint.sum()                     # P(obs) via total probability
        posterior = joint / (marginal + 1e-300)   # P(state_i | obs) via Bayes
        self.prior = posterior                     # Update prior for next step
        return posterior

    def reset(self, initial_prior: np.ndarray = None):
        if initial_prior is not None:
            self.prior = np.array(initial_prior)
        else:
            self.prior = np.ones(self.n_states) / self.n_states


class GaussianEmission:
    """Single-state Gaussian emission model."""

    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std
        self._dist = stats.norm(loc=mean, scale=std)

    def pdf(self, x: float) -> float:
        return float(self._dist.pdf(x))

    def log_pdf(self, x: float) -> float:
        return float(self._dist.logpdf(x))

    def sample(self, n: int = 1) -> np.ndarray:
        return self._dist.rvs(size=n)


def demonstrate_regime_inference():
    """Walk through Bayesian regime inference on a short return sequence."""
    # Bull regime: positive drift, low volatility
    # Bear regime: negative drift, high volatility
    emissions = [
        GaussianEmission(mean=0.001, std=0.010),   # Bull
        GaussianEmission(mean=-0.001, std=0.020),  # Bear
    ]
    state_names = ["Bull", "Bear"]

    # Prior: 60% Bull, 40% Bear
    prior = np.array([0.6, 0.4])
    updater = RegimeBeliefUpdater(n_states=2)
    updater.prior = prior.copy()

    # Observed returns
    observations = [0.015, -0.005, 0.012, -0.020, 0.008]

    print(f"Initial belief: Bull={prior[0]:.1%}, Bear={prior[1]:.1%}")
    for t, obs in enumerate(observations):
        likelihoods = np.array([e.pdf(obs) for e in emissions])
        posterior = updater.update(likelihoods)
        print(
            f"t={t+1}, obs={obs:+.3f}: "
            f"Bull={posterior[0]:.1%}, Bear={posterior[1]:.1%}"
        )


def log_space_product_demo():
    """Show why log-space is necessary for long sequences."""
    n_steps = 200
    small_prob = 0.05  # Each emission probability

    # Direct multiplication — underflows to zero
    direct = small_prob ** n_steps
    print(f"Direct product ({n_steps} terms of {small_prob}): {direct}")
    # Prints: 0.0 (underflow)

    # Log-space — numerically stable
    log_prod = n_steps * np.log(small_prob)
    log_space_result = np.exp(log_prod)
    print(f"Log-space product: {log_space_result:.4e}")
    # Prints the correct (tiny) number


if __name__ == "__main__":
    demonstrate_regime_inference()
    print()
    log_space_product_demo()
```

## Common Pitfalls

**Prosecutor's fallacy: confusing $P(E \mid H)$ with $P(H \mid E)$.** A high-return day is more likely during a bull market than during a bear market. But observing a high-return day does not mean you are definitely in a bull market. The base rate (prior) matters. Always apply Bayes' theorem rather than intuitively inverting conditional probabilities.

**Base rate neglect.** Even if $P(\text{crash} \mid \text{bear market}) = 0.9$, if bear markets are rare ($P(\text{bear}) = 0.05$), the posterior $P(\text{bear} \mid \text{crash})$ might still be modest. Many HMM practitioners are surprised that a single extreme observation does not strongly shift the regime probability.

**Numerical underflow.** Multiplying 500 emission probabilities together drives the product to zero even for moderately small probabilities. Always work in log space: replace $\prod p_t$ with $\sum \log p_t$. The log-sum-exp trick handles cases where you need to exponentiate the sum.

**Forgetting to normalize after Bayes updates.** After computing the joint $P(\text{obs}, q_t = i) = \alpha_t(i) \cdot \beta_t(i)$, the result is not a probability unless you divide by $P(O \mid \lambda) = \sum_i \alpha_t(i) \cdot \beta_t(i)$. Failure to normalize gives gamma values that do not sum to 1, silently corrupting the Baum-Welch M-step.

**Assuming independence when returns are autocorrelated.** The HMM output independence assumption says $o_t \perp o_{t-1} \mid q_t$. Real financial returns exhibit autocorrelation (especially in volatility). A Markov-Switching Autoregressive model (Module 05) relaxes this assumption.

## Connections

**Forward algorithm**: Each time step of the Forward algorithm computes exactly the Bayesian update shown in the regime inference example, but using forward variables $\alpha_t(i)$ that encode the joint probability $P(o_1, \ldots, o_t, q_t = i)$.

**Backward algorithm**: Uses the law of total probability in the reverse direction, computing $\beta_t(i) = P(o_{t+1}, \ldots, o_T \mid q_t = i)$ by marginalizing over all possible next states.

**Gamma and Xi (Baum-Welch E-step)**: The posterior state probability $\gamma_t(i) = P(q_t = i \mid O)$ is computed by combining forward ($\alpha$) and backward ($\beta$) variables through Bayes' theorem: $\gamma_t(i) \propto \alpha_t(i) \cdot \beta_t(i)$.

**Conditional independence structure**: The HMM graphical model explicitly encodes which random variables are conditionally independent. This structure is what allows the $O(TK^2)$ dynamic programming algorithms to be correct — each local computation assumes that future observations are conditionally independent of past observations given the current state.

## Practice Problems

**Problem 1.** A diagnostic test for a rare market regime has sensitivity 95% (correctly identifies the regime when it is present) and specificity 90% (correctly identifies the absence). The regime occurs 5% of the time. What is the probability the regime is actually present when the test is positive? Work through this numerically.

*Answer*: $P(\text{regime} \mid \text{positive}) = \frac{0.95 \times 0.05}{0.95 \times 0.05 + 0.10 \times 0.95} \approx 33\%$. Counterintuitively low because the base rate is small.

**Problem 2.** Extend the `RegimeBeliefUpdater` class to handle a three-state model (Bull, Neutral, Bear). Use these emission parameters:
- Bull: $\mathcal{N}(0.002, 0.010)$
- Neutral: $\mathcal{N}(0.000, 0.015)$
- Bear: $\mathcal{N}(-0.002, 0.025)$

Generate 100 observations from a known sequence of states (50 Bull, 30 Neutral, 20 Bear) and apply the updater. Plot how the belief about each state evolves over time.

**Problem 3.** Implement the log-sum-exp trick:
$$\log \sum_i e^{x_i} = x_{\max} + \log \sum_i e^{x_i - x_{\max}}$$

Demonstrate that it correctly computes `np.log(sum(np.exp(x)))` for a vector $x$ where the naive computation would overflow or underflow.

**Problem 4.** Prove algebraically that the chain rule for the joint probability of an HMM state sequence and observation sequence factorizes as:
$$P(O, Q \mid \lambda) = \pi_{q_1} b_{q_1}(o_1) \prod_{t=2}^{T} a_{q_{t-1}, q_t} b_{q_t}(o_t)$$

using the Markov property and conditional independence of observations given states.

**Problem 5.** The conditional independence assumption says $P(o_t \mid q_t, o_1, \ldots, o_{t-1}) = P(o_t \mid q_t)$. Design a simulation test that checks whether this assumption holds for a fitted Gaussian HMM on real financial return data. Hint: fit the HMM, decode the most likely state sequence, then test whether the returns are autocorrelated within each state.

## Further Reading

- **Bishop (2006)** — *Pattern Recognition and Machine Learning*, Chapter 8. Covers graphical models and conditional independence in full generality.
- **Murphy (2012)** — *Machine Learning: A Probabilistic Perspective*, Chapter 17. Comprehensive treatment of HMMs with Bayesian perspective.
- **Jaynes (2003)** — *Probability Theory: The Logic of Science*. Deep treatment of Bayesian inference; Chapter 4 covers the base rate / prosecutor's fallacy in detail.
- **Rabiner (1989)** — "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition." *Proceedings of the IEEE*, 77(2). The classic reference; Section II reviews the probability foundations specifically for HMMs.
