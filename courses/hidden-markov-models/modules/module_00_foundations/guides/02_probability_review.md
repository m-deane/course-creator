# Probability Review for Hidden Markov Models

> **Reading time:** ~9 min | **Module:** Module 0: Foundations | **Prerequisites:** Basic linear algebra, Python

<div class="callout-key">

**Key Concept Summary:** Essential probability concepts for HMMs: conditional probability, Bayes' theorem, probability distributions, and the law of total probability. These form the mathematical foundation for understanding how hidden states generate observable sequences.

</div>

## In Brief

Essential probability concepts for HMMs: conditional probability, Bayes' theorem, probability distributions, and the law of total probability. These form the mathematical foundation for understanding how hidden states generate observable sequences.

<div class="callout-insight">

**Insight:** HMMs are fundamentally about computing conditional probabilities of hidden states given observations. Bayes' theorem lets us invert relationships: from knowing P(observation|state) to inferring P(state|observation).

</div>
<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## Intuitive Explanation

**Conditional Probability** is updating beliefs with new information. If you know the market was in a bull regime, what's the probability of positive returns?

**Bayes' Theorem** flips this around. You observe positive returns—what's the probability the market is in a bull regime?

**Law of Total Probability** says to find the probability of an observation, consider all possible hidden states that could generate it, weighted by how likely each state is.

## Formal Definition

### Conditional Probability

The probability of event A given event B has occurred:

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

provided $P(B) > 0$.

### Bayes' Theorem

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

In HMM context:
$$P(\text{state}|\text{observations}) = \frac{P(\text{observations}|\text{state}) \cdot P(\text{state})}{P(\text{observations})}$$

### Law of Total Probability

For a partition ${B_1, B_2, ..., B_n}$ of the sample space:

$$P(A) = \sum_{i=1}^{n} P(A|B_i) \cdot P(B_i)$$

### Joint vs Marginal Distributions

For random variables $X$ and $Y$:
- **Joint:** $P(X=x, Y=y)$
- **Marginal:** $P(X=x) = \sum_y P(X=x, Y=y)$
- **Conditional:** $P(X=x|Y=y) = \frac{P(X=x, Y=y)}{P(Y=y)}$

### The HMM Connection

```
Hidden States (unknown):  S₁ → S₂ → S₃ → ...
                           ↓    ↓    ↓
Observations (known):      O₁   O₂   O₃   ...

Question: Given we see O₁, O₂, O₃, what states generated them?

This requires: P(S₁,S₂,S₃|O₁,O₂,O₃) ← Bayes' theorem!
```

## Mathematical Formulation

### Chain Rule of Probability

$$P(A_1, A_2, ..., A_n) = P(A_1) \cdot P(A_2|A_1) \cdot P(A_3|A_1, A_2) \cdots P(A_n|A_1, ..., A_{n-1})$$

For HMM state sequences:

$$P(q_1, q_2, ..., q_T) = P(q_1) \prod_{t=2}^{T} P(q_t | q_{t-1})$$

This simplification uses the Markov property!

### Independence

Events A and B are independent if:

$$P(A \cap B) = P(A) \cdot P(B)$$

Equivalently: $P(A|B) = P(A)$

**Conditional Independence:** A and B are conditionally independent given C if:

$$P(A, B | C) = P(A|C) \cdot P(B|C)$$

In HMMs: Future and past are conditionally independent given the present state.

$$P(q_{t+1}, q_{t+2}, ... | q_t) = P(q_{t+1} | q_t) \cdot P(q_{t+2} | q_{t+1}) \cdots$$

## Code Implementation


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">regimeinference.py</span>
</div>

```python
import numpy as np
from typing import List, Tuple

def conditional_probability(joint_prob: float, condition_prob: float) -> float:
    """
    Compute P(A|B) = P(A,B) / P(B)

    Args:
        joint_prob: P(A ∩ B)
        condition_prob: P(B)

    Returns:
        P(A|B)
    """
    if condition_prob == 0:
        raise ValueError("Conditioning event has zero probability")
    return joint_prob / condition_prob


def bayes_theorem(
    likelihood: float,
    prior: float,
    marginal: float
) -> float:
    """
    Compute P(A|B) using Bayes' theorem.

    Args:
        likelihood: P(B|A) - probability of evidence given hypothesis
        prior: P(A) - prior probability of hypothesis
        marginal: P(B) - probability of evidence

    Returns:
        P(A|B) - posterior probability of hypothesis
    """
    return (likelihood * prior) / marginal


def total_probability(
    conditional_probs: List[float],
    partition_probs: List[float]
) -> float:
    """
    Compute P(A) using law of total probability.

    P(A) = Σᵢ P(A|Bᵢ) · P(Bᵢ)

    Args:
        conditional_probs: [P(A|B₁), P(A|B₂), ...]
        partition_probs: [P(B₁), P(B₂), ...]

    Returns:
        P(A)
    """
    return np.dot(conditional_probs, partition_probs)


# Example: Market regime inference
class RegimeInference:
    """Bayesian inference for market regimes."""

    def __init__(self):
        # Prior probabilities of regimes
        self.p_bull = 0.6
        self.p_bear = 0.4

        # Likelihoods: P(positive return | regime)
        self.p_positive_given_bull = 0.7
        self.p_positive_given_bear = 0.3

    def marginal_positive_return(self) -> float:
        """
        P(positive return) using law of total probability.
        """
        return total_probability(
            conditional_probs=[self.p_positive_given_bull, self.p_positive_given_bear],
            partition_probs=[self.p_bull, self.p_bear]
        )

    def posterior_bull_given_positive(self) -> float:
        """
        P(bull | positive return) using Bayes' theorem.
        """
        marginal = self.marginal_positive_return()
        return bayes_theorem(
            likelihood=self.p_positive_given_bull,
            prior=self.p_bull,
            marginal=marginal
        )

    def posterior_bull_given_negative(self) -> float:
        """
        P(bull | negative return) using Bayes' theorem.
        """
        p_negative_given_bull = 1 - self.p_positive_given_bull
        p_negative_given_bear = 1 - self.p_positive_given_bear

        marginal = total_probability(
            conditional_probs=[p_negative_given_bull, p_negative_given_bear],
            partition_probs=[self.p_bull, self.p_bear]
        )

        return bayes_theorem(
            likelihood=p_negative_given_bull,
            prior=self.p_bull,
            marginal=marginal
        )


# Usage
regime_model = RegimeInference()

print("Prior Probabilities:")
print(f"  P(Bull) = {regime_model.p_bull:.2%}")
print(f"  P(Bear) = {regime_model.p_bear:.2%}")

print("\nLikelihoods:")
print(f"  P(+return | Bull) = {regime_model.p_positive_given_bull:.2%}")
print(f"  P(+return | Bear) = {regime_model.p_positive_given_bear:.2%}")

print("\nPosterior Probabilities:")
print(f"  P(Bull | +return) = {regime_model.posterior_bull_given_positive():.2%}")
print(f"  P(Bull | -return) = {regime_model.posterior_bull_given_negative():.2%}")
```

</div>
</div>

### Discrete Probability Distributions


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">discreteprobdist.py</span>
</div>

```python
class DiscreteProbDist:
    """Discrete probability distribution."""

    def __init__(self, outcomes: List, probabilities: np.ndarray):
        """
        Args:
            outcomes: Possible values
            probabilities: P(X = outcome) for each outcome
        """
        self.outcomes = outcomes
        self.probs = np.array(probabilities)

        # Validate
        assert len(outcomes) == len(probabilities), "Mismatched lengths"
        assert np.allclose(self.probs.sum(), 1.0), "Probabilities must sum to 1"
        assert np.all(self.probs >= 0), "Probabilities must be non-negative"

    def sample(self, n: int = 1) -> List:
        """Draw samples from the distribution."""
        indices = np.random.choice(len(self.outcomes), size=n, p=self.probs)
        return [self.outcomes[i] for i in indices]

    def expectation(self, function=None) -> float:
        """
        Compute E[f(X)] = Σ f(x) · P(X=x)

        If function is None, computes E[X] assuming outcomes are numeric.
        """
        if function is None:
            function = lambda x: x

        return sum(function(outcome) * prob
                  for outcome, prob in zip(self.outcomes, self.probs))

    def variance(self) -> float:
        """Compute Var(X) = E[X²] - E[X]²"""
        mean = self.expectation()
        second_moment = self.expectation(lambda x: x**2)
        return second_moment - mean**2


# Example: State emissions
states = ['Bull', 'Bear']
state_probs = np.array([0.6, 0.4])

state_dist = DiscreteProbDist(states, state_probs)

print("\nState Distribution:")
print(f"  E[State] (not meaningful for categorical)")
samples = state_dist.sample(10)
print(f"  Sample sequence: {samples}")
```

</div>
</div>

### Continuous Distributions


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">gaussiandistribution.py</span>
</div>

```python
from scipy import stats

class GaussianDistribution:
    """Gaussian (Normal) distribution."""

    def __init__(self, mean: float, variance: float):
        self.mean = mean
        self.variance = variance
        self.std = np.sqrt(variance)
        self.dist = stats.norm(loc=mean, scale=self.std)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density function."""
        return self.dist.pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Cumulative distribution function."""
        return self.dist.cdf(x)

    def sample(self, n: int = 1) -> np.ndarray:
        """Draw samples."""
        return self.dist.rvs(size=n)

    def log_likelihood(self, observations: np.ndarray) -> float:
        """Log-likelihood of observations."""
        return np.sum(self.dist.logpdf(observations))


# Example: Return distributions in different regimes
bull_returns = GaussianDistribution(mean=0.05, variance=0.01)  # 5% mean, 10% vol
bear_returns = GaussianDistribution(mean=-0.03, variance=0.04)  # -3% mean, 20% vol

x = np.linspace(-0.15, 0.15, 200)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(x, bull_returns.pdf(x), label='Bull Regime', linewidth=2)
plt.plot(x, bear_returns.pdf(x), label='Bear Regime', linewidth=2)
plt.axvline(0, color='black', linestyle='--', alpha=0.3)
plt.xlabel('Return')
plt.ylabel('Probability Density')
plt.title('Return Distributions by Regime')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Probability of positive return in each regime
print(f"P(return > 0 | Bull) = {1 - bull_returns.cdf(0):.2%}")
print(f"P(return > 0 | Bear) = {1 - bear_returns.cdf(0):.2%}")
```

</div>
</div>

## Common Pitfalls

1. **Confusing P(A|B) with P(B|A)**
   - P(positive return | bull market) ≠ P(bull market | positive return)
   - Always use Bayes' theorem to flip conditional probabilities
   - Example: P(symptoms|disease) is not the same as P(disease|symptoms)

2. **Ignoring the base rate (prior)**
   - Posterior depends on both likelihood AND prior
   - Rare events (low prior) need strong evidence (high likelihood) to become probable
   - In finance: Don't ignore long-run regime frequencies

3. **Assuming independence when it doesn't hold**
   - Returns are often autocorrelated
   - Volatility clusters (high vol follows high vol)
   - Check assumptions before using independence

4. **Forgetting to normalize**
   - Probabilities must sum to 1
   - When using Bayes' theorem, ensure marginal is computed correctly
   - In HMMs: State probabilities and transition probabilities must be proper distributions

5. **Numerical underflow in products**
   - Multiplying many small probabilities → zero
   - Solution: Work in log space
   - log(a × b) = log(a) + log(b)


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Bad: Numerical underflow
probs = [0.01] * 100
product = np.prod(probs)  # → 0.0 (underflow!)

# Good: Log-space computation
log_probs = np.log([0.01] * 100)
log_product = np.sum(log_probs)
product = np.exp(log_product)  # Accurate

print(f"Direct product: {np.prod(probs)}")
print(f"Log-space product: {np.exp(np.sum(np.log(probs)))}")
```


## Connections

### Builds on:
- Basic probability axioms
- Set theory
- Calculus (for continuous distributions)

### Leads to:
- **Markov property**: Conditional independence of future and past given present
- **HMM forward algorithm**: Uses law of total probability
- **HMM backward algorithm**: Computes conditional probabilities
- **Baum-Welch (EM)**: Iterative application of Bayes' theorem
- **State decoding**: Posterior probabilities P(state|observations)

### Related to:
- **Bayesian inference**: Updating beliefs with evidence
- **Information theory**: KL divergence measures distribution differences
- **Statistical inference**: Maximum likelihood estimation

## Practice Problems

1. **Conditional Probability**

   A stock has 60% probability of being in a bull regime. Given a bull regime, there's a 70% probability of a positive daily return. What's the probability of both bull regime AND positive return?

   <details>
   <summary>Solution</summary>

   P(Bull ∩ Positive) = P(Positive|Bull) × P(Bull) = 0.7 × 0.6 = 0.42
   </details>

2. **Bayes' Theorem Application**

   Using the regime model:
   - P(Bull) = 0.6, P(Bear) = 0.4
   - P(+return|Bull) = 0.7, P(+return|Bear) = 0.3

   You observe a positive return. What's P(Bull|+return)?

   <details>
   <summary>Solution</summary>

   P(Bull|+return) = P(+return|Bull) × P(Bull) / P(+return)

   P(+return) = P(+return|Bull)×P(Bull) + P(+return|Bear)×P(Bear)
              = 0.7×0.6 + 0.3×0.4 = 0.42 + 0.12 = 0.54

   P(Bull|+return) = (0.7 × 0.6) / 0.54 = 0.42 / 0.54 ≈ 0.778 = 77.8%
   </details>

3. **Law of Total Probability**

   A 3-state HMM has states {Low Vol, Medium Vol, High Vol} with probabilities {0.5, 0.3, 0.2}. The probability of a large price move in each state is {0.1, 0.3, 0.7}. What's the overall probability of a large price move?

   <details>
   <summary>Solution</summary>

   P(Large Move) = Σᵢ P(Large Move|State_i) × P(State_i)
                 = 0.1×0.5 + 0.3×0.3 + 0.7×0.2
                 = 0.05 + 0.09 + 0.14
                 = 0.28 = 28%
   </details>

4. **Sequential Probability**

   In a 2-state Markov chain with transition matrix:
   ```
   A = [[0.8, 0.2],
        [0.3, 0.7]]
   ```
   Starting in state 0, what's the probability of the sequence: 0 → 0 → 1 → 1?

   <details>
   <summary>Solution</summary>

   P(0,0,1,1) = P(q₁=0) × P(q₂=0|q₁=0) × P(q₃=1|q₂=0) × P(q₄=1|q₃=1)
              = 1.0 × 0.8 × 0.2 × 0.7
              = 0.112 = 11.2%
   </details>

<div class="callout-insight">

**Insight:** Understanding probability review for hidden markov models is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.


## Further Reading

### Essential References

1. **DeGroot & Schervish (2012)** - "Probability and Statistics" (4th ed.)
   - Comprehensive probability foundations
   - Chapter 2: Conditional Probability
   - Chapter 3: Discrete and Continuous Distributions

2. **Bishop (2006)** - "Pattern Recognition and Machine Learning"
   - Section 1.2: Probability Theory
   - Excellent treatment of Bayes' theorem in ML context
   - Connects to graphical models (HMMs are a special case)

3. **Bertsekas & Tsitsiklis (2008)** - "Introduction to Probability" (2nd ed.)
   - Rigorous yet accessible
   - Chapter 1: Conditional Probability
   - Chapter 3: Random Variables

### HMM-Specific

4. **Rabiner (1989)** - "A Tutorial on Hidden Markov Models"
   - Section II: Discrete Markov processes review
   - How probability concepts apply to HMMs
   - Classic reference paper

5. **Murphy (2012)** - "Machine Learning: A Probabilistic Perspective"
   - Chapter 2: Probability review
   - Chapter 17: Markov and HMMs
   - Modern, comprehensive treatment

### Online Resources

6. **Seeing Theory** - https://seeing-theory.brown.edu/
   - Interactive probability visualizations
   - Excellent for building intuition

7. **Khan Academy** - Probability and Statistics
   - Free video lectures on conditional probability
   - Good refresher for basics

**Next Steps:** With this probability foundation, you're ready to understand Markov chains (which add the temporal dimension) and Hidden Markov Models (which add the hidden state layer).

---

## Conceptual Practice Questions

1. Why is a solid understanding of conditional probability essential for HMMs?

2. Explain Bayes' theorem in the context of updating regime beliefs given new market data.

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.


---

## Cross-References

<a class="link-card" href="./02_probability_review_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_markov_chain_basics.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_markov_chains.md">
  <div class="link-card-title">01 Markov Chains</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_transition_matrices.md">
  <div class="link-card-title">02 Transition Matrices</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

