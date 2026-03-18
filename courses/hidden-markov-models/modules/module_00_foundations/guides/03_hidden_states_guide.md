# Hidden States: Why "Hidden" in Hidden Markov Models

## In Brief

In a standard Markov chain, the state at each time step is directly observed. A Hidden Markov Model adds a second layer: the states evolve according to a Markov chain, but we cannot observe them. Instead, each state generates an observation through an emission distribution. The "hidden" terminology reflects this observational gap — we see only the emissions, not the states that generated them. All HMM algorithms are techniques for reasoning about hidden states from observable evidence.

## Key Insight

The key value of hidden states is **structured dimension reduction**. Instead of modeling a potentially infinite observation space directly (stock returns can take any real value), we postulate a small number of discrete hidden states (2–5) and model each state's emission distribution separately. A single +2% return is ambiguous — it could come from a bull market or just a lucky bear market day. But a *sequence* of mostly positive returns with low volatility is strong evidence for the bull state. Hidden states let the model accumulate and weigh this sequence-level evidence probabilistically.

## Formal Definition

An HMM $\lambda = (\pi, A, B)$ consists of:

| Component | Symbol | Definition |
|-----------|--------|------------|
| State space | $S = \{s_1, \ldots, s_K\}$ | Finite set of $K$ hidden states |
| Initial distribution | $\pi_i = P(q_1 = s_i)$ | Probability of starting in state $i$ |
| Transition matrix | $a_{ij} = P(q_{t+1} = s_j \mid q_t = s_i)$ | Hidden state dynamics |
| Emission distribution | $b_i(o) = P(o_t = o \mid q_t = s_i)$ | How states generate observations |

Two independence assumptions make inference tractable:

1. **First-order Markov property**: $P(q_t \mid q_1, \ldots, q_{t-1}) = P(q_t \mid q_{t-1})$
2. **Output independence**: $P(o_t \mid q_1, \ldots, q_T, o_1, \ldots, o_{t-1}) = P(o_t \mid q_t)$

Together they mean the joint probability of states and observations factorizes as:
$$P(O, Q \mid \lambda) = \pi_{q_1} b_{q_1}(o_1) \prod_{t=2}^{T} a_{q_{t-1}, q_t} b_{q_t}(o_t)$$

### Three Fundamental Problems

| Problem | Question | Algorithm |
|---------|----------|-----------|
| Evaluation | $P(O \mid \lambda) = ?$ | Forward algorithm |
| Decoding | $\arg\max_Q P(Q \mid O, \lambda) = ?$ | Viterbi algorithm |
| Learning | $\arg\max_\lambda P(O \mid \lambda) = ?$ | Baum-Welch (EM) |

## Intuitive Explanation

### The Weather-Behind-the-Window Analogy

You are locked in a windowless room. You cannot see the weather outside (the hidden state), but you can observe what your friend wears when they arrive:

```
Hidden States (Weather):   Sunny → Sunny → Rainy → Rainy → Sunny
                             ↓        ↓       ↓        ↓       ↓
Observations (Clothing):  Shorts  T-shirt  Coat  Umbrella  Shorts
```

Given only the clothing sequence, you want to infer the weather sequence. Each clothing item is ambiguous — someone might wear a coat on a cool sunny day — but the pattern over multiple days narrows it down.

### Financial Markets Are Hidden

No single number tells you the market is in a bull or bear regime. Regime is a latent concept:
- Returns today = +2% could come from a bull state (high probability) or a bear state (lower probability)
- Volatility today = 15% annualized is normal in bull markets but low for bear markets
- The *sequence* of returns and volatilities over weeks is what distinguishes regimes

Why bother with hidden states instead of observable clustering? Three reasons:

1. **Continuity across missing data**: The Markov chain transitions even when observations are missing, providing principled interpolation.
2. **Temporal coherence**: States encode persistence — the model learns that bull markets typically last 20+ days, not 1–2 days.
3. **Probabilistic uncertainty**: Instead of hard "bull/bear" labels, the model provides $P(\text{bull at time } t \mid \text{all observations})$, enabling uncertainty-aware position sizing.

### Why Observations Are Ambiguous

The overlap between emission distributions is fundamental:

```python
import numpy as np
from scipy import stats

# Bull: positive drift, low volatility
bull = stats.norm(loc=0.001, scale=0.010)
# Bear: negative drift, high volatility
bear = stats.norm(loc=-0.001, scale=0.020)

# A return of 0.0 has non-zero probability under BOTH states
obs = 0.0
print(f"P(return=0.0 | Bull) = {bull.pdf(obs):.3f}")
print(f"P(return=0.0 | Bear) = {bear.pdf(obs):.3f}")
# P(return=0.0 | Bull) ≈ 39.9
# P(return=0.0 | Bear) ≈ 19.9
# We cannot deterministically assign this observation to a state
```

The overlap means we need inference algorithms, not just a lookup table.

## Code Implementation

```python
import numpy as np
from scipy import stats
from typing import List, Tuple


class GaussianEmission:
    """Per-state Gaussian emission distribution."""

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


class HiddenMarkovModel:
    """Simple HMM with Gaussian emissions for illustration.

    For production use, prefer hmmlearn.hmm.GaussianHMM.
    This class makes the distinction between hidden and observed explicit.
    """

    def __init__(
        self,
        pi: np.ndarray,
        A: np.ndarray,
        emissions: List[GaussianEmission],
    ):
        self.pi = np.array(pi)
        self.A = np.array(A)
        self.emissions = emissions
        self.n_states = len(pi)

        if not np.isclose(self.pi.sum(), 1.0):
            raise ValueError("pi must sum to 1")
        if not np.allclose(self.A.sum(axis=1), 1.0):
            raise ValueError("A must be row-stochastic")

    def simulate(self, n_steps: int) -> Tuple[List[int], List[float]]:
        """Generate a sequence of hidden states AND observations.

        Returns (hidden_states, observations) — in a real problem we
        would only observe the second list.
        """
        # Sample initial state
        state = int(np.random.choice(self.n_states, p=self.pi))
        hidden_states = [state]
        observations = [float(self.emissions[state].sample(1)[0])]

        for _ in range(n_steps - 1):
            # Transition to next hidden state
            state = int(np.random.choice(self.n_states, p=self.A[state]))
            hidden_states.append(state)
            # Emit an observation from the new state
            observations.append(float(self.emissions[state].sample(1)[0]))

        return hidden_states, observations

    def emission_matrix(self, observations: List[float]) -> np.ndarray:
        """Compute emission probability matrix B[t, k] = P(o_t | state k).

        Shape: (T, K)
        """
        T = len(observations)
        B = np.zeros((T, self.n_states))
        for t, obs in enumerate(observations):
            for k in range(self.n_states):
                B[t, k] = self.emissions[k].pdf(obs)
        return B

    def joint_probability(
        self, states: List[int], observations: List[float]
    ) -> float:
        """Compute P(O, Q | lambda) using the factorization.

        This is the quantity the Forward algorithm marginalizes over all Q.
        For any sequence longer than ~20, use log_joint_probability instead.
        """
        prob = self.pi[states[0]] * self.emissions[states[0]].pdf(observations[0])
        for t in range(1, len(observations)):
            prob *= self.A[states[t - 1], states[t]]
            prob *= self.emissions[states[t]].pdf(observations[t])
        return prob

    def log_joint_probability(
        self, states: List[int], observations: List[float]
    ) -> float:
        """Log-space version of joint_probability. Use for sequences > 20 steps."""
        log_prob = (
            np.log(self.pi[states[0]] + 1e-300)
            + self.emissions[states[0]].log_pdf(observations[0])
        )
        for t in range(1, len(observations)):
            log_prob += np.log(self.A[states[t - 1], states[t]] + 1e-300)
            log_prob += self.emissions[states[t]].log_pdf(observations[t])
        return float(log_prob)


def demonstrate_observation_ambiguity():
    """Show that a single observation cannot determine the hidden state."""
    bull_emission = GaussianEmission(mean=0.001, std=0.010)
    bear_emission = GaussianEmission(mean=-0.001, std=0.020)
    emissions = [bull_emission, bear_emission]
    state_names = ["Bull", "Bear"]

    test_observations = [-0.025, -0.005, 0.000, +0.005, +0.025]
    print("Observation ambiguity — P(obs | state):")
    print(f"{'obs':>8} | {'Bull PDF':>10} | {'Bear PDF':>10} | {'Ratio':>8}")
    print("-" * 45)
    for obs in test_observations:
        p_bull = bull_emission.pdf(obs)
        p_bear = bear_emission.pdf(obs)
        ratio = p_bull / (p_bear + 1e-300)
        print(f"{obs:+8.3f} | {p_bull:10.3f} | {p_bear:10.3f} | {ratio:8.2f}")
    print("Even large positive returns have non-zero P under Bear state.")


def compare_observable_vs_hidden():
    """Contrast what we observe vs. what we want to infer."""
    np.random.seed(42)

    # Known parameters
    A = np.array([[0.95, 0.05], [0.10, 0.90]])
    pi = np.array([0.6, 0.4])
    emissions = [
        GaussianEmission(mean=0.001, std=0.010),   # Bull
        GaussianEmission(mean=-0.001, std=0.020),  # Bear
    ]
    hmm = HiddenMarkovModel(pi, A, emissions)

    # Simulate 20 steps
    hidden_states, observations = hmm.simulate(20)

    print("What the model knows (hidden states):")
    print(" ".join(["Bull" if s == 0 else "Bear" for s in hidden_states]))
    print("\nWhat we observe (returns):")
    print(" ".join([f"{o:+.4f}" for o in observations]))
    print("\nOur task: infer the top row from only the bottom row.")


if __name__ == "__main__":
    demonstrate_observation_ambiguity()
    print()
    compare_observable_vs_hidden()
```

## Common Pitfalls

**Using too many hidden states.** More states always reduce training loss but hurt generalization. For financial regime detection, 2–3 states almost always suffice: bull/bear or bull/neutral/bear. Start with 2, then check BIC before adding more.

**Treating hard-decoded states as ground truth.** The Viterbi state sequence is the *most likely* sequence, not the *true* sequence. In periods of regime uncertainty ($P(\text{bull}) \approx 0.5$), hard labels can be misleading. Use posterior probabilities from forward-backward for decision-making, not Viterbi labels.

**Ignoring the output independence assumption.** The model assumes that $o_t$ depends only on $q_t$, not on $o_{t-1}$. Financial returns exhibit volatility clustering (GARCH effects) that violates this within a regime. If residual autocorrelation is detected within decoded states, consider the Markov-Switching AR model (Module 05).

**Confusing filtering and smoothing.** Filtering gives $P(q_t \mid o_1, \ldots, o_t)$ — the regime probability using only past observations. Smoothing gives $P(q_t \mid o_1, \ldots, o_T)$ — using all observations including future ones. For live trading, only filtering is available. For backtesting, smoothing (forward-backward) uses future information and will create look-ahead bias.

**Expecting the model to identify economically meaningful regimes.** HMMs find statistically distinct emission clusters. The states that emerge might not correspond to "bull/bear" in any human-interpretable way. Always post-hoc interpret learned parameters: if state 0 has higher mean and lower variance, label it bull.

## Connections

**Observable Markov chain (Module 00)**: An observable Markov chain is a degenerate HMM where the emission distribution for state $i$ places all probability mass on the single symbol $i$. HMMs generalize this by allowing stochastic emissions.

**Forward algorithm (Module 02)**: Solves the evaluation problem by computing $P(O \mid \lambda)$ in $O(TK^2)$ time. It uses the output independence assumption to factorize the joint probability and sum efficiently over hidden states using dynamic programming.

**Viterbi algorithm (Module 02)**: Solves the decoding problem. Instead of summing over all hidden state sequences, it finds the single most likely sequence using the same dynamic programming structure as the Forward algorithm, but with `max` instead of `sum`.

**Gaussian HMM (Module 03)**: Extends the emission distribution from discrete (categorical) to Gaussian, making the model appropriate for continuous observations like financial returns.

**State-space models**: HMMs are discrete-state state-space models. Kalman Filters are continuous-state state-space models. Both model the same latent-observable structure but make different assumptions about state and observation distributions.

## Practice Problems

**Problem 1.** Implement an `ObservableMarkovChain` class and a `HiddenMarkovModel` class. The key difference: in the observable chain, `simulate()` returns only states; in the HMM, it returns both states and observations. Write a function that verifies the output independence assumption: given the true state sequence, check that the autocorrelation of residuals $(o_t - \mu_{q_t})$ is near zero.

**Problem 2.** Using the bull/bear HMM from the code above, simulate 1000 steps. Plot the observations as a time series. Add vertical shading for the true bull/bear states. Can you visually identify the regimes from the return series alone? What does this tell you about why inference algorithms are necessary?

**Problem 3.** Compute $P(O, Q \mid \lambda)$ for three candidate state sequences on the first 10 observations of your simulation:
- (a) The true hidden state sequence
- (b) A sequence that is all Bull
- (c) A sequence that is all Bear

Which sequence has the highest joint probability? Is it the true sequence?

**Problem 4.** For a 2-state HMM with transition matrix $A$ and initial distribution $\pi$, derive the marginal probability $P(o_T)$ by expanding the law of total probability over all $K^T$ possible state sequences. Then show that the forward algorithm computes the same result in $O(TK^2)$ operations. What is the ratio of computation saved for $K=2$, $T=100$?

**Problem 5.** Design an experiment to test whether the number of hidden states $K$ affects the quality of regime detection. Simulate data from a 3-state HMM. Fit HMMs with $K = 2, 3, 4$ states. For each, compute BIC and the accuracy of Viterbi decoded states against the true states. Plot BIC vs. $K$ and accuracy vs. $K$. What do you observe?

## Further Reading

- **Rabiner (1989)** — "A Tutorial on Hidden Markov Models." Section III defines the three fundamental problems and explains the structure of hidden states in detail.
- **Durbin et al. (1998)** — *Biological Sequence Analysis*. Chapter 3 applies HMMs to biological sequences, providing a complete worked example of hidden state inference.
- **Ang & Bekaert (2002)** — "Regime Switches in Interest Rates." *Journal of Business and Economic Statistics*. Seminal application of HMMs to regime detection in finance.
- **Kim & Nelson (1999)** — *State-Space Models with Regime Switching*. MIT Press. Comprehensive treatment of hidden state models in econometrics.
