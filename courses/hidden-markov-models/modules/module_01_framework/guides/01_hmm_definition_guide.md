# HMM Framework and Definition

## In Brief

A Hidden Markov Model is completely specified by the triple $\lambda = (\pi, A, B)$: the initial state distribution, the transition matrix, and the emission parameters. Everything else — regime detection, parameter learning, sequence generation, likelihood evaluation — follows from this compact specification. This guide formalizes the definition and works through the three fundamental problems that organize all HMM algorithms.

## Key Insight

Adding an emission layer to a Markov chain creates an exponential combinatorial inference problem: there are $K^T$ possible hidden state sequences for $T$ observations with $K$ states. The insight that makes HMMs tractable is that the conditional independence assumptions in the graphical model allow this exponential sum or maximization to be decomposed into $T$ steps of $O(K^2)$ work each — reducing complexity from $O(K^T)$ to $O(TK^2)$.

## Formal Definition

An HMM $\lambda = (A, B, \pi)$ over a state space $S = \{s_1, \ldots, s_K\}$ and observation sequence $O = (o_1, \ldots, o_T)$ is defined by:

**Initial state distribution**: $\pi_i = P(q_1 = s_i)$ for $i = 1, \ldots, K$
- Constraints: $\pi_i \geq 0$, $\sum_i \pi_i = 1$

**Transition matrix**: $a_{ij} = P(q_{t+1} = s_j \mid q_t = s_i)$
- Constraints: $a_{ij} \geq 0$, $\sum_j a_{ij} = 1$

**Emission distribution**: $b_i(o) = P(o_t = o \mid q_t = s_i)$
- Discrete case: $b_{ij} = P(o_t = v_j \mid q_t = s_i)$, rows sum to 1
- Continuous case: $b_i(o) = \mathcal{N}(o; \mu_i, \Sigma_i)$ (Gaussian HMM)

### Joint Probability Factorization

$$P(O, Q \mid \lambda) = \pi_{q_1} \cdot b_{q_1}(o_1) \cdot \prod_{t=2}^{T} a_{q_{t-1}, q_t} \cdot b_{q_t}(o_t)$$

### The Three Fundamental Problems

**Problem 1 — Evaluation**: Given $\lambda$ and $O$, compute $P(O \mid \lambda)$.
$$P(O \mid \lambda) = \sum_{Q} P(O, Q \mid \lambda)$$
Solved by the **Forward algorithm** in $O(TK^2)$.

**Problem 2 — Decoding**: Given $\lambda$ and $O$, find:
$$Q^* = \arg\max_Q P(Q \mid O, \lambda) = \arg\max_Q P(O, Q \mid \lambda)$$
Solved by the **Viterbi algorithm** in $O(TK^2)$.

**Problem 3 — Learning**: Given $O$ (and model structure $K$), find:
$$\lambda^* = \arg\max_\lambda P(O \mid \lambda)$$
Solved by **Baum-Welch (EM)**, guaranteed to increase $P(O \mid \lambda)$ at each iteration.

## Intuitive Explanation

### The Architecture

The HMM has two layers:
- **Hidden layer**: A Markov chain over states $q_1 \to q_2 \to \cdots \to q_T$. We cannot observe these states directly.
- **Observation layer**: Each state $q_t$ generates an observation $o_t$ through the emission distribution $b_{q_t}$.

The two independence assumptions create the layered structure:
- States form a Markov chain: each state depends only on the previous state.
- Observations depend only on the current state: given $q_t$, the observation $o_t$ is independent of everything else.

### Why Each Problem Matters

**Evaluation** answers: "How well does my trained model explain this data?" It is the objective function for model selection (BIC/AIC) and for comparing models with different numbers of states.

**Decoding** answers: "Given these observations, what regime was the market in at each time?" This produces the human-interpretable output: a sequence of labeled market regimes.

**Learning** answers: "What parameters best explain my historical data?" This is training — the unsupervised learning problem of finding $\pi$, $A$, and $B$ from observations alone, without regime labels.

### Weather Example Worked Through

Classical three-symbol HMM (Rabiner 1989):
- States: Sunny (0), Rainy (1)
- Observations: Walk (0), Shop (1), Clean (2)

```
A = [[0.7, 0.3],   B = [[0.6, 0.3, 0.1],   pi = [0.6, 0.4]
     [0.4, 0.6]]        [0.1, 0.4, 0.5]]
```

Given the sequence [Walk, Shop, Clean], we want:
1. $P(\text{Walk, Shop, Clean} \mid \lambda)$ — evaluation
2. Most likely weather sequence — decoding
3. Parameters that best explain months of activity sequences — learning

## Code Implementation

```python
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from scipy import stats


@dataclass
class HMMParams:
    """Container for discrete HMM parameters."""
    pi: np.ndarray       # Shape (K,)  — initial distribution
    A: np.ndarray        # Shape (K, K) — transition matrix
    B: np.ndarray        # Shape (K, M) — emission matrix (discrete)

    @property
    def n_states(self) -> int:
        return len(self.pi)

    @property
    def n_symbols(self) -> int:
        return self.B.shape[1]

    def validate(self) -> List[str]:
        """Return list of validation errors (empty = valid)."""
        errors = []
        if not np.isclose(self.pi.sum(), 1.0, atol=1e-8):
            errors.append(f"pi sums to {self.pi.sum():.6f}, not 1.0")
        if np.any(self.pi < 0):
            errors.append("pi contains negative entries")
        if not np.allclose(self.A.sum(axis=1), 1.0, atol=1e-8):
            errors.append("A rows do not all sum to 1.0")
        if np.any(self.A < 0):
            errors.append("A contains negative entries")
        if not np.allclose(self.B.sum(axis=1), 1.0, atol=1e-8):
            errors.append("B rows do not all sum to 1.0")
        if np.any(self.B < 0):
            errors.append("B contains negative entries")
        return errors


class DiscreteHMM:
    """Hidden Markov Model with discrete (categorical) emissions.

    This is the canonical HMM for learning the algorithms.
    For financial data with continuous observations, use GaussianHMM.
    """

    def __init__(self, n_states: int, n_symbols: int, params: Optional[HMMParams] = None):
        self.n_states = n_states
        self.n_symbols = n_symbols

        if params is not None:
            errors = params.validate()
            if errors:
                raise ValueError(f"Invalid parameters: {errors}")
            self.pi = params.pi.copy()
            self.A = params.A.copy()
            self.B = params.B.copy()
        else:
            self._initialize_random()

    def _initialize_random(self):
        """Dirichlet random initialization (valid probability vectors)."""
        self.pi = np.random.dirichlet(np.ones(self.n_states))
        self.A = np.random.dirichlet(np.ones(self.n_states), size=self.n_states)
        self.B = np.random.dirichlet(np.ones(self.n_symbols), size=self.n_states)

    def generate(self, length: int) -> Tuple[List[int], List[int]]:
        """Generate a sequence of hidden states and observations.

        Returns:
            (states, observations): Both are lists of integer indices.
        """
        state = int(np.random.choice(self.n_states, p=self.pi))
        states = [state]
        observations = [int(np.random.choice(self.n_symbols, p=self.B[state]))]

        for _ in range(length - 1):
            state = int(np.random.choice(self.n_states, p=self.A[state]))
            states.append(state)
            observations.append(int(np.random.choice(self.n_symbols, p=self.B[state])))

        return states, observations

    def joint_log_probability(self, states: List[int], observations: List[int]) -> float:
        """Compute log P(O, Q | lambda) directly from the factorization."""
        log_prob = (
            np.log(self.pi[states[0]] + 1e-300)
            + np.log(self.B[states[0], observations[0]] + 1e-300)
        )
        for t in range(1, len(observations)):
            log_prob += np.log(self.A[states[t - 1], states[t]] + 1e-300)
            log_prob += np.log(self.B[states[t], observations[t]] + 1e-300)
        return float(log_prob)

    def expected_state_durations(self) -> np.ndarray:
        """Expected consecutive time steps in each state: 1/(1 - a_ii)."""
        return 1.0 / (1.0 - np.diag(self.A) + 1e-10)


# --- Classic weather HMM example ---

def build_weather_hmm() -> DiscreteHMM:
    """Create the classic Rabiner (1989) weather HMM.

    States:      0 = Sunny, 1 = Rainy
    Observations: 0 = Walk, 1 = Shop, 2 = Clean
    """
    params = HMMParams(
        pi=np.array([0.6, 0.4]),
        A=np.array([
            [0.7, 0.3],   # Sunny -> Sunny: 70%, Rainy: 30%
            [0.4, 0.6],   # Rainy -> Sunny: 40%, Rainy: 60%
        ]),
        B=np.array([
            [0.6, 0.3, 0.1],   # Sunny: Walk 60%, Shop 30%, Clean 10%
            [0.1, 0.4, 0.5],   # Rainy: Walk 10%, Shop 40%, Clean 50%
        ]),
    )
    return DiscreteHMM(n_states=2, n_symbols=3, params=params)


def three_problems_demo():
    """Illustrate the three HMM problems conceptually."""
    np.random.seed(42)
    hmm = build_weather_hmm()

    # Generate data (normally we would only see observations, not states)
    true_states, observations = hmm.generate(20)
    obs_names = ["Walk", "Shop", "Clean"]
    state_names = ["Sunny", "Rainy"]

    print("True states (normally hidden):")
    print("  " + " ".join(state_names[s] for s in true_states))
    print("Observations (what we see):")
    print("  " + " ".join(obs_names[o] for o in observations))

    print("\nProblem 1 (Evaluation):")
    print("  Use Forward algorithm to compute P(O | lambda)")
    print("  Compare this across models to select best number of states")

    print("\nProblem 2 (Decoding):")
    print("  Use Viterbi algorithm to find most likely state sequence")
    print("  Compare decoded states to true states to evaluate model")

    print("\nProblem 3 (Learning):")
    print("  Use Baum-Welch to find pi, A, B that maximize P(O | lambda)")
    print("  Start from random params, iterate until log-likelihood converges")


if __name__ == "__main__":
    three_problems_demo()

    hmm = build_weather_hmm()
    errors = HMMParams(hmm.pi, hmm.A, hmm.B).validate()
    print(f"\nParameter validation: {'PASS' if not errors else errors}")

    durations = hmm.expected_state_durations()
    print(f"Expected duration in Sunny: {durations[0]:.1f} steps")
    print(f"Expected duration in Rainy: {durations[1]:.1f} steps")
```

## Common Pitfalls

**Initializing $B$ with a single peak.** If all probability mass in $B$ starts in one symbol per state, Baum-Welch may converge immediately to a degenerate solution where each state only ever emits one symbol. Always use Dirichlet random initialization to spread probability mass across symbols.

**Confusing the three problems.** Problem 1 returns a scalar (a probability). Problem 2 returns a sequence (of state indices). Problem 3 returns parameters (updated $\pi$, $A$, $B$). If you ask "what regime is the market in?" you need Problem 2 (Viterbi). If you ask "does this model explain the data well?", you need Problem 1 (Forward).

**Using the joint probability instead of the marginal for evaluation.** $P(O, Q \mid \lambda)$ is computed by `joint_log_probability` but this is not the evaluation answer. The evaluation answer is $P(O \mid \lambda) = \sum_Q P(O, Q \mid \lambda)$, which sums over all state sequences. The Forward algorithm computes this sum efficiently.

**Applying decoding before training.** Viterbi and the Forward algorithm both require known parameters $\lambda$. If you haven't trained the model yet (Problem 3), you don't have parameters to decode with. The standard workflow is: train with Baum-Welch first, then decode.

**Expecting Viterbi to give the same result as marginal argmax.** `argmax_i P(q_t = s_i \mid O, \lambda)` (from forward-backward) and Viterbi can give different state sequences. Viterbi gives the globally optimal sequence; marginal argmax optimizes each time step independently and may produce transitions with zero probability.

## Connections

**Markov chains (Module 00)**: The HMM hidden state layer is a Markov chain. The transition matrix $A$ is exactly the Markov chain transition matrix. Module 00 covered how to analyze, simulate, and estimate Markov chains; all of that applies to the $A$ component of an HMM.

**Forward-Backward (Module 02)**: Solves Problems 1 and 2 (evaluation and smoothing). The forward pass computes $\alpha_t(i)$, the backward pass computes $\beta_t(i)$, and combining them gives the posterior $\gamma_t(i) = P(q_t = s_i \mid O, \lambda)$.

**Viterbi (Module 02)**: Solves Problem 2 (decoding) by finding the sequence that maximizes $P(Q \mid O, \lambda)$.

**Baum-Welch (Module 02)**: Solves Problem 3 (learning) using the EM framework. The E-step runs the Forward-Backward algorithm; the M-step updates $\pi$, $A$, $B$ using sufficient statistics from gamma and xi.

**HMM parameters (next guide)**: A detailed treatment of what each parameter controls, initialization strategies, and validation.

## Practice Problems

**Problem 1.** Given the weather HMM from the code above:
(a) Generate 100 observation sequences of length 50 each.
(b) For each sequence, compute the joint log-probability of the true state sequence.
(c) Generate 100 random alternative state sequences. Compare their joint log-probabilities to the true sequence. What fraction of random sequences have lower joint probability than the true sequence?

**Problem 2.** Create an HMM for coin flipping with two states (Fair and Biased):
- Fair coin: P(Heads) = 0.5, P(Tails) = 0.5
- Biased coin: P(Heads) = 0.8, P(Tails) = 0.2
- Transition: once assigned a coin, you keep it with probability 0.9

Generate 200 flips. Implement the naive joint probability computation (sum over all $2^{200}$ state sequences) and the Forward algorithm. Verify they give the same answer for sequences of length 5. For length 200, show that the naive approach is computationally infeasible.

**Problem 3.** Modify `DiscreteHMM.generate()` to handle missing observations (emit `None` for some time steps with probability `p_missing`). How does the HMM architecture naturally handle missing data? (Hint: the state chain continues even when we don't observe the emission.)

**Problem 4.** Derive the factorization $P(O, Q \mid \lambda) = \pi_{q_1} b_{q_1}(o_1) \prod_{t=2}^{T} a_{q_{t-1},q_t} b_{q_t}(o_t)$ from the chain rule of probability and the two independence assumptions. Identify exactly where each assumption is used.

**Problem 5.** For a 3-state HMM ($K=3$) with sequence length $T=20$: how many possible state sequences are there? How many arithmetic operations does the Forward algorithm require? Compute the speedup factor and express it as a percentage reduction in computation.

## Further Reading

- **Rabiner (1989)** — "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition." The original three-problems framing from Section II is still the clearest exposition.
- **Jurafsky & Martin (2023)** — *Speech and Language Processing*, Chapter 8. Free online. Applied NLP perspective with worked examples.
- **Murphy (2012)** — *Machine Learning: A Probabilistic Perspective*, Chapter 17. Formal ML treatment with connections to graphical models.
- **Bilmes (1998)** — "A Gentle Tutorial of the EM Algorithm and its Application to Parameter Estimation for Gaussian Mixture and Hidden Markov Models." ICSI Technical Report. Accessible derivation of Baum-Welch from EM first principles.
