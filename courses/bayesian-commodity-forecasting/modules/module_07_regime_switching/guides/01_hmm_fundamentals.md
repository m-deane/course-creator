# Hidden Markov Model Fundamentals

> **Reading time:** ~6 min | **Module:** 7 — Regime Switching | **Prerequisites:** Module 3 State-Space Models


## In Brief

A Hidden Markov Model (HMM) describes a system that transitions between discrete hidden states over time, with each state generating observations according to a state-specific distribution. For commodities, states represent market regimes, and observations are prices or returns.

<div class="callout-insight">

<strong>Insight:</strong> **HMMs separate "what regime are we in?" from "what do we observe?"** The regime is latent (unobserved) but influences the distribution of prices. By modeling this two-layer structure, we can infer regimes from prices and make regime-dependent forecasts.

</div>


<div class="callout-key">

<strong>Key Concept Summary:</strong> A Hidden Markov Model (HMM) describes a system that transitions between discrete hidden states over time, with each state generating observations according to a state-specific distribution.

</div>

---

## Formal Definition

### Model Components

An HMM is defined by:

1. **States:** $S = \{1, 2, ..., K\}$ (e.g., Bull, Bear)
2. **Initial Distribution:** $\pi_k = P(S_1 = k)$
3. **Transition Matrix:** $A_{jk} = P(S_t = k | S_{t-1} = j)$
4. **Emission Distribution:** $P(Y_t | S_t = k) = f_k(y_t)$

### Model Assumptions

1. **Markov Property:** $P(S_t | S_{t-1}, ..., S_1) = P(S_t | S_{t-1})$
2. **Conditional Independence:** $P(Y_t | S_t, Y_{1:t-1}) = P(Y_t | S_t)$
3. **Time Homogeneity:** Transition and emission parameters don't change over time

### Graphical Representation

```
         A          A          A
    S₁ ────→ S₂ ────→ S₃ ────→ S₄   (Hidden States)
    │        │        │        │
    │f₁/f₂   │f₁/f₂   │f₁/f₂   │f₁/f₂
    ↓        ↓        ↓        ↓
    Y₁       Y₂       Y₃       Y₄   (Observations)
```

---

## Intuitive Explanation

### The Weather Analogy

Imagine you can't see outside but can observe whether your friend carries an umbrella:

- **Hidden States:** Sunny, Rainy (weather you can't see)
- **Observations:** Umbrella, No Umbrella
- **Transition:** Sunny days tend to follow sunny days
- **Emission:** Rainy → high probability of umbrella

From umbrella observations, you can infer the likely weather sequence.

### Commodity Market Analogy

- **Hidden States:** Bull Market, Bear Market
- **Observations:** Daily returns
- **Transition:** Regimes are persistent (bull follows bull)
- **Emission:** Bull → positive mean return, lower volatility

From returns, we infer the current regime and forecast future regimes.

---

## Two-State Commodity Model

### Specification

**States:** $K = 2$ (Bull, Bear)

**Initial Distribution:**
$$\pi = [\pi_{\text{bull}}, \pi_{\text{bear}}]$$

**Transition Matrix:**
$$A = \begin{bmatrix}
P(S_t = \text{bull} | S_{t-1} = \text{bull}) & P(S_t = \text{bear} | S_{t-1} = \text{bull}) \\
P(S_t = \text{bull} | S_{t-1} = \text{bear}) & P(S_t = \text{bear} | S_{t-1} = \text{bear})
\end{bmatrix}$$

Example:
$$A = \begin{bmatrix} 0.98 & 0.02 \\ 0.05 & 0.95 \end{bmatrix}$$

Interpretation: Bull markets last ~50 days on average ($1/0.02$), bear markets ~20 days.

**Emission Distributions:**
$$Y_t | S_t = \text{bull} \sim \mathcal{N}(\mu_{\text{bull}}, \sigma^2_{\text{bull}})$$
$$Y_t | S_t = \text{bear} \sim \mathcal{N}(\mu_{\text{bear}}, \sigma^2_{\text{bear}})$$

Typical values for daily returns:
- Bull: $\mu = +0.05\%$, $\sigma = 1.5\%$
- Bear: $\mu = -0.10\%$, $\sigma = 2.5\%$

---

## Inference Problems

### 1. Filtering: $P(S_t | Y_{1:t})$

What is the current regime given observations up to now?

**Use:** Real-time regime identification for trading

### 2. Smoothing: $P(S_t | Y_{1:T})$

What was the regime at time $t$ given ALL observations?

**Use:** Retrospective regime labeling for analysis

### 3. Decoding: Most Likely Sequence

$$\hat{S}_{1:T} = \arg\max_{S_{1:T}} P(S_{1:T} | Y_{1:T})$$

**Use:** Finding the single best regime sequence (Viterbi algorithm)

### 4. Parameter Learning

Estimate $\theta = (\pi, A, \{f_k\})$ from data.

**Use:** Fitting the model to historical data

---

## The Forward-Backward Algorithm

### Forward Pass (Filtering)

$$\alpha_t(k) = P(Y_{1:t}, S_t = k)$$

**Recursion:**
$$\alpha_1(k) = \pi_k \cdot f_k(y_1)$$
$$\alpha_t(k) = f_k(y_t) \sum_j \alpha_{t-1}(j) \cdot A_{jk}$$

**Filtered probability:**
$$P(S_t = k | Y_{1:t}) = \frac{\alpha_t(k)}{\sum_j \alpha_t(j)}$$

### Backward Pass (Smoothing)

$$\beta_t(k) = P(Y_{t+1:T} | S_t = k)$$

**Recursion:**
$$\beta_T(k) = 1$$
$$\beta_t(k) = \sum_j A_{kj} \cdot f_j(y_{t+1}) \cdot \beta_{t+1}(j)$$

### Smoothed Probability

$$P(S_t = k | Y_{1:T}) = \frac{\alpha_t(k) \cdot \beta_t(k)}{\sum_j \alpha_t(j) \cdot \beta_t(j)}$$

---

## Code Implementation

### Basic HMM Class


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import numpy as np
from scipy import stats

class GaussianHMM:
    """Two-state Gaussian HMM for regime detection."""

    def __init__(self, means, stds, trans_mat, initial_probs):
        """
        Parameters
        ----------
        means : array (K,)
            Mean for each state
        stds : array (K,)
            Std for each state
        trans_mat : array (K, K)
            Transition matrix A[i,j] = P(S_t=j | S_{t-1}=i)
        initial_probs : array (K,)
            Initial state distribution
        """
        self.K = len(means)
        self.means = np.array(means)
        self.stds = np.array(stds)
        self.A = np.array(trans_mat)
        self.pi = np.array(initial_probs)

    def emission_prob(self, y, state):
        """P(y | state)"""
        return stats.norm.pdf(y, self.means[state], self.stds[state])

    def forward(self, y):
        """Forward algorithm returning log-likelihood and alpha values."""
        T = len(y)
        alpha = np.zeros((T, self.K))

        # Initialize
        for k in range(self.K):
            alpha[0, k] = self.pi[k] * self.emission_prob(y[0], k)

        # Recurse
        for t in range(1, T):
            for k in range(self.K):
                alpha[t, k] = self.emission_prob(y[t], k) * \
                              np.sum(alpha[t-1, :] * self.A[:, k])

        log_likelihood = np.log(np.sum(alpha[-1, :]))
        return alpha, log_likelihood

    def backward(self, y):
        """Backward algorithm returning beta values."""
        T = len(y)
        beta = np.zeros((T, self.K))

        # Initialize
        beta[-1, :] = 1

        # Recurse
        for t in range(T-2, -1, -1):
            for k in range(self.K):
                beta[t, k] = np.sum(
                    self.A[k, :] *
                    np.array([self.emission_prob(y[t+1], j) for j in range(self.K)]) *
                    beta[t+1, :]
                )

        return beta

    def smooth(self, y):
        """Compute smoothed state probabilities P(S_t | Y_{1:T})."""
        alpha, _ = self.forward(y)
        beta = self.backward(y)

        gamma = alpha * beta
        gamma = gamma / gamma.sum(axis=1, keepdims=True)

        return gamma

    def filter(self, y):
        """Compute filtered state probabilities P(S_t | Y_{1:t})."""
        alpha, _ = self.forward(y)
        return alpha / alpha.sum(axis=1, keepdims=True)


# Example usage
hmm = GaussianHMM(
    means=[0.001, -0.002],  # Bull and Bear daily returns
    stds=[0.015, 0.025],
    trans_mat=[[0.98, 0.02], [0.05, 0.95]],
    initial_probs=[0.5, 0.5]
)

# Simulate or use real returns
returns = np.random.randn(252) * 0.02  # Placeholder

# Get regime probabilities
smoothed = hmm.smooth(returns)
print(f"Final regime probabilities: Bull={smoothed[-1, 0]:.2%}, Bear={smoothed[-1, 1]:.2%}")
```

</div>
</div>

---

## Regime Duration and Characteristics

### Expected Duration

For a regime with self-transition probability $p_{kk}$:

$$\mathbb{E}[\text{Duration in state } k] = \frac{1}{1 - p_{kk}}$$

Example: If $p_{\text{bull,bull}} = 0.98$, expected bull market duration = 50 periods.

### Stationary Distribution

Long-run regime probabilities (eigenvector of $A$):

For 2-state: $\pi^*_{\text{bull}} = \frac{p_{\text{bear,bull}}}{p_{\text{bear,bull}} + p_{\text{bull,bear}}}$

---

## Common Pitfalls

### 1. Label Switching

The model doesn't know which state is "Bull" vs "Bear". States may swap between runs.

**Solution:** Impose ordering constraint (e.g., $\mu_1 < \mu_2$) or post-process labels.

### 2. Choosing Number of States

More states = better fit but overfitting risk.

**Solution:** Use model comparison (WAIC, LOO) or domain knowledge.

### 3. Ignoring Uncertainty

Point estimates of regimes ignore classification uncertainty.

**Solution:** Report regime probabilities, not just most likely regime.

---

## Connections

**Builds on:**
- Module 3: State space models (HMM is discrete-state analog)
- Module 6: MCMC for parameter estimation

**Leads to:**
- Module 8: Regime-dependent fundamental models
- Capstone: Regime-aware forecasting systems

---

## Practice Problems

1. For a 2-state HMM with $A = [[0.9, 0.1], [0.2, 0.8]]$, what is the expected duration in each state? What is the stationary distribution?

2. You observe returns: $y = [0.02, 0.03, -0.01, -0.03, -0.02, 0.01]$. Given the HMM parameters above, compute the filtered regime probability at $t=3$.

3. Why might the Viterbi (most likely path) and smoothed probabilities give different regime classifications?

---


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Formal Definition" and why it matters in practice.

2. Given a real-world scenario involving hidden markov model fundamentals, what would be your first three steps to apply the techniques from this guide?




## Further Reading

1. **Rabiner** "A Tutorial on HMMs" (1989) — Classic introduction
2. **Hamilton** "A New Approach to Economic Time Series" (1989) — MS models in economics
3. **Murphy** *Machine Learning* Chapter 17 — Modern treatment

---

*"Regimes are the market's hidden heartbeat. HMMs let us hear it through the noise of daily prices."*

---

## Cross-References

<a class="link-card" href="./01_hmm_fundamentals_slides.md">
  <div class="link-card-title">Companion Slide Deck</div>
  <div class="link-card-description">Visual presentation covering the key concepts from this guide.</div>
</a>

<a class="link-card" href="../notebooks/01_hmm_from_scratch.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
