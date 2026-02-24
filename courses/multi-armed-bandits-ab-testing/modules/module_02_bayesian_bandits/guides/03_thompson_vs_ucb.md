# Thompson Sampling vs UCB: Theory and Practice

## In Brief

Thompson Sampling (TS) and Upper Confidence Bound (UCB) are the two leading bandit algorithms. Both achieve logarithmic regret, but they explore differently: UCB uses deterministic optimism ("pick the arm with highest plausible value"), while Thompson Sampling uses randomized probability matching ("pick arms proportionally to their probability of being best"). In practice, Thompson Sampling often outperforms UCB in non-stationary, delayed-feedback, and batched-update settings.

> 💡 **Key Insight:** **UCB's philosophy:** Always pick the arm with the highest upper confidence bound. Exploration happens because uncertain arms have wide bounds.

**Thompson Sampling's philosophy:** Maintain beliefs about each arm, sample from those beliefs, act on samples. Exploration happens because uncertain beliefs produce diverse samples.

**Why it matters:** UCB is deterministic given current data (same state → same action). Thompson Sampling is stochastic (same state → different actions, weighted by belief). This stochasticity helps in delayed feedback, contextual settings, and non-stationary environments.

## Visual Explanation

```
Scenario: 3 arms, 100 pulls each

UCB Confidence Bounds:
Arm A: [0.45 ████████ 0.55]  ← UCB = 0.55
Arm B: [0.48 ██████ 0.58]    ← UCB = 0.58 (SELECTED)
Arm C: [0.40 ███████████ 0.60] ← UCB = 0.60 (wide but not selected)

UCB picks arm C (highest upper bound)
Deterministic: Always C until bound shrinks

Thompson Sampling Posteriors:
Arm A: Beta(50, 55)    |█|
Arm B: Beta(53, 52)      |█|
Arm C: Beta(50, 60)  |█|

Sample draws: θ̂_A=0.51, θ̂_B=0.54, θ̂_C=0.48
Thompson picks B (highest sample)

Next round, new samples: θ̂_A=0.49, θ̂_B=0.50, θ̂_C=0.53
Thompson picks C

Stochastic: Different samples → different actions
Naturally mixes exploration and exploitation
```

**Key difference:** UCB commits to one arm until confidence bounds shift. Thompson Sampling continuously randomizes across plausible arms.

## Formal Definition

### UCB1 Algorithm
```
For each round t:
    Select arm: aₜ = argmax [μ̂ᵢ + √(2 ln t / nᵢ)]
    where:
        μ̂ᵢ = empirical mean of arm i
        nᵢ = number of pulls of arm i
        t = total rounds so far

Regret bound: E[R_T] ≤ O(K log T / Δ)
where K = number of arms, Δ = gap to best arm
```

### Thompson Sampling (Beta-Bernoulli)
```
For each round t:
    For each arm i: sample θ̂ᵢ ~ Beta(αᵢ, βᵢ)
    Select arm: aₜ = argmax θ̂ᵢ
    Update posterior based on reward

Regret bound: E[R_T] ≤ O(K log T / Δ)
(same asymptotic guarantee)
```

**Theoretical equivalence:** Both achieve logarithmic regret. Thompson Sampling's bound was proven much later (2010s) than UCB's (2000s), but they're asymptotically equivalent.

## Comparison Table

| Criterion | UCB1 | Thompson Sampling |
|-----------|------|-------------------|
| **Exploration mechanism** | Deterministic optimism | Randomized probability matching |
| **Selection rule** | Pick highest UCB | Sample and pick best sample |
| **Tuning parameters** | Exploration constant (often c=2) | Prior distribution (often Beta(1,1)) |
| **Computational cost** | O(K) comparison | O(K) sampling (slightly higher) |
| **Handles delayed feedback** | Poorly (assumes immediate rewards) | Naturally (batch updates work) |
| **Non-stationary environments** | Requires modification (discounted UCB) | Naturally adapts (discount posteriors) |
| **Contextual extension** | LinUCB (complex) | Contextual TS (straightforward) |
| **Empirical performance** | Good | Often better in real settings |
| **Interpretability** | "Always try the most optimistic option" | "Sample plausible worlds, act accordingly" |
| **Theoretical guarantees** | O(log T) regret | O(log T) regret |

## When to Use Which

### Choose UCB when:
- **You need determinism** — Same state always produces same action (reproducibility, auditing)
- **Immediate feedback** — Rewards arrive instantly after each action
- **Simple Bernoulli/Gaussian rewards** — No complicated likelihood structure
- **Theoretical guarantees paramount** — UCB has longer track record of proofs

**Commodity example:** High-frequency trading where every decision must be explainable and reproducible for compliance.

### Choose Thompson Sampling when:
- **Delayed or batched feedback** — Rewards arrive in batches (weekly rebalancing, monthly reports)
- **Non-stationary environments** — Distributions shift over time (regime changes in commodities)
- **Contextual decisions** — Actions depend on features (volatility, term structure, seasonality)
- **Prior information available** — You have genuine beliefs to encode (e.g., from fundamental analysis)
- **Empirical performance matters most** — Thompson often wins in practice

**Commodity example:** Portfolio allocation where you rebalance weekly based on market regimes, and regime shifts require adaptive exploration.

## Theoretical Comparison

### Regret Bounds (Both Logarithmic)

**UCB1:**
```
E[R_T] ≤ 8 Σᵢ (ln T / Δᵢ) + (1 + π²/3) Σᵢ Δᵢ
```

**Thompson Sampling:**
```
E[R_T] ≤ O(K log T / Δ)
```

Both scale as O(log T) — optimal for stochastic bandits.

### Probability Matching Property

Thompson Sampling satisfies **probability matching**: the probability of selecting arm *i* equals the probability that arm *i* is optimal given current beliefs.

```
P(select arm i | data) = P(arm i is best | data)
```

UCB does NOT satisfy probability matching (it's deterministic). This property makes Thompson Sampling naturally Bayesian-optimal in certain formulations.

### Computational Complexity

**Per-round cost:**
- UCB: O(K) to compute K upper bounds
- Thompson Sampling: O(K) to sample K posteriors

Both scale linearly in number of arms. Thompson Sampling requires random number generation (slightly slower), but difference is negligible in practice.

## Practical Trade-offs

### Implementation Complexity
**UCB:** Simpler — just track counts and means
```python
ucb = mu_hat + np.sqrt(2 * np.log(t) / n)
```

**Thompson Sampling:** Requires probability libraries
```python
from scipy.stats import beta
theta_hat = beta.rvs(alpha, beta_param)
```

Winner: UCB (marginally)

### Handling Delayed Feedback
**UCB:** Breaks down when rewards are delayed. If you pull arm A but don't see reward for 10 rounds, how do you update the bound?

**Thompson Sampling:** Naturally handles batched/delayed updates. Pull arm A multiple times, then update posterior with all observed rewards at once.

Winner: Thompson Sampling (decisively)

### Non-Stationary Environments
**UCB:** Requires modification (sliding window UCB, discounted UCB). Standard UCB accumulates all past data forever.

**Thompson Sampling:** Just discount posteriors exponentially:
```python
alpha *= gamma  # e.g., gamma = 0.99
beta_param *= gamma
```

Winner: Thompson Sampling (easier adaptation)

### Prior Knowledge Integration
**UCB:** No natural way to incorporate prior beliefs. You could initialize counts, but it's ad-hoc.

**Thompson Sampling:** Priors are fundamental. If you believe an arm has ~60% success rate based on fundamentals, start with Beta(6,4).

Winner: Thompson Sampling (Bayesian framework)

### Reproducibility & Debugging
**UCB:** Deterministic — same data produces same sequence of actions. Easier to debug.

**Thompson Sampling:** Stochastic — need to set random seed for reproducibility. Harder to debug "why did it pick that arm?" (answer: "it sampled high this time").

Winner: UCB (easier to reason about)

## Commodity Trading Context

### Why Thompson Sampling Dominates in Commodity Allocation

1. **Weekly/Monthly Rebalancing:** You don't trade every millisecond. You rebalance weekly based on week's worth of data. Thompson Sampling handles batched updates naturally.

2. **Regime Changes:** Commodity markets shift between contango/backwardation, risk-on/risk-off, seasonal patterns. Discounted Thompson Sampling adapts faster than standard UCB.

3. **Fundamental Priors:** You have views on commodities based on supply/demand fundamentals. Thompson Sampling lets you encode these as priors.

4. **Noisy Signals:** Commodity returns are extremely noisy (high variance). Thompson Sampling's probabilistic approach handles noise better than UCB's deterministic bounds.

5. **Contextual Features Coming Next:** Module 3 extends to contextual bandits. Contextual Thompson Sampling is more natural than LinUCB.

### Example: Commodity Sector Rotation

**Setup:** Allocate between Energy (WTI), Metals (Copper), Agriculture (Corn) each week.

**UCB approach:**
- Track mean weekly returns for each sector
- Compute UCB for each
- Go all-in on highest UCB
- Problem: If energy has high UCB from recent volatility, you're locked into energy until variance shrinks

**Thompson Sampling approach:**
- Maintain Gaussian posterior over each sector's mean return
- Each week, sample plausible return from each posterior
- Allocate to highest sample
- Problem: None — stochasticity naturally diversifies, and posteriors tighten around true means over time

## Empirical Results from Literature

**Chapelle & Li (2011):** "An Empirical Evaluation of Thompson Sampling"
- Tested on 6 real-world datasets
- Thompson Sampling matched or beat UCB on all datasets
- Biggest wins in non-stationary and delayed-feedback settings

**Russo & Van Roy (2014):** "Learning to Optimize via Information-Directed Sampling"
- Thompson Sampling approximates information-directed sampling (theoretically optimal)
- UCB doesn't — it over-explores in some settings

**Industry adoption:**
- Google, Meta: Thompson Sampling for content recommendations
- E-commerce: Thompson Sampling for pricing and promotions
- Reason: Better performance under real-world messiness (delayed feedback, non-stationarity, context)

## Practice Problems

### Problem 1: Side-by-side comparison
Implement both UCB1 and Thompson Sampling on the same 4-arm Bernoulli bandit with true probabilities [0.4, 0.5, 0.45, 0.48].

Run 1000 rounds for each algorithm (with same random seed for reward generation).

**Compare:**
a) Cumulative regret
b) Fraction of time spent on each arm
c) Exploration rate over time (fraction of pulls to non-empirical-best arm)

### Problem 2: Delayed feedback simulation
Simulate a bandit problem where rewards are delayed by 10 rounds (you pull arm at t=0, but see reward at t=10).

Implement:
a) Standard UCB (try your best to handle delays)
b) Thompson Sampling with batched updates every 10 rounds

**Question:** Which handles delayed feedback better? Why?

### Problem 3: Non-stationary commodity returns
Create a 3-arm Gaussian bandit representing commodity sectors. True means shift every 200 rounds:
- Rounds 0-199: μ = [0.01, 0.02, 0.015]
- Rounds 200-399: μ = [0.02, 0.01, 0.015]  (energy and metals swap)
- Rounds 400-599: μ = [0.015, 0.015, 0.025] (agriculture dominates)

Implement:
a) Standard UCB
b) Discounted UCB (discount old observations)
c) Thompson Sampling with posterior discounting

**Compare:** Which adapts fastest to regime changes?

### Problem 4: Explain to a PM
You're presenting to a portfolio manager who asks: "Why use this Thompson Sampling instead of just tracking which commodity has highest Sharpe ratio?"

**Draft a 3-sentence answer that:**
- Explains the explore-exploit tradeoff
- Shows why Thompson Sampling is better than "always pick highest observed Sharpe"
- Connects to commodity market non-stationarity
