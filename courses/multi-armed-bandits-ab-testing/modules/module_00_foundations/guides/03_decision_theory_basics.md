# Decision Theory Basics

> **Reading time:** ~20 min | **Module:** 00 — Foundations | **Prerequisites:** None (entry point)


## In Brief


<div class="callout-key">

**Key Concept Summary:** Decision theory provides the mathematical foundation for choosing actions under uncertainty. For bandits and commodity trading, the key concepts are: expected value (what you'll earn on average), u...

</div>

Decision theory provides the mathematical foundation for choosing actions under uncertainty. For bandits and commodity trading, the key concepts are: expected value (what you'll earn on average), utility (how you value different outcomes), sequential decision making (your choices now affect future information), and regret minimization (comparing your strategy to the best possible strategy in hindsight).

> 💡 **Key Insight:** **Decision theory formalizes intuition.** When a trader says "I'm not touching volatility trades right now," they're implicitly doing expected utility calculations with risk aversion. When they say "I'll try a small position to see how it performs," they're valuing information acquisition. Decision theory makes these tradeoffs explicit and mathematically tractable.

## Expected Value and Utility

### Expected Value (Risk-Neutral)

The **expected value** of a random variable X is:
```
E[X] = Σ x · P(X = x)    (discrete)
E[X] = ∫ x · f(x) dx     (continuous)
```

**In bandits:** The expected reward of arm k is μ_k = E[r | arm k].

**In commodity trading:**
```
E[P&L] = Σ (return_i × probability_i)
```

If crude oil has:
- 40% chance of +$5/barrel → +$2.00 expected
- 30% chance of $0 → $0.00 expected
- 30% chance of -$3/barrel → -$0.90 expected
- Total: E[P&L] = $1.10/barrel

### Utility Functions (Risk-Averse/Seeking)

Most traders are **risk-averse:** they prefer certain outcomes over risky bets with the same expected value.

**Utility function** U(x) maps outcomes to subjective value:
```
Risk-neutral:  U(x) = x                    (linear)
Risk-averse:   U(x) = √x  or  log(x)       (concave)
Risk-seeking:  U(x) = x²                   (convex)
```

**Expected utility:**
```
EU = Σ U(x_i) · P(x_i)
```

<div class="callout-insight">

**Insight:** The core insight of bandit algorithms is that learning and earning are not separate phases. Every observation contributes to both understanding which option is best and generating value from the best option.

</div>


**Example:** Would you rather have:
- **Option A:** $100 for sure
- **Option B:** 50% chance of $200, 50% chance of $0

Expected value: Both are $100.

Risk-neutral trader: Indifferent (both have U = 100).

Risk-averse trader (U = √x):
- U(A) = √100 = 10
- U(B) = 0.5·√200 + 0.5·√0 = 0.5·14.14 ≈ 7.07
- Prefers A (certain $100).

**Commodity context:** A risk-averse gold trader might prefer a hedged position with 8% expected return and low variance over a naked position with 10% expected return and high variance, even though the naked position has higher expected value.

### Sharpe Ratio: Risk-Adjusted Returns

The **Sharpe ratio** combines expected return and risk:
```
Sharpe = (μ - r_f) / σ
```
where μ = expected return, r_f = risk-free rate, σ = return volatility.

**In bandit terms:** Arms have both mean (μ_k) and variance (σ²_k). A high-Sharpe arm might be preferable to a high-mean arm.

**Example:**
- Arm A: μ = 10%, σ = 20%, Sharpe = 0.5
- Arm B: μ = 8%, σ = 10%, Sharpe = 0.8

Risk-neutral bandit: Prefers A (higher mean).
Risk-adjusted bandit: Prefers B (higher Sharpe).

## Sequential Decision Making Under Uncertainty

<div class="callout-warning">

**Warning:** Bandit algorithms assume the reward distributions are stationary (or slowly changing). In commodity markets, regime shifts can make a historically optimal arm suddenly suboptimal. Always implement change detection alongside your bandit.

</div>


### The Value of Information

In bandits, exploring an uncertain arm has two types of value:

1. **Immediate reward:** The expected payoff of the arm this round
2. **Information value:** Learning reduces future uncertainty, enabling better future decisions

**Formally:** The value of perfect information (VPI) about arm k is:
```
VPI_k = E[future regret reduction | learn true μ_k]
```

**Intuitive example:**
You have two arms:
- Arm 1: Known μ_1 = 0.5 (you've pulled it 1000 times)
- Arm 2: Estimated μ_2 = 0.55, but only 10 pulls (high uncertainty)

**Should you explore Arm 2?**

**Information value reasoning:**
- If you pull Arm 2 and learn its true mean is 0.6, you'll switch to it permanently → huge future gain
- If you learn its true mean is 0.4, you'll stick with Arm 1 → small cost now, big savings later
- Expected information value likely exceeds the immediate regret risk

**Commodity example:** You have high confidence that energy trades return 12% annually (100 trades). You have weak confidence that a new agriculture strategy returns 15% (5 trades). The potential 3% gain, multiplied by future years of trading, makes exploration valuable even if the next few agriculture trades lose money.

### Bayesian vs Frequentist Perspectives

**Frequentist view:**
- Arm means μ_k are fixed but unknown parameters
- You estimate them with sample averages: μ̂_k = (1/n) Σ rewards
- Confidence intervals quantify estimation uncertainty

**Bayesian view:**
- Arm means μ_k are random variables with prior distributions
- Each observation updates the distribution via Bayes' rule
- Posterior distributions quantify belief uncertainty

**For bandits:**
- **Frequentist algorithms:** UCB (Upper Confidence Bound) uses confidence intervals
- **Bayesian algorithms:** Thompson Sampling uses posterior distributions

**Commodity example:**

**Frequentist approach:**
- "Based on 50 gold trades, I estimate return = 8% ± 3% (95% CI)"
- "I'm 95% confident the true mean is in [5%, 11%]"

**Bayesian approach:**
- "My prior belief: gold returns are Normal(6%, 4%)"
- "After 50 trades with avg 8%, my posterior is Normal(7.5%, 2%)"
- "I now believe there's a 73% chance gold beats 6%"

Both are valid; Bayesian makes incorporating prior knowledge easier.

## Regret Minimization vs Reward Maximization

### Two Objectives

**Reward maximization:**
```
maximize  Σ_{t=1}^T r_t
```
Goal: Earn as much total reward as possible.

**Regret minimization:**
```
minimize  R(T) = T·μ* - Σ_{t=1}^T r_t
```
Goal: Perform as close to the oracle (best arm in hindsight) as possible.

### Why They Differ

**Scenario:** Three arms with means [10, 9, 8].

**Reward maximizer:**
- Wants high absolute returns
- Might tolerate more exploration if variance is high (searching for lucky streaks)

**Regret minimizer:**
- Wants to match the best arm (mean 10)
- Focuses exploration on distinguishing between top arms, ignores clearly inferior arms

**Practical difference:**

After 100 pulls, you've identified:
- Arm 1: μ̂ = 10.1 ± 0.5
- Arm 2: μ̂ = 9.0 ± 0.5
- Arm 3: μ̂ = 8.0 ± 2.0 (high variance, few samples)

**Reward maximizer:** Might explore Arm 3 hoping for a lucky high-variance payoff.

**Regret minimizer:** Ignores Arm 3 (even upper bound is below Arm 1), focuses on exploiting Arm 1 and occasionally verifying Arm 2 hasn't improved.

**Commodity context:**

**Reward maximization:** A speculative trader might take outsized bets on high-variance commodities (oil options, agricultural futures during weather events) seeking absolute P&L.

**Regret minimization:** A systematic trader wants to match the best possible strategy. They focus on consistently picking the top-Sharpe sector, not chasing volatile outliers.

### Horizon Effects

**Short horizon (T = 100):**
- Information has limited future value (only 100 rounds to exploit learning)
- Exploration should be minimal
- Focus on exploitation with high-confidence estimates

**Long horizon (T = 100,000):**
- Information compounds over many future rounds
- Early exploration is cheap (regret amortizes over huge horizon)
- Worth thorough exploration to find the best arm

**Formula:** Optimal exploration scales roughly as √T in many algorithms.

**Commodity example:** If you're testing strategies for a 3-month trading period, you can't afford weeks of pure exploration. If you're building a 10-year systematic program, spending the first month thoroughly testing strategies is wise.

## How Commodity Traders Actually Make Decisions

### Simplified Mental Model

1. **Estimate expected returns** (fundamental analysis, historical data, models)
2. **Estimate uncertainty** (volatility, correlation, model confidence)
3. **Apply risk constraints** (VaR limits, position sizing, stop-losses)
4. **Optimize allocation** (maximize Sharpe, minimize regret, achieve diversification)
5. **Monitor and adapt** (update estimates, rebalance, detect regime changes)

### Bandit Framing

Each step maps to bandit concepts:

| Trading Step | Bandit Concept |
|--------------|----------------|
| Estimate expected returns | Arm mean estimation (μ̂_k) |
| Estimate uncertainty | Confidence bounds or posterior variance |
| Risk constraints | Utility function (risk aversion) |
| Allocation | Arm selection policy |
| Monitor and adapt | Non-stationary bandits, change detection |

**Example decision:**

A crude oil trader asks: "Should I increase my WTI position or rotate to Brent?"

**Decision theory translation:**
- **Arms:** WTI, Brent, cash
- **Rewards:** Risk-adjusted returns (Sharpe ratio)
- **Uncertainty:** Recent volatility, correlation shifts, sample size
- **Horizon:** 1 month trading window
- **Objective:** Maximize Sharpe (regret minimization relative to best commodity)

**Bandit solution:** Use UCB or Thompson Sampling with Sharpe-ratio rewards and a 1-month horizon. The algorithm automatically balances "keep trading WTI (exploit current winner)" vs "try Brent to see if it's better (explore uncertain alternative)."

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| K | Number of arms (options/strategies) |
| T | Time horizon (total rounds) |
| a(t) | Arm chosen at time t |
| r(t) | Reward received at time t |
| μ_k | True expected reward of arm k |
| μ* | Best arm's expected reward: max_k μ_k |
| μ̂_k | Estimated reward of arm k |
| N_k(t) | Number of times arm k pulled by time t |
| R(T) | Cumulative regret over T rounds |
| U(x) | Utility function |
| EU | Expected utility |
| σ²_k | Variance of arm k's rewards |

## Key Takeaways

1. **Expected value** is the average outcome; **utility** captures risk preferences
2. **Sequential decisions** have both immediate payoffs and information value
3. **Regret minimization** focuses on matching the best arm; **reward maximization** focuses on absolute returns
4. **Bayesian methods** use probability distributions to represent uncertainty; **frequentist methods** use confidence intervals
5. **Horizon length** determines optimal exploration intensity (more time = more exploration)
6. Commodity trading is full of bandit-like decisions: which sector, which strategy, which contract

## Connections

<div class="callout-danger">

**Danger:** Never deploy a bandit system without a kill switch and maximum allocation limits. An unconstrained bandit can allocate 100% of traffic/capital to a single arm, which creates catastrophic risk if the reward signal is noisy or delayed.

</div>


**Builds on:**
- Basic probability (expected value, variance, distributions)
- Statistical inference (confidence intervals, hypothesis testing)
- Optimization (maximization problems, constraints)

**Leads to:**
- Multi-armed bandit algorithms (epsilon-greedy, UCB, Thompson Sampling)
- Risk-sensitive bandits (mean-variance tradeoffs, CVaR constraints)
- Portfolio optimization (bandits with portfolio constraints)
- Bayesian bandits and Thompson Sampling (Module 3)

**Related fields:**
- **Reinforcement learning:** Adds state transitions to bandits (MDPs)
- **Optimal control:** Dynamic programming for sequential decisions
- **Information theory:** Quantifying value of information (entropy, KL divergence)
- **Behavioral economics:** How humans actually make decisions under uncertainty (often suboptimally)

## Practice Problems

### Conceptual Questions

**1. Risk Aversion Impact:**
You have two commodity strategies:
- **Strategy A:** 10% return with 5% volatility (Sharpe = 2.0)
- **Strategy B:** 12% return with 12% volatility (Sharpe = 1.0)

A risk-neutral trader prefers B (higher expected return). At what level of risk aversion (measured by a simple utility function U(r) = r - λ·σ²) does a trader switch to preferring A?

**Hint:** Solve for λ where U(A) = U(B).

**2. Exploration Horizon:**
Suppose you discover a new commodity trading signal with uncertain quality. You estimate:
- 50% chance it has Sharpe = 1.5 (better than your current 1.2)
- 50% chance it has Sharpe = 0.8 (worse)

You can test it for N weeks, then decide whether to adopt it permanently or stick with your current strategy.

**Question:** How many test weeks N justify spending one week of testing (where you might lose 0.4 Sharpe points if it's the bad signal)? Assume you'll trade for 52 more weeks after testing.

### Implementation Challenge

**3. Utility-Based Arm Selection:**
Implement a bandit algorithm that chooses arms based on **expected utility** instead of expected reward:


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np

def utility_based_bandit(arm_means, arm_stds, utility_fn, n_rounds=1000):
    """
    Bandit algorithm using expected utility for risk-averse preferences.

    Parameters:
    -----------
    arm_means : array-like
        True mean rewards (unknown to algorithm, for simulation)
    arm_stds : array-like
        True reward standard deviations (for reward sampling)
    utility_fn : callable
        Utility function: reward → utility (e.g., sqrt for risk aversion)
    n_rounds : int
        Number of rounds to simulate

    Returns:
    --------
    choices : array
        Arm chosen each round
    utilities : array
        Utility achieved each round
    """
    # TODO: Implement this
    # Hints:
    # - Maintain estimates of reward distributions (mean and variance)
    # - Choose arm with highest expected utility
    # - For risk-averse utility, mean-variance approximation:
    #   EU[U(r)] ≈ U(μ) + 0.5·U''(μ)·σ²
    pass

# Test with different utility functions
def risk_neutral(x):
    return x

def risk_averse(x):
    return np.sqrt(np.maximum(x, 0))

# Two arms: high mean/high variance vs low mean/low variance
arm_means = [10, 8]
arm_stds = [5, 2]

# Compare risk-neutral vs risk-averse preferences
```

</div>

**Expected behavior:**
- Risk-neutral: Prefers arm 0 (mean = 10)
- Risk-averse: Might prefer arm 1 (mean = 8, but lower variance gives higher utility)

**Extension:** How does the risk aversion parameter affect exploration? More risk-averse agents might explore less (uncertainty is costly).


---

## Cross-References

<a class="link-card" href="./01_ab_testing_limits.md">
  <div class="link-card-title">01 Ab Testing Limits</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_ab_testing_limits.md">
  <div class="link-card-title">01 Ab Testing Limits — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./02_explore_exploit_tradeoff.md">
  <div class="link-card-title">02 Explore Exploit Tradeoff</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_explore_exploit_tradeoff.md">
  <div class="link-card-title">02 Explore Exploit Tradeoff — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./cheatsheet.md">
  <div class="link-card-title">Cheatsheet</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./cheatsheet.md">
  <div class="link-card-title">Cheatsheet — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

