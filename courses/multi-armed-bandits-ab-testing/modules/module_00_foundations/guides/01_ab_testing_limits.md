# The Limits of A/B Testing

## In Brief

A/B testing allocates traffic evenly between options until statistical significance is reached, but this means you keep sending 50% of traffic to the inferior option even after it's clearly losing. In commodity trading, this translates to continuing to allocate capital to underperforming strategies while evidence accumulates.

## Key Insight

**A/B testing is a snapshot; bandits are a steering wheel.** Traditional A/B tests prioritize certainty over efficiency — they minimize false positives but maximize opportunity cost. In dynamic environments like commodity markets, you need to learn AND earn simultaneously, not wait for perfect statistical confidence before acting.

## Visual Explanation

```
A/B TESTING (Fixed Allocation)
Time:      0%    25%    50%    75%   100%
           |------|------|------|------|
Option A:  50%    50%    50%    50%    50%   ← Always 50% even if losing
Option B:  50%    50%    50%    50%    50%

BANDIT (Adaptive Allocation)
Time:      0%    25%    50%    75%   100%
           |------|------|------|------|
Option A:  50%    35%    20%    10%     5%   ← Reduces losing arm quickly
Option B:  50%    65%    80%    90%    95%   ← Exploits winner sooner

Cumulative Regret:
A/B Test:  ████████████████████ (wastes ~40-45% of traffic)
Bandit:    ██████░░░░░░░░░░░░░░ (wastes ~15-20% of traffic)
```

## Formal Definition

**A/B Testing Framework:**

An A/B test allocates equal traffic n/2 to each of two treatments A and B, collecting observations until the test statistic:

```
z = (p̂_B - p̂_A) / √(p̂(1-p̂)(2/n))
```

exceeds the critical value z_(α/2) for significance level α, where p̂ is the pooled proportion.

**Sample Size Formula:**
```
n = 2(z_(α/2) + z_β)² · p̄(1-p̄) / (p_B - p_A)²
```

where p̄ = (p_A + p_B)/2, and β is the Type II error rate (1 - power).

**The Cost:** During the entire test duration T, you allocate T/2 observations to the inferior arm, accumulating regret:
```
R(T) = (T/2) · |μ_A - μ_B|
```

This regret is **linear in T**, meaning the longer you test, the more you waste.

## Intuitive Explanation

**It's like testing two restaurant delivery services for your office:**

**A/B Testing approach:**
- Week 1-4: Order from each restaurant 50% of the time
- Week 5: Analyze data, achieve statistical significance
- Result: You ate bad food 50% of the time for a month to be 95% confident

**Bandit approach:**
- Week 1: Try each restaurant 50/50
- Week 2: The good one is clearly better, shift to 70/30
- Week 3: Strong evidence now, shift to 85/15
- Week 4: Occasionally still try the bad one (exploration), but mostly enjoy good food
- Result: You ate good food ~75% of the time AND learned which was better

In commodity trading: You have two crude oil trading strategies. A/B testing keeps allocating 50% of capital to the losing strategy for weeks while you gather "statistically significant evidence." A bandit shifts capital toward the winner within days while maintaining small exploratory bets.

## Code Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

def ab_test_simulation(p_A=0.05, p_B=0.08, n_trials=10000):
    """Simulate A/B test showing wasted traffic on inferior arm."""

    # True conversion rates (unknown to the tester)
    arms = np.array([p_A, p_B])
    best_arm = np.argmax(arms)

    # A/B test: always 50/50 allocation
    choices = np.random.choice([0, 1], size=n_trials)
    rewards = np.random.binomial(1, arms[choices])

    # Calculate cumulative regret
    optimal_reward = arms[best_arm]
    actual_rewards = arms[choices]
    regret = optimal_reward - actual_rewards
    cumulative_regret = np.cumsum(regret)

    # Visualization
    plt.figure(figsize=(10, 4))
    plt.plot(cumulative_regret, linewidth=2)
    plt.xlabel('Trial Number')
    plt.ylabel('Cumulative Regret')
    plt.title('A/B Test Waste: Regret Grows Linearly')
    plt.grid(alpha=0.3)

    total_waste = cumulative_regret[-1]
    print(f"Total regret after {n_trials} trials: {total_waste:.1f}")
    print(f"Could have earned {total_waste:.1f} more conversions")

    return cumulative_regret

# Example: Two crude oil strategies with different win rates
regret = ab_test_simulation(p_A=0.05, p_B=0.08, n_trials=10000)
```

## Common Pitfalls

### 1. The Peeking Problem
**What it is:** Checking results before reaching predetermined sample size and stopping early if you see significance.

**Why it breaks:** Inflates Type I error rate from 5% to 20-30%. Random walks cross significance thresholds spuriously.

**Commodity example:** You test two gold trading signals, see one ahead after 500 trades, stop the test. Turns out it was just luck — the signal degrades over the next month.

### 2. Simpson's Paradox
**What it is:** A trend appears in subgroups but reverses when groups are combined.

**Why it matters:** Aggregating across different market conditions can hide regime-dependent performance.

**Commodity example:** Strategy A beats Strategy B during both high-volatility and low-volatility periods, but appears worse overall because you tested it more during unfavorable conditions.

### 3. Fixed Horizon Fallacy
**What it is:** Believing you must wait until predetermined sample size even when one option is clearly dominant.

**Why it's costly:** In trading, "clearly losing" often emerges within 10-20% of planned samples, but you waste the remaining 80% gathering unnecessary precision.

**Commodity example:** After 200 natural gas trades, one strategy is down 15% and the other is up 12%. You don't need 800 more trades to know which to prefer — but A/B testing says wait.

### 4. Non-Stationarity Blindness
**What it is:** Assuming the true parameters remain constant throughout the test.

**Why it fails in trading:** Market regimes shift. The "winning" strategy during test weeks 1-3 may be the loser in week 4 after a volatility regime change.

**Commodity example:** You test two wheat inventory strategies during a calm market period. Results say Strategy A wins. Then a supply shock hits (Ukraine conflict, drought), and the optimal strategy flips. Your A/B test result is now obsolete.

## Connections

**Builds on:**
- Basic probability and statistics (expected value, variance)
- Hypothesis testing fundamentals (p-values, confidence intervals)
- Statistical power and sample size calculations

**Leads to:**
- Multi-armed bandit algorithms (epsilon-greedy, UCB, Thompson Sampling)
- Regret analysis and theoretical guarantees
- Online learning and adaptive experimentation
- Contextual bandits for personalized decisions

**Related concepts:**
- Sequential analysis and early stopping rules (closer to bandits than fixed A/B tests)
- Bayesian A/B testing (updates beliefs continuously, but still typically uses fixed allocation)
- Multi-variate testing (even worse: K arms means only 1/K traffic per variant)

## Practice Problems

### Conceptual Questions

**1. Sample Size Trap:**
You're testing two commodity trading algorithms. Algorithm A has a true Sharpe ratio of 1.2, Algorithm B has 1.5. Using a standard A/B test with 80% power and α=0.05, you calculate you need 1,000 trades per algorithm.

After 500 trades each, Algorithm B is clearly outperforming (observed Sharpe: 1.51 vs 1.18, p=0.03).

**Question:** What is the opportunity cost of continuing the A/B test to completion? Assume you trade $10,000 per signal and Sharpe ratio approximates excess return / volatility where volatility is ~15% annually.

**Hint:** Calculate expected profit difference over the remaining 500 trades.

**2. Non-Stationarity Scenario:**
You run a 4-week A/B test on two crude oil inventory signals. Week 1-2: Signal A wins (p=0.04). Week 3-4: Signal B wins (p=0.03). Overall 4-week result: Signal A wins (p=0.08).

**Question:** Why is this result misleading? What bandit approach would handle this better? What does this tell you about market regime changes?

### Implementation Challenge

**3. Regret Calculator:**
Implement a function that calculates cumulative regret for any allocation strategy given true arm means:

```python
def calculate_regret(arm_means, choices, rewards=None):
    """
    Calculate cumulative regret for a sequence of arm choices.

    Parameters:
    -----------
    arm_means : array-like
        True expected reward for each arm
    choices : array-like
        Sequence of arm indices chosen (0, 1, 2, ...)
    rewards : array-like, optional
        Actual rewards received (if None, assume deterministic = means)

    Returns:
    --------
    cumulative_regret : np.array
        Cumulative regret at each time step
    """
    # TODO: Implement this
    pass

# Test case:
arm_means = [0.05, 0.08, 0.06]  # Three strategies
ab_test_choices = [0, 1, 2] * 1000  # Round-robin allocation
regret = calculate_regret(arm_means, ab_test_choices)
print(f"Total regret: {regret[-1]:.2f}")
# Should show regret accumulating linearly
```

**Expected behavior:** For round-robin allocation across K arms, regret should grow as T · (μ* - μ_avg) where μ* is the best arm mean and μ_avg is the average arm mean.

**Extension:** Compare the regret curve for:
- Round-robin allocation (A/B test with K arms)
- Random allocation
- Always picking arm 1 (pure exploitation, wrong arm)
- Always picking best arm (oracle, zero regret baseline)
