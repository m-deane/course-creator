# Regret Theory for Multi-Armed Bandits

## TL;DR

Regret measures how much worse your bandit strategy performed compared to always pulling the best arm. The best algorithms achieve logarithmic regret — O(ln T) — meaning the penalty for learning grows very slowly over time.

## Visual Explanation

```
Cumulative Regret Over Time

Regret │
       │   Random (linear)
  500  │   ╱
       │  ╱
       │ ╱      ε-greedy (linear but slower)
  250  │╱       ╱─────────────────────
       │      ╱
       │    ╱     UCB / Thompson (logarithmic)
       │  ╱      ╭─────────────────────
       │╱      ╭─╯
       │─────╭─╯
    0  └────────────────────────────────► Time
       0    500   1000  1500  2000

Key insight: Logarithmic regret means the "cost of learning"
grows slower and slower. After 1000 rounds, you're barely
accumulating any additional regret.
```

## Intuitive Analogy

Think of regret as the "tuition you pay" while learning which restaurant in a new city is best. A naive approach (trying random restaurants forever) keeps paying full tuition. A smart approach (Thompson Sampling) pays most of its tuition in the first few weeks, then rarely picks a bad meal again.

## Formal Definition

**Pseudo-regret** after T rounds:

$$\bar{R}_T = T \cdot \mu^* - \sum_{t=1}^{T} \mu_{A_t}$$

Where:
- $\mu^* = \max_a \mu_a$ is the mean reward of the best arm
- $A_t$ is the arm selected at time $t$
- $\mu_{A_t}$ is the expected reward of the selected arm

**Per-arm gap**: $\Delta_a = \mu^* - \mu_a$

**Lai-Robbins Lower Bound** (1985):

For any consistent policy:

$$\liminf_{T \to \infty} \frac{\bar{R}_T}{\ln T} \geq \sum_{a: \Delta_a > 0} \frac{\Delta_a}{\text{KL}(\mu_a, \mu^*)}$$

This says no algorithm can do better than logarithmic regret asymptotically.

**UCB1 Upper Bound** (Auer et al., 2002):

$$\bar{R}_T \leq 8 \sum_{a: \Delta_a > 0} \frac{\ln T}{\Delta_a} + \left(1 + \frac{\pi^2}{3}\right) \sum_{a=1}^{K} \Delta_a$$

**Thompson Sampling**: Achieves the Lai-Robbins bound asymptotically (Kaufmann et al., 2012).

## When Regret Analysis Matters in Practice

Regret analysis is most useful when:

1. **Comparing algorithms**: UCB and Thompson both achieve O(ln T), but Thompson often has smaller constants
2. **Setting expectations**: If your best arm has reward 0.06 and second-best has 0.04, the gap Δ = 0.02 means regret accumulates slowly (problem is "easy")
3. **Detecting problems**: If your regret is growing linearly, something is wrong (non-stationarity, implementation bug, wrong model)

## Commodity Context

For a commodity allocator:
- **T** = number of weekly allocation decisions (52/year)
- **μ\*** = return of always picking the best commodity sector
- **Regret** = how much less you earned while learning which sector was best

With T=52 and 5 arms, a good algorithm might accumulate total regret equivalent to ~10 weeks of sub-optimal allocation in year 1, but only ~2 additional weeks in year 2. The "learning cost" is front-loaded.

## Key References

- Lai, T.L. and Robbins, H. (1985). "Asymptotically efficient adaptive allocation rules." *Advances in Applied Mathematics*.
- Auer, P., Cesa-Bianchi, N., and Fischer, P. (2002). "Finite-time analysis of the multiarmed bandit problem." *Machine Learning*.
- Kaufmann, E., Korda, N., and Munos, R. (2012). "Thompson Sampling: An Asymptotically Optimal Finite-Time Analysis." *ALT*.
