# Deep Dive: Regret Theory

## TL;DR
Regret measures how much reward you lost by not knowing the best arm from the start. Optimal policies achieve O(log T) regret — logarithmic growth means you learn quickly and waste less over time. Linear regret O(T) means you never learned.

## Visual Explanation

```
REGRET OVER TIME

Cumulative Reward:
  ┌───────────────────────────────────┐
  │                                   │
  │    Optimal (oracle knows best)    │
  │   ╱──────────────────────────     │
  │  ╱                                │
  │ ╱    ← Regret = this gap          │
  │╱                                  │
  │      Thompson Sampling            │
  │     ╱─────────────────            │
  │    ╱                              │
  │   ╱  Random (no learning)         │
  │  ╱  ╱──────────                   │
  │ ╱  ╱                              │
  │╱  ╱                               │
  └───────────────────────────────────┘
   0              Rounds (T)         →

Random: R(T) = O(T) — grows linearly
Thompson: R(T) = O(log T) — flattens out
Optimal: R(T) = 0
```

## Formal Definition

**Cumulative Regret:**
```
R(T) = Σₜ₌₁ᵀ (μ* - μₐₜ)

Where:
  μ* = mean reward of best arm
  μₐₜ = mean reward of arm played at round t
  R(T) = total reward lost vs oracle
```

**Per-Arm Regret:**
```
R(T) = Σᵢ E[nᵢ(T)] · Δᵢ

Where:
  nᵢ(T) = times arm i was pulled
  Δᵢ = μ* - μᵢ (gap from best)
```

**Lai-Robbins Lower Bound:**
Any consistent policy must have:
```
R(T) ≥ Σᵢ:Δᵢ>0 (log T / KL(μᵢ, μ*)) · Δᵢ
```

Where KL = Kullback-Leibler divergence. This is Ω(log T).

**UCB Regret Bound:**
```
R(T) ≤ Σᵢ:Δᵢ>0 (8 log T / Δᵢ) + (1 + π²/3) · Σᵢ Δᵢ
     = O(log T)
```

UCB achieves the optimal rate up to constants.

## Intuitive Explanation

**Analogy:** Job candidates

You're hiring and have 3 candidates. You can interview them repeatedly (with some randomness each time).

- **Random hiring:** R(T) = O(T)
  - You pick randomly forever → never learn → lose Δ every round
  - After 1000 hires, you've wasted ~500 on suboptimal candidates

- **Smart hiring (UCB/Thompson):** R(T) = O(log T)
  - First 10 hires: Try everyone, learn quickly
  - Next 90 hires: Mostly pick best, occasionally verify others
  - After 100 hires: 95% confident in best, rarely pick others
  - After 1000 hires: Total mistakes ≈ 20 (not 500!)

**The logarithm appears because:** 
- Confidence grows as √n
- Need n doublings to gain each bit of certainty
- Doublings: 1, 2, 4, 8, 16, ... → log₂(T) doublings
- Total regret across doublings: O(log T)

## When Regret Analysis Matters in Practice

**Regret is THE metric when:**
1. **You can't pause operations** (commodity trading, content publishing)
2. **Exploration is expensive** (each sub-optimal choice costs real money)
3. **Many rounds** (enough data for logarithmic advantage to matter)
4. **Stationarity** (reward distributions don't change wildly)

**Example: Commodity allocator over 1 year (52 weeks)**
- Random allocation: Regret ≈ 15% (linear growth)
- Thompson Sampling: Regret ≈ 3% (logarithmic, learns quickly)
- Difference: 12% of portfolio value = $12K on $100K portfolio

**Regret is LESS important when:**
1. **One-shot decision** (pick best after fixed exploration, then stop)
2. **Need p-values** (A/B testing with statistical inference)
3. **Non-stationary** (rewards change faster than you can learn)

## Connections

**Builds on:**
- Information theory: Optimal learning requires log(1/ε) samples
- Concentration inequalities: Confidence bounds grow as √(log T / n)

**Leads to:**
- Algorithm design: UCB, Thompson Sampling designed to minimize regret
- Bayesian optimization: Regret bounds for continuous optimization
- Online learning: Adversarial bandits, expert algorithms

**Appears in:**
- Recommendation systems (minimize regret of showing bad content)
- Clinical trials (minimize harm from inferior treatments)
- Online advertising (minimize revenue lost to exploration)

## Commodity Context

**What regret means for a commodity allocator:**

Over 1 year, you make 52 weekly allocation decisions.

**Scenario 1: Pure equal-weight (no learning)**
- Return: 8% (average of all commodities)
- Regret: 8% (vs best single commodity at 16%)

**Scenario 2: Two-wallet bandit (Thompson Sampling)**
- Return: 12% (tilts toward winners)
- Regret: 4% (vs best single at 16%)
- Improvement: 4% = $4K on $100K portfolio

**Logarithmic regret means:**
- Most learning happens in first 10-15 weeks
- By week 30, allocation is ~90% optimal
- Weeks 30-52 have minimal additional regret
- Total regret stays bounded, doesn't grow linearly

**Practical implication:**
After 6 months of bandit learning, you've:
- Found the better commodities (Gold, Copper)
- Reduced exposure to worse ones (NatGas if weak)
- Regret ≈ 3-5% (acceptable for adaptive learning)
- Vs A/B test that takes 12 weeks just to declare a winner

---

**Remember:** Logarithmic regret is the signature of learning. Linear regret means you never learned. Optimal is O(log T) — you can't do better asymptotically.
