# Deep Dive: Bayesian Bandits Theory

## TL;DR
Bayesian bandits maintain probability distributions (beliefs) over each arm's reward, update via Bayes' rule, and make decisions that balance expected reward and information gain. Thompson Sampling is the practical approximation of the theoretically optimal Gittins index policy.

> 💡 **Key Insight:** Classical bandits (UCB) use frequentist statistics: estimate means, compute confidence bounds.

Bayesian bandits use probability: maintain distributions over unknown parameters, sample from posteriors.

**Why Bayesian?**
1. Natural uncertainty quantification (posterior width = exploration need)
2. Incorporates prior knowledge (if you have it)
3. Optimal for finite horizons (Gittins index)
4. Empirically excellent (Thompson Sampling)

## Visual Explanation

```
BAYESIAN UPDATING CYCLE

Prior Belief          Observe Data           Posterior Belief
(Before seeing)    →  (Evidence)          →  (After seeing)

Beta(1,1)             Success!               Beta(2,1)
 Uniform              ───────→               Shifted right
  |█████|                                     |░███░|

Beta(2,1)             Failure                Beta(2,2)
 Optimistic           ───────→               Centered
  |░███░|                                     |░░█░░|

Beta(2,2)             Success!               Beta(3,2)
 Uncertain            ───────→               Learning
  |░░█░░|                                     | ░██░|

After 100 observations:
Beta(55,46) → |███| (confident, ~54% success rate)
```

## Formal Framework

**Bayesian Bandit Model:**

For each arm i:
- True parameter: θᵢ (unknown)
- Prior belief: p(θᵢ) (before data)
- Likelihood: p(r | θᵢ) (reward given parameter)
- Posterior: p(θᵢ | data) ∝ p(data | θᵢ) · p(θᵢ)

**Update Rule (Bayes' Theorem):**
```
p(θᵢ | r₁,...,rₜ) = p(r₁,...,rₜ | θᵢ) · p(θᵢ) / p(r₁,...,rₜ)
```

**Conjugate Pairs (easy updating):**

1. **Beta-Bernoulli** (binary rewards)
   ```
   Prior: θ ~ Beta(α, β)
   Likelihood: r ~ Bernoulli(θ)
   Posterior: θ | r ~ Beta(α+r, β+(1-r))
   ```

2. **Normal-Normal** (continuous rewards, known variance)
   ```
   Prior: μ ~ N(μ₀, σ₀²)
   Likelihood: r ~ N(μ, σ²)
   Posterior: μ | r ~ N((σ²μ₀ + σ₀²r)/(σ² + σ₀²), ...)
   ```

3. **Gamma-Poisson** (count data)
   ```
   Prior: λ ~ Gamma(α, β)
   Likelihood: r ~ Poisson(λ)
   Posterior: λ | r ~ Gamma(α+r, β+1)
   ```

**Thompson Sampling Decision Rule:**
```
For each arm i:
  Sample θ̂ᵢ ~ p(θᵢ | data)
Pick arm i* = argmaxᵢ θ̂ᵢ
```

This is probability matching: allocate trials proportional to P(arm i is best).

## Gittins Index: The Optimal Solution

**Gittins Index Theorem (1979):**

For infinite-horizon discounted bandits, there exists an index Gᵢ(state) for each arm such that the optimal policy is:
```
Always pull arm i* = argmaxᵢ Gᵢ
```

**What is the Gittins index?**
For a given arm in state s (posterior):
```
G(s) = sup {λ : E[Σₜ γᵗ rₜ | continue] ≥ E[Σₜ γᵗ λ | stop]}
```

Interpretation: The fair price at which you'd be indifferent between:
- Continuing to pull this arm
- Switching to a known-reward alternative paying λ forever

**Why we don't use it:**
- Hard to compute (dynamic programming over continuous state space)
- Requires discount factor γ
- Infinite-horizon assumption

**Why Thompson Sampling works:**
Thompson Sampling approximates Gittins index under certain conditions. Empirically matches its performance in most cases.

## Connection to Bayesian Decision Theory

**Bayesian approach to bandits:**

Goal: Maximize expected cumulative reward under posterior beliefs.

```
π* = argmaxπ Eₚ(θ|data) [Σₜ r(aₜ, θ)]
```

**Information value:**
Pulling an arm gives reward r but also information I about θ.

Bayesian bandits implicitly balance:
- **Exploitation:** E[r | current beliefs]
- **Exploration:** E[future value from info | current beliefs]

This is called **information-directed sampling**.

**Theoretical result (Russo & Van Roy):**
Thompson Sampling performs near-optimal information-directed sampling.

## Intuitive Explanation

**Analogy: Job candidate evaluation (Bayesian view)**

You have 3 candidates. Each has true ability θᵢ (unknown).

- **Prior:** Before interviews, assume all are equally skilled (Beta(1,1) = uniform)
- **Interview = Pull:** Each interview reveals noisy evidence about θᵢ
- **Update:** After each interview, update your belief about that candidate
- **Decision:** Sample from beliefs ("If this candidate is as good as I think, pick them")
- **Posterior width = uncertainty:** 
  - Wide posterior (2 interviews) → frequently re-test
  - Narrow posterior (20 interviews) → rarely re-test

**Bayesian vs Frequentist:**
- Frequentist (UCB): "Candidate has mean score μ̂ ± confidence interval"
- Bayesian (Thompson): "I believe candidate's score is θ ~ Beta(α, β)"

Same information, different language. Bayesian is more natural for sequential decisions.

## Why Thompson Sampling Works So Well

**Theoretical properties:**
1. **Optimal for 2 arms:** Matches Gittins index exactly
2. **Optimal asymptotically:** Achieves O(log T) regret
3. **Information-directed:** Near-optimal info/reward tradeoff
4. **Finite-horizon aware:** Adjusts exploration based on remaining time

**Practical advantages:**
1. **No parameter tuning:** Just choose priors (often uniform is fine)
2. **Easy to implement:** 5 lines of code for Beta-Bernoulli
3. **Handles delayed rewards:** Bayesian updating works with batches
4. **Extends naturally:** Contextual, non-stationary, structured bandits

**Empirical observation:**
In practice, Thompson Sampling often beats UCB and other methods, especially:
- Early in learning (finite horizon)
- Non-stationary environments
- Structured reward models

## Connection to the Bayesian Commodity Forecasting Course

**Commodity forecasting is Bayesian modeling:**
- Prior: Belief about commodity returns, volatility, regime
- Data: Observed prices, fundamentals
- Posterior: Updated beliefs about future returns

**Bandit connects forecasting to action:**
- Forecast: p(return | data) for each commodity
- Bandit: How to allocate capital given forecasts
- Thompson Sampling: Sample from forecast distributions, pick best

**Example workflow:**
1. **Bayesian forecast:** "WTI return ~ N(0.5%, 3%)" (posterior)
2. **Thompson Sampling:** Sample from this distribution
3. **Decision:** If WTI sample > others → allocate more to WTI
4. **Observe:** Actual return
5. **Update forecast:** Bayesian update on commodity model
6. **Repeat:** Next week

**Key insight:**
If you're already doing Bayesian commodity modeling, Thompson Sampling is the natural way to turn forecasts into decisions.

## When Bayesian Bandits Excel

**Use Bayesian bandits when:**
1. You have meaningful priors (domain knowledge)
2. Rewards have natural probabilistic interpretation
3. Want automatic exploration tuning
4. Working with non-stationary rewards (discounting is natural in Bayesian framework)
5. Need to explain decisions to stakeholders ("We believe Gold has 70% chance of outperforming")

**Classical bandits (UCB) might be better when:**
1. Need worst-case guarantees (minimax regret)
2. Adversarial environments (rewards chosen to hurt you)
3. Want to avoid Bayesian assumptions

**In commodity trading:**
Bayesian bandits are natural because:
- Commodity returns have distributional structure (not adversarial)
- Prior knowledge exists (historical volatility, correlations)
- Non-stationarity is common (regimes change)
- Stakeholders understand "confidence" language

## Practice: Bayesian Updating

**Exercise:** Commodity win rates

You're tracking 3 commodity strategies. Each generates weekly wins/losses.

Week 1: All go 1-0 (one win each)
```
Prior: Beta(1,1)
After 1 win: Beta(2,1)
Mean = 2/3 ≈ 67% (optimistic, but uncertain)
```

Week 5: Strategy A is 4-1, B is 2-3, C is 3-2
```
A: Beta(5,2) → mean 5/7 ≈ 71%
B: Beta(3,4) → mean 3/7 ≈ 43%
C: Beta(4,3) → mean 4/7 ≈ 57%
```

Thompson Sampling:
```python
samples = [beta.rvs(5,2), beta.rvs(3,4), beta.rvs(4,3)]
# samples ≈ [0.75, 0.38, 0.61]
# Pick A (highest sample)
```

After 50 weeks: A is 32-18, B is 19-31, C is 25-25
```
A: Beta(33,19) → mean 63%, std 7% (confident)
B: Beta(20,32) → mean 38%, std 7% (confident loser)
C: Beta(26,26) → mean 50%, std 7% (coin flip, but confident it's mediocre)

Thompson Sampling now picks:
  A: ~95% of the time
  C: ~5% of the time
  B: ~0% of the time (clearly inferior)
```

**Notice:** Exploration decays naturally as posteriors tighten.

---

**Remember:** Bayesian bandits are about maintaining beliefs and sampling decisions from those beliefs. Thompson Sampling is the practical way to do this optimally.
