# Thompson Sampling

> **Reading time:** ~14 min | **Module:** 02 — Bayesian Bandits | **Prerequisites:** Module 1


## In Brief

<div class="flow">
<div class="flow-step mint">1. Set Prior</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">2. Sample from Posterior</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">3. Pull Best Sample</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Observe Reward</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step rose">5. Update Posterior</div>
</div>


<div class="callout-key">

**Key Concept Summary:** Thompson Sampling is a Bayesian bandit algorithm that maintains a probability distribution (belief) over each arm's true reward, samples a plausible reward from each distribution, and selects the a...

</div>

Thompson Sampling is a Bayesian bandit algorithm that maintains a probability distribution (belief) over each arm's true reward, samples a plausible reward from each distribution, and selects the arm with the highest sample. It naturally balances exploration (wide sampling when uncertain) and exploitation (concentrated sampling when confident) through posterior-guided randomness.

> 💡 **Key Insight:** Instead of deterministically choosing based on estimates (like UCB), Thompson Sampling asks: "Given what I know, what could each arm's true reward be?" It samples one possibility from each belief distribution, then acts as if that possibility were true. Over time, beliefs narrow around truth, and exploration naturally decreases.

The magic: exploration happens automatically without tuning parameters. Arms with uncertain beliefs get diverse samples (high exploration); arms with concentrated beliefs give consistent samples (low exploration unless they're truly best).

## Visual Explanation

```
Round 1: Wide, uncertain beliefs
Beta(1,1)   Beta(1,1)   Beta(1,1)
   |█|         |█|         |█|      Sample → Act
  Arm A       Arm B       Arm C

Round 100: Beliefs concentrating
 Beta(55,46)  Beta(61,40)  Beta(48,53)
    |█|          |██|         |█|
   Arm A        Arm B       Arm C
              (Clearly best)

<div class="callout-insight">

**Insight:** Thompson Sampling is often called probability matching because it selects each arm with probability equal to the posterior probability that it is optimal. This natural calibration is why it tends to outperform in practice.

</div>


Round 500: Truth revealed
 Beta(245,256) Beta(298,203) Beta(231,270)
     |█|          |███|          |█|
    Arm A         Arm B         Arm C
                (Still best)

Thompson Sampling: Sample θᵢ ~ Beta(αᵢ, βᵢ), pick argmax θᵢ
Posteriors narrow → Samples concentrate → Exploration fades
```

**Key observation:** Arm B's posterior is both higher (better mean) and tighter (more certain). Thompson Sampling picks it most often, but still samples A and C occasionally when their random draws exceed B's.

## Formal Definition

<div class="callout-warning">

**Warning:** Bandit algorithms assume the reward distributions are stationary (or slowly changing). In commodity markets, regime shifts can make a historically optimal arm suddenly suboptimal. Always implement change detection alongside your bandit.

</div>


**Beta-Bernoulli Thompson Sampling:**

For each arm *i*, maintain a Beta posterior:
```
θᵢ ~ Beta(αᵢ, βᵢ)
```

Where:
- θᵢ = true success probability of arm *i* (unknown)
- αᵢ = successes observed + prior successes
- βᵢ = failures observed + prior failures

**Algorithm:**
```
Initialize: αᵢ ← 1, βᵢ ← 1 for all arms i (uniform prior)

For each round t = 1, 2, 3, ...:
    1. For each arm i, sample: θ̂ᵢ ~ Beta(αᵢ, βᵢ)
    2. Select arm: aₜ = argmaxᵢ θ̂ᵢ
    3. Observe reward: rₜ ∈ {0, 1}
    4. Update posterior:
       If rₜ = 1: αₐₜ ← αₐₜ + 1
       If rₜ = 0: βₐₜ ← βₐₜ + 1
```

**Regret bound:** Thompson Sampling achieves O(log T) expected regret asymptotically, matching UCB's theoretical guarantee.

## Intuitive Explanation

**It's like asking each option to make its best case, then picking whoever sounds most convincing — but options with more evidence give more consistent pitches.**

Imagine you're choosing between three commodity trading signals. You don't know which works best, but you've seen some results:
- Signal A: 55 wins, 46 losses (seems okay, pretty sure)
- Signal B: 61 wins, 40 losses (looks better, moderately confident)
- Signal C: 48 wins, 53 losses (seems worse, reasonably sure)

Thompson Sampling says: "Each signal, tell me what you COULD be."
- A might claim: "I'm actually 54% accurate!" (sampled from Beta(55,46))
- B might claim: "I'm 62% accurate!" (sampled from Beta(61,40))
- C might claim: "I'm 51% accurate!" (sampled from Beta(48,53))

You pick B. Tomorrow, they make new claims (new samples). Sometimes C gets lucky and claims higher than B — you try C, collect data, update beliefs. Over time, the true best reveals itself through concentrated samples.

**Commodity context:** Each commodity (WTI, Gold, Corn) is a trading signal. Weekly returns are Bernoulli "wins" (positive) or "losses" (negative). Thompson Sampling allocates capital by sampling plausible returns from your posterior beliefs, then going all-in on the best sample. As you trade, posteriors tighten around true performance.

## Code Implementation

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import numpy as np
from scipy.stats import beta

class ThompsonSampling:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)  # Prior successes
        self.beta = np.ones(n_arms)   # Prior failures

    def select_arm(self):
        # Sample theta from each arm's posterior
        samples = beta.rvs(self.alpha, self.beta)
        return np.argmax(samples)

    def update(self, arm, reward):
        # Bernoulli reward: 1 = success, 0 = failure
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

# Usage
bandit = ThompsonSampling(n_arms=3)
for t in range(1000):
    arm = bandit.select_arm()
    reward = np.random.binomial(1, true_probs[arm])
    bandit.update(arm, reward)
```

</div>

**That's it.** 15 lines for a state-of-the-art bandit algorithm.

## Common Pitfalls

### Pitfall 1: Using wrong priors for the reward distribution
**Why it happens:** Assuming all rewards are Bernoulli when they're actually continuous (prices, returns).

**How to avoid:** Match the prior to the likelihood:
- Bernoulli rewards → Beta prior
- Gaussian rewards → Normal prior (or Normal-Gamma for unknown variance)
- Count data → Gamma prior (Poisson likelihood)

**Commodity example:** Commodity returns are Gaussian, not Bernoulli. Use Normal-Normal conjugacy, not Beta-Bernoulli.

### Pitfall 2: Forgetting to handle non-stationarity
**Why it happens:** Posteriors keep accumulating evidence from the past, even when regimes change.

**How to avoid:**
- Use exponential discounting: `alpha[arm] *= 0.99` each period
- Sliding window: only use last N observations
- Change detection: reset posteriors when distribution shifts detected

**Commodity example:** A commodity that worked in contango may fail in backwardation. Discount old evidence or detect regime changes.

### Pitfall 3: Starting with overly strong priors
**Why it happens:** Choosing Beta(10, 10) instead of Beta(1, 1) to "be Bayesian."

**How to avoid:** Weak priors (e.g., Beta(1,1) = uniform) let data dominate quickly. Strong priors (e.g., Beta(100,100)) slow learning dramatically. For bandits, weak priors usually win.

**Trade-off:** Strong priors prevent early over-commitment to lucky arms, but delay convergence. Use Beta(1,1) unless you have genuine prior information.

## Connections

<div class="callout-danger">

**Danger:** Never deploy a bandit system without a kill switch and maximum allocation limits. An unconstrained bandit can allocate 100% of traffic/capital to a single arm, which creates catastrophic risk if the reward signal is noisy or delayed.

</div>


**Builds on:**
- Module 0: Decision theory and regret minimization
- Module 1: Epsilon-greedy and UCB algorithms
- Bayesian Commodity Forecasting course: Posterior updating with conjugate priors

**Leads to:**
- Module 3: Contextual Thompson Sampling with linear models
- Module 6: Non-stationary bandits with discounted Thompson Sampling
- Bayesian optimization: Thompson Sampling for hyperparameter tuning

**Mathematical foundation:**
- Conjugate priors: Beta-Bernoulli, Normal-Normal, Gamma-Poisson
- Posterior predictive distribution: marginalizing over parameter uncertainty
- Information theory: Thompson Sampling performs information-directed sampling

## Practice Problems

### Problem 1: Implement from scratch
Implement Thompson Sampling for a 4-arm Bernoulli bandit with true probabilities [0.3, 0.5, 0.45, 0.4]. Run for 1000 rounds. Plot cumulative regret.

**Extension:** Compare Beta(1,1), Beta(0.5, 0.5), and Beta(5,5) as priors. Which converges fastest?

### Problem 2: Posterior evolution
Create a 3-arm problem where true probabilities are very close: [0.48, 0.50, 0.49]. Run Thompson Sampling for 5000 rounds. Plot the posterior distributions at rounds 100, 500, 1000, 5000.

**Question:** When do the posteriors clearly separate? How much data is needed to reliably identify the best arm?

### Problem 3: Commodity trading signal selection
You have 4 commodity trading signals with unknown win rates. Frame this as a Bernoulli bandit where each signal produces binary outcomes (profit/loss on each trade). Implement Thompson Sampling to adaptively choose which signal to follow each day.

**Data:** Simulate signals with win rates [0.52, 0.48, 0.55, 0.50] (realistic for trading).

**Analysis:** How many trades before you reliably identify the 0.55 signal? What's your cumulative P&L vs equal allocation?

### Problem 4: Exploration decay
Instrument Thompson Sampling to track the "exploration rate" = fraction of times the empirically-best arm is NOT selected. Plot this over time.

**Expected behavior:** Exploration rate should decay roughly as 1/t. Verify this empirically.

**Commodity context:** In live trading, you want exploration to fade but never fully stop (in case regimes change). What exploration rate is acceptable after 500 trades?


---

## Cross-References

<a class="link-card" href="./02_posterior_updating.md">
  <div class="link-card-title">02 Posterior Updating</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_posterior_updating.md">
  <div class="link-card-title">02 Posterior Updating — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./03_thompson_vs_ucb.md">
  <div class="link-card-title">03 Thompson Vs Ucb</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_thompson_vs_ucb.md">
  <div class="link-card-title">03 Thompson Vs Ucb — Companion Slides</div>
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

