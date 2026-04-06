# Conversion Optimization with Bandits

> **Reading time:** ~20 min | **Module:** 04 — Content Growth Optimization | **Prerequisites:** Module 3


## In Brief


<div class="callout-key">

**Key Concept Summary:** Website conversion testing with bandits solves a simple problem: **why waste traffic on inferior landing pages when you could be tilting toward winners while you learn?** Traditional A/B testing locks

</div>

Website conversion testing with bandits solves a simple problem: **why waste traffic on inferior landing pages when you could be tilting toward winners while you learn?** Traditional A/B testing locks you into 50/50 splits for weeks, costing real revenue. Thompson Sampling for Beta-Bernoulli models gives you the optimal balance: explore early when uncertain, exploit heavily once you're confident.

**Why it matters for commodity traders:** When you're testing research report formats, email subject lines, or alert delivery channels, every "conversion" is a trader taking action. Wasting half your audience on the worse variant for 4 weeks means missed trades and lost trust.

## Key Insight

<div class="callout-insight">

**Insight:** The core insight of bandit algorithms is that learning and earning are not separate phases. Every observation contributes to both understanding which option is best and generating value from the best option.

</div>


Conversion optimization is the **perfect use case for Bayesian bandits**:
- **Binary outcomes:** Visitor converts (1) or doesn't (0) → Bernoulli
- **Unknown conversion rate:** Each variant has true p ∈ [0,1] → Beta prior
- **High opportunity cost:** Every visitor sent to a bad page is lost revenue
- **Easy to implement:** Thompson Sampling for Beta-Bernoulli is 10 lines of code

The math is beautiful: after observing successes S and failures F, your posterior belief is Beta(1+S, 1+F). Sample from it, pick the best sample, observe result, update. You'll converge to the best variant faster than any A/B test while losing less money along the way.

## Visual Explanation

<div class="callout-warning">

**Warning:** Bandit algorithms assume the reward distributions are stationary (or slowly changing). In commodity markets, regime shifts can make a historically optimal arm suddenly suboptimal. Always implement change detection alongside your bandit.

</div>


```
TRADITIONAL A/B TEST vs BANDIT

A/B Test (4 weeks, 10,000 visitors):
┌────────────────────────────────────┐
│ Variant A: 5,000 visitors → 150    │  3.0% conversion
│ Variant B: 5,000 visitors → 300    │  6.0% conversion
└────────────────────────────────────┘
Result: B is better (finally!), wasted 5,000 visitors on A
Lost conversions: 5,000 × 0.03 = 150 (should have been 5,000 × 0.06 = 300)
Opportunity cost: 150 conversions over 4 weeks

Thompson Sampling (same 10,000 visitors):
┌────────────────────────────────────┐
│ Week 1: 50/50 split (learning)     │
│ Week 2: 30/70 split (tilting to B) │
│ Week 3: 15/85 split (confident)    │
│ Week 4:  5/95 split (exploiting)   │
└────────────────────────────────────┘
Result: Adaptive allocation saves ~75 conversions
Plus: You can stop early if B is clearly winning

POSTERIOR EVOLUTION (Thompson Sampling):
                 Start           After 100        After 500        After 2000
Variant A:    Beta(1,1)      Beta(4, 96)     Beta(16, 484)    Beta(61, 1939)
              [flat]         [peaked at 3%]   [tight at 3%]    [very tight]
Variant B:    Beta(1,1)      Beta(7, 93)     Beta(31, 469)    Beta(121, 1879)
              [flat]         [peaked at 6%]   [tight at 6%]    [very tight]

Action: Sample from both → B samples higher → send visitor to B → update posterior
```

## Formal Definition

**Problem Setup:**
- K variants (landing pages, headlines, CTAs, prices)
- Each variant k has unknown conversion rate θ_k ∈ [0, 1]
- Visitor t arrives, you show variant a_t, observe conversion r_t ∈ {0, 1}
- Goal: Maximize cumulative conversions Σ r_t

**Thompson Sampling for Conversion Rates:**

Initialize:
```
For each variant k:
  α_k ← 1  (prior successes)
  β_k ← 1  (prior failures)
```

For each visitor t = 1, 2, ..., T:
```
1. Sample: θ̃_k ~ Beta(α_k, β_k) for each variant k
2. Select: a_t = argmax_k θ̃_k (show variant with highest sample)
3. Observe: conversion r_t ∈ {0, 1}
4. Update:
   α_{a_t} ← α_{a_t} + r_t
   β_{a_t} ← β_{a_t} + (1 - r_t)
```

**Regret Bound:**
```
E[Regret_T] = O(K log T)
vs A/B test: Regret_T = Θ(T) (linear waste)
```

**Multi-Step Funnel Extension:**
```
For funnel with steps S_1, S_2, ..., S_n:
  Run independent bandit at each step
  Metric: conversion rate for that step (not end-to-end)
  Why: Faster learning, localized optimization
```

## Intuitive Explanation

Imagine you're a commodity research firm testing two landing pages for your weekly report signup:

- **Variant A:** "Get Professional Market Analysis" (true conversion: 3%)
- **Variant B:** "Join 10,000 Traders Who Trade Smarter" (true conversion: 6%)

**A/B test approach:**
- Week 1-4: Send 50% to each, collect data
- Week 5: Declare B the winner
- Cost: 4 weeks × 1,000 visitors/week × 50% = 2,000 visitors to the worse variant A
- Lost signups: 2,000 × (0.06 - 0.03) = 60 traders you didn't convert

**Thompson Sampling approach:**
- Day 1: Start with 50/50 (no data yet)
- Day 3: B shows 8% conversion (lucky), A shows 2% (unlucky) → shift to 30/70
- Week 2: Posteriors stabilize around true rates → shift to 10/90
- Week 4: Barely showing A anymore (just 5% for monitoring)
- Cost: Maybe 500 visitors to A (vs 2,000 for A/B test)
- Lost signups: ~15 (vs 60 for A/B test)

**Why it works:** Thompson Sampling is "optimistic exploration." When uncertain (wide Beta), you'll sample high values sometimes, giving each variant a chance. As you gain confidence (tight Beta), you stop wasting traffic on losers.

## Code Implementation

Complete Thompson Sampling conversion optimizer:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import numpy as np

class ConversionBandit:
    def __init__(self, n_variants):
        self.n = n_variants
        self.alpha = np.ones(n_variants)  # Prior
        self.beta = np.ones(n_variants)

    def select_variant(self):
        """Thompson Sampling: sample from posteriors"""
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)

    def update(self, variant, converted):
        """Update posterior after observing result"""
        if converted:
            self.alpha[variant] += 1
        else:
            self.beta[variant] += 1

    def get_means(self):
        """Current conversion rate estimates"""
        return self.alpha / (self.alpha + self.beta)

# Simulate conversion testing
true_rates = [0.03, 0.06, 0.04, 0.05]  # 4 variants
bandit = ConversionBandit(len(true_rates))
conversions = np.zeros(len(true_rates))

for visitor in range(10000):
    variant = bandit.select_variant()
    converted = np.random.rand() < true_rates[variant]
    bandit.update(variant, converted)
    conversions[variant] += converted

# Results
for i, rate in enumerate(true_rates):
    pulls = bandit.alpha[i] + bandit.beta[i] - 2
    conv_rate = conversions[i] / pulls if pulls > 0 else 0
    print(f"Variant {i}: {pulls:.0f} visitors, "
          f"{conv_rate:.1%} conv (true: {rate:.1%})")
```

</div>
</div>

**Output example:**
```
Variant 0:  523 visitors, 2.9% conv (true: 3.0%)
Variant 1: 8891 visitors, 6.1% conv (true: 6.0%)
Variant 2:  312 visitors, 4.2% conv (true: 4.0%)
Variant 3:  274 visitors, 4.7% conv (true: 5.0%)
```

Notice: 89% of traffic went to the best variant (B) by the end.

## Commodity Trading Applications

### Application 1: Research Report Formats
**Problem:** You publish weekly crude oil reports. Which format drives the most trader action?

**Arms:**
- Variant A: PDF with charts (15 pages)
- Variant B: Email with key takeaways (5 bullet points + link)
- Variant C: Interactive dashboard
- Variant D: Video walkthrough (10 min)

**Reward:** 1 if trader makes a trade within 24 hours of receiving report, 0 otherwise

**Why bandits beat A/B:** You can't afford to send the wrong format for 4 weeks. Every day a trader gets the wrong format, they might miss a move or lose trust.

### Application 2: Alert Threshold Optimization
**Problem:** You send "crude oil volatility spike" alerts. What threshold minimizes false positives while catching real moves?

**Arms:**
- Threshold 1: Alert when ATR > 2σ (frequent, noisy)
- Threshold 2: Alert when ATR > 3σ (balanced)
- Threshold 3: Alert when ATR > 4σ (rare, high precision)

**Reward:** 1 if trader rates alert as "actionable," 0 if "noise"

**Why bandits:** Optimal threshold might change with market regime. Bandits adapt as volatility patterns shift.

### Application 3: Email Subject Line Testing
**Problem:** Weekly commodities newsletter. Which subject lines get opens AND reads (not just opens)?

**Arms:**
- "This Week in Crude Oil" (descriptive)
- "Why WTI Just Hit $85" (specific)
- "Traders: Don't Miss This Move" (urgency)
- "My Take on the EIA Report" (personal)

**Reward:** (Open rate × Read ratio) — composite metric to avoid clickbait

**Why bandits:** Subject line effectiveness depends on what you recently sent (novelty effect). Bandits adapt to saturation.

## Common Pitfalls

### Pitfall 1: Stopping Too Early
**The trap:** "After 200 visitors, variant B has 8% vs A's 3%. Ship it!"

**Why it fails:** Small sample sizes have huge variance. B might have gotten lucky. Thompson Sampling naturally handles this by keeping wide posteriors → more exploration.

**The fix:** Let the bandit run until posteriors are tight (credible intervals don't overlap). Or use a minimum sample size rule: "No decision until each variant has ≥100 visitors."

### Pitfall 2: Ignoring Multiple Comparisons
**The trap:** Testing 20 variants, one shows 10% conversion by chance.

**Why it fails:** With many variants, false discoveries are likely. Thompson Sampling helps (Bayesian shrinkage) but doesn't eliminate the problem.

**The fix:**
- Start with ≤5 variants (easier to learn)
- Use informative priors: Beta(2, 20) if you expect ~10% conversion
- Monitor credible intervals, not just point estimates

### Pitfall 3: Optimizing for the Wrong Conversion
**The trap:** E-commerce site optimizes for "add to cart" (easy) not "completed purchase" (hard).

**Why it fails:** You've trained the bandit to drive cart adds, which might not correlate with revenue. Result: misleading design that hurts bottom line.

**The fix:**
- Use **final conversion** (purchase, trade, signup) as reward
- If delayed, use **proxy metrics with validation**: "add to cart" is fine IF it correlates 0.8+ with purchase
- For commodity reports: optimize for "trader took action" not "opened email"

## Connections

<div class="callout-danger">

**Danger:** Never deploy a bandit system without a kill switch and maximum allocation limits. An unconstrained bandit can allocate 100% of traffic/capital to a single arm, which creates catastrophic risk if the reward signal is noisy or delayed.



**Builds on:**
- **Module 2 (Bayesian Bandits):** Thompson Sampling with Beta-Bernoulli is the workhorse here
- **Module 1 (UCB):** UCB1 also works for conversions, but Thompson Sampling is simpler and better for binary rewards

**Leads to:**
- **Module 7 (Production Systems):** How to ship this as a website optimization service
- **Module 6 (Non-Stationary Bandits):** What if conversion rates drift over time? (Discounted Thompson Sampling)

**Connects to:**
- **Creator Bandit Playbook (Guide 01):** Same framework, different domain — conversions instead of content engagement
- **Arm Management (Guide 03):** How to add new landing pages without disrupting the bandit

## Practice Problems

### Problem 1: Calculate Opportunity Cost (Quantitative)
You're testing 2 landing pages with true conversion rates 4% and 7%. You get 1,000 visitors per week.

**A/B test (4 weeks):** 2,000 visitors to each variant
**Thompson Sampling:** Approximately 500 to worse, 3,500 to better (empirical average)

**Question:** How many conversions did Thompson Sampling save?

**Solution:**
```
A/B test conversions:
  Variant A: 2,000 × 0.04 = 80
  Variant B: 2,000 × 0.07 = 140
  Total: 220

Thompson Sampling conversions:
  Variant A: 500 × 0.04 = 20
  Variant B: 3,500 × 0.07 = 245
  Total: 265

Saved: 265 - 220 = 45 conversions (20% lift)
```

### Problem 2: Design a Multi-Step Funnel Bandit (Implementation)
Your signup funnel has 3 steps:
1. Landing page → Email signup (4 variants)
2. Email → Confirm email (3 subject lines)
3. Confirmed → First purchase (2 onboarding flows)

**Question:** Should you run one bandit optimizing end-to-end conversion, or 3 separate bandits (one per step)?

**Hint:** Separate bandits learn faster (more immediate feedback) and localize issues. End-to-end is cleaner but slower.

### Problem 3: Detect Bad Metrics (Conceptual)
A SaaS company runs a conversion bandit optimizing for "started free trial." After 6 months, trial starts are up 40% but revenue is flat.

**Question:** What went wrong? What should they optimize for instead?

**Answer:** They optimized for a vanity metric (trial starts) that doesn't correlate with revenue. Should optimize for "trial → paid conversion" or use a composite reward: (trial start) × (predicted LTV based on engagement).


---

## Cross-References

<a class="link-card" href="./01_creator_bandit_playbook.md">
  <div class="link-card-title">01 Creator Bandit Playbook</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_creator_bandit_playbook.md">
  <div class="link-card-title">01 Creator Bandit Playbook — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./03_arm_management.md">
  <div class="link-card-title">03 Arm Management</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_arm_management.md">
  <div class="link-card-title">03 Arm Management — Companion Slides</div>
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

