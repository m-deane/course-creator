# Module 4 Cheatsheet: Bandits for Content & Growth

> **Reading time:** ~20 min | **Module:** 04 — Content Growth Optimization | **Prerequisites:** Module 3


## Creator Bandit Playbook (6-Step Framework)


<div class="callout-key">

**Key Concept Summary:** This guide covers the core concepts of module 4 cheatsheet: bandits for content & growth, with worked examples and practical implementation guidance.

</div>

```
STEP 1: Define Arms (repeatable content types)
  ✓ Topic × Format combinations (e.g., "Market Analysis × Video")
  ✗ One-off posts (can't learn from single events)
  Example: 3 topics × 2 formats = 6 arms

STEP 2: Choose Reward Metric (quality, not vanity)
  ✓ Read ratio (completions / opens)
  ✓ Engaged time (minutes spent)
  ✓ Downstream actions (trades made, signups)
  ✗ Views (optimizes for clickbait)
  ✗ Likes (optimizes for controversy)

STEP 3: Exploration Phase (Weeks 1-3)
  Publish evenly: ~2 posts per arm
  Collect baseline reward data
  Build initial estimates μ̂_k

STEP 4: Exploitation Phase (Weeks 4-12)
  Top 2 arms: 60% of publishing slots
  Remaining arms: 40% (keep exploring)
  Reserve 20% for deliberate experiments

STEP 5: Arm Retirement (Every 12 Weeks)
  Drop worst performer (if n_k ≥ 10 and μ̂_k significantly lower)
  Introduce new topic×format combo
  Reset: 2 weeks even exploration on new arm

STEP 6: Continuous Adaptation
  Track windowed estimates (last 50 posts)
  Detect performance drift
  Prune ruthlessly, test curiously
```

## Conversion Optimization with Thompson Sampling

```
SETUP:
  K variants (landing pages, headlines, CTAs, prices)
  Binary reward: conversion (1) or not (0)
  Prior: Beta(1, 1) for each variant

```

<div class="callout-insight">

**Insight:** The core insight of bandit algorithms is that learning and earning are not separate phases. Every observation contributes to both understanding which option is best and generating value from the best option.

</div>

```
ALGORITHM:
  For each visitor t:
    1. Sample: θ̃_k ~ Beta(α_k, β_k) for all variants
    2. Select: Show variant with max(θ̃_k)
    3. Observe: Conversion r_t ∈ {0, 1}
    4. Update:
       if r_t = 1: α_k ← α_k + 1  (success)
       if r_t = 0: β_k ← β_k + 1  (failure)

WHEN TO USE:
  ✓ Testing landing pages, email subject lines, pricing
  ✓ Binary outcomes (convert or don't)
  ✓ High opportunity cost (can't waste traffic)
  ✗ Continuous outcomes (use Gaussian Thompson Sampling)

COMMODITY APPLICATION:
  Variants = Report formats (PDF, email, video, dashboard)
  Conversion = Trader makes a trade within 24h
  Goal = Maximize actionable research consumption
```

## Arm Management Decision Tree

<div class="callout-warning">

**Warning:** Bandit algorithms assume the reward distributions are stationary (or slowly changing). In commodity markets, regime shifts can make a historically optimal arm suddenly suboptimal. Always implement change detection alongside your bandit.

</div>


```
                    New arm to introduce?
                    /                    \
                  Yes                    No
                  /                       \
        Give 1/K traffic               Review existing arms
        for N pulls                           ↓
             ↓                      Has arm been pulled ≥N_min?
        Evaluate after N                   /            \
             ↓                           Yes             No
        μ̂ > threshold?                   ↓               ↓
        /           \              Is worst performer?  Keep exploring
      Yes           No                  /        \
      ↓             ↓                 Yes        No
    Keep it      Retire it             ↓          ↓
                                   UCB_k < LCB_best?  Keep it
                                      /      \
                                    Yes      No
                                    ↓         ↓
                                 RETIRE    Keep it

RETIREMENT CRITERIA (all must be true):
  1. n_k ≥ N_min (e.g., 50 pulls) — fair evaluation
  2. Worst performer among active arms
  3. Confidence intervals don't overlap with best
  4. [Optional] μ̂_k < absolute threshold (e.g., 0.05)

INTRODUCTION PROTOCOL:
  1. Onboarding (2 weeks): Forced 1/K traffic
  2. Monitoring (2 weeks): Natural selection by bandit
  3. Evaluation (week 4): Keep if μ̂ > threshold
```

## Reward Design Principles

```
GOOD REWARDS (align with business goals):
  Content:     Read ratio, time spent, shares
  Conversion:  Completed purchase, signup, subscription
  Trading:     Profitable trades, risk-adjusted returns
  Engagement:  Comments, saves, actionable responses

BAD REWARDS (vanity metrics, misaligned):
  Views:       Optimizes for clickbait, not quality
  Opens:       Optimizes for misleading subject lines
  Likes:       Optimizes for controversy, not value
  Traffic:     Optimizes for SEO spam, not readers

SAFETY SIGNALS (prevent gaming):
  Read ratio < 20%? → Clickbait penalty
  Bounce rate > 80%? → Misleading content
  Unsubscribe spike? → Annoying format
  Negative feedback? → Hard constraint (never show again)

COMPOSITE REWARDS (balance multiple goals):
  Revenue optimization: (conversion rate) × (average order value)
  Content quality:      (read ratio) × (1 + shares/100)
  Trading value:        (win rate) × (avg profit per trade)
```

## Common Patterns by Use Case

```
┌────────────────────────────────────────────────────────────┐
│ USE CASE              │ ARMS           │ REWARD            │
├────────────────────────────────────────────────────────────┤
│ Content Strategy      │ Topic × Format │ Read ratio        │
│ Landing Page Testing  │ Page variants  │ Conversion rate   │
│ Email Optimization    │ Subject lines  │ Open × read ratio │
│ Pricing Experiments   │ Price points   │ Revenue per visit │
│ Onboarding Flow       │ Step variants  │ Completion rate   │
│ Trading Alerts        │ Thresholds     │ Actionable trades │
│ Report Formats        │ Delivery modes │ Trader engagement │
│ Ad Creative Testing   │ Ad variants    │ CTR × conversion  │
└────────────────────────────────────────────────────────────┘
```

## Quick Reference: Algorithm Selection

```
PROBLEM                          → ALGORITHM
────────────────────────────────────────────────────────────
Binary conversions              → Thompson Sampling (Beta)
Continuous rewards              → Thompson Sampling (Gaussian)
Need interpretability           → UCB1 with confidence intervals
Many variants (K > 20)          → Top-K Thompson Sampling
Delayed feedback                → Sliding window estimates
Non-stationary environment      → Discounted Thompson Sampling
Strict budget constraints       → Epsilon-greedy (fixed exploration)
Multi-step funnel               → Independent bandit per step
```

## Implementation Checklist

```
☐ Define arms (repeatable, mutually exclusive)
☐ Choose reward metric (quality, not vanity)
☐ Add safety signals (prevent gaming)
☐ Set exploration budget (10-20%)
☐ Define minimum pulls before retirement (≥50)
☐ Choose retirement cadence (monthly/quarterly)
☐ Plan introduction protocol (onboarding period)
☐ Implement windowed estimates (handle drift)
☐ Monitor performance metrics (regret, conversion lift)
☐ Document arm retirement decisions (audit trail)
```

## Red Flags (When NOT to Use Bandits)

```
❌ "I need statistical significance" → Use traditional A/B test
   (Bandits optimize for regret, not p-values)

❌ "I can only test for 1 week" → Too short for bandits
   (Need enough time to adapt allocation)

❌ "I have 100 variants to test" → Use successive elimination
   (Too many arms = slow learning)

❌ "Reward arrives after 6 months" → Use proxy metrics
   (Bandits need timely feedback)

❌ "One bad decision = catastrophic" → Use controlled rollout
   (Bandits will try all arms initially)
```

## Quick Wins (Start Here)

```
EASIEST: Email subject line testing
  - 3-4 variants (easy to create)
  - Binary reward (open or not)
  - Immediate feedback
  - Low risk

MEDIUM: Landing page conversion
  - 2-3 page designs
  - Thompson Sampling (Beta-Bernoulli)
  - High business impact
  - 2-week test window

ADVANCED: Content strategy optimization
  - 6 topic×format arms
  - Custom reward (read ratio)
  - Quarterly arm retirement
  - 12-month evolution
```

## Key Equations

<div class="callout-danger">

**Danger:** Never deploy a bandit system without a kill switch and maximum allocation limits. An unconstrained bandit can allocate 100% of traffic/capital to a single arm, which creates catastrophic risk if the reward signal is noisy or delayed.

</div>


```
Thompson Sampling Update (Beta-Bernoulli):
  α_k ← α_k + r_t     (if arm k selected, conversion occurred)
  β_k ← β_k + (1-r_t) (if arm k selected, no conversion)

Windowed Average (Non-Stationary):
  μ̂_k = (1/W) Σ_{i=n_k-W+1}^{n_k} r_i
  where W = window size (e.g., 100)

Exponentially Weighted Average:
  μ̂_k ← α·μ̂_k + (1-α)·r_new
  where α ∈ [0.9, 0.99]

Retirement Criterion (UCB/LCB):
  Retire k if: UCB_k < LCB_best
  where:
    UCB_k = μ̂_k + sqrt(2 log(t) / n_k)
    LCB_best = μ̂_best - sqrt(2 log(t) / n_best)
```

## Resources for Deep Dive

```
Papers:
  - "A/B Testing with Thompson Sampling" (Scott, 2015)
  - "Online Experimentation at Microsoft" (Kohavi et al., 2013)

Blog Posts:
  - Optimizely Engineering: Bandit Testing Guide
  - VWO: When to Use Multi-Armed Bandits
  - Booking.com: Continuous Experimentation

Tools:
  - Epsilon (open-source bandit framework)
  - Ax Platform (Meta's adaptive experimentation)
  - Google Optimize (deprecated, but good case studies)
```


---

## Conceptual Practice Questions

**Practice Question 1:** What is the primary tradeoff this approach makes compared to simpler alternatives?

**Practice Question 2:** Under what conditions would this approach fail or underperform?


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

<a class="link-card" href="./02_conversion_optimization.md">
  <div class="link-card-title">02 Conversion Optimization</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_conversion_optimization.md">
  <div class="link-card-title">02 Conversion Optimization — Companion Slides</div>
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

