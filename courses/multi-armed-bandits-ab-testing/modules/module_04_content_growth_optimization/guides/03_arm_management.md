# Arm Management: Retirement, Introduction, and Evolution

> **Reading time:** ~20 min | **Module:** 04 — Content Growth Optimization | **Prerequisites:** Module 3


## In Brief


<div class="callout-key">

**Key Concept Summary:** Real-world bandit systems aren't static. New options emerge (formats, products, strategies), old options stop working (market regime shifts, audience fatigue), and you can't test infinite arms simu...

</div>

Real-world bandit systems aren't static. New options emerge (formats, products, strategies), old options stop working (market regime shifts, audience fatigue), and you can't test infinite arms simultaneously. **Arm management** is the discipline of deciding when to retire underperforming arms, how to introduce new ones without disrupting learning, and how to keep your bandit system evolving as the world changes.

**Why it matters:** A bandit with 6 arms where 3 are dead weight learns slower and wastes traffic. Periodic pruning + strategic introduction creates an "evolutionary" system: keep what works, test what might work, ruthlessly cut what doesn't.

## Key Insight

The best bandit systems are **gardens, not monuments**:
- **Monument mindset:** "I'll design the perfect 10 arms, test them, and use the winners forever."
- **Garden mindset:** "I'll start with 6 arms, prune the worst quarterly, plant new ones, and let the system evolve."

<div class="callout-insight">

**Insight:** The core insight of bandit algorithms is that learning and earning are not separate phases. Every observation contributes to both understanding which option is best and generating value from the best option.

</div>


This maps directly to how successful companies operate:
- **Amazon:** Constantly tests new product features, retires what doesn't work, scales what does
- **Netflix:** A/B tests are never "finished" — the winner becomes the new baseline for the next test
- **Trading firms:** Strategies that worked in 2020 might fail in 2025 — continuous adaptation or die

The key techniques:
1. **Arm retirement:** Drop arms that are provably worse
2. **Arm introduction:** Add new arms with "onboarding" exploration
3. **Windowed estimates:** Weight recent data more (handles non-stationarity)
4. **Minimum pull constraints:** Don't retire until you've given it a fair shot

## Visual Explanation

<div class="callout-warning">

**Warning:** Bandit algorithms assume the reward distributions are stationary (or slowly changing). In commodity markets, regime shifts can make a historically optimal arm suddenly suboptimal. Always implement change detection alongside your bandit.

</div>


```
EVOLUTIONARY BANDIT SYSTEM (12-month lifecycle)

Month 0: INITIAL ARMS
┌─────────────────────────────────────────────┐
│ Arms: [A, B, C, D, E, F]                    │
│ Status: All active, even exploration        │
└─────────────────────────────────────────────┘
              ↓
Month 3: FIRST PRUNING
┌─────────────────────────────────────────────┐
│ Retire: F (worst, pulled 50x, μ̂=0.02)       │
│ Introduce: G (new topic, unknown μ)         │
│ Reset: Give G even exploration for 2 weeks  │
│ Arms: [A, B, C, D, E, G]                    │
└─────────────────────────────────────────────┘
              ↓
Month 6: SECOND PRUNING
┌─────────────────────────────────────────────┐
│ Retire: E (drifted down, recent μ̂=0.03)     │
│ Introduce: H (format variant)               │
│ Arms: [A, B, C, D, G, H]                    │
└─────────────────────────────────────────────┘
              ↓
Month 12: EVOLVED PORTFOLIO
┌─────────────────────────────────────────────┐
│ Original survivors: [A, C, D] (50%)         │
│ New additions: [G, H, I] (50%)              │
│ Net result: System adapted to audience      │
└─────────────────────────────────────────────┘

RETIREMENT DECISION TREE:
                Has arm been pulled ≥N times?
                /                            \
              Yes                            No
              /                               \
    Is it worst performer?              Keep exploring
           /        \
         Yes        No
         /           \
  Is μ̂ < threshold? Keep it
       /      \
     Yes      No
     /         \
  RETIRE    Keep it (might recover)
```

## Formal Definition

**Arm Retirement Criteria:**

Given arm k at time t, retire if ALL conditions met:
```
1. Minimum pulls: n_k(t) ≥ N_min (e.g., 50)
   → Don't retire before fair evaluation

2. Worst performer: μ̂_k ≤ min_j≠k μ̂_j
   → Only retire if provably worse

3. Statistical significance: UCB_k < LCB_best
   → Confidence intervals don't overlap
   where:
     UCB_k = μ̂_k + sqrt(2 log(t) / n_k)
     LCB_best = μ̂_best - sqrt(2 log(t) / n_best)

Optional:
4. Absolute threshold: μ̂_k < μ_min (e.g., 0.05)
   → Retire if objectively bad, even if others are worse
```

**Arm Introduction Protocol:**

When introducing new arm k_new:
```
1. Onboarding period (weeks 1-2):
   Pull k_new with probability ≥ 1/K (forced exploration)

2. Optimistic initialization (optional):
   Set μ̂_{k_new} = μ_best (give it a chance)
   OR use informative prior if Bayesian

3. Monitoring period (weeks 3-4):
   Let bandit choose k_new naturally, no forcing

4. Evaluation (week 4):
   If n_{k_new} ≥ N_min, add to regular rotation
```

**Non-Stationary Handling (Windowed Estimates):**

Instead of cumulative average:
```
μ̂_k = (1/n_k) Σ_{i=1}^{n_k} r_i  ← Stationary

Use windowed average:
μ̂_k = (1/W) Σ_{i=max(1, n_k-W+1)}^{n_k} r_i  ← Non-stationary
where W = window size (e.g., 100 pulls)
```

Or exponentially weighted:
```
μ̂_k ← α·μ̂_k + (1-α)·r_new
where α ∈ [0.9, 0.99] (higher = more memory)
```

## Intuitive Explanation

Imagine you run a commodity trading newsletter with 6 content types (arms). After 3 months, you notice:
- **Arm A (Market Analysis × Essay):** 52% read ratio, 120 posts — clear winner
- **Arm B (Trading Psychology × Thread):** 38% read ratio, 80 posts — solid performer
- **Arm C (Risk Management × Video):** 45% read ratio, 90 posts — working well
- **Arm D (Market Analysis × Thread):** 28% read ratio, 75 posts — underperforming
- **Arm E (Trading Psychology × Essay):** 22% read ratio, 60 posts — worst performer
- **Arm F (Risk Management × Thread):** 35% read ratio, 70 posts — mediocre

**Retirement decision:**
- Arm E is clearly worst (22%, well-sampled with 60 posts)
- Confidence intervals: [19%, 25%] vs next-worst D at [25%, 31%] — no overlap
- **Decision:** Retire E, it's not working

**Introduction decision:**
- New idea: "Market Analysis × Podcast" (untested format)
- **Protocol:**
  1. Week 1-2: Publish 2 podcasts/week (forced exploration)
  2. Week 3-4: Let bandit decide naturally
  3. Week 4: If read ratio > 30% and ≥8 episodes, keep it
  4. If read ratio < 20%, it's not resonating — retire early

**Result after 12 months:**
- Started with arms [A, B, C, D, E, F]
- Retired [E, D, F] (underperformers)
- Added [G=Podcast, H=Newsletter, I=Dashboard]
- Current portfolio [A, C, G, H, I, B] — 50% new, 50% original survivors
- Net outcome: Strategy evolved with audience preferences

## Code Implementation

Complete arm management system with retirement and introduction:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import numpy as np

class EvolutionaryBandit:
    def __init__(self, initial_arms, min_pulls=50):
        self.arms = initial_arms
        self.n_pulls = {arm: 0 for arm in initial_arms}
        self.rewards = {arm: [] for arm in initial_arms}
        self.min_pulls = min_pulls
        self.retired = []

    def get_mean(self, arm, window=None):
        """Get average reward (optionally windowed)"""
        if not self.rewards[arm]:
            return 0.0
        data = self.rewards[arm]
        if window:
            data = data[-window:]
        return np.mean(data)

    def retire_worst(self, threshold=None):
        """Retire worst arm if criteria met"""
        # Only consider arms with enough data
        eligible = [a for a in self.arms
                    if self.n_pulls[a] >= self.min_pulls]
        if len(eligible) <= 2:  # Keep minimum 2 arms
            return None

        # Find worst by recent performance
        means = {a: self.get_mean(a, window=100)
                 for a in eligible}
        worst = min(eligible, key=lambda a: means[a])

        # Check if significantly worse
        if threshold and means[worst] < threshold:
            self.arms.remove(worst)
            self.retired.append(worst)
            return worst

        return None

    def add_arm(self, new_arm, onboarding_pulls=10):
        """Introduce new arm with forced exploration"""
        self.arms.append(new_arm)
        self.n_pulls[new_arm] = 0
        self.rewards[new_arm] = []
        return onboarding_pulls  # Force this many pulls

# Example usage
bandit = EvolutionaryBandit([
    "Essay", "Thread", "Video",
    "Podcast", "Newsletter", "Dashboard"
], min_pulls=50)

# Simulate 52 weeks
for week in range(52):
    # Quarterly retirement
    if week > 0 and week % 12 == 0:
        retired = bandit.retire_worst(threshold=0.3)
        if retired:
            print(f"Week {week}: Retired '{retired}'")
            # Add new arm
            new = f"Format_{week}"
            bandit.add_arm(new, onboarding_pulls=10)
            print(f"Week {week}: Added '{new}'")
```

</div>

## Commodity Trading Applications

### Application 1: Strategy Retirement in Portfolio Allocation
**Problem:** You're running 8 commodity trading strategies. Some worked in 2023 but fail in 2025.

**Retirement criteria:**
- Sharpe ratio < 0.5 over last 60 days (windowed estimate)
- Pulled at least 100 times (enough data)
- Worst performer among active strategies

**Introduction protocol:**
- New strategy gets 10% of capital for first month (forced exploration)
- Evaluated after 30 trades — if Sharpe > 1.0, keep it; else retire

**Why it matters:** Market regimes change. A backwardation strategy that crushed in 2020 (contango market) might bleed in 2024 (backwardation). Retirement prevents holding dead strategies.

### Application 2: Research Format Evolution
**Problem:** Traders consume your research in different formats. Preferences shift over time.

**Retirement example:**
- "15-page PDF report" was popular in 2020 (desktop traders)
- By 2024, read ratio dropped from 45% to 18% (mobile traders)
- Windowed estimate (last 50 reports) shows consistent decline
- **Decision:** Retire PDF, replace with "5-minute video summary"

**Introduction protocol:**
- First 10 videos: send to 20% of audience (forced exploration)
- Track: watch-through rate, trading activity after watching
- If watch-through > 60% and drives trades, scale to 50% of audience

### Application 3: Alert Channel Optimization
**Problem:** You send crude oil alerts via email, SMS, and push notifications. Which channels work?

**Retirement criteria:**
- Channel has <10% click-through rate on alerts (threshold)
- Channel cost > $0.05 per alert (economics)
- At least 100 alerts sent (minimum pulls)

**Example:**
- Email: 35% CTR, $0.01/alert → Keep
- SMS: 12% CTR, $0.08/alert → Retire (expensive, mediocre)
- Push: 45% CTR, $0.001/alert → Best performer
- **Decision:** Retire SMS, test new channel (Slack integration)

## Common Pitfalls

### Pitfall 1: Retiring Too Aggressively
**The trap:** "Arm C had a bad week, so I'll drop it."

**Why it fails:** Short-term variance doesn't mean the arm is bad. You need sufficient pulls and statistical confidence.

**The fix:**
- Minimum pull requirement (≥50 for content, ≥100 for trading)
- Check confidence intervals, not just point estimates
- Require "worst performer" for multiple weeks, not just one

### Pitfall 2: No Onboarding for New Arms
**The trap:** Add new arm, let bandit choose immediately, new arm never gets selected (existing arms have tight confidence).

**Why it fails:** New arms start with high uncertainty. If you're using UCB or Thompson Sampling, they might get unlucky early and never recover.

**The fix:**
- **Forced exploration:** New arm gets 1/K traffic for first N pulls
- **Optimistic initialization:** Set initial estimate high (give it a chance)
- **Onboarding period:** 2 weeks of guaranteed even exposure

### Pitfall 3: Ignoring Non-Stationarity
**The trap:** Using cumulative average (all-time performance) when recent performance has shifted.

**Why it fails:** An arm that was great for 6 months but terrible for the last month still has a high cumulative average. You'll keep using it even though it's broken.

**The fix:**
- **Windowed estimates:** Use last W=100 pulls, not all-time
- **Exponential weighting:** Recent rewards weighted higher (α=0.95)
- **Drift detection:** Monitor variance of windowed estimates — sudden change = regime shift

### Pitfall 4: Retiring Without Replacement
**The trap:** Over 12 months, you retire 4 arms and don't add new ones. Now you're down to 2 arms.

**Why it fails:** Fewer arms = less exploration capacity. You might miss the next breakthrough format.

**The fix:**
- **1-for-1 replacement:** Every retirement = one new introduction
- **Ideation pipeline:** Maintain a queue of "next to test"
- **Experimentation budget:** Always reserve 10-20% for new arms

## Connections

<div class="callout-danger">

**Danger:** Never deploy a bandit system without a kill switch and maximum allocation limits. An unconstrained bandit can allocate 100% of traffic/capital to a single arm, which creates catastrophic risk if the reward signal is noisy or delayed.

</div>


**Builds on:**
- **Module 1 (UCB):** UCB confidence intervals are perfect for retirement decisions (overlap check)
- **Module 2 (Thompson Sampling):** Bayesian credible intervals for statistical significance
- **Module 6 (Non-Stationary Bandits):** Discounted Thompson Sampling, sliding window estimates

**Leads to:**
- **Module 5 (Commodity Trading):** Strategy retirement is critical for portfolio bandits
- **Module 7 (Production Systems):** How to automate arm management in production

**Connects to:**
- **Creator Playbook (Guide 01):** Quarterly arm retirement is the core mechanism
- **Conversion Optimization (Guide 02):** Same retirement logic for landing page variants

## Practice Problems

### Problem 1: Implement Retirement Logic (Code)
<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def should_retire(arm_stats, arm_id, min_pulls=50):
    """
    Return True if arm should be retired.

    arm_stats = {
        arm_id: {
            'pulls': int,
            'mean': float,
            'ucb': float,
            'lcb': float
        }
    }
    """
    # Your implementation:
    # 1. Check minimum pulls
    # 2. Check if worst performer
    # 3. Check if UCB < best LCB
    pass

# Test case
stats = {
    'A': {'pulls': 100, 'mean': 0.50, 'ucb': 0.55, 'lcb': 0.45},
    'B': {'pulls': 80, 'mean': 0.30, 'ucb': 0.38, 'lcb': 0.22},
    'C': {'pulls': 120, 'mean': 0.48, 'ucb': 0.52, 'lcb': 0.44}
}
assert should_retire(stats, 'B', min_pulls=50) == True  # B is worst and significant
assert should_retire(stats, 'C', min_pulls=50) == False  # C is competitive
```

</div>

### Problem 2: Design an Introduction Protocol (Conceptual)
You're adding a new trading strategy (arm) to a portfolio bandit with 6 existing strategies.

**Design:**
1. How much capital does the new strategy get for the first month?
2. What performance threshold must it hit to stay active?
3. How do you avoid disrupting the existing bandit's learning?

**Hint:** Forced exploration (10-20% allocation) + performance gate (Sharpe > 1.0) + 30-day evaluation period.

### Problem 3: Detect Non-Stationarity (Analysis)
An arm has these monthly average rewards:
```
Month:  1    2    3    4    5    6    7    8    9    10   11   12
Reward: 0.45 0.48 0.47 0.46 0.44 0.42 0.38 0.35 0.32 0.30 0.28 0.25
```

**Questions:**
1. What's the cumulative average after 12 months?
2. What's the windowed average (last 3 months)?
3. Should this arm be retired? Why or why not?

**Solution:**
1. Cumulative: (0.45+...+0.25)/12 = 0.37 (looks okay)
2. Windowed: (0.30+0.28+0.25)/3 = 0.28 (trending down!)
3. **Yes, retire:** Clear downward trend suggests the arm stopped working. Cumulative average is misleading.


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

<a class="link-card" href="./cheatsheet.md">
  <div class="link-card-title">Cheatsheet</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./cheatsheet.md">
  <div class="link-card-title">Cheatsheet — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

