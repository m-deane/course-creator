# Creator Bandit Playbook

> **Reading time:** ~20 min | **Module:** 04 — Content Growth Optimization | **Prerequisites:** Module 3


## In Brief


<div class="callout-key">

**Key Concept Summary:** The **Creator Bandit Playbook** treats content publishing as a multi-armed bandit problem where arms are topic×format combinations you can repeat weekly, and rewards are meaningful engagement metri...

</div>

The **Creator Bandit Playbook** treats content publishing as a multi-armed bandit problem where arms are topic×format combinations you can repeat weekly, and rewards are meaningful engagement metrics (read ratio, not just views). Instead of guessing what your audience wants or running endless A/B tests, you publish evenly for a few weeks, tilt toward top performers, keep exploring, and ruthlessly prune what doesn't work.

**Why it matters:** Most creators either follow their gut (emotional, inconsistent) or their vanity metrics (views optimize for clickbait). Bandits give you the calm, systematic version: learn what works, double down on it, but stay curious about what might work better.

> 💡 **Key Insight:** **Content creation is inherently sequential and non-stationary.** Your audience evolves. Trends shift. New formats emerge. What worked last quarter might not work next quarter. The bandit framework gives you a systematic way to adapt:

1. **Arms = repeatable content types** (topic×format, not individual posts)
2. **Rewards = engagement quality** (read ratio, shares, actionable response — NOT views)
3. **Exploration budget** = deliberate "experimentation slots" so you don't get stuck
4. **Arm retirement** = monthly pruning to make room for new ideas

This is the antidote to "I went viral once, so I'll chase that high forever" and "I haven't changed my content strategy in 3 years."

## Visual Explanation

```
CREATOR BANDIT LIFECYCLE (52-week cycle)

<div class="callout-insight">

**Insight:** The core insight of bandit algorithms is that learning and earning are not separate phases. Every observation contributes to both understanding which option is best and generating value from the best option.

</div>


Weeks 1-3: EXPLORATION PHASE
┌──────────────────────────────────────────┐
│  6 Arms (3 topics × 2 formats)           │
│  Publish evenly: ~2 posts per arm        │
│  Collect reward data (read ratio)        │
└──────────────────────────────────────────┘
              ↓
Weeks 4-12: EXPLOITATION WITH EXPLORATION
┌──────────────────────────────────────────┐
│  Top 2 arms get 60% of publishing budget │
│  Remaining 4 arms get 20% (exploration)  │
│  Bottom 2 get 20% (keep monitoring)      │
└──────────────────────────────────────────┘
              ↓
Week 12: ARM RETIREMENT
┌──────────────────────────────────────────┐
│  Drop worst-performing arm               │
│  Introduce new topic×format combo        │
│  Reset: 2 weeks even exploration on new  │
└──────────────────────────────────────────┘
              ↓
         REPEAT QUARTERLY
```

**Key insight visualization:**
```
Emotional Creator:     🎢 (viral spike → chase → burnout → random pivot)
A/B Test Creator:      📊 (test 8 weeks → pick winner → stuck with it)
Bandit Creator:        📈 (tilt toward what works + 20% experiments + quarterly pruning)
```

## Formal Definition

<div class="callout-warning">

**Warning:** Bandit algorithms assume the reward distributions are stationary (or slowly changing). In commodity markets, regime shifts can make a historically optimal arm suddenly suboptimal. Always implement change detection alongside your bandit.

</div>


Let:
- **K = 6 arms** (3 topics × 2 formats, e.g., "Trading Psychology × Thread" or "Market Analysis × Video")
- **T = 52 weeks** (one year of content)
- **Reward r_t** = read ratio for post published at time t (completions / opens)

**Phase 1 (Weeks 1-3): Uniform Exploration**
```
Pull each arm evenly: n_k(3) ≈ 2-3 posts per arm
Estimate μ̂_k = (1/n_k) Σ r_t for each arm k
```

**Phase 2 (Weeks 4-12): Tilted Exploitation**
```
Rank arms by μ̂_k
Top 2 arms: 60% of publishing slots
Middle 2 arms: 20% (monitoring)
Bottom 2 arms: 20% (keep exploring, might improve)
```

**Phase 3 (Week 12, 24, 36): Arm Retirement**
```
Drop worst arm: k_worst = argmin μ̂_k
Introduce new arm: k_new with unknown μ
Reset exploration: publish k_new evenly for 2-3 weeks
```

**Safety Constraint:**
```
Reward must capture quality, not just volume:
✓ Read ratio (% who finish article)
✓ Engaged time (minutes spent)
✓ Actionable response (trades made, tools used)
✗ Views (trains clickbait)
✗ Likes (trains controversy)
```

## Intuitive Explanation

Imagine you run a commodity research newsletter. You can write about:
- **Topics:** Market analysis, trading psychology, risk management
- **Formats:** Long-form essays, tweet threads, video walkthroughs

That's 3×2 = 6 possible content types (arms). You have time to publish 5 posts per week.

**Emotional approach:** "This tweet thread went viral, so I'll only do threads now!"
- Problem: You're chasing a spike, not a strategy. Next thread flops.

**A/B test approach:** "I'll publish each type for 8 weeks, measure engagement, pick the winner."
- Problem: 8 weeks × 6 arms = 48 weeks. By the time you decide, your audience has changed.

**Bandit approach:**
1. **Weeks 1-3:** Publish all 6 types evenly (~2 each). Measure which get the best read ratio.
2. **Weeks 4-12:** Tilt toward top 2 (3 posts/week) but keep publishing the others (1 post/week each) to stay curious.
3. **Week 12:** Drop the worst-performing type, try a new one (e.g., "Market Analysis × Podcast").
4. **Repeat quarterly.**

After a year, you've learned:
- Which topics your audience truly cares about (not just clicks on)
- Which formats drive deep engagement
- That your strategy evolves as your audience grows

## Code Implementation

Complete simulation of the Creator Bandit Playbook over 52 weeks:


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np

# Define 6 arms: topic × format combinations
arms = [
    "Market Analysis × Essay",
    "Market Analysis × Thread",
    "Trading Psychology × Essay",
    "Trading Psychology × Video",
    "Risk Management × Essay",
    "Risk Management × Thread"
]

# True read ratios (unknown to creator)
true_means = [0.45, 0.38, 0.52, 0.41, 0.35, 0.48]

def publish_content(arm_idx):
    """Simulate publishing: return read ratio"""
    return np.clip(
        np.random.normal(true_means[arm_idx], 0.1),
        0, 1
    )

# Creator Bandit Playbook
n_arms = len(arms)
pulls = np.zeros(n_arms)
rewards = np.zeros(n_arms)
history = []

# Phase 1: Exploration (weeks 1-3, ~12 posts)
for week in range(3):
    for arm in range(n_arms):
        reward = publish_content(arm)
        pulls[arm] += 1
        rewards[arm] += reward
        history.append((week, arm, reward))

# Phase 2: Tilted exploitation (weeks 4-52)
for week in range(3, 52):
    # Rank arms by average reward
    avg_rewards = rewards / np.maximum(pulls, 1)
    ranking = np.argsort(avg_rewards)[::-1]

    # Top 2 get 60%, others get 40%
    week_arms = (
        [ranking[0]] * 3 +  # 60% top
        [ranking[1]] * 2 +  # 40% second
        [ranking[i] for i in range(2, n_arms)]  # Rest
    )
    np.random.shuffle(week_arms)

    for arm in week_arms[:5]:  # 5 posts/week
        reward = publish_content(arm)
        pulls[arm] += 1
        rewards[arm] += reward
        history.append((week, arm, reward))

    # Quarterly arm retirement
    if week in [12, 24, 36]:
        worst = ranking[-1]
        print(f"Week {week}: Retire '{arms[worst]}'")
        # Reset that arm (simulating new topic)
        pulls[worst] = 0
        rewards[worst] = 0

print("\nFinal Performance:")
for i, arm in enumerate(arms):
    if pulls[i] > 0:
        print(f"{arm}: {rewards[i]/pulls[i]:.2%} "
              f"({int(pulls[i])} posts)")
```

</div>
</div>

## Common Pitfalls

### Pitfall 1: Over-Learning from One Viral Spike
**The trap:** "This tweet thread got 50K views! I should only do threads now!"

**Why it fails:** Virality is high-variance noise. One spike doesn't mean threads are your best format — you might have just hit the algorithm lottery. Bandits smooth over variance by tracking **average** performance across many posts.

**The fix:** Use read ratio (quality) not views (quantity) as your reward. A thread with 50K views but 2% read ratio (1K actually read it) loses to an essay with 5K views and 40% read ratio (2K read it).

### Pitfall 2: No Exploration Budget (Pure Exploitation)
**The trap:** "My top 2 arms are working, so I'll only publish those."

**Why it fails:** Your audience evolves. New formats emerge (Reels, Shorts, newsletters). What worked last quarter might saturate. Without exploration, you'll miss the next thing.

**The fix:** Reserve 20% of your publishing for exploration. That's 1 post/week if you publish 5×/week. Feels wasteful until it finds your next breakout format.

### Pitfall 3: Optimizing for the Wrong Metric
**The trap:** "My clickbait headlines get way more opens, so the bandit picks those."

**Why it fails:** You've trained the bandit to optimize for opens, not quality. Result: you become a clickbait factory that alienates your real audience.

**The fix:**
- Use **read ratio** (completions / opens) as reward
- Add **safety signals**: "If read ratio < 20%, it's clickbait — penalize it"
- Track **downstream actions** (for traders: "Did they make a trade?" not "Did they open the email?")

### Pitfall 4: Arms That Can't Be Repeated
**The trap:** Treating individual posts as arms ("Should I write about XYZ news event?")

**Why it fails:** You can't repeat a specific news event. Bandits need **repeatable** arms to learn from.

**The fix:** Arms must be **content types you can publish weekly**:
- ✓ "Market Analysis × Video" (repeatable every week)
- ✗ "Analysis of Feb 2026 OPEC Meeting" (one-time event)

## Connections

<div class="callout-danger">

**Danger:** Never deploy a bandit system without a kill switch and maximum allocation limits. An unconstrained bandit can allocate 100% of traffic/capital to a single arm, which creates catastrophic risk if the reward signal is noisy or delayed.

</div>


**Builds on:**
- **Module 2 (Bayesian Bandits):** Thompson Sampling is perfect for content — each arm has a Beta posterior for read ratio
- **Module 1 (Core Algorithms):** Epsilon-greedy with ε=0.2 (20% exploration) is the simplest version of this playbook
- **Module 3 (Contextual Bandits):** Add context (day of week, audience segment) for personalized content strategies

**Leads to:**
- **Module 5 (Commodity Trading):** Same framework, different domain — arms are trading strategies, rewards are risk-adjusted returns
- **Module 7 (Production Systems):** How to ship this as a recommendation system for a media company or newsletter platform

**Connects to:**
- **Conversion optimization** (Guide 02): Same bandit logic, applied to website testing
- **Arm management** (Guide 03): The retirement/introduction system is what makes this evolutionary, not static

## Practice Problems

### Problem 1: Design Your Creator Bandit (Conceptual)
You run a commodity trading podcast. Design a 6-arm bandit:
- **Topics:** Pick 3 (e.g., market commentary, trader interviews, strategy deep-dives)
- **Formats:** Pick 2 (e.g., solo episode, guest interview)
- **Reward metric:** What metric captures "valuable to my audience"?
- **Exploration budget:** How many episodes per week go to experimentation?

**Self-check:** Did you pick a quality metric (listen-through rate) not a vanity metric (downloads)?

### Problem 2: Detect the Metric Trap (Analysis)
A food blogger uses a bandit to optimize content. Arms are recipe types. Reward is pageviews. After 6 months, the bandit heavily favors "20-Minute Desserts" and "Sheet Pan Dinners."

**Question:** What's the hidden problem? What metric should they use instead?

**Hint:** Pageviews optimize for clicks, not cooking. Better metric: "Did they leave a comment saying they made it?"

### Problem 3: Simulate Arm Retirement (Implementation)
Implement the retirement logic from the playbook:

<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def should_retire(arm_idx, avg_rewards, pulls, week):
    """Return True if arm should be retired this week"""
    # Rules:
    # 1. Only retire every 12 weeks
    # 2. Must have at least 10 pulls
    # 3. Must be worst performer
    pass  # Your implementation

# Test: After 12 weeks, arm 3 has 15 pulls and lowest avg
# should_retire(3, avg_rewards, pulls, 12) → True
```

</div>
</div>

**Self-check:** Did you verify minimum pulls before retiring? (Don't retire arms you haven't given a fair shot.)


---

## Cross-References

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

<a class="link-card" href="./cheatsheet.md">
  <div class="link-card-title">Cheatsheet</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./cheatsheet.md">
  <div class="link-card-title">Cheatsheet — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

