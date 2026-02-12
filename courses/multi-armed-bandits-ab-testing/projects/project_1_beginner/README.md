# Project 1: Content Strategy Optimizer (Beginner)

## What You'll Build

A bandit-powered content publishing system that learns which topic-format combinations drive the best engagement. Inspired by the Creator Bandit Playbook, you'll implement Thompson Sampling to optimize a 52-week content calendar, adaptively tilting toward high-performers while maintaining exploration and periodically retiring underperformers.

By the end, you'll have a working simulation showing how a data-driven content strategy outperforms both random publishing and pure exploitation.

## Learning Goals

1. Implement Thompson Sampling from scratch for Beta-Bernoulli bandits
2. Design repeatable arms (topic × format combinations) for a real-world problem
3. Define reward metrics that capture quality over quantity
4. Implement arm retirement to handle evolving content landscapes
5. Visualize bandit learning over time
6. Compare bandit performance to baseline strategies

## What You'll Learn About Commodity Trading

While this project uses content creation as the domain, the exact same framework applies to commodity portfolio allocation:
- Arms = trading strategies or commodity positions
- Rewards = risk-adjusted returns
- Exploration = capital allocated to testing new strategies
- Retirement = dropping failed strategies to make room for new ones

The principles you learn here transfer directly to the commodity projects.

## Requirements

- Python 3.11+
- numpy
- matplotlib
- scipy (for Beta distribution)

```bash
pip install numpy matplotlib scipy
```

## The Problem

You run a commodity trading newsletter and can publish 5 posts per week. You have 6 possible content types:

**Topics (3):**
1. Market Analysis
2. Trading Psychology
3. Risk Management

**Formats (2):**
1. Long-form Essay
2. Twitter Thread

That's 3 × 2 = 6 arms (content types).

**Unknown:** Which combinations drive the best engagement (measured by read ratio: % of subscribers who finish the article)?

**Goal:** Maximize cumulative engagement over 52 weeks by learning which content types work best, while:
- Exploring new combinations to avoid getting stuck
- Retiring poor performers quarterly
- Maintaining a data-driven publishing calendar

## Project Steps

### Step 1: Environment Setup (5 minutes)

The `BanditEnvironment` class simulates your audience's response to different content types. It's already implemented in `starter_code.py` — you don't need to modify it.

```python
env = BanditEnvironment(arms, true_read_ratios)
reward = env.publish(arm_idx)  # Simulate publishing and get read ratio
```

### Step 2: Implement Thompson Sampling (20 minutes)

Complete the `ThompsonSampler` class in `starter_code.py`:

**Your tasks:**
1. Implement `select_arm()` — sample from Beta posteriors and pick the best
2. Implement `update()` — update the posterior based on observed read ratio

**Hints:**
- Use `scipy.stats.beta.rvs()` for sampling
- Read ratio is between 0-1, so treat as Bernoulli by comparing to threshold OR use it directly
- Start with Beta(1,1) priors (uniform)

### Step 3: Run 52-Week Simulation (10 minutes)

The main simulation loop is implemented. Run it to see:
- Week-by-week arm selection
- Posterior evolution
- Cumulative engagement

**Phases:**
- Weeks 1-3: Exploration (publish all arms evenly)
- Weeks 4-52: Tilted exploitation (60% top performers, 40% exploration)
- Weeks 12, 24, 36: Arm retirement (drop worst, introduce new)

### Step 4: Add Arm Retirement (15 minutes)

Implement the retirement logic:
- Every 12 weeks, identify the worst-performing arm
- Reset its posterior to Beta(1,1) (simulating a new topic)
- Give it 2-3 weeks of even exploration

**Why this matters:** Content trends evolve. What worked in Q1 may not work in Q4. Retirement lets you adapt.

### Step 5: Visualize Results (10 minutes)

Create plots showing:
1. **Posterior evolution:** How beliefs about each arm changed over time
2. **Arm selection frequency:** Which arms were picked most often
3. **Cumulative engagement:** Total engagement vs baseline (random selection)
4. **Regret:** How much engagement you lost vs always picking the best arm

The plotting functions are provided, but you'll need to collect the right data.

### Step 6: Compare to Baselines (10 minutes)

Run three strategies and compare:
1. **Random:** Pick arms uniformly at random
2. **Pure Exploit:** Always pick empirically best arm (no exploration)
3. **Thompson Sampling:** Your implementation

**Expected results:**
- Random: ~42% average read ratio (average of true means)
- Pure Exploit: Gets stuck on early lucky arm, ~44%
- Thompson Sampling: Finds true best, ~47%

## Expected Output

A report showing:

```
=== CONTENT STRATEGY OPTIMIZER RESULTS ===

Week 52 Final Performance:
┌─────────────────────────────────────┬────────┬────────┬──────────┐
│ Arm                                 │ Picks  │ Avg RR │ Posterior│
├─────────────────────────────────────┼────────┼────────┼──────────┤
│ Trading Psychology × Essay          │   87   │ 52.3%  │ β(46,42) │
│ Risk Management × Thread            │   71   │ 48.1%  │ β(35,38) │
│ Market Analysis × Essay             │   52   │ 45.2%  │ β(24,29) │
│ Trading Psychology × Thread         │   38   │ 41.5%  │ β(16,23) │
│ Market Analysis × Thread            │   34   │ 38.7%  │ β(14,22) │
│ Risk Management × Essay             │   23   │ 35.2%  │ β(9,17)  │
└─────────────────────────────────────┴────────┴────────┴──────────┘

Total Engagement: 142.8 (vs 124.3 random baseline)
Regret: 8.4% (vs optimal strategy)
```

Plus visualizations of posterior evolution and learning curves.

## Success Criteria

Your implementation is successful if:

1. Thompson Sampling converges to the true best arm (highest read ratio) within 20 weeks
2. Cumulative engagement beats random baseline by 10%+
3. Posterior distributions narrow over time (increasing certainty)
4. Arm retirement successfully introduces and evaluates new content types
5. Exploration rate decays naturally (you pick best arm more often over time)

## Extensions (Optional)

Once you have the basic system working:

1. **Contextual features:** Add seasonality (some topics work better in certain months)
2. **Non-stationarity:** Simulate audience preferences changing mid-year
3. **Multiple reward metrics:** Track both read ratio AND shares
4. **Budget constraints:** Limited publishing capacity (can't publish all arms)
5. **A/B testing comparison:** Run a traditional A/B test on same problem, compare sample efficiency

## File Structure

```
project_1_beginner/
├── README.md              # This file
├── starter_code.py        # Your working skeleton (complete TODOs)
└── solution.py            # Reference implementation (only look if stuck!)
```

## Getting Started

1. Read this README completely
2. Open `starter_code.py`
3. Run it to see the skeleton structure
4. Complete the TODOs in order
5. Run again and iterate until tests pass
6. Compare to `solution.py` only if truly stuck

## Next Steps

After completing this project:
- **Project 2:** Apply same framework to commodity portfolio allocation
- **Module 5:** Learn the two-wallet framework for live trading
- **Module 7:** Deploy a production bandit system with monitoring

## Questions to Consider

1. Why is read ratio a better reward than views?
2. How does arm retirement prevent the bandit from getting stuck?
3. What happens if you set exploration too low (e.g., 5% instead of 40%)?
4. How would this change for a trader picking between 5 commodity strategies?
5. What if your audience preferences changed mid-year — would the bandit adapt?

---

**Remember:** The goal isn't to predict which content will work. It's to learn what works while wasting as little effort as possible on what doesn't.
