# Logging and Monitoring

> **Reading time:** ~15 min | **Module:** 07 — Production Systems | **Prerequisites:** Module 6


## In Brief


<div class="callout-key">

**Key Concept Summary:** Production bandit systems require comprehensive logging of every decision (arm, context, reward, metadata) and real-time monitoring of key metrics (regret estimates, arm selection distribution, reward

</div>

Production bandit systems require comprehensive logging of every decision (arm, context, reward, metadata) and real-time monitoring of key metrics (regret estimates, arm selection distribution, reward trends) to detect failures before they cause significant losses.

> 💡 **Key Insight:** The difference between a research notebook and a production system is observability. When your bandit starts losing money, you need to answer: Which arm is underperforming? Did the context features change? Is the policy stuck on one arm? Did we violate position limits? Without structured logging and monitoring, you're flying blind.

## Visual Explanation

<div class="callout-insight">

**Insight:** The core insight of bandit algorithms is that learning and earning are not separate phases. Every observation contributes to both understanding which option is best and generating value from the best option.

</div>


```
┌───────────────────────────────────────────────────────────────────┐
│                    WHAT TO LOG (Per Decision)                     │
├───────────────────────────────────────────────────────────────────┤
│                                                                     │
│  {                                                                 │
│    "timestamp": "2026-02-12T10:30:00Z",                           │
│    "decision_id": "uuid-1234",                                    │
│    "policy_version": "thompson_v2.1",                             │
│    "context": {                                                    │
│      "vix": 18.5,                                                  │
│      "term_structure": {"GOLD": 0.02, "OIL": -0.03},              │
│      "regime": "high_volatility"                                   │
│    },                                                              │
│    "active_arms": ["GOLD", "OIL", "NATGAS", "COPPER"],           │
│    "policy_scores": {"GOLD": 0.45, "OIL": 0.30, ...},            │
│    "selected_arm": "GOLD",                                        │
│    "guardrail_override": false,                                    │
│    "final_arm": "GOLD",                                           │
│    "position_size": 0.25,                                         │
│    "metadata": {                                                   │
│      "exploration_type": "thompson_sample",                        │
│      "arm_pull_count": 15                                          │
│    }                                                               │
│  }                                                                 │
│                                                                     │
│  # Later, when reward is observed:                                │
│  {                                                                 │
│    "decision_id": "uuid-1234",                                    │
│    "reward": 0.023,  # 2.3% weekly return                         │
│    "reward_timestamp": "2026-02-19T16:00:00Z",                    │
│    "latency_days": 7                                               │
│  }                                                                 │
│                                                                     │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│                    MONITORING DASHBOARD LAYOUT                     │
├───────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────┐  ┌─────────────────────────┐         │
│  │  Cumulative Regret      │  │  Arm Selection Dist.    │         │
│  │                         │  │                         │         │
│  │      ╱                  │  │  GOLD  ████████ 40%    │         │
│  │     ╱                   │  │  OIL   ████ 20%        │         │
│  │    ╱                    │  │  GAS   ████ 20%        │         │
│  │   ╱                     │  │  COPPER ████ 20%       │         │
│  │  ╱____________________  │  │                         │         │
│  │    Week 1 → Week 52    │  │  Entropy: 1.38 (good)  │         │
│  └─────────────────────────┘  └─────────────────────────┘         │
│                                                                     │
│  ┌─────────────────────────┐  ┌─────────────────────────┐         │
│  │  Reward Moving Avg      │  │  Feature Drift          │         │
│  │  (7-week window)        │  │                         │         │
│  │                         │  │  VIX mean: 18.5→22.3   │         │
│  │   ╱╲    ╱╲              │  │  ⚠️  Shift detected!   │         │
│  │  ╱  ╲  ╱  ╲             │  │                         │         │
│  │ ╱    ╲╱    ╲            │  │  Term struct: stable    │         │
│  │╱            ╲___        │  │  ✓ No drift             │         │
│  └─────────────────────────┘  └─────────────────────────┘         │
│                                                                     │
│  ┌──────────────────────────────────────────────────────┐         │
│  │  ALERTS                                               │         │
│  │  🔴 Policy Collapse: GOLD selected 20 consecutive    │         │
│  │     times (entropy < 0.5)                             │         │
│  │  🟡 Reward degradation: -2.1% below baseline         │         │
│  │  🟢 All systems normal                                │         │
│  └──────────────────────────────────────────────────────┘         │
│                                                                     │
└───────────────────────────────────────────────────────────────────┘
```

## Formal Definition

<div class="callout-warning">

**Warning:** Bandit algorithms assume the reward distributions are stationary (or slowly changing). In commodity markets, regime shifts can make a historically optimal arm suddenly suboptimal. Always implement change detection alongside your bandit.

</div>


**Logging Requirements:**

For each decision at time $t$, log the tuple:
$$L_t = (t, c_t, \mathcal{A}_t, \pi_t, a_t, a_t^*, m_t)$$

Where:
- $c_t \in \mathbb{R}^d$ = context features
- $\mathcal{A}_t$ = set of active arms
- $\pi_t$ = policy scores (probability/value for each arm)
- $a_t$ = policy-selected arm
- $a_t^*$ = final arm after guardrails
- $m_t$ = metadata (version, flags, etc.)

When reward $r_t$ is observed, append:
$$L_t \leftarrow L_t \cup (r_t, t_{observe})$$

**Monitoring Metrics:**

1. **Estimated Cumulative Regret:** $\hat{R}_T = \sum_{t=1}^T (\hat{\mu}^* - \hat{\mu}_{a_t})$ where $\hat{\mu}^* = \max_a \hat{\mu}_a$

2. **Selection Entropy:** $H(T) = -\sum_{a \in \mathcal{A}} p_a \log p_a$ where $p_a = \frac{N_a(T)}{T}$
   - Low entropy (< 0.5): policy collapse
   - High entropy (> 1.5): excessive exploration

3. **Reward Drift:** $\Delta_w = |\text{MA}_w(r_t) - \text{MA}_w(r_{t-k})|$ for window $w$
   - Alert if $\Delta_w > 2\sigma$ (2 standard deviations)

4. **Feature Drift:** KL divergence between recent and historical feature distributions
   - $D_{KL}(P_{\text{recent}} || P_{\text{historical}}) > \tau$ triggers alert

## Intuitive Explanation

**Logging** is like a black box recorder on an airplane. When something goes wrong, you need to replay exactly what happened. But unlike aviation, you can't just record "altitude and speed" — you need to capture:

- **Decision context:** What did the world look like when you made this choice?
- **Policy reasoning:** Why did the algorithm pick this arm? (Store the scores/probabilities)
- **Guardrail interventions:** Did risk management override the policy?
- **Outcome:** What actually happened, and when did you learn about it?

**Monitoring** is like the cockpit instruments. You need real-time signals that tell you:

- **Are we stuck?** (Policy collapse = always picking the same arm)
- **Are we lost?** (High regret = consistently picking suboptimal arms)
- **Is the world changing?** (Feature drift = market regime shift)
- **Are we earning?** (Reward trends)

The key is **actionable alerts**. Don't just track 50 metrics — identify the 5 failure modes that matter and set clear thresholds.

## Code Implementation


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import json
from datetime import datetime
from collections import deque, Counter
import numpy as np

class BanditLogger:
    """Structured logging for bandit decisions."""

    def __init__(self, log_file="bandit_decisions.jsonl"):
        self.log_file = log_file

    def log_decision(self, decision_id, policy_version, context, active_arms,
                    policy_scores, selected_arm, final_arm, metadata):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "decision_id": decision_id,
            "policy_version": policy_version,
            "context": context,
            "active_arms": active_arms,
            "policy_scores": policy_scores,
            "selected_arm": selected_arm,
            "final_arm": final_arm,
            "metadata": metadata
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        return decision_id

    def log_reward(self, decision_id, reward):
        reward_entry = {
            "decision_id": decision_id,
            "reward": reward,
            "reward_timestamp": datetime.now().isoformat()
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(reward_entry) + "\n")

class BanditMonitor:
    """Real-time monitoring and alerting."""

    def __init__(self, window_size=20):
        self.window_size = window_size
        self.recent_arms = deque(maxlen=window_size)
        self.recent_rewards = deque(maxlen=window_size)

    def update(self, arm, reward):
        self.recent_arms.append(arm)
        self.recent_rewards.append(reward)

    def check_policy_collapse(self, threshold=0.5):
        """Alert if selection entropy is too low."""
        if len(self.recent_arms) < self.window_size:
            return False

        counts = Counter(self.recent_arms)
        probs = np.array([counts[a] for a in counts]) / len(self.recent_arms)
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        return entropy < threshold

    def check_reward_degradation(self, baseline, threshold=0.02):
        """Alert if recent rewards significantly below baseline."""
        if len(self.recent_rewards) < 10:
            return False

        recent_mean = np.mean(self.recent_rewards)
        return (baseline - recent_mean) > threshold

    def get_summary(self):
        """Generate monitoring summary."""
        if not self.recent_arms:
            return {}

        arm_counts = Counter(self.recent_arms)
        return {
            "arm_distribution": dict(arm_counts),
            "avg_reward": np.mean(self.recent_rewards) if self.recent_rewards else 0,
            "policy_collapse": self.check_policy_collapse(),
            "recent_entropy": self._compute_entropy(),
        }

    def _compute_entropy(self):
        if not self.recent_arms:
            return 0
        counts = Counter(self.recent_arms)
        probs = np.array([counts[a] for a in counts]) / len(self.recent_arms)
        return -np.sum(probs * np.log(probs + 1e-10))
```

</div>
</div>

**Commodity Application:**

<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Weekly commodity allocation with logging
logger = BanditLogger("commodity_decisions.jsonl")
monitor = BanditMonitor(window_size=12)  # 12 weeks

for week in range(52):
    context = get_market_context(week)
    decision_id = f"week_{week}"

    # Make decision
    arm = bandit.select_arm(context)

    # Log decision
    logger.log_decision(
        decision_id=decision_id,
        policy_version="thompson_v1.0",
        context=context,
        active_arms=["GOLD", "OIL", "NATGAS", "COPPER"],
        policy_scores=bandit.get_arm_scores(),
        selected_arm=arm,
        final_arm=arm,
        metadata={"week": week}
    )

    # Execute trade and observe reward
    reward = execute_and_observe(arm)

    # Log reward
    logger.log_reward(decision_id, reward)

    # Update monitor
    monitor.update(arm, reward)

    # Check alerts
    if monitor.check_policy_collapse():
        print(f"⚠️  Week {week}: Policy collapse detected!")

    if monitor.check_reward_degradation(baseline=0.01):
        print(f"⚠️  Week {week}: Reward degradation!")

# Final summary
print(monitor.get_summary())
```

</div>
</div>

## Common Pitfalls

**Pitfall 1: Logging only successes**
When debugging, you need to see the failures. Log rejected arms, guardrail overrides, and error conditions.

**Solution:** Log every decision path, including what didn't happen. Store policy scores for all arms, not just the selected one.

**Pitfall 2: Monitoring vanity metrics**
Tracking "total decisions made" or "uptime percentage" doesn't tell you if the policy is working.

**Solution:** Monitor business metrics (regret, returns, Sharpe ratio) and operational metrics (policy collapse, feature drift).

**Pitfall 3: Alert fatigue**
Setting thresholds too sensitive creates noise. Too loose and you miss real problems.

**Solution:** Calibrate alert thresholds using historical data. Start conservative, tune based on false positive rate.

**Pitfall 4: No baseline for comparison**
"Average reward is 0.8%" — is that good or bad? You need a benchmark (equal weighting, best single arm, etc.).

**Solution:** Always log baseline performance alongside bandit performance. Monitor the difference.

## Connections

<div class="callout-danger">

**Danger:** Never deploy a bandit system without a kill switch and maximum allocation limits. An unconstrained bandit can allocate 100% of traffic/capital to a single arm, which creates catastrophic risk if the reward signal is noisy or delayed.

</div>


**Builds on:**
- Module 0: Regret definitions and cumulative regret calculation
- Previous guide: Production system architecture (what to log where)

**Leads to:**
- Offline evaluation (using logged data to evaluate new policies)
- A/B testing integration (comparing bandit to fixed allocation)
- Post-deployment analysis and debugging

**Related concepts:**
- Observability in distributed systems (metrics, logs, traces)
- Statistical process control (detecting anomalies in time series)
- Change point detection (identifying regime shifts)

## Practice Problems

1. **Design Challenge:** You're monitoring a commodity bandit that rebalances weekly. Design a dashboard with exactly 4 metrics that would catch the most critical failures. Justify each choice.

2. **Implementation:** Extend `BanditMonitor` to detect feature drift. Given historical context features and recent context features, compute KL divergence and alert if it exceeds a threshold. (Hint: discretize continuous features into bins first.)

3. **Real Scenario:** Your monitor shows:
   - Entropy: 0.3 (very low)
   - Recent reward: 0.5% (below 1% baseline)
   - GOLD selected 18 out of last 20 weeks

   What's happening? Is this necessarily bad? What would you investigate next?

4. **Code Review:** What's missing from this logging approach?

<span class="filename">example.py</span>
</div>

   <div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
   def make_decision(context):
       arm = policy.select(context)
       print(f"Selected: {arm}")
       return arm
   ```

</div>
</div>

   Rewrite it with proper structured logging that would support debugging 6 months later.


---

## Cross-References

<a class="link-card" href="./01_bandit_system_architecture.md">
  <div class="link-card-title">01 Bandit System Architecture</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_bandit_system_architecture.md">
  <div class="link-card-title">01 Bandit System Architecture — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./03_offline_evaluation.md">
  <div class="link-card-title">03 Offline Evaluation</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_offline_evaluation.md">
  <div class="link-card-title">03 Offline Evaluation — Companion Slides</div>
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

