# Bandit System Architecture

> **Reading time:** ~14 min | **Module:** 07 — Production Systems | **Prerequisites:** Module 6


## In Brief


<div class="callout-key">

**Key Concept Summary:** A production bandit system separates policy logic (how to select arms), data management (tracking rewards and contexts), and business logic (guardrails and overrides) into distinct components that can

</div>

A production bandit system separates policy logic (how to select arms), data management (tracking rewards and contexts), and business logic (guardrails and overrides) into distinct components that can be developed, tested, and monitored independently.

> 💡 **Key Insight:** The biggest mistake in production bandit systems is coupling the machine learning policy too tightly with business logic. When your Thompson Sampling algorithm is tangled with position limits, stop-loss checks, and data validation, you can't update one without breaking the other. Clean architecture with clear interfaces enables safe iteration and reliable monitoring.

## Visual Explanation

<div class="callout-insight">

**Insight:** The core insight of bandit algorithms is that learning and earning are not separate phases. Every observation contributes to both understanding which option is best and generating value from the best option.

</div>


```
┌─────────────────────────────────────────────────────────────────┐
│                     PRODUCTION BANDIT SYSTEM                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌───────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │  Arm Registry │      │  Context     │      │  Guardrails  │  │
│  │               │      │  Provider    │      │  Engine      │  │
│  │ • arm_id      │      │              │      │              │  │
│  │ • metadata    │      │ • features   │      │ • pos limits │  │
│  │ • enabled     │      │ • enrichment │      │ • stop loss  │  │
│  └───────┬───────┘      └──────┬───────┘      └──────┬───────┘  │
│          │                     │                     │           │
│          └─────────────────────┼─────────────────────┘           │
│                                │                                 │
│                        ┌───────▼────────┐                        │
│                        │  Policy Engine │                        │
│                        │                │                        │
│                        │ • select_arm() │                        │
│                        │ • update()     │                        │
│                        │ • get_stats()  │                        │
│                        └───────┬────────┘                        │
│                                │                                 │
│          ┌─────────────────────┼─────────────────────┐           │
│          │                     │                     │           │
│  ┌───────▼───────┐    ┌────────▼────────┐    ┌──────▼──────┐   │
│  │  Reward       │    │  Structured     │    │  Monitoring │   │
│  │  Tracker      │    │  Logger         │    │  Dashboard  │   │
│  │               │    │                 │    │             │   │
│  │ • history     │    │ • JSON logs     │    │ • alerts    │   │
│  │ • aggregates  │    │ • audit trail   │    │ • metrics   │   │
│  └───────────────┘    └─────────────────┘    └─────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Data Flow:**
1. Context Provider → extracts features from market data
2. Policy Engine → selects arm based on context and history
3. Guardrails Engine → validates selection (position limits, risk checks)
4. Arm Registry → returns arm configuration and metadata
5. Logger → records decision with full context
6. Reward Tracker → observes outcome, updates policy
7. Monitor → checks for anomalies, triggers alerts

## Formal Definition

<div class="callout-warning">

**Warning:** Bandit algorithms assume the reward distributions are stationary (or slowly changing). In commodity markets, regime shifts can make a historically optimal arm suddenly suboptimal. Always implement change detection alongside your bandit.

</div>


A production bandit system is a tuple $(R, C, P, G, L, M)$ where:

- **R** = Arm Registry: maps arm IDs to configurations and enables/disables arms dynamically
- **C** = Context Provider: function $c: t \to \mathbb{R}^d$ producing feature vectors at time $t$
- **P** = Policy Engine: implements $\pi: (c, H) \to a$ where $H$ is history
- **G** = Guardrails: constraint function $g: a \to \{accept, reject, override\}$
- **L** = Logger: records tuples $(t, c_t, a_t, r_t, \text{metadata})$ to persistent storage
- **M** = Monitor: evaluates metrics $m: H \to \mathbb{R}^k$ and triggers alerts

The system processes decisions as:
$$\text{action} = G(P(C(t), H), t)$$

Where $G$ can override policy decisions based on business rules.

## Intuitive Explanation

Think of a production bandit system like an automated trading desk with specialized roles:

- **Arm Registry** = The list of tradable assets (with circuit breakers to disable problematic ones)
- **Context Provider** = Market data analysts who prepare relevant indicators
- **Policy Engine** = The portfolio manager who decides allocations
- **Guardrails** = Risk management team who can veto or modify decisions
- **Logger** = Compliance department recording every decision for audits
- **Reward Tracker** = Performance attribution analysts
- **Monitor** = Real-time surveillance watching for anomalies

Just like you wouldn't let a portfolio manager bypass risk limits, your policy engine shouldn't be able to violate guardrails. And just like compliance needs complete audit trails, your logger must capture everything.

## Code Implementation


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import json

@dataclass
class Arm:
    arm_id: str
    metadata: Dict
    enabled: bool = True

class ArmRegistry:
    def __init__(self):
        self.arms: Dict[str, Arm] = {}

    def register(self, arm: Arm):
        self.arms[arm.arm_id] = arm

    def get_active_arms(self) -> List[str]:
        return [a.arm_id for a in self.arms.values() if a.enabled]

class PolicyEngine:
    def select_arm(self, context: Dict, active_arms: List[str]) -> str:
        raise NotImplementedError

    def update(self, arm: str, reward: float, context: Dict):
        raise NotImplementedError

class ProductionBanditSystem:
    def __init__(self, policy: PolicyEngine, guardrails: Optional[callable] = None):
        self.registry = ArmRegistry()
        self.policy = policy
        self.guardrails = guardrails or (lambda x: x)
        self.logger = []
        self.history = []

    def make_decision(self, context: Dict) -> str:
        active_arms = self.registry.get_active_arms()
        selected_arm = self.policy.select_arm(context, active_arms)
        validated_arm = self.guardrails(selected_arm, context)

        decision = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "selected_arm": selected_arm,
            "final_arm": validated_arm,
            "active_arms": active_arms
        }
        self.logger.append(json.dumps(decision))
        return validated_arm

    def record_reward(self, arm: str, reward: float, context: Dict):
        self.history.append((arm, reward, context))
        self.policy.update(arm, reward, context)
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
# Weekly commodity allocation system
def commodity_guardrails(arm: str, context: Dict) -> str:
    """Validate commodity selection before execution."""
    # Don't allocate to commodities in backwardation > 5%
    if context.get(f"{arm}_term_structure", 0) < -0.05:
        return "CASH"  # Override to cash

    # Don't allocate if volatility > 40% annualized
    if context.get(f"{arm}_volatility", 0) > 0.40:
        return "CASH"

    return arm  # Accept policy decision

# Usage
system = ProductionBanditSystem(
    policy=ThompsonSamplingPolicy(),
    guardrails=commodity_guardrails
)

# Register commodities
for commodity in ["GOLD", "OIL", "NATGAS", "COPPER"]:
    system.registry.register(Arm(commodity, metadata={"sector": "energy"}))

# Weekly decision
context = get_market_features()  # VIX, term structure, momentum
allocation = system.make_decision(context)
```

</div>
</div>

## Common Pitfalls

**Pitfall 1: Monolithic design**
Putting everything in one giant class makes testing and debugging impossible. You can't unit test your Thompson Sampling logic if it's entangled with database calls and position limits.

**Solution:** Separate concerns. Policy engine should only know about arm selection. Guardrails should only validate. Logger should only record.

**Pitfall 2: Synchronous reward tracking**
Waiting for reward observation before returning from `make_decision()` creates coupling and latency. In commodity trading, you might not observe the weekly return until Friday close.

**Solution:** Decouple decision making from reward observation. `make_decision()` returns immediately. `record_reward()` is called asynchronously when the outcome is known.

**Pitfall 3: No versioning**
Your policy changes over time (new features, different algorithms). If you don't version decisions in your logs, you can't reproduce past behavior or debug issues.

**Solution:** Log policy version and all hyperparameters with every decision. Include git commit hash if possible.

**Pitfall 4: Missing arm metadata**
When a commodity allocation performs poorly, you need to know: Was the arm enabled? What was the market regime? What features were available?

**Solution:** The Arm Registry should store rich metadata. The logger should capture it on every decision.

## Connections

<div class="callout-danger">

**Danger:** Never deploy a bandit system without a kill switch and maximum allocation limits. An unconstrained bandit can allocate 100% of traffic/capital to a single arm, which creates catastrophic risk if the reward signal is noisy or delayed.

</div>


**Builds on:**
- Module 1: Core bandit algorithms (epsilon-greedy, Thompson Sampling)
- Module 3: Contextual bandits (LinUCB, neural bandits)
- Module 5: Commodity trading applications

**Leads to:**
- Logging and monitoring (next guide)
- Offline evaluation using logged data
- Multi-environment deployments (paper trading → production)
- A/B testing integration

**Related concepts:**
- MLOps: model versioning, feature stores, monitoring
- Event sourcing: append-only logs as source of truth
- Circuit breaker pattern: disabling failing arms
- Microservices: each component could be a separate service

## Practice Problems

1. **Architectural Decision:** You're deploying a commodity bandit that rebalances weekly. Should the Policy Engine, Guardrails, and Logger be separate microservices or modules in one application? Justify your answer considering latency, failure modes, and operational complexity.

2. **Implementation Challenge:** Extend the `ProductionBanditSystem` class to support:
   - A/B testing mode (override policy with fixed allocation for a subset of decisions)
   - Feature flag for enabling/disabling specific arms remotely
   - Audit log export to S3/GCS for compliance

3. **Real-World Scenario:** Your commodity allocation system needs to handle:
   - New commodities added mid-deployment (cold start problem)
   - Commodities temporarily disabled due to liquidity issues
   - Emergency stop-loss override when portfolio drawdown exceeds 10%

   Design the interfaces for ArmRegistry and Guardrails that support these requirements.

4. **Code Review:** What's wrong with this architecture?

<span class="filename">example.py</span>
</div>

   <div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
   class BanditSystem:
       def decide(self, context):
           arm = self.policy.select(context)
           reward = execute_trade(arm)  # Execute immediately
           self.policy.update(arm, reward)
           return arm
   ```

</div>
</div>

   List at least 3 architectural problems and how to fix them.


---

## Cross-References

<a class="link-card" href="./02_logging_and_monitoring.md">
  <div class="link-card-title">02 Logging And Monitoring</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_logging_and_monitoring.md">
  <div class="link-card-title">02 Logging And Monitoring — Companion Slides</div>
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

