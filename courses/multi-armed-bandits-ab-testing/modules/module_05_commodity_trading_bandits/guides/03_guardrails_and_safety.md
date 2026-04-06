# Guardrails and Safety

> **Reading time:** ~20 min | **Module:** 05 — Commodity Trading Bandits | **Prerequisites:** Module 4


## In Brief


<div class="callout-key">

**Key Concept Summary:** Bandit algorithms without constraints become dangerous optimization machines. Guardrails are hard limits that prevent concentration risk, overtrading, and regime-blind allocation. They're not weakn...

</div>

Bandit algorithms without constraints become dangerous optimization machines. Guardrails are hard limits that prevent concentration risk, overtrading, and regime-blind allocation. They're not weakness—they're the difference between a learning system and self-sabotage.

> 💡 **Key Insight:** "Without guardrails, bandit investing becomes a dopamine machine."

A pure bandit will:
- Concentrate entirely in the recent winner (concentration risk)
- Abandon arms after one bad result (premature exploration termination)
- Rebalance constantly (transaction cost explosion)
- Ignore regime changes (optimization in the wrong context)

Guardrails force the bandit to:
- Maintain diversification
- Give arms sufficient trials before reducing exposure
- Limit portfolio churn
- Adapt allocation speed to market conditions

## Visual Explanation

<div class="callout-insight">

**Insight:** The core insight of bandit algorithms is that learning and earning are not separate phases. Every observation contributes to both understanding which option is best and generating value from the best option.

</div>


```
┌────────────────────────────────────────────────────────────┐
│  Guardrail Checkpoint Flow                                 │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  STEP 1: Bandit Proposes Allocation                        │
│  ┌──────────────────────────────────────┐                  │
│  │ WTI: 60%, Gold: 30%, Copper: 10%     │                  │
│  │ NatGas: 0%, Corn: 0%                 │ ← Raw bandit     │
│  └──────────────────────────────────────┘   suggestion     │
│                    ↓                                        │
│  CHECKPOINT 1: Position Limits                             │
│  ┌──────────────────────────────────────┐                  │
│  │ Max per arm: 40%                     │                  │
│  │ WTI: 60% → 40% ✗ CAPPED              │                  │
│  │ Gold: 30% → 30% ✓                    │                  │
│  └──────────────────────────────────────┘                  │
│                    ↓                                        │
│  CHECKPOINT 2: Minimum Allocation                          │
│  ┌──────────────────────────────────────┐                  │
│  │ Min per arm: 5%                      │                  │
│  │ NatGas: 0% → 5% ✗ RAISED             │                  │
│  │ Corn: 0% → 5% ✗ RAISED               │                  │
│  └──────────────────────────────────────┘                  │
│                    ↓                                        │
│  CHECKPOINT 3: Tilt Speed Limit                            │
│  ┌──────────────────────────────────────┐                  │
│  │ Max change: 15% per week             │                  │
│  │ WTI: was 25%, now 40% → +15% ✓       │                  │
│  │ Gold: was 20%, now 30% → +10% ✓      │                  │
│  └──────────────────────────────────────┘                  │
│                    ↓                                        │
│  CHECKPOINT 4: Volatility Dampening                        │
│  ┌──────────────────────────────────────┐                  │
│  │ VIX > 30? Reduce tilt aggressiveness │                  │
│  │ Current VIX: 25 → Normal mode ✓      │                  │
│  └──────────────────────────────────────┘                  │
│                    ↓                                        │
│  FINAL ALLOCATION (Post-Guardrails)                        │
│  ┌──────────────────────────────────────┐                  │
│  │ WTI: 35%, Gold: 28%, Copper: 20%     │                  │
│  │ NatGas: 10%, Corn: 7%                │ ← Safe, diverse  │
│  └──────────────────────────────────────┘                  │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

## The Five Essential Guardrails

<div class="callout-warning">

**Warning:** Bandit algorithms assume the reward distributions are stationary (or slowly changing). In commodity markets, regime shifts can make a historically optimal arm suddenly suboptimal. Always implement change detection alongside your bandit.

</div>


### Guardrail 1: Position Limits

**Purpose:** Prevent concentration risk.

**Rule:** No single arm can exceed a maximum weight in the bandit sleeve.


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def apply_position_limits(weights, max_weight=0.40):
    """
    Cap any position above max_weight.

    Args:
        weights: Array of proposed allocation weights
        max_weight: Maximum weight for any single arm
    """
    weights = np.clip(weights, 0, max_weight)
    # Re-normalize to sum to 1
    return weights / weights.sum()
```

</div>
</div>

**Typical values:**
- Conservative: 30% max per arm
- Moderate: 40% max per arm
- Aggressive: 50% max per arm

**Commodity example:**

<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Bandit proposes: [0.70, 0.15, 0.10, 0.05, 0.00]
# After position limit (40%): [0.44, 0.17, 0.11, 0.06, 0.22]
#   → Forced diversification

weights = np.array([0.70, 0.15, 0.10, 0.05, 0.00])
safe_weights = apply_position_limits(weights, max_weight=0.40)
# Result: WTI capped at 40%, excess reallocated
```

</div>
</div>

**Why it matters:**
- One commodity can dominate without this (e.g., 95% WTI after strong streak)
- Portfolio becomes a levered bet on single name
- One bad event wipes you out

### Guardrail 2: Minimum Allocation

**Purpose:** Prevent premature abandonment of arms.

**Rule:** Every arm must maintain a minimum weight, even after poor performance.


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def apply_minimum_allocation(weights, min_weight=0.05):
    """
    Ensure no arm falls below min_weight.

    Args:
        weights: Array of proposed allocation weights
        min_weight: Minimum weight for any single arm
    """
    K = len(weights)
    weights = np.maximum(weights, min_weight)
    # Re-normalize
    return weights / weights.sum()
```

</div>
</div>

**Typical values:**
- Conservative: 10% min per arm (forces near-equal weight)
- Moderate: 5% min per arm
- Aggressive: 2% min per arm (allows more tilt)

**Commodity example:**
```python
# Bandit proposes: [0.50, 0.30, 0.15, 0.05, 0.00]
# After min allocation (5%): [0.48, 0.29, 0.14, 0.05, 0.05]
#   → Corn stays alive despite zero allocation

weights = np.array([0.50, 0.30, 0.15, 0.05, 0.00])
safe_weights = apply_minimum_allocation(weights, min_weight=0.05)
# Result: Even worst-performing arm gets 5%
```

**Why it matters:**
- Sample size of 1 week is insufficient to judge an arm
- Volatility ≠ bad arm (might just be noisy)
- Maintains exploration even during exploitation phase

### Guardrail 3: Tilt Speed Limits

**Purpose:** Prevent excessive portfolio turnover.

**Rule:** Limit how much allocation can change from one period to the next.

```python
def apply_tilt_speed_limit(
    new_weights,
    old_weights,
    max_change=0.15
):
    """
    Limit allocation change between periods.

    Args:
        new_weights: Proposed new weights
        old_weights: Current weights
        max_change: Maximum change in allocation per arm
    """
    change = new_weights - old_weights
    # Clip change to [-max_change, +max_change]
    clipped_change = np.clip(change, -max_change, max_change)
    adjusted_weights = old_weights + clipped_change
    # Re-normalize
    return adjusted_weights / adjusted_weights.sum()
```

**Typical values:**
- Conservative: 10% max change per period
- Moderate: 15% max change per period
- Aggressive: 25% max change per period

**Commodity example:**
```python
# Current: [0.20, 0.20, 0.20, 0.20, 0.20]
# Proposed: [0.50, 0.30, 0.10, 0.05, 0.05]
# After speed limit (15%): [0.35, 0.28, 0.15, 0.12, 0.10]
#   → Gradual tilt, not sudden swing

old = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
new = np.array([0.50, 0.30, 0.10, 0.05, 0.05])
safe = apply_tilt_speed_limit(new, old, max_change=0.15)
# Result: WTI increases by max 15%, not full 30%
```

**Why it matters:**
- Transaction costs (bid-ask spread, commissions)
- Market impact (large trades move prices)
- Smoother equity curve (easier to hold during drawdowns)

### Guardrail 4: Core Protection

**Purpose:** Separate strategic (core) from tactical (bandit) allocation.

**Rule:** Core allocation rebalances on a different, slower timescale than bandit sleeve.

```python
class TwoWalletGuardrail:
    """
    Separate core (monthly rebalance) from bandit (weekly rebalance).
    """
    def __init__(self, core_pct=0.80, bandit_pct=0.20):
        self.core_pct = core_pct
        self.bandit_pct = bandit_pct
        self.core_weights = None
        self.last_core_rebalance = None

    def get_allocation(
        self,
        bandit_weights,
        strategic_weights,
        current_date,
        core_rebalance_freq='M'  # Monthly
    ):
        """
        Combine core and bandit with different rebalance frequencies.
        """
        # Check if core needs rebalancing
        if (self.last_core_rebalance is None or
            self._should_rebalance(current_date, core_rebalance_freq)):
            self.core_weights = strategic_weights
            self.last_core_rebalance = current_date

        # Combine
        total_weights = (
            self.core_pct * self.core_weights +
            self.bandit_pct * bandit_weights
        )
        return total_weights

    def _should_rebalance(self, date, freq):
        """Check if rebalance is due."""
        if freq == 'M':  # Monthly
            return date.day == 1
        elif freq == 'Q':  # Quarterly
            return date.day == 1 and date.month in [1, 4, 7, 10]
        return False
```

**Why it matters:**
- Core provides stability and strategic direction
- Bandit provides tactical tilts without disrupting core
- Different timescales prevent bandit from dominating

### Guardrail 5: Volatility Dampening

**Purpose:** Reduce tilt aggressiveness during high-volatility regimes.

**Rule:** When market volatility spikes, reduce bandit sleeve or tilt speed.

```python
def apply_volatility_dampening(
    bandit_weights,
    core_weights,
    current_vix,
    vix_threshold=30,
    dampening_factor=0.5
):
    """
    Reduce bandit influence during high volatility.

    Args:
        bandit_weights: Proposed bandit allocation
        core_weights: Strategic core allocation
        current_vix: Current VIX level
        vix_threshold: VIX level to trigger dampening
        dampening_factor: How much to reduce bandit (0=full, 1=none)
    """
    if current_vix > vix_threshold:
        # Blend bandit back toward core
        damped_weights = (
            dampening_factor * bandit_weights +
            (1 - dampening_factor) * core_weights
        )
        return damped_weights
    else:
        return bandit_weights
```

**Typical thresholds:**
- VIX < 20: Normal operation
- VIX 20-30: Light dampening (0.7 factor)
- VIX > 30: Heavy dampening (0.5 factor)
- VIX > 40: Crisis mode (revert to core)

**Commodity example:**
```python
# Normal regime (VIX=18):
#   Bandit: [0.40, 0.30, 0.15, 0.10, 0.05]
#   Core: [0.20, 0.20, 0.20, 0.20, 0.20]
#   Final: bandit weights (no dampening)

# Crisis regime (VIX=35):
#   Bandit: [0.40, 0.30, 0.15, 0.10, 0.05]
#   Core: [0.20, 0.20, 0.20, 0.20, 0.20]
#   Final: 0.5 * bandit + 0.5 * core
#        = [0.30, 0.25, 0.175, 0.15, 0.125]
#   → Moved halfway back to equal-weight
```

**Why it matters:**
- High volatility = higher uncertainty
- Bandit beliefs less reliable in regime shifts
- Prevents disaster during market dislocations

## Commodity-Specific Guardrails

### Guardrail 6: Correlation Limits

**Purpose:** Prevent concentration in highly correlated commodities.

```python
def apply_correlation_limit(
    weights,
    correlation_matrix,
    max_correlated_weight=0.60
):
    """
    Limit combined weight of highly correlated arms.

    Args:
        weights: Proposed allocation
        correlation_matrix: K x K correlation matrix
        max_correlated_weight: Max combined weight for correlated arms
    """
    K = len(weights)
    adjusted = weights.copy()

    # Find highly correlated pairs (corr > 0.7)
    for i in range(K):
        correlated_group = [i]
        for j in range(i+1, K):
            if correlation_matrix[i, j] > 0.7:
                correlated_group.append(j)

        # Check combined weight
        group_weight = adjusted[correlated_group].sum()
        if group_weight > max_correlated_weight:
            # Scale down proportionally
            scale = max_correlated_weight / group_weight
            adjusted[correlated_group] *= scale

    # Re-normalize
    return adjusted / adjusted.sum()
```

**Commodity example:**
- WTI and Brent are 95% correlated
- Don't allow 40% WTI + 40% Brent = 80% oil exposure
- Cap combined oil exposure at 50%

### Guardrail 7: Sector Exposure Caps

**Purpose:** Limit exposure to single commodity sectors.

```python
def apply_sector_limits(
    weights,
    arm_to_sector,
    max_sector_weight=0.50
):
    """
    Limit total allocation to any commodity sector.

    Args:
        weights: Proposed allocation
        arm_to_sector: Dict mapping arm index to sector
        max_sector_weight: Max weight per sector
    """
    sectors = set(arm_to_sector.values())
    adjusted = weights.copy()

    for sector in sectors:
        sector_arms = [i for i, s in arm_to_sector.items() if s == sector]
        sector_weight = adjusted[sector_arms].sum()

        if sector_weight > max_sector_weight:
            # Scale down sector proportionally
            scale = max_sector_weight / sector_weight
            adjusted[sector_arms] *= scale

    # Re-normalize
    return adjusted / adjusted.sum()
```

**Commodity sectors:**
- Energy: WTI, Brent, NatGas, Gasoline
- Metals: Gold, Silver, Copper, Platinum
- Grains: Corn, Soybeans, Wheat
- Softs: Coffee, Sugar, Cotton
- Livestock: Cattle, Hogs

**Example:**
- Don't allow 70% in Energy sector
- Cap at 50% even if all energy commodities individually perform well

### Guardrail 8: Inventory-Based Position Sizing

**Purpose:** Reduce exposure when inventory levels signal oversupply.

```python
def apply_inventory_guardrail(
    weights,
    inventory_percentiles,
    low_threshold=20,
    high_threshold=80
):
    """
    Adjust weights based on inventory levels.

    Args:
        weights: Proposed allocation
        inventory_percentiles: Array of inventory levels (0-100 percentile)
        low_threshold: Percentile below which to boost allocation
        high_threshold: Percentile above which to reduce allocation
    """
    adjusted = weights.copy()

    for i, inv_pct in enumerate(inventory_percentiles):
        if inv_pct < low_threshold:
            # Low inventory = bullish, allow full weight
            pass
        elif inv_pct > high_threshold:
            # High inventory = bearish, reduce weight
            adjusted[i] *= 0.7  # 30% haircut

    # Re-normalize
    return adjusted / adjusted.sum()
```

**Example:**
- WTI inventories at 90th percentile (oversupply)
- Reduce WTI allocation by 30% even if recent performance is good
- Prevents buying into building glut

## Complete Guardrail System

```python
class GuardrailSystem:
    """
    Complete commodity trading guardrail system.
    """
    def __init__(
        self,
        max_position=0.40,
        min_position=0.05,
        max_tilt_speed=0.15,
        vix_threshold=30,
        max_sector_weight=0.50
    ):
        self.max_pos = max_position
        self.min_pos = min_position
        self.max_speed = max_tilt_speed
        self.vix_threshold = vix_threshold
        self.max_sector = max_sector_weight
        self.last_weights = None

    def apply_all_guardrails(
        self,
        proposed_weights,
        core_weights,
        current_vix,
        arm_to_sector
    ):
        """
        Apply all guardrails in sequence.
        """
        weights = proposed_weights.copy()

        # 1. Volatility dampening
        if current_vix > self.vix_threshold:
            dampening = 0.5 if current_vix > 35 else 0.7
            weights = (
                dampening * weights +
                (1 - dampening) * core_weights
            )

        # 2. Position limits
        weights = np.clip(weights, 0, self.max_pos)
        weights = weights / weights.sum()

        # 3. Minimum allocation
        weights = np.maximum(weights, self.min_pos)
        weights = weights / weights.sum()

        # 4. Tilt speed limit (if we have prior weights)
        if self.last_weights is not None:
            change = weights - self.last_weights
            clipped_change = np.clip(
                change,
                -self.max_speed,
                self.max_speed
            )
            weights = self.last_weights + clipped_change
            weights = weights / weights.sum()

        # 5. Sector limits
        weights = self._apply_sector_limits(weights, arm_to_sector)

        # Store for next iteration
        self.last_weights = weights.copy()

        return weights

    def _apply_sector_limits(self, weights, arm_to_sector):
        """Apply sector exposure caps."""
        sectors = set(arm_to_sector.values())
        adjusted = weights.copy()

        for sector in sectors:
            sector_arms = [
                i for i, s in arm_to_sector.items() if s == sector
            ]
            sector_weight = adjusted[sector_arms].sum()

            if sector_weight > self.max_sector:
                scale = self.max_sector / sector_weight
                adjusted[sector_arms] *= scale

        return adjusted / adjusted.sum()
```

## When to Relax Guardrails

Guardrails should be tight by default, but can be loosened in specific scenarios:

**Scenario 1: High Conviction**
- You have strong fundamental thesis
- Multiple independent signals confirm
- Relax: Position limits (40% → 50%), but keep others

**Scenario 2: Low Volatility Regime**
- VIX < 15, stable markets
- Correlations are low
- Relax: Tilt speed limits (15% → 20%), but keep position limits

**Scenario 3: Long Time Horizon**
- Accumulating position over 12+ months
- Not worried about monthly volatility
- Relax: Volatility dampening, but keep position and sector limits

**Never relax:**
- Minimum allocation (always maintain exploration)
- Sector limits (concentration risk is structural)

## Common Pitfalls

### Pitfall 1: No Minimum Allocation
**What happens:** Bandit zeros out an arm after one bad week, arm recovers, you miss it.

**Example:** NatGas has one bad week (-10%). Bandit sets weight to 0%. Next 4 weeks: +5%, +8%, +6%, +7%. You earned 0% instead of 26%.

**Fix:** Set `min_allocation >= 5%`.

### Pitfall 2: No Tilt Speed Limit
**What happens:** Massive portfolio turnover, transaction costs eat gains.

**Example:** Bandit changes from 50% WTI to 50% Gold. Sell $50K WTI (10 bps cost), buy $50K Gold (10 bps cost) = $100 in costs for a $100K portfolio = 10 bps. Do this weekly = 5% annual cost.

**Fix:** Set `max_tilt_speed <= 15%`.

### Pitfall 3: Ignoring Correlations
**What happens:** Bandit concentrates in multiple highly correlated arms.

**Example:** 40% WTI + 40% Brent = 80% oil, despite being "two arms."

**Fix:** Add correlation guardrail or sector limits.

### Pitfall 4: Same Timescale for Core and Bandit
**What happens:** Bandit dominates and core allocation becomes meaningless.

**Example:** Both core and bandit rebalance weekly. Bandit overwhelms core. You lose strategic allocation benefits.

**Fix:** Core monthly/quarterly, bandit weekly.

## Connections

<div class="callout-danger">

**Danger:** Never deploy a bandit system without a kill switch and maximum allocation limits. An unconstrained bandit can allocate 100% of traffic/capital to a single arm, which creates catastrophic risk if the reward signal is noisy or delayed.

</div>


### Builds On
- **Module 1**: Thompson Sampling (what guardrails constrain)
- **Reward design**: Guardrails complement rewards

### Leads To
- **Production deployment**: These guardrails make bandits production-safe
- **Risk management**: Integration with broader risk frameworks

### Related Concepts
- **Portfolio constraints**: Modern Portfolio Theory with constraints
- **Risk budgeting**: Allocating risk, not just capital
- **Circuit breakers**: Market-wide trading halts (macro version)

## Practice Problems

### Problem 1: Design Guardrails
You're trading 8 commodities across 4 sectors with a $500K portfolio. Your risk tolerance:
- No single position > 30%
- No sector > 50%
- Comfortable with 5% weekly volatility
- Want smooth transitions (not whipsaw)

Design a complete guardrail system. Write the parameters and justify each choice.

### Problem 2: Guardrail Tradeoffs
You have two guardrail sets:
- **Set A**: max_pos=30%, min_pos=10%, max_speed=10%
- **Set B**: max_pos=50%, min_pos=2%, max_speed=25%

Run a simulation on historical commodity data. Which set:
- Achieves higher returns?
- Has lower volatility?
- Has lower turnover?
- Would you use in practice?

### Problem 3: Crisis Guardrails
March 2020: VIX spikes from 15 to 80 in two weeks. Your bandit wants to:
- Exit all positions (flight to cash)
- Concentrate 90% in Gold

Your guardrails currently:
- min_allocation = 5%
- max_position = 40%
- No volatility dampening

Should you relax any guardrails? Tighten any? Why?

---

**Next Steps:**
- Read [Regime-Aware Allocation](04_regime_aware_allocation.md) for contextual extensions
- Try [Two-Wallet Framework Notebook](../notebooks/01_two_wallet_framework.ipynb) to implement guardrails


---

## Cross-References

<a class="link-card" href="./01_accumulator_bandit_playbook.md">
  <div class="link-card-title">01 Accumulator Bandit Playbook</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_accumulator_bandit_playbook.md">
  <div class="link-card-title">01 Accumulator Bandit Playbook — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./02_reward_design_commodities.md">
  <div class="link-card-title">02 Reward Design Commodities</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_reward_design_commodities.md">
  <div class="link-card-title">02 Reward Design Commodities — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./04_regime_aware_allocation.md">
  <div class="link-card-title">04 Regime Aware Allocation</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./04_regime_aware_allocation.md">
  <div class="link-card-title">04 Regime Aware Allocation — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

