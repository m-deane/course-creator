---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Commodity Trading Bandits Cheatsheet

## Module 5 Quick Reference
### Multi-Armed Bandits for Commodity Trading

<!-- Speaker notes: This deck covers Commodity Trading Bandits Cheatsheet. Set the context for the audience and explain how this topic fits into the broader course on multi-armed bandits for commodity trading. -->
---

## 6-Step Accumulator Bandit Playbook

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#e8f5e9", "primaryBorderColor": "#4caf50", "primaryTextColor": "#212121", "secondaryColor": "#e3f2fd", "tertiaryColor": "#fff8e1", "lineColor": "#757575", "fontFamily": "Inter, sans-serif", "fontSize": "14px"}}}%%
flowchart LR
    S1["1. Core (80%)<br/>Equal-weight"] --> S2["2. Sleeve (20%)<br/>Thompson Sampling"]
    S2 --> S3["3. Arms (5-10)<br/>Commodities/sectors"]
    S3 --> S4["4. Execute<br/>Weekly rebalance"]
    S4 --> S5["5. Score<br/>Risk-adjusted reward"]
    S5 --> S6["6. Guardrails<br/>Position + speed limits"]
    S6 --> S4
```

<!-- Speaker notes: The diagram on 6-Step Accumulator Bandit Playbook illustrates the key relationships visually. Walk through the flow step by step, pointing out decision points and outcomes. Visual representations like this help students build mental models of the concepts. -->

<div class="callout-key">

Bandits learn AND earn simultaneously -- the core advantage over traditional A/B testing.

</div>

---

## Two-Wallet Framework

$$w_{\text{total}}(t) = 0.8 \cdot w_{\text{core}} + 0.2 \cdot w_{\text{bandit}}(t)$$

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
class TwoWalletBandit:
    def __init__(self, K, core_pct=0.8, bandit_pct=0.2):
        self.core = np.ones(K) / K  # Equal-weight
        self.bandit = ThompsonSamplingBandit(K)

    def get_weights(self):
        return (self.core_pct * self.core +
                self.bandit_pct * self.bandit.get_allocation())
```

</div>

<!-- Speaker notes: This slide connects theory to implementation for Two-Wallet Framework. Start with the mathematical formulation, then show how each term maps to a line of code. This bridge between theory and practice is one of the most valuable aspects of the course. -->

<div class="callout-insight">

**Insight:** The exploration-exploitation tradeoff is not a fixed ratio -- it should adapt as uncertainty decreases over time.

</div>

---

## Reward Function Comparison

| Reward | Formula | Use When |
|--------|---------|----------|
| Raw Returns | $r_t$ | **NEVER** |
| Risk-Adjusted | $r/\sigma - \lambda \cdot DD$ | General accumulation |
| Regret-Relative | $r - r_{\text{best}}$ | Relative performance |
| Stability | $r - \lambda \cdot \text{turnover}$ | Cost-sensitive |
| Thesis-Aligned | $r - \lambda \cdot \|w - w_s\|$ | Strategic overlay |
| Multi-Objective | Weighted combination | Explicit tradeoffs |

> **Your reward IS your strategy.**

<!-- Speaker notes: This comparison table on Reward Function Comparison is a key reference. Walk through each row, highlighting the most important distinctions. Students should understand when to use each option based on the criteria shown. -->

<div class="callout-warning">

**Warning:** Non-stationary reward distributions violate bandit assumptions. Always implement change detection in production systems.

</div>

---

## Guardrail Parameters

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#e8f5e9", "primaryBorderColor": "#4caf50", "primaryTextColor": "#212121", "secondaryColor": "#e3f2fd", "tertiaryColor": "#fff8e1", "lineColor": "#757575", "fontFamily": "Inter, sans-serif", "fontSize": "14px"}}}%%
flowchart TD
    subgraph Conservative["Conservative"]
        C1["Max: 30%, Min: 10%"]
        C2["Speed: 10%, Core: 85%"]
        C3["VIX threshold: 25"]
    end
    subgraph Moderate["Moderate"]
        M1["Max: 40%, Min: 5%"]
        M2["Speed: 15%, Core: 80%"]
        M3["VIX threshold: 30"]
    end
    subgraph Aggressive["Aggressive"]
        A1["Max: 50%, Min: 2%"]
        A2["Speed: 25%, Core: 70%"]
        A3["VIX threshold: 35"]
    end
```

<!-- Speaker notes: The diagram on Guardrail Parameters illustrates the key relationships visually. Walk through the flow step by step, pointing out decision points and outcomes. Visual representations like this help students build mental models of the concepts. -->

<div class="callout-info">

**Info:** The regret of the best bandit algorithms grows logarithmically with time, compared to linearly for A/B testing.

</div>

---

## Common Arms

<div class="columns">
<div>

### Broad Sectors (5 arms)
<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
arms = ['Energy', 'Metals',
        'Grains', 'Softs',
        'Livestock']
```

</div>

### Strategy Factors
```python
arms = ['Momentum',
        'Mean_Reversion',
        'Carry', 'Seasonality']
```

</div>
<div>

### Granular (8-10 arms)
```python
arms = ['WTI', 'NatGas',
        'Gold', 'Copper',
        'Corn', 'Soybeans',
        'Coffee', 'Cattle']
```

</div>
</div>

<!-- Speaker notes: This code example for Common Arms is production-ready. Walk through the implementation, noting any important design patterns or potential modifications for different use cases. -->
---

## Regime Features Quick Reference

```python
# Volatility: Low (<15%), Med (15-25%), High (>25%)
vol = returns.rolling(20).std() * np.sqrt(252)

# Term Structure: Contango (>0) vs Backwardation (<0)
ts_slope = (back_month - front_month) / front_month

# Trend: Up (>10%), Neutral (-5 to 5%), Down (<-5%)
trend = (ma_20 - ma_50) / ma_50

# Risk Sentiment: On (>0) vs Off (<0)
sentiment = sp500.rolling(5).mean() - (vix - 20) / 100
```

<!-- Speaker notes: This code example for Regime Features Quick Reference is production-ready. Walk through the implementation, noting any important design patterns or potential modifications for different use cases. -->
---

## Decision Flowchart

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#e8f5e9", "primaryBorderColor": "#4caf50", "primaryTextColor": "#212121", "secondaryColor": "#e3f2fd", "tertiaryColor": "#fff8e1", "lineColor": "#757575", "fontFamily": "Inter, sans-serif", "fontSize": "14px"}}}%%
flowchart TD
    Q1{"Goal?"} -->|Steady accumulation| R1["Stability-weighted reward"]
    Q1 -->|Max returns| R2["Risk-adjusted reward"]
    Q1 -->|Beat benchmark| R3["Regret-relative reward"]

    R1 & R2 & R3 --> Q2{"Capital?"}
    Q2 -->|"< $100K"| A1["Broad sectors (5 arms)"]
    Q2 -->|"$100K-$1M"| A2["Granular (8-10 arms)"]

    A1 & A2 --> Q3{"Risk tolerance?"}
    Q3 -->|Conservative| G1["85/15, tight guardrails"]
    Q3 -->|Moderate| G2["80/20, moderate"]
    Q3 -->|Aggressive| G3["70/30, loose"]
```

<!-- Speaker notes: The diagram on Decision Flowchart illustrates the key relationships visually. Walk through the flow step by step, pointing out decision points and outcomes. Visual representations like this help students build mental models of the concepts. -->
---

## Common Pitfalls Checklist

- [ ] Reward = raw returns? -> Change to risk-adjusted
- [ ] No minimum allocation? -> Add 5% min per arm
- [ ] No position limits? -> Add 40% max per arm
- [ ] No tilt speed limit? -> Add 15% max change
- [ ] Pure bandit (no core)? -> Add 60-80% core
- [ ] Too many arms (>15)? -> Reduce to 5-10
- [ ] Ignoring transaction costs? -> Add turnover penalty
- [ ] No volatility dampening? -> Add VIX-based adjustment

<!-- Speaker notes: Walk through Common Pitfalls Checklist carefully. Emphasize why this mistake is common and how to recognize it in practice. The commodity trading example makes it concrete -- ask if anyone has encountered this in their own work. -->
---

## Quick Debugging Guide

| Symptom | Cause | Fix |
|---------|-------|-----|
| Always equal-weight | Bandit % too low | Increase to 20% |
| Extreme concentration | No position limits | Add max_pos=0.40 |
| Constant churning | No speed limit | Add max_speed=0.15 |
| Underperforms benchmark | Wrong reward | Match reward to goal |
| High transaction costs | Too much turnover | Add stability penalty |
| Ignores regime changes | Non-contextual | Add regime features |

<!-- Speaker notes: This comparison table on Quick Debugging Guide is a key reference. Walk through each row, highlighting the most important distinctions. Students should understand when to use each option based on the criteria shown. -->
---

## Visual Summary

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#e8f5e9", "primaryBorderColor": "#4caf50", "primaryTextColor": "#212121", "secondaryColor": "#e3f2fd", "tertiaryColor": "#fff8e1", "lineColor": "#757575", "fontFamily": "Inter, sans-serif", "fontSize": "14px"}}}%%
flowchart TD
    M5["Module 5: Commodity Trading"] --> TW_v["Two-Wallet Framework"]
    M5 --> RD_v["Reward Design"]
    M5 --> GR_v["Guardrails"]
    M5 --> RA_v["Regime-Aware"]

    TW_v --> |"80/20 core/bandit"| Stable["Stability + adaptiveness"]
    RD_v --> |"Risk-adjusted, not raw"| Smart["Smart optimization"]
    GR_v --> |"Position + speed limits"| Safe_v["Safety constraints"]
    RA_v --> |"Context-dependent"| Adaptive["Regime adaptation"]

    Stable & Smart & Safe_v & Adaptive --> Production["Production-Ready Allocator"]
```

<!-- Speaker notes: This visual summary captures the key relationships from the entire deck. Walk through each branch of the diagram, connecting back to the main concepts covered. This slide works well as a reference -- encourage students to screenshot it for later review. -->