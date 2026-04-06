# Storage Theory: Economic Foundations of Commodity Pricing

> **Reading time:** ~7 min | **Module:** 8 — Fundamentals Integration | **Prerequisites:** Modules 1-7


## In Brief

Storage theory explains how inventory levels, storage costs, and the convenience of holding physical commodities determine the relationship between spot and futures prices. Understanding these economics is essential for building fundamentals-based forecasting models.

<div class="callout-insight">

<strong>Insight:</strong> **Inventories are the bridge between supply and demand across time.** When supply exceeds demand, inventories build and futures trade at a premium (contango). When demand exceeds supply, inventories draw and futures trade at a discount (backwardation). This is the physical foundation of commodity prices.

</div>


<div class="callout-key">

<strong>Key Concept Summary:</strong> Storage theory explains how inventory levels, storage costs, and the convenience of holding physical commodities determine the relationship between spot and futures prices.

</div>

---

## Formal Framework

### The Cost of Carry Model

The theoretical relationship between spot ($S$) and futures ($F_T$) prices:
<div class="callout-key">

<strong>Key Point:</strong> The theoretical relationship between spot ($S$) and futures ($F_T$) prices:

</div>


$$F_T = S \cdot e^{(r + u - y)T}$$

Where:
- $S$: Current spot price
- $F_T$: Futures price for delivery at time $T$
- $r$: Risk-free interest rate
- $u$: Storage cost (as percentage of price)
- $y$: Convenience yield
- $T$: Time to delivery

### Components Explained

**Interest Rate ($r$):**
- Opportunity cost of capital tied up in inventory
- Higher rates → higher futures prices
- Currently ~5% annualized

**Storage Costs ($u$):**
- Physical storage, insurance, handling
- Varies by commodity and location
- Crude oil: ~$0.50/barrel/month at Cushing
- Natural gas: ~$0.02/MMBtu/month

**Convenience Yield ($y$):**
- Benefit of holding physical commodity vs. futures
- High when supply is tight (need physical now)
- Low when supply is ample (no urgency)

> 💡 **Key variable:** Convenience yield $y$ is the single most important input for fundamentals-based forecasting. When you cannot observe it directly, it must be inferred from the spot-futures spread — making it a latent variable that Bayesian methods are ideally suited to estimate.

---

## Contango vs. Backwardation

### Contango ($F > S$)

Futures trade above spot: $y < r + u$

**Conditions:**
- Ample inventories
- Low convenience yield
- Markets expect stable/higher prices

**Economic meaning:**
- Storage is profitable (buy spot, sell futures, store)
- "Carry trade" incentivizes inventory building

### Backwardation ($F < S$)

Futures trade below spot: $y > r + u$

**Conditions:**
- Tight inventories
- High convenience yield
- Immediate demand exceeds supply

**Economic meaning:**
- Storage is unprofitable
- Markets signal "we need it NOW"

---

## The Convenience Yield

### Definition

The convenience yield is the implicit benefit of holding physical commodity rather than a futures contract.

$$y = r + u - \frac{1}{T}\ln\left(\frac{F_T}{S}\right)$$

### What Drives Convenience Yield?

1. **Inventory Levels:** Lower inventories → higher convenience yield
2. **Production Uncertainty:** More uncertainty → higher convenience yield
3. **Demand Spikes:** Unexpected demand → higher convenience yield
4. **Optionality:** Physical commodity can be used/sold anytime

### Empirical Relationship

Convenience yield is typically modeled as a function of inventory:

$$y_t = \alpha + \beta \cdot I_t + \epsilon_t$$

Where $I_t$ is inventory relative to normal (days of supply or z-score).

**Typical finding:** $\beta < 0$ (higher inventory → lower convenience yield)

---

## Inventory-Price Relationships

### The Fundamental Relationship

```
Inventories (I)
    │
    │ High I → Low convenience yield → Contango
    │          Bearish price pressure
    │
    │ Low I  → High convenience yield → Backwardation
    │          Bullish price pressure
    ▼
Price Level and Term Structure
```

### Empirical Model

$$\Delta P_t = \alpha + \beta_1 (I_t - I^*) + \beta_2 \Delta I_t + \epsilon_t$$

Where:
- $I^*$: "Normal" inventory level (5-year average)
- $I_t - I^*$: Inventory deviation from normal
- $\Delta I_t$: Inventory change (surprise)

**Expected signs:**
- $\beta_1 < 0$: High inventory levels → lower prices
- $\beta_2 < 0$: Inventory builds → price declines

### Days of Supply

A useful normalization:

$$\text{Days of Supply} = \frac{\text{Inventory}}{\text{Daily Demand}}$$

Standardizes across commodities with different scales.

---

## Using Storage Theory in Bayesian Models

### Prior Information from Theory

Storage theory suggests:

1. **Inventory coefficient is negative:** Prior: $\beta_I \sim \mathcal{N}(-0.5, 0.5)$
2. **Convenience yield is mean-reverting:** Prior: $y_t$ follows AR(1)
3. **Term structure has bounds:** Extreme contango is unsustainable (arbitrage)

### Model Specification


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import pymc as pm

with pm.Model() as storage_model:
    # Prior informed by storage theory: inventory effect is negative
    beta_inventory = pm.Normal('beta_inventory', mu=-0.5, sigma=0.5)

    # Control variables
    beta_production = pm.Normal('beta_production', mu=0, sigma=0.5)

    # Intercept (base price level)
    alpha = pm.Normal('alpha', mu=80, sigma=10)

    # Noise
    sigma = pm.HalfNormal('sigma', sigma=5)

    # Likelihood
    mu = alpha + beta_inventory * inventory_zscore + beta_production * production_zscore
    price = pm.Normal('price', mu=mu, sigma=sigma, observed=observed_prices)
```

</div>
</div>

### Dynamic Coefficient Model

The relationship may vary over time (regime-dependent):

$$\Delta P_t = \alpha_{r(t)} + \beta_{r(t)} \cdot I_t + \epsilon_t$$

Where $r(t) \in \{1, 2\}$ indicates the regime.

---

## Practical Applications

### 1. Fair Value Estimation

Use inventory levels to estimate "fair" price:

$$\hat{P} = \alpha + \beta \cdot (I - I^*)$$

Deviation from fair value may signal mean reversion opportunity.

### 2. Term Structure Trading

- **Extreme contango:** Expect convergence (short front, long back)
- **Extreme backwardation:** Expect convergence (long front, short back)

### 3. Inventory Surprise Trading

EIA releases inventory data weekly. Surprise = Actual - Consensus.

$$\text{Return}_{\text{post-release}} = \gamma \cdot \text{Surprise} + \epsilon$$

### 4. Forecasting Inventory Changes

Model inventory as:

$$\Delta I_t = \text{Production}_t - \text{Consumption}_t + \text{Net Imports}_t$$

Forecast components separately, aggregate to forecast inventory.

---

## Code Example: Convenience Yield Calculation


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import numpy as np
import pandas as pd

def calculate_convenience_yield(spot, futures, T, r=0.05, u=0.02):
    """
    Calculate implied convenience yield from spot and futures prices.

    Parameters
    ----------
    spot : float or array
        Spot price
    futures : float or array
        Futures price for delivery at time T
    T : float
        Time to delivery in years
    r : float
        Risk-free rate (annualized)
    u : float
        Storage cost (annualized, as proportion)

    Returns
    -------
    float or array
        Implied convenience yield (annualized)
    """
    # From F = S * exp((r + u - y) * T)
    # Solve for y: y = r + u - (1/T) * ln(F/S)
    convenience_yield = r + u - (1/T) * np.log(futures / spot)
    return convenience_yield


def calculate_inventory_zscore(inventory, lookback=260):
    """
    Calculate z-score of inventory relative to rolling history.

    Parameters
    ----------
    inventory : pd.Series
        Inventory time series
    lookback : int
        Lookback window in periods

    Returns
    -------
    pd.Series
        Z-score of inventory
    """
    rolling_mean = inventory.rolling(lookback).mean()
    rolling_std = inventory.rolling(lookback).std()
    zscore = (inventory - rolling_mean) / rolling_std
    return zscore


# Example usage
spot_price = 75.0
futures_1m = 76.0  # 1-month futures
T = 1/12  # 1 month in years

cy = calculate_convenience_yield(spot_price, futures_1m, T)
print(f"Implied convenience yield: {cy:.2%}")

if cy < 0:
    print("Market in CONTANGO (futures > spot)")
else:
    print("Market in BACKWARDATION (spot > futures)")
```

</div>

---

## Common Pitfalls

### 1. Ignoring Storage Constraints

Physical storage capacity limits how much contango is possible. Extreme contango can collapse when storage fills.

### 2. Wrong Inventory Measure

- Total commercial inventory vs. specific location (Cushing)
- Days of supply vs. absolute levels
- Year-over-year change vs. level

### 3. Look-Ahead Bias

Inventory data has release delays. Don't use Friday's inventory data for Monday's forecast!

### 4. Regime Dependence

The inventory-price relationship varies:
- Strong in tight markets (backwardation)
- Weak in glutted markets (contango)

---

## Connections

**Builds on:**
- Module 2: Commodity data and fundamentals
- Module 7: Regime-dependent relationships

**Leads to:**
- Capstone: Full fundamentals-integrated forecasting system
- Trading: Fair value and mean reversion signals

---

## Practice Problems

1. Oil is trading at $80/bbl spot and $82/bbl for 3-month futures. With $r = 5\%$ and $u = 2\%$ annually, what is the implied convenience yield?

2. If crude inventories are 50 million barrels above the 5-year average and the estimated coefficient is -$0.10/million barrels, what is the estimated price impact?

3. Why might the inventory-price relationship be stronger in backwardated markets?

---


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Formal Framework" and why it matters in practice.

2. Given a real-world scenario involving storage theory: economic foundations of commodity pricing, what would be your first three steps to apply the techniques from this guide?


## Further Reading

1. **Working, H.** "The Theory of Price of Storage" (1949) — Original theory
2. **Fama & French** "Commodity Futures Prices" (1987) — Empirical evidence
3. **Geman, H.** *Commodities and Commodity Derivatives* — Modern treatment

---

*"Storage theory isn't just academic—it's how physical traders actually price commodities. Models that ignore it are missing the economic fundamentals."*

---

## Cross-References

<a class="link-card" href="./01_storage_theory_slides.md">
  <div class="link-card-title">Companion Slide Deck</div>
  <div class="link-card-description">Visual presentation covering the key concepts from this guide.</div>
</a>

<a class="link-card" href="../notebooks/01_storage_theory.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
