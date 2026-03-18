# Introduction to Commodity Markets

## In Brief

Commodity markets trade physical goods—energy, metals, and agricultural products. Understanding their structure is essential for building meaningful forecasting models.

> 💡 **Key Insight:** Commodity prices are driven by **physical fundamentals** (supply, demand, storage) overlaid with **financial dynamics** (speculation, hedging, risk premia). Effective forecasting models must capture both.

---

## Formal Definition

A **commodity** is a standardized, fungible physical good that can be stored, transported, and graded to a common specification, making it interchangeable with other units of the same quality. Formally, commodity prices in competitive markets are governed by:

**Cost of Carry (no-arbitrage):**
$$F_{t,T} = S_t \cdot e^{(r_t + u_t - y_t)(T - t)}$$

where $S_t$ is the spot price, $F_{t,T}$ is the futures price for delivery at $T$, $r_t$ is the risk-free rate, $u_t$ is the proportional storage cost, and $y_t$ is the **convenience yield** — the implied benefit of holding physical inventory.

**Supply-demand balance (storage equation):**
$$I_t = I_{t-1} + \text{Production}_t - \text{Consumption}_t$$

where $I_t$ is ending inventory. This single equation links the physical balance to price dynamics: when $I_t$ falls below normal levels, convenience yield rises and backwardation deepens.

---

## Intuitive Explanation

Imagine crude oil as water flowing through pipes. Production wells are the source, refineries are the drain, and storage tanks at Cushing, Oklahoma are the reservoir. When the reservoir is full (high inventory), there is no urgency to buy today — you can always buy from storage next month. Futures trade at a premium to spot (contango). When the reservoir is nearly empty (low inventory), buyers pay up to secure physical supply now. Futures trade at a discount to spot (backwardation).

Bayesian forecasting models this reservoir level as a latent state evolving through time. Prior knowledge about seasonal fills and draws (summer build, winter draw for natural gas) enters as an informative prior. Weekly EIA data updates the posterior on inventory, which then flows through to price forecasts.

---

## Code Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Simulate the contango/backwardation relationship
# Cost of carry: F = S * exp((r + u - y) * T)

def futures_price(spot, r, storage_cost, convenience_yield, T):
    """
    Theoretical futures price from cost of carry.

    Parameters
    ----------
    spot : float
        Current spot price ($/barrel for crude)
    r : float
        Risk-free rate (annualized)
    storage_cost : float
        Storage cost as fraction of price per year
    convenience_yield : float
        Convenience yield (annualized)
    T : float
        Time to expiration in years

    Returns
    -------
    float
        Theoretical futures price
    """
    return spot * np.exp((r + storage_cost - convenience_yield) * T)

# Example: WTI crude at $80/barrel
spot = 80.0
r = 0.05          # 5% risk-free rate
storage = 0.04    # 4% annual storage cost ($3.20/barrel/year)

# Maturities: 1 to 24 months
T_values = np.arange(1, 25) / 12

# Scenario 1: Contango (ample supply, low convenience yield)
y_contango = 0.02  # convenience yield below cost of carry
futures_contango = [futures_price(spot, r, storage, y_contango, T) for T in T_values]

# Scenario 2: Backwardation (tight supply, high convenience yield)
y_backwardation = 0.12  # convenience yield above cost of carry
futures_backwardation = [futures_price(spot, r, storage, y_backwardation, T) for T in T_values]

months = np.arange(1, 25)

fig, ax = plt.subplots(figsize=(10, 6))
ax.axhline(spot, color='black', linestyle='--', lw=1.5, label=f'Spot: ${spot}')
ax.plot(months, futures_contango, 'b-o', markersize=4, label='Contango (y=2%)')
ax.plot(months, futures_backwardation, 'r-o', markersize=4, label='Backwardation (y=12%)')
ax.set_xlabel('Months to Expiration')
ax.set_ylabel('Price ($/barrel)')
ax.set_title('Crude Oil Term Structure: Contango vs. Backwardation')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Contango: F(12M) - Spot =", f"${futures_contango[11] - spot:.2f}/bbl")
print("Backwardation: F(12M) - Spot =", f"${futures_backwardation[11] - spot:.2f}/bbl")
```

---

## 1. What Are Commodities?

### Definition

**Commodities** are standardized, fungible goods traded on exchanges. Key categories:

| Category | Examples | Key Drivers |
|----------|----------|-------------|
| **Energy** | Crude oil (WTI, Brent), Natural gas, Gasoline, Heating oil | OPEC policy, inventories, weather, economic growth |
| **Metals** | Gold, Silver, Copper, Aluminum, Platinum | Industrial demand, monetary policy, mining output |
| **Agriculture** | Corn, Wheat, Soybeans, Coffee, Sugar, Cotton | Weather, planting/harvest, trade policy, biofuels |
| **Livestock** | Cattle, Hogs, Lean hogs | Feed costs, disease, consumer demand |
| **Softs** | Cocoa, Coffee, Sugar, Orange juice | Weather in producing regions, disease |

### Why Trade Commodities?

1. **Hedgers:** Producers and consumers lock in prices (airline hedging fuel costs)
2. **Speculators:** Profit from price movements
3. **Arbitrageurs:** Exploit pricing inefficiencies
4. **Index investors:** Diversification and inflation protection

---

## 2. Market Structure

### Spot vs. Futures Markets

**Spot Market:** Immediate delivery at current price
- Physical transaction
- Location-specific pricing

**Futures Market:** Agreement to buy/sell at future date at agreed price
- Standardized contracts (size, quality, delivery)
- Margin-based (leverage)
- Most positions closed before delivery

### Key Exchanges

| Exchange | Location | Commodities |
|----------|----------|-------------|
| CME Group (NYMEX, COMEX, CBOT) | Chicago | Energy, Metals, Grains |
| ICE | Atlanta/London | Energy, Softs |
| LME | London | Base Metals |

### Contract Specifications

Example: WTI Crude Oil (NYMEX CL)
- **Size:** 1,000 barrels
- **Quote:** USD per barrel
- **Tick size:** $0.01 = $10 per contract
- **Delivery:** Cushing, Oklahoma
- **Months:** All calendar months

---

## 3. Fundamentals That Drive Prices

### Supply Factors

**Production:**
- OPEC decisions (oil)
- Mine output and disruptions (metals)
- Planting decisions and acreage (agriculture)
- Weather during growing season

**Inventories:**
- Storage levels relative to historical norms
- Days of supply coverage
- Strategic reserves (SPR for oil)

### Demand Factors

**Consumption:**
- Economic growth (GDP, industrial production)
- Seasonal patterns (heating oil in winter, gasoline in summer)
- Substitution effects (nat gas vs. coal)

**Emerging market growth:**
- China's infrastructure and manufacturing
- India's energy consumption

### Storage Economics

**Cost of Carry:** $F = S \cdot e^{(r + u - y)T}$

Where:
- $F$ = Futures price
- $S$ = Spot price
- $r$ = Risk-free rate
- $u$ = Storage costs
- $y$ = Convenience yield (benefit of holding physical)
- $T$ = Time to expiration

**Contango:** $F > S$ (futures above spot)
- Normal when storage is cheap and inventory is high
- Incentivizes storage

**Backwardation:** $F < S$ (futures below spot)
- Occurs when supply is tight
- High convenience yield (need physical now)

---

## 4. Seasonality in Commodities

### Agricultural Seasonality

**Corn (Northern Hemisphere):**
- April-May: Planting
- June-July: Pollination (critical weather period)
- September-October: Harvest

Price pattern: Uncertainty peaks in summer, harvest pressure in fall.

**Wheat:**
- Winter wheat: Planted fall, harvested early summer
- Spring wheat: Planted spring, harvested late summer

### Energy Seasonality

**Natural Gas:**
- Summer: Cooling demand (electricity for AC)
- Winter: Heating demand (direct consumption)
- Shoulder seasons: Lower demand, injection season

**Gasoline:**
- Summer: Driving season demand
- Spring: Refinery maintenance, summer blend transition

### Metals Seasonality

Generally less seasonal, but:
- Construction-related metals (copper, steel): Spring pickup
- Jewelry demand (gold): Q4 holiday season

---

## 5. Key Data Sources

### Energy (EIA - Energy Information Administration)

**Weekly Petroleum Status Report:**
- Crude oil inventories
- Refinery utilization
- Product supplied (demand proxy)
- Import/export data

**Natural Gas Weekly Update:**
- Storage levels
- Injection/withdrawal
- Henry Hub prices

### Agriculture (USDA - Department of Agriculture)

**WASDE (World Agricultural Supply and Demand Estimates):**
- Monthly report
- Production, consumption, ending stocks
- Global balance sheets

**Crop Progress Reports:**
- Weekly during growing season
- Planting/harvest progress
- Crop condition ratings

### Metals (LME, COMEX)

**LME Warehouse Stocks:**
- Daily updates
- On-warrant vs. cancelled warrants

**COMEX Inventory:**
- Registered vs. eligible gold/silver

### Positioning Data (CFTC)

**Commitments of Traders (COT):**
- Weekly report (Tuesday data, Friday release)
- Positions by trader category
- Commercial (hedgers) vs. Non-commercial (speculators)

---

## 6. Price Dynamics and Stylized Facts

### Returns Characteristics

**Fat Tails:**
- Commodity returns have heavier tails than Normal
- Extreme moves more frequent than Gaussian predicts
- Implication: Use t-distributions or similar

**Volatility Clustering:**
- High volatility tends to follow high volatility
- GARCH effects prominent
- Implication: Stochastic volatility models

**Mean Reversion (in some markets):**
- Agricultural spreads tend to revert to cost of carry
- Convenience yield fluctuates around equilibrium
- Implication: State space models with mean-reverting dynamics

**Regime Switching:**
- Commodity super-cycles (multi-year bull/bear)
- Structural breaks (OPEC policy changes, shale revolution)
- Implication: Hidden Markov Models

### Term Structure Dynamics

**Parallel shifts:** Entire curve moves up/down
**Slope changes:** Front vs. back spread widens/narrows
**Curvature:** Roll dynamics, expiration effects

---

## 7. Why Bayesian Methods for Commodities?

### Handling Uncertainty

Commodity forecasts must quantify uncertainty:
- Trading decisions need confidence intervals
- Risk management requires tail risk estimates
- Bayesian methods provide full predictive distributions

### Incorporating Prior Knowledge

Domain expertise can be encoded:
- Storage costs bound futures-spot spreads
- Seasonality patterns are known
- Fundamental relationships are established

### Sparse Data Challenges

Many fundamental series are infrequent:
- Monthly WASDE reports
- Weekly inventory data
- Bayesian methods handle small samples gracefully

### Hierarchical Structure

Related commodities share information:
- Crude oil prices affect product cracks
- Grain prices are interconnected
- Hierarchical Bayes enables learning across markets

---

## 8. Course Data Focus

Throughout this course, we will primarily work with:

**Energy:**
- WTI Crude Oil (front month and term structure)
- Natural Gas (Henry Hub)
- EIA inventory and production data

**Agriculture:**
- Corn, Soybeans, Wheat
- USDA balance sheet data
- Crop progress indicators

**Metals:**
- Copper (industrial bellwether)
- Gold (safe haven, monetary metal)
- LME inventory data

---

## Common Pitfalls

### "Commodity prices are random walks"
Not entirely. While short-term movements are hard to predict, fundamentals drive long-term levels. Mean reversion exists in spreads and relative values.

### "More data is always better"
For commodities, structural changes (shale revolution, OPEC policy) mean old data may not be relevant. Bayesian priors can downweight stale information.

### "Fundamentals are everything"
Financial flows and positioning matter too. Large speculative positions can move markets short-term. Models should incorporate both.

---

## Practice Problems

1. WTI crude is at $85/bbl. The 12-month futures contract trades at $88/bbl. Assuming a 5% risk-free rate and 4% annual storage cost, what is the implied convenience yield? Is the market in contango or backwardation?

2. Natural gas typically draws from storage during November through March and injects April through October. How would you encode this seasonal pattern as an informative prior in a Bayesian model?

3. The CFTC COT report shows managed money (speculators) hold a record net long position in corn. Does this change your fundamental supply/demand forecast? How would you incorporate positioning data into a Bayesian model?

---

## Connections

**Builds on:**
- Basic economics (supply/demand)
- Financial markets understanding

**Leads to:**
- Module 3: State space models for price dynamics
- Module 4: Hierarchical models for related commodities
- Module 8: Bayesian models integrating storage theory and fundamentals

---

## Further Reading

1. **Geman, H.** *Commodities and Commodity Derivatives* - Comprehensive reference
2. **Pirrong, C.** *Commodity Price Dynamics: A Structural Approach* - Economic foundations
3. **EIA.gov** - Official US energy data and analysis
4. **USDA.gov/oce/commodity/** - Agricultural market reports

---

*The commodity markets are where physical reality meets financial abstraction. Good models respect both.*
