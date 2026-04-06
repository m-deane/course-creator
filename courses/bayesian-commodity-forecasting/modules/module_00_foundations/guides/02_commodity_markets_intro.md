# Introduction to Commodity Markets

> **Reading time:** ~7 min | **Module:** 0 — Foundations | **Prerequisites:** Basic probability, linear algebra


## In Brief

Commodity markets trade physical goods—energy, metals, and agricultural products. Understanding their structure is essential for building meaningful forecasting models.

<div class="callout-insight">

<strong>Insight:</strong> Commodity prices are driven by **physical fundamentals** (supply, demand, storage) overlaid with **financial dynamics** (speculation, hedging, risk premia). Effective forecasting models must capture both.

</div>


<div class="callout-key">

<strong>Key Concept Summary:</strong> Commodity markets trade physical goods—energy, metals, and agricultural products.

</div>

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

## Common Misconceptions

### "Commodity prices are random walks"
Not entirely. While short-term movements are hard to predict, fundamentals drive long-term levels. Mean reversion exists in spreads and relative values.

### "More data is always better"
For commodities, structural changes (shale revolution, OPEC policy) mean old data may not be relevant. Bayesian priors can downweight stale information.

### "Fundamentals are everything"
Financial flows and positioning matter too. Large speculative positions can move markets short-term. Models should incorporate both.

---

## Connections

**Builds on:**
- Basic economics (supply/demand)
- Financial markets understanding

**Leads to:**
- State space models for price dynamics (Module 3)
- Hierarchical models for related commodities (Module 4)
- Fundamentals integration (Module 8)

---


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "What Are Commodities?" and why it matters in practice.

2. Given a real-world scenario involving introduction to commodity markets, what would be your first three steps to apply the techniques from this guide?

</div>

## Further Reading

1. **Geman, H.** *Commodities and Commodity Derivatives* - Comprehensive reference
2. **Pirrong, C.** *Commodity Price Dynamics: A Structural Approach* - Economic foundations
3. **EIA.gov** - Official US energy data and analysis
4. **USDA.gov/oce/commodity/** - Agricultural market reports

---

*The commodity markets are where physical reality meets financial abstraction. Good models respect both.*

---

## Cross-References

<a class="link-card" href="./02_commodity_markets_intro_slides.md">
  <div class="link-card-title">Companion Slide Deck</div>
  <div class="link-card-description">Visual presentation covering the key concepts from this guide.</div>
</a>

<a class="link-card" href="../notebooks/01_environment_setup.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
