# Module 2 Quiz: Commodity Data and Seasonality Analysis

**Course:** Bayesian Commodity Forecasting
**Module:** 02 - Commodity Data Sources and Feature Engineering
**Time Limit:** 30 minutes
**Total Points:** 100 points
**Instructions:** Select the best answer for each question. Show your work for mathematical problems.

---

## Section A: Data Sources and Quality (30 points)

### Question 1 (5 points)
The EIA Weekly Petroleum Status Report is released on Wednesday at 10:30 AM ET. You are backtesting a trading strategy using historical crude oil inventory data. On Tuesday, December 15, 2023, what is the most recent inventory data you should use in your model to avoid look-ahead bias?

A) Inventory data through Friday, December 12, 2023
B) Inventory data through Tuesday, December 15, 2023
C) Inventory data through Friday, December 5, 2023
D) Inventory data through Wednesday, December 9, 2023

**Answer:** C

**Explanation:**
- EIA reports are released Wednesday with data through the prior Friday
- On Tuesday Dec 15, the most recent report would have been Wednesday Dec 10
- That report contains data through Friday Dec 5
- Using any more recent data would be look-ahead bias since it wasn't available on Dec 15
- **Common mistake:** Choosing A assumes the current week's report is already available
- **Key insight:** Always account for reporting lag when backtesting

---

### Question 2 (6 points)
The USDA WASDE (World Agricultural Supply and Demand Estimates) report contains production forecasts and ending stocks estimates. Why is the "ending stocks" metric particularly important for commodity price forecasting?

A) It directly measures current consumption rates
B) It represents the buffer available to meet unexpected demand shocks
C) It is the most accurately measured variable in the report
D) It determines export quotas for the following year

**Answer:** B

**Explanation:**
- Ending stocks = Beginning stocks + Production - Consumption
- Low ending stocks → tight supply → price sensitive to shocks → higher volatility
- This is the **stocks-to-use ratio** concept: Ending Stocks / Annual Use
- Historical pattern: When stocks/use < 15%, prices become highly volatile
- **Example:** 2012 corn: Drought reduced production, stocks fell to 10-year low, prices spiked 50%
- Options A and D are incorrect: Ending stocks don't measure current consumption or determine quotas
- Option C is misleading: Ending stocks are estimates, subject to revision

---

### Question 3 (7 points)
You are building a forecasting model for crude oil prices using EIA weekly data. Over the past year, you notice three instances where reported crude oil inventories were revised by more than 5 million barrels in subsequent weeks. Which data pipeline strategy best addresses this issue?

A) Use only the initial release values to maintain consistency
B) Re-download all historical data before each model run to capture revisions
C) Exclude revised data points from the training set
D) Apply a moving average to smooth out revision effects

**Answer:** B

**Explanation:**
- **Real-time data problem:** Initial releases contain estimates; revisions provide better accuracy
- For model training: Use revised data (best estimate of true values)
- For backtesting: Use vintage data (what was available at decision time)
- Best practice: Maintain two datasets
  - `data_revised/`: Latest revisions (for training)
  - `data_vintage/`: Historical snapshots (for backtesting)
- **Option A is wrong:** Using unrevised data trains model on measurement errors
- **Option C is wrong:** Losing data reduces sample size unnecessarily
- **Option D is wrong:** Smoothing masks important information about inventory changes
- **Academic reference:** Croushore & Stark (2001) on real-time data vintages

---

### Question 4 (6 points)
The CFTC Commitments of Traders (COT) report categorizes market participants into Commercial, Non-Commercial, and Non-Reportable traders. A forecasting model finds that extreme Non-Commercial net long positions (top 10th percentile) are associated with subsequent price declines. This suggests:

A) Non-Commercial traders have superior information and should be followed
B) Non-Commercial positioning may reflect overcrowded trades vulnerable to reversal
C) The COT report should not be used for forecasting due to its contrarian nature
D) Commercial hedgers are manipulating the market

**Answer:** B

**Explanation:**
- **Non-Commercial = Speculators** (hedge funds, CTAs, prop traders)
- Extreme positioning → contrarian indicator → mean reversion likely
- **Mechanism:**
  - Crowded long positions → few buyers left → vulnerable to profit-taking
  - Any negative news → rush to exit → amplified price decline
- **Empirical evidence:** Extreme speculator positions often near price peaks/troughs
- **Option A is wrong:** Superior info would lead to positions predicting price continuation, not reversal
- **Option C is wrong:** Contrarian nature is valuable for risk management
- **Option D is wrong:** No evidence of manipulation; reflects market dynamics
- **Related concept:** Sentiment indicators, positioning risk

---

### Question 5 (6 points)
When retrieving commodity price data from Yahoo Finance using the `yfinance` library for continuous futures contracts (e.g., `CL=F` for WTI crude), which data quality issue requires careful handling?

A) Prices are always delayed by 15 minutes
B) Contract roll dates create artificial price jumps in the continuous series
C) Weekend data is missing
D) All prices are quoted in local currency rather than USD

**Answer:** B

**Explanation:**
- **Roll problem:** Continuous futures switch from near to next contract monthly
- **Price gap:** Near contract (expiring) ≠ next contract (longer maturity)
- **Example:** WTI May contract at $75, June contract at $77 → $2 jump at roll
- **Solutions:**
  1. **Panama Canal (constant maturity):** Interpolate between contracts
  2. **Return-based splicing:** Adjust historical prices for roll gap
  3. **Ratio splicing:** Scale by price ratio at roll
- **Option A:** While some data may be delayed, yfinance provides end-of-day data
- **Option C:** Missing weekend data is expected and not an "issue"
- **Option D:** Commodity futures are typically USD-denominated
- **Practical impact:** Ignoring rolls creates spurious volatility in backtests

---

## Section B: Seasonality Analysis (35 points)

### Question 6 (7 points)
A time series $y_t$ exhibits seasonality with period $m = 12$ (monthly data). Write the mathematical decomposition for an **additive** seasonal model and explain when additive is preferred over multiplicative seasonality.

**Answer:**

**Decomposition:**
$$y_t = T_t + S_t + e_t$$

Where:
- $T_t$ = trend component
- $S_t$ = seasonal component with $S_t = S_{t+12}$ (periodic)
- $e_t$ = irregular/noise component
- Constraint: $\sum_{i=1}^{12} S_i = 0$ (seasonal effects sum to zero)

**When to use additive:**
- Seasonal fluctuations are **constant in absolute magnitude** regardless of trend level
- Example: Natural gas prices may rise/fall by $2/MMBtu in winter vs summer, whether average price is $3 or $5
- **Diagnostic:** Plot seasonal amplitude over time; if constant → additive

**When to use multiplicative ($y_t = T_t \times S_t \times e_t$):**
- Seasonal fluctuations **scale with the level** of the series
- Example: If trend doubles, seasonal swings also double
- **Diagnostic:** If seasonal variance increases with trend → multiplicative
- **Solution:** Log-transform converts multiplicative to additive: $\log(y_t) = \log(T_t) + \log(S_t) + \log(e_t)$

**Scoring:**
- Correct decomposition formula: 3 points
- Explanation of when to use additive: 2 points
- Mention of multiplicative alternative: 2 points

---

### Question 7 (8 points)
Consider natural gas prices with monthly data. You want to represent seasonality using Fourier terms:

$$S_t = \sum_{k=1}^K \left[ a_k \sin\left(\frac{2\pi k t}{12}\right) + b_k \cos\left(\frac{2\pi k t}{12}\right) \right]$$

**(a)** How many parameters are needed for $K=2$ Fourier pairs? (2 points)

**(b)** What is the advantage of using $K=2$ Fourier terms instead of 11 monthly dummy variables? (3 points)

**(c)** For natural gas with strong annual seasonality (high winter, low summer), would you expect the $k=1$ or $k=2$ terms to have larger coefficients? Explain. (3 points)

**Answer:**

**(a)** For $K=2$:
- $k=1$: $a_1, b_1$ (2 parameters)
- $k=2$: $a_2, b_2$ (2 parameters)
- **Total: 4 parameters**

**(b)** Advantages of Fourier representation:
1. **Parsimony:** 4 parameters vs 11 dummy variables → reduced overfitting risk
2. **Smoothness:** Continuous seasonal curve, not discrete jumps between months
3. **Interpretability:** Frequency components (annual cycle, semi-annual, etc.)
4. **Forecasting:** Easier to extrapolate beyond sample period
5. **Flexibility:** Can choose $K$ based on model selection (AIC/BIC)

**(c)** $k=1$ (annual cycle) would have larger coefficients because:
- Natural gas seasonality is primarily **annual**: One peak (winter) and one trough (summer)
- $k=1$ captures the fundamental frequency: $\frac{2\pi \cdot 1 \cdot t}{12}$ completes one full cycle per year
- $k=2$ captures **semi-annual** harmonics (two cycles per year) → useful for secondary features like shoulder seasons, but smaller amplitude
- **Intuition:** Single dominant season → low-frequency dominates
- If there were two equal peaks (e.g., winter heating AND summer cooling), $k=1$ and $k=2$ might be comparable

**Scoring:**
- Part (a): 2 points for correct count
- Part (b): 3 points (1 point each for three advantages)
- Part (c): 3 points (2 for correct answer, 1 for solid reasoning)

---

### Question 8 (6 points)
STL (Seasonal and Trend decomposition using Loess) decomposition is preferred over classical decomposition for commodity prices. Which characteristic of STL is most valuable for handling the 2020 crude oil price crash (when WTI briefly went negative)?

A) STL allows time-varying seasonal patterns
B) STL is robust to outliers
C) STL automatically detects change points
D) STL works with non-Gaussian distributions

**Answer:** B

**Explanation:**
- The 2020 WTI negative price event was an extreme outlier (April 2020: -$37.63)
- Classical decomposition uses **global averaging** → outliers distort seasonal/trend estimates
- **STL uses iterative, locally weighted regression (Loess):**
  - Assigns lower weight to outliers
  - Prevents single extreme observation from corrupting entire decomposition
  - **Robustness step:** Identifies residuals > 6 median absolute deviations, downweights them
- **Option A** is true but not most relevant to handling an outlier event
- **Option C** is false: STL doesn't explicitly detect change points (though you could analyze the decomposition afterward)
- **Option D** is misleading: STL works with any distribution, but doesn't explicitly model it
- **Practical impact:** Without robustness, the negative price would distort seasonal estimates for months before/after

---

### Question 9 (7 points)
You are modeling corn prices with a Bayesian seasonal model in PyMC:

```python
with pm.Model() as seasonal_model:
    seasonal_raw = pm.Normal('seasonal_raw', mu=0, sigma=10, shape=12)
    seasonal = pm.Deterministic('seasonal', seasonal_raw - seasonal_raw.mean())
```

**(a)** Why is the sum-to-zero constraint (`seasonal_raw - seasonal_raw.mean()`) necessary? (4 points)

**(b)** What would happen without this constraint? (3 points)

**Answer:**

**(a)** The sum-to-zero constraint is necessary for **identifiability**:
- Without constraint: **Infinite parameter combinations** produce identical likelihood
- **Non-identifiable problem:**
  - Can increase overall mean by $c$ and decrease all seasonal effects by $c$ → same predictions
  - Example: $\mu=100, S_{\text{Jan}}=5$ vs $\mu=105, S_{\text{Jan}}=0$ → both predict 105 for January
- **Mathematical:** With intercept $\mu$ and seasonal effects $S_j$:
  - Model: $y_t = \mu + S_j + \epsilon_t$
  - If $\mu' = \mu + c$ and $S_j' = S_j - c$, then $\mu' + S_j' = \mu + S_j$ → same model
- **Constraint resolves:** Forces seasonal effects to represent deviations from the mean
  - $\sum_{j=1}^{12} S_j = 0$ → seasonal effects are centered
  - Intercept $\mu$ captures grand mean, seasonal effects capture deviations

**(b)** Without constraint, two problems arise:
1. **Non-convergence:** MCMC explores all equivalent parameter combinations → poor mixing, high autocorrelation
2. **Uninterpretable posteriors:** Posterior distributions for $\mu$ and $S_j$ are non-stationary; cannot distinguish baseline from seasonal
3. **Computational waste:** Sampler explores redundant parameter space

**Alternative constraints:**
- Fix one seasonal effect (e.g., $S_{\text{Dec}} = 0$): December becomes reference month
- Sum-to-zero is preferred for **symmetry** (no arbitrary reference choice)

**Scoring:**
- Part (a): 4 points (2 for identifiability concept, 2 for explanation)
- Part (b): 3 points (1 per problem identified)

---

### Question 10 (7 points)
Agricultural commodity prices often show both **seasonal patterns** (harvest cycle) and **trends** (long-term supply/demand shifts). A analyst detrends corn prices by first-differencing: $\Delta y_t = y_t - y_{t-1}$, then analyzes seasonality in the differenced series. Explain one advantage and one disadvantage of this approach compared to decomposing the original level series.

**Answer:**

**Advantage:**
- **Stationarity achieved:** First-differencing removes unit root/trend
  - Makes seasonal patterns stationary and easier to estimate
  - Residuals are well-behaved (constant mean/variance)
- **Avoids trend misspecification:** Don't need to assume parametric trend form (linear, exponential, etc.)
- **Suitable for forecasting models:** Many time series models (SARIMA) work better with stationary data

**Disadvantage:**
- **Confounds trend and seasonality:** If trend is not truly a unit root but a smooth deterministic trend, differencing can:
  1. Create spurious dynamics (overdifferencing)
  2. Muddle seasonal effects with trend changes
  3. Reduce interpretability (seasonal patterns in changes, not levels)
- **Example:** If corn prices have slow upward trend (5%/year) plus harvest seasonality:
  - In levels: Seasonal effect is "August prices 20% above November prices"
  - In differences: Harder to interpret; seasonal pattern in month-to-month changes
- **Information loss:** Cannot decompose into interpretable level/trend/seasonal components
- **Mixing frequencies:** Differencing creates MA(1) component that can interfere with seasonal identification

**Better approach:**
- Use structural decomposition (STL) or state space model (Basic Structural Model)
- Separately model stochastic trend and deterministic seasonality
- Preserves interpretation and handles non-stationary trend appropriately

**Scoring:**
- Advantage (3 points): 2 for correct concept, 1 for explanation
- Disadvantage (4 points): 2 for correct concept, 2 for explanation

---

## Section C: Feature Engineering and Integration (35 points)

### Question 11 (8 points)
You are building features for a crude oil price forecasting model. The theory of storage suggests that inventory levels relative to capacity (storage utilization) should predict future price movements. You have:

- Weekly US crude oil stocks: $I_t$ (million barrels)
- Estimated total storage capacity: $C = 650$ million barrels (constant)

**(a)** Define a storage utilization feature $U_t$ and explain why a non-linear transformation might be appropriate. (4 points)

**(b)** Historically, when $U_t > 0.80$, crude oil prices tend to fall sharply. Propose a feature that captures this threshold effect. (4 points)

**Answer:**

**(a)** Storage utilization:
$$U_t = \frac{I_t}{C} = \frac{I_t}{650}$$

**Why non-linear transformation:**
- **Economic intuition:** Storage constraints bind asymmetrically
  - At low utilization (U < 0.5): Plenty of storage, price insensitive
  - At high utilization (U > 0.8): Storage scarce, price very sensitive
  - **Convenience yield theory:** As stocks deplete, marginal value increases non-linearly
- **Suggested non-linear features:**
  1. **Inverse transformation:** $\frac{1}{1 - U_t}$ → explodes as $U_t \to 1$
  2. **Squared term:** $U_t^2$ → captures acceleration at extremes
  3. **Logarithmic:** $\log(C - I_t)$ → log of available capacity
- **Example:** If $U = 0.5$, adding 50M barrels → small price impact. If $U = 0.9$, adding 50M barrels → huge price impact (approaching physical limit).

**(b)** Threshold feature capturing high-storage regime:
$$\text{HighStorage}_t = \begin{cases} 1 & \text{if } U_t > 0.80 \\ 0 & \text{otherwise} \end{cases}$$

Or continuous "exceedance" feature:
$$\text{StorageExcess}_t = \max(0, U_t - 0.80)$$

**Model specification:**
$$\text{PriceChange}_{t,t+1} = \beta_0 + \beta_1 U_t + \beta_2 \text{HighStorage}_t + \epsilon_t$$

Where $\beta_2 < 0$ captures additional downward pressure when storage is critically high.

**Alternative:** Piecewise linear or spline regression to allow different slopes below/above threshold.

**Scoring:**
- Part (a): 4 points (2 for formula, 2 for non-linearity explanation)
- Part (b): 4 points (2 for feature definition, 2 for justification)

---

### Question 12 (9 points)
Missing data is common in commodity datasets (exchange holidays, data collection errors). For a Bayesian state space model in PyMC, how does the framework naturally handle missing observations compared to classical methods?

Specifically, consider a local level model:
$$y_t = \mu_t + \epsilon_t$$
$$\mu_{t+1} = \mu_t + \eta_t$$

If observation $y_t$ is missing at time $t$, what happens to the state $\mu_t$ inference?

**Answer:**

**Bayesian state space handling of missing data:**

**Mechanism:**
1. **State evolution continues:** $\mu_t$ still evolves according to transition equation
   - Even if $y_t$ is unobserved, $\mu_t = \mu_{t-1} + \eta_{t-1}$ proceeds
   - State is latent (always unobserved), so missing data just means no update from observation

2. **Conditional posterior:**
   - **With data:** $p(\mu_t | y_1, ..., y_t) \propto p(y_t | \mu_t) p(\mu_t | y_1, ..., y_{t-1})$
     - Observation $y_t$ refines estimate of $\mu_t$
   - **Without data:** $p(\mu_t | y_1, ..., y_{t-1})$ (no update from $y_t$)
     - State distribution is just the one-step-ahead prediction from prior time

3. **Increased uncertainty:** $\text{Var}(\mu_t)$ is larger when $y_t$ missing
   - No observation to "correct" the state prediction
   - Uncertainty accumulates due to process noise $\eta_t$

**Comparison to classical methods:**

| Method | Missing Data Handling |
|--------|----------------------|
| **Bayesian (PyMC)** | Natural; just condition on observed data. Missing $y_t$ → wider credible interval for $\mu_t$ |
| **OLS Regression** | Cannot handle missing $y$; requires complete cases or imputation |
| **Kalman Filter** | Can handle missing $y$ by skipping update step (prediction only) |
| **ARIMA** | Requires complete data or ad-hoc imputation (forward-fill, interpolation) |

**PyMC Implementation:**
```python
# Missing data via masking
y_observed = np.array([3.2, 4.1, np.nan, 4.5, 4.8])  # Missing at t=3
y_masked = np.ma.masked_invalid(y_observed)

with pm.Model():
    # ... define states ...
    y_obs = pm.Normal('y', mu=level, sigma=sigma_obs, observed=y_masked)
    # PyMC automatically handles the mask
```

**Key insight:** In Bayesian inference, we're always marginalizing over unobserved variables. A missing observation is just another latent variable to marginalize over—no special treatment needed.

**Scoring:**
- Explanation of state evolution with missing data: 3 points
- Increased uncertainty concept: 2 points
- Comparison to at least one other method: 2 points
- Technical accuracy: 2 points

---

### Question 13 (6 points)
The CFTC Commitments of Traders report is released Friday at 3:30 PM ET with data as of Tuesday. You want to incorporate Non-Commercial net positions into a predictive model for WTI crude oil prices. What is the effective forecast horizon given this reporting lag, and how should you structure your backtesting to avoid look-ahead bias?

A) 4-day horizon; use Friday's report to predict next Tuesday's price
B) 3-day horizon; use Tuesday's position to predict Friday's price
C) Same-day horizon; information is already in the market by publication
D) 7-day horizon; use Friday's report to predict next Friday's price

**Answer:** B (with important nuance)

**Explanation:**

**Reporting timeline:**
- Data snapshot: Tuesday close
- Report release: Friday 3:30 PM ET
- Lag: 3 days (Tuesday → Friday)

**Practical implications:**
1. **Information incorporation:** By Friday 3:30 PM, Tuesday's positioning is public
   - Markets may have partially impounded this info through informed trading
   - But not all participants observe COT report, so some predictive value may remain

2. **Backtesting structure:**
   - **Valid:** Use Friday's report to predict price changes from Friday close onward
   - **Invalid:** Use Friday's report to predict price changes from Tuesday to Friday (look-ahead bias)
   - **Conservative:** Assume information incorporates on Monday after report publication

3. **Optimal forecast horizon:**
   - **Not** Tuesday-to-Friday (data wasn't public)
   - Friday-to-next-Tuesday is 4 days, but includes weekend (2 non-trading days)
   - **Practical:** Predict next week's price changes (Friday-to-Friday = 7 days)

**Why not other options:**
- **A:** Incorrect lag calculation
- **C:** Unrealistic; market doesn't have the data until Friday
- **D:** This is actually the most practical horizon for weekly trading strategies

**Refined answer:** Both B and D are defensible depending on use case, but B is technically correct for the minimal lag. In practice, weekly (D) is more common.

**Scoring:**
- Correct answer: 3 points
- Proper explanation of lag: 2 points
- Backtesting implications: 1 point

---

### Question 14 (6 points)
When combining price data from different commodity exchanges (e.g., NYMEX WTI and ICE Brent), you notice that NYMEX prices are timestamped in US Eastern Time while ICE prices are in UTC. Both exchanges close at different times. What is the correct approach to align these time series for joint modeling?

A) Convert all timestamps to UTC and merge on matching timestamps
B) Use business day calendar to align regardless of exact time
C) Use daily closing prices and align by calendar date in exchange local time
D) Resample both to hourly frequency before merging

**Answer:** C

**Explanation:**

**Problem:** Closing time differences create misalignment
- NYMEX WTI: Settles at 2:30 PM ET (14:30 ET)
- ICE Brent: Settles at 3:00 PM London time (15:00 GMT = 10:00 AM ET typically)
- Using exact timestamps would not match

**Correct approach:**
1. Use **daily closing (settlement) prices** for each contract
2. Align by **calendar date in exchange local time**
3. For analysis, use common timezone (typically UTC) for consistency
4. Key: Both represent "end-of-day" price for that market

**Why other options fail:**
- **A:** Exact timestamp matching fails because close times differ; would result in no matches
- **B:** Too coarse; loses information about which prices are contemporaneous
- **D:** Intraday resampling introduces noise and doesn't address fundamental misalignment
  - Commodities are typically analyzed on daily frequency anyway

**Additional considerations:**
- **Time zone awareness:** Use `pandas` `DatetimeIndex` with `tz` attribute
- **Weekends/holidays:** Use `pd.bdate_range()` for business day alignment
- **Lead-lag relationships:** If analyzing causality, account for close time differences

```python
# Proper implementation
wti = wti.tz_localize('America/New_York')  # ET
brent = brent.tz_localize('Europe/London')  # London
wti_utc = wti.tz_convert('UTC')
brent_utc = brent.tz_convert('UTC')
combined = pd.merge(wti_utc, brent_utc, left_index=True, right_index=True, how='inner')
```

**Scoring:**
- Correct answer: 3 points
- Explanation of time zone issues: 2 points
- Practical considerations: 1 point

---

### Question 15 (6 points)
Feature engineering for commodities often involves creating **lagged fundamental variables**. You create features for natural gas prices including:
- Lagged storage levels: $\text{Storage}_{t-1}, \text{Storage}_{t-2}, ...$
- Lagged storage changes: $\Delta \text{Storage}_{t-1}, \Delta \text{Storage}_{t-2}, ...$

Why might including both levels and changes improve model performance compared to using only levels?

**Answer:**

**Levels capture:**
- **Absolute scarcity:** How tight supply is overall
- **Long-term relationships:** Equilibrium conditions
- **Example:** Storage = 2000 Bcf (low) → expect higher prices

**Changes capture:**
- **Momentum:** Direction and speed of inventory movements
- **Short-term dynamics:** Recent supply/demand shocks
- **Example:** $\Delta \text{Storage} = -100$ Bcf (large withdrawal) → demand spike → immediate price pressure

**Why both together:**
1. **Different information content:**
   - Levels → where we are (state)
   - Changes → where we're going (dynamics)

2. **Conditional relationships:**
   - Large withdrawal matters more when storage is already low
   - Interaction: $\text{Storage}_{t-1} \times \Delta \text{Storage}_{t-1}$

3. **Economic interpretation:**
   - **Theory of storage:** Spot premium = $f(\text{inventory level}, \text{convenience yield})$
   - Convenience yield relates to both stock levels and flow rates

4. **Statistical properties:**
   - Levels may be non-stationary (I(1))
   - Changes are typically stationary (I(0))
   - Including both captures long-run equilibrium (levels) and short-run adjustment (changes)

**Model form:**
$$\text{Price}_t = \beta_0 + \beta_1 \text{Storage}_{t-1} + \beta_2 \Delta \text{Storage}_{t-1} + \epsilon_t$$

This resembles an **error correction model** structure.

**Empirical evidence:**
- Models with both typically have higher $R^2$
- Changes capture weekly volatility, levels capture seasonal baseline

**Scoring:**
- Explanation of what levels capture: 2 points
- Explanation of what changes capture: 2 points
- Why both together is beneficial: 2 points

---

## Answer Key Summary

| Question | Answer | Points | Topic |
|----------|--------|--------|-------|
| 1 | C | 5 | Look-ahead bias, data lag |
| 2 | B | 6 | Ending stocks importance |
| 3 | B | 7 | Data revisions |
| 4 | B | 6 | COT positioning |
| 5 | B | 6 | Futures roll adjustment |
| 6 | Essay | 7 | Seasonal decomposition |
| 7 | Essay | 8 | Fourier seasonality |
| 8 | B | 6 | STL robustness |
| 9 | Essay | 7 | Bayesian seasonality identifiability |
| 10 | Essay | 7 | Detrending approaches |
| 11 | Essay | 8 | Storage features |
| 12 | Essay | 9 | Missing data in state space |
| 13 | B/D | 6 | COT reporting lag |
| 14 | C | 6 | Time zone alignment |
| 15 | Essay | 6 | Levels vs changes |
| **Total** | | **100** | |

---

## Scoring Rubric

**A (90-100):** Demonstrates mastery of data sourcing, quality control, seasonality analysis, and feature engineering. Correctly applies mathematical concepts and explains economic intuition.

**B (80-89):** Solid understanding with minor gaps. May miss some technical details or struggle with advanced feature engineering concepts.

**C (70-79):** Basic competency but lacks depth in seasonality mathematics or practical data pipeline considerations.

**D (60-69):** Significant gaps in understanding. Struggles with data quality issues or seasonal decomposition.

**F (<60):** Does not demonstrate minimum competency for commodity data analysis.

---

## Study Resources

- EIA API Documentation: https://www.eia.gov/opendata/
- USDA NASS QuickStats: https://quickstats.nass.usda.gov/
- CFTC COT Reports: https://www.cftc.gov/MarketReports/CommitmentsofTraders/
- Hyndman & Athanasopoulos (2021): *Forecasting: Principles and Practice*
- Cleveland et al. (1990): "STL: A Seasonal-Trend Decomposition Procedure Based on Loess"

---

**Estimated Time Distribution:**
- Section A (Data Sources): 10 minutes
- Section B (Seasonality): 12 minutes
- Section C (Feature Engineering): 8 minutes

**Good luck!**
