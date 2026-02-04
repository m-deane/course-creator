# Nowcasting

```
┌─────────────────────────────────────────────────────────────────────┐
│ NOWCASTING (FORECASTING THE PRESENT)                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Timeline: Estimate current quarter GDP before official release   │
│                                                                     │
│   Q1 ──────────── Q2 ──────────── Q3 ───────────┐                  │
│   Jan  Feb  Mar  Apr  May  Jun  Jul  Aug  Sep  │ Oct               │
│    │    │    │    │    │    │    │    │    │   ↓                   │
│   ✓IP ✓IP ✓IP ✓IP ✓IP ✓IP ✓IP ✓IP ✓IP  [Now]                     │
│   ✓Ret ✓Ret ✓Ret ✓Ret ✓Ret ✓Ret ✓Ret ✓Ret✗                       │
│   ✗GDP ✗GDP ✓GDP ✗GDP ✗GDP ✗GDP  ❓              Official          │
│                                                  GDP Q3?            │
│                                                                     │
│   Problem: GDP for Q3 won't be released until late October!        │
│   Solution: Use high-frequency indicators (monthly/weekly) to      │
│            estimate Q3 GDP in real-time as data arrives.           │
│                                                                     │
│   Key challenges:                                                   │
│   1. Ragged edge: Different release dates (IP monthly, retail lag) │
│   2. Mixed frequency: Daily, weekly, monthly, quarterly            │
│   3. Revisions: Data gets updated after initial release            │
│   4. Publication lag: Most data published with 1-2 month delay     │
│                                                                     │
│   DFM handles all of these elegantly via state-space + Kalman!     │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ TL;DR: Produce real-time estimates of current-quarter GDP using    │
│        all available high-frequency data, even with missing values.│
├─────────────────────────────────────────────────────────────────────┤
│ Code (< 15 lines):                                                  │
│                                                                     │
│   from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor│
│                                                                     │
│   # Create ragged-edge dataset (NaN for missing)                   │
│   data = fetch_realtime_indicators()  # Some series more current   │
│                                                                     │
│   # Estimate DFM (Kalman handles missing data automatically)       │
│   model = DynamicFactor(data, k_factors=3, factor_order=2)         │
│   result = model.fit(maxiter=1000)                                  │
│                                                                     │
│   # Nowcast current quarter GDP                                     │
│   factors_current = result.filtered_state[:, -1]                    │
│   gdp_nowcast = result.predict(start=len(data), end=len(data))[-1] │
│                                                                     │
│   print(f"Q3 2024 GDP nowcast: {gdp_nowcast:.2f}%")                │
├─────────────────────────────────────────────────────────────────────┤
│ Common Pitfall: Using smoothed factors for nowcast! This would     │
│                 use future data. Always use FILTERED factors.      │
└─────────────────────────────────────────────────────────────────────┘
```

## The Nowcasting Problem

**Central bank scenario:** It's October 5, 2024. You need to:
- Estimate GDP growth for Q3 2024 (Jul-Sep)
- But official GDP won't be released until October 30
- Make a decision at the October 8 policy meeting

**Solution:** Nowcasting - use all available high-frequency indicators to estimate current-quarter GDP **now**.

## Why Dynamic Factor Models Win

Nowcasting requires handling:

### 1. Mixed Frequencies
- **Quarterly:** GDP (target variable)
- **Monthly:** Industrial production, retail sales, employment
- **Weekly:** Initial jobless claims
- **Daily:** Stock prices, yields

**DFM solution:** State-space framework naturally handles different frequencies via observation equation.

### 2. Ragged Edge (Asynchronous Data)
At any point in time, different series have different "latest" observations:

```
         Sep    Oct    Nov
GDP      ❓     ?      ?     ← What we want to estimate
IndProd  ✓     ❓      ?     ← Available with 2-week lag
Retail   ✓     ?      ?     ← Available with 1-month lag
Claims   ✓     ✓      ?     ← Available weekly
```

**DFM solution:** Kalman filter treats missing values as "unobserved" and optimally estimates them.

### 3. Publication Lags
Each indicator has a different publication lag:
- Initial claims: 3 days
- Industrial production: 15 days
- Retail sales: 14 days
- GDP: 30 days

**DFM solution:** Update nowcast as each new data release arrives ("news" decomposition).

### 4. Revisions
Initial data releases get revised:
- GDP: Revised after 1 month, 2 months, then annual revisions
- Employment: Revised monthly

**DFM solution:** Use real-time vintages in estimation (ALFRED database).

## The Nowcasting Workflow

### Step 1: Setup Factor Model
```python
# Monthly indicators (100+ series)
indicators = {
    'real_activity': ['IP', 'emp', 'hours', 'retail'],
    'prices': ['CPI', 'PPI', 'wages'],
    'financial': ['rates', 'spreads', 'stocks']
}

# Target: Quarterly GDP
target = 'GDP_growth'
```

### Step 2: Estimate Model on Historical Data
```python
# Use full sample with actual GDP to estimate parameters
model = DynamicFactor(historical_data, k_factors=3, factor_order=2)
result = model.fit()

# Save parameters: Lambda, T, Q, H
params = result.params
```

### Step 3: Real-Time Nowcasting
```python
# Load latest data (with ragged edge)
current_data = fetch_latest_data()  # NaNs for unreleased data

# Filter with fixed parameters
kf = model.clone(current_data, params=params)
filtered_state = kf.filter()

# Nowcast = E[GDP_t | data up to now]
gdp_nowcast = filtered_state.filtered_state[-1, -1]
gdp_uncertainty = filtered_state.filtered_state_cov[-1, -1, -1]
```

### Step 4: Update as New Data Arrives
```python
# New indicator released (e.g., retail sales)
current_data['retail_sales'][-1] = 2.3  # Fill in NaN

# Update nowcast
kf_updated = model.clone(current_data, params=params)
filtered_state_updated = kf_updated.filter()

# "News" = change in nowcast
news = filtered_state_updated.filtered_state[-1, -1] - gdp_nowcast
print(f"Retail sales release changed nowcast by {news:.2f}pp")
```

## Bridge Equations (Simple Alternative)

Before DFM, nowcasting used "bridge equations":

```python
# Aggregate monthly indicators to quarterly
IP_q = monthly_to_quarterly(IP)
retail_q = monthly_to_quarterly(retail_sales)

# Simple regression
GDP_nowcast = β₀ + β₁*IP_q + β₂*retail_q + ε
```

**Problems with bridge equations:**
- Manual aggregation loses information
- Can't handle ragged edge naturally
- No uncertainty quantification for missing data
- Each target needs separate model

**DFM advantages:**
- Automatic aggregation via state-space
- Kalman filter handles missing data optimally
- Single model for all targets
- Proper uncertainty quantification

## The "News" Decomposition

Key insight from Banbura & Modugno (2014): Decompose nowcast revisions into:

```
Δ Nowcast = ∑ (impact of each data release)
```

Example:
```python
# Nowcast before retail sales release: 2.5%
nowcast_before = 2.5

# Retail sales comes in higher than expected
# Nowcast after: 2.8%
nowcast_after = 2.8

# News from retail = 0.3pp
# This measures information content of retail sales for GDP
```

This helps central banks understand **which indicators matter most**.

## Real-World Example: NY Fed Nowcast

The New York Fed publishes real-time nowcasts:
- Updated every Friday
- Uses ~50 indicators
- 3-factor DFM
- Provides probability distributions

Typical accuracy:
- 1 month before release: RMSE ≈ 0.5pp
- Day before release: RMSE ≈ 0.3pp
- Better than consensus forecasts!

## Mixed-Frequency State-Space

Technical detail: How to handle quarterly GDP + monthly indicators?

**Approach 1: Monthly model with aggregation**
```python
# GDP is average of monthly latent GDP
GDP_q = (GDP_m1 + GDP_m2 + GDP_m3) / 3

# State includes 3 months
state = [factors, GDP_m1, GDP_m2, GDP_m3]
```

**Approach 2: Quarterly model with disaggregation**
```python
# Interpolate quarterly factors to monthly
factors_m = interpolate_to_monthly(factors_q)
```

Approach 1 is more common in practice.

## Evaluation: Use Real-Time Vintages!

**Wrong way:**
```python
# Fit on full sample, test on held-out period
rmse = evaluate(model, test_data)  # BIASED!
```

This uses revised data that wouldn't have been available in real-time.

**Right way:**
```python
# Use ALFRED real-time vintages
for vintage_date in test_dates:
    data_available = fetch_vintage(vintage_date)  # Only data available then
    nowcast = model.filter(data_available)
    actual = first_release_gdp[vintage_date]  # Compare to first release
    errors.append(nowcast - actual)

rmse = np.sqrt(np.mean(errors**2))
```

## Quick Diagnostic Checks

After nowcasting, verify:

1. **Factor interpretation:** Do factors align with known drivers?
2. **Revision pattern:** Does nowcast improve as data arrives?
3. **Confidence bands:** Are they well-calibrated? (68% should contain truth)
4. **Contribution analysis:** Do important indicators matter most?

## Extensions

Modern nowcasting adds:

1. **Machine learning:** Random forests for factor extraction
2. **Big data:** Google trends, satellite imagery, credit card data
3. **Ensemble:** Combine DFM with other models (VAR, ML)
4. **Density forecasts:** Full distribution, not just mean

But DFM remains the workhorse for its elegant handling of data irregularities.
