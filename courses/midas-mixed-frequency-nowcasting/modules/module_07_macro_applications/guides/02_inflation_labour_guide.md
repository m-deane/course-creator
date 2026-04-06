# Inflation and Labour Market Nowcasting

> **Reading time:** ~16 min | **Module:** 07 — Macro Applications | **Prerequisites:** Module 6


## Learning Objectives

<div class="flow">
<div class="flow-step mint">1. Load Ragged Data</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">2. Estimate MIDAS</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">3. Generate Nowcast</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Update as Data Arrives</div>
</div>


<div class="callout-key">

**Key Concept Summary:** CPI and PCE are released with a ~2-week lag after month-end. During the month, several high-frequency signals are available:

</div>

By the end of this guide you will be able to:

1. Build an inflation nowcasting model using daily commodity prices and weekly PPI data
2. Implement labour market nowcasting (weekly claims → monthly payrolls)
3. Understand the specific data structures for inflation and employment nowcasting
4. Evaluate these models appropriately for their target variables

---

## 1. Inflation Nowcasting

### 1.1 Why Inflation Nowcasting is Challenging

CPI and PCE are released with a ~2-week lag after month-end. During the month, several high-frequency signals are available:

- **Daily**: Commodity prices (oil, food grains, metals), energy prices, FX rates
- **Weekly**: Gasoline prices (EIA survey), food-at-home price surveys
- **Monthly**: PPI components (released before CPI by ~1 week), import prices

The challenge: CPI has many components (energy 7%, food 14%, core goods 22%, core services 57%) with very different data structures. Energy and food components have good daily predictors; core services (rent, medical, education) have very little high-frequency predictors.

### 1.2 Component-Level vs Aggregate Forecasting

**Bottom-up approach**: Nowcast each component separately, aggregate with CPI weights.

$$\hat{\pi}^{\text{CPI}}_t = w_E \hat{\pi}^E_t + w_F \hat{\pi}^F_t + w_{CS} \hat{\pi}^{CS}_t + w_{CG} \hat{\pi}^{CG}_t$$

**Top-down approach**: Forecast aggregate CPI directly using all available signals.

Bottom-up tends to outperform when component-specific predictors have high information content (energy: daily oil prices are very informative).

### 1.3 Energy Component MIDAS

Daily crude oil and natural gas prices directly predict the energy CPI component:

$$\pi^E_t = \mu + \phi_1 \sum_{j=0}^{K_d-1} B_1(j) \Delta \log P^{\text{oil}}_{t-j/d} + \phi_2 \sum_{j=0}^{K_d-1} B_2(j) \Delta \log P^{\text{gas}}_{t-j/d} + \varepsilon_t$$

Typical $R^2$ for the energy component nowcast: 0.6–0.8 (daily energy prices are highly informative for monthly energy CPI).


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def build_inflation_midas_features(
    daily_oil_ret, daily_gasoline_ret, monthly_ppi,
    target_dates, K_daily=22, K_ppi=2
):
    """
    Build MIDAS feature matrix for inflation nowcasting.

    Features:
    - K_daily daily oil price returns (lagged)
    - K_daily daily gasoline price returns (lagged)
    - K_ppi monthly PPI changes (lagged)
    """
    rows = []
    dates = []

    for date in target_dates:
        row = {}

```

<div class="callout-insight">

**Insight:** Real-time nowcasting is fundamentally different from pseudo out-of-sample backtesting. The ragged-edge data structure means your model sees different information at different points within a quarter.

</div>

```python
        # Daily oil lags
        d_avail_oil = daily_oil_ret[daily_oil_ret.index < date]
        if len(d_avail_oil) < K_daily:
            continue
        oil_lags = d_avail_oil.tail(K_daily).values[::-1]
        for j, v in enumerate(oil_lags, start=1):
            row[f'oil_L{j}'] = v

        # Daily gasoline lags (if available)
        if daily_gasoline_ret is not None:
            d_avail_gas = daily_gasoline_ret[daily_gasoline_ret.index < date]
            if len(d_avail_gas) >= K_daily:
                gas_lags = d_avail_gas.tail(K_daily).values[::-1]
                for j, v in enumerate(gas_lags, start=1):
                    row[f'gas_L{j}'] = v

        # Monthly PPI lags
        m_avail_ppi = monthly_ppi[monthly_ppi.index < date]
        if len(m_avail_ppi) >= K_ppi:
            ppi_lags = m_avail_ppi.tail(K_ppi).values[::-1]
            for l, v in enumerate(ppi_lags, start=1):
                row[f'ppi_L{l}'] = v

        # Summary statistics
        row['oil_mean'] = np.mean(oil_lags)
        row['oil_momentum'] = oil_lags[0] - oil_lags[-1]
        row['oil_vol'] = np.std(oil_lags)

        rows.append(row)
        dates.append(date)

    X = pd.DataFrame(rows, index=dates)
    return X
```

</div>
</div>

### 1.4 Core Inflation Nowcasting

Core services (the "super-core") are the hardest to nowcast because:
- No daily predictor with good correlation
- Sticky prices: rents and services change slowly
- Seasonal adjustment complicates signal extraction

**Practical approach**: Use a simple AR model for core services and MIDAS for energy/food.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def core_services_ar_forecast(core_cpi_monthly, h=1):
    """
    Simple AR(1) model for core services CPI — best available for this component.
    """
    T = len(core_cpi_monthly)
    y = core_cpi_monthly.values

    # AR(1) parameters from last 24 months
    lookback = min(24, T - h - 1)
    y_fit = y[-(lookback+h+1):-h]
    y_lag = y[-(lookback+h+2):-h-1]

    X_ar = np.column_stack([np.ones(len(y_fit)), y_lag])
    beta, _, _, _ = np.linalg.lstsq(X_ar, y_fit, rcond=None)
    return beta[0] + beta[1] * y[-1]
```

</div>
</div>

---

## 2. Labour Market Nowcasting

<div class="callout-warning">

**Warning:** Pseudo out-of-sample exercises that do not properly account for the real-time data vintage will overstate nowcast accuracy. Always use the ragged-edge structure that would have been available at each historical nowcast date.

</div>


### 2.1 Weekly Initial Claims → Monthly Payrolls

The most important high-frequency indicator for labour market nowcasting is **initial jobless claims** (weekly, released every Thursday). The relationship between claims and payrolls is:

- High claims → workers entering unemployment → future payrolls negative signal
- Low claims → labour market strengthening → payrolls will likely be positive

The MIDAS structure is ideal: monthly payrolls as target, 4-5 weekly claims observations as predictors.

$$\Delta \text{Payrolls}^{(m)}_t = \mu + \phi \sum_{k=0}^{K_w-1} B(k;\theta_1,\theta_2) \text{Claims}^{(w)}_{t-k} + \varepsilon_t$$

where $K_w$ = 4 (approximately 4 weekly claims reports per month).

### 2.2 Multiple High-Frequency Indicators

Beyond claims, several other high-frequency indicators are available:

| Indicator | Frequency | Source | Signal |
|-----------|-----------|--------|--------|
| Initial Claims | Weekly | DOL/BLS | Leading (announces job losses) |
| Continuing Claims | Weekly | DOL/BLS | Level of unemployment |
| ADP Employment | Monthly | ADP | Pre-payrolls estimate |
| Monster Job Ads Index | Monthly | Monster | Leading (demand side) |
| Job Openings (JOLTS) | Monthly (lagged) | BLS | Demand side with lag |
| PMI Employment | Monthly | ISM | Diffusion index |
| Conference Board Employment Trends | Monthly | CB | Composite |

A MIDAS-X model can incorporate several of these simultaneously.

### 2.3 Implementation


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def build_labour_midas_features(
    weekly_claims, monthly_adp, monthly_pmi,
    target_dates, K_weekly=4, K_monthly_lag=2
):
    """
    Build MIDAS feature matrix for payrolls nowcasting.

    Parameters
    ----------
    weekly_claims : Series — weekly initial jobless claims (thousands)
    monthly_adp : Series — ADP private payrolls estimate (monthly)
    monthly_pmi : Series — ISM Manufacturing employment PMI
    target_dates : DatetimeIndex — monthly payrolls release dates (target variable)
    K_weekly : int — number of weekly claims lags per month
    K_monthly_lag : int — number of monthly indicator lags
    """
    rows = []
    dates = []

    for date in target_dates:
        row = {}

        # Weekly claims: K_weekly most recent observations before month end
        w_avail = weekly_claims[weekly_claims.index < date]
        if len(w_avail) < K_weekly:
            continue

        w_lags = w_avail.tail(K_weekly).values[::-1]
        for k, v in enumerate(w_lags, start=1):
            row[f'claims_W{k}'] = v

        # Claims summary statistics
        row['claims_mean'] = np.mean(w_lags)
        row['claims_4wk_chg'] = w_lags[0] - w_lags[-1]  # 4-week change
        row['claims_trend'] = np.polyfit(range(K_weekly), w_lags[::-1], 1)[0]

        # ADP employment: K_monthly_lag monthly lags before this payrolls date
        if monthly_adp is not None:
            m_avail_adp = monthly_adp[monthly_adp.index < date]
            if len(m_avail_adp) >= K_monthly_lag:
                adp_lags = m_avail_adp.tail(K_monthly_lag).values[::-1]
                for l, v in enumerate(adp_lags, start=1):
                    row[f'adp_L{l}'] = v

        # PMI Employment: monthly lags
        if monthly_pmi is not None:
            m_avail_pmi = monthly_pmi[monthly_pmi.index < date]
            if len(m_avail_pmi) >= K_monthly_lag:
                pmi_lags = m_avail_pmi.tail(K_monthly_lag).values[::-1]
                for l, v in enumerate(pmi_lags, start=1):
                    row[f'pmi_L{l}'] = v

        rows.append(row)
        dates.append(date)

    X = pd.DataFrame(rows, index=dates)
    return X
```


### 2.4 Relationship: Claims → Unemployment → Payrolls

The mechanistic chain is:

1. **Initial claims rise** → More workers filing for unemployment insurance
2. **Continuing claims rise** → Unemployment rate increases in 2-3 months
3. **Payrolls fall** → GDP employment component weakens

MIDAS captures this cascade through the lag structure. The claim-payrolls relationship tends to have a lag of 2-4 weeks, well-captured by 4 weekly claims lags.

---

## 3. Unemployment Rate Nowcasting

### 3.1 From Claims to Unemployment Rate

The unemployment rate $u_t$ is related to flows in and out of unemployment:

$$\Delta u_t \approx s_t (1 - u_{t-1}) - f_t u_{t-1}$$

where $s_t$ is the job separation rate (related to initial claims) and $f_t$ is the job finding rate (related to job openings, PMI, ADP).

A simpler direct MIDAS model:

$$u^{(m)}_t = \mu + \phi_1 \sum_{k=0}^{K_w-1} B_1(k) \text{Claims}^{(w)}_{t-k} + \phi_2 u^{(m)}_{t-1} + \varepsilon_t$$

### 3.2 Labour Force Participation

Labour force participation (LFP) is harder to nowcast than the unemployment rate:

- Lower frequency drivers: demographic trends, social factors
- Limited high-frequency predictors
- More structural than cyclical

Approach: Use an AR(2) for LFP as the baseline, with the unemployment rate as an additional predictor.

---

## 4. Evaluating Macro Nowcasts

### 4.1 Standard Metrics

For macro variables, evaluation requires:

1. **RMSFE** (Root Mean Square Forecast Error): Overall accuracy
2. **Bias**: Systematic over- or under-prediction
3. **Correlation**: Are directional movements correct?
4. **Sign accuracy**: What fraction of expansions/contractions are correctly called?

```python
def evaluate_macro_nowcast(actuals, forecasts, variable_name):
    """
    Comprehensive evaluation of macro nowcast accuracy.
    """
    errors = actuals - forecasts
    T = len(errors)

    rmsfe = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    bias = np.mean(errors)
    corr = np.corrcoef(actuals, forecasts)[0, 1]
    sign_acc = np.mean(np.sign(actuals) == np.sign(forecasts))

    print(f"\n{variable_name} Nowcast Evaluation:")
    print(f"  RMSFE:      {rmsfe:.4f}")
    print(f"  MAE:        {mae:.4f}")
    print(f"  Bias:       {bias:.4f} ({'over-predicts' if bias < 0 else 'under-predicts'})")
    print(f"  Correlation:{corr:.4f}")
    print(f"  Sign acc.:  {sign_acc:.1%}")

    return {'rmsfe': rmsfe, 'mae': mae, 'bias': bias, 'corr': corr, 'sign': sign_acc}
```

### 4.2 Recession Detection

For nowcasting in the context of policy, correctly identifying recessions (consecutive negative quarters) is often more important than minimising RMSFE. Evaluate:

- Precision and recall for recession quarters
- First-recession-quarter detection lag
- Number of false alarms (positive RMSFE but negative GDP)

---

## 5. Integrating Inflation and Labour Market Nowcasts

In practice, GDP, inflation, and employment nowcasts are produced jointly because they share common macro drivers (aggregate demand, supply shocks). A joint model:

$$\begin{pmatrix} \hat{\pi}_t \\ \Delta \hat{u}_t \\ \Delta \log \hat{\text{GDP}}_t \end{pmatrix} = \mu + \sum_{j} B(j) \cdot x_{t-j/m} + \varepsilon_t$$

where $x$ contains the common predictors. The key benefit: if payrolls surprise positively, both the GDP and inflation nowcasts should be revised up simultaneously.

---

## 6. Key References

<div class="callout-danger">

**Danger:** Never use future information when constructing the high-frequency regressor matrix. In a real-time nowcasting context, you only have data up to the current date -- using the full quarter of monthly data when nowcasting mid-quarter is a look-ahead bias that invalidates your results.



- Brave, S., & Butters, R. A. (2010). Chicago Fed National Activity Index trends and cycles. *Economic Perspectives*, 34(Q1), 13–23.
- Atkeson, A., & Ohanian, L. E. (2001). Are Phillips curves useful for forecasting inflation? *FRBM Quarterly Review*, 25(1), 2–11.
- Modugno, M. (2013). Now-casting inflation using high-frequency data. *International Journal of Forecasting*, 29(4), 664–675.
- Knotek, E. S., & Zaman, S. (2017). Nowcasting U.S. headline and core inflation. *Journal of Money, Credit and Banking*, 49(5), 931–968.

---

## Summary

Inflation and labour market nowcasting exploit specific high-frequency signals:

**Inflation**:
- Daily oil/gasoline prices → energy CPI component (high predictability, R²≈0.7)
- PPI components → CPI goods component (with 1-week lead)
- Core services: limited high-frequency predictors → AR(1) is competitive

**Labour market**:
- Weekly initial claims → monthly payrolls (4-week lead, MIDAS-W model)
- ADP + PMI employment → payrolls supplement
- Unemployment rate: claims + job finding proxies in MIDAS-X

Both applications use the same MIDAS-X framework with Beta polynomial weights. The key difference from GDP nowcasting is the specific predictor-target alignment (daily energy prices for inflation; weekly claims for payrolls).

Next: `01_simplified_nyfed_nowcast.ipynb`


---

## Conceptual Practice Questions

**Practice Question 1:** How does the ragged-edge problem affect the reliability of real-time nowcasts compared to pseudo out-of-sample exercises?

**Practice Question 2:** What is the key difference between direct and iterated multi-step forecasts in a MIDAS context?


---

## Cross-References

<a class="link-card" href="./01_gdp_nowcasting_practice_guide.md">
  <div class="link-card-title">01 Gdp Nowcasting Practice</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_gdp_nowcasting_practice_slides.md">
  <div class="link-card-title">01 Gdp Nowcasting Practice — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

