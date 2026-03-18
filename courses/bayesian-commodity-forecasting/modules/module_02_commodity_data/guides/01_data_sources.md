# Commodity Data Sources Guide

## In Brief

Quality commodity forecasting requires reliable data on prices, fundamentals, and positioning. This guide maps the data landscape and provides practical retrieval strategies.

> 💡 **Key Insight:** **Free, official data often beats expensive commercial feeds for fundamentals.** EIA, USDA, and CFTC provide authoritative data that moves markets. Commercial data adds value primarily for higher frequency or alternative datasets.

---

## Formal Definition

**Fundamental data** in commodity markets refers to the supply-demand balance sheet variables that determine equilibrium prices according to storage theory. The core relationship is:

$$\text{Ending Stocks}_t = \text{Beginning Stocks}_t + \text{Production}_t + \text{Imports}_t - \text{Consumption}_t - \text{Exports}_t$$

This **stock-flow accounting identity** is exact — there is no approximation. Every commodity forecasting model ultimately reduces to beliefs about the future evolution of this balance sheet. Data sources differ by:

- **Frequency:** Daily (prices, LME stocks) → Weekly (EIA, CFTC) → Monthly (USDA WASDE) → Quarterly (company reports)
- **Revision frequency:** Weekly EIA estimates are revised monthly; WASDE estimates are revised and replaced each month
- **Look-back window:** Older data more reliable; newer data subject to revision
- **Coverage:** US-only (EIA) vs. global (WASDE world tables, IEA)

**Survey vs. administrative data:**
- EIA petroleum: Survey-based (sample of operators), subject to sampling error
- USDA NASS: Survey and administrative records
- CFTC COT: Administrative (regulatory filing), nearly exact

---

## Intuitive Explanation

Think of commodity data sources as different lenses on the same physical reality. The EIA weekly petroleum report is a blurry photo taken quickly every Wednesday — it captures the rough shape of US crude inventories but has measurement noise and is revised. The annual EIA-914 natural gas production survey is a sharper, longer-exposure photo — more accurate but only available months later.

Bayesian models handle these different data qualities naturally: attach a likelihood with wider variance to noisier sources, and a tighter likelihood to more reliable data. The posterior automatically down-weights noisy observations. When you update your inventory model with a fresh EIA report, you're doing exactly what Bayes' theorem prescribes: weighting new information by its precision relative to your prior uncertainty.

The look-ahead bias danger is the most important practical point: using revised data in backtests makes models look better than they actually were. The real-time data vintage — what a trader saw on the day — is what matters. Several data vendors (Refinitiv, Bloomberg, FactSet) provide vintage-stamped data specifically for this reason.

---

## 1. Energy Data (EIA)

### Weekly Petroleum Status Report

The most market-moving energy report in the world.

**Release:** Wednesday 10:30 AM ET (Thursday if Monday holiday)
**URL:** https://www.eia.gov/petroleum/supply/weekly/

**Key Series:**
| Series | EIA Code | Description |
|--------|----------|-------------|
| Crude Stocks | WCESTUS1 | Total US crude oil inventories |
| Cushing Stocks | WCRSTUS1 | Cushing, OK crude stocks (WTI delivery) |
| Gasoline Stocks | WGTSTUS1 | Total motor gasoline inventories |
| Distillate Stocks | WDISTUS1 | Heating oil + diesel inventories |
| Refinery Utilization | WPULEUS3 | Percent of capacity in use |
| Crude Production | WCRFPUS2 | Weekly crude oil production |

**Python Retrieval:**
```python
import pandas as pd

def get_eia_petroleum(series_id, api_key):
    """
    Retrieve EIA petroleum data.

    Parameters
    ----------
    series_id : str
        EIA series identifier (e.g., 'PET.WCESTUS1.W')
    api_key : str
        EIA API key (free registration)

    Returns
    -------
    pd.DataFrame
        Time series with date index
    """
    url = f"https://api.eia.gov/v2/seriesid/{series_id}"
    params = {"api_key": api_key, "frequency": "weekly"}

    response = requests.get(url, params=params)
    data = response.json()

    df = pd.DataFrame(data['response']['data'])
    df['date'] = pd.to_datetime(df['period'])
    df = df.set_index('date').sort_index()

    return df[['value']].rename(columns={'value': series_id})
```

### Natural Gas Weekly Update

**Release:** Thursday 10:30 AM ET
**URL:** https://www.eia.gov/naturalgas/weekly/

**Key Series:**
| Series | Description |
|--------|-------------|
| Working Gas in Storage | Total lower-48 storage |
| Net Change | Weekly injection/withdrawal |
| Henry Hub Spot | Daily spot price |

### Short-Term Energy Outlook (STEO)

**Release:** Monthly
**Content:** 18-month forecasts for supply, demand, prices

---

## 2. Agricultural Data (USDA)

### WASDE Report

World Agricultural Supply and Demand Estimates - the most important agricultural report.

**Release:** Monthly, around 12th
**URL:** https://www.usda.gov/oce/commodity/wasde

**Key Tables:**
- US Supply & Use (production, consumption, ending stocks)
- World Supply & Use
- Price forecasts

**Retrieval via USDA NASS API:**
```python
import requests

def get_usda_nass(commodity, year, api_key):
    """
    Retrieve USDA production/stocks data.

    Parameters
    ----------
    commodity : str
        'CORN', 'SOYBEANS', 'WHEAT'
    year : int
        Marketing year
    api_key : str
        USDA NASS API key
    """
    base_url = "https://quickstats.nass.usda.gov/api/api_GET/"
    params = {
        "key": api_key,
        "commodity_desc": commodity,
        "year": year,
        "format": "JSON"
    }

    response = requests.get(base_url, params=params)
    return pd.DataFrame(response.json()['data'])
```

### Crop Progress Report

**Release:** Weekly during growing season (April-November)
**Content:** Planting progress, crop condition ratings

**Condition Categories:**
- Excellent, Good, Fair, Poor, Very Poor
- Market watches "Good/Excellent" percentage

### Export Sales Report

**Release:** Weekly (Thursday 8:30 AM ET)
**Content:** US agricultural export sales and shipments

---

## 3. Metals Data (LME, COMEX)

### LME Warehouse Stocks

**Release:** Daily
**Metals:** Copper, Aluminum, Zinc, Lead, Nickel, Tin

**Key Metrics:**
- Total stocks
- Cancelled warrants (metal earmarked for delivery)
- In vs. Out flows

### COMEX Inventory

**Release:** Daily
**Metals:** Gold, Silver, Copper

**Categories:**
- Registered (eligible for delivery)
- Eligible (in COMEX warehouses but not registered)

---

## 4. Positioning Data (CFTC)

### Commitments of Traders (COT)

**Release:** Friday 3:30 PM ET (Tuesday data)
**URL:** https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm

**Trader Categories:**
| Category | Description |
|----------|-------------|
| Commercial | Hedgers (producers, consumers) |
| Non-Commercial | Speculators (hedge funds, CTAs) |
| Non-Reportable | Small traders |

**Key Metrics:**
- Net position = Long - Short
- Open interest
- Changes from prior week

**Interpretation:**
- Extreme speculator positioning often precedes reversals
- Commercial hedgers are often "smart money"

---

## 5. Price Data (Free Sources)

### Yahoo Finance (yfinance)

```python
import yfinance as yf

# Futures contracts
cl = yf.download('CL=F', start='2020-01-01')  # WTI Crude
ng = yf.download('NG=F', start='2020-01-01')  # Natural Gas
gc = yf.download('GC=F', start='2020-01-01')  # Gold
zc = yf.download('ZC=F', start='2020-01-01')  # Corn
zs = yf.download('ZS=F', start='2020-01-01')  # Soybeans

# Continuous front-month contracts
# Note: These roll automatically but may have gaps
```

### FRED (Federal Reserve)

```python
from fredapi import Fred

fred = Fred(api_key='your_key')

# Commodity prices
wti = fred.get_series('DCOILWTICO')  # WTI spot
brent = fred.get_series('DCOILBRENTEU')  # Brent spot

# Macro variables useful for commodities
dxy = fred.get_series('DTWEXBGS')  # Dollar index
fedfunds = fred.get_series('FEDFUNDS')  # Fed funds rate
```

---

## 6. Data Quality Considerations

### Missing Data Patterns

| Source | Common Gaps | Handling |
|--------|-------------|----------|
| EIA Weekly | Holidays | Forward fill or interpolate |
| USDA WASDE | Not released monthly Jan | Use prior month |
| yfinance | Weekends, exchange holidays | Business day index |
| LME | Exchange holidays | Forward fill |

### Revisions

**EIA:** Historical data revised monthly
**USDA:** WASDE revisions rare but significant when they occur
**Strategy:** Always re-download recent history before modeling

### Look-Ahead Bias

**Critical:** When backtesting, only use data available at the time.

```python
def get_available_data(date, release_schedule):
    """
    Return only data that would have been available on given date.

    Example: EIA reports Wednesday with data through Friday.
    On Monday, latest available data is from prior week.
    """
    # Implementation depends on specific release schedule
    pass
```

---

## 7. Building a Data Pipeline

### Recommended Architecture

```
data/
├── raw/                    # Original downloads
│   ├── eia/
│   ├── usda/
│   └── prices/
├── processed/              # Cleaned, aligned data
│   ├── fundamentals.parquet
│   └── prices.parquet
├── features/               # Engineered features
│   └── features.parquet
└── logs/                   # Download logs
    └── data_log.csv
```

### Pipeline Principles

1. **Idempotent:** Running twice produces same result
2. **Logged:** Track when data was downloaded
3. **Versioned:** Store raw data, process into features
4. **Tested:** Validate data quality automatically

---

## Common Pitfalls

### 1. Look-Ahead Bias in Backtests

Using revised data to backtest a model that would have seen only preliminary data produces overly optimistic results. The EIA revises crude inventory estimates monthly — using the final revised number pretends you had information you didn't.

**Fix:** Download EIA data with revision timestamps, or use only data older than 6 months where revisions are mostly complete.

### 2. Ignoring Release Timing

EIA petroleum data released Wednesday covers the week ending Friday (5 days prior). Using it as same-day information in a daily model is look-ahead bias.

**Fix:** In your data pipeline, always store `release_date` separately from `reference_period`.

### 3. Mixing Frequencies Without Alignment

Merging daily prices with weekly EIA inventory and monthly WASDE data requires careful date alignment. A naive join produces NaN-filled rows that can contaminate model training.

**Fix:** Decide on a target frequency, then use `ffill()` (forward-fill) for slower-frequency series — the EIA inventory figure was valid from release date until the next release.

### 4. EIA API Pagination

EIA's v2 API returns max 5,000 rows per call. Long historical pulls silently truncate data.

**Fix:** Implement pagination with `offset` parameter or use date-range chunking.

### 5. USDA WASDE Marketing Year vs. Calendar Year

USDA uses marketing years (corn: September-August) not calendar years. Merging WASDE with calendar-year macro data on year alone gives 4-month misalignment.

**Fix:** Always join on the `period_end` date, not the year field.

---

## Connections

**Builds on:**
- Module 0: Commodity markets introduction — the supply-demand balance sheet structure
- Module 0: Probability review — measurement error as a likelihood component

**Leads to:**
- Module 2 (Guide 2): Seasonality analysis — extracting seasonal patterns from these data series
- Module 3: State space models — the inventory stock-flow equations as state transitions
- Module 8: Fundamental variables — using balance sheet data directly in Bayesian forecasting models

**Related to:**
- Data engineering: pipelines, idempotency, vintage management
- Measurement error models: explicitly modeling the gap between survey estimates and true inventories

---

## Practice Problems

1. Write a function to retrieve the last 52 weeks of EIA crude inventory data and calculate the year-over-year change.

2. The WASDE report for corn includes "Ending Stocks." Why is this variable important for price forecasting?

3. Design a data pipeline that updates weekly with the latest EIA, USDA (during growing season), and price data.

---

## Further Reading

- EIA API Documentation: https://www.eia.gov/opendata/
- USDA NASS Quick Stats: https://quickstats.nass.usda.gov/
- CFTC COT Explanatory Notes: https://www.cftc.gov/MarketReports/CommitmentsofTraders/ExplanatoryNotes/index.htm

---

*"Know your data sources better than you know your models. The data is the foundation."*
