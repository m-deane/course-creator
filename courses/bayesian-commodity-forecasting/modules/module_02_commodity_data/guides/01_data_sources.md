# Commodity Data Sources Guide

## In Brief

Quality commodity forecasting requires reliable data on prices, fundamentals, and positioning. This guide maps the data landscape and provides practical retrieval strategies.

> 💡 **Key Insight:** **Free, official data often beats expensive commercial feeds for fundamentals.** EIA, USDA, and CFTC provide authoritative data that moves markets. Commercial data adds value primarily for higher frequency or alternative datasets.

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
