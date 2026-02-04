# Project 3: Real-Time Forecasting Dashboard (Advanced)

## What You'll Build

A **production-grade real-time forecasting system** that continuously monitors economic indicators, updates factor estimates as new data arrives, and provides live forecasts through an interactive web dashboard.

**End result:** A complete forecasting platform that:
- Ingests data automatically (daily/weekly/monthly)
- Handles ragged edge and mixed frequencies
- Updates factors and forecasts in real-time
- Tracks forecast accuracy and model performance
- Provides interactive dashboard with drill-down capabilities
- Sends alerts when economic conditions change significantly

**Difficulty:** Advanced
**Time:** 10-12 hours
**Prerequisites:** Projects 1 & 2, experience with web frameworks, databases

---

## Learning Goals

1. **Production ML Systems**
   - Real-time data pipelines
   - Model versioning and A/B testing
   - Performance monitoring and alerting
   - Graceful degradation and error handling

2. **Advanced Nowcasting**
   - Mixed-frequency state-space models
   - Real-time vintage data handling
   - News decomposition (impact of each data release)
   - Density forecasts (full distribution, not just mean)

3. **Web Application Development**
   - RESTful API design
   - Database integration (PostgreSQL/SQLite)
   - Interactive dashboards (Plotly Dash/Streamlit)
   - User authentication and permissions

4. **DevOps & Deployment**
   - Containerization (Docker)
   - CI/CD pipelines (GitHub Actions)
   - Cloud deployment (AWS/GCP/Azure)
   - Monitoring and logging (CloudWatch, Sentry)

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     DATA LAYER                               │
├──────────────────────────────────────────────────────────────┤
│ ┌────────────┐  ┌────────────┐  ┌────────────┐             │
│ │ FRED API   │  │ BLS API    │  │ Other APIs │             │
│ └─────┬──────┘  └─────┬──────┘  └─────┬──────┘             │
│       └─────────────────┴──────────────┘                     │
│                         │                                    │
│                         ↓                                    │
│               ┌─────────────────┐                            │
│               │ Data Scheduler  │ (AWS EventBridge/Airflow) │
│               │ - Fetch daily   │                            │
│               │ - Validate      │                            │
│               │ - Store         │                            │
│               └────────┬────────┘                            │
└────────────────────────│─────────────────────────────────────┘
                         │
                         ↓
┌──────────────────────────────────────────────────────────────┐
│                   STORAGE LAYER                              │
├──────────────────────────────────────────────────────────────┤
│ ┌────────────────┐  ┌─────────────────┐                     │
│ │ Raw Data (S3)  │  │ Time Series DB  │ (InfluxDB/TimescaleDB)
│ │ - Vintages     │  │ - Latest data   │                     │
│ │ - Archives     │  │ - Fast queries  │                     │
│ └────────────────┘  └─────────────────┘                     │
└──────────────────────────────────────────────────────────────┘
                         │
                         ↓
┌──────────────────────────────────────────────────────────────┐
│                   MODEL LAYER                                │
├──────────────────────────────────────────────────────────────┤
│ ┌──────────────────────────────────────────┐                │
│ │ Model Updater (scheduled: daily/weekly)  │                │
│ │ - Extract factors                        │                │
│ │ - Update parameters                      │                │
│ │ - Generate forecasts                     │                │
│ │ - Compute uncertainty                    │                │
│ └──────────────┬───────────────────────────┘                │
│                │                                             │
│                ↓                                             │
│ ┌─────────────────────────────────────────┐                 │
│ │ Model Registry                          │                 │
│ │ - Versioned models (MLflow/DVC)         │                 │
│ │ - Performance metrics                   │                 │
│ │ - A/B test different specifications     │                 │
│ └─────────────────────────────────────────┘                 │
└──────────────────────────────────────────────────────────────┘
                         │
                         ↓
┌──────────────────────────────────────────────────────────────┐
│                   API LAYER                                  │
├──────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────┐                 │
│ │ REST API (FastAPI)                      │                 │
│ │ Endpoints:                              │                 │
│ │ - GET /nowcast/{variable}               │                 │
│ │ - GET /factors/latest                   │                 │
│ │ - GET /forecast/{variable}/{horizon}    │                 │
│ │ - GET /news/{date}                      │                 │
│ │ - POST /backtest                        │                 │
│ └─────────────────────────────────────────┘                 │
└──────────────────────────────────────────────────────────────┘
                         │
                         ↓
┌──────────────────────────────────────────────────────────────┐
│                PRESENTATION LAYER                            │
├──────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────┐                 │
│ │ Interactive Dashboard (Dash/Streamlit)  │                 │
│ │ - Live nowcasts                         │                 │
│ │ - Factor trends                         │                 │
│ │ - News feed (what moved the forecast)   │                 │
│ │ - Historical accuracy                   │                 │
│ │ - Downloadable reports                  │                 │
│ └─────────────────────────────────────────┘                 │
│ ┌─────────────────────────────────────────┐                 │
│ │ Monitoring Dashboard (Grafana)          │                 │
│ │ - Data freshness                        │                 │
│ │ - Model performance drift               │                 │
│ │ - API latency                           │                 │
│ └─────────────────────────────────────────┘                 │
└──────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
project_3_advanced/
├── README.md
├── starter_code.py
├── solution.py
├── deploy.md
│
├── backend/
│   ├── api/
│   │   ├── main.py              # FastAPI application
│   │   ├── models.py            # Pydantic models
│   │   └── routers/
│   │       ├── nowcast.py
│   │       ├── factors.py
│   │       └── news.py
│   ├── database/
│   │   ├── models.py            # SQLAlchemy models
│   │   └── connection.py
│   ├── services/
│   │   ├── data_fetcher.py
│   │   ├── factor_estimator.py
│   │   ├── nowcaster.py
│   │   └── news_analyzer.py
│   └── tests/
│
├── frontend/
│   ├── app.py                   # Dash/Streamlit app
│   ├── components/
│   │   ├── nowcast_chart.py
│   │   ├── factor_plot.py
│   │   └── news_feed.py
│   └── assets/
│       └── custom.css
│
├── infrastructure/
│   ├── docker-compose.yml       # Local development
│   ├── Dockerfile
│   ├── kubernetes/              # Production deployment
│   └── terraform/               # Infrastructure as code
│
├── pipelines/
│   ├── data_ingestion.py        # Airflow DAG
│   ├── model_training.py
│   └── forecast_update.py
│
└── monitoring/
    ├── grafana_dashboard.json
    └── alerts.yaml
```

---

## Core Features to Implement

### Feature 1: Real-Time Data Pipeline

**Requirements:**
- Fetch data from multiple sources daily
- Handle API failures gracefully (retry logic)
- Validate data quality (detect outliers, missing values)
- Store as vintages (track data revisions)

**Implementation:**
```python
class DataPipeline:
    def __init__(self, sources: List[DataSource]):
        self.sources = sources
        self.storage = VintageStorage()

    async def fetch_latest(self):
        """Fetch latest data from all sources."""
        tasks = [source.fetch() for source in self.sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle failures
        for source, result in zip(self.sources, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch {source.name}: {result}")
                # Use cached data or skip
            else:
                self.storage.save(result, vintage=datetime.now())

    def validate(self, data: pd.DataFrame) -> bool:
        """Validate data quality."""
        # Check: no all-NaN columns, reasonable ranges, etc.
        pass
```

### Feature 2: Mixed-Frequency State-Space Model

**Requirements:**
- Handle daily, weekly, monthly, quarterly data simultaneously
- Kalman filter with missing observations
- Update factors as each new data point arrives

**Implementation:**
```python
from statsmodels.tsa.statespace.mlemodel import MLEModel

class MixedFrequencyDFM(MLEModel):
    """
    State-space DFM with mixed frequencies.

    States: Monthly factors
    Observations: Daily, weekly, monthly, quarterly indicators
    """

    def __init__(self, data_dict, k_factors=3):
        """
        data_dict: {
            'daily': pd.DataFrame,
            'monthly': pd.DataFrame,
            'quarterly': pd.DataFrame
        }
        """
        # Aggregate all to monthly frequency with NaNs
        # Kalman filter handles missing observations naturally
        pass

    def update(self, k_endog=None, endog=None, **kwargs):
        """Update model parameters and filter states."""
        # This is called by statsmodels during filtering
        pass
```

### Feature 3: News Decomposition

**Requirements:**
- Track how each new data release changes the nowcast
- Attribute forecast revisions to specific indicators
- Present in user-friendly format

**Implementation:**
```python
class NewsAnalyzer:
    """Decompose forecast revisions into news from each indicator."""

    def compute_news(self, nowcast_before, nowcast_after, new_data):
        """
        Compare nowcasts before and after new data release.

        Returns: Impact of each indicator on forecast revision.
        """
        # Method from Bańbura & Modugno (2014)
        # News = K * innovation
        # where K = Kalman gain
        pass

    def generate_narrative(self, news_dict):
        """Convert news into natural language."""
        # "Stronger retail sales added 0.2pp to GDP nowcast"
        pass
```

### Feature 4: Forecast Performance Tracking

**Requirements:**
- Store all historical nowcasts and forecasts
- Compare to actual releases (RMSE, MAE, coverage of CIs)
- Detect performance degradation → trigger re-estimation

**Implementation:**
```python
class PerformanceMonitor:
    def __init__(self, db_connection):
        self.db = db_connection

    def evaluate_vintage(self, vintage_date):
        """Evaluate forecasts made on vintage_date."""
        forecasts = self.db.get_forecasts(vintage_date)
        actuals = self.db.get_actuals()

        errors = forecasts - actuals
        metrics = {
            'rmse': np.sqrt((errors**2).mean()),
            'mae': errors.abs().mean(),
            'coverage_90': self.compute_coverage(forecasts, actuals, 0.90)
        }

        # Alert if performance degraded
        if metrics['rmse'] > self.threshold:
            send_alert("Model performance degraded - consider re-estimation")

        return metrics
```

### Feature 5: Interactive Dashboard

**Requirements:**
- Real-time updates (WebSocket or polling)
- Drill-down capabilities (click factor → see contributors)
- Downloadable reports (PDF/Excel)
- Mobile-responsive

**Key Components:**
```python
# Dash app
import dash
from dash import dcc, html, Input, Output

app = dash.Dash(__name__)

app.layout = html.Div([
    # Header
    html.H1("Real-Time Economic Nowcasting"),

    # Nowcast gauge
    dcc.Graph(id='nowcast-gauge'),

    # Factor time series
    dcc.Graph(id='factors-chart'),

    # News feed
    html.Div(id='news-feed'),

    # Auto-update every 60 seconds
    dcc.Interval(id='interval', interval=60*1000)
])

@app.callback(
    Output('nowcast-gauge', 'figure'),
    Input('interval', 'n_intervals')
)
def update_nowcast(n):
    # Fetch latest nowcast from API
    response = requests.get('http://api:8000/nowcast/GDP')
    data = response.json()

    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = data['nowcast'],
        title = {'text': "GDP Q3 2024 Nowcast"},
        delta = {'reference': data['previous']},
        gauge = {'axis': {'range': [-2, 6]}}
    ))

    return fig
```

---

## Tasks

### Phase 1: Backend Infrastructure (3-4 hours)

**Task 1.1:** Setup database schema
```sql
-- Vintages table
CREATE TABLE data_vintages (
    id SERIAL PRIMARY KEY,
    series_id VARCHAR(50),
    vintage_date DATE,
    observation_date DATE,
    value FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Nowcasts table
CREATE TABLE nowcasts (
    id SERIAL PRIMARY KEY,
    target_variable VARCHAR(50),
    target_period DATE,
    vintage_date DATE,
    nowcast FLOAT,
    lower_90 FLOAT,
    upper_90 FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Model versions
CREATE TABLE model_versions (
    id SERIAL PRIMARY KEY,
    version VARCHAR(20),
    n_factors INT,
    parameters JSONB,
    performance_metrics JSONB,
    is_active BOOLEAN,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Task 1.2:** Implement REST API
```python
# backend/api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Nowcasting API")

class NowcastResponse(BaseModel):
    target_variable: str
    target_period: str
    nowcast: float
    lower_90: float
    upper_90: float
    vintage_date: str

@app.get("/nowcast/{variable}", response_model=NowcastResponse)
async def get_nowcast(variable: str):
    """Get latest nowcast for a variable."""
    # TODO: Query database for latest nowcast
    pass

@app.get("/factors/latest")
async def get_latest_factors():
    """Get current factor values."""
    # TODO: Return latest filtered factors
    pass

@app.get("/news")
async def get_news_feed():
    """Get recent data releases and their impacts."""
    # TODO: Return news decomposition
    pass
```

**Task 1.3:** Setup data pipeline with Airflow
```python
# pipelines/data_ingestion.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-team',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'daily_data_ingestion',
    default_args=default_args,
    schedule_interval='0 10 * * *',  # 10 AM daily
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:

    fetch_fred = PythonOperator(
        task_id='fetch_fred',
        python_callable=fetch_fred_data
    )

    validate_data = PythonOperator(
        task_id='validate_data',
        python_callable=validate_and_clean
    )

    store_vintage = PythonOperator(
        task_id='store_vintage',
        python_callable=save_to_database
    )

    update_model = PythonOperator(
        task_id='update_model',
        python_callable=update_factors_and_nowcast
    )

    fetch_fred >> validate_data >> store_vintage >> update_model
```

### Phase 2: Advanced Nowcasting (3-4 hours)

**Task 2.1:** Implement mixed-frequency Kalman filter
**Task 2.2:** Add news decomposition
**Task 2.3:** Generate density forecasts (not just point estimates)
**Task 2.4:** Implement real-time backtest evaluation

### Phase 3: Dashboard (2-3 hours)

**Task 3.1:** Build interactive charts with Plotly
**Task 3.2:** Add drill-down functionality
**Task 3.3:** Implement real-time updates
**Task 3.4:** Create downloadable reports

### Phase 4: Production Deployment (2-3 hours)

**Task 4.1:** Containerize with Docker
**Task 4.2:** Setup CI/CD pipeline
**Task 4.3:** Deploy to cloud (AWS/GCP)
**Task 4.4:** Configure monitoring and alerts

---

## Success Criteria

1. **System uptime:** > 99% availability
2. **Data freshness:** All data fetched within 2 hours of release
3. **Forecast accuracy:** RMSE < 0.8pp for GDP nowcasts
4. **API latency:** < 200ms for nowcast queries
5. **Dashboard load time:** < 2 seconds

---

## Extensions

1. **Multi-model ensemble:** Combine DFM with ML models
2. **Scenario analysis:** "What-if" tool for user-defined shocks
3. **Explainable AI:** SHAP values for factor contributions
4. **Mobile app:** iOS/Android app for alerts
5. **Collaboration features:** Team sharing and annotations

---

## Example Dashboard

**Homepage:**
```
┌────────────────────────────────────────────────────┐
│  Real-Time Economic Dashboard            [Login]   │
├────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐│
│  │  GDP Growth  │  │  Inflation   │  │ Unemp.   ││
│  │   2.8% ↑     │  │   3.2% ↓     │  │  3.7% →  ││
│  │  [Gauge]     │  │  [Gauge]     │  │ [Gauge]  ││
│  └──────────────┘  └──────────────┘  └──────────┘│
│                                                    │
│  Latest News:                                      │
│  ✓ Retail sales (+0.5%) added 0.2pp to GDP        │
│  ✗ Employment report (-20K) subtracted 0.1pp      │
│  ✓ Industrial production (+0.3%) added 0.15pp     │
│                                                    │
│  [Factors Chart - Time Series]                    │
│  [Historical Accuracy - Line Chart]               │
│  [Download Report]  [View API Docs]               │
└────────────────────────────────────────────────────┘
```

---

## Resources

- **FastAPI Docs:** https://fastapi.tiangolo.com/
- **Dash by Plotly:** https://dash.plotly.com/
- **Airflow:** https://airflow.apache.org/
- **Docker:** https://docs.docker.com/
- **AWS Deployment:** https://aws.amazon.com/getting-started/

---

This is your capstone project - showcase your skills and build something production-ready!
