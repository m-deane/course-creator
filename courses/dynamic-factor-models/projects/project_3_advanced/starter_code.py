"""
Real-Time Forecasting Dashboard - Starter Code
===============================================

This starter provides the skeleton for a production nowcasting system.
Complete the TODOs to build a full real-time dashboard.

Components:
1. Data pipeline (fetch + validate + store)
2. Mixed-frequency state-space model
3. News analyzer
4. REST API
5. Interactive dashboard
"""

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import sqlite3
from fastapi import FastAPI
from pydantic import BaseModel
import dash
from dash import dcc, html
import plotly.graph_objects as go

# ============================================================================
# DATABASE SETUP
# ============================================================================

class Database:
    """Simple SQLite database for storing vintages and nowcasts."""

    def __init__(self, db_path='data/nowcasting.db'):
        Path(db_path).parent.mkdir(exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        """Create database schema."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS data_vintages (
                series_id TEXT,
                vintage_date TEXT,
                observation_date TEXT,
                value REAL
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS nowcasts (
                target_variable TEXT,
                target_period TEXT,
                vintage_date TEXT,
                nowcast REAL,
                lower_90 REAL,
                upper_90 REAL
            )
        """)

        self.conn.commit()

    def save_nowcast(self, nowcast_dict):
        """Save nowcast to database."""
        # TODO: Implement
        pass

    def get_latest_nowcast(self, variable='GDP'):
        """Retrieve latest nowcast."""
        # TODO: Implement
        pass


# ============================================================================
# DATA PIPELINE
# ============================================================================

class DataFetcher:
    """Fetch data from FRED and other sources."""

    def __init__(self, api_key):
        self.api_key = api_key

    def fetch_latest(self, series_list):
        """
        Fetch latest data for all series.

        TODO: Implement with error handling and retry logic
        """
        pass

    def validate(self, data):
        """
        Validate data quality.

        Checks:
        - No all-NaN columns
        - Values in reasonable range
        - No sudden jumps > 5 std
        """
        # TODO: Implement validation
        pass


# ============================================================================
# MIXED-FREQUENCY MODEL
# ============================================================================

class MixedFrequencyNowcaster:
    """
    State-space model with mixed frequencies.

    Handles:
    - Daily financial indicators
    - Monthly real activity
    - Quarterly GDP
    """

    def __init__(self, n_factors=3):
        self.n_factors = n_factors

    def fit(self, data_dict):
        """
        Estimate model.

        data_dict: {
            'monthly': pd.DataFrame,
            'quarterly': pd.DataFrame
        }

        TODO: Implement mixed-frequency state-space estimation
        Hint: Use statsmodels.tsa.statespace or pykalman
        """
        pass

    def nowcast(self, current_data):
        """Generate nowcast using latest data."""
        # TODO: Run Kalman filter to get current state estimate
        pass

    def forecast(self, steps=4):
        """Forecast h quarters ahead."""
        # TODO: Forecast factor dynamics, then map to observables
        pass


# ============================================================================
# NEWS ANALYZER
# ============================================================================

class NewsAnalyzer:
    """Decompose forecast revisions into news from each indicator."""

    def compute_news(self, model, new_data, series_name):
        """
        Compute impact of new data release on nowcast.

        Returns: Change in nowcast attributable to this release.

        TODO: Implement using Kalman gain
        News = K * innovation
        """
        pass

    def generate_feed(self, news_dict):
        """
        Convert news decomposition to readable format.

        Example:
        "Strong retail sales (+0.5%) added 0.2pp to Q3 GDP nowcast"
        """
        feed = []
        for series, impact in news_dict.items():
            if abs(impact) > 0.05:  # Only show significant impacts
                direction = "added" if impact > 0 else "subtracted"
                feed.append(f"{series} {direction} {abs(impact):.2f}pp")

        return feed


# ============================================================================
# REST API
# ============================================================================

app = FastAPI(title="Nowcasting API")
db = Database()

class NowcastResponse(BaseModel):
    target_variable: str
    nowcast: float
    lower_90: float
    upper_90: float
    vintage_date: str


@app.get("/nowcast/{variable}", response_model=NowcastResponse)
def get_nowcast(variable: str):
    """
    Get latest nowcast for a variable.

    TODO: Query database and return latest nowcast
    """
    # Placeholder
    return NowcastResponse(
        target_variable=variable,
        nowcast=2.8,
        lower_90=1.5,
        upper_90=4.1,
        vintage_date=str(datetime.now())
    )


@app.get("/news")
def get_news():
    """
    Get recent data releases and their impacts.

    TODO: Return news feed from analyzer
    """
    return {
        "news": [
            {"series": "Retail Sales", "impact": 0.2, "direction": "up"},
            {"series": "Employment", "impact": -0.1, "direction": "down"}
        ]
    }


@app.get("/factors/latest")
def get_factors():
    """Get current factor estimates."""
    # TODO: Return latest filtered factors
    pass


# ============================================================================
# DASHBOARD
# ============================================================================

dash_app = dash.Dash(__name__)

dash_app.layout = html.Div([
    html.H1("Real-Time Economic Dashboard"),

    # Nowcast gauge
    html.Div([
        dcc.Graph(id='nowcast-gauge'),
    ]),

    # News feed
    html.Div([
        html.H3("Latest Updates"),
        html.Div(id='news-feed'),
    ]),

    # Factors chart
    html.Div([
        dcc.Graph(id='factors-chart'),
    ]),

    # Auto-update interval (every 60 seconds)
    dcc.Interval(id='interval', interval=60*1000, n_intervals=0)
])


@dash_app.callback(
    dash.dependencies.Output('nowcast-gauge', 'figure'),
    dash.dependencies.Input('interval', 'n_intervals')
)
def update_nowcast_gauge(n):
    """
    Update nowcast gauge chart.

    TODO: Fetch latest nowcast from API/database
    """
    # Placeholder data
    nowcast = 2.8
    previous = 2.5

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=nowcast,
        title={'text': "GDP Q3 2024 Nowcast (%)"},
        delta={'reference': previous, 'relative': False},
        gauge={
            'axis': {'range': [-2, 6]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-2, 0], 'color': "lightcoral"},
                {'range': [0, 2], 'color': "lightyellow"},
                {'range': [2, 6], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 3.0  # Potential threshold
            }
        }
    ))

    return fig


@dash_app.callback(
    dash.dependencies.Output('news-feed', 'children'),
    dash.dependencies.Input('interval', 'n_intervals')
)
def update_news_feed(n):
    """
    Update news feed with latest data releases.

    TODO: Fetch from news analyzer
    """
    # Placeholder news
    news_items = [
        "✓ Retail Sales (+0.5%): Added 0.2pp to GDP nowcast",
        "✗ Employment Report (-20K): Subtracted 0.1pp from GDP nowcast",
        "✓ Industrial Production (+0.3%): Added 0.15pp to GDP nowcast"
    ]

    return [html.P(item) for item in news_items]


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_pipeline():
    """
    Main nowcasting pipeline (run daily).

    1. Fetch latest data
    2. Update model
    3. Generate nowcast
    4. Save to database
    5. Send alerts if significant change
    """
    print("Running nowcasting pipeline...")

    # TODO: Implement full pipeline
    # 1. fetcher = DataFetcher(api_key)
    # 2. data = fetcher.fetch_latest(series_list)
    # 3. model.update(data)
    # 4. nowcast = model.nowcast()
    # 5. db.save_nowcast(nowcast)

    print("Pipeline complete!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'pipeline':
        # Run data pipeline
        run_pipeline()
    elif len(sys.argv) > 1 and sys.argv[1] == 'api':
        # Run API server
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # Run dashboard
        dash_app.run_server(debug=True, port=8050)
