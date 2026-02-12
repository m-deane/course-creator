# Deployment Guide: Production Commodity Allocator

## Overview

This guide walks through deploying the regime-aware bandit allocator for live trading. The system runs weekly, allocating capital across commodities based on adaptive learning and regime detection.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                 WEEKLY PIPELINE                      │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Trigger: Monday 9:00 AM EST (cron/Airflow)        │
│     ↓                                               │
│  1. Data Ingestion                                  │
│     ├─ Fetch commodity prices (yfinance/API)       │
│     ├─ Load macro indicators (FRED)                │
│     └─ Validate data quality                       │
│     ↓                                               │
│  2. Feature Pipeline                                │
│     ├─ Extract regime features                     │
│     ├─ Compute contextual vector                   │
│     └─ Log feature values                          │
│     ↓                                               │
│  3. Allocation Engine                               │
│     ├─ LinUCB prediction                           │
│     ├─ Apply guardrails                            │
│     └─ Generate final allocation                   │
│     ↓                                               │
│  4. Execution                                       │
│     ├─ Generate trade orders                       │
│     ├─ Submit to broker API                        │
│     └─ Confirm execution                           │
│     ↓                                               │
│  5. Monitoring & Reporting                          │
│     ├─ Log decision                                │
│     ├─ Update monitoring dashboard                 │
│     ├─ Generate weekly report                      │
│     └─ Send alerts if needed                       │
│                                                      │
└─────────────────────────────────────────────────────┘
```

## Prerequisites

### 1. Environment Setup

```bash
# Create production environment
conda create -n bandit-prod python=3.11
conda activate bandit-prod

# Install dependencies
pip install numpy pandas scipy matplotlib
pip install yfinance fredapi  # Data sources
pip install pyyaml python-dotenv  # Configuration
pip install schedule  # Scheduling (or use cron/Airflow)

# Optional: Production tools
pip install sentry-sdk  # Error tracking
pip install boto3  # AWS for logging/storage
```

### 2. Configuration Management

Create `config/production.yaml`:

```yaml
# Production Configuration
allocator:
  commodities:
    - WTI
    - Gold
    - Copper
    - NatGas
    - Corn

  core_pct: 0.80
  bandit_pct: 0.20

  guardrails:
    max_position: 0.40
    min_position: 0.05
    max_drawdown: 0.15
    vol_scale_threshold: 30

  linucb:
    alpha: 1.0
    lambda_reg: 1.0

data:
  sources:
    prices: yfinance
    macro: fred

  tickers:
    WTI: CL=F
    Gold: GC=F
    Copper: HG=F
    NatGas: NG=F
    Corn: ZC=F

  lookback_days: 365

execution:
  broker: interactive_brokers  # or alpaca, etc.
  api_endpoint: ${BROKER_API_URL}
  account_id: ${ACCOUNT_ID}
  dry_run: false  # Set true for paper trading

monitoring:
  log_dir: ./logs
  dashboard_url: ${GRAFANA_URL}
  alerts:
    - type: email
      recipients:
        - trader@firm.com
    - type: slack
      webhook: ${SLACK_WEBHOOK}

schedule:
  day: monday
  time: "09:00"
  timezone: America/New_York
```

### 3. Environment Variables

Create `.env` file (NEVER commit this):

```bash
# Broker API
BROKER_API_URL=https://api.broker.com
BROKER_API_KEY=your_api_key_here
ACCOUNT_ID=your_account_id

# Data Sources
FRED_API_KEY=your_fred_key

# Monitoring
SLACK_WEBHOOK=https://hooks.slack.com/services/YOUR/WEBHOOK
SENTRY_DSN=https://your-sentry-dsn

# AWS (for logging/storage)
AWS_ACCESS_KEY=your_key
AWS_SECRET_KEY=your_secret
AWS_S3_BUCKET=bandit-logs
```

## Deployment Options

### Option 1: Cron (Simple)

For single-server deployment:

```bash
# Add to crontab
crontab -e

# Run every Monday at 9 AM EST
0 9 * * 1 /home/user/bandit-prod/run_allocator.sh >> /var/log/bandit.log 2>&1
```

Create `run_allocator.sh`:

```bash
#!/bin/bash
set -e

# Activate environment
source /home/user/miniconda3/bin/activate bandit-prod

# Run allocator
cd /home/user/bandit-prod
python -m allocator.main --config config/production.yaml

# Check for errors
if [ $? -eq 0 ]; then
    echo "✅ Allocation completed successfully"
else
    echo "❌ Allocation failed" | mail -s "Bandit Alert" trader@firm.com
fi
```

### Option 2: Airflow (Recommended for Production)

Create `dags/commodity_allocator.py`:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
sys.path.append('/home/user/bandit-prod')

from allocator.main import run_allocation_pipeline

default_args = {
    'owner': 'trading',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['trader@firm.com'],
    'email_on_failure': True,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'commodity_allocator',
    default_args=default_args,
    schedule_interval='0 9 * * 1',  # Every Monday 9 AM
    catchup=False
)

def load_data(**context):
    from allocator.data import load_commodity_data
    data = load_commodity_data()
    return data.to_json()

def extract_features(**context):
    from allocator.features import FeaturePipeline
    data = context['task_instance'].xcom_pull(task_ids='load_data')
    pipeline = FeaturePipeline()
    features = pipeline.extract(data)
    return features

def allocate(**context):
    from allocator.engine import run_allocation
    features = context['task_instance'].xcom_pull(task_ids='features')
    allocation = run_allocation(features)
    return allocation

def execute_trades(**context):
    from allocator.execution import execute
    allocation = context['task_instance'].xcom_pull(task_ids='allocate')
    results = execute(allocation)
    return results

def generate_report(**context):
    from allocator.reporting import generate_weekly_report
    results = context['task_instance'].xcom_pull(task_ids='execute')
    generate_weekly_report(results)

# Define tasks
t1 = PythonOperator(task_id='load_data', python_callable=load_data, dag=dag)
t2 = PythonOperator(task_id='features', python_callable=extract_features, dag=dag)
t3 = PythonOperator(task_id='allocate', python_callable=allocate, dag=dag)
t4 = PythonOperator(task_id='execute', python_callable=execute_trades, dag=dag)
t5 = PythonOperator(task_id='report', python_callable=generate_report, dag=dag)

t1 >> t2 >> t3 >> t4 >> t5
```

### Option 3: Cloud Deployment (AWS Lambda / GCP Cloud Functions)

For serverless deployment:

```python
# lambda_handler.py
import json
import boto3
from allocator.main import run_allocation_pipeline

def lambda_handler(event, context):
    """
    AWS Lambda handler for weekly allocation.
    Triggered by CloudWatch Events (cron).
    """
    try:
        # Run allocation
        results = run_allocation_pipeline()

        # Store results in S3
        s3 = boto3.client('s3')
        s3.put_object(
            Bucket='bandit-logs',
            Key=f"allocations/{datetime.now().isoformat()}.json",
            Body=json.dumps(results)
        )

        return {
            'statusCode': 200,
            'body': json.dumps('Allocation completed')
        }
    except Exception as e:
        # Send alert
        sns = boto3.client('sns')
        sns.publish(
            TopicArn='arn:aws:sns:us-east-1:123456:bandit-alerts',
            Subject='Allocation Failed',
            Message=str(e)
        )
        raise
```

CloudWatch Events Rule (cron):
```json
{
  "schedule": "cron(0 14 ? * MON *)",
  "targets": [{
    "arn": "arn:aws:lambda:us-east-1:123456:function:commodity-allocator"
  }]
}
```

## Data Source Management

### Commodity Prices (yfinance)

```python
# allocator/data.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def load_commodity_data(tickers, lookback_days=365):
    """Load commodity data with fallback and validation."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    data = {}
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)

            # Validate
            if len(df) < 100:
                raise ValueError(f"Insufficient data for {name}")

            if df['Adj Close'].isna().sum() > 10:
                raise ValueError(f"Too many missing values for {name}")

            data[name] = df['Adj Close']

        except Exception as e:
            print(f"❌ Error loading {name}: {e}")
            # Alert and fail (don't trade without data!)
            raise

    return pd.DataFrame(data)
```

### Macro Data (FRED)

```python
from fredapi import Fred

def load_macro_indicators(fred_api_key):
    """Load supporting macro data."""
    fred = Fred(api_key=fred_api_key)

    indicators = {
        'VIX': 'VIXCLS',
        'DXY': 'DTWEXBGS',  # Dollar index
        'EFFR': 'EFFR'  # Fed funds rate
    }

    data = {}
    for name, series_id in indicators.items():
        try:
            data[name] = fred.get_series(series_id, observation_start='2023-01-01')
        except:
            print(f"⚠️  Could not load {name}, using default")
            data[name] = None

    return data
```

## Monitoring & Alerting

### Logging

```python
# allocator/logging.py
import logging
import json
from datetime import datetime

def setup_logging(log_dir='./logs'):
    """Configure structured logging."""

    # File handler
    log_file = f"{log_dir}/allocator_{datetime.now():%Y%m%d}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger('allocator')

def log_decision(logger, allocation, features, guardrails_fired):
    """Log allocation decision with context."""
    logger.info(json.dumps({
        'event': 'allocation_decision',
        'timestamp': datetime.now().isoformat(),
        'allocation': allocation.tolist(),
        'features': features,
        'guardrails': guardrails_fired,
    }))
```

### Alerts

```python
# allocator/alerts.py
import requests
from email.mime.text import MIMEText
import smtplib

def send_slack_alert(webhook_url, message):
    """Send alert to Slack."""
    payload = {'text': message}
    requests.post(webhook_url, json=payload)

def send_email_alert(recipients, subject, body):
    """Send email alert."""
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = 'bandit@firm.com'
    msg['To'] = ', '.join(recipients)

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(user, password)
        server.send_message(msg)

def check_and_alert(guardrails_fired, performance_metrics):
    """Check conditions and send alerts."""
    if 'CIRCUIT_BREAKER' in guardrails_fired:
        send_slack_alert(
            webhook,
            "🚨 CIRCUIT BREAKER ACTIVATED - Large drawdown detected"
        )

    if performance_metrics['sharpe'] < 0.5:
        send_email_alert(
            ['trader@firm.com'],
            'Low Sharpe Alert',
            f"Sharpe ratio dropped to {performance_metrics['sharpe']:.2f}"
        )
```

## Testing Before Live Deployment

### 1. Paper Trading

Run in dry-run mode for 4-8 weeks:

```yaml
# config/paper.yaml
execution:
  dry_run: true
  log_trades: true
```

Monitor:
- Decision quality
- Guardrail activations
- Feature pipeline stability
- Execution timing

### 2. Stress Testing

Test extreme scenarios:

```python
# tests/test_stress.py
def test_2008_crisis():
    """Test on 2008 financial crisis data."""
    data = load_historical_data('2008-09-01', '2008-12-31')
    allocator = run_backtest(data)
    assert allocator.max_drawdown < 0.25  # Survived with < 25% loss

def test_vol_spike():
    """Test behavior when VIX spikes to 80."""
    # Simulate extreme volatility
    synthetic_data = generate_crisis_scenario(vix=80)
    allocator = run_backtest(synthetic_data)
    assert all(allocator.allocations < 0.3)  # Reduced exposure
```

## Production Checklist

Before going live:

- [ ] Backtested on 2+ years of data
- [ ] Paper traded for 4+ weeks
- [ ] All guardrails tested and validated
- [ ] Logging and monitoring operational
- [ ] Alerts tested and routed correctly
- [ ] Data sources validated and redundant
- [ ] Configuration version-controlled
- [ ] Rollback procedure documented
- [ ] Manual override process defined
- [ ] Team trained on system operation

## Maintenance

### Weekly

- Review allocation report
- Check guardrail activations
- Verify data quality

### Monthly

- Analyze prediction accuracy
- Review feature importance
- Assess regime detection quality
- Check for model drift

### Quarterly

- Retrain/recalibrate model
- Update guardrail thresholds
- Review and update features
- Stress test on recent data

## Troubleshooting

### Issue: Data not loading

```bash
# Check data sources
python -c "import yfinance; yf.download('CL=F')"

# Verify API keys
echo $FRED_API_KEY
```

### Issue: Allocation fails

Check logs:
```bash
tail -f logs/allocator_$(date +%Y%m%d).log
```

Common causes:
- Missing data
- Matrix inversion failure (increase lambda_reg)
- Guardrail deadlock (conflicting constraints)

### Issue: Poor performance

- Check feature drift (are regimes still predictive?)
- Verify guardrails not too restrictive
- Compare to equal-weight benchmark
- Analyze by regime (is model good in all regimes?)

---

**You're ready to deploy.** Start with paper trading, monitor closely, and gradually increase confidence in the system.
