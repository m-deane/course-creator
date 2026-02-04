# Deployment Guide: GDP Nowcasting Model

This guide shows you how to deploy your GDP nowcasting model to production, enabling automated updates and real-time monitoring.

---

## Deployment Options

### Option 1: Scheduled Local Execution (Easiest)
Run the model automatically on your local machine using cron/Task Scheduler.

### Option 2: Cloud Deployment (Recommended)
Deploy to AWS Lambda, Google Cloud Functions, or Azure Functions for serverless execution.

### Option 3: Web Dashboard (Most Professional)
Create an interactive Streamlit/Dash dashboard hosted on cloud.

---

## Option 1: Scheduled Local Execution

### Setup (macOS/Linux)

**1. Create a shell script to run the model:**

```bash
#!/bin/bash
# File: run_nowcast.sh

# Activate environment
source ~/anaconda3/bin/activate gdp-nowcast

# Run model
cd /path/to/project_1_beginner
python solution.py >> logs/nowcast_$(date +%Y%m%d).log 2>&1

# Optional: Email results
if [ $? -eq 0 ]; then
    echo "Nowcast completed successfully at $(date)" | \
        mail -s "GDP Nowcast Update" your.email@example.com
fi
```

**2. Make it executable:**
```bash
chmod +x run_nowcast.sh
```

**3. Schedule with cron (runs every Monday at 9 AM):**
```bash
crontab -e
```

Add line:
```
0 9 * * 1 /path/to/run_nowcast.sh
```

### Setup (Windows)

**1. Create PowerShell script (`run_nowcast.ps1`):**

```powershell
# Activate conda environment
conda activate gdp-nowcast

# Run model
cd C:\path\to\project_1_beginner
python solution.py

# Email notification (optional, requires Send-MailMessage setup)
if ($LASTEXITCODE -eq 0) {
    Send-MailMessage -To "your.email@example.com" `
        -From "nowcast@yourorg.com" `
        -Subject "GDP Nowcast Update" `
        -Body "Nowcast completed successfully" `
        -SmtpServer "smtp.gmail.com"
}
```

**2. Schedule with Task Scheduler:**
- Open Task Scheduler
- Create Basic Task
- Trigger: Weekly (Monday, 9:00 AM)
- Action: Start a program
  - Program: `powershell.exe`
  - Arguments: `-File C:\path\to\run_nowcast.ps1`

---

## Option 2: Cloud Deployment (AWS Lambda)

### Prerequisites
- AWS account
- AWS CLI installed and configured
- Docker installed (for packaging dependencies)

### Step 1: Prepare Lambda Function

**Create `lambda_function.py`:**

```python
import json
import boto3
import pandas as pd
from pathlib import Path
import tempfile

# Import your nowcasting code
from solution import (
    fetch_fred_data, transform_series, standardize_data,
    StockWatsonFactorModel, GDPNowcaster
)

def lambda_handler(event, context):
    """AWS Lambda handler for nowcasting."""

    # Get API key from environment or AWS Secrets Manager
    import os
    api_key = os.environ['FRED_API_KEY']

    # Run nowcasting pipeline
    try:
        # Fetch data
        indicators = ['INDPRO', 'PAYEMS', 'CPIAUCSL', ...]  # Your list
        data_raw = fetch_fred_data(api_key, indicators)

        # Transform
        data_transformed = transform_series(data_raw)
        data_std = standardize_data(data_transformed)

        # Estimate model
        model = StockWatsonFactorModel(n_factors=3)
        model.fit(data_std)

        # Nowcast
        nowcaster = GDPNowcaster(model)
        # ... (complete the pipeline)

        # Save results to S3
        s3 = boto3.client('s3')
        results = {
            'nowcast': float(nowcast_result['nowcast']),
            'lower': float(nowcast_result['lower']),
            'upper': float(nowcast_result['upper']),
            'timestamp': str(pd.Timestamp.now())
        }

        s3.put_object(
            Bucket='your-nowcast-bucket',
            Key='latest_nowcast.json',
            Body=json.dumps(results)
        )

        return {
            'statusCode': 200,
            'body': json.dumps(results)
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

### Step 2: Package Dependencies

**Create `Dockerfile`:**

```dockerfile
FROM public.ecr.aws/lambda/python:3.11

# Copy requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy code
COPY solution.py lambda_function.py ./

CMD ["lambda_function.lambda_handler"]
```

**requirements.txt:**
```
pandas
numpy
statsmodels
scikit-learn
fredapi
boto3
```

### Step 3: Build and Deploy

```bash
# Build Docker image
docker build -t gdp-nowcast-lambda .

# Tag for ECR
aws ecr create-repository --repository-name gdp-nowcast
docker tag gdp-nowcast-lambda:latest \
    123456789012.dkr.ecr.us-east-1.amazonaws.com/gdp-nowcast:latest

# Push to ECR
aws ecr get-login-password | docker login --username AWS \
    --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/gdp-nowcast:latest

# Create Lambda function
aws lambda create-function \
    --function-name gdp-nowcast \
    --package-type Image \
    --code ImageUri=123456789012.dkr.ecr.us-east-1.amazonaws.com/gdp-nowcast:latest \
    --role arn:aws:iam::123456789012:role/lambda-execution-role \
    --timeout 300 \
    --memory-size 1024
```

### Step 4: Schedule Execution

**Create EventBridge rule (runs every Monday at 9 AM EST):**

```bash
aws events put-rule \
    --name weekly-nowcast \
    --schedule-expression "cron(0 14 ? * MON *)"

aws lambda add-permission \
    --function-name gdp-nowcast \
    --statement-id weekly-nowcast-event \
    --action lambda:InvokeFunction \
    --principal events.amazonaws.com \
    --source-arn arn:aws:events:us-east-1:123456789012:rule/weekly-nowcast

aws events put-targets \
    --rule weekly-nowcast \
    --targets "Id"="1","Arn"="arn:aws:lambda:us-east-1:123456789012:function:gdp-nowcast"
```

---

## Option 3: Interactive Web Dashboard

### Using Streamlit

**Create `app.py`:**

```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from solution import (
    load_config, fetch_fred_data, transform_series,
    standardize_data, StockWatsonFactorModel, GDPNowcaster
)

st.set_page_config(page_title="GDP Nowcasting", layout="wide")

st.title("🇺🇸 GDP Nowcasting Dashboard")
st.markdown("Real-time GDP growth estimates using dynamic factor models")

# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("FRED API Key", type="password")
    n_factors = st.slider("Number of Factors", 1, 5, 3)
    update_button = st.button("Update Nowcast")

# Main content
if update_button and api_key:
    with st.spinner("Fetching data and estimating model..."):
        # Run pipeline
        config = load_config()
        data_raw = fetch_fred_data(api_key, config['indicators'])
        data_std = standardize_data(transform_series(data_raw))

        model = StockWatsonFactorModel(n_factors=n_factors)
        model.fit(data_std)

        # Display results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Current Nowcast", "2.8%", "+0.3pp")
        with col2:
            st.metric("90% CI", "[1.5%, 4.1%]")
        with col3:
            st.metric("Model R²", "0.73")

        # Factor plot
        st.subheader("Estimated Factors")
        fig = go.Figure()
        for i, col in enumerate(model.factors.columns):
            fig.add_trace(go.Scatter(
                x=model.factors.index,
                y=model.factors[col],
                name=f'Factor {i+1}'
            ))
        st.plotly_chart(fig, use_container_width=True)

        # Factor loadings
        st.subheader("Factor Interpretation")
        interpretations = model.get_factor_interpretation(data_std.columns)
        for factor, loadings in interpretations.items():
            st.write(f"**{factor}**")
            st.dataframe(loadings.to_frame('Loading'))

else:
    st.info("👈 Enter your FRED API key and click 'Update Nowcast'")
```

### Deploy Streamlit App

**Option 3a: Streamlit Community Cloud (Free)**

1. Push code to GitHub
2. Go to https://share.streamlit.io/
3. Connect repository
4. Deploy (auto-updates on git push)

**Option 3b: AWS/Heroku**

```bash
# Create Procfile
echo "web: streamlit run app.py --server.port=$PORT" > Procfile

# Deploy to Heroku
heroku create gdp-nowcast
git push heroku main
heroku ps:scale web=1
```

---

## Production Best Practices

### 1. Environment Variables

Store secrets securely:

```bash
# .env file (never commit!)
FRED_API_KEY=your_api_key_here
AWS_ACCESS_KEY_ID=your_aws_key
DATABASE_URL=postgresql://user:pass@host/db
```

Load in Python:
```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('FRED_API_KEY')
```

### 2. Error Handling and Logging

```python
import logging
import sentry_sdk  # Optional: error tracking

# Configure logging
logging.basicConfig(
    filename='nowcast.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Sentry for error tracking (optional)
sentry_sdk.init("your_sentry_dsn")

try:
    # Your nowcasting code
    result = run_nowcast()
    logging.info(f"Nowcast: {result['nowcast']:.2f}%")
except Exception as e:
    logging.error(f"Nowcast failed: {str(e)}")
    sentry_sdk.capture_exception(e)
    raise
```

### 3. Data Versioning

Track which data vintage was used:

```python
def save_data_vintage(data, nowcast_result):
    """Save data and results for reproducibility."""
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

    # Save data snapshot
    data.to_csv(f'data/vintage_{timestamp}.csv')

    # Save results
    results = {
        'timestamp': timestamp,
        'nowcast': nowcast_result,
        'data_hash': pd.util.hash_pandas_object(data).sum()
    }

    with open(f'output/results_{timestamp}.json', 'w') as f:
        json.dump(results, f)
```

### 4. Model Monitoring

Track model performance over time:

```python
def monitor_nowcast_accuracy():
    """Compare nowcasts to actual GDP releases."""
    nowcasts = pd.read_csv('output/nowcast_history.csv')
    actuals = pd.read_csv('data/gdp_actuals.csv')

    merged = nowcasts.merge(actuals, on='quarter')
    merged['error'] = merged['nowcast'] - merged['actual']

    rmse = (merged['error']**2).mean()**0.5

    # Alert if RMSE exceeds threshold
    if rmse > 1.5:
        send_alert(f"Model performance degraded: RMSE = {rmse:.2f}pp")
```

### 5. Automated Testing

```python
# tests/test_nowcast.py
import pytest
from solution import StockWatsonFactorModel

def test_factor_extraction():
    """Test that factor extraction works."""
    data = generate_test_data()
    model = StockWatsonFactorModel(n_factors=3)
    model.fit(data)

    assert model.factors.shape[1] == 3
    assert model.explained_variance.sum() > 0.5

def test_nowcast_bounds():
    """Test that nowcast is reasonable."""
    result = run_complete_nowcast()

    # GDP growth typically -5% to +10%
    assert -5 < result['nowcast'] < 10
    assert result['lower'] < result['nowcast'] < result['upper']
```

Run tests before deployment:
```bash
pytest tests/ --cov=solution
```

---

## Monitoring and Alerts

### Email Alerts

```python
import smtplib
from email.mime.text import MIMEText

def send_alert(subject, message):
    """Send email alert when issues occur."""
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = 'nowcast@yourorg.com'
    msg['To'] = 'team@yourorg.com'

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login('your_email@gmail.com', 'your_app_password')
        server.send_message(msg)

# Usage
if abs(nowcast_change) > 0.5:
    send_alert(
        "Large Nowcast Revision",
        f"GDP nowcast changed by {nowcast_change:.1f}pp"
    )
```

### Slack Integration

```python
import requests

def send_slack_notification(nowcast_result):
    """Post nowcast to Slack channel."""
    webhook_url = os.getenv('SLACK_WEBHOOK_URL')

    message = {
        "text": f"📊 *GDP Nowcast Update*",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"Current Q GDP: *{nowcast_result['nowcast']:.2f}%*\n"
                            f"90% CI: [{nowcast_result['lower']:.2f}%, {nowcast_result['upper']:.2f}%]"
                }
            }
        ]
    }

    requests.post(webhook_url, json=message)
```

---

## Cost Estimates

### AWS Lambda (Recommended for most users)
- **Compute:** ~$0.20/month (4 runs/month, 2GB RAM, 60s execution)
- **Storage (S3):** ~$0.01/month (store results)
- **Data transfer:** Negligible
- **Total:** **~$0.25/month**

### Streamlit Community Cloud
- **Free tier:** Unlimited public apps
- **Limitations:** 1 GB RAM, shared resources

### Self-hosted VM (AWS EC2 t3.small)
- **Cost:** ~$15/month (always-on)
- **Better for:** High-frequency updates, complex dashboards

---

## Troubleshooting

### Common Issues

**1. FRED API rate limits**
```
Error: 429 Too Many Requests
```
**Solution:** Cache data, limit requests to once per day

**2. Lambda timeout**
```
Error: Task timed out after 30.00 seconds
```
**Solution:** Increase timeout (max 900s), optimize code

**3. Missing data**
```
Error: ValueError: Input contains NaN
```
**Solution:** Implement robust missing data handling

---

## Next Steps

After deployment:

1. **Monitor performance** - Track RMSE vs actual GDP releases
2. **Iterate model** - Add new indicators, tune parameters
3. **Expand coverage** - Nowcast other variables (inflation, employment)
4. **Build API** - Expose nowcasts via REST API for other applications

---

## Resources

- **AWS Lambda Docs:** https://docs.aws.amazon.com/lambda/
- **Streamlit Deployment:** https://docs.streamlit.io/streamlit-community-cloud
- **GitHub Actions (CI/CD):** https://docs.github.com/en/actions
- **Monitoring Tools:** Sentry, Datadog, CloudWatch

---

**Questions?** Open an issue on GitHub or contact the course staff.
