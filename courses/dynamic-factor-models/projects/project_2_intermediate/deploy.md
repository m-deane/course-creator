# Deployment: Multi-Country Factor Model

## Overview

Deploy the multi-country factor model as a monitoring dashboard for international macroeconomic conditions.

---

## Deployment Architecture

```
┌─────────────────────┐
│  Data Sources       │
│  - FRED             │
│  - World Bank       │
│  - OECD             │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  Data Pipeline      │
│  (AWS Lambda)       │
│  - Fetch daily      │
│  - Transform        │
│  - Store to S3      │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  Model Estimation   │
│  (AWS Batch/SageMaker)
│  - Monthly update   │
│  - Factor extraction│
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  Dashboard          │
│  (Streamlit Cloud)  │
│  - Live factors     │
│  - Spillover viz    │
└─────────────────────┘
```

---

## Quick Start: Streamlit Dashboard

**app.py:**
```python
import streamlit as st
import plotly.graph_objects as go
from solution import HierarchicalFactorModel

st.title("🌍 Global Macro Monitor")

# Sidebar
with st.sidebar:
    st.header("Settings")
    n_global = st.slider("Global Factors", 1, 5, 2)
    update = st.button("Update Model")

# Main dashboard
if update:
    with st.spinner("Estimating model..."):
        model = HierarchicalFactorModel(n_global=n_global)
        # ... (run full pipeline)

    # Display global factors
    st.subheader("Global Factors")
    fig = create_factor_plot(model.global_factors)
    st.plotly_chart(fig)

    # Spillover network
    st.subheader("International Spillovers")
    network_fig = create_network_plot(spillover_matrix)
    st.plotly_chart(network_fig)
```

**Deploy:**
```bash
streamlit run app.py
# Or deploy to cloud:
# streamlit cloud deploy
```

---

## Production Deployment

See Project 1 deployment guide for AWS Lambda setup.

**Key differences for multi-country model:**

1. **Larger data volume** → Use S3 for caching
2. **Monthly updates** (not weekly) → Less frequent Lambda runs
3. **Longer computation** → Consider AWS Batch for model estimation

---

## Monitoring Alerts

**Alert conditions:**
```python
# 1. Spillover index spike
if spillover_index > historical_mean + 2*std:
    send_alert("High spillover detected")

# 2. Factor divergence
if global_factor_1 < -2:  # Recession signal
    send_alert("Global growth factor weak")

# 3. Contagion test
if contagion_detected(correlations):
    send_alert("Correlation structure changed")
```

---

## Use Cases

1. **Central banks:** Monitor global shocks vs domestic
2. **International portfolios:** Forecast correlations for diversification
3. **Trade policy:** Measure interconnectedness over time
4. **Research:** Track globalization trends

---

## Resources

- **ECB Global Model:** Similar multi-country framework
- **IMF Spillover Reports:** Methodology references
- **BIS International Banking Statistics:** Financial linkages data
