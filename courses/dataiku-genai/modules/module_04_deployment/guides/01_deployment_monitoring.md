# Deployment and Monitoring in Dataiku

## Deployment Options

### API Node Deployment

Deploy LLM applications as REST APIs:

```
┌──────────────────────────────────────────────────────────────┐
│                    Dataiku API Node                           │
├──────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  Endpoint 1 │  │  Endpoint 2 │  │  Endpoint 3 │          │
│  │  Q&A API    │  │  Analysis   │  │  Signals    │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                          │                                    │
│                    Load Balancer                              │
│                          │                                    │
│              ┌───────────┼───────────┐                       │
│              ▼           ▼           ▼                       │
│         Container 1  Container 2  Container 3                │
└──────────────────────────────────────────────────────────────┘
```

### Creating an API Service

```python
# In Dataiku API Designer

# 1. Create endpoint function
def process_commodity_query(query: str, commodity: str = None) -> dict:
    """Process commodity market query."""
    from dataiku.llm import LLM
    from dataiku.knowledge_bank import KnowledgeBank

    # Initialize
    kb = KnowledgeBank("commodity_reports_kb")
    llm = LLM("anthropic-claude")

    # Retrieve context
    filters = {"commodity": commodity} if commodity else None
    results = kb.search(query=query, top_k=5, filters=filters)

    context = "\n\n".join([r.text for r in results])

    # Generate response
    prompt = f"""Answer based on this context:
    {context}

    Question: {query}"""

    response = llm.complete(prompt, max_tokens=500)

    return {
        "answer": response.text,
        "sources": [r.metadata.get("source") for r in results],
        "confidence": "high" if results[0].score > 0.8 else "medium"
    }

# 2. Define endpoint schema
endpoint_config = {
    "name": "commodity_qa",
    "function": "process_commodity_query",
    "inputs": [
        {"name": "query", "type": "string", "required": True},
        {"name": "commodity", "type": "string", "required": False}
    ],
    "outputs": [
        {"name": "answer", "type": "string"},
        {"name": "sources", "type": "array"},
        {"name": "confidence", "type": "string"}
    ]
}
```

### Deployment Configuration

```yaml
# api_deployer_config.yaml
api_service:
  name: commodity-analysis-api
  version: "1.0.0"

  infrastructure:
    type: kubernetes
    replicas: 3
    resources:
      cpu: "2"
      memory: "4Gi"

  endpoints:
    - name: commodity_qa
      path: /api/v1/query
      method: POST
      rate_limit: 100  # requests per minute

    - name: batch_analysis
      path: /api/v1/batch
      method: POST
      timeout: 300  # seconds for long-running

  authentication:
    type: api_key
    header: X-API-Key

  monitoring:
    enabled: true
    metrics_endpoint: /metrics
```

## Monitoring Setup

### Built-in Metrics

Dataiku automatically tracks:

| Metric | Description |
|--------|-------------|
| `llm_requests_total` | Total LLM API calls |
| `llm_tokens_total` | Total tokens consumed |
| `llm_latency_seconds` | Response time histogram |
| `llm_errors_total` | Error count by type |
| `llm_cost_usd` | Estimated cost |

### Custom Metrics

```python
import dataiku
from dataiku.llm import LLM
from dataiku.monitoring import MetricsClient
import time

class MonitoredLLM:
    """LLM wrapper with custom metrics."""

    def __init__(self, connection: str):
        self.llm = LLM(connection)
        self.metrics = MetricsClient()

    def complete(self, prompt: str, **kwargs) -> dict:
        """Complete with metrics tracking."""
        start_time = time.time()

        try:
            response = self.llm.complete(prompt, **kwargs)

            # Record metrics
            latency = time.time() - start_time
            self.metrics.gauge("llm_latency", latency, tags={
                "connection": self.llm.connection_name,
                "status": "success"
            })

            self.metrics.increment("llm_requests", tags={
                "connection": self.llm.connection_name,
                "status": "success"
            })

            self.metrics.increment("llm_tokens",
                value=response.usage.total_tokens,
                tags={"connection": self.llm.connection_name}
            )

            return response

        except Exception as e:
            self.metrics.increment("llm_requests", tags={
                "connection": self.llm.connection_name,
                "status": "error",
                "error_type": type(e).__name__
            })
            raise
```

### Dashboard Setup

Create monitoring dashboard in Dataiku:

```python
# dashboard_config.py
dashboard = {
    "name": "LLM Operations Dashboard",
    "tiles": [
        {
            "type": "metric",
            "title": "Requests (24h)",
            "metric": "llm_requests_total",
            "aggregation": "sum",
            "time_range": "24h"
        },
        {
            "type": "timeseries",
            "title": "Latency P95",
            "metric": "llm_latency_seconds",
            "aggregation": "p95",
            "time_range": "24h",
            "interval": "1h"
        },
        {
            "type": "metric",
            "title": "Token Usage",
            "metric": "llm_tokens_total",
            "aggregation": "sum",
            "time_range": "24h"
        },
        {
            "type": "timeseries",
            "title": "Error Rate",
            "metric": "llm_errors_total",
            "aggregation": "rate",
            "time_range": "24h"
        },
        {
            "type": "metric",
            "title": "Estimated Cost",
            "metric": "llm_cost_usd",
            "aggregation": "sum",
            "time_range": "24h"
        }
    ]
}
```

## Alerting

### Alert Configuration

```yaml
# alerts_config.yaml
alerts:
  - name: high_error_rate
    condition: |
      rate(llm_errors_total[5m]) / rate(llm_requests_total[5m]) > 0.05
    severity: critical
    notification:
      - type: email
        recipients: [ops-team@company.com]
      - type: slack
        channel: "#commodity-alerts"

  - name: high_latency
    condition: |
      histogram_quantile(0.95, llm_latency_seconds) > 10
    severity: warning
    notification:
      - type: slack
        channel: "#commodity-alerts"

  - name: daily_cost_exceeded
    condition: |
      sum(llm_cost_usd{period="daily"}) > 100
    severity: warning
    notification:
      - type: email
        recipients: [finance@company.com]

  - name: rate_limit_approaching
    condition: |
      rate(llm_requests_total[1m]) > 80  # 80% of 100/min limit
    severity: info
    notification:
      - type: slack
        channel: "#commodity-ops"
```

### Programmatic Alerts

```python
from dataiku.monitoring import AlertManager

alert_manager = AlertManager()

def check_llm_health():
    """Check LLM system health and raise alerts."""

    metrics = get_recent_metrics(minutes=5)

    # Check error rate
    error_rate = metrics['errors'] / max(metrics['requests'], 1)
    if error_rate > 0.05:
        alert_manager.raise_alert(
            name="high_error_rate",
            severity="critical",
            message=f"LLM error rate at {error_rate:.1%}",
            metadata={"error_rate": error_rate}
        )

    # Check latency
    if metrics['p95_latency'] > 10:
        alert_manager.raise_alert(
            name="high_latency",
            severity="warning",
            message=f"P95 latency at {metrics['p95_latency']:.1f}s"
        )

    # Check costs
    if metrics['daily_cost'] > 100:
        alert_manager.raise_alert(
            name="cost_exceeded",
            severity="warning",
            message=f"Daily cost at ${metrics['daily_cost']:.2f}"
        )
```

## Cost Management

### Cost Tracking

```python
from dataiku.llm import LLM
from dataiku.monitoring import MetricsClient
from datetime import datetime, timedelta

class CostTracker:
    """Track and manage LLM costs."""

    # Cost per 1M tokens by model
    COSTS = {
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
        "gpt-4o": {"input": 2.5, "output": 10.0},
        "claude-3-haiku": {"input": 0.25, "output": 1.25}
    }

    def __init__(self):
        self.metrics = MetricsClient()
        self.daily_costs = {}

    def record_usage(self, model: str, input_tokens: int, output_tokens: int):
        """Record token usage and cost."""
        costs = self.COSTS.get(model, {"input": 5.0, "output": 15.0})

        cost = (
            input_tokens * costs["input"] / 1_000_000 +
            output_tokens * costs["output"] / 1_000_000
        )

        # Track in metrics
        self.metrics.increment("llm_cost_usd", value=cost, tags={
            "model": model
        })

        # Track daily
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in self.daily_costs:
            self.daily_costs[today] = 0
        self.daily_costs[today] += cost

        return cost

    def get_daily_cost(self, date: str = None) -> float:
        """Get cost for a specific day."""
        date = date or datetime.now().strftime("%Y-%m-%d")
        return self.daily_costs.get(date, 0)

    def check_budget(self, daily_limit: float) -> dict:
        """Check if within daily budget."""
        today = datetime.now().strftime("%Y-%m-%d")
        current = self.daily_costs.get(today, 0)

        return {
            "current_cost": current,
            "daily_limit": daily_limit,
            "remaining": daily_limit - current,
            "percent_used": (current / daily_limit) * 100,
            "within_budget": current < daily_limit
        }
```

### Budget Enforcement

```python
class BudgetEnforcedLLM:
    """LLM wrapper with budget enforcement."""

    def __init__(self, connection: str, daily_budget: float):
        self.llm = LLM(connection)
        self.cost_tracker = CostTracker()
        self.daily_budget = daily_budget

    def complete(self, prompt: str, **kwargs) -> dict:
        """Complete with budget check."""

        # Check budget
        budget_status = self.cost_tracker.check_budget(self.daily_budget)
        if not budget_status["within_budget"]:
            raise Exception(
                f"Daily budget exceeded: ${budget_status['current_cost']:.2f} "
                f"of ${self.daily_budget:.2f}"
            )

        # Make request
        response = self.llm.complete(prompt, **kwargs)

        # Record cost
        model = kwargs.get("model", "claude-sonnet-4-20250514")
        cost = self.cost_tracker.record_usage(
            model=model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens
        )

        return response
```

## Production Checklist

### Pre-Deployment

- [ ] All test cases passing
- [ ] Rate limits configured appropriately
- [ ] Error handling implemented
- [ ] Logging enabled
- [ ] Cost tracking active
- [ ] Alerts configured
- [ ] API authentication set up
- [ ] Backup LLM connection configured

### Post-Deployment

- [ ] Monitor error rates
- [ ] Track latency percentiles
- [ ] Review daily costs
- [ ] Check rate limit utilization
- [ ] Validate response quality
- [ ] Review audit logs

## Key Takeaways

1. **API Node** provides scalable deployment for LLM applications

2. **Comprehensive monitoring** is essential - track requests, latency, errors, and costs

3. **Alerting** enables proactive issue detection

4. **Cost management** prevents budget overruns

5. **Production checklist** ensures nothing is missed before go-live
