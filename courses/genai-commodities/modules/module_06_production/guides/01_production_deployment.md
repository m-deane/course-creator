# Production Deployment for Commodity Gen AI

> **Reading time:** ~7 min | **Module:** Module 6: Production | **Prerequisites:** Modules 0-5

<div class="callout-key">

**Key Concept Summary:** Moving from prototype to production requires addressing reliability, cost, latency, and monitoring. This guide covers production deployment patterns for commodity trading Gen AI systems.

</div>

## Overview

Moving from prototype to production requires addressing reliability, cost, latency, and monitoring. This guide covers production deployment patterns for commodity trading Gen AI systems.

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## Architecture Patterns

### Production Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Load Balancer                               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                      API Gateway                                 │
│  - Rate limiting  - Authentication  - Request validation         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Report    │  │  Sentiment  │  │   Signal    │
│  Processor  │  │  Analyzer   │  │  Generator  │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┼────────────────┘
                        ▼
              ┌─────────────────┐
              │   Message Queue │
              │   (Redis/SQS)   │
              └────────┬────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   Worker 1  │ │   Worker 2  │ │   Worker N  │
└─────────────┘ └─────────────┘ └─────────────┘
         │             │             │
         └─────────────┼─────────────┘
                       ▼
              ┌─────────────────┐
              │    Database     │
              │ (PostgreSQL/    │
              │  TimescaleDB)   │
              └─────────────────┘
```

## Reliability Patterns

### Retry with Exponential Backoff

LLM APIs fail transiently — rate limits, timeouts, and overloaded endpoints are routine in production. The retry decorator below handles these failures automatically: it waits longer after each attempt (exponential backoff) and adds randomness (jitter) to prevent simultaneous retries from a fleet of workers all hitting the API at once.

```python
import time
import random
from functools import wraps
from typing import TypeVar, Callable
import logging

T = TypeVar('T')

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True
) -> Callable:
    """
    Retry decorator with exponential backoff.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries - 1:
                        raise

                    # Calculate delay
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )

                    # Add jitter
                    if jitter:
                        delay *= (0.5 + random.random())

                    logging.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)

            raise last_exception

        return wrapper
    return decorator

# Usage
@retry_with_backoff(max_retries=3)
def call_llm_api(prompt: str) -> str:
    """Call LLM with retry."""
    from anthropic import Anthropic
    client = Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```

### Circuit Breaker


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">from.py</span>
</div>

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import threading

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreaker:
    """
    Circuit breaker for external service calls.
    """
    failure_threshold: int = 5
    recovery_timeout: timedelta = timedelta(seconds=30)
    half_open_max_calls: int = 3

    def __post_init__(self):
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        self._lock = threading.Lock()

    def can_execute(self) -> bool:
        """Check if call should be allowed."""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if datetime.now() - self.last_failure_time > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    return True
                return False

            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls < self.half_open_max_calls:
                    self.half_open_calls += 1
                    return True
                return False

        return False

    def record_success(self):
        """Record successful call."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
            self.failures = 0

    def record_failure(self):
        """Record failed call."""
        with self._lock:
            self.failures += 1
            self.last_failure_time = datetime.now()

            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
            elif self.failures >= self.failure_threshold:
                self.state = CircuitState.OPEN

# Usage
llm_circuit_breaker = CircuitBreaker()

def call_llm_with_circuit_breaker(prompt: str) -> str:
    """Call LLM with circuit breaker protection."""
    if not llm_circuit_breaker.can_execute():
        raise Exception("Circuit breaker is open")

    try:
        result = call_llm_api(prompt)
        llm_circuit_breaker.record_success()
        return result
    except Exception as e:
        llm_circuit_breaker.record_failure()
        raise
```

</div>
</div>

## Cost Management

### Token Budgeting


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">from.py</span>
</div>

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading

@dataclass
class TokenBudget:
    """
    Manage LLM token budget.
    """
    daily_limit: int
    hourly_limit: int

    def __post_init__(self):
        self.daily_usage = 0
        self.hourly_usage = 0
        self.day_start = datetime.now().replace(hour=0, minute=0, second=0)
        self.hour_start = datetime.now().replace(minute=0, second=0)
        self._lock = threading.Lock()

    def _reset_if_needed(self):
        """Reset counters if period has passed."""
        now = datetime.now()
        new_day = now.replace(hour=0, minute=0, second=0)
        new_hour = now.replace(minute=0, second=0)

        if new_day > self.day_start:
            self.daily_usage = 0
            self.day_start = new_day

        if new_hour > self.hour_start:
            self.hourly_usage = 0
            self.hour_start = new_hour

    def can_spend(self, tokens: int) -> bool:
        """Check if token spend is within budget."""
        with self._lock:
            self._reset_if_needed()
            return (
                self.daily_usage + tokens <= self.daily_limit and
                self.hourly_usage + tokens <= self.hourly_limit
            )

    def record_usage(self, tokens: int):
        """Record token usage."""
        with self._lock:
            self._reset_if_needed()
            self.daily_usage += tokens
            self.hourly_usage += tokens

    def get_remaining(self) -> dict:
        """Get remaining budget."""
        with self._lock:
            self._reset_if_needed()
            return {
                'daily_remaining': self.daily_limit - self.daily_usage,
                'hourly_remaining': self.hourly_limit - self.hourly_usage
            }

# Usage
token_budget = TokenBudget(
    daily_limit=1_000_000,  # 1M tokens/day
    hourly_limit=100_000    # 100K tokens/hour
)

def budgeted_llm_call(prompt: str, estimated_tokens: int) -> str:
    """Make LLM call within budget."""
    if not token_budget.can_spend(estimated_tokens):
        raise Exception("Token budget exceeded")

    result = call_llm_api(prompt)

    # Record actual usage (would get from API response)
    token_budget.record_usage(estimated_tokens)

    return result
```

</div>
</div>

### Caching Strategy


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">llmcache.py</span>
</div>

```python
import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional
import redis

class LLMCache:
    """
    Cache LLM responses to reduce costs.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        default_ttl: timedelta = timedelta(hours=1)
    ):
        self.redis = redis.from_url(redis_url)
        self.default_ttl = default_ttl

    def _cache_key(self, prompt: str, model: str) -> str:
        """Generate cache key from prompt."""
        content = f"{model}:{prompt}"
        return f"llm_cache:{hashlib.sha256(content.encode()).hexdigest()}"

    def get(self, prompt: str, model: str) -> Optional[str]:
        """Get cached response."""
        key = self._cache_key(prompt, model)
        cached = self.redis.get(key)
        if cached:
            return json.loads(cached)['response']
        return None

    def set(
        self,
        prompt: str,
        model: str,
        response: str,
        ttl: Optional[timedelta] = None
    ):
        """Cache response."""
        key = self._cache_key(prompt, model)
        ttl = ttl or self.default_ttl

        self.redis.setex(
            key,
            ttl,
            json.dumps({
                'response': response,
                'cached_at': datetime.now().isoformat()
            })
        )

# Usage
llm_cache = LLMCache()

def cached_llm_call(prompt: str, model: str = "claude-sonnet-4-20250514") -> str:
    """Make LLM call with caching."""
    # Check cache
    cached = llm_cache.get(prompt, model)
    if cached:
        return cached

    # Make API call
    response = call_llm_api(prompt)

    # Cache response
    llm_cache.set(prompt, model, response)

    return response
```

</div>
</div>

## Monitoring and Observability

### Metrics Collection


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">from.py</span>
</div>

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List
import json

@dataclass
class LLMMetrics:
    """
    Track LLM usage metrics.
    """
    timestamp: datetime
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    success: bool
    error_type: str = None
    cache_hit: bool = False

class MetricsCollector:
    """
    Collect and export metrics.
    """

    def __init__(self):
        self.metrics: List[LLMMetrics] = []

    def record(self, metric: LLMMetrics):
        """Record a metric."""
        self.metrics.append(metric)

    def get_summary(self, hours: int = 24) -> dict:
        """Get metrics summary."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [m for m in self.metrics if m.timestamp > cutoff]

        if not recent:
            return {}

        total_calls = len(recent)
        successful = sum(1 for m in recent if m.success)
        cache_hits = sum(1 for m in recent if m.cache_hit)
        total_tokens = sum(m.prompt_tokens + m.completion_tokens for m in recent)
        avg_latency = sum(m.latency_ms for m in recent) / total_calls

        return {
            'total_calls': total_calls,
            'success_rate': successful / total_calls,
            'cache_hit_rate': cache_hits / total_calls,
            'total_tokens': total_tokens,
            'avg_latency_ms': avg_latency,
            'estimated_cost_usd': total_tokens * 0.000003  # Approximate
        }

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        summary = self.get_summary()
        lines = [
            f"llm_total_calls {summary.get('total_calls', 0)}",
            f"llm_success_rate {summary.get('success_rate', 0)}",
            f"llm_cache_hit_rate {summary.get('cache_hit_rate', 0)}",
            f"llm_total_tokens {summary.get('total_tokens', 0)}",
            f"llm_avg_latency_ms {summary.get('avg_latency_ms', 0)}"
        ]
        return "\n".join(lines)
```

</div>
</div>

### Alerting


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">from.py</span>

```python
from dataclasses import dataclass
from typing import Callable, List
import logging

@dataclass
class AlertRule:
    """Alert rule definition."""
    name: str
    condition: Callable[[dict], bool]
    severity: str  # critical, warning, info
    message_template: str

class AlertManager:
    """
    Manage alerts based on metrics.
    """

    def __init__(self):
        self.rules: List[AlertRule] = []
        self.alerts_sent: List[dict] = []

    def add_rule(self, rule: AlertRule):
        """Add alert rule."""
        self.rules.append(rule)

    def check_alerts(self, metrics_summary: dict):
        """Check all rules against metrics."""
        for rule in self.rules:
            if rule.condition(metrics_summary):
                self._send_alert(rule, metrics_summary)

    def _send_alert(self, rule: AlertRule, metrics: dict):
        """Send alert notification."""
        alert = {
            'name': rule.name,
            'severity': rule.severity,
            'message': rule.message_template.format(**metrics),
            'timestamp': datetime.now().isoformat()
        }
        self.alerts_sent.append(alert)
        logging.warning(f"ALERT [{rule.severity}]: {alert['message']}")

# Setup alerts
alert_manager = AlertManager()

alert_manager.add_rule(AlertRule(
    name="high_error_rate",
    condition=lambda m: m.get('success_rate', 1) < 0.95,
    severity="critical",
    message="LLM error rate above 5%: {success_rate:.1%}"
))

alert_manager.add_rule(AlertRule(
    name="high_latency",
    condition=lambda m: m.get('avg_latency_ms', 0) > 5000,
    severity="warning",
    message="LLM latency above 5s: {avg_latency_ms:.0f}ms"
))

alert_manager.add_rule(AlertRule(
    name="token_budget_warning",
    condition=lambda m: m.get('total_tokens', 0) > 800_000,
    severity="warning",
    message="Approaching daily token limit: {total_tokens:,} tokens"
))
```


<div class="callout-insight">

**Insight:** Understanding production deployment for commodity gen ai is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.


## Key Takeaways

1. **Reliability first** - retries, circuit breakers, and graceful degradation

2. **Cost control** - token budgets, caching, and usage monitoring

3. **Observability** - metrics, logging, and alerting for production visibility

4. **Async processing** - use queues for non-blocking report processing

5. **Gradual rollout** - test thoroughly before market-impacting deployment

---

## Conceptual Practice Questions

1. What are the key considerations when deploying LLM-based commodity systems to production?

2. How do you handle LLM API latency in real-time trading applications?

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.


---

## Cross-References

<a class="link-card" href="../notebooks/01_pipeline_build.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_commodity_agents.md">
  <div class="link-card-title">01 Commodity Agents</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_monitoring.md">
  <div class="link-card-title">02 Monitoring</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

