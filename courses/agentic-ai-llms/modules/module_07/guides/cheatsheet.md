# Production Deployment Cheatsheet

## Key Concepts

| Concept | Definition |
|---------|-----------|
| **Observability** | Ability to understand system behavior through logs, metrics, and traces |
| **Distributed Tracing** | Tracking requests across multiple services to diagnose performance issues |
| **Circuit Breaker** | Pattern that prevents cascading failures by stopping requests to failing services |
| **Prompt Caching** | Reusing cached LLM responses for identical or similar prompts to reduce cost |
| **Model Routing** | Selecting appropriate models based on task complexity and requirements |
| **Semantic Caching** | Caching responses for semantically similar queries, not just exact matches |
| **Exponential Backoff** | Retry strategy with increasing delays between attempts |
| **Rate Limiting** | Controlling request frequency to prevent abuse and manage costs |
| **Canary Deployment** | Gradual rollout to subset of users to detect issues before full deployment |
| **Token Budget** | Maximum token limit per request or time period to control costs |

## Common Patterns

### Structured Logging

```python
import structlog
from datetime import datetime

logger = structlog.get_logger()

class ProductionAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.logger = logger.bind(agent_id=agent_id)

    def process(self, request):
        request_id = generate_request_id()
        log = self.logger.bind(
            request_id=request_id,
            user_id=request.user_id
        )

        log.info("request_started", query=request.query)

        try:
            # Process request
            start_time = time.time()
            result = self.execute(request)
            latency = time.time() - start_time

            log.info(
                "request_completed",
                latency_ms=latency * 1000,
                tokens_used=result.tokens,
                cost_usd=result.cost
            )

            return result

        except Exception as e:
            log.error(
                "request_failed",
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc()
            )
            raise

# Logs output as JSON for easy parsing
# {"event": "request_started", "agent_id": "agent-123", "request_id": "req-456", ...}
```

### Distributed Tracing with OpenTelemetry

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Setup
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter())
)
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer(__name__)

class TracedAgent:
    def process(self, query):
        with tracer.start_as_current_span("agent.process") as span:
            span.set_attribute("query.length", len(query))

            # Planning phase
            with tracer.start_as_current_span("agent.plan") as plan_span:
                plan = self.create_plan(query)
                plan_span.set_attribute("plan.steps", len(plan))

            # Execution phase
            with tracer.start_as_current_span("agent.execute") as exec_span:
                for i, step in enumerate(plan):
                    with tracer.start_as_current_span(f"step.{i}") as step_span:
                        step_span.set_attribute("step.action", step.action)

                        # LLM call
                        with tracer.start_as_current_span("llm.call") as llm_span:
                            result = llm.generate(step.prompt)
                            llm_span.set_attribute("tokens.input", result.input_tokens)
                            llm_span.set_attribute("tokens.output", result.output_tokens)

                        # Tool call
                        if step.requires_tool:
                            with tracer.start_as_current_span("tool.call") as tool_span:
                                tool_span.set_attribute("tool.name", step.tool)
                                tool_result = self.call_tool(step.tool, step.input)

            return result
```

### Retry Logic with Circuit Breaker

```python
import time
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)

            # Success
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
            self.failures = 0
            return result

        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()

            if self.failures >= self.failure_threshold:
                self.state = CircuitState.OPEN

            raise e

def retry_with_backoff(func, max_retries=3, base_delay=1, max_delay=30):
    """Exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise

            delay = min(base_delay * (2 ** attempt), max_delay)
            logger.warning(
                "retry_attempt",
                attempt=attempt + 1,
                delay_seconds=delay,
                error=str(e)
            )
            time.sleep(delay)

# Usage
circuit_breaker = CircuitBreaker()

def call_llm_with_protection(prompt):
    def llm_call():
        return llm.generate(prompt)

    return circuit_breaker.call(retry_with_backoff, llm_call)
```

### Semantic Caching

```python
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticCache:
    def __init__(self, similarity_threshold=0.95):
        self.cache = {}  # {embedding_hash: (query, response, embedding)}
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = similarity_threshold

    def get(self, query):
        """Get cached response if semantically similar query exists."""
        query_embedding = self.embedder.encode(query)

        for cached_hash, (cached_query, response, cached_embedding) in self.cache.items():
            similarity = np.dot(query_embedding, cached_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
            )

            if similarity >= self.threshold:
                logger.info(
                    "cache_hit",
                    query=query,
                    cached_query=cached_query,
                    similarity=similarity
                )
                return response

        return None

    def set(self, query, response):
        """Cache query-response pair with embedding."""
        embedding = self.embedder.encode(query)
        embedding_hash = hashlib.md5(embedding.tobytes()).hexdigest()

        self.cache[embedding_hash] = (query, response, embedding)

        # Simple size limit
        if len(self.cache) > 1000:
            # Remove oldest
            first_key = next(iter(self.cache))
            del self.cache[first_key]

# Usage
cache = SemanticCache()

def cached_agent_call(query):
    # Check cache
    cached = cache.get(query)
    if cached:
        return cached

    # Generate response
    response = agent.process(query)

    # Cache it
    cache.set(query, response)

    return response
```

### Model Routing

```python
class ModelRouter:
    def __init__(self):
        self.models = {
            "fast": {
                "name": "claude-3-haiku-20240307",
                "cost_per_1k": 0.00025,
                "latency_ms": 500
            },
            "balanced": {
                "name": "claude-3-5-sonnet-20241022",
                "cost_per_1k": 0.003,
                "latency_ms": 1000
            },
            "powerful": {
                "name": "claude-opus-4-5-20251101",
                "cost_per_1k": 0.015,
                "latency_ms": 2000
            }
        }

    def select_model(self, query, context):
        """Route to appropriate model based on complexity."""

        # Calculate complexity score
        complexity = self.assess_complexity(query, context)

        if complexity < 0.3:
            model_tier = "fast"
        elif complexity < 0.7:
            model_tier = "balanced"
        else:
            model_tier = "powerful"

        logger.info(
            "model_selected",
            complexity=complexity,
            tier=model_tier,
            model=self.models[model_tier]["name"]
        )

        return self.models[model_tier]

    def assess_complexity(self, query, context):
        """Estimate task complexity (0-1)."""
        features = {
            "query_length": len(query) / 1000,  # Normalize
            "requires_reasoning": self.needs_reasoning(query),
            "context_size": len(context) / 10000,
            "requires_tools": self.needs_tools(query)
        }

        # Weighted sum
        weights = {
            "query_length": 0.2,
            "requires_reasoning": 0.4,
            "context_size": 0.2,
            "requires_tools": 0.2
        }

        complexity = sum(
            features[k] * weights[k] for k in features
        )

        return min(complexity, 1.0)

# Usage
router = ModelRouter()
model = router.select_model(user_query, context)
response = anthropic.messages.create(model=model["name"], ...)
```

### Monitoring Dashboard Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
request_counter = Counter(
    'agent_requests_total',
    'Total agent requests',
    ['status', 'model']
)

latency_histogram = Histogram(
    'agent_request_latency_seconds',
    'Request latency distribution',
    ['model']
)

token_counter = Counter(
    'agent_tokens_total',
    'Total tokens used',
    ['model', 'type']  # type: input/output
)

cost_counter = Counter(
    'agent_cost_usd_total',
    'Total cost in USD',
    ['model']
)

active_requests = Gauge(
    'agent_active_requests',
    'Number of requests in progress'
)

class MonitoredAgent:
    def process(self, query, model="claude-3-5-sonnet-20241022"):
        active_requests.inc()

        try:
            with latency_histogram.labels(model=model).time():
                result = self.execute(query, model)

            # Record success
            request_counter.labels(status="success", model=model).inc()

            # Record tokens
            token_counter.labels(model=model, type="input").inc(result.input_tokens)
            token_counter.labels(model=model, type="output").inc(result.output_tokens)

            # Record cost
            cost = self.calculate_cost(result, model)
            cost_counter.labels(model=model).inc(cost)

            return result

        except Exception as e:
            request_counter.labels(status="error", model=model).inc()
            raise

        finally:
            active_requests.dec()
```

### Rate Limiting

```python
import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, requests_per_minute=60):
        self.limit = requests_per_minute
        self.requests = defaultdict(list)  # {user_id: [timestamps]}

    def allow_request(self, user_id):
        now = time.time()
        minute_ago = now - 60

        # Remove old requests
        self.requests[user_id] = [
            ts for ts in self.requests[user_id]
            if ts > minute_ago
        ]

        # Check limit
        if len(self.requests[user_id]) >= self.limit:
            return False

        # Record this request
        self.requests[user_id].append(now)
        return True

# Token budget limiter
class TokenBudgetLimiter:
    def __init__(self, tokens_per_day=1_000_000):
        self.daily_limit = tokens_per_day
        self.usage = {}  # {date: token_count}

    def check_budget(self, user_id, estimated_tokens):
        today = datetime.now().date()

        if today not in self.usage:
            self.usage = {today: 0}  # Reset

        if self.usage[today] + estimated_tokens > self.daily_limit:
            raise Exception(f"Daily token budget exceeded: {self.usage[today]}/{self.daily_limit}")

        self.usage[today] += estimated_tokens

# Usage
rate_limiter = RateLimiter(requests_per_minute=10)
token_limiter = TokenBudgetLimiter(tokens_per_day=100_000)

def protected_endpoint(user_id, query):
    if not rate_limiter.allow_request(user_id):
        raise Exception("Rate limit exceeded")

    estimated_tokens = estimate_tokens(query)
    token_limiter.check_budget(user_id, estimated_tokens)

    return agent.process(query)
```

## Gotchas

### Problem: Logs overwhelm storage
**Symptom:** Logging costs exceed LLM costs, storage fills up quickly
**Solution:**
- Use log levels appropriately (DEBUG for dev, INFO/WARN for prod)
- Sample verbose logs (e.g., log 1% of successful requests, 100% of errors)
- Set retention policies (7 days for DEBUG, 30 days for INFO, 1 year for ERROR)
- Use structured logging for efficient querying

```python
# Bad: Log everything
logger.debug(f"Full prompt: {prompt}")  # Expensive!

# Good: Sample and summarize
if random.random() < 0.01 or error:  # 1% sampling
    logger.info("request_details", prompt_length=len(prompt), tokens=tokens)
```

### Problem: Distributed tracing adds latency
**Symptom:** Tracing overhead slows down requests significantly
**Solution:**
- Use async exporters
- Batch span exports
- Sample traces (not every request needs full trace)
- Optimize span creation (don't create spans for trivial operations)

### Problem: Cache invalidation is hard
**Symptom:** Users get stale responses after updates
**Solution:**
- Use TTL (time to live) for all cache entries
- Implement cache versioning
- Add manual invalidation endpoints
- Monitor cache hit rate vs accuracy tradeoff

### Problem: Circuit breaker false positives
**Symptom:** Circuit opens during temporary blips, rejecting valid requests
**Solution:**
- Tune failure threshold based on actual error rates
- Use sliding window instead of fixed counter
- Implement half-open state to test recovery
- Different thresholds for different error types

### Problem: Monitoring alerts are too noisy
**Symptom:** Team ignores alerts due to false positives
**Solution:**
- Set thresholds based on historical data, not guesses
- Use anomaly detection instead of fixed thresholds
- Implement alert fatigue detection
- Group related alerts

### Problem: Cost tracking is inaccurate
**Symptom:** Actual bills don't match tracked costs
**Solution:**
- Track all LLM calls (including retries and failures)
- Include provider markup and API fees
- Track non-LLM costs (embeddings, infrastructure)
- Reconcile daily with provider bills

### Problem: Canary deployments detect issues too late
**Symptom:** Bad deploy reaches 20% of users before caught
**Solution:**
- Start with 1-5% traffic
- Use automated quality checks, not just error rate
- Include business metrics (task completion, user satisfaction)
- Have instant rollback capability

## Quick Decision Guide

**When to add observability?**
- Always: Structured logging, basic metrics
- Production: Distributed tracing, dashboards
- Scale: Advanced monitoring, anomaly detection
- Enterprise: Full observability stack with alerting

**When to optimize for cost vs latency?**
- Cost priority: Batch processing, async workflows, non-user-facing
- Latency priority: Interactive chat, real-time agents, user-facing
- Balance: Most production applications

**When to use caching?**
- Exact match cache: Repeated identical queries (FAQ, common requests)
- Semantic cache: Similar queries with same answer
- Prompt cache: Long system prompts that don't change
- Don't cache: User-specific, time-sensitive, or unique queries

**When to implement circuit breakers?**
- External API calls (especially third-party)
- Database queries
- LLM provider calls (optional, they usually have good reliability)
- Any dependency with known instability

**When to use model routing?**
- Mixed workload (simple and complex tasks)
- Cost optimization is priority
- Have established complexity heuristics
- Don't use: All tasks same complexity, latency critical

## Production Checklist

### Before Launch
- [ ] Structured logging implemented
- [ ] Metrics tracked (latency, tokens, cost, errors)
- [ ] Error handling and retries configured
- [ ] Rate limiting implemented
- [ ] Monitoring dashboards created
- [ ] Alerts configured with on-call rotation
- [ ] Load testing completed
- [ ] Security review passed
- [ ] Cost budgets set
- [ ] Incident response plan documented

### Day 1
- [ ] Monitor error rates closely
- [ ] Check cost vs estimates
- [ ] Validate latency meets SLA
- [ ] Review user feedback
- [ ] Check cache hit rates

### Week 1
- [ ] Analyze usage patterns
- [ ] Optimize slow queries
- [ ] Tune caching strategy
- [ ] Review and adjust alerts
- [ ] Cost optimization review

### Ongoing
- [ ] Weekly metrics review
- [ ] Monthly cost analysis
- [ ] Quarterly load testing
- [ ] Regular security audits
- [ ] Model version upgrades
