# Production Architecture: Designing for Reliability

> **Reading time:** ~12 min | **Module:** 7 — Production Deployment | **Prerequisites:** Module 5 — Multi-Agent Systems

Production agents face challenges prototypes never see: high load, network failures, malicious inputs, and cost constraints. This guide covers architectural patterns that make agents reliable, observable, and maintainable at scale.

<div class="callout-insight">

**Insight:** Production readiness is about failure handling. Assume everything will fail—APIs timeout, models hallucinate, users send garbage. Build systems that fail gracefully and recover automatically.

</div>

---

## Production Agent Architecture

### Reference Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        API Gateway                          │
│              (Rate limiting, Auth, Load balancing)          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     Request Router                          │
│           (Model selection, Request classification)         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      Agent Workers                          │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐              │
│  │  Worker 1 │  │  Worker 2 │  │  Worker N │              │
│  └───────────┘  └───────────┘  └───────────┘              │
│                                                             │
│  Components per worker:                                     │
│  • Input Validation                                         │
│  • Context Manager (Memory, RAG)                           │
│  • LLM Client (with retries)                               │
│  • Tool Executor (sandboxed)                               │
│  • Output Validator                                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   External Services                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │ LLM API │  │ Vector  │  │  Tools  │  │  Cache  │       │
│  │(Claude) │  │   DB    │  │  APIs   │  │ (Redis) │       │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### Implementation

`ProductionAgent` wraps the core agent loop with input validation, caching, structured error handling, and metrics collection—solving the gap between a working prototype and a system that can handle real load, malformed inputs, and partial failures without crashing. It uses `AgentConfig` to centralise all tunable parameters so deployment settings never live inside business logic.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>
<div class="code-body">

```python
from dataclasses import dataclass
from typing import Optional
import asyncio


@dataclass
class AgentConfig:
    model: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 4096
    max_retries: int = 3
    timeout_seconds: int = 120
    max_tool_calls: int = 10
    enable_caching: bool = True
    log_level: str = "INFO"


class ProductionAgent:
    """Production-ready agent with reliability features."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = anthropic.Anthropic()
        self.cache = RedisCache() if config.enable_caching else None
        self.metrics = MetricsCollector()
        self.logger = StructuredLogger(config.log_level)

    async def run(self, request_id: str, user_input: str) -> dict:
        """Process a user request with full production handling."""

        span = self.start_trace(request_id)

        try:
            # Input validation
            validated_input = self.validate_input(user_input)

            # Check cache
            if self.cache:
                cached = await self.cache.get(validated_input)
                if cached:
                    self.metrics.increment("cache_hit")
                    return cached

            # Process request
            result = await self._process_with_retry(validated_input)

            # Validate output
            validated_output = self.validate_output(result)

            # Cache result
            if self.cache:
                await self.cache.set(validated_input, validated_output)

            self.metrics.increment("success")
            return {"status": "success", "result": validated_output}

        except ValidationError as e:
            self.metrics.increment("validation_error")
            return {"status": "error", "error": str(e), "code": "INVALID_INPUT"}

        except RateLimitError as e:
            self.metrics.increment("rate_limit")
            return {"status": "error", "error": "Service busy", "code": "RATE_LIMITED"}

        except Exception as e:
            self.logger.error("Unexpected error", error=str(e), request_id=request_id)
            self.metrics.increment("error")
            return {"status": "error", "error": "Internal error", "code": "INTERNAL"}

        finally:
            span.end()

    async def _process_with_retry(self, input_text: str) -> str:
        """Process with exponential backoff retry."""

        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                return await asyncio.wait_for(
                    self._execute(input_text),
                    timeout=self.config.timeout_seconds
                )
            except (asyncio.TimeoutError, TransientError) as e:
                last_error = e
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        raise last_error

    async def _execute(self, input_text: str) -> str:
        """Core execution logic."""
        # Implementation here
        pass
```

</div>
</div>

---

## Reliability Patterns

### Circuit Breaker

Prevent cascading failures:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>
<div class="code-body">

```python
from datetime import datetime, timedelta
from enum import Enum


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: timedelta = timedelta(seconds=60),
        half_open_requests: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests

        self.state = CircuitState.CLOSED
        self.failures = 0
        self.successes_in_half_open = 0
        self.last_failure_time: Optional[datetime] = None

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""

        if self.state == CircuitState.OPEN:
            if self._should_attempt_recovery():
                self.state = CircuitState.HALF_OPEN
                self.successes_in_half_open = 0
            else:
                raise CircuitOpenError("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise

    def _should_attempt_recovery(self) -> bool:
        return (
            self.last_failure_time and
            datetime.now() - self.last_failure_time >= self.recovery_timeout
        )

    def _record_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self.successes_in_half_open += 1
            if self.successes_in_half_open >= self.half_open_requests:
                self.state = CircuitState.CLOSED
                self.failures = 0

    def _record_failure(self):
        self.failures += 1
        self.last_failure_time = datetime.now()

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
        elif self.failures >= self.failure_threshold:
            self.state = CircuitState.OPEN


# Usage
llm_circuit = CircuitBreaker()

async def call_llm_with_protection(prompt):
    return await llm_circuit.call(client.messages.create, ...)
```

</div>
</div>

### Fallback Strategies

```python
class FallbackAgent:
    """Agent with fallback strategies."""

    def __init__(self):
        self.primary_model = "claude-3-5-sonnet-20241022"
        self.fallback_model = "claude-3-haiku-20240307"
        self.emergency_response = "I'm having technical difficulties. Please try again."

    async def run(self, query: str) -> str:
        # Try primary model
        try:
            return await self._call_model(query, self.primary_model)
        except (RateLimitError, TimeoutError):
            pass

        # Try fallback model
        try:
            return await self._call_model(query, self.fallback_model)
        except (RateLimitError, TimeoutError):
            pass

        # Emergency fallback
        return self.emergency_response
```

### Bulkhead Isolation

Isolate failures to prevent system-wide impact:

```python
from asyncio import Semaphore


class BulkheadAgent:
    """Agent with bulkhead isolation per component."""

    def __init__(self, max_concurrent_llm: int = 10, max_concurrent_tools: int = 5):
        self.llm_semaphore = Semaphore(max_concurrent_llm)
        self.tool_semaphore = Semaphore(max_concurrent_tools)

    async def call_llm(self, messages):
        async with self.llm_semaphore:
            return await self._llm_call(messages)

    async def call_tool(self, name, params):
        async with self.tool_semaphore:
            return await self._tool_call(name, params)
```

---

## Request Handling

### Rate Limiting

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>
<div class="code-body">

```python
from collections import defaultdict
import time


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, requests_per_minute: int = 60):
        self.rate = requests_per_minute / 60.0  # per second
        self.tokens = defaultdict(lambda: requests_per_minute)
        self.last_update = defaultdict(time.time)

    def allow(self, key: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        elapsed = now - self.last_update[key]
        self.last_update[key] = now

        # Add tokens based on elapsed time
        self.tokens[key] = min(
            self.tokens[key] + elapsed * self.rate,
            self.rate * 60  # Max bucket size
        )

        if self.tokens[key] >= 1:
            self.tokens[key] -= 1
            return True
        return False


# Usage in FastAPI
from fastapi import HTTPException, Request


rate_limiter = RateLimiter()


async def rate_limit_middleware(request: Request):
    client_id = request.client.host
    if not rate_limiter.allow(client_id):
        raise HTTPException(status_code=429, detail="Too many requests")
```

</div>
</div>

### Request Prioritization

```python
from enum import Enum
from heapq import heappush, heappop


class Priority(Enum):
    HIGH = 1
    NORMAL = 2
    LOW = 3


class PriorityQueue:
    """Priority queue for agent requests."""

    def __init__(self):
        self.queue = []
        self.counter = 0

    def enqueue(self, priority: Priority, request: dict):
        heappush(self.queue, (priority.value, self.counter, request))
        self.counter += 1

    def dequeue(self):
        if self.queue:
            _, _, request = heappop(self.queue)
            return request
        return None
```

---

## Deployment Patterns

### FastAPI Agent Service

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>
<div class="code-body">

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uuid


app = FastAPI()
agent = ProductionAgent(AgentConfig())


class AgentRequest(BaseModel):
    query: str
    user_id: str
    priority: str = "normal"


class AgentResponse(BaseModel):
    request_id: str
    status: str
    result: Optional[str] = None
    error: Optional[str] = None


@app.post("/agent/query", response_model=AgentResponse)
async def query_agent(request: AgentRequest):
    request_id = str(uuid.uuid4())

    result = await agent.run(request_id, request.query)

    return AgentResponse(
        request_id=request_id,
        status=result["status"],
        result=result.get("result"),
        error=result.get("error")
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}
```

</div>
</div>

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-service
  template:
    metadata:
      labels:
        app: agent-service
    spec:
      containers:
      - name: agent
        image: agent-service:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: anthropic
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
```

---

## Error Handling

### Structured Error Responses

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>
<div class="code-body">

```python
from enum import Enum


class ErrorCode(Enum):
    INVALID_INPUT = "INVALID_INPUT"
    RATE_LIMITED = "RATE_LIMITED"
    MODEL_ERROR = "MODEL_ERROR"
    TOOL_ERROR = "TOOL_ERROR"
    TIMEOUT = "TIMEOUT"
    INTERNAL = "INTERNAL"


@dataclass
class AgentError:
    code: ErrorCode
    message: str
    details: Optional[dict] = None
    retry_after: Optional[int] = None

    def to_response(self) -> dict:
        return {
            "status": "error",
            "error": {
                "code": self.code.value,
                "message": self.message,
                "details": self.details,
                "retry_after": self.retry_after
            }
        }


def handle_llm_error(error: Exception) -> AgentError:
    """Convert LLM errors to structured responses."""

    if isinstance(error, anthropic.RateLimitError):
        return AgentError(
            code=ErrorCode.RATE_LIMITED,
            message="Request rate limit exceeded",
            retry_after=60
        )

    if isinstance(error, anthropic.APITimeoutError):
        return AgentError(
            code=ErrorCode.TIMEOUT,
            message="Request timed out"
        )

    return AgentError(
        code=ErrorCode.INTERNAL,
        message="An unexpected error occurred"
    )
```

</div>
</div>

<div class="callout-key">

**Key Concept Summary:** This guide covered the core concepts. Review the companion slides for visual summaries and the hands-on notebook for practice implementations.

</div>

---

*Production deployment is where agents prove their value. Build for failure, measure everything, and iterate based on real-world performance.*


## Practice Questions

1. Explain in your own words how the concepts in this guide relate to building production agents.
2. What are the key tradeoffs you need to consider when applying these techniques?
3. Describe a scenario where the approach from this guide would be the wrong choice, and what you would use instead.

---

**Next Steps:**

<a class="link-card" href="./01_production_architecture_slides.md">
  <div class="link-card-title">Production Agent Architecture — Companion Slides</div>
  <div class="link-card-description">Visual slide deck with diagrams, speaker notes, and key takeaways.</div>
</a>

<a class="link-card" href="../notebooks/01_deployment_patterns.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with working code and guided exercises.</div>
</a>
