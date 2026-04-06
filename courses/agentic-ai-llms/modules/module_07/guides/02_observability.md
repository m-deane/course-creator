# Observability for AI Agents

> **Reading time:** ~10 min | **Module:** 7 — Production Deployment | **Prerequisites:** Module 7 — Production Architecture

Observability is the practice of making agent systems transparent and debuggable through comprehensive logging, metrics, and tracing. While traditional software observability focuses on requests and errors, agent observability must capture reasoning traces, tool usage, and decision-making processes.

<div class="callout-insight">

**Insight:** You can't fix what you can't see. Agents make complex multi-step decisions involving reasoning, tool calls, and data retrieval. Without observability, debugging agent failures is like troubleshooting a black box—you see the wrong output but have no insight into why it occurred.

</div>

Good observability answers:
- **What** did the agent do?
- **Why** did it make each decision?
- **How long** did each step take?
- **What went wrong** when it failed?

## Formal Definition

**Observability** for AI agents consists of three pillars:

1. **Logs:** Detailed records of agent execution including inputs, outputs, reasoning, and tool calls
2. **Metrics:** Quantitative measurements of performance (latency, cost, success rate, errors)
3. **Traces:** End-to-end execution paths showing the flow through multi-step agent processes

**Key Properties:**
- **Granularity:** Capture details at LLM call, tool call, and overall session levels
- **Context:** Include relevant metadata (user, timestamp, version, configuration)
- **Actionability:** Logs should enable rapid debugging and root cause analysis
- **Efficiency:** Observability overhead should be minimal (<5% performance impact)

## Intuitive Explanation

Think of observability like a flight data recorder for aircraft:

**Without Observability:**
```
Agent crashed. Error: "Failed to complete task"
[No additional information]
```

**With Observability:**
```
Session: abc-123
User: user@example.com
Timestamp: 2026-02-02T10:30:00Z
Duration: 45.3s
Cost: $0.23

Trace:
1. [10:30:00.000] User input received: "Analyze Q4 revenue"
2. [10:30:00.100] Planning step: Need to fetch revenue data
3. [10:30:00.150] Tool call: fetch_financial_data(quarter=4, year=2025)
4. [10:30:05.200] Tool result: 15MB of data received
5. [10:30:05.300] Reasoning step: Analyzing revenue trends
6. [10:30:25.400] LLM call: Anthropic Claude (2000 tokens in, 800 tokens out)
7. [10:30:35.500] Generated analysis
8. [10:30:35.600] Safety check: passed
9. [10:30:35.700] Response delivered

Metrics:
- Total tool calls: 1
- Total LLM calls: 1
- Total tokens: 2800
- Total cost: $0.23
- p95 latency: 42.1s
```

This visibility makes debugging and optimization possible.

## Code Implementation

### Structured Logging

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>
<div class="code-body">

```python
import logging
import json
from datetime import datetime
from typing import Any, Optional
import uuid

class AgentLogger:
    """Structured logging for agent execution."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.logger = logging.getLogger(f"agent.{self.session_id}")

        # Configure JSON structured logging
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_event(
        self,
        event_type: str,
        message: str,
        **kwargs
    ):
        """Log a structured event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": self.session_id,
            "event_type": event_type,
            "message": message,
            **kwargs
        }
        self.logger.info(json.dumps(event))

    def log_user_input(self, user_input: str, user_id: str):
        """Log user input."""
        self.log_event(
            "user_input",
            "User input received",
            user_id=user_id,
            input_length=len(user_input),
            input_preview=user_input[:100]
        )

    def log_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cost: float
    ):
        """Log LLM API call."""
        self.log_event(
            "llm_call",
            f"LLM call to {model}",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_ms=latency_ms,
            cost=cost
        )

    def log_tool_call(
        self,
        tool_name: str,
        parameters: dict,
        result: Any,
        latency_ms: float,
        success: bool
    ):
        """Log tool execution."""
        self.log_event(
            "tool_call",
            f"Tool call: {tool_name}",
            tool_name=tool_name,
            parameters=parameters,
            result_preview=str(result)[:200],
            latency_ms=latency_ms,
            success=success
        )

    def log_reasoning(self, thought: str, step_number: int):
        """Log agent reasoning step."""
        self.log_event(
            "reasoning",
            "Agent reasoning",
            thought=thought,
            step_number=step_number
        )

    def log_error(self, error: Exception, context: dict):
        """Log error with full context."""
        self.log_event(
            "error",
            f"Error: {str(error)}",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context,
            level="ERROR"
        )


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logs."""

    def format(self, record):
        return record.getMessage()
```

</div>
</div>

### Metrics Collection

```python
from dataclasses import dataclass, field
from typing import Dict, List
import time

@dataclass
class AgentMetrics:
    """Metrics for agent execution."""

    # Latency metrics
    total_duration_ms: float = 0
    llm_latency_ms: List[float] = field(default_factory=list)
    tool_latency_ms: List[float] = field(default_factory=list)

    # Token metrics
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # Cost metrics
    total_cost: float = 0.0

    # Count metrics
    llm_calls: int = 0
    tool_calls: int = 0
    errors: int = 0

    # Success metrics
    successful_completions: int = 0
    failed_completions: int = 0

    def add_llm_call(
        self,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cost: float
    ):
        """Record LLM call metrics."""
        self.llm_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.llm_latency_ms.append(latency_ms)
        self.total_cost += cost

    def add_tool_call(self, latency_ms: float, success: bool):
        """Record tool call metrics."""
        self.tool_calls += 1
        self.tool_latency_ms.append(latency_ms)
        if not success:
            self.errors += 1

    def compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        import statistics

        return {
            "total_duration_ms": self.total_duration_ms,
            "avg_llm_latency_ms": statistics.mean(self.llm_latency_ms) if self.llm_latency_ms else 0,
            "p95_llm_latency_ms": statistics.quantiles(self.llm_latency_ms, n=20)[18] if len(self.llm_latency_ms) > 1 else 0,
            "avg_tool_latency_ms": statistics.mean(self.tool_latency_ms) if self.tool_latency_ms else 0,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": self.total_cost,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "errors": self.errors,
            "success_rate": self.successful_completions / (self.successful_completions + self.failed_completions) if (self.successful_completions + self.failed_completions) > 0 else 0
        }
```

### Distributed Tracing

```python
from contextlib import contextmanager
from typing import Generator

class Span:
    """Represents a traced operation."""

    def __init__(self, name: str, parent: Optional['Span'] = None):
        self.name = name
        self.parent = parent
        self.span_id = str(uuid.uuid4())
        self.trace_id = parent.trace_id if parent else str(uuid.uuid4())
        self.start_time = time.time()
        self.end_time = None
        self.attributes = {}
        self.events = []

    def set_attribute(self, key: str, value: Any):
        """Add attribute to span."""
        self.attributes[key] = value

    def add_event(self, name: str, **attributes):
        """Add event to span."""
        self.events.append({
            "timestamp": time.time(),
            "name": name,
            **attributes
        })

    def end(self):
        """Mark span as complete."""
        self.end_time = time.time()

    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0


class AgentTracer:
    """Distributed tracing for agents."""

    def __init__(self):
        self.current_span: Optional[Span] = None
        self.completed_spans: List[Span] = []

    @contextmanager
    def start_span(self, name: str) -> Generator[Span, None, None]:
        """Start a new span."""
        span = Span(name, parent=self.current_span)
        previous_span = self.current_span
        self.current_span = span

        try:
            yield span
        finally:
            span.end()
            self.completed_spans.append(span)
            self.current_span = previous_span

    def get_trace(self) -> List[Dict]:
        """Get all spans as trace."""
        return [
            {
                "span_id": span.span_id,
                "trace_id": span.trace_id,
                "parent_id": span.parent.span_id if span.parent else None,
                "name": span.name,
                "start_time": span.start_time,
                "duration_ms": span.duration_ms(),
                "attributes": span.attributes,
                "events": span.events
            }
            for span in self.completed_spans
        ]


# Usage in agent
class ObservableAgent:
    """Agent with full observability."""

    def __init__(self, client):
        self.client = client
        self.logger = AgentLogger()
        self.metrics = AgentMetrics()
        self.tracer = AgentTracer()

    def execute(self, user_input: str, user_id: str) -> str:
        """Execute with full observability."""

        start_time = time.time()

        with self.tracer.start_span("agent_execution") as span:
            span.set_attribute("user_id", user_id)
            span.set_attribute("input_length", len(user_input))

            try:
                # Log input
                self.logger.log_user_input(user_input, user_id)

                # Execute agent logic
                result = self._execute_with_tracing(user_input, span)

                # Record success
                self.metrics.successful_completions += 1
                span.set_attribute("success", True)

                return result

            except Exception as e:
                # Log error
                self.logger.log_error(e, {"user_input": user_input})
                self.metrics.failed_completions += 1
                self.metrics.errors += 1
                span.set_attribute("success", False)
                span.set_attribute("error", str(e))
                raise

            finally:
                # Record total duration
                duration_ms = (time.time() - start_time) * 1000
                self.metrics.total_duration_ms = duration_ms
                span.set_attribute("duration_ms", duration_ms)

    def _execute_with_tracing(self, user_input: str, parent_span: Span) -> str:
        """Execute agent with nested tracing."""

        # Step 1: LLM call
        with self.tracer.start_span("llm_call") as span:
            llm_start = time.time()

            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": user_input}]
            )

            llm_latency = (time.time() - llm_start) * 1000
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = (input_tokens * 0.003 + output_tokens * 0.015) / 1000

            # Log and record metrics
            self.logger.log_llm_call(
                "claude-3-5-sonnet",
                input_tokens,
                output_tokens,
                llm_latency,
                cost
            )
            self.metrics.add_llm_call(input_tokens, output_tokens, llm_latency, cost)

            span.set_attribute("model", "claude-3-5-sonnet")
            span.set_attribute("tokens", input_tokens + output_tokens)
            span.set_attribute("cost", cost)

            return response.content[0].text

    def get_observability_data(self) -> Dict:
        """Get all observability data."""
        return {
            "session_id": self.logger.session_id,
            "metrics": self.metrics.compute_summary(),
            "trace": self.tracer.get_trace()
        }
```

### Integration with Observability Platforms

```python
# Example: Sending to OpenTelemetry
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

class OTelAgentObservability:
    """Agent observability using OpenTelemetry."""

    def __init__(self):
        # Setup tracer
        trace.set_tracer_provider(TracerProvider())
        span_processor = BatchSpanProcessor(OTLPSpanExporter())
        trace.get_tracer_provider().add_span_processor(span_processor)
        self.tracer = trace.get_tracer(__name__)

    @contextmanager
    def trace_llm_call(self, model: str):
        """Trace LLM call."""
        with self.tracer.start_as_current_span("llm_call") as span:
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.provider", "anthropic")
            yield span

    @contextmanager
    def trace_tool_call(self, tool_name: str):
        """Trace tool execution."""
        with self.tracer.start_as_current_span("tool_call") as span:
            span.set_attribute("tool.name", tool_name)
            yield span
```

## Common Pitfalls

### 1. Logging Too Little
**Problem:** Can't debug because critical information isn't logged.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>
<div class="code-body">

```python
# DON'T: Minimal logging
logger.info("Agent executed")

# DO: Comprehensive logging
logger.log_event(
    "agent_execution",
    "Agent completed task",
    user_id=user_id,
    duration_ms=duration,
    tokens=tokens,
    cost=cost,
    steps=num_steps,
    tools_used=tools
)
```

</div>
</div>

### 2. Logging Too Much
**Problem:** Logs are noisy and expensive to store.

```python
# DON'T: Log everything verbatim
logger.info(f"Full prompt: {ten_thousand_character_prompt}")

# DO: Log strategically with previews
logger.info(f"Prompt: {prompt[:200]}... (length: {len(prompt)})")
```

### 3. No Structured Logging
**Problem:** Can't query or analyze logs.

```python
# DON'T: Unstructured strings
logger.info(f"LLM call took {latency}ms and cost ${cost}")

# DO: Structured JSON
logger.info(json.dumps({
    "event": "llm_call",
    "latency_ms": latency,
    "cost_usd": cost
}))
```

### 4. Missing Error Context
**Problem:** Errors logged without context for reproduction.

```python
# DON'T: Minimal error logging
logger.error(str(e))

# DO: Full context
logger.error({
    "error": str(e),
    "error_type": type(e).__name__,
    "user_input": user_input,
    "agent_state": agent_state,
    "stack_trace": traceback.format_exc()
})
```

## Connections

**Builds on:**
- Production Architecture (designing observable systems)
- Evaluation Frameworks (metrics for measurement)
- Agent Fundamentals (understanding what to observe)

**Leads to:**
- Optimization (using observability data to improve performance)
- Debugging (using traces to fix issues)
- Cost Management (tracking and controlling costs)

**Related to:**
- APM (Application Performance Monitoring)
- Distributed Tracing (Jaeger, Zipkin)
- Log Aggregation (ELK stack, Splunk)

## Practice Problems

### 1. Design Observability Schema
Design a logging schema for a multi-agent research system with:
- Orchestrator agent
- Research agent
- Writing agent
- Review agent

What should be logged at each level? What metrics matter?

### 2. Build Dashboard
Using observability data, create a dashboard showing:
- Request volume over time
- p50/p95/p99 latency
- Cost per user
- Error rate
- Most-used tools

What visualizations would you choose? Why?

### 3. Implement Sampling
You're logging 1M agent executions per day. Storage costs are too high. Implement intelligent sampling that:
- Always logs errors
- Logs 100% of slow requests (>5s)
- Logs 10% of normal requests
- Logs 1% of cached responses

### 4. Trace Analysis
Given this trace:
```
1. User input (0ms)
2. Planning (100ms)
3. Tool call 1: search_web (2000ms)
4. Tool call 2: search_papers (3000ms)
5. Tool call 3: fetch_data (1000ms)
6. LLM synthesis (500ms)
7. Response (0ms)
Total: 6600ms
```

What optimizations would you suggest? How would you measure improvement?

<div class="callout-key">

**Key Concept Summary:** This guide covered the core concepts. Review the companion slides for visual summaries and the hands-on notebook for practice implementations.

</div>

## Further Reading

**Foundational Concepts:**
- "Observability Engineering" (Majors, Fong-Jones, Miranda) - O'Reilly
- "Distributed Systems Observability" (Sridharan) - O'Reilly
- OpenTelemetry Documentation

**LLM-Specific Observability:**
- LangSmith Documentation (LangChain's observability platform)
- Phoenix Documentation (Arize AI's LLM observability)
- Weights & Biases: "LLM Observability"
- Helicone: "LLM Monitoring Best Practices"

**Tools & Platforms:**
- OpenTelemetry (open standard)
- LangSmith (LangChain)
- Phoenix (Arize AI)
- Helicone
- Weights & Biases
- MLflow

**Advanced Topics:**
- Sampling strategies for high-volume systems
- Real-time anomaly detection
- Cost attribution in multi-tenant systems
- Privacy-preserving logging

---

**Next Steps:**

<a class="link-card" href="./02_observability_slides.md">
  <div class="link-card-title">Agent Observability and Monitoring — Companion Slides</div>
  <div class="link-card-description">Visual slide deck with diagrams, speaker notes, and key takeaways.</div>
</a>

<a class="link-card" href="../notebooks/01_deployment_patterns.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with working code and guided exercises.</div>
</a>
