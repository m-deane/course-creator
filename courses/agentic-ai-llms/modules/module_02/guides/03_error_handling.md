# Error Handling: Graceful Failure and Recovery

> **Reading time:** ~10 min | **Module:** 2 — Tool Use & Function Calling | **Prerequisites:** Module 2 — Tool Fundamentals

Tools fail. APIs timeout, data is malformed, permissions are denied. Robust agents don't crash on errors—they understand failures, communicate clearly, and either recover automatically or guide users to resolution.

<div class="callout-insight">

**Insight:** Errors are information, not just failures. A well-structured error message helps the LLM understand what went wrong and what to try next. Turn errors into actionable context.

</div>

---

## Error Categories

### Transient Errors (Retry-able)

Temporary failures that may succeed on retry:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>
<div class="code-body">

```python
TRANSIENT_ERRORS = {
    "rate_limit": {
        "retry": True,
        "max_retries": 3,
        "backoff": "exponential",
        "message": "Rate limit reached. Waiting before retry."
    },
    "timeout": {
        "retry": True,
        "max_retries": 2,
        "backoff": "linear",
        "message": "Request timed out. Retrying."
    },
    "connection_error": {
        "retry": True,
        "max_retries": 3,
        "backoff": "exponential",
        "message": "Connection failed. Retrying."
    },
    "server_error": {  # 5xx
        "retry": True,
        "max_retries": 2,
        "backoff": "exponential",
        "message": "Server error. Retrying."
    }
}
```

</div>
</div>

### Client Errors (Fix Required)

User or input errors requiring correction:

```python
CLIENT_ERRORS = {
    "invalid_input": {
        "retry": False,
        "message": "Invalid input provided",
        "action": "Correct the input and try again"
    },
    "not_found": {
        "retry": False,
        "message": "Requested resource not found",
        "action": "Verify the identifier exists"
    },
    "permission_denied": {
        "retry": False,
        "message": "Insufficient permissions",
        "action": "Request elevated access or try different resource"
    },
    "validation_error": {
        "retry": False,
        "message": "Data validation failed",
        "action": "Review and correct the data format"
    }
}
```

### Fatal Errors (Cannot Proceed)

Unrecoverable failures:

```python
FATAL_ERRORS = {
    "authentication_failed": {
        "retry": False,
        "message": "Authentication failed",
        "action": "Check API credentials"
    },
    "service_unavailable": {
        "retry": False,
        "message": "Service is unavailable",
        "action": "Service may be down for maintenance"
    },
    "quota_exceeded": {
        "retry": False,
        "message": "Usage quota exceeded",
        "action": "Upgrade plan or wait for quota reset"
    }
}
```

---

## Retry Strategies

### Exponential Backoff

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>
<div class="code-body">

```python
import time
from functools import wraps


def with_retry(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
    """Decorator for retrying functions with exponential backoff."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            last_error = None

            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except TransientError as e:
                    last_error = e
                    retries += 1

                    if retries > max_retries:
                        break

                    # Exponential backoff with jitter
                    delay = min(base_delay * (2 ** (retries - 1)), max_delay)
                    delay *= (0.5 + random.random())  # Add jitter
                    time.sleep(delay)

            raise MaxRetriesExceeded(f"Failed after {max_retries} retries: {last_error}")

        return wrapper
    return decorator


@with_retry(max_retries=3)
def call_external_api(endpoint: str, data: dict):
    """Call API with automatic retry on transient failures."""
    response = requests.post(endpoint, json=data, timeout=30)
    if response.status_code >= 500:
        raise TransientError(f"Server error: {response.status_code}")
    return response.json()
```

</div>
</div>

### Circuit Breaker Pattern

Prevent cascading failures:

```python
from datetime import datetime, timedelta
from enum import Enum


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        success_threshold: int = 2
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""

        if self.state == CircuitState.OPEN:
            if self._should_attempt_recovery():
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitOpenError("Circuit is open, request rejected")

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise

    def _should_attempt_recovery(self) -> bool:
        if self.last_failure_time is None:
            return True
        elapsed = datetime.now() - self.last_failure_time
        return elapsed > timedelta(seconds=self.recovery_timeout)

    def _record_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0

    def _record_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        self.success_count = 0

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN


# Usage
api_circuit = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

def safe_api_call(endpoint: str):
    return api_circuit.call(requests.get, endpoint)
```

---

## Error Response Structure

### Standard Error Format

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>
<div class="code-body">

```python
from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class ToolError:
    """Standardized error response for tools."""
    error_type: str
    message: str
    suggestion: Optional[str] = None
    details: Optional[dict] = None
    retryable: bool = False

    def to_json(self) -> str:
        return json.dumps({
            "status": "error",
            "error": {
                "type": self.error_type,
                "message": self.message,
                "suggestion": self.suggestion,
                "details": self.details,
                "retryable": self.retryable
            }
        })


# Usage examples
def handle_api_error(e: Exception, context: dict) -> str:
    """Convert exceptions to structured error responses."""

    if isinstance(e, requests.exceptions.Timeout):
        return ToolError(
            error_type="timeout",
            message="The request timed out",
            suggestion="Try again with a simpler query or later",
            details={"timeout_seconds": 30},
            retryable=True
        ).to_json()

    elif isinstance(e, requests.exceptions.ConnectionError):
        return ToolError(
            error_type="connection_error",
            message="Could not connect to the service",
            suggestion="Check if the service is available",
            retryable=True
        ).to_json()

    elif isinstance(e, PermissionError):
        return ToolError(
            error_type="permission_denied",
            message=f"Access denied: {context.get('resource', 'unknown')}",
            suggestion="Request access or try a different resource",
            retryable=False
        ).to_json()

    else:
        return ToolError(
            error_type="unknown",
            message=str(e),
            suggestion="Contact support if this persists",
            details={"exception_type": type(e).__name__},
            retryable=False
        ).to_json()
```

</div>
</div>

### Informative Error Messages

```python
# Bad: Uninformative
{"error": "Failed"}

# Good: Actionable
{
    "status": "error",
    "error": {
        "type": "not_found",
        "message": "User 'john_doe123' not found in database",
        "suggestion": "Check the username spelling or search by email instead",
        "details": {
            "searched_in": "users",
            "search_field": "username",
            "alternatives": ["search_by_email", "list_all_users"]
        }
    }
}
```

---

## Agent-Level Error Handling

### Graceful Degradation

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>
<div class="code-body">

```python
class RobustToolAgent:
    """Agent with comprehensive error handling."""

    def execute_tool(self, name: str, arguments: dict) -> dict:
        """Execute tool with full error handling."""

        # Pre-execution validation
        validation_errors = self.validate_arguments(name, arguments)
        if validation_errors:
            return {
                "status": "error",
                "error_type": "validation",
                "errors": validation_errors,
                "suggestion": "Correct the arguments and retry"
            }

        # Attempt execution with retry
        for attempt in range(3):
            try:
                result = self.tool_handlers[name](**arguments)
                return {"status": "success", "data": result}

            except TransientError as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                return self.format_error(e, retryable=True)

            except ClientError as e:
                return self.format_error(e, retryable=False)

            except Exception as e:
                return self.format_error(e, retryable=False)

        return {
            "status": "error",
            "error_type": "max_retries",
            "message": "Failed after maximum retries"
        }

    def run_with_fallbacks(self, query: str) -> str:
        """Run with fallback strategies."""

        messages = [{"role": "user", "content": query}]

        # Try primary model
        try:
            response = self.call_model(messages, model="claude-3-5-sonnet-20241022")
        except Exception as e:
            # Fallback to faster model
            try:
                response = self.call_model(messages, model="claude-3-haiku-20240307")
            except Exception:
                return "I'm having trouble processing your request. Please try again later."

        return self.process_response(response)
```

</div>
</div>

### Error Recovery Strategies

```python
def execute_with_recovery(self, tool_name: str, arguments: dict) -> dict:
    """Execute tool with automatic recovery strategies."""

    result = self.execute_tool(tool_name, arguments)

    if result["status"] == "success":
        return result

    error_type = result.get("error_type")

    # Strategy 1: Parameter adjustment
    if error_type == "validation":
        adjusted = self.adjust_parameters(tool_name, arguments, result["errors"])
        if adjusted:
            return self.execute_tool(tool_name, adjusted)

    # Strategy 2: Alternative tool
    if error_type == "not_found":
        alternative = self.find_alternative_tool(tool_name)
        if alternative:
            return self.execute_tool(alternative, arguments)

    # Strategy 3: Decompose request
    if error_type == "too_complex":
        subtasks = self.decompose_task(tool_name, arguments)
        results = [self.execute_tool(t["name"], t["args"]) for t in subtasks]
        return self.combine_results(results)

    # No recovery possible
    return result
```

---

## Communicating Errors to Users

### System Prompt Error Handling

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>
<div class="code-body">

```python
system_prompt = """You are a helpful assistant with access to tools.

When a tool returns an error:
1. Acknowledge the issue briefly
2. Explain what went wrong in simple terms
3. Suggest next steps or alternatives
4. Do NOT repeatedly retry the same failing action

Example good response to errors:
"I wasn't able to find that user in our system. The username might be misspelled,
or the account may not exist yet. Would you like me to search by email instead,
or help you create a new account?"

Example bad response:
"Error: User not found. Let me try again... Error: User not found. Let me try again..."
"""
```

</div>
</div>

### Error-Aware Agent Loop

```python
def run_with_error_awareness(self, query: str) -> str:
    """Agent loop that handles errors intelligently."""

    messages = [{"role": "user", "content": query}]
    error_count = {}
    max_consecutive_errors = 2

    for turn in range(10):
        response = self.call_model(messages)

        if response.stop_reason != "tool_use":
            return self.extract_text(response)

        # Handle tool calls
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tool_call in self.get_tool_calls(response):
            result = self.execute_tool(tool_call.name, tool_call.input)

            # Track errors
            if result["status"] == "error":
                error_key = f"{tool_call.name}:{result.get('error_type')}"
                error_count[error_key] = error_count.get(error_key, 0) + 1

                # Prevent infinite error loops
                if error_count[error_key] >= max_consecutive_errors:
                    result["suggestion"] = (
                        f"This is the {error_count[error_key]}th failure. "
                        "Consider a different approach or inform the user."
                    )

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": json.dumps(result)
            })

        messages.append({"role": "user", "content": tool_results})

    return "I've reached the maximum number of attempts. Please try rephrasing your request."
```

---

## Logging and Observability

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


class ToolLogger:
    """Logger for tool execution with structured output."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

    def log_execution(
        self,
        tool_name: str,
        arguments: dict,
        result: dict,
        duration_ms: float
    ):
        """Log a tool execution."""

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "tool_execution",
            "tool": tool_name,
            "arguments": arguments,
            "status": result.get("status"),
            "duration_ms": duration_ms
        }

        if result.get("status") == "error":
            log_entry["error"] = result.get("error")
            self.logger.warning(json.dumps(log_entry))
        else:
            self.logger.info(json.dumps(log_entry))

    def log_retry(self, tool_name: str, attempt: int, error: str):
        """Log a retry attempt."""
        self.logger.warning(json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "event": "retry",
            "tool": tool_name,
            "attempt": attempt,
            "error": error
        }))
```

</div>
</div>

### Metrics Collection

```python
from collections import defaultdict
import time


class ToolMetrics:
    """Collect metrics on tool execution."""

    def __init__(self):
        self.execution_times = defaultdict(list)
        self.success_count = defaultdict(int)
        self.error_count = defaultdict(int)
        self.error_types = defaultdict(lambda: defaultdict(int))

    def record_execution(
        self,
        tool_name: str,
        success: bool,
        duration_ms: float,
        error_type: str = None
    ):
        self.execution_times[tool_name].append(duration_ms)

        if success:
            self.success_count[tool_name] += 1
        else:
            self.error_count[tool_name] += 1
            if error_type:
                self.error_types[tool_name][error_type] += 1

    def get_stats(self, tool_name: str) -> dict:
        times = self.execution_times[tool_name]
        total = self.success_count[tool_name] + self.error_count[tool_name]

        return {
            "total_calls": total,
            "success_rate": self.success_count[tool_name] / total if total else 0,
            "avg_duration_ms": sum(times) / len(times) if times else 0,
            "p95_duration_ms": sorted(times)[int(len(times) * 0.95)] if times else 0,
            "error_breakdown": dict(self.error_types[tool_name])
        }
```

<div class="callout-key">

**Key Concept Summary:** This guide covered the core concepts. Review the companion slides for visual summaries and the hands-on notebook for practice implementations.

</div>

---

*Errors are inevitable—poor error handling is not. Build agents that fail gracefully, communicate clearly, and recover when possible.*



## Practice Questions

1. Explain in your own words how the concepts in this guide relate to building production agents.
2. What are the key tradeoffs you need to consider when applying these techniques?
3. Describe a scenario where the approach from this guide would be the wrong choice, and what you would use instead.

---

**Next Steps:**

<a class="link-card" href="./03_error_handling_slides.md">
  <div class="link-card-title">Error Handling for Tool Calls — Companion Slides</div>
  <div class="link-card-description">Visual slide deck with diagrams, speaker notes, and key takeaways.</div>
</a>

<a class="link-card" href="../notebooks/01_basic_tools.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with working code and guided exercises.</div>
</a>
