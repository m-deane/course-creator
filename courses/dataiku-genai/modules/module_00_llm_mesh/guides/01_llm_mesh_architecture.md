# LLM Mesh Architecture

## In Brief

Dataiku LLM Mesh is an abstraction layer that provides unified access to multiple Large Language Model providers through a single, governed interface. It centralizes configuration, cost tracking, access control, and monitoring while allowing seamless switching between providers without code changes.

> 💡 **Key Insight:** **The core value proposition:** LLM Mesh transforms enterprise LLM usage from fragmented, ungoverned API calls scattered across projects into a centralized, auditable, cost-managed platform that enables experimentation while maintaining control.

Think of LLM Mesh as a "router" sitting between your applications and multiple LLM providers—it handles authentication, load balancing, fallback, cost tracking, and compliance while presenting a simple, unified API to developers.

## Formal Definition

**LLM Mesh** is a provider-agnostic orchestration layer that:

1. **Abstracts** multiple LLM provider APIs behind a unified interface
2. **Routes** requests to appropriate providers based on configuration or load
3. **Tracks** usage metrics including tokens, costs, and latency
4. **Enforces** access control policies and rate limits
5. **Monitors** health and performance of all connected providers
6. **Logs** all interactions for audit and compliance

The architecture follows a **middleware pattern** where the mesh intercepts all LLM requests, applies governance policies, routes to providers, and captures telemetry before returning responses.

## Intuitive Explanation

### The Problem Without LLM Mesh

> ⚠️ **Without LLM Mesh:**

Imagine an organization where:
- Team A uses OpenAI directly, hardcoding API keys in scripts
- Team B uses Anthropic with keys in environment variables
- Team C uses Azure OpenAI with yet another authentication method
- Nobody knows the total monthly LLM spend
- No visibility into who's using which models
- Switching providers requires code changes across dozens of projects
- No consistent error handling or retry logic

### The Solution With LLM Mesh

Now imagine a central "LLM traffic control center":
- All teams connect through one interface
- Administrators configure provider connections once
- Usage automatically tracked by team, project, and user
- Switch from GPT-4 to Claude by changing one setting
- Automatic failover if primary provider is down
- Centralized audit logs for compliance
- Budget alerts when spending approaches limits

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dataiku DSS Platform                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                │
│  │  Project A │  │  Project B │  │  Project C │                │
│  │  (Recipes) │  │ (Notebooks)│  │  (Webapps) │                │
│  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘                │
│         │                │                │                      │
│         └────────────────┼────────────────┘                      │
│                          │                                       │
│                          ▼                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    LLM Mesh Layer                         │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐  │  │
│  │  │   Router    │  │ Gov & Audit  │  │ Cost Tracker    │  │  │
│  │  │ Load Balance│  │ Access Ctrl  │  │ Metrics         │  │  │
│  │  └─────────────┘  └──────────────┘  └─────────────────┘  │  │
│  │                                                            │  │
│  │  ┌──────────────────────────────────────────────────────┐ │  │
│  │  │        Connection Pool (Provider Configs)            │ │  │
│  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐             │ │  │
│  │  │  │Claude-1 │  │ GPT-4-1 │  │ Azure-1 │             │ │  │
│  │  │  │Primary  │  │ Backup  │  │ Regional│             │ │  │
│  │  │  └─────────┘  └─────────┘  └─────────┘             │ │  │
│  │  └──────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          │                                       │
└──────────────────────────┼───────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────┐ ┌──────────────────┐
│ Anthropic API   │ │ OpenAI API  │ │ Azure OpenAI API │
│ Claude Models   │ │ GPT Models  │ │ GPT Models       │
└─────────────────┘ └─────────────┘ └──────────────────┘
```

## Code Implementation

### Basic Architecture Components

```python
# Conceptual implementation showing key architecture components

class LLMMeshRouter:
    """
    Routes LLM requests to appropriate providers based on
    configuration, load, and availability.
    """

    def __init__(self, connections: dict):
        self.connections = connections
        self.load_balancer = LoadBalancer()
        self.health_checker = HealthChecker()

    def route_request(
        self,
        request: LLMRequest,
        routing_strategy: str = "primary"
    ) -> LLMConnection:
        """
        Select appropriate connection for request.

        Strategies:
        - primary: Use configured primary connection
        - round_robin: Distribute load evenly
        - least_latency: Route to fastest provider
        - cost_optimized: Route to cheapest provider
        """
        available_connections = [
            conn for conn in self.connections.values()
            if self.health_checker.is_healthy(conn)
        ]

        if not available_connections:
            raise NoHealthyConnectionsError("All LLM connections are down")

        if routing_strategy == "primary":
            return available_connections[0]
        elif routing_strategy == "round_robin":
            return self.load_balancer.get_next(available_connections)
        elif routing_strategy == "least_latency":
            return min(available_connections, key=lambda c: c.avg_latency)
        elif routing_strategy == "cost_optimized":
            return min(available_connections, key=lambda c: c.cost_per_token)


class GovernanceLayer:
    """
    Enforces access control, rate limits, and budget constraints.
    """

    def __init__(self):
        self.access_control = AccessControl()
        self.rate_limiter = RateLimiter()
        self.budget_enforcer = BudgetEnforcer()

    def check_permissions(
        self,
        user: str,
        connection: str,
        project: str
    ) -> bool:
        """Verify user has access to connection in project."""
        return self.access_control.has_permission(
            user=user,
            connection=connection,
            project=project,
            action="llm.generate"
        )

    def check_rate_limit(
        self,
        connection: str,
        user: str
    ) -> tuple[bool, dict]:
        """Check if request would exceed rate limits."""
        status = self.rate_limiter.check(
            key=f"{connection}:{user}",
            limits={
                "requests_per_minute": 100,
                "tokens_per_minute": 50000
            }
        )
        return status['allowed'], status

    def check_budget(
        self,
        project: str,
        estimated_cost: float
    ) -> tuple[bool, dict]:
        """Check if request would exceed budget."""
        status = self.budget_enforcer.check(
            project=project,
            cost=estimated_cost,
            period="daily"
        )
        return status['within_budget'], status


class TelemetryCollector:
    """
    Collects metrics, logs, and traces for all LLM interactions.
    """

    def __init__(self):
        self.metrics = MetricsStore()
        self.audit_log = AuditLog()

    def record_request(
        self,
        request: LLMRequest,
        response: LLMResponse,
        metadata: dict
    ):
        """Record request details for monitoring and audit."""

        # Metrics
        self.metrics.increment("llm_requests_total", tags={
            "connection": metadata['connection'],
            "model": metadata['model'],
            "status": "success" if response.success else "error"
        })

        self.metrics.observe("llm_latency_seconds",
            value=response.latency,
            tags={"connection": metadata['connection']}
        )

        self.metrics.increment("llm_tokens_total",
            value=response.usage.total_tokens,
            tags={"connection": metadata['connection']}
        )

        # Audit log
        self.audit_log.write({
            "timestamp": datetime.now().isoformat(),
            "user": metadata['user'],
            "project": metadata['project'],
            "connection": metadata['connection'],
            "model": metadata['model'],
            "prompt_preview": request.prompt[:200],
            "tokens": response.usage.total_tokens,
            "cost": response.estimated_cost,
            "latency_ms": response.latency * 1000,
            "success": response.success
        })


class LLMMesh:
    """
    Main LLM Mesh orchestrator that ties all components together.
    """

    def __init__(self):
        self.router = LLMMeshRouter(connections={})
        self.governance = GovernanceLayer()
        self.telemetry = TelemetryCollector()

    def generate(
        self,
        connection_name: str,
        prompt: str,
        user: str,
        project: str,
        **kwargs
    ) -> LLMResponse:
        """
        Main entry point for LLM generation with full governance.
        """

        # 1. Check permissions
        if not self.governance.check_permissions(user, connection_name, project):
            raise PermissionDeniedError(f"User {user} cannot access {connection_name}")

        # 2. Check rate limits
        allowed, rate_status = self.governance.check_rate_limit(connection_name, user)
        if not allowed:
            raise RateLimitExceededError(
                f"Rate limit exceeded: {rate_status['remaining']} remaining"
            )

        # 3. Estimate cost and check budget
        estimated_tokens = len(prompt) / 4  # Rough estimate
        estimated_cost = estimated_tokens * 0.00001  # Placeholder
        within_budget, budget_status = self.governance.check_budget(
            project, estimated_cost
        )
        if not within_budget:
            raise BudgetExceededError(
                f"Budget exceeded: ${budget_status['used']:.2f} of "
                f"${budget_status['limit']:.2f}"
            )

        # 4. Route to appropriate provider
        connection = self.router.route_request(
            request=LLMRequest(prompt=prompt, **kwargs),
            routing_strategy=kwargs.get('routing', 'primary')
        )

        # 5. Execute request
        start_time = time.time()
        try:
            response = connection.complete(prompt, **kwargs)
            response.latency = time.time() - start_time
            response.success = True
        except Exception as e:
            response = LLMResponse(
                text="",
                error=str(e),
                success=False,
                latency=time.time() - start_time
            )

        # 6. Record telemetry
        self.telemetry.record_request(
            request=LLMRequest(prompt=prompt, **kwargs),
            response=response,
            metadata={
                'user': user,
                'project': project,
                'connection': connection_name,
                'model': kwargs.get('model', connection.default_model)
            }
        )

        return response
```

### Using LLM Mesh in Dataiku

```python
# In a Dataiku Python recipe or notebook

import dataiku
from dataiku.llm import LLM

# Simple usage - all governance automatic
llm = LLM("claude-production")

response = llm.complete(
    prompt="Analyze this commodity report: ...",
    max_tokens=500
)

print(response.text)
print(f"Tokens used: {response.usage.total_tokens}")
print(f"Cost: ${response.estimated_cost:.4f}")

# The LLM Mesh automatically:
# - Verified your permissions
# - Checked rate limits
# - Validated budget
# - Routed to healthy connection
# - Logged the interaction
# - Tracked metrics
```

## Common Pitfalls

### 1. Hardcoding Connection Names

**Problem:**
```python
# Bad - hardcoded connection
llm = LLM("claude-production")
```

**Solution:**
```python
# Good - use project variables
import dataiku
project = dataiku.api_client().get_project(dataiku.default_project_key())
connection_name = project.get_variable("llm_connection", "claude-production")
llm = LLM(connection_name)
```

### 2. Ignoring Rate Limits

**Problem:**
```python
# Will hit rate limits quickly
for text in large_dataset:
    response = llm.complete(text)  # No delay
```

**Solution:**
```python
import time
from concurrent.futures import ThreadPoolExecutor

def process_with_backoff(text, llm):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return llm.complete(text)
        except RateLimitError:
            time.sleep(2 ** attempt)
    raise Exception("Max retries exceeded")

# Controlled parallelism
with ThreadPoolExecutor(max_workers=5) as executor:
    results = executor.map(lambda t: process_with_backoff(t, llm), large_dataset)
```

### 3. No Error Handling

**Problem:**
```python
response = llm.complete(prompt)  # What if it fails?
data = json.loads(response.text)  # What if not valid JSON?
```

**Solution:**
```python
try:
    response = llm.complete(prompt, max_tokens=200)
    data = json.loads(response.text)
except RateLimitError as e:
    # Handle rate limiting
    logger.warning(f"Rate limited: {e}")
    time.sleep(60)
except JSONDecodeError:
    # Handle invalid JSON
    logger.error(f"Invalid JSON response: {response.text[:100]}")
    data = {"error": "parse_failed", "raw": response.text}
except Exception as e:
    # Handle other errors
    logger.error(f"LLM request failed: {e}")
    raise
```

### 4. Not Monitoring Costs

**Problem:**
```python
# No cost awareness
for row in df.iterrows():
    llm.complete(row['long_text'])  # Could be expensive!
```

**Solution:**
```python
from dataiku.monitoring import CostTracker

tracker = CostTracker()

for row in df.iterrows():
    # Estimate before calling
    estimated_cost = tracker.estimate_cost(
        text=row['long_text'],
        model="claude-sonnet-4-20250514"
    )

    if tracker.daily_total + estimated_cost > tracker.daily_budget:
        logger.warning("Approaching daily budget, stopping")
        break

    response = llm.complete(row['long_text'])
    tracker.record(response.usage, response.estimated_cost)

print(f"Total cost today: ${tracker.daily_total:.2f}")
```

## Connections to Other Topics

### Prerequisites
- **API fundamentals**: Understanding REST APIs, authentication, rate limiting
- **Dataiku basics**: Projects, datasets, recipes, connections
- **Python fundamentals**: Functions, error handling, async programming

### Enables
- **Prompt Design (Module 1)**: LLM Mesh connections are used in Prompt Studios
- **RAG Applications (Module 2)**: Knowledge Banks use LLM Mesh for embeddings and generation
- **Custom Applications (Module 3)**: Python recipes leverage LLM Mesh API
- **Deployment (Module 4)**: API endpoints expose LLM Mesh capabilities

### Related Concepts
- **API Gateway Pattern**: LLM Mesh is essentially an API gateway for LLM providers
- **Circuit Breaker**: Health checking and failover prevent cascading failures
- **Observability**: Telemetry collection enables monitoring and debugging
- **Multi-tenancy**: Access control and quotas enable safe sharing

## Practice Problems

### Problem 1: Basic Architecture Understanding

**Question:** Draw a diagram showing the flow of an LLM request from a Dataiku Python recipe through LLM Mesh to an external provider. Label each component and the data it captures.

**Difficulty:** Easy

<details>
<summary>Solution</summary>

```
Python Recipe
     │
     ├─ User: data_scientist
     ├─ Project: COMMODITY_ANALYSIS
     └─ Request: "Analyze this report..."
     │
     ▼
LLM Mesh
     │
     ├─ Governance Layer
     │    ├─ Check: User permissions ✓
     │    ├─ Check: Rate limit ✓
     │    └─ Check: Budget ✓
     │
     ├─ Router
     │    └─ Select: claude-production (healthy, primary)
     │
     ├─ Connection Pool
     │    └─ Get: Claude connection with API key
     │
     └─ Telemetry
          ├─ Log: Audit entry
          ├─ Metric: llm_requests_total++
          └─ Metric: llm_tokens_total += 1234
     │
     ▼
Anthropic API
     │
     └─ Response: {text: "...", usage: {tokens: 1234}}
     │
     ▼
LLM Mesh (return path)
     │
     ├─ Telemetry
     │    ├─ Metric: llm_latency_seconds = 2.3
     │    └─ Metric: llm_cost_usd += 0.0123
     │
     └─ Return response to recipe
```
</details>

### Problem 2: Failover Configuration

**Question:** Design a failover strategy for a production system that needs 99.9% uptime. You have access to Anthropic Claude (primary, fastest, most expensive), OpenAI GPT-4 (secondary, medium speed/cost), and Azure OpenAI (tertiary, slowest, cheapest). Write pseudocode for the failover logic.

**Difficulty:** Medium

<details>
<summary>Solution</summary>

```python
class ResilientLLMRouter:
    def __init__(self):
        self.connections = [
            {
                'name': 'claude-primary',
                'priority': 1,
                'max_retries': 2,
                'timeout': 30
            },
            {
                'name': 'gpt4-secondary',
                'priority': 2,
                'max_retries': 2,
                'timeout': 45
            },
            {
                'name': 'azure-tertiary',
                'priority': 3,
                'max_retries': 3,
                'timeout': 60
            }
        ]

    def generate_with_failover(self, prompt, **kwargs):
        last_error = None

        for connection in sorted(self.connections, key=lambda c: c['priority']):
            for attempt in range(connection['max_retries']):
                try:
                    llm = LLM(connection['name'])
                    response = llm.complete(
                        prompt,
                        timeout=connection['timeout'],
                        **kwargs
                    )

                    # Log successful failover if not primary
                    if connection['priority'] > 1:
                        logger.warning(
                            f"Failed over to {connection['name']} "
                            f"(priority {connection['priority']})"
                        )

                    return response

                except TimeoutError as e:
                    logger.warning(
                        f"{connection['name']} timeout on attempt {attempt+1}"
                    )
                    last_error = e
                    time.sleep(2 ** attempt)  # Exponential backoff

                except RateLimitError as e:
                    # Don't retry on rate limit, move to next connection
                    logger.warning(f"{connection['name']} rate limited")
                    last_error = e
                    break

                except Exception as e:
                    logger.error(
                        f"{connection['name']} error: {e}"
                    )
                    last_error = e

        # All connections failed
        raise AllConnectionsFailedError(
            f"All LLM connections failed. Last error: {last_error}"
        )
```
</details>

### Problem 3: Cost Optimization

**Question:** You have a dataset of 10,000 customer reviews to analyze. Claude Sonnet costs $3/$15 per million input/output tokens, Haiku costs $0.25/$1.25. Each review is ~200 tokens, expected output is ~100 tokens. Your budget is $50. Design an optimization strategy.

**Difficulty:** Hard

<details>
<summary>Solution</summary>

```python
def optimize_batch_processing(reviews, budget=50.0):
    """
    Strategy:
    1. Use cheap model (Haiku) for simple cases
    2. Use expensive model (Sonnet) for complex cases
    3. Track costs in real-time
    4. Stop if budget exceeded
    """

    # Model costs per million tokens
    COSTS = {
        'haiku': {'input': 0.25, 'output': 1.25},
        'sonnet': {'input': 3.0, 'output': 15.0}
    }

    def estimate_cost(input_tokens, output_tokens, model):
        return (
            input_tokens * COSTS[model]['input'] / 1_000_000 +
            output_tokens * COSTS[model]['output'] / 1_000_000
        )

    def classify_complexity(review_text):
        """Determine if review needs sophisticated model."""
        # Simple heuristic: length and keyword presence
        complex_indicators = [
            'however', 'although', 'mixed', 'complicated',
            'on one hand', 'sarcasm', 'irony'
        ]

        if len(review_text) > 500:
            return 'complex'
        if any(ind in review_text.lower() for ind in complex_indicators):
            return 'complex'
        return 'simple'

    results = []
    total_cost = 0.0
    haiku_llm = LLM('claude-haiku')
    sonnet_llm = LLM('claude-sonnet')

    for i, review in enumerate(reviews):
        # Classify complexity
        complexity = classify_complexity(review['text'])

        # Select model
        if complexity == 'simple':
            model_name = 'haiku'
            llm = haiku_llm
        else:
            model_name = 'sonnet'
            llm = sonnet_llm

        # Estimate cost
        estimated_cost = estimate_cost(200, 100, model_name)

        if total_cost + estimated_cost > budget:
            logger.warning(
                f"Budget exhausted after {i} reviews. "
                f"${total_cost:.2f} spent."
            )
            break

        # Process
        try:
            response = llm.complete(
                f"Analyze sentiment: {review['text']}",
                max_tokens=100
            )

            actual_cost = estimate_cost(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                model_name
            )
            total_cost += actual_cost

            results.append({
                'review_id': review['id'],
                'analysis': response.text,
                'model': model_name,
                'complexity': complexity,
                'cost': actual_cost
            })

        except Exception as e:
            logger.error(f"Failed to process review {review['id']}: {e}")

    print(f"\nProcessing Summary:")
    print(f"Reviews processed: {len(results)}")
    print(f"Total cost: ${total_cost:.2f}")
    print(f"Avg cost per review: ${total_cost/len(results):.4f}")

    # Model distribution
    haiku_count = sum(1 for r in results if r['model'] == 'haiku')
    sonnet_count = len(results) - haiku_count
    print(f"Haiku: {haiku_count} ({haiku_count/len(results)*100:.1f}%)")
    print(f"Sonnet: {sonnet_count} ({sonnet_count/len(results)*100:.1f}%)")

    return results

# Expected outcome:
# - ~8,000-9,000 reviews processed (within $50 budget)
# - ~70-80% use Haiku (simple sentiment)
# - ~20-30% use Sonnet (complex/nuanced cases)
# - Average cost: ~$0.005-0.006 per review
```
</details>

## Further Reading

### Official Documentation
- [Dataiku LLM Mesh Documentation](https://doc.dataiku.com/dss/latest/llm-mesh/)
- [Dataiku Generative AI Features](https://www.dataiku.com/product/key-capabilities/generative-ai/)

### Architecture Patterns
- **API Gateway Pattern**: *Microservices Patterns* by Chris Richardson
- **Circuit Breaker**: Martin Fowler's article on resilience patterns
- **Observability**: *Distributed Systems Observability* by Cindy Sridharan

### Related Topics
- **Rate Limiting Algorithms**: Token bucket, leaky bucket, sliding window
- **Load Balancing**: Round-robin, least connections, consistent hashing
- **Cost Optimization**: Resource allocation, budget constraints, optimization under uncertainty

### Dataiku-Specific
- Guide 02: Provider Setup (detailed connection configuration)
- Guide 03: Governance (access control and compliance)
- Module 3: Custom Applications (programmatic LLM Mesh usage)
- Module 4: Deployment (production monitoring and alerting)
