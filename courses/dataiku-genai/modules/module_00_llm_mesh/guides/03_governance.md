# Governance and Access Control in LLM Mesh

> **Reading time:** ~8 min | **Module:** 0 — Llm Mesh | **Prerequisites:** Basic Python, familiarity with LLM concepts

## In Brief

LLM Mesh governance enables centralized control over LLM access, cost tracking, and usage policies across your organization. It provides visibility into who uses LLMs, how much they cost, and ensures compliance with security policies.

<div class="callout-insight">

<strong>Key Insight:</strong> Enterprise Gen AI requires governance layer that balances innovation with control. LLM Mesh provides this through connection-level access control, project-based quotas, and comprehensive audit trails without requiring changes to application code.

</div>

<div class="callout-key">

<strong>Key Concept:</strong> LLM Mesh governance enables centralized control over LLM access, cost tracking, and usage policies across your organization. It provides visibility into who uses LLMs, how much they cost, and ensures compliance with security policies.

</div>

## Formal Definition

**LLM Governance** is the systematic management of generative AI usage through:
- **Access Control**: Role-based permissions at connection and project levels
- **Cost Management**: Token usage tracking and budget allocation
- **Audit Logging**: Complete history of all LLM interactions
- **Rate Limiting**: Request throttling to prevent abuse and manage costs
- **Compliance Enforcement**: Policy application across all LLM usage

## Intuitive Explanation

Think of LLM Mesh governance like corporate credit card management. Each employee (user) has access to certain cards (LLM connections) with spending limits (quotas). All transactions (API calls) are logged for accounting (audit trail), and the finance team (admins) can see spending patterns and enforce budgets without being involved in every purchase decision.

## Visual Representation

```

┌─────────────────────────────────────────────────────────────┐
│                   LLM Mesh Governance Layer                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │Access Control│  │Cost Tracking │  │  Audit Logging  │  │
│  │              │  │              │  │                 │  │
│  │ • Groups     │  │ • By user    │  │ • All prompts   │  │
│  │ • Projects   │  │ • By project │  │ • All responses │  │
│  │ • Connections│  │ • By model   │  │ • Token counts  │  │
│  │ • Rate limits│  │ • Daily/month│  │ • Timestamps    │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬────────┘  │
│         │                  │                    │           │
│         └──────────────────┴────────────────────┘           │
│                            │                                │
└────────────────────────────┼────────────────────────────────┘
                             ▼
              ┌──────────────────────────────┐
              │      LLM API Calls           │
              │  (governed and monitored)    │
              └──────────────────────────────┘
```

## Code Implementation

### Setting Up Access Control


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import dataiku

# Admin operations - connection access management
client = dataiku.api_client()

def configure_connection_access(
    connection_name: str,
    allowed_groups: list[str],
    daily_token_limit: int = None
) -> dict:
    """
    Configure access control for an LLM connection.

    Args:
        connection_name: Name of LLM Mesh connection
        allowed_groups: List of group names with access
        daily_token_limit: Optional daily token quota per user

    Returns:
        Configuration details
    """
    connection = client.get_connection(connection_name)

    # Set group permissions
    connection.set_permission(
        groups=allowed_groups,
        permissions=['READ', 'USE']
    )

    # Configure rate limits
    if daily_token_limit:
        connection.set_rate_limit(
            type='daily_tokens',
            limit=daily_token_limit,
            scope='user'
        )

    return {
        'connection': connection_name,
        'allowed_groups': allowed_groups,
        'daily_token_limit': daily_token_limit
    }

# Example usage
configure_connection_access(
    connection_name='anthropic-claude',
    allowed_groups=['data-scientists', 'analysts'],
    daily_token_limit=100_000
)
```

</div>
</div>

### Project-Level Quotas


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def set_project_llm_quota(
    project_key: str,
    monthly_budget_usd: float,
    alert_threshold: float = 0.8
) -> dict:
    """
    Set LLM usage quota for a project.

    Args:
        project_key: Dataiku project key
        monthly_budget_usd: Maximum monthly spend
        alert_threshold: Alert when usage reaches this fraction (0-1)

    Returns:
        Quota configuration
    """
    project = client.get_project(project_key)

    # Configure project-level quota
    project.set_llm_quota(
        monthly_budget=monthly_budget_usd,
        alert_at=alert_threshold,
        hard_limit=True  # Stop requests when quota exceeded
    )

    # Set up notifications
    project.add_quota_alert(
        recipients=['project-admins@company.com'],
        when='threshold_reached'
    )

    return {
        'project': project_key,
        'monthly_budget': monthly_budget_usd,
        'alert_threshold': alert_threshold
    }

# Example
set_project_llm_quota(
    project_key='GENAI_PILOT',
    monthly_budget_usd=500.00,
    alert_threshold=0.8  # Alert at 80% usage
)
```

</div>
</div>

### Cost Tracking and Reporting


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
from datetime import datetime, timedelta
import pandas as pd

def get_llm_usage_report(
    project_key: str = None,
    start_date: datetime = None,
    end_date: datetime = None,
    group_by: str = 'user'
) -> pd.DataFrame:
    """
    Generate LLM usage report.

    Args:
        project_key: Optional project filter
        start_date: Report start date
        end_date: Report end date
        group_by: Aggregation level ('user', 'project', 'model')

    Returns:
        DataFrame with usage metrics
    """
    # Default to last 30 days
    if not end_date:
        end_date = datetime.now()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    # Query usage logs
    usage_logs = client.get_llm_usage_logs(
        project=project_key,
        start_date=start_date,
        end_date=end_date
    )

    # Convert to DataFrame
    df = pd.DataFrame(usage_logs)

    # Calculate costs (example pricing)
    model_costs = {
        'claude-sonnet-4': {'input': 3.0, 'output': 15.0},  # per 1M tokens
        'gpt-4o': {'input': 2.5, 'output': 10.0}
    }

    def calculate_cost(row):
        costs = model_costs.get(row['model'], {'input': 0, 'output': 0})
        input_cost = (row['input_tokens'] / 1_000_000) * costs['input']
        output_cost = (row['output_tokens'] / 1_000_000) * costs['output']
        return input_cost + output_cost

    df['cost_usd'] = df.apply(calculate_cost, axis=1)

    # Aggregate by specified dimension
    report = df.groupby(group_by).agg({
        'request_id': 'count',
        'input_tokens': 'sum',
        'output_tokens': 'sum',
        'cost_usd': 'sum'
    }).rename(columns={'request_id': 'request_count'})

    report['total_tokens'] = report['input_tokens'] + report['output_tokens']

    return report.sort_values('cost_usd', ascending=False)

# Usage by project
project_usage = get_llm_usage_report(group_by='project')
print(project_usage)

# Usage by user in specific project
user_usage = get_llm_usage_report(
    project_key='GENAI_PILOT',
    group_by='user'
)
print(f"\nTop users by cost:")
print(user_usage.head(10))
```

</div>
</div>

### Audit Logging


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def query_audit_logs(
    connection_name: str = None,
    user: str = None,
    start_date: datetime = None,
    end_date: datetime = None,
    include_prompts: bool = False
) -> list[dict]:
    """
    Query LLM Mesh audit logs.

    Args:
        connection_name: Filter by connection
        user: Filter by user
        start_date: Start of time range
        end_date: End of time range
        include_prompts: Include full prompt content

    Returns:
        List of audit log entries
    """
    logs = client.get_llm_audit_logs(
        connection=connection_name,
        user=user,
        start_date=start_date,
        end_date=end_date,
        include_content=include_prompts
    )

    # Format logs
    formatted_logs = []
    for log in logs:
        entry = {
            'timestamp': log['timestamp'],
            'user': log['user'],
            'project': log['project'],
            'connection': log['connection'],
            'model': log['model'],
            'input_tokens': log['input_tokens'],
            'output_tokens': log['output_tokens'],
            'latency_ms': log['latency_ms'],
            'status': log['status']
        }

        if include_prompts:
            entry['prompt_preview'] = log['prompt'][:200] + '...'
            entry['response_preview'] = log['response'][:200] + '...'

        formatted_logs.append(entry)

    return formatted_logs

# Get recent failures for debugging
recent_errors = query_audit_logs(
    connection_name='anthropic-claude',
    start_date=datetime.now() - timedelta(hours=24)
)

error_logs = [log for log in recent_errors if log['status'] != 'success']
print(f"Errors in last 24h: {len(error_logs)}")
```

</div>

### Rate Limiting Implementation


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
from dataiku.llm import LLM
import time

class RateLimitedLLM:
    """
    Wrapper for LLM with client-side rate limiting.
    Useful for additional control beyond connection limits.
    """

    def __init__(
        self,
        connection_name: str,
        max_requests_per_minute: int = 10,
        max_tokens_per_minute: int = 50_000
    ):
        self.llm = LLM(connection_name)
        self.max_rpm = max_requests_per_minute
        self.max_tpm = max_tokens_per_minute

        # Tracking
        self.request_times = []
        self.token_usage = []

    def _check_rate_limit(self, estimated_tokens: int):
        """Check if request would exceed rate limits."""
        now = time.time()

        # Remove entries older than 1 minute
        cutoff = now - 60
        self.request_times = [t for t in self.request_times if t > cutoff]
        self.token_usage = [
            (t, tokens) for t, tokens in self.token_usage if t > cutoff
        ]

        # Check limits
        if len(self.request_times) >= self.max_rpm:
            raise RuntimeError(
                f"Rate limit: {self.max_rpm} requests/minute exceeded"
            )

        total_tokens = sum(tokens for _, tokens in self.token_usage)
        if total_tokens + estimated_tokens > self.max_tpm:
            raise RuntimeError(
                f"Rate limit: {self.max_tpm} tokens/minute exceeded"
            )

    def complete(
        self,
        prompt: str,
        estimated_tokens: int = 1000,
        **kwargs
    ) -> str:
        """
        Generate completion with rate limiting.

        Args:
            prompt: The prompt text
            estimated_tokens: Estimated total tokens for rate limit check
            **kwargs: Additional arguments for LLM.complete()

        Returns:
            Completion text
        """
        # Check rate limits before making request
        self._check_rate_limit(estimated_tokens)

        # Make request
        now = time.time()
        response = self.llm.complete(prompt, **kwargs)

        # Track usage
        self.request_times.append(now)
        actual_tokens = response.usage.total_tokens
        self.token_usage.append((now, actual_tokens))

        return response.text

# Usage
rate_limited_llm = RateLimitedLLM(
    connection_name='anthropic-claude',
    max_requests_per_minute=10,
    max_tokens_per_minute=50_000
)

# This will respect rate limits
for prompt in batch_prompts:
    try:
        result = rate_limited_llm.complete(prompt, estimated_tokens=2000)
    except RuntimeError as e:
        print(f"Rate limit hit: {e}")
        time.sleep(60)  # Wait before retrying
```


## Common Pitfalls

**Pitfall 1: Not Setting Up Alerts**
- Setting quotas without alerts means you discover overages after the fact
- Always configure alerts at 75-80% of quota threshold
- Include multiple recipients (admins and project leads)

**Pitfall 2: Overly Restrictive Permissions**
- Blocking all users by default can stifle innovation
- Start permissive in pilot phase, then tighten based on actual usage patterns
- Use rate limits instead of hard blocks for experimentation

**Pitfall 3: Ignoring Audit Logs**
- Audit logs are invaluable for debugging and optimization
- Regularly review logs to identify inefficient prompts or error patterns
- Use log analysis to educate users on best practices

**Pitfall 4: Not Accounting for Cost Variability**
- Different models have vastly different costs per token
- Claude Opus costs 5x more than Haiku
- Set quotas based on expected model mix, not just request count

**Pitfall 5: Forgetting About Compliance**
- Some industries require specific retention policies for LLM interactions
- Configure audit log retention to meet compliance requirements
- Consider data residency requirements when choosing providers

## Connections

<div class="callout-info">

<strong>How this connects to the rest of the course:</strong>


**Builds on:**
- LLM Mesh architecture and setup (Module 0.1)
- Provider connections and configuration (Module 0.2)

**Leads to:**
- Production deployment patterns (Module 4)
- Cost optimization strategies (Module 4)
- Enterprise scaling considerations

**Related to:**
- Dataiku security groups and permissions
- Project-level access control
- Cost accounting and chargeback

## Practice Problems

1. **Basic Governance Setup**
   - Configure three LLM connections with different access levels: dev (all users, high limit), staging (specific groups, medium limit), prod (admins only, low limit for testing)
   - Set up email alerts at 80% quota usage
   - Generate a usage report showing cost by user

2. **Cost Analysis Challenge**
   - Given 30 days of audit logs, identify the top 5 most expensive prompts
   - Calculate what switching Claude Opus to Claude Sonnet would save
   - Design a quota system that allocates budget proportionally to team size

3. **Rate Limiting Design**
   - Design rate limiting rules for a chatbot application expected to receive 1000 requests/hour during business hours
   - Balance user experience (no errors) with cost control
   - Account for burst traffic and graceful degradation

4. **Compliance Implementation**
   - Your company requires 90-day audit log retention and prohibition on sending PII to external LLMs
   - Design a governance strategy using LLM Mesh features
   - What additional tooling might you need?

5. **Multi-Project Governance**
   - You manage 10 projects with a total LLM budget of $5000/month
   - 3 projects are production (need reliability), 7 are experimental (need flexibility)
   - Design a quota allocation strategy with automated rebalancing

## Further Reading

- **Dataiku Documentation**: [LLM Mesh Governance Features](https://doc.dataiku.com/dss/latest/generative-ai/llm-mesh-governance.html) - Official governance configuration guide

- **OpenAI Enterprise Guidance**: [Managing GPT-4 Access in Organizations](https://platform.openai.com/docs/guides/safety-best-practices) - Best practices applicable to any LLM provider

- **Cloud FinOps Foundation**: [Unit Economics for AI/ML](https://www.finops.org/wg/ai-ml/) - Framework for cost management in AI workloads

- **NIST AI Risk Management Framework**: [AI RMF 1.0](https://www.nist.gov/itl/ai-risk-management-framework) - Comprehensive governance framework for AI systems

- **Research Paper**: "Measuring and Managing LLM Costs at Scale" - Practical patterns from companies running production LLM applications (fictional but representative of real-world challenges)


## Resources

<a class="link-card" href="../notebooks/01_first_connection.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>
