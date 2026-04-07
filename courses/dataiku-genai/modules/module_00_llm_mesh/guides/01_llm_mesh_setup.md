# LLM Mesh Setup and Configuration

> **Reading time:** ~5 min | **Module:** 0 — Llm Mesh | **Prerequisites:** Basic Python, familiarity with LLM concepts

## What is LLM Mesh?

Dataiku LLM Mesh provides a unified interface to multiple LLM providers:

```

┌─────────────────────────────────────────────────────────────┐
│                    Your Dataiku Project                      │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      LLM Mesh Layer                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  • Unified API  • Cost tracking  • Access control    │   │
│  │  • Rate limiting • Logging • Model routing           │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Anthropic  │    │   OpenAI    │    │ Azure/GCP/  │
│   Claude    │    │   GPT-4     │    │   Bedrock   │
└─────────────┘    └─────────────┘    └─────────────┘
```

## Benefits of LLM Mesh

| Benefit | Description |
|---------|-------------|
| **Provider Abstraction** | Switch models without code changes |
| **Centralized Governance** | Single point for access control |
| **Cost Management** | Track and allocate costs by project |
| **Compliance** | Audit logs for all LLM interactions |
| **Reliability** | Failover between providers |

## Configuring LLM Connections

### Anthropic Claude Setup

1. Navigate to **Administration > Connections**
2. Click **+ New Connection** > **LLM**
3. Select **Anthropic Claude**

```yaml

# Connection settings
connection_name: anthropic-claude
provider: anthropic
api_key: ${ANTHROPIC_API_KEY}  # From secrets
default_model: claude-sonnet-4-20250514

# Rate limiting
max_requests_per_minute: 60
max_tokens_per_minute: 100000

# Timeout settings
timeout_seconds: 120
```

### OpenAI Setup

```yaml
connection_name: openai-gpt4
provider: openai
api_key: ${OPENAI_API_KEY}
default_model: gpt-4o
organization_id: org-xxx  # Optional

# Rate limiting
max_requests_per_minute: 100
max_tokens_per_minute: 150000
```

### Azure OpenAI Setup

```yaml
connection_name: azure-openai
provider: azure_openai
api_key: ${AZURE_OPENAI_KEY}
endpoint: https://your-resource.openai.azure.com/
deployment_name: gpt-4-deployment
api_version: 2024-02-15-preview

# Azure-specific
resource_name: your-resource-name
```

## Using LLM Mesh in Python

### Basic Usage


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import dataiku
from dataiku.llm import LLM

# Get the LLM handle
llm = LLM("anthropic-claude")  # Connection name

# Simple completion
response = llm.complete(
    prompt="Summarize the key factors affecting oil prices.",
    max_tokens=500
)
print(response.text)
```

</div>
</div>

### Chat Interface


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
# Pseudocode — ChatSession is a conceptual pattern, not a real Dataiku import.
# Verify the multi-turn conversation API against your Dataiku version's docs.
# The real API uses project.get_llm() or dataiku.api_client() methods.

import dataiku

client = dataiku.api_client()
project = client.get_default_project()
llm = project.get_llm("anthropic-claude")

# Multi-turn conversation (conceptual pattern)
messages = [
    {"role": "system", "content": "You are a commodity market analyst specializing in energy markets."},
    {"role": "user", "content": "What drove oil prices this week?"}
]

completion = llm.new_completion()
for msg in messages:
    if msg["role"] == "system":
        completion.with_message(msg["content"], role="system")
    else:
        completion.with_message(msg["content"], role="user")

response = completion.execute()
print(response.text)
```

</div>
</div>

### Structured Output


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
from dataiku.llm import LLM
import json

llm = LLM("anthropic-claude")

prompt = """Extract the following from this EIA report:
- Crude inventory change (million barrels)
- Gasoline inventory change
- Distillate inventory change

Report: U.S. commercial crude oil inventories decreased by 5.2 million barrels...

Return as JSON only."""

response = llm.complete(
    prompt=prompt,
    max_tokens=200,
    temperature=0  # Deterministic for extraction
)

data = json.loads(response.text)
print(data)
```

</div>
</div>

## Connection Management

### Listing Available Connections


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import dataiku

# List all LLM connections
client = dataiku.api_client()
connections = client.list_connections()

llm_connections = [
    c for c in connections
    if c.get('type') == 'LLM'
]

for conn in llm_connections:
    print(f"Name: {conn['name']}, Provider: {conn.get('params', {}).get('provider')}")
```

</div>
</div>

### Connection Testing


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def test_llm_connection(connection_name: str) -> dict:
    """Test an LLM connection."""
    try:
        llm = LLM(connection_name)
        response = llm.complete(
            prompt="Say 'connection successful'",
            max_tokens=10
        )
        return {
            'status': 'success',
            'connection': connection_name,
            'response': response.text
        }
    except Exception as e:
        return {
            'status': 'error',
            'connection': connection_name,
            'error': str(e)
        }

# Test all connections
for conn_name in ['anthropic-claude', 'openai-gpt4']:
    result = test_llm_connection(conn_name)
    print(f"{conn_name}: {result['status']}")
```

</div>
</div>

## Model Routing

### Automatic Failover


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
from dataiku.llm import LLM

class LLMRouter:
    """Route requests with failover."""

    def __init__(self, primary: str, fallback: str):
        self.primary = LLM(primary)
        self.fallback = LLM(fallback)

    def complete(self, prompt: str, **kwargs) -> str:
        """Try primary, fallback on error."""
        try:
            response = self.primary.complete(prompt, **kwargs)
            return response.text
        except Exception as e:
            print(f"Primary failed: {e}, trying fallback")
            response = self.fallback.complete(prompt, **kwargs)
            return response.text

# Usage
router = LLMRouter("anthropic-claude", "openai-gpt4")
result = router.complete("Summarize recent OPEC decisions")
```

</div>

### Cost-Based Routing


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
MODEL_COSTS = {
    'claude-sonnet-4-20250514': {'input': 3.0, 'output': 15.0},  # per 1M tokens
    'gpt-4o': {'input': 2.5, 'output': 10.0},
    'claude-3-haiku': {'input': 0.25, 'output': 1.25}
}

def select_model_by_budget(
    estimated_tokens: int,
    max_cost: float,
    prefer_quality: bool = True
) -> str:
    """Select model based on budget constraints."""

    # Calculate costs
    viable_models = []
    for model, costs in MODEL_COSTS.items():
        estimated_cost = (
            estimated_tokens * costs['input'] / 1_000_000 +
            estimated_tokens * costs['output'] / 1_000_000
        )
        if estimated_cost <= max_cost:
            viable_models.append((model, estimated_cost))

    if not viable_models:
        raise ValueError("No model within budget")

    # Sort by cost (ascending) or quality (descending cost = better quality)
    viable_models.sort(key=lambda x: x[1], reverse=prefer_quality)

    return viable_models[0][0]
```


## Access Control

### Project-Level Permissions

In Dataiku, configure access at the project level:

1. **Project Settings** > **Security**
2. Add groups with LLM access
3. Configure per-connection permissions

```yaml

# Project security settings
llm_permissions:
  - connection: anthropic-claude
    groups: [data-scientists, analysts]
    daily_token_limit: 100000

  - connection: openai-gpt4
    groups: [senior-analysts]
    daily_token_limit: 50000
```

### Usage Quotas


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
# Pseudocode — UsageTracker is not a real Dataiku import.
# Token usage is tracked automatically by the LLM Mesh and visible
# in the Dataiku admin console. For programmatic tracking, build
# a custom wrapper around LLM calls.

import dataiku

client = dataiku.api_client()
project = client.get_default_project()
llm = project.get_llm("anthropic-claude")

completion = llm.new_completion()
completion.with_message("Summarize the key factors affecting oil prices.")
response = completion.execute()

# Token usage is available in the response metadata
print(f"Response: {response.text}")
# Check the Dataiku admin console for detailed cost/usage tracking
```


## Key Takeaways

1. **LLM Mesh centralizes** all LLM interactions through a single governed layer

2. **Multiple providers** can be configured and swapped without code changes

3. **Cost tracking** is built-in at the connection and project level

4. **Access control** enables fine-grained permissions by group

5. **Failover routing** improves reliability across providers

<div class="callout-key">

<strong>Key Concept:</strong> 5. **Failover routing** improves reliability across providers





## Resources

<a class="link-card" href="../notebooks/01_first_connection.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>
