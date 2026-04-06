# Configuring LLM Connections in Dataiku

> **Reading time:** ~5 min | **Module:** 0 — Llm Mesh | **Prerequisites:** Basic Python, familiarity with LLM concepts

## Overview

Dataiku's LLM Mesh provides a unified interface to connect various LLM providers. This guide covers setting up and managing these connections.

## Supported LLM Providers

### Cloud Providers

| Provider | Models | Features |
|----------|--------|----------|
| OpenAI | GPT-4, GPT-3.5-turbo | Chat, embeddings, function calling |
| Azure OpenAI | GPT-4, GPT-3.5-turbo | Enterprise security, regional deployment |
| Anthropic | Claude 3, Claude 2 | Long context, constitutional AI |
| Google | Gemini, PaLM | Multimodal, code generation |
| AWS Bedrock | Claude, Titan, Llama | AWS integration, fine-tuning |
| Cohere | Command, Embed | RAG-optimized, embeddings |

### Self-Hosted

- Ollama integration for local models
- Custom API endpoints
- HuggingFace models

## Connection Setup Workflow

### Step 1: Navigate to LLM Mesh

```

Administration → Connections → LLM Mesh
```

### Step 2: Create New Connection


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Conceptual API structure (actual UI-based)
connection_config = {
    "name": "openai-production",
    "provider": "openai",
    "api_key": "sk-...",  # Stored securely
    "organization_id": "org-...",
    "default_model": "gpt-4-turbo-preview",
    "settings": {
        "timeout_seconds": 60,
        "max_retries": 3,
        "rate_limit_rpm": 3500
    }
}
```

</div>
</div>

### Step 3: Configure Model Parameters


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
model_settings = {
    "model_id": "gpt-4-turbo-preview",
    "temperature": 0.7,
    "max_tokens": 4096,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "stop_sequences": []
}
```

</div>
</div>

## Provider-Specific Configuration

### OpenAI


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# OpenAI connection configuration
openai_config = {
    "type": "openai",
    "params": {
        "api_key": "${secrets.OPENAI_API_KEY}",
        "organization": "org-xxx",
        "api_base": "https://api.openai.com/v1"  # Custom endpoint optional
    },
    "models": {
        "gpt-4-turbo-preview": {
            "context_window": 128000,
            "max_output": 4096,
            "supports_functions": True,
            "supports_vision": True
        },
        "gpt-3.5-turbo": {
            "context_window": 16385,
            "max_output": 4096,
            "supports_functions": True,
            "supports_vision": False
        }
    }
}
```

</div>
</div>

### Azure OpenAI


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Azure OpenAI requires deployment-specific configuration
azure_config = {
    "type": "azure_openai",
    "params": {
        "api_key": "${secrets.AZURE_OPENAI_KEY}",
        "api_base": "https://your-resource.openai.azure.com/",
        "api_version": "2024-02-15-preview",
        "deployment_name": "gpt4-deployment"
    },
    "security": {
        "managed_identity": True,
        "tenant_id": "xxx-xxx-xxx"
    }
}
```

</div>
</div>

### AWS Bedrock


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# AWS Bedrock uses IAM authentication
bedrock_config = {
    "type": "aws_bedrock",
    "params": {
        "region": "us-east-1",
        "access_key_id": "${secrets.AWS_ACCESS_KEY}",
        "secret_access_key": "${secrets.AWS_SECRET_KEY}",
        # Or use IAM role
        "role_arn": "arn:aws:iam::123456789:role/bedrock-access"
    },
    "models": {
        "anthropic.claude-3-sonnet": {
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
        },
        "amazon.titan-embed": {
            "model_id": "amazon.titan-embed-text-v1"
        }
    }
}
```

</div>
</div>

## Connection Testing

### Health Check


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# In Dataiku Python recipe or notebook
import dataiku

def test_llm_connection(connection_name):
    """Test LLM connection health."""
    client = dataiku.api_client()
    llm_connection = client.get_llm_connection(connection_name)

    try:
        # Simple completion test
        response = llm_connection.generate(
            prompt="Hello, respond with 'Connection successful'",
            max_tokens=50
        )

        if "successful" in response.text.lower():
            return {"status": "healthy", "response_time_ms": response.timing}
        else:
            return {"status": "unexpected_response", "response": response.text}

    except Exception as e:
        return {"status": "error", "error": str(e)}

# Run test
result = test_llm_connection("openai-production")
print(f"Connection status: {result['status']}")
```

</div>
</div>

### Latency Benchmarking


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import time

def benchmark_connection(connection_name, n_requests=10):
    """Benchmark LLM connection performance."""
    client = dataiku.api_client()
    llm = client.get_llm_connection(connection_name)

    latencies = []
    errors = 0

    for i in range(n_requests):
        start = time.time()
        try:
            response = llm.generate(
                prompt=f"Count from 1 to 5. Request {i+1}.",
                max_tokens=50
            )
            latencies.append((time.time() - start) * 1000)
        except Exception as e:
            errors += 1

        # Rate limiting pause
        time.sleep(0.5)

    return {
        "mean_latency_ms": np.mean(latencies) if latencies else None,
        "p50_latency_ms": np.percentile(latencies, 50) if latencies else None,
        "p95_latency_ms": np.percentile(latencies, 95) if latencies else None,
        "error_rate": errors / n_requests
    }
```

</div>
</div>

## Connection Groups and Fallbacks

### Load Balancing


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Configure connection group for redundancy
connection_group = {
    "name": "llm-primary-group",
    "connections": [
        {
            "name": "openai-primary",
            "weight": 70,  # 70% of traffic
            "priority": 1
        },
        {
            "name": "azure-openai-backup",
            "weight": 30,  # 30% of traffic
            "priority": 2
        }
    ],
    "fallback_strategy": "priority_order",
    "health_check_interval_seconds": 60
}
```

</div>

### Automatic Failover


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
class LLMConnectionManager:
    """Manager for LLM connections with failover."""

    def __init__(self, primary_connection, fallback_connections):
        self.primary = primary_connection
        self.fallbacks = fallback_connections
        self.current = primary_connection

    def generate(self, prompt, **kwargs):
        """Generate with automatic failover."""
        connections = [self.primary] + self.fallbacks

        for conn in connections:
            try:
                response = conn.generate(prompt, **kwargs)
                self.current = conn
                return response
            except Exception as e:
                print(f"Connection {conn.name} failed: {e}")
                continue

        raise Exception("All LLM connections failed")

    def get_status(self):
        """Get status of all connections."""
        status = {}
        for conn in [self.primary] + self.fallbacks:
            try:
                conn.generate("test", max_tokens=5)
                status[conn.name] = "healthy"
            except:
                status[conn.name] = "unhealthy"
        return status
```


## Cost Tracking

### Token Usage Monitoring


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
class TokenUsageTracker:
    """Track token usage across LLM calls."""

    def __init__(self):
        self.usage_log = []

    def log_usage(self, connection_name, model, prompt_tokens,
                  completion_tokens, cost_per_1k_input, cost_per_1k_output):
        """Log a single usage event."""
        cost = (prompt_tokens / 1000 * cost_per_1k_input +
                completion_tokens / 1000 * cost_per_1k_output)

        self.usage_log.append({
            "timestamp": datetime.now().isoformat(),
            "connection": connection_name,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "estimated_cost_usd": cost
        })

    def get_summary(self, start_date=None, end_date=None):
        """Get usage summary."""
        import pandas as pd

        df = pd.DataFrame(self.usage_log)
        if df.empty:
            return {}

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        if start_date:
            df = df[df['timestamp'] >= start_date]
        if end_date:
            df = df[df['timestamp'] <= end_date]

        return {
            "total_tokens": df['total_tokens'].sum(),
            "total_cost_usd": df['estimated_cost_usd'].sum(),
            "by_model": df.groupby('model')['estimated_cost_usd'].sum().to_dict(),
            "by_connection": df.groupby('connection')['estimated_cost_usd'].sum().to_dict()
        }
```


## Security Best Practices

### API Key Management


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python

# NEVER hardcode API keys

# Use Dataiku's secrets management

# In project settings or global variables:

# ${secrets.OPENAI_API_KEY}

# Access in code:
import dataiku

project = dataiku.api_client().get_project(dataiku.default_project_key())
api_key = project.get_variable("OPENAI_API_KEY")  # From project variables

# Or use connection directly (key handled internally)
llm = dataiku.LLMConnection("openai-production")
```


### Access Control

```

Connection Security Settings:
├── Allowed Projects: [project1, project2]
├── Allowed Users: [user1, user2, group:data-science]
├── Rate Limits:
│   ├── Per User: 100 requests/minute
│   └── Per Project: 1000 requests/minute
└── Audit Logging: Enabled
```

## Key Takeaways

1. **Centralized management** through LLM Mesh simplifies multi-provider usage

2. **Secure credential storage** using Dataiku's secrets management

3. **Failover configuration** ensures reliability for production applications

4. **Cost tracking** is essential for budget management

5. **Connection groups** enable load balancing and redundancy

6. **Test connections** thoroughly before production deployment

<div class="callout-key">

<strong>Key Concept:</strong> 6. **Test connections** thoroughly before production deployment



## Resources

<a class="link-card" href="../notebooks/01_first_connection.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>
