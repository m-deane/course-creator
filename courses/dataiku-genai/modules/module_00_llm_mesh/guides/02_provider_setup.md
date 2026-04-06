# Provider Setup and Configuration

> **Reading time:** ~9 min | **Module:** 0 — Llm Mesh | **Prerequisites:** Basic Python, familiarity with LLM concepts

## In Brief

Provider setup in Dataiku LLM Mesh involves configuring connections to external LLM APIs (Anthropic, OpenAI, Azure, etc.) with proper authentication, model selection, and operational parameters. Each provider connection becomes a managed resource that can be used across all projects with centralized governance.

<div class="callout-insight">

<strong>Key Insight:</strong> **The power of provider abstraction:** Configure once, use everywhere. A well-configured provider connection eliminates the need for developers to manage API keys, handle rate limits, or implement retry logic—all of this becomes infrastructure managed by the platform.

</div>

<div class="callout-key">

<strong>Key Concept:</strong> Provider setup in Dataiku LLM Mesh involves configuring connections to external LLM APIs (Anthropic, OpenAI, Azure, etc.) with proper authentication, model selection, and operational parameters. Each provider connection becomes a managed resource that can be used across all projects with centrali...

</div>

## Formal Definition

A **Provider Connection** in LLM Mesh is a configured endpoint that encapsulates:

1. **Authentication**: API keys, OAuth tokens, or managed identities
2. **Endpoint Configuration**: Base URLs, API versions, deployment names
3. **Model Selection**: Available models and default selections
4. **Operational Limits**: Rate limits, timeouts, retry policies
5. **Cost Parameters**: Pricing per token for budget tracking
6. **Access Control**: Which users/groups/projects can use the connection

## Intuitive Explanation

Think of provider setup like configuring a company's phone system. Instead of each employee managing their own phone contracts with different carriers, the IT department:

- Negotiates contracts with carriers (configures provider APIs)
- Sets up extensions (creates connections)
- Defines who can make international calls (access control)
- Monitors usage and bills by department (cost tracking)
- Automatically routes calls if one carrier is down (failover)

Similarly, LLM Mesh provider setup centralizes all LLM infrastructure management so data scientists can focus on building applications rather than managing API credentials.

### Configuration Levels

```

┌─────────────────────────────────────────────────────────┐
│          Administrator (Platform Level)                  │
│  • Configure provider connections                        │
│  • Store API keys securely                               │
│  • Set global rate limits                                │
│  • Define cost parameters                                │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Connection  │ │  Connection  │ │  Connection  │
│   Claude-1   │ │   GPT-4-1    │ │  Azure-OAI   │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
  ┌───────────┐  ┌───────────┐  ┌───────────┐
  │ Project A │  │ Project B │  │ Project C │
  │ (granted) │  │ (granted) │  │ (granted) │
  └───────────┘  └───────────┘  └───────────┘
```

## Code Implementation

### Anthropic Claude Setup


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Administrative setup (done via UI or API)

# This is the conceptual structure - actual setup is done in Dataiku UI

connection_config = {
    "name": "claude-production",
    "type": "anthropic",
    "description": "Production Claude Sonnet for commodity analysis",

    # Authentication
    "auth": {
        "api_key": "${secrets.ANTHROPIC_API_KEY}",  # Stored securely
        "api_base": "https://api.anthropic.com"     # Optional override
    },

    # Model configuration
    "models": {
        "default": "claude-sonnet-4-20250514",
        "available": [
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-haiku-20240307"
        ]
    },

    # Operational parameters
    "limits": {
        "max_requests_per_minute": 60,
        "max_tokens_per_minute": 100000,
        "max_concurrent_requests": 10,
        "timeout_seconds": 120,
        "max_retries": 3,
        "retry_delay_seconds": 2
    },

    # Cost tracking
    "pricing": {
        "claude-sonnet-4-20250514": {
            "input_per_million": 3.00,
            "output_per_million": 15.00,
            "currency": "USD"
        },
        "claude-3-haiku-20240307": {
            "input_per_million": 0.25,
            "output_per_million": 1.25,
            "currency": "USD"
        }
    },

    # Access control
    "access": {
        "allowed_groups": ["data-scientists", "ml-engineers"],
        "allowed_projects": ["*"],  # All projects
        "require_approval": False
    }
}
```

</div>
</div>

### OpenAI Setup


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
connection_config = {
    "name": "openai-gpt4",
    "type": "openai",
    "description": "OpenAI GPT-4 for general purpose use",

    "auth": {
        "api_key": "${secrets.OPENAI_API_KEY}",
        "organization_id": "org-xxxxxxxx",  # Optional
        "api_base": "https://api.openai.com/v1"
    },

    "models": {
        "default": "gpt-4o",
        "available": [
            "gpt-4o",
            "gpt-4-turbo-preview",
            "gpt-3.5-turbo"
        ]
    },

    "limits": {
        "max_requests_per_minute": 100,
        "max_tokens_per_minute": 150000,
        "timeout_seconds": 60
    },

    "pricing": {
        "gpt-4o": {
            "input_per_million": 2.50,
            "output_per_million": 10.00
        },
        "gpt-3.5-turbo": {
            "input_per_million": 0.50,
            "output_per_million": 1.50
        }
    }
}
```

</div>
</div>

### Azure OpenAI Setup


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
connection_config = {
    "name": "azure-openai-prod",
    "type": "azure_openai",
    "description": "Azure OpenAI for enterprise workloads",

    "auth": {
        "api_key": "${secrets.AZURE_OPENAI_KEY}",
        # OR use managed identity
        "use_managed_identity": True,
        "tenant_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
    },

    "endpoint": {
        "resource_name": "your-resource-name",
        "deployment_name": "gpt-4-deployment",
        "api_base": "https://your-resource.openai.azure.com/",
        "api_version": "2024-02-15-preview"
    },

    "models": {
        "default": "gpt-4",
        # Note: In Azure, model = deployment name
        "deployment_mappings": {
            "gpt-4-deployment": "gpt-4-turbo-preview",
            "gpt-35-deployment": "gpt-3.5-turbo"
        }
    },

    "limits": {
        "max_requests_per_minute": 120,  # Per deployment
        "max_tokens_per_minute": 200000
    },

    # Azure-specific features
    "azure": {
        "region": "eastus",
        "private_endpoint": False,
        "vnet_integration": False
    }
}
```

</div>
</div>

### Using Configured Connections


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# In a Dataiku Python recipe or notebook
import dataiku
from dataiku.llm import LLM

# Simple usage - connection name only
llm = LLM("claude-production")

response = llm.complete(
    prompt="Analyze this EIA report: ...",
    max_tokens=500,
    temperature=0.7
)

print(response.text)

# Advanced usage - override model, pass parameters
llm_fast = LLM("claude-production")
response = llm_fast.complete(
    prompt="Quick sentiment analysis: ...",
    model="claude-3-haiku-20240307",  # Override default
    max_tokens=100,
    temperature=0.3
)

# Multi-connection fallback
def generate_with_fallback(prompt, **kwargs):
    connections = ["claude-production", "openai-gpt4", "azure-openai-prod"]

    for conn_name in connections:
        try:
            llm = LLM(conn_name)
            return llm.complete(prompt, **kwargs)
        except Exception as e:
            print(f"{conn_name} failed: {e}")
            continue

    raise Exception("All connections failed")

result = generate_with_fallback("Summarize this report: ...")
```

</div>
</div>

## Common Pitfalls

### 1. Hardcoding API Keys

**Problem:**

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# NEVER DO THIS
import anthropic
client = anthropic.Anthropic(api_key="sk-ant-api03-...")  # Exposed!
```

</div>
</div>

**Solution:**

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Use Dataiku connection - API key managed securely
from dataiku.llm import LLM
llm = LLM("claude-production")  # API key in secure storage
```

</div>
</div>

### 2. Incorrect Azure Deployment Names

**Problem:**

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Azure requires deployment name, not model name
llm = LLM("azure-openai-prod")
response = llm.complete(
    prompt="...",
    model="gpt-4-turbo-preview"  # ✗ Wrong! This is a model name
)
```

</div>
</div>

**Solution:**

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Use deployment name configured in Azure
llm = LLM("azure-openai-prod")
response = llm.complete(
    prompt="...",
    model="gpt-4-deployment"  # ✓ Correct deployment name
)
```

</div>
</div>

### 3. Ignoring Provider-Specific Limits

**Problem:**

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Different providers have different rate limits
for i in range(1000):
    llm.complete("...")  # May hit limits quickly
```

</div>
</div>

**Solution:**

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import time
from dataiku.llm import LLM, RateLimitError

llm = LLM("claude-production")

for i in range(1000):
    try:
        response = llm.complete("...")
        process_response(response)
    except RateLimitError as e:
        # Respect rate limits
        wait_time = e.retry_after or 60
        print(f"Rate limited, waiting {wait_time}s")
        time.sleep(wait_time)
```

</div>
</div>

### 4. Not Testing Connections

**Problem:**

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Deploying without testing
llm = LLM("new-connection")

# Hope it works in production!
```

</div>

**Solution:**

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
def test_connection(connection_name):
    """Test LLM connection before deployment."""
    try:
        llm = LLM(connection_name)

        # Simple test
        response = llm.complete(
            prompt="Say 'connection successful' if you receive this.",
            max_tokens=10
        )

        assert "successful" in response.text.lower()

        return {
            "status": "healthy",
            "latency_ms": response.latency * 1000,
            "connection": connection_name
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "connection": connection_name
        }

# Test all connections
for conn in ["claude-production", "openai-gpt4", "azure-openai-prod"]:
    result = test_connection(conn)
    print(f"{conn}: {result['status']}")
```


### 5. Mixed Provider Assumptions

**Problem:**

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python

# Anthropic-specific parameter used with OpenAI
llm = LLM("openai-gpt4")
response = llm.complete(
    prompt="...",
    metadata={"user_id": "user123"}  # Anthropic parameter, not OpenAI
)
```


**Solution:**

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python

# Check provider before using provider-specific features
def get_provider_type(connection_name):
    client = dataiku.api_client()
    conn = client.get_connection(connection_name)
    return conn.type

provider = get_provider_type("openai-gpt4")

if provider == "anthropic":
    response = llm.complete(prompt, metadata={"user_id": "user123"})
elif provider == "openai":
    response = llm.complete(prompt, user="user123")  # OpenAI format
```


## Connections to Other Topics

### Prerequisites
- **Dataiku Administration**: Understanding connections and security
- **API Authentication**: API keys, OAuth, managed identities
- **Cloud Services**: AWS, Azure, GCP basics

### Enables
- **Prompt Studios**: Require configured LLM connections
- **Knowledge Banks**: Use LLM connections for embeddings
- **Python Recipes**: Programmatic LLM access
- **API Deployment**: Expose LLM capabilities

### Related Concepts
- **Secrets Management**: Secure credential storage
- **Service Mesh**: Similar abstraction pattern for microservices
- **Connection Pooling**: Reusing connections for efficiency
- **Health Checks**: Monitoring provider availability

## Practice Problems

### Problem 1: Multi-Provider Setup

**Question:** You need to set up three connections: Anthropic Claude (primary, cost $3/$15 per M tokens), OpenAI GPT-4 (secondary, $2.50/$10), and GPT-3.5 (budget option, $0.50/$1.50). Design the connection configurations with appropriate rate limits and access controls. Data scientists should use Claude, analysts should use GPT-3.5.

**Difficulty:** Medium

<details>
<summary>Solution</summary>


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python

# Connection 1: Claude for Data Scientists
claude_config = {
    "name": "claude-ds",
    "type": "anthropic",
    "auth": {"api_key": "${secrets.ANTHROPIC_API_KEY}"},
    "models": {
        "default": "claude-sonnet-4-20250514"
    },
    "limits": {
        "max_requests_per_minute": 60,
        "max_tokens_per_minute": 100000
    },
    "pricing": {
        "claude-sonnet-4-20250514": {
            "input_per_million": 3.00,
            "output_per_million": 15.00
        }
    },
    "access": {
        "allowed_groups": ["data-scientists", "ml-engineers"],
        "allowed_projects": ["*"]
    }
}

# Connection 2: GPT-4 Backup
gpt4_config = {
    "name": "gpt4-backup",
    "type": "openai",
    "auth": {"api_key": "${secrets.OPENAI_API_KEY}"},
    "models": {"default": "gpt-4o"},
    "limits": {
        "max_requests_per_minute": 80,
        "max_tokens_per_minute": 120000
    },
    "pricing": {
        "gpt-4o": {
            "input_per_million": 2.50,
            "output_per_million": 10.00
        }
    },
    "access": {
        "allowed_groups": ["data-scientists"],
        "allowed_projects": ["*"]
    }
}

# Connection 3: GPT-3.5 for Analysts
gpt35_config = {
    "name": "gpt35-analysts",
    "type": "openai",
    "auth": {"api_key": "${secrets.OPENAI_API_KEY}"},
    "models": {"default": "gpt-3.5-turbo"},
    "limits": {
        "max_requests_per_minute": 100,
        "max_tokens_per_minute": 200000
    },
    "pricing": {
        "gpt-3.5-turbo": {
            "input_per_million": 0.50,
            "output_per_million": 1.50
        }
    },
    "access": {
        "allowed_groups": ["analysts", "business-users"],
        "allowed_projects": ["REPORTING", "DASHBOARDS"]
    }
}

# Usage example with automatic selection by group
import dataiku
from dataiku.llm import LLM

def get_appropriate_connection():
    """Select connection based on user's group."""
    client = dataiku.api_client()
    user_info = client.get_auth_info()

    user_groups = user_info.get('groups', [])

    if 'data-scientists' in user_groups or 'ml-engineers' in user_groups:
        return "claude-ds"
    elif 'analysts' in user_groups or 'business-users' in user_groups:
        return "gpt35-analysts"
    else:
        raise PermissionError("User not in authorized groups")

# Use appropriate connection
connection_name = get_appropriate_connection()
llm = LLM(connection_name)
```

</details>

### Problem 2: Azure-Specific Configuration

**Question:** Configure Azure OpenAI with three deployments in different regions for geo-redundancy: East US (primary), West Europe (failover), and Southeast Asia (APAC). Implement automatic regional selection based on user location.

**Difficulty:** Hard

<details>
<summary>Solution</summary>


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python

# Three regional connections
azure_configs = [
    {
        "name": "azure-eastus",
        "type": "azure_openai",
        "auth": {"api_key": "${secrets.AZURE_OPENAI_KEY_EASTUS}"},
        "endpoint": {
            "resource_name": "openai-eastus",
            "deployment_name": "gpt-4-eastus",
            "api_base": "https://openai-eastus.openai.azure.com/",
            "api_version": "2024-02-15-preview"
        },
        "azure": {"region": "eastus"},
        "priority": 1,  # Primary
        "geo": ["US", "CA", "MX"]
    },
    {
        "name": "azure-westeurope",
        "type": "azure_openai",
        "auth": {"api_key": "${secrets.AZURE_OPENAI_KEY_WESTEUROPE}"},
        "endpoint": {
            "resource_name": "openai-westeurope",
            "deployment_name": "gpt-4-westeurope",
            "api_base": "https://openai-westeurope.openai.azure.com/",
            "api_version": "2024-02-15-preview"
        },
        "azure": {"region": "westeurope"},
        "priority": 2,
        "geo": ["GB", "DE", "FR", "IT", "ES", "NL", "BE"]
    },
    {
        "name": "azure-southeastasia",
        "type": "azure_openai",
        "auth": {"api_key": "${secrets.AZURE_OPENAI_KEY_SOUTHEASTASIA}"},
        "endpoint": {
            "resource_name": "openai-southeastasia",
            "deployment_name": "gpt-4-southeastasia",
            "api_base": "https://openai-southeastasia.openai.azure.com/",
            "api_version": "2024-02-15-preview"
        },
        "azure": {"region": "southeastasia"},
        "priority": 3,
        "geo": ["SG", "MY", "TH", "VN", "ID", "PH"]
    }
]

# Geo-aware router
import dataiku
from dataiku.llm import LLM
import requests

class GeoAwareLLMRouter:
    """Route to nearest Azure region with failover."""

    REGION_MAP = {
        "eastus": ["US", "CA", "MX", "BR"],
        "westeurope": ["GB", "DE", "FR", "IT", "ES", "NL", "BE", "SE", "NO"],
        "southeastasia": ["SG", "MY", "TH", "VN", "ID", "PH", "AU", "NZ", "JP", "KR"]
    }

    def __init__(self):
        self.connections = {
            "eastus": LLM("azure-eastus"),
            "westeurope": LLM("azure-westeurope"),
            "southeastasia": LLM("azure-southeastasia")
        }

    def get_user_country(self):
        """Detect user's country from IP (simplified)."""
        # In production, use request headers or user profile
        try:
            response = requests.get("https://ipapi.co/json/", timeout=2)
            return response.json().get('country_code', 'US')
        except:
            return 'US'  # Default

    def select_region(self, country_code):
        """Select best region for country."""
        for region, countries in self.REGION_MAP.items():
            if country_code in countries:
                return region
        return "eastus"  # Default

    def generate_with_failover(self, prompt, **kwargs):
        """Generate with regional selection and failover."""
        # Determine primary region
        user_country = self.get_user_country()
        primary_region = self.select_region(user_country)

        # Attempt primary
        try:
            llm = self.connections[primary_region]
            return llm.complete(prompt, **kwargs)
        except Exception as e:
            print(f"Primary region {primary_region} failed: {e}")

        # Failover to other regions
        for region, llm in self.connections.items():
            if region == primary_region:
                continue  # Already tried

            try:
                print(f"Failing over to {region}")
                return llm.complete(prompt, **kwargs)
            except Exception as e:
                print(f"Region {region} failed: {e}")

        raise Exception("All regions failed")

# Usage
router = GeoAwareLLMRouter()
response = router.generate_with_failover(
    "Analyze this commodity report: ..."
)
```

</details>

## Further Reading

### Official Documentation
- [Dataiku LLM Connections](https://doc.dataiku.com/dss/latest/llm-mesh/connections.html)
- [Anthropic API Documentation](https://docs.anthropic.com/en/api/getting-started)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/)

### Best Practices
- **Secrets Management**: HashiCorp Vault, AWS Secrets Manager
- **Rate Limiting**: Token bucket algorithm, sliding window
- **Failover Strategies**: Circuit breaker pattern, exponential backoff

### Next Steps
- Guide 03: Governance (access control and compliance)
- Module 1: Prompt Design (using configured connections)
- Module 4: Deployment (monitoring connection health)


## Resources

<a class="link-card" href="../notebooks/01_first_connection.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>
