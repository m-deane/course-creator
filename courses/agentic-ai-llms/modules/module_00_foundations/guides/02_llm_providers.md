# LLM Providers: Choosing the Right Model

> **Reading time:** ~10 min | **Module:** 0 — Foundations | **Prerequisites:** None

Different LLM providers offer distinct trade-offs in capability, cost, latency, and features. Understanding these differences helps you select the right model for your agent system and build provider-agnostic architectures.

<div class="callout-insight">

**Insight:** No single model is best for everything. Claude excels at reasoning and instruction-following, GPT-4 at broad knowledge, and open-source models at cost-efficiency and privacy. Production agents often use multiple models strategically.

</div>

---

## Provider Comparison

### Claude (Anthropic)

**Models:**
- Claude 3.5 Sonnet: Best balance of capability and cost
- Claude 3 Opus: Maximum capability, higher cost
- Claude 3 Haiku: Fast, cheap, good for simple tasks

**Strengths:**
- Superior instruction following
- Strong reasoning and analysis
- Excellent at structured output
- Large context window (200K tokens)
- Built-in tool use support

**Considerations:**
- Newer ecosystem than OpenAI
- Fewer fine-tuning options


<span class="filename">agent.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain quantum computing in one paragraph."}
    ]
)
print(response.content[0].text)
```

</div>
</div>

### GPT-4 (OpenAI)

**Models:**
- GPT-4 Turbo: Latest, 128K context
- GPT-4o: Optimized for speed
- GPT-3.5 Turbo: Fast and cheap

**Strengths:**
- Largest ecosystem and tooling
- Strong general knowledge
- Good code generation
- Extensive fine-tuning options
- Mature function calling

**Considerations:**
- Higher costs for GPT-4
- Rate limits can be restrictive

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "user", "content": "Explain quantum computing in one paragraph."}
    ]
)
print(response.choices[0].message.content)
```

### Open-Source Models

**Popular Options:**
- Llama 3 (Meta): Strong general capability
- Mistral/Mixtral: Excellent efficiency
- DeepSeek: Strong at code
- Qwen: Good multilingual support

**Strengths:**
- No API costs (if self-hosted)
- Full data privacy
- Customizable and fine-tunable
- No rate limits

**Considerations:**
- Requires infrastructure
- Lower capability than top proprietary models
- More operational complexity

```python
# Using Ollama for local inference
import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3:70b",
        "prompt": "Explain quantum computing in one paragraph.",
        "stream": False
    }
)
print(response.json()["response"])
```

---

## Model Selection Framework

### Decision Matrix

| Factor | Claude | GPT-4 | Open-Source |
|--------|--------|-------|-------------|
| Reasoning | ★★★★★ | ★★★★☆ | ★★★☆☆ |
| Speed | ★★★★☆ | ★★★☆☆ | ★★★★★ |
| Cost | ★★★☆☆ | ★★☆☆☆ | ★★★★★ |
| Privacy | ★★★☆☆ | ★★★☆☆ | ★★★★★ |
| Ecosystem | ★★★☆☆ | ★★★★★ | ★★★☆☆ |
| Context | ★★★★★ | ★★★★☆ | ★★☆☆☆ |

### Selection by Use Case

**High-Stakes Reasoning (Legal, Medical, Financial):**
- Primary: Claude 3 Opus or GPT-4
- Rationale: Maximum accuracy, strong instruction following

**High-Volume Processing:**
- Primary: Claude 3 Haiku or GPT-3.5 Turbo
- Rationale: Cost efficiency at scale

**Code Generation:**
- Primary: Claude 3.5 Sonnet or GPT-4
- Alternative: DeepSeek Coder (open-source)

**Privacy-Sensitive:**
- Primary: Self-hosted Llama 3 or Mistral
- Rationale: Data never leaves your infrastructure

**Multi-Step Agents:**
- Primary: Claude 3.5 Sonnet
- Rationale: Strong tool use, good cost/capability balance

---

## Building Provider-Agnostic Systems

### Abstraction Layer


<span class="filename">agent.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class Message:
    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int
    output_tokens: int


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> LLMResponse:
        pass


class ClaudeProvider(LLMProvider):
    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> LLMResponse:
        # Extract system message if present
        system = None
        chat_messages = []
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                chat_messages.append({"role": m.role, "content": m.content})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=chat_messages
        )

        return LLMResponse(
            content=response.content[0].text,
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens
        )


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4-turbo-preview"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> LLMResponse:
        formatted = [{"role": m.role, "content": m.content} for m in messages]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=formatted,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens
        )
```

</div>
</div>

### Using the Abstraction

```python
def create_provider(provider_name: str, **kwargs) -> LLMProvider:
    """Factory function to create LLM providers."""
    providers = {
        "claude": ClaudeProvider,
        "openai": OpenAIProvider,
    }
    return providers[provider_name](**kwargs)


# Usage - easily switch providers
provider = create_provider("claude", model="claude-3-5-sonnet-20241022")

response = provider.chat([
    Message(role="system", content="You are a helpful assistant."),
    Message(role="user", content="What is the capital of France?")
])

print(f"Response: {response.content}")
print(f"Tokens used: {response.input_tokens + response.output_tokens}")
```

---

## Cost Optimization Strategies

### Model Routing

Use cheaper models for simple tasks, expensive models for complex ones:


<span class="filename">agent.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def route_to_model(query: str, complexity: str = "auto") -> LLMProvider:
    """Route queries to appropriate models based on complexity."""

    if complexity == "auto":
        # Simple heuristic - improve with classification model
        complexity = "simple" if len(query) < 100 else "complex"

    if complexity == "simple":
        return create_provider("claude", model="claude-3-haiku-20240307")
    else:
        return create_provider("claude", model="claude-3-5-sonnet-20241022")
```

</div>
</div>

### Caching

Cache responses for repeated queries:

```python
import hashlib
import json
from functools import lru_cache


def hash_messages(messages: list[Message]) -> str:
    """Create a hash of the message list for caching."""
    content = json.dumps([(m.role, m.content) for m in messages])
    return hashlib.sha256(content.encode()).hexdigest()


@lru_cache(maxsize=1000)
def cached_chat(provider_name: str, messages_hash: str, messages_json: str) -> str:
    """Cache LLM responses by message hash."""
    messages = [Message(**m) for m in json.loads(messages_json)]
    provider = create_provider(provider_name)
    return provider.chat(messages).content


def chat_with_cache(provider_name: str, messages: list[Message]) -> str:
    """Chat with caching enabled."""
    messages_json = json.dumps([{"role": m.role, "content": m.content} for m in messages])
    messages_hash = hash_messages(messages)
    return cached_chat(provider_name, messages_hash, messages_json)
```

---

## API Key Management

### Environment Variables (Recommended)

```bash
# .env file
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```


<span class="filename">agent.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import os
from dotenv import load_dotenv

load_dotenv()

# Keys are automatically picked up by clients
anthropic_client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY
openai_client = OpenAI()  # Uses OPENAI_API_KEY
```

</div>
</div>

### Secrets Management (Production)

```python
import boto3
from functools import lru_cache


@lru_cache
def get_api_key(secret_name: str) -> str:
    """Retrieve API key from AWS Secrets Manager."""
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=secret_name)
    return response["SecretString"]


# Usage
anthropic_key = get_api_key("prod/anthropic-api-key")
```

---

## Rate Limiting and Retries


<span class="filename">agent.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import time
from tenacity import retry, stop_after_attempt, wait_exponential


class RateLimitedProvider(LLMProvider):
    """Wrapper that adds rate limiting and retries."""

    def __init__(self, provider: LLMProvider, requests_per_minute: int = 60):
        self.provider = provider
        self.min_interval = 60.0 / requests_per_minute
        self.last_request = 0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> LLMResponse:
        # Rate limiting
        elapsed = time.time() - self.last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

        self.last_request = time.time()
        return self.provider.chat(messages, temperature, max_tokens)
```

</div>
</div>

---

## Provider Feature Comparison

| Feature | Claude | GPT-4 | Llama 3 |
|---------|--------|-------|---------|
| Tool/Function Calling | ✅ | ✅ | ✅ (with prompting) |
| Vision | ✅ | ✅ | ❌ |
| Streaming | ✅ | ✅ | ✅ |
| Fine-tuning | ❌ | ✅ | ✅ |
| JSON Mode | ✅ | ✅ | ❌ |
| System Prompts | ✅ | ✅ | ✅ |
| Batch API | ✅ | ✅ | N/A |

<div class="callout-key">

**Key Concept Summary:** This guide covered the core concepts. Review the companion slides for visual summaries and the hands-on notebook for practice implementations.

</div>

---

*Choosing the right LLM provider is about matching capabilities to requirements. Build abstractions that let you switch providers easily, and don't hesitate to use multiple models for different purposes.*


## Practice Questions

1. Explain in your own words how the concepts in this guide relate to building production agents.
2. What are the key tradeoffs you need to consider when applying these techniques?
3. Describe a scenario where the approach from this guide would be the wrong choice, and what you would use instead.

---

**Next Steps:**

<a class="link-card" href="./02_llm_providers_slides.md">
  <div class="link-card-title">LLM Providers: Choosing the Right Model — Companion Slides</div>
  <div class="link-card-description">Visual slide deck with diagrams, speaker notes, and key takeaways.</div>
</a>

<a class="link-card" href="../notebooks/01_api_setup.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with working code and guided exercises.</div>
</a>
