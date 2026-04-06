# Module 0: Foundations Cheatsheet

> **Reading time:** ~5 min | **Module:** 0 — Foundations | **Prerequisites:** Module 0 guides

## Key Concepts

| Concept | Definition |
|---------|-----------|
| **Transformer** | Neural network architecture using self-attention to process sequences in parallel |
| **Token** | Smallest unit of text an LLM processes (subword, word, or character) |
| **Context Window** | Maximum number of tokens (input + output) an LLM can process in one request |
| **Embedding** | Dense vector representation of tokens capturing semantic meaning |
| **Attention** | Mechanism allowing each token to weigh relevance of all other tokens |
| **Autoregressive** | Generating one token at a time based on all previous tokens |
| **Temperature** | Controls randomness in output (0 = deterministic, 2 = very random) |
| **Top-p (Nucleus)** | Sampling from smallest set of tokens whose cumulative probability exceeds p |

## Common API Patterns

### Basic Claude API Call

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>

```python
import anthropic

client = anthropic.Anthropic(api_key="your-key")
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude!"}
    ]
)
print(message.content[0].text)
```

</div>
</div>

### OpenAI API Call
```python
from openai import OpenAI

client = OpenAI(api_key="your-key")
response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "user", "content": "Hello, GPT!"}
    ]
)
print(response.choices[0].message.content)
```

### Streaming Responses
```python
with client.messages.stream(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Tell me a story"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

## Token Economics Quick Reference

| Model | Context | Input ($/1M tokens) | Output ($/1M tokens) |
|-------|---------|---------------------|----------------------|
| Claude 3.5 Sonnet | 200K | $3.00 | $15.00 |
| GPT-4 Turbo | 128K | $10.00 | $30.00 |
| GPT-3.5 Turbo | 16K | $0.50 | $1.50 |
| Llama 3 70B | 8K | Free (self-host) | Free (self-host) |

**Rule of Thumb:** 1 word ≈ 1.3 tokens in English

## Temperature Guide


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>

```python

# Factual, deterministic (Q&A, extraction, classification)
temperature=0.0

# Slightly creative (summaries, explanations)
temperature=0.3

# Balanced (general conversation)
temperature=0.7

# Creative writing (stories, brainstorming)
temperature=1.0
```

</div>
</div>

## Gotchas

- **Context overflow** - Always count tokens before sending. Use `tiktoken` (OpenAI) or Anthropic's count API


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>

```python

# Count tokens before sending
token_count = client.count_tokens(text)
if token_count > max_context:
    # Truncate or summarize
```


- **Output token limits** - Set `max_tokens` explicitly; models won't complete responses without it

```python

# Bad: might truncate mid-sentence
max_tokens=100

# Good: allow full responses
max_tokens=2048
```

- **Rate limits** - Implement exponential backoff for production

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def call_api():
    return client.messages.create(...)
```

- **API key security** - Never hardcode keys; use environment variables

```python
import os
api_key = os.getenv("ANTHROPIC_API_KEY")
```

- **Tokenization differences** - Each model uses different tokenizers; always use the correct counter
  - Claude: Anthropic API token counter
  - GPT: `tiktoken` library
  - Llama: Model-specific tokenizer

- **Billing surprises** - Input + output tokens both count; monitor usage

```python
usage = message.usage
cost = (usage.input_tokens * 0.000003) + (usage.output_tokens * 0.000015)
print(f"Request cost: ${cost:.4f}")
```
