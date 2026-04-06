# AI Agents Cheat Sheet

> **Reading time:** ~3 min | **Topic:** Quick Reference | **Prerequisites:** None

<span class="badge mint">Beginner</span> <span class="badge amber">Reference</span> <span class="badge blue">Copy-Paste Ready</span>

## Quick Setup

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">setup.sh</span>
</div>

```bash
pip install anthropic chromadb
export ANTHROPIC_API_KEY="sk-ant-..."
```

</div>

## Basic API Call

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">basic_call.py</span>
</div>

```python
import anthropic
client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.content[0].text)
```

</div>

## With System Prompt

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">system_prompt.py</span>
</div>

```python
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system="You are a helpful assistant.",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

</div>

## Conversation Memory

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">memory.py</span>
</div>

```python
messages = []
messages.append({"role": "user", "content": "I'm Alex"})
response = client.messages.create(model="claude-sonnet-4-20250514", messages=messages)
messages.append({"role": "assistant", "content": response.content[0].text})
messages.append({"role": "user", "content": "What's my name?"})
# Now it remembers!
```

</div>

## Tool Calling

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">tool_calling.py</span>
</div>

```python
tools = [{
    "name": "calculator",
    "description": "Calculate arithmetic expressions like 2+2 or sqrt(144)",
    "input_schema": {
        "type": "object",
        "properties": {"expr": {"type": "string"}},
        "required": ["expr"]
    }
}]

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    tools=tools,
    messages=[{"role": "user", "content": "What's 2+2?"}]
)

if response.stop_reason == "tool_use":
    tool = response.content[0]  # tool.name, tool.input
```

</div>

## RAG (3 Lines)

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">rag_minimal.py</span>
</div>

```python
import chromadb
col = chromadb.Client().create_collection("docs")
col.add(documents=["doc1", "doc2"], ids=["1", "2"])
results = col.query(query_texts=["question"], n_results=2)
```

</div>

## Streaming

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">streaming.py</span>
</div>

```python
with client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

</div>

## Models

| Model | Use Case | Input / Output Cost (per M tokens) |
|-------|----------|-------------------------------------|
| `claude-sonnet-4-20250514` | Best balance of quality and speed | $3 / $15 |
| `claude-3-5-haiku-20241022` | Fast and cheap | $0.25 / $1.25 |
| `claude-3-opus-20240229` | Complex reasoning | $15 / $75 |

<div class="callout-key">

**Key Point:** Use Haiku for simple tasks (classification, extraction, short answers) and Sonnet for complex tasks (reasoning, code generation, multi-step). This single decision can reduce your API costs by 10-20x without meaningful quality loss on simple tasks.

</div>

## Common Errors Quick Reference

| Error | Fix |
|-------|-----|
| `AuthenticationError` | Check `ANTHROPIC_API_KEY` environment variable |
| `RateLimitError` | Add exponential backoff retry logic |
| `Context too long` | Trim messages: `messages = [messages[0]] + messages[-10:]` |
| `Tool not found` | Check tool name matches between definition and handler |
| `JSONDecodeError` | Add `"Return ONLY valid JSON"` to system prompt |

---

<a class="link-card" href="../recipes/troubleshooting.md">
  <div class="link-card-title">Troubleshooting Guide</div>
  <div class="link-card-description">Detailed fixes for every common error with copy-paste solutions.</div>
</a>

<a class="link-card" href="../recipes/common_patterns.py">
  <div class="link-card-title">Common Patterns</div>
  <div class="link-card-description">Self-contained code recipes for API calls, streaming, retries, and structured output.</div>
</a>

<a class="link-card" href="../concepts/visual_guides/tool_calling.md">
  <div class="link-card-title">Tool Calling Visual Guide</div>
  <div class="link-card-description">Understand the tool-calling loop with sequence diagrams and flow charts.</div>
</a>
