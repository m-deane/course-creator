# AI Agents Cheat Sheet

## Quick Setup
```bash
pip install anthropic chromadb
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Basic API Call
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

## With System Prompt
```python
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system="You are a helpful assistant.",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Conversation Memory
```python
messages = []
messages.append({"role": "user", "content": "I'm Alex"})
response = client.messages.create(model="...", messages=messages)
messages.append({"role": "assistant", "content": response.content[0].text})
messages.append({"role": "user", "content": "What's my name?"})
# Now it remembers!
```

## Tool Calling
```python
tools = [{
    "name": "calculator",
    "description": "Do math",
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

## RAG (3 Lines)
```python
import chromadb
col = chromadb.Client().create_collection("docs")
col.add(documents=["doc1", "doc2"], ids=["1", "2"])
results = col.query(query_texts=["question"], n_results=2)
```

## Streaming
```python
with client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="")
```

## Models

| Model | Use Case | Cost |
|-------|----------|------|
| claude-sonnet-4-20250514 | Best balance | $3/$15 per M |
| claude-3-5-haiku | Fast & cheap | $0.25/$1.25 per M |
| claude-3-opus | Complex reasoning | $15/$75 per M |

## Common Errors

| Error | Fix |
|-------|-----|
| `AuthenticationError` | Check API key |
| `RateLimitError` | Add retry with backoff |
| `Context too long` | Trim messages |
| `Tool not found` | Check tool name matches |

## Links

- [Anthropic Docs](https://docs.anthropic.com)
- [Claude Cookbook](https://github.com/anthropics/anthropic-cookbook)
- [ChromaDB Docs](https://docs.trychroma.com)
