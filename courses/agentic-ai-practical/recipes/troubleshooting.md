# Troubleshooting Guide

> **Reading time:** ~8 min | **Topic:** Debugging AI Agents | **Prerequisites:** None

<div class="callout-key">

**Key Concept Summary:** Agent debugging follows a predictable pattern: API errors are configuration issues, tool-calling errors are schema or handler mismatches, and performance issues are prompt or model-selection problems. This guide covers the most common errors in order of frequency. For each error, the fix is shown as copy-paste code you can drop into your project.

</div>

## API Errors

These errors occur before your agent logic runs. They are almost always configuration problems.

### `AuthenticationError: Invalid API key`

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">fix_auth.sh</span>
</div>

```bash
# Check your API key is set
echo $ANTHROPIC_API_KEY

# Set it if missing
export ANTHROPIC_API_KEY="sk-ant-..."

# Verify it works
python -c "import anthropic; print(anthropic.Anthropic().messages.create(
    model='claude-sonnet-4-20250514', max_tokens=10,
    messages=[{'role': 'user', 'content': 'ping'}]).content[0].text)"
```

</div>

<div class="callout-warning">

**Warning:** Never hardcode API keys in source files. Use environment variables or a `.env` file (add `.env` to `.gitignore`). A leaked API key on GitHub will be exploited within minutes by automated scanners.

</div>

### `RateLimitError: Rate limit exceeded`

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">retry_logic.py</span>
</div>

```python
import time
import anthropic

def call_with_retry(client, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError:
            wait = 2 ** attempt  # 1s, 2s, 4s
            print(f"Rate limited. Waiting {wait}s...")
            time.sleep(wait)
    raise Exception("Max retries exceeded")
```

</div>

### `APIConnectionError: Connection failed`

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">diagnose_connection.sh</span>
</div>

```bash
# 1. Check internet
ping -c 1 google.com

# 2. Check API endpoint is reachable
curl -s https://api.anthropic.com/v1/messages -w "%{http_code}" -o /dev/null

# 3. Check for proxy/firewall issues
curl -v https://api.anthropic.com/v1/messages 2>&1 | head -20
```

</div>

---

## ChromaDB Errors

These errors occur in the RAG pipeline during document indexing or retrieval.

### `Collection already exists`

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">fix_collection.py</span>
</div>

```python
# Use get_or_create instead of create
collection = chroma.get_or_create_collection("name")
```

</div>

### `No documents found` / `Embedding dimension mismatch`

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">fix_chroma.py</span>
</div>

```python
# Diagnose: check document count
print(f"Document count: {collection.count()}")

# Fix dimension mismatch: delete and recreate
chroma.delete_collection("name")
collection = chroma.create_collection("name")
# Re-index all documents
```

</div>

<div class="callout-insight">

**Insight:** Embedding dimension mismatches happen when you switch embedding models between indexing and querying. ChromaDB stores the dimension from the first `add()` call. If you change your embedding function, you must recreate the collection from scratch.

</div>

---

## Tool Calling Errors

These are the most common runtime errors in agent loops.

### `Tool not found`

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">fix_tool_name.py</span>
</div>

```python
# The tool name in definitions MUST match the handler key exactly
tools = [{"name": "my_tool", ...}]       # Definition
handlers = {"my_tool": my_function}       # Handler -- must match!

# Common mistake: "my-tool" vs "my_tool" (hyphens vs underscores)
```

</div>

### `Tool returned None`

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">fix_tool_return.py</span>
</div>

```python
# Always return a string from tool handlers
def my_tool(x):
    result = do_something(x)
    return str(result)  # Never return None -- convert to string
```

</div>

### `Max turns exceeded` / Agent stuck in loop

<div class="callout-danger">

**Danger:** An agent stuck in a tool-calling loop will keep making API calls (and running up costs) until you stop it. Always set a `max_turns` limit and add logging to see what the agent is doing on each turn. If the agent calls the same tool with the same arguments twice in a row, break the loop.

</div>

---

## JSON Parsing Errors

### `JSONDecodeError`

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">fix_json.py</span>
</div>

```python
# Fix: Use structured output prompting
system = "Return ONLY valid JSON. No markdown, no explanation."

# Or use defensive parsing
import json
try:
    data = json.loads(response)
except json.JSONDecodeError:
    data = {"error": "Failed to parse", "raw": response}
```

</div>

---

## Memory / Context Errors

### `Context length exceeded`

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">fix_context.py</span>
</div>

```python
# Trim old messages, preserving system prompt
if len(messages) > 20:
    messages = [messages[0]] + messages[-10:]  # system + last 10
```

</div>

### `Lost in the middle`

<div class="callout-info">

**Info:** LLMs pay more attention to the beginning and end of the context window. If critical information is buried in the middle of a long conversation, the model may ignore it. Put important instructions in the system prompt (beginning) and the most relevant context near the user's question (end).

</div>

---

## Performance Issues

### Slow responses

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">fix_performance.py</span>
</div>

```python
# Fix 1: Use a faster model for simple tasks
model = "claude-3-5-haiku-20241022"

# Fix 2: Reduce max_tokens to only what you need
max_tokens = 256

# Fix 3: Use streaming for perceived speed
with client.messages.stream(
    model="claude-sonnet-4-20250514", max_tokens=1024, messages=messages
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

</div>

### High costs

| Strategy | Impact | Effort |
|----------|--------|--------|
| Use Haiku for simple tasks | 10-20x cheaper | Low -- just change model string |
| Cache repeated queries | Varies | Medium -- add caching layer |
| Trim conversation history | Linear cost reduction | Low -- sliding window |
| Count tokens before calling | Prevents surprise bills | Low -- add a guard |

---

## Quick Diagnostic Checklist

<div class="flow">
<div class="flow-step rose">1. Check API key</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">2. Check tool names match</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">3. Add logging</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step mint">4. Print full response</div>
</div>

<div class="callout-key">

**Key Point:** When debugging an agent, always start with `print(response)` to see the full API response. Most issues become obvious when you can see exactly what the LLM returned -- including stop_reason, tool_use blocks, and content blocks.

</div>

---

<a class="link-card" href="./common_patterns.py">
  <div class="link-card-title">Common Patterns</div>
  <div class="link-card-description">Copy-paste code recipes for API calls, streaming, retries, and structured output.</div>
</a>

<a class="link-card" href="../resources/cheat_sheet.md">
  <div class="link-card-title">Cheat Sheet</div>
  <div class="link-card-description">Quick reference for API calls, tool definitions, RAG, and streaming patterns.</div>
</a>
