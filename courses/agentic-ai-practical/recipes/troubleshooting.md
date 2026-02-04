# Troubleshooting Guide

Common errors and how to fix them.

## API Errors

### `AuthenticationError: Invalid API key`
```
Fix: Check your ANTHROPIC_API_KEY environment variable
     export ANTHROPIC_API_KEY="sk-ant-..."
```

### `RateLimitError: Rate limit exceeded`
```
Fix: Add retry logic with exponential backoff
     See: common_patterns.py → RECIPE 3
```

### `APIConnectionError: Connection failed`
```
Fix: 1. Check internet connection
     2. Check if api.anthropic.com is accessible
     3. Try: curl https://api.anthropic.com/v1/messages
```

## ChromaDB Errors

### `Collection already exists`
```python
# Fix: Use get_or_create instead of create
collection = chroma.get_or_create_collection("name")
```

### `No documents found`
```python
# Fix: Check if documents were added
print(f"Document count: {collection.count()}")
```

### `Embedding dimension mismatch`
```python
# Fix: Delete and recreate collection
chroma.delete_collection("name")
collection = chroma.create_collection("name")
```

## Tool Calling Errors

### `Tool not found`
```python
# Fix: Ensure tool name matches exactly
tools = [{"name": "my_tool", ...}]  # Define
handlers = {"my_tool": my_function}  # Must match
```

### `Tool returned None`
```python
# Fix: Always return a string from tool handlers
def my_tool(x):
    result = do_something(x)
    return str(result)  # Convert to string
```

### `Max turns exceeded`
```python
# Fix: Check if tool is returning useful results
# The agent might be stuck in a loop
# Add logging to see what's happening
```

## JSON Parsing Errors

### `JSONDecodeError`
```python
# Fix: Use structured output prompting
system = "Return ONLY valid JSON. No markdown, no explanation."

# Or use try/except
import json
try:
    data = json.loads(response)
except json.JSONDecodeError:
    data = {"error": "Failed to parse", "raw": response}
```

## Memory/Context Errors

### `Context length exceeded`
```python
# Fix: Trim old messages
if len(messages) > 20:
    messages = [messages[0]] + messages[-10:]  # Keep system + last 10
```

### `Lost in the middle`
```
Fix: Put important info at START and END of prompts
     Middle content gets less attention
```

## Performance Issues

### Slow responses
```python
# Fix 1: Use a faster model
model = "claude-3-5-haiku-20241022"  # Faster than Sonnet

# Fix 2: Reduce max_tokens
max_tokens = 256  # Only what you need

# Fix 3: Use streaming for perceived speed
with client.messages.stream(...) as stream:
    for text in stream.text_stream:
        print(text, end="")
```

### High costs
```python
# Fix 1: Use Haiku for simple tasks
# Fix 2: Cache responses
# Fix 3: Count tokens before calling
tokens = count_tokens(prompt)
if tokens > 10000:
    print("Warning: Large prompt will be expensive")
```

## Still stuck?

1. Check the [Anthropic docs](https://docs.anthropic.com)
2. Print everything: `print(response)` to see full response
3. Use `logging.basicConfig(level=logging.DEBUG)`
