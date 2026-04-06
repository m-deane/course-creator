# Module 2: Tool Use & Function Calling Cheatsheet

> **Reading time:** ~5 min | **Module:** 2 — Tool Use & Function Calling | **Prerequisites:** Module 2 guides

## Key Concepts

| Concept | Definition |
|---------|-----------|
| **Tool/Function** | External capability that LLM can invoke (API, database, file system, computation) |
| **Tool Schema** | JSON definition describing tool name, purpose, parameters, and types |
| **Function Calling** | LLM's ability to generate structured calls to external functions |
| **Tool Use Loop** | Cycle of: LLM decides tool needed → Execute tool → Return result → LLM processes |
| **ReAct Pattern** | Reasoning + Acting: LLM reasons about what to do, takes action, observes result |
| **Tool Chaining** | Using output of one tool as input to another for multi-step tasks |
| **Sandboxing** | Isolated execution environment to safely run untrusted tool calls |
| **Parameter Validation** | Checking tool inputs meet schema requirements before execution |

## Common Patterns

### Basic Tool Definition (Claude)
<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>
<div class="code-body">

```python
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location. Use this when user asks about weather, temperature, or conditions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state, e.g., 'San Francisco, CA'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }
    }
]
```

</div>
</div>

### Basic Tool Definition (OpenAI)
```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }
]
```

### Tool Use Loop (Claude)
```python
import anthropic
import json

client = anthropic.Anthropic()

def process_tool_call(tool_name, tool_input):
    """Execute the actual tool logic"""
    if tool_name == "get_weather":
        # Call weather API
        return {"temperature": 72, "conditions": "sunny"}
    # ... other tools

# Initial message
messages = [{"role": "user", "content": "What's the weather in NYC?"}]

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    messages=messages
)

# Process tool calls
while response.stop_reason == "tool_use":
    tool_use = next(block for block in response.content if block.type == "tool_use")

    # Execute tool
    tool_result = process_tool_call(tool_use.name, tool_use.input)

    # Add tool result to conversation
    messages.append({"role": "assistant", "content": response.content})
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": json.dumps(tool_result)
            }
        ]
    })

    # Get next response
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )

# Final answer
print(response.content[0].text)
```

### Tool Use Loop (OpenAI)
```python
from openai import OpenAI
import json

client = OpenAI()

messages = [{"role": "user", "content": "What's the weather in NYC?"}]

response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=messages,
    tools=tools
)

# Process function calls
while response.choices[0].finish_reason == "tool_calls":
    tool_calls = response.choices[0].message.tool_calls

    messages.append(response.choices[0].message)

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        # Execute function
        function_result = process_tool_call(function_name, function_args)

        # Add result to messages
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": function_name,
            "content": json.dumps(function_result)
        })

    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=messages,
        tools=tools
    )

print(response.choices[0].message.content)
```

### Error Handling Wrapper
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def safe_tool_execution(tool_name, tool_input):
    try:
        # Validate input
        validate_tool_input(tool_name, tool_input)

        # Execute with timeout
        result = execute_with_timeout(tool_name, tool_input, timeout=30)

        # Validate output
        validate_tool_output(tool_name, result)

        return {"success": True, "data": result}

    except ValidationError as e:
        return {"success": False, "error": f"Invalid input: {e}"}
    except TimeoutError:
        return {"success": False, "error": "Tool execution timed out"}
    except Exception as e:
        return {"success": False, "error": f"Tool failed: {e}"}
```

### Parallel Tool Execution
```python
import asyncio

async def execute_tools_parallel(tool_calls):
    """Execute multiple independent tools concurrently"""
    tasks = [
        asyncio.create_task(async_tool_execution(call))
        for call in tool_calls
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

# Use in tool loop
if len(tool_calls) > 1 and tools_are_independent(tool_calls):
    results = asyncio.run(execute_tools_parallel(tool_calls))
```

## Tool Design Principles

### Good Tool Names
<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>
<div class="code-body">

```python
# Use clear verbs
"search_documents"    # Good
"doc_search"          # Okay
"find"               # Too vague

# Be specific about action
"create_calendar_event"  # Good
"calendar"               # Too vague
```

</div>
</div>

### Good Tool Descriptions
```python
# Bad: Vague
"description": "Searches for stuff"

# Good: Specific with guidance
"description": "Search company documents using keywords. Use this when user asks about internal policies, procedures, or documentation. Returns top 5 most relevant documents."
```

### Parameter Design
```python
{
    "properties": {
        # Required params: clear, specific type
        "query": {
            "type": "string",
            "description": "Search keywords or question"
        },

        # Optional params: include default
        "limit": {
            "type": "integer",
            "description": "Max results to return (default: 5)",
            "default": 5,
            "minimum": 1,
            "maximum": 20
        },

        # Enums for constrained choices
        "sort_by": {
            "type": "string",
            "enum": ["relevance", "date", "title"],
            "description": "How to sort results (default: relevance)"
        }
    }
}
```

## Security Checklist

- [ ] **Input validation** - Validate all parameters against schema
  <div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>
<div class="code-body">

```python
  from jsonschema import validate
  validate(instance=tool_input, schema=tool_schema)
  ```

</div>
</div>

- [ ] **Parameter sanitization** - Clean potentially dangerous inputs
  ```python
  # SQL queries
  from sqlalchemy import text
  query = text("SELECT * FROM users WHERE id = :id")
  result = connection.execute(query, {"id": user_id})

  # File paths
  from pathlib import Path
  safe_path = Path("/safe/directory") / Path(user_path).name
  ```

- [ ] **Rate limiting** - Prevent abuse of expensive tools
  ```python
  from functools import wraps
  from time import time

  def rate_limit(max_calls, period):
      calls = []
      def decorator(func):
          @wraps(func)
          def wrapper(*args, **kwargs):
              now = time()
              calls[:] = [c for c in calls if c > now - period]
              if len(calls) >= max_calls:
                  raise Exception("Rate limit exceeded")
              calls.append(now)
              return func(*args, **kwargs)
          return wrapper
      return decorator
  ```

- [ ] **Timeout protection** - Don't let tools hang forever
  ```python
  import signal

  def timeout_handler(signum, frame):
      raise TimeoutError("Tool execution exceeded timeout")

  signal.signal(signal.SIGALRM, timeout_handler)
  signal.alarm(30)  # 30 second timeout
  try:
      result = execute_tool()
  finally:
      signal.alarm(0)  # Cancel timeout
  ```

- [ ] **Audit logging** - Track all tool executions
  ```python
  import logging

  logger.info(f"Tool execution: {tool_name}", extra={
      "tool_name": tool_name,
      "parameters": tool_input,
      "user_id": user_id,
      "timestamp": datetime.now()
  })
  ```

## Gotchas

- **Tool description quality** - LLM relies heavily on description to choose tools
  <div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>
<div class="code-body">

```python
  # Bad: LLM won't know when to use this
  "description": "Gets data"

  # Good: Clear trigger conditions
  "description": "Gets user profile data including name, email, and preferences. Use when user asks about their account or settings."
  ```

</div>
</div>

- **Too many tools** - LLM performance degrades with >20 tools
  ```python
  # Solution: Group related tools or use hierarchical selection
  # First: choose category tool
  # Then: provide tools for that category
  ```

- **Non-deterministic tool selection** - Same input might choose different tool
  ```python
  # Use temperature=0 for consistent tool selection
  response = client.messages.create(temperature=0, tools=tools, ...)
  ```

- **Tool result formatting** - Always return JSON-serializable data
  ```python
  # Bad: Python objects
  return {"user": User(id=1, name="Alice")}

  # Good: Serializable dict
  return {"user": {"id": 1, "name": "Alice"}}
  ```

- **Circular tool loops** - Agent keeps calling tools without progress
  ```python
  # Solution: Limit tool use iterations
  max_iterations = 10
  iteration = 0
  while response.stop_reason == "tool_use" and iteration < max_iterations:
      # process tools
      iteration += 1
  ```

- **Expensive tool costs** - Tool execution can be more expensive than LLM calls
  ```python
  # Cache tool results for repeated queries
  from functools import lru_cache

  @lru_cache(maxsize=100)
  def cached_tool_execution(tool_name, tool_input_hash):
      return execute_tool(tool_name, tool_input)
  ```
