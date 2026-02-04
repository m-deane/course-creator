# Tool Calling

```
┌─────────────────────────────────────────────────────────────────┐
│                       TOOL CALLING                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   User: "What's 123 × 456?"                                     │
│            │                                                    │
│            ▼                                                    │
│   ┌─────────────────────────────────────────────┐               │
│   │                    LLM                       │               │
│   │  "I should use the calculator tool"          │               │
│   └─────────────────┬───────────────────────────┘               │
│                     │                                           │
│                     ▼ tool_use                                  │
│   ┌─────────────────────────────────────────────┐               │
│   │  {                                           │               │
│   │    "name": "calculator",                     │               │
│   │    "input": {"expression": "123 * 456"}      │               │
│   │  }                                           │               │
│   └─────────────────┬───────────────────────────┘               │
│                     │                                           │
│                     ▼ your code runs                            │
│   ┌─────────────────────────────────────────────┐               │
│   │  def calculator(expression):                 │               │
│   │      return str(eval(expression))  # 56088   │               │
│   └─────────────────┬───────────────────────────┘               │
│                     │                                           │
│                     ▼ tool_result                               │
│   ┌─────────────────────────────────────────────┐               │
│   │                    LLM                       │               │
│   │  "The answer is 56,088"                      │               │
│   └─────────────────┬───────────────────────────┘               │
│                     │                                           │
│                     ▼                                           │
│   Agent: "123 × 456 = 56,088"                                   │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ TL;DR: LLM decides WHAT tool to call, YOUR code runs it         │
├─────────────────────────────────────────────────────────────────┤
│ Code:                                                           │
│   tools = [{"name": "calc", "description": "...", "schema": {}}]│
│   response = client.messages.create(tools=tools, ...)           │
│   if response.stop_reason == "tool_use":                        │
│       result = run_tool(response.content[0])                    │
├─────────────────────────────────────────────────────────────────┤
│ Pitfall: Tool descriptions matter! Be specific.                │
│          Bad:  "Does math"                                      │
│          Good: "Calculate arithmetic expressions like 2+2"      │
└─────────────────────────────────────────────────────────────────┘
```

## Tool Definition Schema

```python
{
    "name": "get_weather",           # Unique identifier
    "description": "Get weather...", # LLM reads this to decide
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"}
        },
        "required": ["city"]
    }
}
```

## The Loop

```
┌──────────────────────────────────────────────────────┐
│                                                      │
│   ┌─────────┐    tool_use    ┌──────────┐           │
│   │   LLM   │───────────────▶│ Your Code│           │
│   └────▲────┘                └────┬─────┘           │
│        │                          │                 │
│        │      tool_result         │                 │
│        └──────────────────────────┘                 │
│                                                      │
│   Loop until: stop_reason == "end_turn"              │
└──────────────────────────────────────────────────────┘
```

## Common Tools

| Tool | Use Case |
|------|----------|
| `calculator` | Math operations |
| `search` | Web/database search |
| `get_weather` | External API calls |
| `read_file` | Local file access |
| `execute_code` | Run Python (careful!) |

→ Full template: [../templates/agent_template.py](../templates/agent_template.py)
