# Tool Fundamentals: How LLM Tool Calling Works

## In Brief

Tool calling (also called function calling) lets LLMs decide when to invoke external functions and with what arguments. The model generates a structured tool call, your code executes it, and the result is fed back for the model to use.

> 💡 **Key Insight:** **The model doesn't execute tools—it generates instructions for you to execute.** Tool calling is a structured output format that your code interprets. This separation is crucial for security, control, and integration.

---

## The Tool Calling Flow

### Complete Lifecycle

```
1. USER INPUT
   "What's the weather in Tokyo?"
                ↓
2. LLM RECEIVES (system prompt + tools + user message)
   Tools available: [get_weather, search_web, ...]
                ↓
3. LLM DECIDES (based on query and tool descriptions)
   "I need get_weather for this query"
                ↓
4. LLM GENERATES TOOL CALL
   {"name": "get_weather", "arguments": {"city": "Tokyo"}}
                ↓
5. YOUR CODE EXECUTES
   result = get_weather("Tokyo")  # → {"temp": 22, "conditions": "sunny"}
                ↓
6. RESULT RETURNED TO LLM
   [tool_result: {"temp": 22, "conditions": "sunny"}]
                ↓
7. LLM GENERATES FINAL RESPONSE
   "The weather in Tokyo is sunny with a temperature of 22°C."
```

### Code Implementation

```python
import anthropic
import json

client = anthropic.Anthropic()


# Step 1: Define tools
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city. Use this when users ask about weather conditions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name, e.g., 'Tokyo' or 'New York'"
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature units"
                }
            },
            "required": ["city"]
        }
    }
]


# Step 2: Implement tool execution
def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool and return the result as a string."""
    if name == "get_weather":
        # In reality, call a weather API
        city = arguments["city"]
        return json.dumps({
            "city": city,
            "temperature": 22,
            "conditions": "sunny",
            "humidity": 65
        })
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


# Step 3: Run the agent loop
def run_agent(user_message: str) -> str:
    """Run a tool-using agent."""

    messages = [{"role": "user", "content": user_message}]

    # First API call
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )

    # Check if model wants to use a tool
    while response.stop_reason == "tool_use":
        # Extract tool calls from response
        tool_calls = [
            block for block in response.content
            if block.type == "tool_use"
        ]

        # Add assistant's response to messages
        messages.append({"role": "assistant", "content": response.content})

        # Execute each tool and collect results
        tool_results = []
        for tool_call in tool_calls:
            result = execute_tool(tool_call.name, tool_call.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": result
            })

        # Add tool results to messages
        messages.append({"role": "user", "content": tool_results})

        # Continue the conversation
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )

    # Extract final text response
    return "".join(
        block.text for block in response.content
        if hasattr(block, "text")
    )


# Usage
answer = run_agent("What's the weather like in Tokyo?")
print(answer)
```

---

## Tool Schema Structure

### JSON Schema Basics

Tools use JSON Schema to define their parameters:

```python
tool = {
    "name": "search_database",
    "description": "Search a database for records matching criteria",
    "input_schema": {
        "type": "object",
        "properties": {
            "table": {
                "type": "string",
                "description": "The table to search",
                "enum": ["users", "orders", "products"]
            },
            "query": {
                "type": "string",
                "description": "Search query string"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results to return",
                "default": 10,
                "minimum": 1,
                "maximum": 100
            },
            "filters": {
                "type": "object",
                "description": "Additional filters as key-value pairs",
                "additionalProperties": {"type": "string"}
            }
        },
        "required": ["table", "query"]
    }
}
```

### Common Schema Types

```python
# String with constraints
{
    "type": "string",
    "minLength": 1,
    "maxLength": 1000,
    "pattern": "^[a-z]+$"  # Regex pattern
}

# Number with range
{
    "type": "number",
    "minimum": 0,
    "maximum": 100,
    "multipleOf": 0.5  # Must be divisible by 0.5
}

# Array of items
{
    "type": "array",
    "items": {"type": "string"},
    "minItems": 1,
    "maxItems": 10,
    "uniqueItems": True
}

# Enum (fixed choices)
{
    "type": "string",
    "enum": ["option1", "option2", "option3"]
}

# Nested object
{
    "type": "object",
    "properties": {
        "nested_field": {"type": "string"}
    }
}
```

---

## Multiple Tools

### Providing Multiple Tools

```python
tools = [
    {
        "name": "search_web",
        "description": "Search the web for current information. Use for questions about recent events or facts you're uncertain about.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_weather",
        "description": "Get current weather for a location. Use only for weather-specific questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City or location name"}
            },
            "required": ["location"]
        }
    },
    {
        "name": "calculate",
        "description": "Perform mathematical calculations. Use for any math that requires precision.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate"}
            },
            "required": ["expression"]
        }
    }
]
```

### Tool Selection Logic

The model selects tools based on:
1. **Description matching**: Query keywords vs tool description
2. **Capability matching**: What the tool can do vs what's needed
3. **Specificity**: More specific tools preferred when applicable

```python
# Query: "What's 15% of 84.50?"
# → Model selects: calculate(expression="0.15 * 84.50")

# Query: "What's the weather in Paris and what's 20% of the temperature?"
# → Model might call both tools sequentially
```

---

## Parallel Tool Calls

Some models can request multiple tools at once:

```python
def run_agent_parallel(user_message: str) -> str:
    """Agent that handles parallel tool calls."""

    messages = [{"role": "user", "content": user_message}]

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )

    while response.stop_reason == "tool_use":
        tool_calls = [b for b in response.content if b.type == "tool_use"]

        messages.append({"role": "assistant", "content": response.content})

        # Execute ALL tool calls (potentially in parallel)
        tool_results = []
        for tool_call in tool_calls:
            result = execute_tool(tool_call.name, tool_call.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": result
            })

        messages.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )

    return "".join(b.text for b in response.content if hasattr(b, "text"))


# Query: "What's the weather in Tokyo and New York?"
# Model might call get_weather twice in parallel
```

### Truly Parallel Execution

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor


async def execute_tools_parallel(tool_calls: list) -> list:
    """Execute multiple tool calls in parallel."""

    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor() as executor:
        tasks = [
            loop.run_in_executor(
                executor,
                execute_tool,
                tc.name,
                tc.input
            )
            for tc in tool_calls
        ]

        results = await asyncio.gather(*tasks)

    return [
        {
            "type": "tool_result",
            "tool_use_id": tc.id,
            "content": result
        }
        for tc, result in zip(tool_calls, results)
    ]
```

---

## Tool Choice Control

### Forcing Tool Use

```python
# Force the model to use a specific tool
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    tool_choice={"type": "tool", "name": "get_weather"},  # Force this tool
    messages=[{"role": "user", "content": "Tell me about Paris"}]
)
```

### Allowing Any Tool

```python
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    tool_choice={"type": "auto"},  # Model decides (default)
    messages=[{"role": "user", "content": user_message}]
)
```

### Requiring Some Tool

```python
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    tool_choice={"type": "any"},  # Must use at least one tool
    messages=[{"role": "user", "content": user_message}]
)
```

---

## Understanding Stop Reasons

```python
response = client.messages.create(...)

if response.stop_reason == "end_turn":
    # Model finished responding, no tool needed
    pass
elif response.stop_reason == "tool_use":
    # Model wants to use a tool
    pass
elif response.stop_reason == "max_tokens":
    # Response was truncated
    pass
elif response.stop_reason == "stop_sequence":
    # Hit a stop sequence
    pass
```

---

## Streaming with Tools

```python
def stream_with_tools(user_message: str):
    """Stream responses while handling tool calls."""

    messages = [{"role": "user", "content": user_message}]

    with client.messages.stream(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=tools,
        messages=messages
    ) as stream:
        for event in stream:
            if event.type == "content_block_delta":
                if hasattr(event.delta, "text"):
                    print(event.delta.text, end="", flush=True)

        response = stream.get_final_message()

    # Handle tool calls if any
    if response.stop_reason == "tool_use":
        # ... execute tools and continue
        pass
```

---

## Tool Result Formatting

### Success Results

```python
# Return structured data
tool_results.append({
    "type": "tool_result",
    "tool_use_id": tool_call.id,
    "content": json.dumps({
        "status": "success",
        "data": {"temperature": 22, "conditions": "sunny"}
    })
})
```

### Error Results

```python
# Return error information
tool_results.append({
    "type": "tool_result",
    "tool_use_id": tool_call.id,
    "content": json.dumps({
        "status": "error",
        "error": "City not found",
        "suggestion": "Check spelling or try a nearby major city"
    })
})
```

### Large Results

```python
def truncate_result(result: str, max_length: int = 10000) -> str:
    """Truncate large tool results to avoid context overflow."""
    if len(result) <= max_length:
        return result

    return result[:max_length] + f"\n\n[Truncated: {len(result) - max_length} characters omitted]"
```

---

## Complete Agent Example

```python
class ToolAgent:
    """A complete tool-using agent."""

    def __init__(self, tools: list, max_turns: int = 10):
        self.client = anthropic.Anthropic()
        self.tools = tools
        self.max_turns = max_turns
        self.tool_handlers = {}

    def register_handler(self, tool_name: str, handler: callable):
        """Register a function to handle a tool."""
        self.tool_handlers[tool_name] = handler

    def execute_tool(self, name: str, arguments: dict) -> str:
        """Execute a registered tool."""
        if name in self.tool_handlers:
            try:
                result = self.tool_handlers[name](**arguments)
                return json.dumps(result)
            except Exception as e:
                return json.dumps({"error": str(e)})
        return json.dumps({"error": f"Unknown tool: {name}"})

    def run(self, user_message: str, system_prompt: str = None) -> str:
        """Run the agent on a user message."""
        messages = [{"role": "user", "content": user_message}]

        for _ in range(self.max_turns):
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4096,
                system=system_prompt,
                tools=self.tools,
                messages=messages
            )

            if response.stop_reason != "tool_use":
                return "".join(
                    b.text for b in response.content if hasattr(b, "text")
                )

            # Handle tool calls
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = self.execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            messages.append({"role": "user", "content": tool_results})

        return "Max turns reached"


# Usage
agent = ToolAgent(tools)
agent.register_handler("get_weather", lambda city, units="celsius": {
    "city": city,
    "temperature": 22 if units == "celsius" else 72,
    "conditions": "sunny"
})
agent.register_handler("calculate", lambda expression: {
    "result": eval(expression)  # Note: eval is dangerous! Use a safe parser
})

result = agent.run("What's the weather in London and what's 15% of 85?")
```

---

*Tool calling transforms language models from conversationalists into actors. Master this fundamental pattern—every advanced agent capability builds on it.*
