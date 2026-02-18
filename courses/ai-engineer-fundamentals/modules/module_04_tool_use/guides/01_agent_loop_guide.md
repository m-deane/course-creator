# The Agent Loop: From Text Generation to World Interaction

## In Brief

An agent is an LLM that can take actions in the world, observe results, and iterate until a goal is achieved. The agent loop is the fundamental pattern that makes this possible.

## Key Insight

> **The agent loop transforms a "text predictor" into a "goal achiever" by closing the feedback loop between generation and world state.**

## Visual Explanation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          THE AGENT LOOP                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│     USER GOAL                                                                │
│         │                                                                    │
│         ▼                                                                    │
│   ┌─────────────┐                                                            │
│   │  INTERPRET  │  "What is the user trying to achieve?"                     │
│   └──────┬──────┘                                                            │
│          │                                                                   │
│          ▼                                                                   │
│   ┌─────────────┐    ┌─────────────────────────────────────┐                │
│   │   DECIDE    │───►│ Should I use a tool, or respond?    │                │
│   └──────┬──────┘    │ If tool: which one? with what args? │                │
│          │           └─────────────────────────────────────┘                │
│          │                                                                   │
│     ┌────┴────┐                                                              │
│     │         │                                                              │
│     ▼         ▼                                                              │
│ [Respond]  [Call Tool]                                                       │
│     │         │                                                              │
│     │         ▼                                                              │
│     │   ┌─────────────┐                                                      │
│     │   │   EXECUTE   │  Tool runs in external environment                   │
│     │   └──────┬──────┘                                                      │
│     │          │                                                             │
│     │          ▼                                                             │
│     │   ┌─────────────┐                                                      │
│     │   │   OBSERVE   │  Read tool result/error                              │
│     │   └──────┬──────┘                                                      │
│     │          │                                                             │
│     │          ▼                                                             │
│     │   ┌─────────────┐                                                      │
│     │   │   UPDATE    │  Add observation to context                          │
│     │   └──────┬──────┘                                                      │
│     │          │                                                             │
│     │          └──────────► Loop back to DECIDE                              │
│     │                                                                        │
│     ▼                                                                        │
│  [Final Response to User]                                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## The Loop Components

### 1. Interpret Goal

Transform user input into actionable understanding.

```python
def interpret_goal(user_message: str, context: list) -> str:
    """
    Understand what the user wants to achieve.

    This happens implicitly through the system prompt and
    conversation context, but can be made explicit.
    """
    system_prompt = """You are a helpful assistant with access to tools.

    When the user asks for something:
    1. Identify the core goal
    2. Determine what information or actions are needed
    3. Use tools when they can help achieve the goal
    4. Only respond when you have sufficient information
    """
    return system_prompt
```

### 2. Decide Action

The LLM chooses whether to call a tool, and if so, which one.

```python
import anthropic

client = anthropic.Anthropic()

def decide_action(messages: list, tools: list) -> dict:
    """Let the model decide next action."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )

    # Response contains either text or tool_use
    return {
        "stop_reason": response.stop_reason,
        "content": response.content
    }
```

**The decision process:**
- Model evaluates available tools against current goal
- Considers tool descriptions and parameter schemas
- Chooses the most appropriate action (or responds if no tool needed)

### 3. Execute Tool

Run the selected tool and capture the result.

```python
import json

def execute_tool(tool_name: str, tool_input: dict, tool_registry: dict) -> str:
    """Execute a tool and return the result."""

    if tool_name not in tool_registry:
        return f"Error: Unknown tool '{tool_name}'"

    tool_function = tool_registry[tool_name]

    try:
        result = tool_function(**tool_input)
        return json.dumps(result) if isinstance(result, dict) else str(result)
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"
```

### 4. Observe Result

Feed the tool result back to the model.

```python
def observe_result(tool_use_id: str, result: str) -> dict:
    """Format tool result for the model."""
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result
            }
        ]
    }
```

### 5. Update Context & Iterate

Add the observation to context and let the model decide next steps.

```python
def agent_loop(user_message: str, tools: list, tool_registry: dict, max_iterations: int = 10):
    """Complete agent loop implementation."""

    messages = [{"role": "user", "content": user_message}]

    for iteration in range(max_iterations):
        # Decide action
        response = decide_action(messages, tools)

        # Check if done (text response, no tool call)
        if response["stop_reason"] == "end_turn":
            # Extract final text response
            for block in response["content"]:
                if hasattr(block, "text"):
                    return block.text

        # Process tool calls
        tool_results = []
        for block in response["content"]:
            if block.type == "tool_use":
                # Execute tool
                result = execute_tool(
                    block.name,
                    block.input,
                    tool_registry
                )

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

        # Update messages with assistant response and tool results
        messages.append({"role": "assistant", "content": response["content"]})
        messages.append({"role": "user", "content": tool_results})

    return "Max iterations reached without completion"
```

## Complete Implementation

```python
import anthropic
import json

class Agent:
    def __init__(self, tools: list, tool_registry: dict):
        self.client = anthropic.Anthropic()
        self.tools = tools
        self.tool_registry = tool_registry
        self.system_prompt = """You are a helpful assistant with access to tools.
        Use tools when they help achieve the user's goal.
        Think step by step about what information you need."""

    def run(self, user_message: str, max_iterations: int = 10) -> str:
        """Run the agent loop until completion."""

        messages = [{"role": "user", "content": user_message}]

        for _ in range(max_iterations):
            # Get model response
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=self.system_prompt,
                tools=self.tools,
                messages=messages
            )

            # If no tool use, we're done
            if response.stop_reason == "end_turn":
                return self._extract_text(response.content)

            # Process tool calls
            assistant_content = response.content
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    result = self._execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            # Update conversation
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})

        return "Agent did not complete within iteration limit"

    def _execute_tool(self, name: str, input: dict) -> str:
        """Execute a tool by name."""
        if name not in self.tool_registry:
            return json.dumps({"error": f"Unknown tool: {name}"})

        try:
            result = self.tool_registry[name](**input)
            return json.dumps(result) if isinstance(result, dict) else str(result)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _extract_text(self, content: list) -> str:
        """Extract text from response content."""
        for block in content:
            if hasattr(block, "text"):
                return block.text
        return ""


# Example usage
def get_weather(location: str, units: str = "celsius") -> dict:
    """Mock weather function."""
    return {"location": location, "temperature": 22, "units": units, "conditions": "sunny"}

def search_web(query: str) -> dict:
    """Mock search function."""
    return {"results": [{"title": "Result 1", "snippet": "..."}]}

tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    },
    {
        "name": "search_web",
        "description": "Search the web for information",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    }
]

tool_registry = {
    "get_weather": get_weather,
    "search_web": search_web
}

agent = Agent(tools, tool_registry)
result = agent.run("What's the weather in Tokyo?")
print(result)
```

## Loop Termination

The agent loop should terminate when:

| Condition | What Happens |
|-----------|--------------|
| **Goal achieved** | Model responds with final answer |
| **Max iterations** | Force stop, return partial result |
| **Error threshold** | Too many consecutive errors |
| **User cancellation** | External interrupt signal |
| **Cost limit** | Token budget exhausted |

```python
class LoopController:
    def __init__(self, max_iterations=10, max_errors=3, max_tokens=10000):
        self.max_iterations = max_iterations
        self.max_errors = max_errors
        self.max_tokens = max_tokens
        self.iteration = 0
        self.error_count = 0
        self.token_count = 0

    def should_continue(self, response) -> tuple[bool, str]:
        """Check if loop should continue."""
        self.iteration += 1
        self.token_count += response.usage.total_tokens

        if self.iteration >= self.max_iterations:
            return False, "max_iterations"

        if self.error_count >= self.max_errors:
            return False, "max_errors"

        if self.token_count >= self.max_tokens:
            return False, "max_tokens"

        if response.stop_reason == "end_turn":
            return False, "completed"

        return True, "continue"

    def record_error(self):
        self.error_count += 1

    def reset_errors(self):
        self.error_count = 0
```

## Common Pitfalls

### Pitfall 1: No termination condition
**Problem:** Agent loops forever.
**Solution:** Always set max_iterations and implement explicit exit conditions.

### Pitfall 2: Lost context
**Problem:** Model forgets earlier tool results.
**Solution:** Maintain full message history, consider summarization for long conversations.

### Pitfall 3: Tool description mismatch
**Problem:** Model calls tools incorrectly.
**Solution:** Write clear, specific tool descriptions with examples.

### Pitfall 4: No error handling
**Problem:** Single tool failure crashes the agent.
**Solution:** Wrap tool execution in try/catch, provide error feedback to model.

## Connections

**Builds on:**
- Module 03: Memory Systems (context management)

**Leads to:**
- Guide 02: ReAct Pattern (structured reasoning)
- Guide 03: Tool Design (effective tools)
- Module 05: MCP Protocols (standardized tool integration)

## Practice Problems

1. **Implement:** Build an agent with a calculator tool. Have it solve multi-step math problems.

2. **Debug:** An agent is calling the same tool repeatedly with the same arguments. What's wrong?

3. **Design:** How would you modify the loop to support parallel tool execution?
