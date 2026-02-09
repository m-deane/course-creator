# Portfolio Project 2: Multi-Tool Agent

> **Build an agent that can query databases, call APIs, execute code, and maintain memory.**

## What You'll Build

A fully functional agent that can:
- Use multiple tools to accomplish complex tasks
- Maintain conversation memory across turns
- Handle errors gracefully
- Explain its reasoning

**Demo:** "Find the top 5 trending GitHub repos in Python, summarize their READMEs, and create a comparison table."

## Learning Goals

By completing this project, you will demonstrate:
- Agent loop implementation (ReAct pattern)
- Tool design and registration
- Error handling and recovery
- Memory management
- Multi-step task execution

## Requirements

### Functional Requirements
- [ ] Implement at least 5 tools from different categories
- [ ] ReAct-style reasoning traces
- [ ] Persistent conversation memory
- [ ] Graceful error handling with retries
- [ ] Task completion detection

### Required Tools (implement at least 5)

| Category | Tool Options |
|----------|--------------|
| **Search** | Web search, Wikipedia, ArXiv |
| **Data** | SQL query, API calls, file read |
| **Compute** | Calculator, code execution, data analysis |
| **Output** | File write, format converter, chart generator |
| **Memory** | Save note, recall notes, clear memory |

### Technical Requirements
- [ ] Clean agent loop with max iterations
- [ ] Structured tool definitions (JSON schema)
- [ ] Tool execution with timeout and error capture
- [ ] Conversation history management
- [ ] Token budget awareness

### Quality Requirements
- [ ] Task success rate > 80% on test tasks
- [ ] Average steps to completion < 10
- [ ] No infinite loops
- [ ] Clear reasoning traces

## Suggested Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   MULTI-TOOL AGENT ARCHITECTURE                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User Goal                                                      │
│      │                                                          │
│      ▼                                                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                     AGENT CORE                             │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │  │
│  │  │   Planner   │  │  Executor   │  │  Evaluator  │       │  │
│  │  │ (decide     │  │ (call tools)│  │ (check if   │       │  │
│  │  │  next step) │  │             │  │  done)      │       │  │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘       │  │
│  │         │                │                │               │  │
│  │         └────────────────┼────────────────┘               │  │
│  │                          │                                │  │
│  └──────────────────────────┼────────────────────────────────┘  │
│                             │                                   │
│         ┌───────────────────┼───────────────────┐              │
│         ▼                   ▼                   ▼              │
│  ┌───────────┐       ┌───────────┐       ┌───────────┐        │
│  │  Memory   │       │   Tools   │       │  Claude   │        │
│  │  Manager  │       │  Registry │       │   API     │        │
│  └───────────┘       └─────┬─────┘       └───────────┘        │
│                            │                                   │
│      ┌─────────────────────┼─────────────────────┐            │
│      ▼           ▼         ▼         ▼           ▼            │
│  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐       │
│  │Search │  │  SQL  │  │ Code  │  │ File  │  │ API   │       │
│  │       │  │ Query │  │ Exec  │  │ I/O   │  │ Call  │       │
│  └───────┘  └───────┘  └───────┘  └───────┘  └───────┘       │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
project_2_multi_tool_agent/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── agent.py            # Main agent loop
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py         # Tool base class
│   │   ├── search.py       # Search tools
│   │   ├── data.py         # Database/API tools
│   │   ├── compute.py      # Calculator, code exec
│   │   ├── output.py       # File write, formatting
│   │   └── memory.py       # Note-taking tools
│   ├── memory.py           # Conversation memory
│   ├── executor.py         # Tool execution with error handling
│   └── config.py           # Configuration
├── tests/
│   ├── test_agent.py       # Agent behavior tests
│   ├── test_tools.py       # Individual tool tests
│   └── test_tasks/         # Multi-step task tests
├── evaluation/
│   ├── task_suite.py       # Test task definitions
│   └── evaluate.py         # Run evaluation
└── examples/
    └── demo.py             # Interactive demo
```

## Getting Started

### Step 1: Define Tool Interface

```python
# src/tools/base.py - Starter code
from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel

class ToolDefinition(BaseModel):
    """Tool definition for Claude."""
    name: str
    description: str
    input_schema: dict

class Tool(ABC):
    """Base class for all tools."""

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Return tool definition for registration."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> dict:
        """Execute the tool and return result."""
        pass

    def handle_error(self, error: Exception) -> dict:
        """Handle tool execution error."""
        return {
            "success": False,
            "error": str(error),
            "error_type": type(error).__name__
        }
```

### Step 2: Implement Tools

```python
# src/tools/search.py - Starter code
from .base import Tool, ToolDefinition

class WebSearchTool(Tool):
    """Search the web for information."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="web_search",
            description="Search the web for current information. Use for recent events, facts, or when you need up-to-date information.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        )

    async def execute(self, query: str, num_results: int = 5) -> dict:
        """Execute web search."""
        # TODO: Implement using search API (SerpAPI, Tavily, etc.)
        pass
```

### Step 3: Build Agent Loop

```python
# src/agent.py - Starter code
import anthropic
from .tools.base import Tool
from .memory import ConversationMemory
from .executor import ToolExecutor

class Agent:
    def __init__(self, tools: list[Tool], max_iterations: int = 15):
        self.client = anthropic.Anthropic()
        self.tools = {t.definition.name: t for t in tools}
        self.tool_definitions = [t.definition.model_dump() for t in tools]
        self.max_iterations = max_iterations
        self.memory = ConversationMemory()
        self.executor = ToolExecutor()

    async def run(self, user_message: str) -> str:
        """Run agent loop until task completion."""
        messages = self.memory.get_context()
        messages.append({"role": "user", "content": user_message})

        for iteration in range(self.max_iterations):
            # TODO: Get model response
            # TODO: Check if done (no tool use)
            # TODO: Execute tool calls
            # TODO: Add results to messages
            # TODO: Continue loop
            pass

        return "Max iterations reached"

    def _format_tool_result(self, tool_use_id: str, result: dict) -> dict:
        """Format tool result for the model."""
        # TODO: Implement
        pass
```

### Step 4: Implement Memory

```python
# src/memory.py - Starter code
from datetime import datetime

class ConversationMemory:
    """Manage conversation history and working memory."""

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self.messages = []
        self.notes = []  # Agent's saved notes

    def add_message(self, role: str, content: any):
        """Add a message to history."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self._trim_if_needed()

    def get_context(self) -> list:
        """Get messages for context window."""
        # TODO: Return messages, potentially with summarization
        pass

    def save_note(self, note: str):
        """Save a note to working memory."""
        self.notes.append({
            "content": note,
            "timestamp": datetime.now().isoformat()
        })

    def recall_notes(self, query: str = None) -> list:
        """Recall saved notes, optionally filtered by query."""
        # TODO: Implement with optional semantic search
        pass

    def _trim_if_needed(self):
        """Trim old messages if over limit."""
        if len(self.messages) > self.max_turns * 2:
            # TODO: Summarize old messages instead of dropping
            self.messages = self.messages[-self.max_turns * 2:]
```

### Step 5: Add Error Handling

```python
# src/executor.py - Starter code
import asyncio
from typing import Any

class ToolExecutor:
    """Execute tools with error handling and timeouts."""

    def __init__(self, default_timeout: float = 30.0, max_retries: int = 2):
        self.default_timeout = default_timeout
        self.max_retries = max_retries

    async def execute(self, tool, **kwargs) -> dict:
        """Execute a tool with retries and timeout."""
        for attempt in range(self.max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    tool.execute(**kwargs),
                    timeout=self.default_timeout
                )
                return {"success": True, "result": result}
            except asyncio.TimeoutError:
                if attempt == self.max_retries:
                    return {"success": False, "error": "Tool execution timed out"}
            except Exception as e:
                if attempt == self.max_retries:
                    return tool.handle_error(e)
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        return {"success": False, "error": "Unknown error"}
```

## Test Tasks

Your agent should be able to complete these tasks:

### Task 1: Research & Summarize
"Find the top 3 papers on chain-of-thought prompting from ArXiv, summarize each in 2 sentences, and save a note with the key findings."

### Task 2: Data Analysis
"Query the sales database for last month's top products, calculate the total revenue, and create a summary report."

### Task 3: Multi-Step Workflow
"Search for the current weather in Tokyo, convert the temperature to Fahrenheit, and save a note with the full forecast."

### Task 4: Error Recovery
"Try to fetch data from this broken URL [invalid url], handle the error gracefully, and search for alternative sources."

## Evaluation Criteria

| Criterion | Weight | Passing |
|-----------|--------|---------|
| Task success rate | 30% | > 80% on test suite |
| Tool implementation | 20% | 5+ tools working correctly |
| Error handling | 15% | Graceful recovery from failures |
| Memory management | 15% | Context maintained across turns |
| Code quality | 10% | Clean, documented, tested |
| Reasoning traces | 10% | Clear, logical explanations |

## Stretch Goals

- [ ] Add tool chaining (output of one tool as input to another)
- [ ] Implement parallel tool execution
- [ ] Add cost tracking and budget limits
- [ ] Create an MCP server for your tools
- [ ] Add streaming responses
- [ ] Implement human-in-the-loop for sensitive operations

## Resources

- [Module 04: Tool Use](../../modules/module_04_tool_use/)
- [Agent Loop Guide](../../modules/module_04_tool_use/guides/01_agent_loop_guide.md)
- [Paper Summary: ReAct](../../resources/paper_summaries.md#react)

---

*"An agent that can use tools reliably is worth more than a model that can only talk about using them."*
