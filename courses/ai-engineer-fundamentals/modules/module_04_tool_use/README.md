# Module 04: Tool Use - The Agent Loop

> **"Tool use is how you move from 'chatbot' to 'system'."**

## Learning Objectives

By the end of this module, you will:
- Understand the ReAct pattern (Reason + Act)
- Build agents that reliably use external tools
- Implement error handling and recovery strategies
- Design tools that LLMs can use effectively
- Orchestrate multi-agent systems

## The Core Insight

A pure LLM produces text. A useful system **changes the world**.

That requires tools: search, databases, code execution, APIs, file systems.

The agentic pattern is:
```
┌─────────────────────────────────────────────────────────────────┐
│                     THE AGENT LOOP                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐               │
│   │INTERPRET │────►│  DECIDE  │────►│   CALL   │               │
│   │   GOAL   │     │ TOOL?    │     │   TOOL   │               │
│   └──────────┘     └────┬─────┘     └────┬─────┘               │
│                         │                 │                     │
│                    No   │                 │                     │
│                         ▼                 ▼                     │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐               │
│   │ RESPOND  │◄────│  UPDATE  │◄────│  READ    │               │
│   │          │     │   PLAN   │     │  RESULT  │               │
│   └──────────┘     └──────────┘     └──────────┘               │
│        ▲                                   │                    │
│        └───────────────────────────────────┘                    │
│                    (iterate until done)                         │
└─────────────────────────────────────────────────────────────────┘
```

## Why Tool Use Matters

| Chatbot | Agent with Tools |
|---------|------------------|
| "I think the weather is..." | *Actually checks weather API* |
| "You could try this code..." | *Executes code, shows result* |
| "The answer might be..." | *Searches database, cites source* |
| "I'll try to remember..." | *Writes to persistent storage* |

**Tool use creates a new kind of intelligence:**
- **Closed-loop correction** - Tools provide ground truth feedback
- **Stateful workflows** - Multi-step tasks with real progress
- **Real-world reliability** - Less guessing, more checking

## Module Contents

### Guides
| Guide | Description | Time |
|-------|-------------|------|
| [01_agent_loop_guide.md](guides/01_agent_loop_guide.md) | The fundamental Interpret→Decide→Act→Observe loop | 15 min |
| [02_react_pattern_guide.md](guides/02_react_pattern_guide.md) | Reasoning + Acting for interpretable agents | 15 min |
| [03_tool_design_guide.md](guides/03_tool_design_guide.md) | How to design tools LLMs can use reliably | 15 min |
| [04_error_handling_guide.md](guides/04_error_handling_guide.md) | Retry logic, fallbacks, graceful degradation | 15 min |
| [05_multi_agent_guide.md](guides/05_multi_agent_guide.md) | Patterns for agent coordination | 15 min |
| [cheatsheet.md](guides/cheatsheet.md) | Quick reference for agent patterns | 5 min |

### Notebooks
| Notebook | Description | Time |
|----------|-------------|------|
| [01_first_tool_call.ipynb](notebooks/01_first_tool_call.ipynb) | Make your first tool call with Claude | 10 min |
| [02_react_implementation.ipynb](notebooks/02_react_implementation.ipynb) | Build a ReAct agent from scratch | 15 min |
| [03_multi_tool_agent.ipynb](notebooks/03_multi_tool_agent.ipynb) | Agent with multiple tools | 15 min |

### Exercises
Self-check exercises for agent development.

### Resources
- [additional_readings.md](resources/additional_readings.md) - ReAct, Toolformer papers
- [figures/](resources/figures/) - Agent architecture diagrams

## Key Concepts

### The ReAct Pattern

```
Thought: I need to find the current stock price of AAPL
Action: get_stock_price(symbol="AAPL")
Observation: {"price": 178.52, "currency": "USD", "timestamp": "2024-01-15T10:30:00Z"}
Thought: Now I have the price. The user also asked about the 52-week high...
Action: get_stock_info(symbol="AAPL")
Observation: {"52_week_high": 199.62, "52_week_low": 164.08, ...}
Thought: I now have all the information to answer the question.
Action: respond(message="AAPL is currently trading at $178.52...")
```

### Tool Definition

```python
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location. Use when user asks about weather conditions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or coordinates"
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature units"
                }
            },
            "required": ["location"]
        }
    }
]
```

### Error Handling Patterns

| Pattern | Use Case | Implementation |
|---------|----------|----------------|
| **Retry with backoff** | Transient failures | Exponential backoff, max 3 attempts |
| **Fallback tools** | Primary tool fails | Try alternative data source |
| **Graceful degradation** | Tool unavailable | Acknowledge limitation, proceed |
| **Human escalation** | Critical decision | Pause and request approval |

## Templates

```
templates/
├── single_agent_template.py     # Basic agent with tool registry
├── tool_definition_template.py  # Standard tool interface
├── multi_agent_template.py      # Coordinator + specialists
└── error_handler_template.py    # Robust error handling
```

## Prerequisites

- Module 00: AI Engineer Mindset
- Module 03: Memory Systems (recommended)
- Working Claude API key

## Next Steps

After this module:
- **Need standardized tool integration?** → Module 05: Protocols (MCP)
- **Ready to deploy?** → Module 08: Production Systems
- **Want to evaluate agents?** → Module 07: Evaluation

## Time Estimate

- Quick path: 45 minutes (notebooks only)
- Full path: 2-3 hours (guides + notebooks + exercises)

---

*"A pure LLM produces text. A useful system changes the world."*
