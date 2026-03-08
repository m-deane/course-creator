# Module 04: MCP Server Integration

## Overview

The Model Context Protocol (MCP) standardizes how agents discover and call tools. Instead of hard-coding tool interfaces into your training environment, you expose a local MCP server — and the agent learns to use any tool it finds there, just by inspecting the schema.

This module covers the full MCP integration path for RL agent training: building a FastMCP database server, connecting the ART framework to it, and auto-generating diverse training scenarios from the tool schemas themselves.

## Learning Objectives

By completing this module, you will be able to:

1. Explain what MCP is and why standardized tool interfaces accelerate agent training
2. Build a working MCP server with FastMCP exposing database tools
3. Connect an ART-based agent to an MCP server and inspect discovered tools
4. Auto-generate training scenarios from tool schemas using an LLM
5. Design multi-step scenarios that require chaining multiple tools in sequence
6. Control scenario diversity and difficulty for effective curriculum learning

## Prerequisites

- Module 02: ART Framework (agent client architecture, rollout format)
- Module 03: RULER Rewards (reward functions, LLM-as-a-judge)
- Python async/await patterns
- Basic SQL knowledge (SELECT, JOIN, GROUP BY)

## Module Contents

```
module_04_mcp_integration/
├── README.md                                   # This file
├── guides/
│   ├── 01_mcp_overview_guide.md               # What MCP is and why it matters
│   ├── 01_mcp_overview_slides.md              # Companion slides
│   ├── 02_fastmcp_server_guide.md             # Building a database MCP server
│   ├── 02_fastmcp_server_slides.md            # Companion slides
│   ├── 03_scenario_generation_guide.md        # Auto-generating training scenarios
│   └── 03_scenario_generation_slides.md       # Companion slides
├── exercises/
│   └── 01_mcp_server_exercise.py              # Self-check: build your own MCP server
└── resources/
    └── additional_readings.md                  # FastMCP docs, MCP protocol spec
```

## Estimated Time

- Guides: 45-60 minutes total
- Exercise: 30-45 minutes
- Total: approximately 90 minutes (split across 2 sessions)

## Key Concepts

| Concept | What It Means |
|---------|---------------|
| MCP (Model Context Protocol) | Anthropic's open standard for agent-tool communication |
| FastMCP | Python library for building MCP servers with minimal boilerplate |
| Tool discovery | Agent queries the server to list available tools and their schemas |
| Schema inspection | Agent reads parameter types and descriptions before calling a tool |
| Scenario generation | Using an LLM to create diverse training tasks from tool schemas |
| Curriculum diversity | Mixing single-tool, multi-tool, and edge-case scenarios for robust training |

## Connection to the Course

```
Module 02 (ART)    → provides the agent training architecture
Module 03 (RULER)  → provides the reward function for scoring tool use
Module 04 (MCP)    → provides the tool environment the agent learns to navigate
Module 05 (Loop)   → assembles ART + RULER + MCP into a complete training run
Module 06 (SQL)    → applies everything to train a text-to-SQL agent end-to-end
```

MCP is the training environment layer. Without a standardized tool interface, every new environment requires custom integration code. With MCP, the agent's tool-use skills transfer across any MCP-compatible server.

## Core Pattern

```python
# 1. Start your MCP server
server = DatabaseMCPServer("company.db")
server.start()

# 2. Connect ART agent — tools discovered automatically
agent = art.Agent(model="Qwen/Qwen2.5-7B-Instruct", backend=backend)
tools = await agent.discover_tools(server.url)

# 3. Generate scenarios from the schemas
scenarios = await generate_scenarios(tools, n=100)

# 4. Train — agent learns to chain tools through RL
await art.train(agent, scenarios, reward_fn=sql_reward)
```
