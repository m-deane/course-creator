# MCP Overview: Standardized Tool Interfaces for Agents

## In Brief

The Model Context Protocol (MCP) is Anthropic's open standard that lets agents discover and call tools through a consistent interface. Instead of building custom connectors for every data source or service, you expose one MCP server — and any MCP-compatible agent can immediately discover what tools are available and how to use them.

---

## The Problem MCP Solves

Without a standard, every agent-tool integration is custom:

```
Agent A  ─── custom connector ──► Database
Agent A  ─── custom connector ──► Weather API
Agent A  ─── custom connector ──► File System
Agent B  ─── different connector ► Database   (duplicate work)
```

With MCP, you build the server once:

```
Database ──► MCP Server
                │
                ├─── Agent A (discovers tools automatically)
                ├─── Agent B (same interface)
                └─── Agent C (any MCP-compatible client)
```

The agent does not need to know in advance what tools exist. It queries the server, reads the schemas, and starts calling tools — all through the same protocol.

---

## MCP Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MCP Architecture                      │
│                                                         │
│  ┌──────────────┐      MCP Protocol      ┌───────────┐  │
│  │  MCP Client  │◄──────────────────────►│ MCP Server│  │
│  │  (Agent)     │                        │           │  │
│  │              │  1. list_tools()       │  Tools:   │  │
│  │  Discovers   │◄──────────────────────►│  - list_  │  │
│  │  tools       │  2. call_tool(name,    │    tables │  │
│  │  dynamically │     args)              │  - run_   │  │
│  │              │◄──────────────────────►│    query  │  │
│  └──────────────┘                        └───────────┘  │
│                                               │          │
│                                          ┌────▼──────┐   │
│                                          │ SQLite DB │   │
│                                          └───────────┘   │
└─────────────────────────────────────────────────────────┘
```

The protocol defines three core interactions:

1. **List tools** — client asks server what tools exist (names, descriptions, parameter schemas)
2. **Describe tool** — client reads the full schema for a specific tool
3. **Call tool** — client invokes a tool with arguments, receives a structured result

---

## Why Agents Need Standardized Tool Interfaces

### The Training Environment Problem

Training an RL agent to use tools requires thousands of rollouts. Each rollout, the agent must:
1. Understand what tools are available
2. Choose the right tool for the current step
3. Formulate valid arguments
4. Interpret the result and decide what to do next

If the tool interface changes between training runs — different function names, different argument formats, different result structures — the agent's learned behaviors break. MCP eliminates this by making the interface contractual.

### Schema-Driven Generalization

When the agent reads tool schemas, it is not memorizing specific function calls. It is learning to read and interpret any schema. This generalization is what allows a model trained on one MCP server to rapidly adapt to a new one.

```json
{
  "name": "run_query",
  "description": "Execute a SQL query on the database",
  "inputSchema": {
    "type": "object",
    "properties": {
      "sql": {
        "type": "string",
        "description": "The SQL query to execute"
      }
    },
    "required": ["sql"]
  }
}
```

The agent reads this schema the same way a developer reads API documentation — and uses that understanding to construct valid calls.

---

## Tool Discovery: How Agents Learn What Tools Exist

Tool discovery is the first thing an agent does when it connects to an MCP server. The sequence:

```
Agent connects to server
    │
    ▼
Agent sends: list_tools request
    │
    ▼
Server responds with tool list:
    ├── name: "list_tables"
    │   description: "Return all table names in the database"
    │   parameters: (none required)
    │
    ├── name: "describe_table"
    │   description: "Return column names and types for a table"
    │   parameters: table_name (string, required)
    │
    └── name: "run_query"
        description: "Execute a SQL SELECT query"
        parameters: sql (string, required)
    │
    ▼
Agent builds internal tool registry
    │
    ▼
Agent uses tools in reasoning
```

For RL training, this discovery step is part of every episode. The agent must internalize: "before I can query data, I need to know what tables exist and what columns they have."

---

## Schema Inspection: Understanding Tool Parameters

Schema inspection is more granular than tool discovery. After listing tools, the agent reads the full parameter schema for each tool before calling it. This prevents calling tools with wrong argument types or missing required fields.

Example inspection flow for a database scenario:

```
Task: "What is the average salary by department?"

Step 1: Call list_tables()
  → ["employees", "departments", "projects"]

Step 2: Call describe_table("employees")
  → [("id", "INTEGER"), ("name", "TEXT"),
     ("dept_id", "INTEGER"), ("salary", "REAL")]

Step 3: Call describe_table("departments")
  → [("id", "INTEGER"), ("name", "TEXT"), ("manager_id", "INTEGER")]

Step 4: Now I can write the JOIN query
  → Call run_query("SELECT d.name, AVG(e.salary)
                    FROM employees e
                    JOIN departments d ON e.dept_id = d.id
                    GROUP BY d.name")
```

Each schema inspection call adds information that enables the next step. This multi-step reasoning pattern is exactly what RL training teaches the agent to execute reliably.

---

## MCP Server vs Client Architecture

### Server Responsibilities

The MCP server owns the tools and the environment:

```python
# Server defines what tools exist and what they do
@server.tool()
def list_tables() -> list[str]:
    """Return all table names in the database."""
    ...

@server.tool()
def run_query(sql: str) -> list[dict]:
    """Execute a SQL SELECT query and return results."""
    ...
```

The server is environment-stable across training episodes. It does not change between rollouts — only the agent changes as it learns.

### Client Responsibilities

The MCP client (the agent framework) handles the transport layer:

```python
# Client discovers and calls tools through the protocol
async with mcp.client_session(server_url) as session:
    tools = await session.list_tools()
    result = await session.call_tool("list_tables", {})
```

In the ART framework, the client layer is handled for you — you configure the server URL and ART manages discovery and tool calling on each rollout.

---

## How ART Connects to MCP Servers

ART (Agent Reinforcement Training) has native MCP support. The connection process:

```python
import art
from art.tools import MCPToolset

# Point ART at your MCP server
toolset = MCPToolset(server_url="http://localhost:8000")

# ART discovers tools automatically on first connect
agent = art.TrainableAgent(
    model="Qwen/Qwen2.5-7B-Instruct",
    toolset=toolset,
)

# During each rollout, the agent can call any discovered tool
# ART formats tool results back into the conversation
trajectory = await agent.rollout(scenario)
```

The key property: the agent sees exactly what a human would see when reading the tool schemas. Training teaches the agent to go from schema → reasoning → correct tool calls — the same generalization that makes trained agents useful on new problems.

---

## Common Pitfalls

- **Skipping describe_table before run_query:** Agents that do not inspect schemas write SQL with wrong column names. Training must include scenarios that reward the full inspection sequence.
- **Wide-open query permissions:** For training, restrict `run_query` to SELECT statements. Agents can discover injection-style reward hacks if UPDATE/DELETE are available.
- **Unstable server state:** If the database schema changes between training episodes, the agent cannot build stable tool-use patterns. Keep the training database fixed.
- **Missing tool descriptions:** Empty or vague `description` fields in the schema make tool selection much harder. Write descriptions as if they are the only documentation the agent will ever see.

---

## Connections

- **Builds on:** Module 02 (ART Framework) — agent rollout architecture, tool call formatting
- **Builds on:** Module 03 (RULER Rewards) — reward functions that evaluate tool use quality
- **Leads to:** Guide 02 (FastMCP Server) — building the actual database server
- **Leads to:** Guide 03 (Scenario Generation) — creating training tasks from the schemas
- **Applied in:** Module 06 (Text-to-SQL Agent) — end-to-end training with a real database

---

## Further Reading

See `resources/additional_readings.md` for the MCP specification, FastMCP documentation, and related papers.
