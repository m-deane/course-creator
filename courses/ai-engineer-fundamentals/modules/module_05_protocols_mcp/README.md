# Module 05: Protocols - MCP and the Integration Layer

> **"If Transformers are the 'brain', then MCP is the 'bus system' connecting the brain to the environment."**

## Learning Objectives

By the end of this module, you will:
- Understand why protocols matter for AI systems
- Build MCP servers in Python and TypeScript
- Connect agents to multiple MCP servers
- Design tools and resources following MCP patterns
- Handle security, permissions, and error cases

## The Core Insight

Once your agent uses tools, you hit a scaling problem:

```
Without protocols:     With protocols (MCP):
┌─────────┐            ┌─────────┐
│ Agent 1 │──┐         │ Agent 1 │───┐
└─────────┘  │         └─────────┘   │
┌─────────┐  │  ┌───┐  ┌─────────┐   │   ┌─────────────┐
│ Agent 2 │──┼──│API│  │ Agent 2 │───┼───│ MCP Server  │───┌───┐
└─────────┘  │  └───┘  └─────────┘   │   └─────────────┘   │API│
┌─────────┐  │  ┌───┐  ┌─────────┐   │   ┌─────────────┐   └───┘
│ Agent 3 │──┼──│ DB│  │ Agent 3 │───┼───│ MCP Server  │───┌───┐
└─────────┘  │  └───┘  └─────────┘   │   └─────────────┘   │ DB│
             │  ┌───┐                │                     └───┘
             └──│...│                │
                └───┘                └── Standard Protocol

N agents × M tools        N agents × 1 protocol × M servers
= N×M integrations       = N + M integrations
```

**Protocols enable ecosystems.** Just as HTTP enabled the web, MCP enables interoperable AI tools.

## Module Contents

### Guides
| Guide | Description | Time |
|-------|-------------|------|
| [01_why_protocols_matter.md](guides/01_why_protocols_matter.md) | The N×M integration problem | 10 min |
| [02_mcp_architecture.md](guides/02_mcp_architecture.md) | Clients, servers, transports | 15 min |
| [03_building_mcp_servers.md](guides/03_building_mcp_servers.md) | Python (FastMCP) and TypeScript | 20 min |
| [04_security_permissions.md](guides/04_security_permissions.md) | Auth, rate limits, sandboxing | 15 min |
| [05_ecosystem_design.md](guides/05_ecosystem_design.md) | Building for interoperability | 10 min |
| [cheatsheet.md](guides/cheatsheet.md) | Quick reference | 5 min |

### Notebooks
| Notebook | Description | Time |
|----------|-------------|------|
| [01_mcp_hello_world.ipynb](notebooks/01_mcp_hello_world.ipynb) | Connect to an MCP server | 10 min |
| [02_build_mcp_server.ipynb](notebooks/02_build_mcp_server.ipynb) | Create your own server | 15 min |
| [03_multi_server_agent.ipynb](notebooks/03_multi_server_agent.ipynb) | Orchestrate multiple servers | 15 min |

### Templates
| Template | Description |
|----------|-------------|
| `mcp_server_python_template.py` | FastMCP production server |
| `mcp_server_typescript_template.ts` | MCP SDK server |
| `mcp_client_template.py` | Multi-server orchestration |

## Key Concepts

### MCP Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MCP ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────┐          Protocol           ┌────────────┐  │
│  │               │◄────────────────────────────►│            │  │
│  │    CLIENT     │           MCP               │   SERVER   │  │
│  │  (Claude,     │                             │ (Your app, │  │
│  │   Agent)      │◄────────────────────────────►│  DB, API)  │  │
│  │               │         Transport            │            │  │
│  └───────────────┘      (stdio, HTTP)          └────────────┘  │
│         │                                             │         │
│         │ Requests:                                   │         │
│         │ • tools/list                               │         │
│         │ • tools/call                               │ Exposes: │
│         │ • resources/list                           │ • Tools  │
│         │ • resources/read                           │ • Resources│
│         │ • prompts/list                             │ • Prompts │
│         │ • prompts/get                              │         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### MCP Capabilities

| Capability | Description | Use Case |
|------------|-------------|----------|
| **Tools** | Functions the agent can call | Search, calculate, API calls |
| **Resources** | Data the agent can read | Files, database records, configs |
| **Prompts** | Dynamic prompt templates | Domain-specific instructions |
| **Sampling** | Server requests model completion | Agentic sub-tasks |

### Quick Example: Python Server

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server

server = Server("my-server")

@server.tool()
async def get_weather(location: str) -> str:
    """Get weather for a location."""
    # Implementation
    return f"Weather in {location}: Sunny, 22°C"

@server.resource("config://settings")
async def get_settings() -> str:
    """Application settings."""
    return json.dumps({"theme": "dark", "language": "en"})

async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write)

asyncio.run(main())
```

## The N×M Problem Solved

**Before MCP:**
- Every agent needs custom code for every tool
- Authentication handled differently everywhere
- Error handling is inconsistent
- Tool discovery is manual

**With MCP:**
- Standard protocol for all tool interactions
- Consistent authentication patterns
- Unified error handling
- Automatic tool discovery via `tools/list`

## Prerequisites

- Module 04: Tool Use (agent loops)
- Basic understanding of client-server architecture
- Python or TypeScript familiarity

## Next Steps

After this module:
- **Need efficiency?** → Module 06: Efficiency
- **Need evaluation?** → Module 07: Evaluation
- **Ready to deploy?** → Module 08: Production Systems

## Resources

- [Model Context Protocol Documentation](https://modelcontextprotocol.io)
- [MCP Python SDK (FastMCP)](https://github.com/jlowin/fastmcp)
- [MCP TypeScript SDK](https://github.com/anthropics/mcp)

## Time Estimate

- Quick path: 40 minutes (notebooks only)
- Full path: 2 hours (guides + notebooks)

---

*"Protocols reduce integration cost and unlock ecosystems."*
