# Additional Readings — Module 04: MCP Server Integration

## MCP Protocol

### Official Specification

**Model Context Protocol — Specification**
https://spec.modelcontextprotocol.io/

The authoritative specification for the MCP protocol. Covers the complete message format, transport options (stdio, SSE, HTTP streaming), capability negotiation, and tool schema format. Read the "Core Architecture" and "Server Features → Tools" sections first. The rest is reference material.

**MCP GitHub Organization**
https://github.com/modelcontextprotocol

Canonical repositories for the protocol spec, TypeScript SDK, Python SDK, and reference server implementations. The `servers` repository contains production-quality MCP server examples for common tools (filesystem, GitHub, databases).

---

## FastMCP

**FastMCP Documentation**
https://gofastmcp.com/

The primary reference for the FastMCP Python library. The "Getting Started" section covers the decorator pattern used in Guide 02. The "Tools" section covers parameter types, error handling, and return type serialization. The "Servers" section covers transport configuration (SSE vs stdio).

**FastMCP GitHub Repository**
https://github.com/jlowin/fastmcp

Source code and issue tracker. The `examples/` directory contains working MCP servers for common use cases. The test suite is a good reference for edge cases in tool behavior.

**FastMCP Transport Options**
The library supports three transports:
- `stdio` — standard input/output, used by Claude Desktop and local MCP clients
- `sse` — Server-Sent Events over HTTP, best for training environments with persistent connections
- `streamable-http` — HTTP with streaming, for production deployments

For training environments (as built in Guide 02), `sse` is recommended because it maintains a persistent connection across multiple agent rollouts without reconnection overhead.

---

## Automatic Scenario Generation

**Self-play and Automatic Curriculum Learning**
Portelas et al. (2020). "Automatic Curriculum Learning for Deep RL: A Short Survey."
https://arxiv.org/abs/2003.04664

Covers the theoretical foundations for why curriculum learning improves sample efficiency in RL. The "Teacher" algorithms described here are the formal basis for the difficulty-progression approach used in Guide 03. Section 3 (Absolute Learning Progress) is directly applicable to the MCP scenario generation setup.

**LLM-Generated Training Data**
Chung et al. (2022). "Scaling Instruction-Finetuned Language Models."
https://arxiv.org/abs/2210.11416

While focused on instruction tuning rather than RL, Section 4 on dataset generation is relevant: it demonstrates that LLM-generated tasks are diverse enough to produce generalizing models. The diversity analysis methodology translates directly to evaluating scenario generator output.

---

## Text-to-SQL Background

**Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task**
Yu et al. (2018).
https://arxiv.org/abs/1809.08887

The benchmark dataset that established the standard difficulty tiers for text-to-SQL tasks. The "Easy/Medium/Hard/Extra Hard" taxonomy from Spider is the basis for the three-tier scenario classification in Guide 03. Useful for calibrating what "hard" means in practice.

**BIRD: A Big Bench for Large-Scale Database Grounded Text-to-SQLs Evaluation**
Li et al. (2023).
https://arxiv.org/abs/2305.03111

More recent benchmark that focuses on realistic database schemas and noisy conditions. The "external knowledge" category (questions requiring real-world knowledge beyond the schema) points toward the kind of scenarios that generalize to production databases.

---

## ART Framework and GRPO

**ART: Agentic Reasoning and Tool-use (OpenPipe)**
https://github.com/OpenPipe/ART

The training framework used throughout this course. The `examples/` directory includes a complete text-to-SQL training example that connects directly to the patterns in Modules 04-06.

**DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models**
Shao et al. (2024).
https://arxiv.org/abs/2402.03300

The paper that introduced GRPO (covered in Module 01) and demonstrated that RL with minimal infrastructure can match much larger supervised models on structured reasoning tasks. Relevant here because text-to-SQL is a structured reasoning task with the same properties.

---

## SQLite Reference

**SQLite Documentation**
https://www.sqlite.org/docs.html

The authoritative reference for SQLite. For Module 04 specifically:
- "File URIs" section explains the `file:path?mode=ro` pattern used in `get_connection()`
- "PRAGMA Statements" section documents `PRAGMA table_info` used in `describe_table()`
- "Built-in Aggregate Functions" covers AVG, COUNT, SUM, MIN, MAX for scenario generation

**SQLite — How SQLite Is Tested**
https://www.sqlite.org/testing.html

Understanding SQLite's testing approach helps you build reliable training environments. The "100% branch coverage" claim means SQL behavior is predictable across the edge cases that trained agents will encounter.

---

## MCP Integration Patterns

**Building Production MCP Servers**
When moving from training environments (Guide 02 pattern) to production MCP servers, consider:

1. **Authentication**: Production servers need API key or OAuth authentication. FastMCP supports custom middleware for this.
2. **Rate limiting**: Tool calls from concurrent agent rollouts can overload production backends. Add rate limiting at the MCP server level.
3. **Result caching**: For read-only tools (like database queries during inference), caching frequent queries reduces latency.
4. **Schema versioning**: When the underlying database schema changes, update tool docstrings and descriptions before redeploying. Agents trained on old schemas may break silently.

These patterns are covered in Module 07 (Production) in the context of deploying trained agents.
