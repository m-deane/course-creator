# Module 06: Text-to-SQL Agent — End-to-End RL Training

## Overview

This is the hands-on capstone module. You build, connect, and train a complete text-to-SQL agent from scratch using every component covered in previous modules: GRPO (Module 01), the ART framework (Module 02), RULER rewards (Module 03), and MCP tool servers (Module 04).

The agent starts knowing nothing about your specific database schema. By the end of training, it learns to explore the schema autonomously, write correct multi-table JOIN queries, and recover gracefully from SQL syntax errors — all without a single hand-labeled example.

**The result:** a fine-tuned model that outperforms GPT-4 on your specific database at a fraction of the inference cost.

## Learning Objectives

After completing this module, you will be able to:

1. Design a SQLite database with realistic schema and populate it with production-representative data
2. Build a FastMCP server exposing `list_tables`, `describe_table`, and `run_query` tools to an agent
3. Auto-generate training scenarios of varying difficulty directly from tool schemas — no manual labeling
4. Configure a RULER judge (o4-mini) to score agent answers against ground truth
5. Implement the rollout function that wires question → tool calls → agent answer into one trajectory
6. Run a complete GRPO training loop and read its logs to diagnose convergence
7. Compare before-training and after-training agent behavior on held-out queries

## Prerequisites

- Module 00 complete (policy gradients, reward signals)
- Module 01 complete (GRPO algorithm)
- Module 02 complete (ART framework: vLLM backend, Unsloth LoRA)
- Module 03 complete (RULER: LLM-as-a-judge, relative scoring)
- Module 04 complete (FastMCP server creation, tool discovery)
- Module 05 complete (rollout functions, training loop structure)
- SQLite familiarity (basic CREATE TABLE, SELECT, JOIN)
- Python 3.10+, `fastmcp`, `art-trainer`, `openai`

## Guides

| File | Topic | Time |
|------|-------|------|
| `guides/01_database_setup_guide.md` | Create company database: schema design, realistic data, why SQLite | 20 min |
| `guides/02_mcp_server_build_guide.md` | Build FastMCP server, expose three tools, connect ART, discover tools | 25 min |
| `guides/03_training_the_agent_guide.md` | Scenario generation, RULER judge, rollout function, GRPO loop, before/after | 35 min |

Each guide has a companion `_slides.md` for lecture delivery.

## Exercises

| File | Topic |
|------|-------|
| `exercises/01_sql_agent_exercise.py` | Build database, implement tool functions, run agent loop, score responses |

## Module Map

```
Module 06: Text-to-SQL Agent
    │
    ├── Guide 01: Database Setup
    │       Schema design → CREATE TABLE → INSERT realistic data → verify
    │
    ├── Guide 02: MCP Server Build
    │       list_tables() → describe_table() → run_query() → run server → connect ART
    │
    └── Guide 03: Training the Agent
            Scenario generation → RULER judge → rollout function → GRPO loop
            └── Before/after behavior comparison
```

## Connections

- **Builds on:** All previous modules (this is the integration project)
- **Leads to:** Module 07 — Production (cost optimization, benchmarking, deployment)
- **Real-world analog:** OpenPipe's SQL agent that beat o3 on agentic tasks — same architecture you build here

## Time Estimate

- Guides (reading): 80 minutes total
- Slides (lecture): 60 minutes total
- Exercise: 60 minutes
- Total: approximately 3.5 hours
