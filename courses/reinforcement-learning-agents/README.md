# Reinforcement Learning for AI Agents

## Course Overview

Train LLM-based agents that learn from experience using reinforcement fine-tuning (RFT). Move beyond prompt engineering and supervised fine-tuning to build agents that self-improve through trial and error using GRPO, the ART framework, and automatic reward functions.

**Level:** Advanced
**Prerequisites:** Python proficiency, LLM API experience, basic ML knowledge
**Duration:** 8 modules (8-10 weeks)
**Effort:** 8-10 hours per week

## Why Reinforcement Learning for Agents?

Supervised fine-tuning teaches models *what to say*. Reinforcement learning teaches them *how to succeed*.

For agents that search, call APIs, and reason across multiple steps, imitation is not enough. You need self-improvement through trial and error:

- **SFT** = studying a textbook (memorizing answers to known questions)
- **RL** = on-the-job training (learning from trial, error, and feedback)

A 14B open-source model trained with RL on a single GPU for under $80 outperformed OpenAI's o3 on a realistic agentic task — 5x faster and 64x cheaper.

## Learning Outcomes

By completing this course, you will:

1. **Explain** why SFT fails for multi-step agentic workflows and when RL is needed
2. **Implement** GRPO-based reinforcement learning from scratch
3. **Build** the ART client-backend architecture for agent training
4. **Design** automatic reward functions using RULER's LLM-as-a-judge approach
5. **Create** MCP tool servers with FastMCP for agent environments
6. **Train** a text-to-SQL agent that learns to master database queries through RL
7. **Evaluate** cost, speed, and accuracy tradeoffs for production deployment

## Course Structure

| Module | Topic | Key Skills |
|--------|-------|------------|
| 0 | Foundations | SFT vs RL, policy optimization basics, reward signals |
| 1 | GRPO Algorithm | Group sampling, relative advantage, policy updates |
| 2 | ART Framework | Client/backend architecture, vLLM, Unsloth, LoRA |
| 3 | RULER Rewards | LLM-as-a-judge, relative scoring, automatic reward generation |
| 4 | MCP Integration | FastMCP servers, tool discovery, agent environments |
| 5 | Training Loop | Rollouts, trajectories, checkpoint management |
| 6 | Text-to-SQL Agent | End-to-end RL training on a real database |
| 7 | Production | Cost optimization, benchmarking, deployment patterns |

## Technology Stack

**Core Frameworks:**
- ART (OpenPipe) — agent reinforcement training
- Unsloth — efficient GRPO training
- vLLM — fast inference serving

**Models:**
- Qwen 2.5 (3B, 7B, 14B)
- Llama 3.x
- Any vLLM-compatible model

**Tools:**
- FastMCP — tool server creation
- SQLite — database environments
- RULER — automatic reward functions

## Quick Start

```bash
# Install dependencies
pip install art-trainer vllm unsloth fastmcp

# Run your first RL training loop
cd quick-starts/
jupyter notebook 01_first_rl_loop.ipynb
```

## Project Structure

```
reinforcement-learning-agents/
├── modules/
│   ├── module_00_foundations/       # SFT vs RL, reward signals
│   ├── module_01_grpo_algorithm/   # GRPO deep-dive
│   ├── module_02_art_framework/    # ART architecture
│   ├── module_03_ruler_rewards/    # Automatic reward functions
│   ├── module_04_mcp_integration/  # MCP tool servers
│   ├── module_05_training_loop/    # Full training pipeline
│   ├── module_06_text_to_sql_agent/# End-to-end project
│   └── module_07_production/       # Deployment & optimization
├── quick-starts/                   # Entry-point notebooks (<2 min)
├── templates/                      # Production-ready scaffolds
├── recipes/                        # Copy-paste code patterns
└── projects/                       # Portfolio projects
```
