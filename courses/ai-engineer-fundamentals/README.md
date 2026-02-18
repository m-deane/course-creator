# AI Engineer Fundamentals: Building Production LLM Systems

> **"Transformer is just the engine. The winners will build the full loop: alignment, memory, tools, protocols, systems — and the evaluation discipline that keeps everything honest."**

## Start Here

This course transforms you from "Transformer diagram reader" into a **closed-loop engineer** who can build, deploy, and maintain production LLM systems.

**Time to first working code:** 2 minutes
**Total course duration:** 8 modules, self-paced
**Prerequisites:** Python proficiency, basic ML concepts

## The Core Insight

Most people think an LLM is a model. It's not.

**An LLM is a system:**
```
Goal → Context → Plan/Generate → Act/Call Tools → Observe Results → Update Memory → Evaluate → Iterate
```

Whoever runs this loop **faster and cleaner** wins.

## Quick Start (Choose Your Path)

| If you want to... | Start with... |
|-------------------|---------------|
| Get something working NOW | [00_your_first_llm_call.ipynb](quick-starts/00_your_first_llm_call.ipynb) |
| Understand the big picture | [Module 00: AI Engineer Mindset](modules/module_00_ai_engineer_mindset/) |
| Build a RAG system | [02_rag_in_5_minutes.ipynb](quick-starts/02_rag_in_5_minutes.ipynb) |
| Create an agent | [03_your_first_agent.ipynb](quick-starts/03_your_first_agent.ipynb) |
| Deploy to production | [Module 08: Production Systems](modules/module_08_production_systems/) |

## The Full LLM Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRODUCTION LLM SYSTEM                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ EVALUATION  │  │ OBSERVABILITY│  │ FEEDBACK FLYWHEEL      │  │
│  │ & TESTING   │  │ & MONITORING │  │ (continuous improvement)│  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
│         │                │                      │                │
│  ┌──────┴────────────────┴──────────────────────┴─────────────┐  │
│  │                    AGENT ORCHESTRATION                      │  │
│  │         (goal interpretation, planning, execution)          │  │
│  └──────┬────────────────┬──────────────────────┬─────────────┘  │
│         │                │                      │                │
│  ┌──────┴──────┐  ┌──────┴──────┐  ┌───────────┴─────────────┐  │
│  │   MEMORY    │  │    TOOLS    │  │      PROTOCOLS          │  │
│  │ RAG + Long  │  │  APIs, DBs  │  │   (MCP, Standards)      │  │
│  │   Term      │  │  Code Exec  │  │                         │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
│         │                │                      │                │
│  ┌──────┴────────────────┴──────────────────────┴─────────────┐  │
│  │                      ALIGNMENT                              │  │
│  │            (SFT → RLHF/DPO → Safety Policies)              │  │
│  └──────────────────────────┬─────────────────────────────────┘  │
│                             │                                    │
│  ┌──────────────────────────┴─────────────────────────────────┐  │
│  │                     TRANSFORMER                             │  │
│  │    (the differentiable reasoning engine - NOT the product) │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Course Modules

### Track A: Model & Training Core
| Module | Topic | Key Outcome |
|--------|-------|-------------|
| 00 | [AI Engineer Mindset](modules/module_00_ai_engineer_mindset/) | Understand the full loop |
| 01 | [Transformer Fundamentals](modules/module_01_transformer_fundamentals/) | Build attention from scratch |
| 06 | [Efficiency](modules/module_06_efficiency/) | LoRA, quantization, FlashAttention |

### Track B: Alignment & Safety
| Module | Topic | Key Outcome |
|--------|-------|-------------|
| 02 | [Alignment](modules/module_02_alignment/) | SFT, RLHF, DPO implementation |
| 07 | [Evaluation](modules/module_07_evaluation/) | Build evaluation harnesses |

### Track C: Agent Systems
| Module | Topic | Key Outcome |
|--------|-------|-------------|
| 03 | [Memory Systems](modules/module_03_memory_systems/) | RAG + long-term memory |
| 04 | [Tool Use](modules/module_04_tool_use/) | Build multi-tool agents |
| 05 | [Protocols (MCP)](modules/module_05_protocols_mcp/) | Standard tool integration |
| 08 | [Production Systems](modules/module_08_production_systems/) | Deploy the full loop |

## Portfolio Projects

Build real things that demonstrate mastery:

1. **[Domain-Specific RAG Chatbot](projects/project_1_rag_chatbot/)** - Beginner
2. **[Multi-Tool Agent](projects/project_2_multi_tool_agent/)** - Intermediate
3. **[Production LLM System](projects/project_3_production_system/)** - Advanced

## Canonical Reading List

Every paper that shaped modern LLM systems, with summaries:
→ [Paper Summaries & Reading Guide](resources/paper_summaries.md)

## Setup

```bash
# Clone and setup
git clone <repo>
cd ai-engineer-fundamentals

# Create environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Set API keys
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"  # optional
```

## Philosophy

This course follows the **practical-first** methodology:

1. **Working code in 2 minutes** - Every concept starts with runnable code
2. **Visual-first explanations** - Diagram before text, always
3. **Copy-paste ready** - All code works in your own projects
4. **15-minute max** - No notebook longer than 15 minutes
5. **Portfolio over grades** - Build real things, not pass tests

## The Closed-Loop Engineer

By the end of this course, you'll be able to:

- [ ] Build and deploy a Transformer-based system
- [ ] Align models using SFT, RLHF, or DPO
- [ ] Implement RAG with production-grade retrieval
- [ ] Create agents that use tools reliably
- [ ] Connect systems via MCP protocols
- [ ] Optimize for cost and latency
- [ ] Evaluate beyond benchmarks
- [ ] Ship systems that improve after deployment

**The future rewards the person who can build a system that keeps getting better after it ships.**

---

*Based on the "Closed-Loop Engineer" framework. See [resources/paper_summaries.md](resources/paper_summaries.md) for the research foundation.*
