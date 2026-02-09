# Module 00: The AI Engineer Mindset

> **"An LLM is not a model. It's a policy + a world interface."**

## Learning Objectives

By the end of this module, you will:
- Understand why "just scale the Transformer" isn't enough
- Internalize the closed-loop mental model for LLM systems
- Know the three tracks of AI Engineering (Model, Alignment, Agents)
- Be ready to navigate the rest of this course strategically

## The Core Insight

Most people meet LLMs through a clean Transformer diagram and think:
> "If you scale parameters and data, intelligence just happens."

Then they try to build something real and discover:
- Hallucinations that look confident
- Knowledge that's outdated the day you deploy
- Tool calls that fail silently
- Users who judge you by one bad answer
- Costs that explode with success

**The uncomfortable truth:** Modern LLMs are not a single model. They are a stack.

## The Full Loop

```
┌──────────────────────────────────────────────────────────────────┐
│                     THE CLOSED LOOP                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────┐                                                    │
│   │  GOAL   │ ─────────────────────────────────────────────┐     │
│   └────┬────┘                                              │     │
│        │                                                   │     │
│        ▼                                                   │     │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐             │     │
│   │ CONTEXT │ ──► │  PLAN/  │ ──► │   ACT   │             │     │
│   │ BUILD   │     │GENERATE │     │  TOOLS  │             │     │
│   └─────────┘     └─────────┘     └────┬────┘             │     │
│        ▲                               │                   │     │
│        │                               ▼                   │     │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐             │     │
│   │ UPDATE  │ ◄── │EVALUATE │ ◄── │ OBSERVE │             │     │
│   │ MEMORY  │     │         │     │ RESULTS │             │     │
│   └─────────┘     └────┬────┘     └─────────┘             │     │
│                        │                                   │     │
│                        └───────────────────────────────────┘     │
│                              (iterate)                           │
└──────────────────────────────────────────────────────────────────┘
```

**Whoever runs this loop faster and cleaner wins.**

## Module Contents

### Guides
| Guide | Description | Time |
|-------|-------------|------|
| [01_from_transformer_to_system.md](guides/01_from_transformer_to_system.md) | Why the model is just the beginning | 10 min |
| [02_the_closed_loop.md](guides/02_the_closed_loop.md) | The mental model for modern AI Engineering | 15 min |
| [03_three_tracks.md](guides/03_three_tracks.md) | Model Core / Alignment / Agent Systems | 10 min |
| [cheatsheet.md](guides/cheatsheet.md) | Quick reference for the AI Engineer role | 5 min |

### Notebooks
| Notebook | Description | Time |
|----------|-------------|------|
| [01_your_first_llm_call.ipynb](notebooks/01_your_first_llm_call.ipynb) | Working API call in 2 minutes | 5 min |
| [02_the_full_loop_demo.ipynb](notebooks/02_the_full_loop_demo.ipynb) | See the entire loop in action | 15 min |

### Exercises
Self-check exercises to verify understanding (no grades).

### Resources
- [additional_readings.md](resources/additional_readings.md) - Papers and articles for deeper exploration
- [figures/](resources/figures/) - Diagrams and visual assets

## Key Concepts

### The Stack (Not Just a Model)

| Layer | What It Does | Example |
|-------|--------------|---------|
| **Transformer** | Generates text, reasons in token space | GPT, Claude base |
| **Alignment** | Shapes behavior to be helpful/safe | SFT, RLHF, DPO |
| **Memory** | Provides updated, relevant context | RAG, long-term stores |
| **Tools** | Takes actions beyond text | API calls, code exec |
| **Protocols** | Standardizes tool integration | MCP |
| **Evaluation** | Measures progress, prevents regression | Benchmarks + custom |

### Three Types of AI Engineers

1. **Model & Training Core** (Track A)
   - Deep learning fundamentals
   - Transformer architecture
   - Training recipes and data quality
   - Distributed training

2. **Alignment & Safety** (Track B)
   - Instruction tuning (SFT)
   - Preference learning (RLHF, DPO)
   - Safety policies and red-teaming
   - Evaluation methodology

3. **Agent Systems** (Track C)
   - RAG systems
   - Memory management
   - Tool use and error handling
   - Protocols and integration

**The best AI Engineers can do all three.**

## The Chatbot vs System Distinction

| Chatbot | System |
|---------|--------|
| Takes prompt, returns text | Takes goal, achieves outcome |
| Stateless | Maintains memory |
| Text-only | Uses tools |
| Single inference | Loop until done |
| Hopes it's right | Verifies and corrects |

## Prerequisites

- Python proficiency
- Basic understanding of neural networks
- API access (Claude or OpenAI)

## Next Steps

After this module:
- **Want to understand the engine?** → Module 01: Transformer Fundamentals
- **Want to shape behavior?** → Module 02: Alignment
- **Want to build agents now?** → Module 03: Memory Systems

## Time Estimate

- Quick path: 30 minutes (notebooks only)
- Full path: 1-2 hours (guides + notebooks)

---

*"The future won't reward the person who can draw the attention diagram. It will reward the person who can build a system that keeps getting better after it ships."*
