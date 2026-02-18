# Module 03: Memory Systems

> **"Memory is an evolving state the agent can read and write."**

## Learning Objectives

By the end of this module, you will:
- Understand the taxonomy of memory types (short-term, external, long-term)
- Build production RAG systems with proper chunking, embedding, and retrieval
- Implement memory operators: formation, retrieval, and evolution
- Know when to use RAG vs fine-tuning vs weight editing
- Deploy memory systems that scale to millions of documents

## The Core Insight

Most people think memory is just "shoving context into the prompt." It's not.

**Memory is a system for managing what the model knows, when it knows it, and how it evolves.**

The Transformer has a fundamental limitation:
```
Knowledge in weights is frozen at training time.
Context windows are finite.
Without memory systems, your agent is blind to:
  - Updated information
  - Long histories
  - User preferences
  - Domain knowledge
```

## Memory Architecture at a Glance

```
┌──────────────────────────────────────────────────────────────────┐
│                    MEMORY SYSTEM LAYERS                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  SHORT-TERM MEMORY (Context Window)                        │  │
│  │  Current conversation, immediate task state                │  │
│  │  Size: 4k-200k tokens  |  Latency: 0ms  |  Cost: $$$      │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  EXTERNAL KNOWLEDGE (RAG)                                  │  │
│  │  Documents, databases, APIs - retrieved on demand          │  │
│  │  Size: Unlimited  |  Latency: 10-100ms  |  Cost: $$       │  │
│  │                                                            │  │
│  │  Query → Embed → Retrieve → Rerank → Inject               │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  LONG-TERM MEMORY (Persistent Stores)                      │  │
│  │  User preferences, learned facts, conversation summaries   │  │
│  │  Size: Unlimited  |  Latency: 50-200ms  |  Cost: $        │  │
│  │                                                            │  │
│  │  Formation → Storage → Retrieval → Evolution               │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  WEIGHT MEMORY (Model Parameters) - RARELY EDITED          │  │
│  │  Baked-in knowledge from pre-training                      │  │
│  │  Size: Fixed  |  Latency: 0ms  |  Cost: $$$$$ (to change) │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**The key question:** Which memory layer should handle each piece of information?

## Module Contents

### Guides
| Guide | Description | Time |
|-------|-------------|------|
| [01_memory_taxonomy_guide.md](guides/01_memory_taxonomy_guide.md) | Memory types and the form × function × dynamics matrix | 15 min |
| [02_rag_architecture_guide.md](guides/02_rag_architecture_guide.md) | Full RAG pipeline with production patterns | 20 min |
| [03_memory_operators_guide.md](guides/03_memory_operators_guide.md) | Formation, retrieval, evolution operators | 15 min |
| [04_chunking_strategies_guide.md](guides/04_chunking_strategies_guide.md) | Document chunking: fixed, semantic, recursive | 15 min |
| [05_when_to_edit_weights_guide.md](guides/05_when_to_edit_weights_guide.md) | ROME, MEMIT, and when RAG isn't enough | 15 min |
| [cheatsheet.md](guides/cheatsheet.md) | Memory decision trees and quick reference | 5 min |

### Notebooks
| Notebook | Description | Time |
|----------|-------------|------|
| [01_rag_from_scratch.ipynb](notebooks/01_rag_from_scratch.ipynb) | Build a complete RAG system in 15 minutes | 15 min |
| [02_chunking_experiments.ipynb](notebooks/02_chunking_experiments.ipynb) | Compare chunking strategies visually | 10 min |
| [03_advanced_retrieval.ipynb](notebooks/03_advanced_retrieval.ipynb) | Reranking, hybrid search, query expansion | 15 min |
| [04_memory_lifecycle.ipynb](notebooks/04_memory_lifecycle.ipynb) | Formation, evolution, and decay | 15 min |

### Exercises
Self-check exercises to verify understanding (no grades).

### Resources
- [additional_readings.md](resources/additional_readings.md) - RAG, MemGPT, ROME/MEMIT papers
- [figures/](resources/figures/) - Architecture diagrams and visual assets

## Key Concepts

### The Memory Taxonomy

| Type | Purpose | Example | When to Use |
|------|---------|---------|-------------|
| **Short-term** | Active task context | Current conversation | Always (automatic) |
| **External (RAG)** | Retrieved knowledge | Company docs, Wikipedia | Frequently-changing info |
| **Long-term** | Persistent facts | User preferences, history | Stable personal data |
| **Weight editing** | Core knowledge update | Fix factual errors | Rare (last resort) |

### RAG vs Fine-tuning vs Weight Editing

```
Need to add knowledge?
├── Changes frequently → RAG
├── Source attribution needed → RAG
├── Domain knowledge but stable → Fine-tune (SFT)
├── Behavior change → Fine-tune (SFT/DPO)
└── Fix specific facts in weights → Weight editing (ROME/MEMIT)
```

### The Memory Operators

1. **Formation** - How memories are created
   - Extract relevant information
   - Summarize and compress
   - Deduplicate
   - Store with metadata

2. **Retrieval** - How memories are accessed
   - Semantic search (vector similarity)
   - Keyword search (BM25)
   - Hybrid approaches
   - Reranking

3. **Evolution** - How memories change over time
   - Consolidation (merge related memories)
   - Decay (reduce importance over time)
   - Update (new information supersedes old)

## Production RAG Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User Query                                                     │
│      │                                                          │
│      ▼                                                          │
│  ┌─────────────────┐                                            │
│  │ Query Processing│  ← Expansion, rewriting                   │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │    Embedding    │  ← text-embedding-3-large, etc.           │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │  Vector Search  │  ← Chroma, Pinecone, pgvector             │
│  │  (Top 50-100)   │                                            │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │    Reranking    │  ← Cohere, cross-encoders                 │
│  │  (Top 5-10)     │                                            │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │ Context Builder │  ← Format for prompt injection            │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │   LLM Generate  │  ← Claude, GPT-4, etc.                    │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│     Final Response                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Common Pitfalls

### Pitfall 1: Ignoring Chunking Strategy
```
"I'll just split every 500 tokens..."

Reality: Chunking is the most important RAG decision.
         Bad chunking destroys retrieval quality.
         You need semantic coherence, not arbitrary boundaries.
```

### Pitfall 2: Using Only Vector Search
```
"Embeddings capture everything..."

Reality: Hybrid search (vector + keyword) consistently outperforms.
         Some queries need exact matches, not semantic similarity.
```

### Pitfall 3: No Reranking
```
"Top-k from vector DB is good enough..."

Reality: Initial retrieval optimizes for recall (get all relevant).
         Reranking optimizes for precision (best at top).
         Production systems need both.
```

### Pitfall 4: Treating RAG as Static
```
"Set it up once and deploy..."

Reality: Memory should evolve - update docs, consolidate info,
         learn from feedback. Static RAG degrades over time.
```

## Prerequisites

- Module 00 (AI Engineer Mindset)
- Basic understanding of embeddings
- Familiarity with vector databases (helpful but not required)

## Next Steps

After this module:
- **Want to use retrieved tools?** → Module 04: Tool Use
- **Want standardized tool protocols?** → Module 05: MCP
- **Want to optimize memory?** → Module 06: Efficiency

## Time Estimate

- Quick path: 1 hour (notebooks only)
- Full path: 3-4 hours (guides + notebooks + exercises)

---

*"An LLM without memory is like a person with amnesia. It can reason, but it can't learn or remember context. Memory systems are what turn language models into intelligent agents."*
