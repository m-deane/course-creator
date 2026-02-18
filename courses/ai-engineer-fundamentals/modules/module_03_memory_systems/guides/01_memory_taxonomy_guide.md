# Memory Taxonomy: Understanding LLM Memory Systems

## In Brief

Memory in LLM systems is not a single thing - it's a taxonomy of mechanisms that provide different types of information at different timescales. Understanding this taxonomy is the foundation for building effective agent systems.

## Key Insight

> **Memory is an evolving state the agent can read and write while making decisions.**

It's not just storage - it has a lifecycle (formation → retrieval → evolution) and comes in different forms optimized for different functions.

## Visual Explanation

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MEMORY TAXONOMY                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    MEMORY FORMS                              │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │                                                              │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │   │
│  │  │   CONTEXT    │  │  PARAMETRIC  │  │    EXTERNAL      │   │   │
│  │  │   WINDOW     │  │   (WEIGHTS)  │  │    STORES        │   │   │
│  │  ├──────────────┤  ├──────────────┤  ├──────────────────┤   │   │
│  │  │ • Prompt     │  │ • Trained    │  │ • Vector DB      │   │   │
│  │  │ • System msg │  │   knowledge  │  │ • Key-value      │   │   │
│  │  │ • History    │  │ • Behaviors  │  │ • Graph DB       │   │   │
│  │  │ • Retrieved  │  │ • Patterns   │  │ • Relational DB  │   │   │
│  │  │   docs       │  │              │  │ • File storage   │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘   │   │
│  │       ▲                  ▲                  ▲                │   │
│  │       │                  │                  │                │   │
│  │  Fast access       Hard to update     Flexible, scalable    │   │
│  │  Limited size      Expensive          Retrieval overhead    │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## The Three Memory Types

### 1. Short-Term Memory: The Context Window

**What it is:** The tokens currently in the prompt that the model can attend to.

**Characteristics:**
- **Fast:** Direct attention access, no retrieval overhead
- **Limited:** Fixed size (4K, 8K, 32K, 128K tokens depending on model)
- **Ephemeral:** Gone when the request ends (unless persisted)

**What goes here:**
- System prompts
- Conversation history
- Retrieved documents (from RAG)
- Current task context

**The key limitation:** When full, something must be dropped. This drives the need for smart memory management.

```python
# Example: Managing context window
class ContextManager:
    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = max_tokens
        self.system_prompt_tokens = 500  # Reserve for system
        self.buffer_tokens = 500  # Safety buffer
        self.available = max_tokens - self.system_prompt_tokens - self.buffer_tokens

    def fit_context(self, messages: list, retrieved_docs: list) -> tuple:
        """Prioritize what fits in context."""
        # Priority: recent messages > relevant docs > older messages
        # Implementation truncates oldest messages first
        pass
```

### 2. External Knowledge: RAG (Retrieval-Augmented Generation)

**What it is:** Documents stored externally, retrieved at inference time based on relevance.

**Characteristics:**
- **Scalable:** Can store millions of documents
- **Updatable:** Change documents without retraining
- **Traceable:** Can cite sources
- **Retrieval cost:** Embedding + search adds latency

**What goes here:**
- Knowledge bases
- Documentation
- Historical data
- Domain-specific content

**The key insight:** Don't memorize everything in weights. Retrieve what's relevant when needed.

```python
# Example: Basic RAG setup
from chromadb import Client
from sentence_transformers import SentenceTransformer

# Index documents
embedder = SentenceTransformer('all-MiniLM-L6-v2')
db = Client()
collection = db.create_collection("knowledge_base")

for doc in documents:
    embedding = embedder.encode(doc.content)
    collection.add(
        documents=[doc.content],
        embeddings=[embedding],
        ids=[doc.id],
        metadatas=[doc.metadata]
    )

# Retrieve at inference time
def retrieve(query: str, k: int = 5) -> list:
    query_embedding = embedder.encode(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    return results['documents'][0]
```

### 3. Long-Term Memory: Persistent Agent State

**What it is:** Information that persists across sessions, learned from experience.

**Characteristics:**
- **Persistent:** Survives across sessions
- **Selective:** Not everything is stored
- **Evolving:** Memories consolidate, decay, update
- **Agent-managed:** The agent controls read/write operations

**What goes here:**
- User preferences
- Learned facts about the user/domain
- Past task outcomes
- Summarized experiences

**The key insight:** Long-term memory requires a lifecycle policy - not just storage, but formation, retrieval, and evolution.

```python
# Example: Long-term memory with lifecycle
class LongTermMemory:
    def __init__(self, db):
        self.db = db

    # FORMATION: Extract and store
    def remember(self, content: str, importance: float = 0.5):
        """Store a memory with importance score."""
        memory = {
            "content": content,
            "importance": importance,
            "created_at": datetime.now(),
            "access_count": 0,
            "last_accessed": None
        }
        self.db.insert(memory)

    # RETRIEVAL: Get relevant memories
    def recall(self, query: str, k: int = 5) -> list:
        """Retrieve memories relevant to query."""
        memories = self.db.semantic_search(query, k=k*2)  # Over-retrieve
        # Apply recency and importance weighting
        scored = self._score_memories(memories, query)
        return sorted(scored, key=lambda m: m['score'], reverse=True)[:k]

    # EVOLUTION: Maintain memory health
    def consolidate(self):
        """Periodically merge and prune memories."""
        # Merge similar memories
        # Decay old, unused memories
        # Update importance based on access patterns
        pass
```

## Memory Form × Function × Dynamics

A useful framework for classifying memory along three axes:

### Forms (How it's stored)
| Form | Example | Access Pattern |
|------|---------|----------------|
| Token/context | Prompt text | Direct attention |
| Parametric | Model weights | Forward pass |
| Vector store | Embeddings in DB | Similarity search |
| Key-value | Redis, DynamoDB | Exact key lookup |
| Graph | Neo4j | Relationship traversal |
| Relational | PostgreSQL | SQL queries |

### Functions (What it stores)
| Function | Description | Example |
|----------|-------------|---------|
| Factual | World knowledge | "Paris is the capital of France" |
| Experiential | Past interactions | "User prefers concise answers" |
| Working | Current task state | "We're on step 3 of 5" |
| Procedural | How to do things | Tool usage patterns |
| Episodic | Specific events | "Last week we discussed X" |

### Dynamics (How it changes)
| Phase | Operations | Purpose |
|-------|------------|---------|
| Formation | Extract, summarize, deduplicate, store | Get memories into the system |
| Retrieval | Search, rank, inject | Get memories out when needed |
| Evolution | Consolidate, decay, merge, update | Keep memory healthy over time |

## The Memory Matrix

Use this to design your memory system:

```
                    │ Context  │ RAG      │ Long-term │ Weights
────────────────────┼──────────┼──────────┼───────────┼─────────
Factual knowledge   │ ✓ (temp) │ ✓✓✓      │ ✓         │ ✓✓
User preferences    │ ✓        │          │ ✓✓✓       │
Current task state  │ ✓✓✓      │          │ ✓         │
Domain expertise    │          │ ✓✓       │           │ ✓✓✓
Conversation history│ ✓✓       │          │ ✓✓        │
Recent events       │ ✓✓✓      │ ✓        │ ✓         │
────────────────────┴──────────┴──────────┴───────────┴─────────
Legend: ✓✓✓ = primary, ✓✓ = good fit, ✓ = possible
```

## Common Pitfalls

### Pitfall 1: Treating all memory as context
**Problem:** Stuffing everything into the prompt until it overflows.
**Solution:** Use hierarchical memory - only retrieve what's relevant.

### Pitfall 2: Storing everything
**Problem:** Memory bloats, retrieval quality degrades.
**Solution:** Selective storage with importance scoring and periodic pruning.

### Pitfall 3: No memory evolution
**Problem:** Stale, redundant, or contradictory memories accumulate.
**Solution:** Implement consolidation, decay, and conflict resolution.

### Pitfall 4: Wrong memory form for the function
**Problem:** Using RAG for rapidly-changing task state, or context for static knowledge.
**Solution:** Match memory form to function using the matrix above.

## Connections

**Builds on:**
- Transformer attention (how context is processed)
- Embeddings (how semantic similarity works)

**Leads to:**
- Module 04: Tool Use (memory enables stateful agents)
- Module 08: Production Systems (memory management at scale)

## Practice Problems

1. **Conceptual:** You're building a customer support agent. Map each of these to a memory type: (a) product documentation, (b) customer's name and account info, (c) current ticket being discussed, (d) past interactions with this customer.

2. **Implementation:** Design a memory system for a research assistant that needs to: track papers it has read, remember user's research interests, and maintain state across a multi-day research project.

3. **Decision:** A chatbot needs to remember that "the user's name is Alice" - should this go in context, RAG, long-term memory, or model weights? What are the tradeoffs?
