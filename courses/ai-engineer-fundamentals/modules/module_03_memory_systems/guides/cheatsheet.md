# Memory Systems Cheatsheet

## The Memory Decision Tree

```
What kind of information?
│
├── Changes frequently (news, prices, events)
│   └── RAG with frequent re-indexing
│
├── User-specific (preferences, history)
│   └── Long-term memory store (per-user)
│
├── Domain knowledge (docs, manuals)
│   └── RAG (vector DB)
│
├── Current task state (steps, variables)
│   └── Working memory (context window)
│
├── Conversation context (recent messages)
│   └── Context window + summarization
│
└── Needs behavior change (not just knowledge)
    └── Fine-tuning (SFT/DPO), not memory
```

---

## Memory Forms Comparison

| Form | Access Speed | Capacity | Update Cost | Best For |
|------|--------------|----------|-------------|----------|
| **Context Window** | Instant | Limited (4K-128K) | Free | Current task |
| **Vector DB** | Fast (~10ms) | Unlimited | Index rebuild | Knowledge base |
| **Key-Value Store** | Very fast | Large | Instant | User prefs, state |
| **Graph DB** | Medium | Large | Medium | Relationships |
| **Model Weights** | Instant | Fixed | Very expensive | Core behaviors |

---

## RAG Pipeline Quick Reference

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  Query  │────►│  Embed  │────►│Retrieve │────►│ Rerank  │────►│Generate │
└─────────┘     └─────────┘     └─────────┘     └─────────┘     └─────────┘
                    │               │               │
                    ▼               ▼               ▼
               Same model      Over-retrieve    Cross-encoder
               as indexing       (k×3)          to top-k
```

---

## Chunking Strategies

| Strategy | Chunk Size | Overlap | Best For |
|----------|------------|---------|----------|
| **Fixed** | 500 tokens | 50 tokens | General purpose |
| **Sentence** | 3-5 sentences | 1 sentence | Natural boundaries |
| **Paragraph** | 1 paragraph | 0 | Well-structured docs |
| **Semantic** | Variable | Context-aware | Quality-critical |
| **Recursive** | Target size | 10-20% | Mixed content |

**Rule of thumb:** Start with 500 tokens, 10% overlap. Adjust based on retrieval quality.

---

## Embedding Models

| Model | Dims | Speed | Quality | Cost |
|-------|------|-------|---------|------|
| `all-MiniLM-L6-v2` | 384 | ⚡⚡⚡ | ★★★ | Free |
| `bge-small-en-v1.5` | 384 | ⚡⚡⚡ | ★★★★ | Free |
| `bge-base-en-v1.5` | 768 | ⚡⚡ | ★★★★★ | Free |
| `text-embedding-3-small` | 1536 | ⚡⚡ | ★★★★★ | $0.02/1M |
| `voyage-2` | 1024 | ⚡⚡ | ★★★★★ | $0.10/1M |

---

## Vector Databases

| Database | Type | Scale | Complexity | Cost |
|----------|------|-------|------------|------|
| **Chroma** | Embedded | <1M | Low | Free |
| **Pinecone** | Managed | Unlimited | Low | Pay per use |
| **Qdrant** | Self-host/Cloud | Large | Medium | Free/Paid |
| **Weaviate** | Self-host/Cloud | Large | Medium | Free/Paid |
| **pgvector** | PostgreSQL ext | Medium | Low | Free |

---

## Memory Operators

### Formation
```python
# Key operations
extract()      # Identify memory candidates
summarize()    # Compress content
deduplicate()  # Remove redundant
score()        # Assign importance
store()        # Write to appropriate store
```

### Retrieval
```python
# Key operations
search()       # Vector similarity search
filter()       # Apply metadata filters
rerank()       # Cross-encoder reordering
inject()       # Format for prompt
```

### Evolution
```python
# Key operations
decay()        # Reduce unused memory importance
consolidate()  # Merge similar memories
prune()        # Remove low-value memories
reinforce()    # Boost accessed memories
```

---

## Retrieval Metrics

| Metric | Formula | Good Value |
|--------|---------|------------|
| **Recall@5** | Relevant in top-5 / Total relevant | >0.8 |
| **Precision@5** | Relevant in top-5 / 5 | >0.6 |
| **MRR** | Mean(1/rank of first relevant) | >0.5 |
| **Latency p95** | 95th percentile response time | <100ms |

---

## Code Snippets

### Quick RAG Setup
```python
import chromadb
from sentence_transformers import SentenceTransformer

# Setup
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
db = chromadb.PersistentClient("./db")
collection = db.get_or_create_collection("docs")

# Index
collection.add(
    documents=["doc1", "doc2"],
    embeddings=embedder.encode(["doc1", "doc2"]).tolist(),
    ids=["1", "2"]
)

# Query
results = collection.query(
    query_embeddings=embedder.encode(["query"]).tolist(),
    n_results=5
)
```

### Memory Formation
```python
def should_remember(content: str, source: str) -> bool:
    """Quick importance check."""
    if source == "user" and len(content) > 20:
        return True
    if any(w in content.lower() for w in ["prefer", "always", "never"]):
        return True
    return False
```

### Memory Decay
```python
def decay_importance(importance: float, days_unused: int) -> float:
    """Exponential decay with 30-day half-life."""
    return importance * (0.95 ** days_unused)
```

---

## Common Patterns

### Hybrid Search
```
query → [vector_search, keyword_search] → reciprocal_rank_fusion → top_k
```

### Hierarchical Memory
```
Hot (context) ←→ Warm (vector DB) ←→ Cold (archive)
         ↑ promote                demote ↓
```

### Memory-Augmented Generation
```
user_query + retrieved_memories + system_prompt → LLM → response
                                                    ↓
                                            form_new_memory
```

---

## Anti-Patterns

| Don't | Do Instead |
|-------|------------|
| Store everything | Filter by importance |
| Never update memories | Implement evolution |
| Single retrieval strategy | Adaptive retrieval |
| Ignore metadata | Use metadata for filtering |
| Same embedding for all content | Domain-specific when beneficial |
| Retrieve once, use forever | Re-retrieve on context change |

---

## When to Use What

```
RAG:
✓ External knowledge needed
✓ Information changes over time
✓ Need source attribution
✓ Large knowledge base

Long-term memory:
✓ User-specific information
✓ Cross-session persistence
✓ Learning from interactions

Fine-tuning:
✓ Need behavior change
✓ Domain-specific language
✓ Consistent style/format

Weight editing (ROME/MEMIT):
✓ Specific fact corrections
✓ Can't use retrieval
✓ Small number of edits
```

---

## Quick Debugging

| Problem | Likely Cause | Fix |
|---------|--------------|-----|
| Poor retrieval | Wrong chunk size | Try 200-1000 range |
| Missing context | No overlap | Add 10-20% overlap |
| Slow queries | Too many results | Reduce k, add filtering |
| Stale answers | Old documents | Re-index, add timestamps |
| Hallucination | Retrieved but not used | Check prompt formatting |
| Memory bloat | No deduplication | Add similarity threshold |
