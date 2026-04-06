# Memory Systems Cheatsheet

> **Reading time:** ~5 min | **Module:** 3 — Memory Systems | **Prerequisites:** None

<span class="badge mint">All Levels</span> <span class="badge amber">~5 min</span> <span class="badge blue">Module 3</span>

## The Memory Decision Tree

```
What kind of information?
+-- Changes frequently (news, prices, events)
|   --> RAG with frequent re-indexing
+-- User-specific (preferences, history)
|   --> Long-term memory store (per-user)
+-- Domain knowledge (docs, manuals)
|   --> RAG (vector DB)
+-- Current task state (steps, variables)
|   --> Working memory (context window)
+-- Conversation context (recent messages)
|   --> Context window + summarization
+-- Needs behavior change (not just knowledge)
    --> Fine-tuning (SFT/DPO), not memory
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

<div class="flow">
  <div class="flow-step mint">Query</div>
  <div class="flow-arrow">&#8594;</div>
  <div class="flow-step amber">Embed</div>
  <div class="flow-arrow">&#8594;</div>
  <div class="flow-step blue">Retrieve (k*3)</div>
  <div class="flow-arrow">&#8594;</div>
  <div class="flow-step lavender">Rerank (top-k)</div>
  <div class="flow-arrow">&#8594;</div>
  <div class="flow-step rose">Generate</div>
</div>

<div class="callout-info">

<strong>Info:</strong> Use the same embedding model for indexing and querying. Over-retrieve, then rerank to the final count.

</div>

---

## Chunking Strategies

| Strategy | Chunk Size | Overlap | Best For |
|----------|------------|---------|----------|
| **Fixed** | 500 tokens | 50 tokens | General purpose |
| **Sentence** | 3-5 sentences | 1 sentence | Natural boundaries |
| **Paragraph** | 1 paragraph | 0 | Well-structured docs |
| **Semantic** | Variable | Context-aware | Quality-critical |
| **Recursive** | Target size | 10-20% | Mixed content |

<div class="callout-key">

<strong>Key Point:</strong> Start with 500 tokens, 10% overlap. Adjust based on retrieval quality.

</div>

---

## Embedding Models

| Model | Dims | Speed | Quality | Cost |
|-------|------|-------|---------|------|
| `all-MiniLM-L6-v2` | 384 | Very fast | Good | Free |
| `bge-small-en-v1.5` | 384 | Fast | Very good | Free |
| `bge-base-en-v1.5` | 768 | Medium | Excellent | Free |
| `text-embedding-3-small` | 1536 | API | Excellent | $0.02/1M |
| `voyage-2` | 1024 | API | State-of-art | $0.10/1M |

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


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">formation_ops.py</span>
</div>

```python
extract()      # Identify memory candidates
summarize()    # Compress content
deduplicate()  # Remove redundant
score()        # Assign importance
store()        # Write to appropriate store
```

</div>
</div>

### Retrieval


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">retrieval_ops.py</span>
</div>

```python
search()       # Vector similarity search
filter()       # Apply metadata filters
rerank()       # Cross-encoder reordering
inject()       # Format for prompt
```

</div>
</div>

### Evolution


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">evolution_ops.py</span>
</div>

```python
decay()        # Reduce unused memory importance
consolidate()  # Merge similar memories
prune()        # Remove low-value memories
reinforce()    # Boost accessed memories
```

</div>
</div>

---

## Retrieval Metrics

| Metric | Formula | Good Value |
|--------|---------|------------|
| **Recall@5** | Relevant in top-5 / Total relevant | >0.8 |
| **Precision@5** | Relevant in top-5 / 5 | >0.6 |
| **MRR** | Mean(1/rank of first relevant) | >0.5 |
| **Latency p95** | 95th percentile response time | <100ms |

---

## Anti-Patterns

<div class="compare">
  <div class="compare-card">
    <div class="header before">Don't</div>
    <div class="body">Store everything. Never update memories. Single retrieval strategy. Ignore metadata. Same embedding for all content. Retrieve once, use forever.</div>
  <div class="compare-card">
    <div class="header after">Do Instead</div>
    <div class="body">Filter by importance. Implement evolution. Adaptive retrieval. Use metadata for filtering. Domain-specific when beneficial. Re-retrieve on context change.</div>

---

## When to Use What

| Use Case | Solution |
|----------|----------|
| External knowledge needed, changes over time, need source attribution | **RAG** |
| User-specific, cross-session, learning from interactions | **Long-term memory** |
| Need behavior change, domain-specific language, consistent style | **Fine-tuning** |
| Specific fact corrections, can't use retrieval | **Weight editing (ROME/MEMIT)** |

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
