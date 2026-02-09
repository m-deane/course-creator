# Memory Operators: Formation, Retrieval, and Evolution

## In Brief

Memory isn't just storage - it's a living system with three core operators: **Formation** (how memories enter), **Retrieval** (how memories are accessed), and **Evolution** (how memories change over time). Master these operators to build agents that truly learn.

## Key Insight

> **The practical product question is never "How do I add memory?" It's: "Which memory form + which memory function + which lifecycle policy actually improves decision quality for my agent?"**

## Visual Explanation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MEMORY LIFECYCLE                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    ┌────────────────┐     ┌────────────────┐     ┌────────────────┐         │
│    │   FORMATION    │     │   RETRIEVAL    │     │   EVOLUTION    │         │
│    │                │     │                │     │                │         │
│    │ • Extract      │     │ • Search       │     │ • Consolidate  │         │
│    │ • Summarize    │────►│ • Rank         │────►│ • Decay        │         │
│    │ • Deduplicate  │     │ • Inject       │     │ • Merge        │         │
│    │ • Store        │     │ • Trigger      │     │ • Update       │         │
│    │                │     │                │     │                │         │
│    └───────┬────────┘     └───────┬────────┘     └───────┬────────┘         │
│            │                      │                      │                   │
│            ▼                      ▼                      ▼                   │
│    ┌───────────────────────────────────────────────────────────────┐        │
│    │                      MEMORY STATE (M_t)                        │        │
│    │                                                                │        │
│    │  Text buffers │ Key-value stores │ Vector DBs │ Graphs        │        │
│    │                                                                │        │
│    └───────────────────────────────────────────────────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Operator 1: Formation

**Purpose:** Transform raw experiences into stored memories.

### Formation Operations

| Operation | What It Does | Why It Matters |
|-----------|--------------|----------------|
| **Extract** | Identify memory candidates from artifacts | Not everything should be remembered |
| **Summarize** | Compress verbose content | Efficient storage and retrieval |
| **Normalize** | Standardize format | Consistent retrieval |
| **Deduplicate** | Remove redundant memories | Prevent bloat |
| **Score** | Assign importance/relevance | Prioritize valuable memories |
| **Store** | Write to appropriate memory form | Match form to function |

### Implementation Pattern

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import hashlib

@dataclass
class Memory:
    id: str
    content: str
    summary: Optional[str]
    importance: float
    memory_type: str  # factual, experiential, procedural
    source: str
    created_at: datetime
    embedding: Optional[list] = None
    metadata: dict = None

class MemoryFormation:
    def __init__(self, embedder, summarizer, vector_db):
        self.embedder = embedder
        self.summarizer = summarizer
        self.vector_db = vector_db
        self.importance_threshold = 0.3

    def process(self, artifact: dict) -> Optional[Memory]:
        """Full formation pipeline for a single artifact."""

        # 1. Extract: Determine if this should become a memory
        if not self._is_memorable(artifact):
            return None

        # 2. Summarize: Compress if needed
        content = artifact["content"]
        summary = None
        if len(content) > 500:
            summary = self._summarize(content)

        # 3. Score importance
        importance = self._score_importance(artifact)
        if importance < self.importance_threshold:
            return None

        # 4. Deduplicate: Check for existing similar memories
        if self._is_duplicate(content):
            return None

        # 5. Create memory object
        memory = Memory(
            id=self._generate_id(content),
            content=content,
            summary=summary,
            importance=importance,
            memory_type=self._classify_type(artifact),
            source=artifact.get("source", "unknown"),
            created_at=datetime.now(),
            metadata=artifact.get("metadata", {})
        )

        # 6. Embed for vector storage
        memory.embedding = self.embedder.encode(
            summary or content
        ).tolist()

        # 7. Store
        self._store(memory)

        return memory

    def _is_memorable(self, artifact: dict) -> bool:
        """Determine if artifact should become a memory."""
        # Skip system messages, errors, etc.
        skip_patterns = ["error:", "system:", "[internal]"]
        content = artifact.get("content", "").lower()
        return not any(p in content for p in skip_patterns)

    def _summarize(self, content: str) -> str:
        """Compress content to key points."""
        return self.summarizer.summarize(content, max_length=100)

    def _score_importance(self, artifact: dict) -> float:
        """Score memory importance 0-1."""
        score = 0.5  # Base score

        # Boost for explicit user statements
        if artifact.get("source") == "user":
            score += 0.2

        # Boost for task outcomes
        if artifact.get("type") == "task_result":
            score += 0.2

        # Boost for preferences/corrections
        if any(w in artifact.get("content", "").lower()
               for w in ["prefer", "don't", "always", "never"]):
            score += 0.1

        return min(score, 1.0)

    def _is_duplicate(self, content: str) -> bool:
        """Check if similar memory already exists."""
        embedding = self.embedder.encode(content).tolist()
        similar = self.vector_db.query(
            query_embeddings=[embedding],
            n_results=1
        )
        if similar["distances"][0]:
            # Cosine similarity > 0.95 = duplicate
            return (1 - similar["distances"][0][0]) > 0.95
        return False

    def _classify_type(self, artifact: dict) -> str:
        """Classify memory type."""
        content = artifact.get("content", "").lower()
        if "how to" in content or "steps" in content:
            return "procedural"
        if artifact.get("type") == "interaction":
            return "experiential"
        return "factual"

    def _generate_id(self, content: str) -> str:
        """Generate unique memory ID."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _store(self, memory: Memory):
        """Store memory in vector DB."""
        self.vector_db.add(
            ids=[memory.id],
            documents=[memory.content],
            embeddings=[memory.embedding],
            metadatas=[{
                "summary": memory.summary,
                "importance": memory.importance,
                "type": memory.memory_type,
                "source": memory.source,
                "created_at": memory.created_at.isoformat()
            }]
        )
```

## Operator 2: Retrieval

**Purpose:** Select and inject relevant memories into current context.

### Retrieval Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| **Event-based** | Retrieve at specific triggers (task start) | Structured workflows |
| **Continuous** | Retrieve every turn | Conversational agents |
| **Uncertainty-triggered** | Retrieve when confidence is low | Efficiency-focused |
| **Explicit** | Agent calls retrieval tool | Maximum control |

### Implementation Pattern

```python
class MemoryRetrieval:
    def __init__(self, vector_db, reranker=None):
        self.vector_db = vector_db
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        k: int = 5,
        memory_types: list = None,
        recency_weight: float = 0.1,
        importance_weight: float = 0.2
    ) -> list:
        """Retrieve relevant memories with multi-factor ranking."""

        # 1. Initial vector search (over-retrieve)
        results = self.vector_db.query(
            query_texts=[query],
            n_results=k * 3,
            where={"type": {"$in": memory_types}} if memory_types else None
        )

        # 2. Build memory objects with scores
        memories = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            semantic_score = 1 - dist  # Convert distance to similarity

            # Recency score (exponential decay)
            created = datetime.fromisoformat(meta["created_at"])
            days_old = (datetime.now() - created).days
            recency_score = math.exp(-days_old / 30)  # 30-day half-life

            # Importance from metadata
            importance_score = meta.get("importance", 0.5)

            # Combined score
            final_score = (
                (1 - recency_weight - importance_weight) * semantic_score +
                recency_weight * recency_score +
                importance_weight * importance_score
            )

            memories.append({
                "content": doc,
                "metadata": meta,
                "score": final_score,
                "semantic_score": semantic_score,
                "recency_score": recency_score
            })

        # 3. Rerank if available
        if self.reranker:
            pairs = [(query, m["content"]) for m in memories]
            rerank_scores = self.reranker.predict(pairs)
            for m, rs in zip(memories, rerank_scores):
                m["rerank_score"] = float(rs)
                m["score"] = 0.5 * m["score"] + 0.5 * rs

        # 4. Sort and return top k
        memories.sort(key=lambda x: x["score"], reverse=True)
        return memories[:k]

    def format_for_context(self, memories: list) -> str:
        """Format retrieved memories for injection into prompt."""
        if not memories:
            return ""

        formatted = "Relevant memories:\n"
        for i, mem in enumerate(memories, 1):
            summary = mem["metadata"].get("summary") or mem["content"][:200]
            formatted += f"{i}. [{mem['metadata']['type']}] {summary}\n"

        return formatted
```

### Retrieval Triggers

```python
class RetrievalPolicy:
    """Decide when to retrieve memories."""

    def __init__(self, retriever):
        self.retriever = retriever
        self.last_retrieval = None
        self.turn_count = 0

    def should_retrieve(self, context: dict) -> bool:
        """Determine if retrieval should happen."""

        # Always retrieve on task start
        if context.get("event") == "task_start":
            return True

        # Retrieve on explicit request
        if "[recall]" in context.get("user_message", "").lower():
            return True

        # Retrieve every N turns
        self.turn_count += 1
        if self.turn_count >= 3:
            self.turn_count = 0
            return True

        # Retrieve on uncertainty signals
        if context.get("model_confidence", 1.0) < 0.5:
            return True

        return False

    def get_memories(self, query: str, context: dict) -> str:
        """Get formatted memories if retrieval is triggered."""
        if not self.should_retrieve(context):
            return ""

        memories = self.retriever.retrieve(query)
        return self.retriever.format_for_context(memories)
```

## Operator 3: Evolution

**Purpose:** Maintain memory health over time through consolidation, decay, and updates.

### Evolution Operations

| Operation | What It Does | When To Apply |
|-----------|--------------|---------------|
| **Consolidate** | Merge related memories into summaries | Periodically (daily/weekly) |
| **Decay** | Reduce importance of unused memories | Continuous or periodic |
| **Prune** | Remove low-value memories | When storage limits approached |
| **Update** | Modify memories with new information | On contradiction detection |
| **Reinforce** | Boost importance of accessed memories | On each retrieval |

### Implementation Pattern

```python
class MemoryEvolution:
    def __init__(self, vector_db, embedder, summarizer):
        self.vector_db = vector_db
        self.embedder = embedder
        self.summarizer = summarizer

    def evolve(self):
        """Run full evolution cycle."""
        self.decay_unused()
        self.consolidate_similar()
        self.prune_low_value()

    def decay_unused(self, decay_rate: float = 0.95):
        """Reduce importance of memories not accessed recently."""
        all_memories = self.vector_db.get()

        for id, meta in zip(all_memories["ids"], all_memories["metadatas"]):
            last_accessed = meta.get("last_accessed")
            if last_accessed:
                days_since = (datetime.now() -
                             datetime.fromisoformat(last_accessed)).days
                if days_since > 7:  # Only decay after 7 days
                    new_importance = meta["importance"] * (decay_rate ** days_since)
                    self.vector_db.update(
                        ids=[id],
                        metadatas=[{**meta, "importance": new_importance}]
                    )

    def consolidate_similar(self, similarity_threshold: float = 0.85):
        """Merge highly similar memories into summaries."""
        all_memories = self.vector_db.get(include=["embeddings", "documents"])

        # Find clusters of similar memories
        clusters = self._cluster_memories(
            all_memories["embeddings"],
            threshold=similarity_threshold
        )

        for cluster in clusters:
            if len(cluster) < 3:
                continue  # Only consolidate 3+ memories

            # Get cluster contents
            cluster_docs = [all_memories["documents"][i] for i in cluster]
            cluster_ids = [all_memories["ids"][i] for i in cluster]

            # Create consolidated memory
            combined = "\n".join(cluster_docs)
            summary = self.summarizer.summarize(combined, max_length=200)

            # Store consolidated memory
            self.vector_db.add(
                ids=[f"consolidated_{datetime.now().timestamp()}"],
                documents=[summary],
                embeddings=[self.embedder.encode(summary).tolist()],
                metadatas=[{
                    "type": "consolidated",
                    "source_count": len(cluster),
                    "importance": 0.7,
                    "created_at": datetime.now().isoformat()
                }]
            )

            # Remove originals
            self.vector_db.delete(ids=cluster_ids)

    def prune_low_value(self, min_importance: float = 0.1, max_memories: int = 10000):
        """Remove lowest value memories."""
        count = self.vector_db.count()

        if count <= max_memories:
            return

        # Get all memories sorted by importance
        all_memories = self.vector_db.get()
        sorted_by_importance = sorted(
            zip(all_memories["ids"], all_memories["metadatas"]),
            key=lambda x: x[1].get("importance", 0)
        )

        # Remove lowest importance until under limit
        to_remove = count - max_memories
        remove_ids = [
            id for id, meta in sorted_by_importance[:to_remove]
            if meta.get("importance", 0) < min_importance
        ]

        if remove_ids:
            self.vector_db.delete(ids=remove_ids)

    def reinforce(self, memory_id: str, boost: float = 0.1):
        """Boost importance when memory is accessed."""
        memory = self.vector_db.get(ids=[memory_id])
        if memory["ids"]:
            meta = memory["metadatas"][0]
            new_importance = min(meta["importance"] + boost, 1.0)
            self.vector_db.update(
                ids=[memory_id],
                metadatas=[{
                    **meta,
                    "importance": new_importance,
                    "last_accessed": datetime.now().isoformat(),
                    "access_count": meta.get("access_count", 0) + 1
                }]
            )

    def _cluster_memories(self, embeddings, threshold):
        """Simple clustering by cosine similarity."""
        # Implementation using sklearn or custom clustering
        pass
```

## Putting It All Together

```python
class AgentMemory:
    """Complete memory system with all three operators."""

    def __init__(self, config):
        self.formation = MemoryFormation(...)
        self.retrieval = MemoryRetrieval(...)
        self.evolution = MemoryEvolution(...)
        self.policy = RetrievalPolicy(self.retrieval)

    def remember(self, artifact: dict) -> Optional[Memory]:
        """Form a new memory from an artifact."""
        return self.formation.process(artifact)

    def recall(self, query: str, context: dict) -> str:
        """Retrieve relevant memories for current context."""
        memories = self.policy.get_memories(query, context)

        # Reinforce accessed memories
        for mem in memories:
            self.evolution.reinforce(mem.get("id"))

        return self.retrieval.format_for_context(memories)

    def maintain(self):
        """Run periodic maintenance (call daily/weekly)."""
        self.evolution.evolve()
```

## Common Pitfalls

### Pitfall 1: No formation filtering
**Problem:** Everything becomes a memory, causing bloat.
**Solution:** Apply importance scoring and deduplication.

### Pitfall 2: Static retrieval
**Problem:** Always retrieve the same way regardless of context.
**Solution:** Adaptive retrieval with multiple strategies.

### Pitfall 3: No evolution
**Problem:** Memories become stale and contradictory.
**Solution:** Implement decay, consolidation, and pruning.

## Connections

**Builds on:**
- Module 03 guides 01-02 (memory taxonomy, RAG architecture)

**Leads to:**
- Module 04: Tool Use (memory-aware agents)
- Module 08: Production (memory at scale)

## Practice Problems

1. **Design:** Create a formation policy for a customer support agent. What should be remembered? What filtered out?

2. **Implement:** Build a retrieval system that weights recency, importance, and semantic similarity. Test different weight combinations.

3. **Analyze:** An agent's memory has grown to 100K entries and retrieval is slow. Design an evolution strategy to maintain quality while reducing size.
