# RAG Architecture: Production-Ready Retrieval

## In Brief

RAG (Retrieval-Augmented Generation) combines the reasoning power of LLMs with the ability to access external knowledge at inference time. Instead of memorizing everything in weights, retrieve what's relevant when you need it.

## Key Insight

> **Separate "what to know" (retriever) from "how to express it" (generator). This makes knowledge updatable without retraining.**

## Visual Explanation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌────────┐ │
│  │  QUERY  │────►│  EMBED  │────►│RETRIEVE │────►│ RERANK  │────►│GENERATE│ │
│  │         │     │         │     │         │     │         │     │        │ │
│  └─────────┘     └────┬────┘     └────┬────┘     └────┬────┘     └───┬────┘ │
│                       │               │               │              │      │
│                       ▼               ▼               ▼              ▼      │
│                  ┌─────────┐    ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│                  │Embedding│    │ Vector   │   │Cross-    │   │  LLM     │  │
│                  │ Model   │    │   DB     │   │encoder   │   │ Context  │  │
│                  └─────────┘    └──────────┘   └──────────┘   │ + Docs   │  │
│                                                               └──────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

                     INDEXING PIPELINE (offline)
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐               │
│  │DOCUMENTS│────►│  CHUNK  │────►│  EMBED  │────►│  STORE  │               │
│  │         │     │         │     │         │     │         │               │
│  └─────────┘     └─────────┘     └─────────┘     └─────────┘               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## The RAG Pipeline

### Step 1: Document Ingestion (Offline)

Before you can retrieve, you need to index your documents.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# 1. Load documents
documents = load_documents("./docs/")

# 2. Chunk documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = []
for doc in documents:
    doc_chunks = splitter.split_text(doc.content)
    for i, chunk in enumerate(doc_chunks):
        chunks.append({
            "id": f"{doc.id}_{i}",
            "content": chunk,
            "metadata": {
                "source": doc.source,
                "chunk_index": i
            }
        })

# 3. Embed chunks
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
embeddings = embedder.encode([c["content"] for c in chunks])

# 4. Store in vector DB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"}
)
collection.add(
    ids=[c["id"] for c in chunks],
    documents=[c["content"] for c in chunks],
    embeddings=embeddings.tolist(),
    metadatas=[c["metadata"] for c in chunks]
)
```

### Step 2: Query Embedding

Transform the user's query into the same embedding space as your documents.

```python
def embed_query(query: str) -> list:
    """Embed query using same model as documents."""
    return embedder.encode(query).tolist()
```

**Best Practice:** Use the same embedding model for queries and documents. Mismatched models = poor retrieval.

### Step 3: Retrieval

Find the most relevant chunks using vector similarity.

```python
def retrieve(query: str, k: int = 5) -> list:
    """Retrieve top-k relevant chunks."""
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    return [
        {
            "content": doc,
            "metadata": meta,
            "score": 1 - dist  # Convert distance to similarity
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
    ]
```

### Step 4: Reranking (Optional but Recommended)

Initial retrieval is fast but imprecise. Reranking uses a more powerful model to reorder results.

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, documents: list, top_k: int = 3) -> list:
    """Rerank retrieved documents using cross-encoder."""
    pairs = [(query, doc["content"]) for doc in documents]
    scores = reranker.predict(pairs)

    # Add scores and sort
    for doc, score in zip(documents, scores):
        doc["rerank_score"] = float(score)

    return sorted(documents, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
```

**Why rerank?**
- Bi-encoders (embedding models) are fast but less accurate
- Cross-encoders see query and document together, more accurate but slower
- Retrieve many (k=20), rerank to few (k=3)

### Step 5: Generation

Combine retrieved context with the query and generate a response.

```python
import anthropic

client = anthropic.Anthropic()

def generate_with_context(query: str, context_docs: list) -> str:
    """Generate response using retrieved context."""

    # Format context
    context = "\n\n---\n\n".join([
        f"Source: {doc['metadata']['source']}\n{doc['content']}"
        for doc in context_docs
    ])

    # Create prompt with context
    prompt = f"""Use the following context to answer the question.
If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {query}

Answer:"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text
```

## Full RAG Pipeline

```python
class RAGPipeline:
    def __init__(self, collection, embedder, reranker=None):
        self.collection = collection
        self.embedder = embedder
        self.reranker = reranker
        self.client = anthropic.Anthropic()

    def query(self, question: str, retrieve_k: int = 10, final_k: int = 3) -> dict:
        """Full RAG pipeline: retrieve, rerank, generate."""

        # 1. Embed query
        query_embedding = self.embedder.encode(question).tolist()

        # 2. Retrieve
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=retrieve_k
        )

        docs = [
            {"content": doc, "metadata": meta}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]

        # 3. Rerank (if available)
        if self.reranker:
            pairs = [(question, doc["content"]) for doc in docs]
            scores = self.reranker.predict(pairs)
            for doc, score in zip(docs, scores):
                doc["score"] = float(score)
            docs = sorted(docs, key=lambda x: x["score"], reverse=True)[:final_k]
        else:
            docs = docs[:final_k]

        # 4. Generate
        context = "\n\n".join([doc["content"] for doc in docs])

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            }]
        )

        return {
            "answer": response.content[0].text,
            "sources": docs
        }
```

## Embedding Model Selection

| Model | Dimensions | Speed | Quality | Use Case |
|-------|------------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 384 | Very fast | Good | Prototyping, low-resource |
| `BAAI/bge-small-en-v1.5` | 384 | Fast | Very good | Production, balanced |
| `BAAI/bge-base-en-v1.5` | 768 | Medium | Excellent | When quality matters |
| `text-embedding-3-small` | 1536 | API call | Excellent | OpenAI users |
| `voyage-2` | 1024 | API call | State-of-art | Best quality, higher cost |

**Selection criteria:**
- **Prototyping:** Use small, fast models
- **Production:** Balance quality vs latency
- **Domain-specific:** Consider fine-tuned embeddings

## Vector Database Comparison

| Database | Type | Best For | Limitations |
|----------|------|----------|-------------|
| **Chroma** | Embedded | Local dev, small scale | Single machine |
| **Pinecone** | Managed | Production, serverless | Cost at scale |
| **Weaviate** | Self-hosted/Cloud | Hybrid search | Complexity |
| **Qdrant** | Self-hosted/Cloud | Performance | Operational overhead |
| **pgvector** | PostgreSQL extension | Existing Postgres users | Scale limits |

```python
# Chroma (local)
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")

# Pinecone (managed)
from pinecone import Pinecone
pc = Pinecone(api_key="your-key")
index = pc.Index("your-index")

# pgvector (PostgreSQL)
# CREATE EXTENSION vector;
# CREATE TABLE items (id bigserial, embedding vector(384));
```

## Advanced Patterns

### Hybrid Search (Vector + Keyword)

Combine semantic search with keyword matching for better coverage.

```python
def hybrid_search(query: str, k: int = 5, alpha: float = 0.5) -> list:
    """Combine vector and keyword search."""
    # Vector search
    vector_results = vector_search(query, k=k*2)

    # Keyword search (BM25)
    keyword_results = bm25_search(query, k=k*2)

    # Merge with reciprocal rank fusion
    scores = {}
    for rank, doc in enumerate(vector_results):
        scores[doc["id"]] = scores.get(doc["id"], 0) + alpha / (rank + 60)
    for rank, doc in enumerate(keyword_results):
        scores[doc["id"]] = scores.get(doc["id"], 0) + (1-alpha) / (rank + 60)

    # Return top k by combined score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
```

### Query Expansion

Improve retrieval by expanding the query with related terms.

```python
def expand_query(query: str) -> str:
    """Use LLM to generate query variations."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"Generate 3 alternative phrasings of this search query:\n{query}"
        }]
    )
    return query + " " + response.content[0].text
```

### Contextual Compression

Reduce retrieved chunks to only the relevant portions.

```python
def compress_context(query: str, document: str) -> str:
    """Extract only query-relevant portions of document."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"""Extract only the parts of this document relevant to the query.

Query: {query}
Document: {document}

Relevant excerpt:"""
        }]
    )
    return response.content[0].text
```

## Common Pitfalls

### Pitfall 1: Wrong chunk size
**Problem:** Too large = irrelevant content pollutes context. Too small = lost context.
**Solution:** 200-500 tokens is a good starting point. Test with your data.

### Pitfall 2: No overlap between chunks
**Problem:** Information at chunk boundaries gets lost.
**Solution:** Use 10-20% overlap.

### Pitfall 3: Ignoring metadata
**Problem:** Can't filter by source, date, or other attributes.
**Solution:** Store rich metadata, use it for filtering.

### Pitfall 4: Retrieving too much
**Problem:** Context window pollution, slower generation, higher cost.
**Solution:** Retrieve more, rerank to less. Quality over quantity.

## Evaluation Metrics

| Metric | What it measures | Calculation |
|--------|------------------|-------------|
| **Recall@k** | Coverage of relevant docs | Relevant in top-k / Total relevant |
| **Precision@k** | Relevance of retrieved docs | Relevant in top-k / k |
| **MRR** | Ranking quality | Mean of 1/rank of first relevant |
| **NDCG** | Graded relevance | Normalized discounted cumulative gain |
| **Answer faithfulness** | Generated answer accuracy | Does answer match sources? |

```python
def evaluate_retrieval(queries: list, ground_truth: list, k: int = 5) -> dict:
    """Evaluate retrieval quality."""
    recalls, precisions = [], []

    for query, relevant_ids in zip(queries, ground_truth):
        retrieved = retrieve(query, k=k)
        retrieved_ids = {doc["id"] for doc in retrieved}

        hits = len(retrieved_ids & set(relevant_ids))
        recalls.append(hits / len(relevant_ids))
        precisions.append(hits / k)

    return {
        "recall@{k}": sum(recalls) / len(recalls),
        "precision@{k}": sum(precisions) / len(precisions)
    }
```

## Connections

**Builds on:**
- Embeddings (semantic representations)
- Vector similarity search

**Leads to:**
- Module 04: Tool Use (RAG as a tool in agent loops)
- Module 08: Production (scaling RAG systems)

## Practice Problems

1. **Implement:** Build a RAG system for a PDF collection. Compare results with and without reranking.

2. **Evaluate:** Measure retrieval quality on a test set. What chunk size gives the best recall@5?

3. **Optimize:** Your RAG system is too slow. What are three ways to reduce latency while maintaining quality?
