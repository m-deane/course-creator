# Vector Stores: Embedding and Storing Knowledge

## In Brief

Vector stores are databases optimized for similarity search over embeddings. They enable RAG systems to find semantically relevant documents from millions of entries in milliseconds.

> 💡 **Key Insight:** **Vector search finds meaning, not keywords.** Traditional databases match exact terms. Vector stores find documents with similar meaning—"automobile" finds "car," "vehicle," and "transportation."

---

## Vector Store Fundamentals

### How Vector Search Works

```
1. Documents → Embed → Vectors stored in index

   "Electric cars are efficient"  →  [0.12, -0.34, 0.56, ...]
   "Solar panels generate power"  →  [0.08, -0.28, 0.61, ...]
   "Gas vehicles emit CO2"        →  [0.15, -0.31, 0.52, ...]

2. Query → Embed → Find nearest neighbors

   "renewable energy vehicles"    →  [0.11, -0.32, 0.58, ...]
                                           ↓
                                  Similarity search
                                           ↓
                           [Match 1: "Electric cars are efficient"]
                           [Match 2: "Gas vehicles emit CO2"]
```

### Distance Metrics

| Metric | Formula | Range | Use Case |
|--------|---------|-------|----------|
| Cosine | 1 - cos(θ) | [0, 2] | Text similarity (normalized) |
| Euclidean (L2) | √Σ(a-b)² | [0, ∞) | Dense vectors |
| Dot Product | Σ(a·b) | [-∞, ∞) | Pre-normalized vectors |

---

## Vector Store Options

### ChromaDB (Local/Embedded)

Best for: Development, small-medium datasets, serverless

```python
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# Initialize
client = chromadb.Client()  # In-memory
# client = chromadb.PersistentClient(path="./chroma_db")  # Persistent

# Create collection with embedding function
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.create_collection(
    name="documents",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}  # Distance metric
)

# Add documents
collection.add(
    documents=["Electric cars are efficient", "Solar power is renewable"],
    metadatas=[{"source": "article1"}, {"source": "article2"}],
    ids=["doc1", "doc2"]
)

# Query
results = collection.query(
    query_texts=["green energy transportation"],
    n_results=2
)

print(results["documents"])
print(results["distances"])
```

### Pinecone (Cloud)

Best for: Production, large scale, managed infrastructure

```python
import pinecone
from pinecone import Pinecone, ServerlessSpec


# Initialize
pc = Pinecone(api_key="your-api-key")

# Create index
pc.create_index(
    name="documents",
    dimension=384,  # Match your embedding model
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

index = pc.Index("documents")

# Upsert vectors
embeddings = embedding_model.encode(documents).tolist()

index.upsert(
    vectors=[
        {"id": f"doc_{i}", "values": emb, "metadata": {"text": doc}}
        for i, (emb, doc) in enumerate(zip(embeddings, documents))
    ]
)

# Query
query_embedding = embedding_model.encode(["green energy"]).tolist()[0]

results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
)

for match in results["matches"]:
    print(f"{match['id']}: {match['score']:.3f} - {match['metadata']['text']}")
```

### Weaviate (Self-hosted or Cloud)

Best for: GraphQL queries, hybrid search, multimodal

```python
import weaviate


client = weaviate.Client("http://localhost:8080")

# Create schema
class_obj = {
    "class": "Document",
    "vectorizer": "text2vec-transformers",
    "properties": [
        {"name": "content", "dataType": ["text"]},
        {"name": "source", "dataType": ["string"]}
    ]
}
client.schema.create_class(class_obj)

# Add data
client.data_object.create(
    data_object={"content": "Electric cars are efficient", "source": "article1"},
    class_name="Document"
)

# Query with GraphQL
result = client.query.get("Document", ["content", "source"])\
    .with_near_text({"concepts": ["green energy"]})\
    .with_limit(5)\
    .do()
```

### Qdrant (Self-hosted or Cloud)

Best for: High performance, complex filtering, production

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


client = QdrantClient(host="localhost", port=6333)

# Create collection
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# Insert vectors
points = [
    PointStruct(
        id=i,
        vector=embedding_model.encode(doc).tolist(),
        payload={"text": doc, "source": f"source_{i}"}
    )
    for i, doc in enumerate(documents)
]

client.upsert(collection_name="documents", points=points)

# Search with filters
results = client.search(
    collection_name="documents",
    query_vector=embedding_model.encode("green energy").tolist(),
    limit=5,
    query_filter={
        "must": [{"key": "source", "match": {"value": "source_1"}}]
    }
)
```

---

## Index Types

### HNSW (Hierarchical Navigable Small World)

Most common index for vector search:

```python
# ChromaDB HNSW settings
collection = client.create_collection(
    name="documents",
    metadata={
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 100,  # Build quality (higher = better, slower)
        "hnsw:search_ef": 50,         # Search quality
        "hnsw:M": 16                   # Connections per layer
    }
)
```

**Trade-offs:**
- Higher `construction_ef`: Better index quality, slower build
- Higher `search_ef`: Better search quality, slower search
- Higher `M`: Better recall, more memory

### IVF (Inverted File Index)

Good for very large datasets:

```python
# Pinecone with IVF
pc.create_index(
    name="large-docs",
    dimension=384,
    metric="cosine",
    spec=PodSpec(
        environment="us-west1-gcp",
        pod_type="p1.x1"
    )
)
```

### Flat (Brute Force)

Exact search, best for small datasets:

```python
# FAISS flat index
import faiss

dimension = 384
index = faiss.IndexFlatL2(dimension)  # Exact L2 distance
index.add(embeddings)

distances, indices = index.search(query_embedding, k=5)
```

---

## Metadata Filtering

### Filter Patterns

```python
# ChromaDB filtering
results = collection.query(
    query_texts=["electric vehicles"],
    n_results=10,
    where={
        "$and": [
            {"category": {"$eq": "technology"}},
            {"year": {"$gte": 2020}},
            {"source": {"$in": ["reuters", "bbc"]}}
        ]
    }
)

# Qdrant filtering
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

results = client.search(
    collection_name="documents",
    query_vector=query_vector,
    query_filter=Filter(
        must=[
            FieldCondition(key="category", match=MatchValue(value="technology")),
            FieldCondition(key="year", range=Range(gte=2020))
        ],
        must_not=[
            FieldCondition(key="status", match=MatchValue(value="draft"))
        ]
    )
)
```

### Metadata Schema Design

```python
# Good metadata design
metadata = {
    "source": "company_wiki",
    "category": "engineering",
    "subcategory": "infrastructure",
    "author": "john_doe",
    "created_at": "2024-01-15",
    "updated_at": "2024-03-20",
    "version": 3,
    "is_public": True,
    "tags": ["kubernetes", "deployment", "production"]
}

# Bad: Unstructured, inconsistent
metadata = {
    "info": "Engineering doc by John, updated March 2024",
    "misc": "k8s stuff"
}
```

---

## Ingestion Pipelines

### Document Processing Pipeline

```python
from dataclasses import dataclass
from typing import Generator
import hashlib


@dataclass
class ProcessedChunk:
    content: str
    metadata: dict
    id: str


def process_documents(
    documents: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Generator[ProcessedChunk, None, None]:
    """Process documents into chunks for vector store."""

    for doc in documents:
        text = doc["content"]
        base_metadata = doc.get("metadata", {})

        # Chunk the document
        chunks = chunk_by_size(text, chunk_size, chunk_overlap)

        for i, chunk in enumerate(chunks):
            # Create unique ID
            chunk_id = hashlib.md5(
                f"{doc.get('id', '')}_{i}_{chunk[:50]}".encode()
            ).hexdigest()

            yield ProcessedChunk(
                content=chunk,
                metadata={
                    **base_metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "doc_id": doc.get("id")
                },
                id=chunk_id
            )


def ingest_to_vector_store(
    chunks: list[ProcessedChunk],
    collection,
    embedder,
    batch_size: int = 100
):
    """Ingest processed chunks into vector store."""

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        embeddings = embedder.encode([c.content for c in batch]).tolist()

        collection.add(
            documents=[c.content for c in batch],
            embeddings=embeddings,
            metadatas=[c.metadata for c in batch],
            ids=[c.id for c in batch]
        )

        print(f"Ingested {min(i + batch_size, len(chunks))}/{len(chunks)}")
```

### Incremental Updates

```python
class VectorStoreManager:
    """Manage vector store with incremental updates."""

    def __init__(self, collection, embedder):
        self.collection = collection
        self.embedder = embedder
        self.doc_hashes = {}  # Track document hashes

    def _hash_document(self, content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()

    def upsert_documents(self, documents: list[dict]):
        """Insert or update documents."""
        to_add = []
        to_update = []

        for doc in documents:
            doc_hash = self._hash_document(doc["content"])
            doc_id = doc["id"]

            if doc_id not in self.doc_hashes:
                to_add.append(doc)
            elif self.doc_hashes[doc_id] != doc_hash:
                to_update.append(doc)

            self.doc_hashes[doc_id] = doc_hash

        # Add new documents
        if to_add:
            self._add_documents(to_add)

        # Update changed documents
        if to_update:
            self._update_documents(to_update)

    def _add_documents(self, documents: list[dict]):
        embeddings = self.embedder.encode([d["content"] for d in documents]).tolist()
        self.collection.add(
            documents=[d["content"] for d in documents],
            embeddings=embeddings,
            ids=[d["id"] for d in documents],
            metadatas=[d.get("metadata", {}) for d in documents]
        )

    def _update_documents(self, documents: list[dict]):
        embeddings = self.embedder.encode([d["content"] for d in documents]).tolist()
        self.collection.update(
            documents=[d["content"] for d in documents],
            embeddings=embeddings,
            ids=[d["id"] for d in documents],
            metadatas=[d.get("metadata", {}) for d in documents]
        )
```

---

## Performance Optimization

### Embedding Caching

```python
import diskcache


class CachedEmbedder:
    """Cache embeddings to avoid recomputation."""

    def __init__(self, embedder, cache_dir: str = ".embedding_cache"):
        self.embedder = embedder
        self.cache = diskcache.Cache(cache_dir)

    def encode(self, texts: list[str]) -> list[list[float]]:
        results = []
        to_compute = []
        indices = []

        for i, text in enumerate(texts):
            cache_key = hashlib.md5(text.encode()).hexdigest()
            cached = self.cache.get(cache_key)

            if cached is not None:
                results.append(cached)
            else:
                to_compute.append(text)
                indices.append(i)
                results.append(None)  # Placeholder

        # Compute uncached embeddings
        if to_compute:
            computed = self.embedder.encode(to_compute).tolist()

            for idx, emb, text in zip(indices, computed, to_compute):
                cache_key = hashlib.md5(text.encode()).hexdigest()
                self.cache.set(cache_key, emb)
                results[idx] = emb

        return results
```

### Query Optimization

```python
def optimized_query(
    query: str,
    collection,
    embedder,
    n_results: int = 10,
    rerank_top_k: int = 50
) -> list[dict]:
    """Optimized query with over-fetch and rerank."""

    # Over-fetch for better reranking
    query_embedding = embedder.encode([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=rerank_top_k
    )

    # Rerank with cross-encoder (more accurate but slower)
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    pairs = [(query, doc) for doc in results["documents"][0]]
    scores = reranker.predict(pairs)

    # Sort by reranked scores
    ranked = sorted(
        zip(results["documents"][0], results["metadatas"][0], scores),
        key=lambda x: x[2],
        reverse=True
    )

    return [
        {"content": doc, "metadata": meta, "score": float(score)}
        for doc, meta, score in ranked[:n_results]
    ]
```

---

## Best Practices

1. **Choose embedding model carefully**: Match model to your domain
2. **Chunk thoughtfully**: Preserve context, not arbitrary boundaries
3. **Use rich metadata**: Enable filtering to reduce search space
4. **Benchmark your use case**: Different stores excel at different scales
5. **Monitor retrieval quality**: Track precision/recall in production
6. **Plan for updates**: Design for incremental ingestion
7. **Test at scale**: Performance varies dramatically with dataset size

---

*Vector stores are the long-term memory of your agent. Design your knowledge architecture as carefully as you would design a database schema.*
