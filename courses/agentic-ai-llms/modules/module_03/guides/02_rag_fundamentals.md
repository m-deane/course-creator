# RAG Fundamentals: Retrieval-Augmented Generation

> **Reading time:** ~12 min | **Module:** 3 — Memory & Context | **Prerequisites:** Module 0 — Context Windows

RAG grounds LLM responses in external knowledge by retrieving relevant documents before generation. This reduces hallucinations, enables access to private data, and keeps responses current beyond training cutoffs.

<div class="callout-insight">

**Insight:** RAG separates what the model knows from what it can access. Instead of cramming all knowledge into model weights, RAG fetches relevant information at runtime—making knowledge updatable, verifiable, and scalable.

</div>

---

## The RAG Pipeline

### Architecture Overview

```
User Query: "What are our refund policies?"
                    ↓
            [1. EMBED QUERY]
         Convert to vector embedding
                    ↓
            [2. RETRIEVE]
         Search vector database
         Find top-k similar documents
                    ↓
            [3. AUGMENT]
         Add retrieved context to prompt
                    ↓
            [4. GENERATE]
         LLM produces grounded response
                    ↓
Response: "According to our policy doc..."
```

### Basic Implementation


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>

```python
import anthropic
from sentence_transformers import SentenceTransformer
import chromadb


class SimpleRAG:
    """Basic RAG implementation."""

    def __init__(self):
        self.llm = anthropic.Anthropic()
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.db = chromadb.Client()
        self.collection = self.db.create_collection("documents")

    def add_documents(self, documents: list[str], ids: list[str] = None):
        """Add documents to the knowledge base."""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        embeddings = self.embedder.encode(documents).tolist()

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids
        )

    def query(self, question: str, n_results: int = 3) -> str:
        """Answer a question using RAG."""

        # 1. Embed the query
        query_embedding = self.embedder.encode([question]).tolist()

        # 2. Retrieve relevant documents
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

        # 3. Augment prompt with context
        context = "\n\n".join(results["documents"][0])

        prompt = f"""Answer the question based on the provided context.
If the context doesn't contain the answer, say so.

Context:
{context}

Question: {question}

Answer:"""

        # 4. Generate response
        response = self.llm.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text


# Usage
rag = SimpleRAG()

# Add knowledge
rag.add_documents([
    "Our refund policy allows returns within 30 days of purchase.",
    "Refunds are processed within 5-7 business days.",
    "Products must be in original packaging for full refund.",
])

# Query
answer = rag.query("How long do I have to return a product?")
print(answer)
```

</div>
</div>

---

## Document Processing

### Chunking Strategies

Breaking documents into retrieval-friendly pieces:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>

```python
from typing import Generator


def chunk_by_size(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into fixed-size overlapping chunks."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks


def chunk_by_sentences(text: str, sentences_per_chunk: int = 5) -> list[str]:
    """Split text by sentence boundaries."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []

    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = ' '.join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)

    return chunks


def chunk_by_paragraphs(text: str) -> list[str]:
    """Split text by paragraph boundaries."""
    paragraphs = text.split('\n\n')
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_markdown(text: str) -> list[str]:
    """Split markdown by headers, preserving structure."""
    import re
    sections = re.split(r'\n(?=#)', text)
    return [s.strip() for s in sections if s.strip()]
```

</div>
</div>

### Semantic Chunking

Split at natural semantic boundaries:

```python
from sentence_transformers import SentenceTransformer
import numpy as np


class SemanticChunker:
    """Chunk text at semantic boundaries."""

    def __init__(self, threshold: float = 0.5):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = threshold

    def chunk(self, text: str) -> list[str]:
        # Split into sentences
        sentences = text.replace('\n', ' ').split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]

        if len(sentences) <= 1:
            return [text]

        # Get embeddings
        embeddings = self.embedder.encode(sentences)

        # Calculate similarity between consecutive sentences
        chunks = []
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            # Cosine similarity
            sim = np.dot(embeddings[i-1], embeddings[i]) / (
                np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
            )

            if sim < self.threshold:
                # Semantic break - start new chunk
                chunks.append(' '.join(current_chunk))
                current_chunk = []

            current_chunk.append(sentences[i])

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks
```

### Adding Metadata

Rich metadata improves retrieval:

```python
def process_document(
    filepath: str,
    chunk_size: int = 1000
) -> list[dict]:
    """Process a document into chunks with metadata."""
    import os
    from datetime import datetime

    with open(filepath, 'r') as f:
        text = f.read()

    chunks = chunk_by_size(text, chunk_size)

    documents = []
    for i, chunk in enumerate(chunks):
        documents.append({
            "content": chunk,
            "metadata": {
                "source": filepath,
                "filename": os.path.basename(filepath),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "processed_at": datetime.utcnow().isoformat(),
            }
        })

    return documents
```

---

## Embedding Models

### Choosing an Embedding Model

| Model | Dimensions | Speed | Quality | Use Case |
|-------|------------|-------|---------|----------|
| all-MiniLM-L6-v2 | 384 | Fast | Good | General purpose |
| all-mpnet-base-v2 | 768 | Medium | Better | Higher accuracy |
| text-embedding-3-large | 3072 | API | Best | Production |
| BAAI/bge-large-en | 1024 | Medium | Excellent | Open-source best |

### Using OpenAI Embeddings


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>

```python
from openai import OpenAI


class OpenAIEmbedder:
    """Embeddings via OpenAI API."""

    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = OpenAI()
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [item.embedding for item in response.data]
```

</div>
</div>

### Batch Embedding

```python
def embed_in_batches(
    texts: list[str],
    embedder,
    batch_size: int = 32
) -> list[list[float]]:
    """Embed texts in batches to avoid memory issues."""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = embedder.encode(batch).tolist()
        all_embeddings.extend(embeddings)

    return all_embeddings
```

---

## Retrieval Strategies

### Basic Semantic Search


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>

```python
def semantic_search(
    query: str,
    collection,
    embedder,
    n_results: int = 5
) -> list[dict]:
    """Simple semantic similarity search."""

    query_embedding = embedder.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )

    return [
        {
            "content": doc,
            "score": 1 - dist,  # Convert distance to similarity
            "metadata": meta
        }
        for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0]
        )
    ]
```

</div>
</div>

### Hybrid Search (Keyword + Semantic)

```python
from rank_bm25 import BM25Okapi


class HybridRetriever:
    """Combine BM25 keyword search with semantic search."""

    def __init__(self, documents: list[str], embedder):
        self.documents = documents
        self.embedder = embedder

        # BM25 index
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

        # Semantic embeddings
        self.embeddings = embedder.encode(documents)

    def search(
        self,
        query: str,
        n_results: int = 5,
        alpha: float = 0.5  # Weight for semantic vs keyword
    ) -> list[tuple[int, float]]:
        """Hybrid search combining BM25 and semantic scores."""

        # BM25 scores
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_scores = bm25_scores / (bm25_scores.max() + 1e-6)  # Normalize

        # Semantic scores
        query_embedding = self.embedder.encode([query])[0]
        semantic_scores = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Combine scores
        combined = alpha * semantic_scores + (1 - alpha) * bm25_scores

        # Get top results
        top_indices = np.argsort(combined)[-n_results:][::-1]

        return [(idx, combined[idx]) for idx in top_indices]
```

### Multi-Query Retrieval

Generate multiple query variations for better recall:

```python
class MultiQueryRetriever:
    """Generate multiple queries to improve retrieval."""

    def __init__(self, llm, base_retriever):
        self.llm = llm
        self.base_retriever = base_retriever

    def generate_queries(self, original_query: str, n_queries: int = 3) -> list[str]:
        """Generate query variations."""

        prompt = f"""Generate {n_queries} different versions of this question
that might help find relevant documents. Make them diverse.

Original question: {original_query}

Variations (one per line):"""

        response = self.llm.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )

        variations = response.content[0].text.strip().split('\n')
        return [original_query] + [v.strip() for v in variations if v.strip()]

    def retrieve(self, query: str, n_results: int = 5) -> list[dict]:
        """Retrieve using multiple query variations."""

        queries = self.generate_queries(query)

        all_results = {}
        for q in queries:
            results = self.base_retriever.search(q, n_results)
            for r in results:
                key = r["content"][:100]  # Dedupe key
                if key not in all_results or r["score"] > all_results[key]["score"]:
                    all_results[key] = r

        # Sort by score and return top n
        sorted_results = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results[:n_results]
```

---

## Prompt Engineering for RAG

### Basic RAG Prompt


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>

```python
RAG_PROMPT = """Answer the question based on the provided context.
If the context doesn't contain enough information, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
```


### Citation-Aware Prompt

```python
CITATION_PROMPT = """Answer the question using the provided sources.
Cite your sources using [1], [2], etc.

Sources:
{numbered_sources}

Question: {question}

Provide a comprehensive answer with citations:"""


def format_sources(documents: list[dict]) -> str:
    """Format documents as numbered sources."""
    return "\n\n".join(
        f"[{i+1}] {doc['content']}"
        for i, doc in enumerate(documents)
    )
```

### Instructed RAG

```python
INSTRUCTED_RAG_PROMPT = """You are a helpful assistant with access to a knowledge base.

Instructions:
1. Base your answer ONLY on the provided context
2. If the context is insufficient, say "I don't have enough information"
3. Quote relevant passages to support your answer
4. If sources conflict, acknowledge the discrepancy

Context from knowledge base:
{context}

User question: {question}

Your response:"""
```

---

## Evaluation

### Retrieval Quality Metrics


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>

```python
def evaluate_retrieval(
    queries: list[str],
    ground_truth: list[list[str]],  # Relevant doc IDs for each query
    retriever,
    k: int = 5
) -> dict:
    """Evaluate retrieval quality."""

    precisions = []
    recalls = []
    mrrs = []

    for query, relevant_ids in zip(queries, ground_truth):
        results = retriever.search(query, n_results=k)
        retrieved_ids = [r["id"] for r in results]

        # Precision@k
        relevant_retrieved = len(set(retrieved_ids) & set(relevant_ids))
        precisions.append(relevant_retrieved / k)

        # Recall@k
        recalls.append(relevant_retrieved / len(relevant_ids))

        # MRR
        for i, rid in enumerate(retrieved_ids):
            if rid in relevant_ids:
                mrrs.append(1 / (i + 1))
                break
        else:
            mrrs.append(0)

    return {
        "precision@k": sum(precisions) / len(precisions),
        "recall@k": sum(recalls) / len(recalls),
        "mrr": sum(mrrs) / len(mrrs)
    }
```


### End-to-End Evaluation

```python
def evaluate_rag_answers(
    qa_pairs: list[dict],
    rag_system,
    llm_judge
) -> dict:
    """Evaluate RAG answer quality using LLM-as-judge."""

    scores = []

    for qa in qa_pairs:
        generated = rag_system.query(qa["question"])

        judge_prompt = f"""Rate the answer quality from 1-5:
1 = Incorrect or irrelevant
3 = Partially correct
5 = Fully correct and comprehensive

Question: {qa["question"]}
Expected Answer: {qa["expected_answer"]}
Generated Answer: {generated}

Score (just the number):"""

        response = llm_judge.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": judge_prompt}]
        )

        try:
            score = int(response.content[0].text.strip())
            scores.append(score)
        except ValueError:
            scores.append(3)  # Default

    return {
        "average_score": sum(scores) / len(scores),
        "score_distribution": {i: scores.count(i) for i in range(1, 6)}
    }
```

<div class="callout-key">

**Key Concept Summary:** This guide covered the core concepts. Review the companion slides for visual summaries and the hands-on notebook for practice implementations.


---

*RAG transforms LLMs from closed knowledge systems to open retrieval engines. Master this pattern—it's the foundation for most production agent systems.*




## Practice Questions

1. Explain in your own words how the concepts in this guide relate to building production agents.
2. What are the key tradeoffs you need to consider when applying these techniques?
3. Describe a scenario where the approach from this guide would be the wrong choice, and what you would use instead.

---

**Next Steps:**

<a class="link-card" href="./02_rag_fundamentals_slides.md">
  <div class="link-card-title">RAG Fundamentals for Agents — Companion Slides</div>
  <div class="link-card-description">Visual slide deck with diagrams, speaker notes, and key takeaways.</div>
</a>

<a class="link-card" href="../notebooks/01_memory_patterns.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with working code and guided exercises.</div>
</a>
