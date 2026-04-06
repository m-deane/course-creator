# Retrieval Strategies for RAG in Dataiku

> **Reading time:** ~6 min | **Module:** 2 — Rag | **Prerequisites:** Module 1 — Prompt Studios

<div class="callout-key">

<strong>Key Concept:</strong> RAG quality is 80% retrieval quality. The best LLM in the world produces hallucinations if it receives irrelevant or incomplete context. Optimising retrieval is the highest-leverage investment in any RAG system.

</div>

## Introduction

Effective retrieval is critical for RAG performance. This guide covers strategies for optimizing document retrieval in Dataiku's Knowledge Banks.

<div class="compare">
<div class="compare-card">
<div class="header before">Naive Retrieval</div>
<div class="body">

- Single embedding similarity search
- Fixed chunk size for all documents
- No metadata filtering
- Misses multi-hop reasoning

</div>

</div>
<div class="compare-card">
<div class="header after">Optimised Retrieval</div>
<div class="body">

- Hybrid search (vector + keyword)
- Adaptive chunking by document type
- Metadata filters narrow search space
- Re-ranking for relevance

</div>

</div>

</div>

## Embedding-Based Retrieval

### Vector Similarity Search


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Conceptual implementation within Dataiku
class VectorRetriever:
    """
    Basic vector similarity retrieval.
    """

    def __init__(self, knowledge_bank, embedding_model, top_k=5):
        self.kb = knowledge_bank
        self.embedding_model = embedding_model
        self.top_k = top_k

    def retrieve(self, query):
        """
        Retrieve documents by vector similarity.
        """
        # Embed query
        query_embedding = self.embedding_model.encode(query)

        # Search knowledge bank
        results = self.kb.similarity_search(
            query_embedding,
            k=self.top_k
        )

        return [
            {
                "content": r.content,
                "score": r.similarity_score,
                "metadata": r.metadata
            }
            for r in results
        ]
```

</div>
</div>

### Distance Metrics


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import numpy as np

def cosine_similarity(a, b):
    """Standard cosine similarity."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def euclidean_distance(a, b):
    """Euclidean distance (lower is more similar)."""
    return np.linalg.norm(a - b)

def dot_product(a, b):
    """Dot product (higher is more similar for normalized vectors)."""
    return np.dot(a, b)

# Dataiku Knowledge Bank configuration
retrieval_config = {
    "distance_metric": "cosine",  # Options: cosine, euclidean, dot_product
    "normalize_embeddings": True,
    "index_type": "hnsw",  # Hierarchical Navigable Small World
    "hnsw_params": {
        "M": 16,  # Number of connections per layer
        "ef_construction": 200,  # Size of dynamic candidate list
        "ef_search": 100  # Search-time dynamic list size
    }
}
```

</div>
</div>

## Hybrid Retrieval

Combine vector search with keyword matching:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
class HybridRetriever:
    """
    Combine vector and keyword search.
    """

    def __init__(self, knowledge_bank, embedding_model,
                 vector_weight=0.7, keyword_weight=0.3, top_k=5):
        self.kb = knowledge_bank
        self.embedding_model = embedding_model
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.top_k = top_k

    def retrieve(self, query):
        """
        Hybrid retrieval combining vector and keyword scores.
        """
        # Vector search
        query_embedding = self.embedding_model.encode(query)
        vector_results = self.kb.similarity_search(
            query_embedding,
            k=self.top_k * 2  # Get more candidates for reranking
        )

        # Keyword search (BM25 or similar)
        keyword_results = self.kb.keyword_search(
            query,
            k=self.top_k * 2
        )

        # Combine scores
        combined_scores = {}

        for r in vector_results:
            doc_id = r.document_id
            combined_scores[doc_id] = {
                "content": r.content,
                "vector_score": r.similarity_score * self.vector_weight,
                "keyword_score": 0,
                "metadata": r.metadata
            }

        for r in keyword_results:
            doc_id = r.document_id
            if doc_id in combined_scores:
                combined_scores[doc_id]["keyword_score"] = r.score * self.keyword_weight
            else:
                combined_scores[doc_id] = {
                    "content": r.content,
                    "vector_score": 0,
                    "keyword_score": r.score * self.keyword_weight,
                    "metadata": r.metadata
                }

        # Calculate final scores
        results = []
        for doc_id, data in combined_scores.items():
            data["final_score"] = data["vector_score"] + data["keyword_score"]
            results.append(data)

        # Sort and return top-k
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:self.top_k]


# Configuration in Dataiku
hybrid_config = {
    "retrieval_mode": "hybrid",
    "vector_search": {
        "enabled": True,
        "weight": 0.7,
        "embedding_model": "text-embedding-3-small"
    },
    "keyword_search": {
        "enabled": True,
        "weight": 0.3,
        "algorithm": "bm25",
        "bm25_k1": 1.5,
        "bm25_b": 0.75
    }
}
```

</div>
</div>

## Query Expansion

Improve retrieval by expanding the original query:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
class QueryExpander:
    """
    Expand queries for better retrieval coverage.
    """

    def __init__(self, llm_connection):
        self.llm = llm_connection

    def expand_query(self, query, n_expansions=3):
        """
        Generate query variations using LLM.
        """
        prompt = f"""Generate {n_expansions} alternative phrasings of this search query.
The alternatives should capture the same intent but use different words.

Original query: {query}

Return only the alternative queries, one per line, without numbering."""

        response = self.llm.generate(prompt, max_tokens=200)

        expansions = [query]  # Include original
        for line in response.text.strip().split('\n'):
            if line.strip():
                expansions.append(line.strip())

        return expansions

    def retrieve_with_expansion(self, query, retriever, deduplicate=True):
        """
        Retrieve using expanded queries.
        """
        expanded_queries = self.expand_query(query)

        all_results = []
        seen_ids = set()

        for q in expanded_queries:
            results = retriever.retrieve(q)
            for r in results:
                if not deduplicate or r.get('document_id') not in seen_ids:
                    all_results.append(r)
                    if r.get('document_id'):
                        seen_ids.add(r['document_id'])

        # Re-rank by score
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)

        return all_results


# Example usage

# expander = QueryExpander(llm_connection)

# results = expander.retrieve_with_expansion("crude oil inventory report", retriever)
```

</div>
</div>

## Contextual Compression

Reduce retrieved content to most relevant portions:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
class ContextualCompressor:
    """
    Compress retrieved documents to relevant portions.
    """

    def __init__(self, llm_connection, max_compressed_length=500):
        self.llm = llm_connection
        self.max_length = max_compressed_length

    def compress(self, query, documents):
        """
        Extract relevant portions from documents.
        """
        compressed = []

        for doc in documents:
            prompt = f"""Given this query and document, extract only the sentences
that are directly relevant to answering the query.

Query: {query}

Document:
{doc['content'][:2000]}

Relevant excerpts (return only the relevant sentences, nothing else):"""

            response = self.llm.generate(prompt, max_tokens=self.max_length)

            compressed.append({
                "original_content": doc['content'],
                "compressed_content": response.text.strip(),
                "score": doc.get('score', 0),
                "metadata": doc.get('metadata', {})
            })

        return compressed
```

</div>
</div>

## Multi-Index Retrieval

Search across multiple knowledge banks:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
class MultiIndexRetriever:
    """
    Retrieve from multiple knowledge banks.
    """

    def __init__(self, knowledge_banks, weights=None):
        """
        Parameters:
        -----------
        knowledge_banks : dict
            {name: knowledge_bank_connection}
        weights : dict
            {name: weight} for score weighting
        """
        self.kbs = knowledge_banks
        self.weights = weights or {name: 1.0 for name in knowledge_banks}

    def retrieve(self, query, top_k=5, strategy='merge'):
        """
        Retrieve from all knowledge banks.

        Strategies:
        - merge: Combine and re-rank all results
        - round_robin: Alternate between sources
        - source_balanced: Equal results per source
        """
        all_results = []

        for name, kb in self.kbs.items():
            results = kb.search(query, k=top_k)
            weight = self.weights.get(name, 1.0)

            for r in results:
                r['source'] = name
                r['weighted_score'] = r.get('score', 0) * weight
                all_results.append(r)

        if strategy == 'merge':
            all_results.sort(key=lambda x: x['weighted_score'], reverse=True)
            return all_results[:top_k]

        elif strategy == 'round_robin':
            # Group by source
            by_source = {name: [] for name in self.kbs}
            for r in all_results:
                by_source[r['source']].append(r)

            # Interleave
            result = []
            i = 0
            while len(result) < top_k:
                for name in self.kbs:
                    if i < len(by_source[name]):
                        result.append(by_source[name][i])
                        if len(result) >= top_k:
                            break
                i += 1
            return result

        elif strategy == 'source_balanced':
            # Equal from each source
            per_source = max(1, top_k // len(self.kbs))
            result = []
            for name in self.kbs:
                source_results = [r for r in all_results if r['source'] == name]
                result.extend(source_results[:per_source])
            return result[:top_k]


# Configuration example
multi_index_config = {
    "knowledge_banks": {
        "company_docs": {
            "connection": "kb-company-docs",
            "weight": 1.0
        },
        "industry_reports": {
            "connection": "kb-industry",
            "weight": 0.8
        },
        "news_archive": {
            "connection": "kb-news",
            "weight": 0.6
        }
    },
    "retrieval_strategy": "merge",
    "top_k": 10
}
```


## Reranking

Improve result quality with a reranking model:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
class Reranker:
    """
    Rerank retrieved documents for better relevance.
    """

    def __init__(self, rerank_model='cross-encoder'):
        self.model = rerank_model

    def rerank(self, query, documents, top_k=5):
        """
        Rerank documents using cross-encoder or LLM.
        """
        if self.model == 'cross-encoder':
            return self._rerank_cross_encoder(query, documents, top_k)
        elif self.model == 'llm':
            return self._rerank_llm(query, documents, top_k)

    def _rerank_cross_encoder(self, query, documents, top_k):
        """
        Rerank using cross-encoder model.
        """
        # In Dataiku, this would use a deployed model
        from sentence_transformers import CrossEncoder

        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        pairs = [(query, doc['content'][:512]) for doc in documents]
        scores = model.predict(pairs)

        for i, doc in enumerate(documents):
            doc['rerank_score'] = float(scores[i])

        documents.sort(key=lambda x: x['rerank_score'], reverse=True)
        return documents[:top_k]

    def _rerank_llm(self, query, documents, top_k):
        """
        Rerank using LLM scoring.
        """
        scored_docs = []

        for doc in documents:
            prompt = f"""Rate the relevance of this document to the query on a scale of 1-10.
Return only the number.

Query: {query}

Document: {doc['content'][:500]}

Relevance score (1-10):"""

            response = self.llm.generate(prompt, max_tokens=5)

            try:
                score = int(response.text.strip())
            except:
                score = 5

            doc['llm_score'] = score
            scored_docs.append(doc)

        scored_docs.sort(key=lambda x: x['llm_score'], reverse=True)
        return scored_docs[:top_k]


# Dataiku configuration for reranking
rerank_config = {
    "enabled": True,
    "model": "cross-encoder",
    "model_endpoint": "reranker-v1",
    "batch_size": 32,
    "top_k_rerank": 20,  # Rerank top 20, return top 5
    "top_k_final": 5
}
```


## Performance Optimization

### Caching


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
class CachedRetriever:
    """
    Retriever with query caching.
    """

    def __init__(self, base_retriever, cache_ttl=3600):
        self.retriever = base_retriever
        self.cache = {}
        self.cache_ttl = cache_ttl

    def retrieve(self, query, **kwargs):
        """Retrieve with caching."""
        import hashlib
        import time

        cache_key = hashlib.md5(query.encode()).hexdigest()

        # Check cache
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if time.time() - entry['timestamp'] < self.cache_ttl:
                return entry['results']

        # Retrieve and cache
        results = self.retriever.retrieve(query, **kwargs)

        self.cache[cache_key] = {
            'results': results,
            'timestamp': time.time()
        }

        return results
```


## Key Takeaways

1. **Hybrid retrieval** combines strengths of vector and keyword search

2. **Query expansion** improves recall for ambiguous queries

3. **Contextual compression** reduces noise in retrieved content

4. **Multi-index** enables searching across diverse knowledge sources

5. **Reranking** significantly improves precision

6. **Caching** is essential for production performance


## Resources

<a class="link-card" href="../notebooks/01_kb_creation.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>
