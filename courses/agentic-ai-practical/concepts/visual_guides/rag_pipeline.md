# RAG Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   INDEXING (once)                    QUERYING (every question)  │
│   ┌─────────────┐                    ┌─────────────┐            │
│   │  Documents  │                    │  Question   │            │
│   └──────┬──────┘                    └──────┬──────┘            │
│          │                                  │                   │
│          ▼                                  ▼                   │
│   ┌─────────────┐                    ┌─────────────┐            │
│   │   Chunk     │                    │   Embed     │            │
│   │  (split)    │                    │  (vectorize)│            │
│   └──────┬──────┘                    └──────┬──────┘            │
│          │                                  │                   │
│          ▼                                  ▼                   │
│   ┌─────────────┐                    ┌─────────────┐            │
│   │   Embed     │───────────────────▶│   Search    │            │
│   │  (vectorize)│     Vector DB      │  (top-k)    │            │
│   └──────┬──────┘                    └──────┬──────┘            │
│          │                                  │                   │
│          ▼                                  ▼                   │
│   ┌─────────────┐                    ┌─────────────┐            │
│   │   Store     │                    │  Context +  │            │
│   │  (vectors)  │                    │  Question   │            │
│   └─────────────┘                    └──────┬──────┘            │
│                                             │                   │
│                                             ▼                   │
│                                      ┌─────────────┐            │
│                                      │     LLM     │            │
│                                      │  (answer)   │            │
│                                      └──────┬──────┘            │
│                                             │                   │
│                                             ▼                   │
│                                      ┌─────────────┐            │
│                                      │   Answer    │            │
│                                      └─────────────┘            │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ TL;DR: Search your docs, stuff results into prompt, ask LLM    │
├─────────────────────────────────────────────────────────────────┤
│ Code:                                                           │
│   results = vectordb.search(question, k=3)                      │
│   context = "\n".join(results)                                  │
│   answer = llm(f"Context: {context}\n\nQ: {question}")          │
├─────────────────────────────────────────────────────────────────┤
│ Pitfall: Chunks too big = irrelevant noise                     │
│          Chunks too small = missing context                     │
│          Sweet spot: 500-1000 chars with 20% overlap            │
└─────────────────────────────────────────────────────────────────┘
```

## When to Use RAG

| Use RAG When | Don't Use RAG When |
|--------------|-------------------|
| Docs change frequently | Static, small knowledge base |
| Need citations/sources | General knowledge questions |
| Domain-specific content | Creative tasks |
| Can't fine-tune model | Can fine-tune for your domain |

## Quick Start

```python
import chromadb
collection = chromadb.Client().create_collection("docs")
collection.add(documents=["Doc 1", "Doc 2"], ids=["1", "2"])
results = collection.query(query_texts=["question"], n_results=2)
```

→ Full template: [../templates/rag_template.py](../templates/rag_template.py)
