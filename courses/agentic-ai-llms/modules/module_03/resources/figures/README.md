# Figures Directory

This directory contains visual diagrams and figures for Module 3: Memory & Context Management.

## Recommended Diagrams

The following diagrams should be created to support this module's content:

1. **memory_types_comparison.png**
   - Side-by-side comparison of Buffer, Summary, Vector, Entity, and Graph memory
   - Persistence characteristics
   - Use case mapping for each type

2. **rag_architecture.png**
   - Complete RAG pipeline flow
   - Document ingestion → Chunking → Embedding → Storage
   - Query embedding → Vector search → Context augmentation → LLM generation
   - Feedback loop

3. **vector_search_visualization.png**
   - Embedding space with query and document vectors
   - Similarity calculation (cosine, dot product)
   - k-nearest neighbors selection
   - Visual representation of semantic proximity

4. **chunking_strategies.png**
   - Fixed-size chunking with overlap
   - Semantic chunking at sentence/paragraph boundaries
   - Recursive chunking hierarchy
   - Document-aware chunking (respecting headers, sections)

5. **context_window_management.png**
   - Sliding window pattern for long conversations
   - Summary-based compression
   - Hybrid approach (recent + summary + relevant)
   - Token budget allocation

6. **hybrid_search_flow.png**
   - Parallel keyword search and vector search
   - Result fusion and reranking
   - Score normalization and combination

7. **conversation_memory_patterns.png**
   - Buffer window (last N messages)
   - Summary memory (compress old messages)
   - Selective memory (extract important entities/facts)
   - Time-decay weighting

8. **rag_evaluation_metrics.png**
   - Retrieval precision/recall
   - Answer relevance
   - Faithfulness (grounded in retrieved docs)
   - Context utilization

9. **embedding_process.png**
   - Text input → Tokenization → Embedding model → Dense vector
   - Dimensionality visualization (e.g., 1536-dim for OpenAI)
   - Semantic similarity preservation

## Usage

Reference these figures in notebooks and guides using:
```markdown
![Description](../resources/figures/filename.png)
```

For interactive notebooks:
```python
from IPython.display import Image, display
display(Image('../resources/figures/filename.png'))
```
