# Module 2: RAG for Commodity Research

## Overview

Build retrieval-augmented generation systems for commodity research. Create knowledge bases from reports, implement semantic search, and generate grounded analysis.

**Time Estimate:** 8-10 hours

## Learning Objectives

By completing this module, you will:
1. Design knowledge bases for commodity data
2. Implement chunking strategies for reports
3. Build semantic search over commodity documents
4. Generate grounded research with citations

## Contents

### Guides
- `01_knowledge_base_design.md` - Structuring commodity knowledge
- `02_document_processing.md` - Chunking, embedding, indexing
- `03_retrieval_strategies.md` - Search and reranking

### Notebooks
- `01_eia_knowledge_base.ipynb` - Building an EIA report KB
- `02_research_assistant.ipynb` - RAG-powered research

## Key Concepts

### Knowledge Base Architecture

```
Document Sources          Processing          Vector Store
┌─────────────┐          ┌─────────┐         ┌─────────┐
│ EIA Reports │    →     │ Chunk   │    →    │ Chroma  │
│ USDA WASDE  │    →     │ Embed   │    →    │ Pinecone│
│ News Feed   │    →     │ Index   │    →    │         │
└─────────────┘          └─────────┘         └─────────┘
```

### Commodity-Specific Chunking

- Preserve table structures
- Keep metrics with context
- Maintain temporal references
- Extract entities and values

### Query Patterns

| Query Type | Strategy |
|------------|----------|
| Factual | Direct retrieval |
| Comparative | Multi-doc retrieval |
| Trend | Time-filtered retrieval |
| Forecast | Latest + historical |

## Prerequisites

- Module 0-1 completed
- Vector database basics
