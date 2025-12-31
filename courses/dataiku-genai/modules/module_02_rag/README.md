# Module 2: RAG with Knowledge Banks

## Overview

Build retrieval-augmented generation applications using Dataiku's Knowledge Banks. Configure vector stores, chunking, and retrieval for enterprise documents.

**Time Estimate:** 8-10 hours

## Learning Objectives

By completing this module, you will:
1. Create and configure Knowledge Banks
2. Ingest documents with optimal chunking
3. Configure retrieval parameters
4. Build RAG-powered applications

## Contents

### Guides
- `01_knowledge_banks.md` - Creating vector stores
- `02_document_ingestion.md` - Chunking and embedding
- `03_rag_applications.md` - Building RAG flows

### Notebooks
- `01_kb_creation.ipynb` - Knowledge Bank setup
- `02_rag_workflow.ipynb` - Complete RAG pipeline

## Key Concepts

### Knowledge Bank Architecture

```
Documents           Knowledge Bank         RAG Application
┌─────────┐        ┌─────────────┐        ┌────────────┐
│  PDFs   │   →    │  Chunking   │        │   Query    │
│  Docs   │   →    │  Embedding  │   →    │  Retrieve  │
│  HTML   │   →    │  Indexing   │        │  Generate  │
└─────────┘        └─────────────┘        └────────────┘
```

### Configuration Options

| Setting | Options | Default |
|---------|---------|---------|
| Chunk Size | 256-2048 tokens | 512 |
| Overlap | 0-50% | 10% |
| Embedding | OpenAI, Cohere | OpenAI |
| Top-K | 1-20 | 5 |

### Knowledge Bank Features

- Automatic document parsing
- Multiple file format support
- Metadata extraction
- Source attribution
- Incremental updates

## Prerequisites

- Module 0-1 completed
- Documents for ingestion
