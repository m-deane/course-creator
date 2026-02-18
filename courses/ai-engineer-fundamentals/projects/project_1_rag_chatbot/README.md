# Portfolio Project 1: Domain-Specific RAG Chatbot

> **Build a production-ready RAG system for a specific domain.**

## What You'll Build

A fully functional chatbot that can answer questions about a specific domain (legal, medical, technical documentation, etc.) using Retrieval-Augmented Generation.

**Demo:** Upload documents → Ask questions → Get accurate, sourced answers

## Learning Goals

By completing this project, you will demonstrate:
- Document ingestion and chunking strategies
- Vector database setup and management
- Retrieval pipeline with reranking
- Generation with source attribution
- Evaluation of retrieval and generation quality
- Basic deployment

## Requirements

### Functional Requirements
- [ ] Upload and process documents (PDF, TXT, MD)
- [ ] Chunk documents with configurable strategy
- [ ] Store embeddings in vector database
- [ ] Retrieve relevant chunks for queries
- [ ] Generate answers with source citations
- [ ] Handle "I don't know" gracefully

### Technical Requirements
- [ ] Python backend with FastAPI or similar
- [ ] Vector database (Chroma, Pinecone, or pgvector)
- [ ] Claude or OpenAI for generation
- [ ] At least 2 chunking strategies implemented
- [ ] Reranking for improved retrieval
- [ ] Evaluation script for retrieval quality

### Quality Requirements
- [ ] Retrieval Recall@5 > 0.8 on test set
- [ ] No hallucinated sources
- [ ] Response latency < 3 seconds
- [ ] Handles 10+ concurrent users

## Suggested Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAG CHATBOT ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐                         │
│  │   Frontend   │────►│   FastAPI    │                         │
│  │  (optional)  │     │   Backend    │                         │
│  └──────────────┘     └──────┬───────┘                         │
│                              │                                  │
│         ┌────────────────────┼────────────────────┐            │
│         ▼                    ▼                    ▼            │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│  │  Document    │     │   Query      │     │   Answer     │   │
│  │  Processor   │     │   Handler    │     │   Generator  │   │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘   │
│         │                    │                    │            │
│         ▼                    ▼                    ▼            │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│  │  Embedding   │     │  Retriever   │     │   Claude     │   │
│  │   Model      │     │  + Reranker  │     │    API       │   │
│  └──────┬───────┘     └──────┬───────┘     └──────────────┘   │
│         │                    │                                  │
│         └────────────────────┘                                  │
│                    │                                            │
│                    ▼                                            │
│             ┌──────────────┐                                   │
│             │  Vector DB   │                                   │
│             │  (Chroma)    │                                   │
│             └──────────────┘                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
project_1_rag_chatbot/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── ingest.py           # Document processing
│   ├── embeddings.py       # Embedding generation
│   ├── retrieval.py        # Vector search + reranking
│   ├── generation.py       # Answer generation
│   ├── api.py              # FastAPI endpoints
│   └── config.py           # Configuration
├── tests/
│   ├── test_retrieval.py   # Retrieval quality tests
│   ├── test_generation.py  # Generation quality tests
│   └── test_data/          # Test documents and queries
├── evaluation/
│   ├── eval_retrieval.py   # Compute retrieval metrics
│   └── eval_generation.py  # Compute generation metrics
├── deploy/
│   ├── Dockerfile
│   └── modal_deploy.py     # Modal deployment
└── docs/
    └── API.md              # API documentation
```

## Getting Started

### Step 1: Set Up Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export ANTHROPIC_API_KEY="your-key"
```

### Step 2: Implement Document Ingestion

```python
# src/ingest.py - Starter code
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def process_file(self, file_path: str) -> list[dict]:
        """Process a file into chunks."""
        # TODO: Implement file reading (PDF, TXT, MD)
        # TODO: Split into chunks
        # TODO: Add metadata (source, page number, etc.)
        pass
```

### Step 3: Build Retrieval Pipeline

```python
# src/retrieval.py - Starter code
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

class Retriever:
    def __init__(self):
        self.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.db = chromadb.PersistentClient("./chroma_db")
        self.collection = self.db.get_or_create_collection("documents")

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Retrieve and rerank relevant documents."""
        # TODO: Embed query
        # TODO: Vector search (retrieve k*3)
        # TODO: Rerank to top k
        # TODO: Return with scores
        pass
```

### Step 4: Implement Generation

```python
# src/generation.py - Starter code
import anthropic

class AnswerGenerator:
    def __init__(self):
        self.client = anthropic.Anthropic()

    def generate(self, query: str, contexts: list[dict]) -> dict:
        """Generate answer with source attribution."""
        # TODO: Format contexts into prompt
        # TODO: Generate answer
        # TODO: Extract and validate citations
        # TODO: Return answer with sources
        pass
```

### Step 5: Create API

```python
# src/api.py - Starter code
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    confidence: float

@app.post("/upload")
async def upload_document(file: UploadFile):
    """Upload and process a document."""
    # TODO: Implement
    pass

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Answer a question using RAG."""
    # TODO: Implement
    pass
```

### Step 6: Evaluate

```python
# evaluation/eval_retrieval.py - Starter code
def evaluate_retrieval(test_queries: list, ground_truth: list, k: int = 5):
    """Evaluate retrieval quality."""
    # TODO: Compute Recall@k
    # TODO: Compute Precision@k
    # TODO: Compute MRR
    pass
```

### Step 7: Deploy

```python
# deploy/modal_deploy.py - Starter code
import modal

app = modal.App("rag-chatbot")

# TODO: Define image with dependencies
# TODO: Create endpoint
# TODO: Handle document storage
```

## Evaluation Criteria

| Criterion | Weight | Passing |
|-----------|--------|---------|
| Retrieval quality | 25% | Recall@5 > 0.8 |
| Answer accuracy | 25% | No hallucinated facts |
| Source attribution | 15% | All claims have valid sources |
| Code quality | 15% | Clean, documented, tested |
| Deployment | 10% | Working deployed endpoint |
| Documentation | 10% | Clear README, API docs |

## Stretch Goals

- [ ] Add conversation memory (multi-turn)
- [ ] Implement hybrid search (vector + keyword)
- [ ] Add query expansion
- [ ] Create a simple web UI
- [ ] Add document management (delete, update)
- [ ] Implement user feedback collection

## Resources

- [Module 03: Memory Systems](../../modules/module_03_memory_systems/)
- [RAG Architecture Guide](../../modules/module_03_memory_systems/guides/02_rag_architecture_guide.md)
- [Paper Summary: RAG](../../resources/paper_summaries.md#retrieval-augmented-generation)

## Submission

When complete:
1. Ensure all tests pass
2. Deploy to Modal or Railway
3. Document the API
4. Write a brief summary of design decisions

---

*"Build something real. A working RAG system demonstrates more than any benchmark score."*
