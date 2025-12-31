# Knowledge Banks for RAG Applications

## What are Knowledge Banks?

Knowledge Banks are Dataiku's managed solution for Retrieval-Augmented Generation (RAG):

```
┌─────────────────────────────────────────────────────────────┐
│                      Knowledge Bank                          │
├─────────────────────────────────────────────────────────────┤
│  Documents → Chunking → Embedding → Vector Store → Retrieval │
│                                                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐ │
│  │  PDFs   │  │  Chunks │  │ Vectors │  │   Query +       │ │
│  │  Text   │→ │  ~500   │→ │  1536d  │→ │   Retrieve      │ │
│  │  HTML   │  │  tokens │  │         │  │   + Generate    │ │
│  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Creating a Knowledge Bank

### From the UI

1. **Project** > **Knowledge Banks** > **+ New Knowledge Bank**
2. Configure settings:

```yaml
knowledge_bank:
  name: commodity_reports_kb
  description: Historical commodity market reports

  # Document sources
  sources:
    - type: folder
      path: /data/reports/eia
    - type: dataset
      name: usda_wasde_reports

  # Chunking settings
  chunking:
    method: recursive
    chunk_size: 500  # tokens
    chunk_overlap: 50

  # Embedding
  embedding:
    model: text-embedding-3-small
    connection: openai-embeddings

  # Vector store
  vector_store:
    type: faiss  # or pinecone, weaviate
    index_type: IVF_FLAT
```

### Programmatic Creation

```python
import dataiku
from dataiku.knowledge_bank import KnowledgeBank

# Create knowledge bank
kb = KnowledgeBank.create(
    project_key="COMMODITY_ANALYSIS",
    name="commodity_reports_kb",
    embedding_connection="openai-embeddings",
    embedding_model="text-embedding-3-small"
)

# Add documents from dataset
kb.add_documents_from_dataset(
    dataset_name="eia_reports",
    text_column="report_text",
    metadata_columns=["report_date", "report_type", "commodity"]
)

# Build the index
kb.build()

print(f"Knowledge bank created with {kb.document_count} documents")
```

## Chunking Strategies

### Fixed Size Chunking

```python
# Simple fixed-size chunks
chunking_config = {
    "method": "fixed",
    "chunk_size": 500,
    "chunk_overlap": 50
}
```

### Semantic Chunking

```python
# Chunk by semantic boundaries
chunking_config = {
    "method": "semantic",
    "max_chunk_size": 1000,
    "similarity_threshold": 0.8,
    "respect_paragraphs": True
}
```

### Document-Specific Chunking

```python
# Different strategies by document type
def custom_chunker(document: dict) -> list:
    """Custom chunking based on document type."""
    doc_type = document.get('type', 'text')
    content = document['content']

    if doc_type == 'eia_report':
        # Chunk by sections
        sections = content.split('\n\n')
        return [{'text': s, 'section': i} for i, s in enumerate(sections)]

    elif doc_type == 'earnings_transcript':
        # Chunk by speaker turns
        import re
        turns = re.split(r'\n(?=[A-Z][a-z]+ [A-Z][a-z]+:)', content)
        return [{'text': t} for t in turns if len(t) > 100]

    else:
        # Default recursive chunking
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        return [{'text': c} for c in splitter.split_text(content)]
```

## Querying Knowledge Banks

### Basic Retrieval

```python
from dataiku.knowledge_bank import KnowledgeBank

kb = KnowledgeBank("commodity_reports_kb")

# Search for relevant chunks
results = kb.search(
    query="What were crude oil inventory changes last week?",
    top_k=5
)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Content: {result.text[:200]}...")
    print(f"Source: {result.metadata.get('source')}")
    print("---")
```

### Filtered Retrieval

```python
# Filter by metadata
results = kb.search(
    query="OPEC production decisions",
    top_k=5,
    filters={
        "report_type": "iea_omr",
        "date_range": {
            "start": "2024-01-01",
            "end": "2024-06-30"
        }
    }
)
```

### Hybrid Search

```python
# Combine vector and keyword search
results = kb.search(
    query="crude inventory draw",
    top_k=5,
    search_type="hybrid",
    keyword_weight=0.3,  # 30% keyword, 70% semantic
    required_keywords=["EIA", "crude"]  # Must contain these
)
```

## RAG Pipeline

### Complete RAG Implementation

```python
from dataiku.llm import LLM
from dataiku.knowledge_bank import KnowledgeBank

class CommodityRAG:
    """RAG system for commodity market questions."""

    def __init__(
        self,
        kb_name: str,
        llm_connection: str,
        top_k: int = 5
    ):
        self.kb = KnowledgeBank(kb_name)
        self.llm = LLM(llm_connection)
        self.top_k = top_k

    def query(self, question: str, filters: dict = None) -> dict:
        """Answer question using RAG."""

        # 1. Retrieve relevant context
        results = self.kb.search(
            query=question,
            top_k=self.top_k,
            filters=filters
        )

        # 2. Format context
        context = "\n\n---\n\n".join([
            f"Source: {r.metadata.get('source', 'Unknown')}\n"
            f"Date: {r.metadata.get('date', 'Unknown')}\n"
            f"Content: {r.text}"
            for r in results
        ])

        # 3. Generate answer
        prompt = f"""Answer the question based on the provided context.
If the context doesn't contain enough information, say so.
Cite specific sources when possible.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

        response = self.llm.complete(prompt, max_tokens=500)

        return {
            'answer': response.text,
            'sources': [
                {
                    'text': r.text[:200],
                    'source': r.metadata.get('source'),
                    'score': r.score
                }
                for r in results
            ]
        }

# Usage
rag = CommodityRAG(
    kb_name="commodity_reports_kb",
    llm_connection="anthropic-claude"
)

result = rag.query(
    "What was the last EIA crude inventory change?",
    filters={"report_type": "eia_wpsr"}
)

print(f"Answer: {result['answer']}")
print(f"\nSources used: {len(result['sources'])}")
```

### Evaluation

```python
def evaluate_rag_response(
    question: str,
    response: dict,
    ground_truth: str = None
) -> dict:
    """Evaluate RAG response quality."""
    from dataiku.llm import LLM

    llm = LLM("anthropic-claude")

    # Check relevance of retrieved sources
    relevance_prompt = f"""Rate how relevant these sources are to the question (1-5):

Question: {question}

Sources:
{chr(10).join([s['text'] for s in response['sources']])}

Return just a number 1-5."""

    relevance = llm.complete(relevance_prompt, max_tokens=10)

    # Check answer quality
    quality_prompt = f"""Rate this answer's quality (1-5):
- Accuracy
- Completeness
- Citation of sources

Question: {question}
Answer: {response['answer']}

Return just a number 1-5."""

    quality = llm.complete(quality_prompt, max_tokens=10)

    return {
        'relevance_score': float(relevance.text.strip()),
        'quality_score': float(quality.text.strip()),
        'num_sources': len(response['sources']),
        'answer_length': len(response['answer'])
    }
```

## Maintaining Knowledge Banks

### Incremental Updates

```python
def update_knowledge_bank(kb_name: str, new_documents: list):
    """Add new documents to existing knowledge bank."""
    kb = KnowledgeBank(kb_name)

    # Add new documents
    for doc in new_documents:
        kb.add_document(
            text=doc['text'],
            metadata=doc.get('metadata', {}),
            document_id=doc.get('id')
        )

    # Rebuild index (incremental)
    kb.rebuild(incremental=True)

    return kb.document_count
```

### Scheduled Refresh

```python
# In a Dataiku scenario
import dataiku
from dataiku.knowledge_bank import KnowledgeBank

def refresh_kb():
    """Daily knowledge bank refresh."""

    kb = KnowledgeBank("commodity_reports_kb")

    # Get new reports from dataset
    ds = dataiku.Dataset("new_reports")
    df = ds.get_dataframe()

    # Add to knowledge bank
    for _, row in df.iterrows():
        kb.add_document(
            text=row['report_text'],
            metadata={
                'date': row['report_date'],
                'type': row['report_type'],
                'source': row['source']
            }
        )

    # Rebuild
    kb.rebuild(incremental=True)

    # Log stats
    print(f"KB now has {kb.document_count} documents")
```

## Key Takeaways

1. **Knowledge Banks manage** the full RAG pipeline: chunking, embedding, storage, retrieval

2. **Chunking strategy matters** - choose based on document type and use case

3. **Metadata enables filtering** - include relevant attributes for targeted retrieval

4. **Hybrid search** combines semantic and keyword matching for better results

5. **Regular maintenance** keeps the knowledge bank current and accurate
