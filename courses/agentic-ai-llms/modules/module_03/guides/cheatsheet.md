# Module 3: Memory & Context Management Cheatsheet

> **Reading time:** ~5 min | **Module:** 3 — Memory & Context | **Prerequisites:** Module 3 guides

## Key Concepts

| Concept | Definition |
|---------|-----------|
| **Conversation Memory** | Storing and managing chat history within context window limits |
| **RAG** | Retrieval-Augmented Generation: enhancing LLM responses with retrieved context |
| **Vector Database** | Database optimized for storing and searching high-dimensional embeddings |
| **Embedding** | Dense numerical vector representing semantic meaning of text |
| **Chunking** | Splitting documents into smaller pieces for embedding and retrieval |
| **Semantic Search** | Finding information based on meaning rather than keyword matching |
| **Hybrid Search** | Combining keyword search and vector search for better retrieval |
| **Reranking** | Re-scoring retrieved results to improve relevance ordering |
| **Context Window** | Maximum tokens LLM can process (input + output) in single request |

## Common Patterns

### Basic Conversation Memory (Buffer)

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>

```python
class ConversationBuffer:
    def __init__(self, max_messages=10):
        self.messages = []
        self.max_messages = max_messages

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
        # Keep only recent messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_messages(self):
        return self.messages

# Usage
memory = ConversationBuffer(max_messages=10)
memory.add_message("user", "What's the weather?")
memory.add_message("assistant", "It's sunny today.")

# Send to LLM
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=memory.get_messages()
)
```

</div>
</div>

### Conversation Summary Memory
```python
class SummaryMemory:
    def __init__(self, client, summary_threshold=5):
        self.client = client
        self.messages = []
        self.summary = ""
        self.summary_threshold = summary_threshold

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

        # Summarize when threshold reached
        if len(self.messages) >= self.summary_threshold:
            self.summary = self._create_summary()
            self.messages = []  # Clear buffer after summarizing

    def _create_summary(self):
        conversation = "\n".join([f"{m['role']}: {m['content']}" for m in self.messages])
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": f"Summarize this conversation concisely:\n\n{conversation}"
            }]
        )
        return response.content[0].text

    def get_context(self):
        context = ""
        if self.summary:
            context += f"Previous conversation summary: {self.summary}\n\n"
        context += "Recent messages:\n" + "\n".join([f"{m['role']}: {m['content']}" for m in self.messages])
        return context
```

### Basic RAG Pipeline
```python
import chromadb
from chromadb.utils import embedding_functions

# 1. Setup vector database
client = chromadb.Client()
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-key",
    model_name="text-embedding-3-small"
)

collection = client.create_collection(
    name="knowledge_base",
    embedding_function=openai_ef
)

# 2. Add documents
documents = [
    "The company was founded in 2010 in San Francisco.",
    "We have 500 employees across 3 offices.",
    "Our main product is cloud-based analytics software."
]

collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

# 3. Query and retrieve
def rag_query(question):
    # Retrieve relevant context
    results = collection.query(
        query_texts=[question],
        n_results=3
    )

    context = "\n".join(results['documents'][0])

    # Generate answer with context
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""Answer the question based on the context below.

Context:
{context}

Question: {question}

Answer:"""
        }]
    )

    return response.content[0].text

# Usage
answer = rag_query("How many employees does the company have?")
```

### Document Chunking Strategies
```python

# Fixed-size chunking with overlap
def chunk_text_fixed(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# Semantic chunking (by paragraphs)
def chunk_text_semantic(text):
    return [p.strip() for p in text.split('\n\n') if p.strip()]

# Recursive chunking
def chunk_text_recursive(text, max_chunk_size=1000):
    if len(text) <= max_chunk_size:
        return [text]

    # Try splitting by paragraphs
    paragraphs = text.split('\n\n')
    if all(len(p) <= max_chunk_size for p in paragraphs):
        return paragraphs

    # Split paragraphs further by sentences
    chunks = []
    for para in paragraphs:
        if len(para) <= max_chunk_size:
            chunks.append(para)
        else:
            sentences = para.split('. ')
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= max_chunk_size:
                    current_chunk += sentence + ". "
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
            if current_chunk:
                chunks.append(current_chunk.strip())

    return chunks
```

### Hybrid Search (Keyword + Vector)
```python
from rank_bm25 import BM25Okapi
import numpy as np

class HybridSearch:
    def __init__(self, documents, embedding_function):
        self.documents = documents
        self.embedding_function = embedding_function

        # Setup BM25 for keyword search
        tokenized = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

        # Setup vector search
        self.embeddings = embedding_function(documents)

    def search(self, query, top_k=5, alpha=0.5):
        # Keyword search
        bm25_scores = self.bm25.get_scores(query.split())

        # Vector search
        query_embedding = self.embedding_function([query])[0]
        vector_scores = np.dot(self.embeddings, query_embedding)

        # Normalize scores
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-10)
        vector_scores = (vector_scores - vector_scores.min()) / (vector_scores.max() - vector_scores.min() + 1e-10)

        # Combine scores
        combined_scores = alpha * bm25_scores + (1 - alpha) * vector_scores

        # Get top-k results
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]
```

### Context Window Management
```python
import tiktoken

def manage_context_window(messages, max_tokens=180000, model="claude-3-5-sonnet-20241022"):
    # Estimate tokens (rough approximation for Claude)
    total_tokens = sum(len(m['content']) // 4 for m in messages)

    if total_tokens <= max_tokens:
        return messages

    # Keep system message and recent messages
    system_msg = [m for m in messages if m['role'] == 'system']
    other_msgs = [m for m in messages if m['role'] != 'system']

    # Calculate tokens for recent messages
    recent_tokens = 0
    keep_count = 0
    for msg in reversed(other_msgs):
        msg_tokens = len(msg['content']) // 4
        if recent_tokens + msg_tokens <= max_tokens * 0.8:  # Use 80% of budget
            recent_tokens += msg_tokens
            keep_count += 1
        else:
            break

    # Keep most recent messages
    recent_msgs = other_msgs[-keep_count:]

    # Summarize older messages if any
    if len(other_msgs) > keep_count:
        older_msgs = other_msgs[:-keep_count]
        summary = summarize_conversation(older_msgs)
        summary_msg = {"role": "system", "content": f"Previous conversation summary: {summary}"}
        return system_msg + [summary_msg] + recent_msgs

    return system_msg + recent_msgs
```

## RAG Best Practices

### Chunk Size Guidelines

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>

```python

# Use case based chunking
CHUNK_SIZES = {
    "qa": 400,              # Question answering
    "summarization": 1000,  # Document summarization
    "code": 200,            # Code search
    "chat": 600             # Conversational context
}

OVERLAP = {
    "qa": 50,
    "summarization": 200,
    "code": 20,
    "chat": 100
}
```

</div>
</div>

### Metadata for Filtering
```python

# Add metadata to improve retrieval
collection.add(
    documents=["Document text..."],
    metadatas=[{
        "source": "company_handbook.pdf",
        "page": 15,
        "section": "Benefits",
        "date": "2024-01-15",
        "category": "HR"
    }],
    ids=["doc_1"]
)

# Query with metadata filtering
results = collection.query(
    query_texts=["What are the benefits?"],
    n_results=5,
    where={"category": "HR"}  # Filter by metadata
)
```

### Citation and Source Tracking
```python
def rag_with_citations(question):
    results = collection.query(
        query_texts=[question],
        n_results=3
    )

    # Build context with source markers
    context_parts = []
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        context_parts.append(f"[Source {i+1}] {doc}")

    context = "\n\n".join(context_parts)

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""Answer based on the sources below. Cite sources using [Source N] notation.

{context}

Question: {question}"""
        }]
    )

    return {
        "answer": response.content[0].text,
        "sources": results['metadatas'][0]
    }
```

## Gotchas

- **Embedding model consistency** - Always use same embedding model for indexing and querying


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>

```python

# Bad: Different models

# Index with: text-embedding-3-small

# Query with: text-embedding-ada-002

# Good: Same model
EMBEDDING_MODEL = "text-embedding-3-small"

# Use everywhere
```


- **Chunk size too large** - Dilutes relevant information

```python

# Bad: 5000 token chunks (too much irrelevant context)

# Good: 400-1000 token chunks (focused context)
```

- **Chunk size too small** - Loses surrounding context

```python

# Bad: 50 token chunks (fragments sentences)

# Good: 200-400 token minimum for coherent meaning
```

- **No overlap in chunking** - Splits concepts across chunks

```python

# Bad: No overlap
chunks = [text[0:500], text[500:1000], ...]

# Good: 10-20% overlap
chunks = [text[0:500], text[450:950], ...]
```

- **Ignoring metadata** - Harder to filter and debug

```python

# Bad: No metadata
collection.add(documents=[doc], ids=["1"])

# Good: Rich metadata
collection.add(
    documents=[doc],
    ids=["1"],
    metadatas=[{"source": "file.pdf", "page": 3, "type": "technical"}]
)
```

- **Top-k too low** - Misses relevant context

```python

# Bad: Only retrieve 1 document
results = collection.query(query, n_results=1)

# Good: Retrieve 3-5, let LLM select relevant parts
results = collection.query(query, n_results=5)
```

- **Context position bias** - LLMs pay more attention to start/end of context

```python

# Solution: Put most relevant retrieved docs at start and end
context = f"{most_relevant}\n\n{other_docs}\n\n{second_most_relevant}"
```

- **Stale embeddings** - Documents updated but embeddings not refreshed

```python

# Solution: Track document versions
metadata = {
    "doc_id": "123",
    "version": 2,
    "updated_at": "2024-01-15"
}

# Periodic re-embedding
if doc_updated_since(last_embed_time):
    collection.update(ids=[doc_id], documents=[new_content])
```

- **Vector database memory** - Large collections consume RAM

```python

# Solution: Use persistent storage
client = chromadb.PersistentClient(path="/path/to/db")

# Or use cloud vector DB for production

# Pinecone, Weaviate, Qdrant, etc.
```
