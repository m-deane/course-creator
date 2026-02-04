"""
RAG Pipeline Template - Copy and customize for your use case
Works with: Anthropic Claude + ChromaDB
Time to working: 5 minutes

Usage:
    python rag_template.py --add path/to/docs/
    python rag_template.py --query "What is X?"
"""

import anthropic
import chromadb
import argparse
import logging
from pathlib import Path

# ============================================================
# CUSTOMIZE THESE
# ============================================================

COLLECTION_NAME = "my_documents"  # TODO: Change for your project
PERSIST_DIR = "./chroma_db"  # TODO: Where to store vectors
CHUNK_SIZE = 1000  # TODO: Characters per chunk
CHUNK_OVERLAP = 200  # TODO: Overlap between chunks
N_RESULTS = 3  # TODO: How many chunks to retrieve
MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = """Answer the question based only on the provided context.
If the context doesn't contain the answer, say "I don't have information about that."
Be concise and direct."""

# ============================================================
# COPY THIS ENTIRE BLOCK (production-ready)
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """Production-ready RAG pipeline."""

    def __init__(self, collection_name: str = COLLECTION_NAME):
        self.client = anthropic.Anthropic()
        self.chroma = chromadb.PersistentClient(path=PERSIST_DIR)
        self.collection = self.chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"RAG initialized with {self.collection.count()} documents")

    def chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end - CHUNK_OVERLAP
        return chunks

    def add_document(self, text: str, doc_id: str, metadata: dict = None):
        """Add a single document to the vector store."""
        chunks = self.chunk_text(text)
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": doc_id, "chunk": i, **(metadata or {})} for i in range(len(chunks))]

        self.collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        logger.info(f"Added {len(chunks)} chunks from {doc_id}")

    def add_file(self, filepath: str):
        """Add a file to the vector store."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        text = path.read_text(encoding="utf-8")
        self.add_document(text, doc_id=path.name, metadata={"path": str(path)})

    def add_directory(self, dirpath: str, extensions: list[str] = None):
        """Add all files from a directory."""
        extensions = extensions or [".txt", ".md", ".py", ".json"]
        path = Path(dirpath)

        for file in path.rglob("*"):
            if file.is_file() and file.suffix in extensions:
                try:
                    self.add_file(str(file))
                except Exception as e:
                    logger.error(f"Failed to add {file}: {e}")

    def query(self, question: str, n_results: int = N_RESULTS) -> dict:
        """Query the RAG pipeline."""
        # Retrieve relevant chunks
        results = self.collection.query(query_texts=[question], n_results=n_results)

        if not results["documents"][0]:
            return {"answer": "No relevant documents found.", "sources": []}

        # Build context
        context_parts = []
        sources = []
        for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
            context_parts.append(doc)
            sources.append(metadata.get("source", "unknown"))

        context = "\n---\n".join(context_parts)

        # Generate answer
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }]
        )

        return {
            "answer": response.content[0].text,
            "sources": list(set(sources)),
            "context_used": len(context_parts)
        }

    def clear(self):
        """Clear all documents from the collection."""
        self.chroma.delete_collection(self.collection.name)
        self.collection = self.chroma.create_collection(self.collection.name)
        logger.info("Collection cleared")


# ============================================================
# RUN IT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Pipeline")
    parser.add_argument("--add", help="Add file or directory to index")
    parser.add_argument("--query", "-q", help="Query the RAG system")
    parser.add_argument("--clear", action="store_true", help="Clear all documents")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    rag = RAGPipeline()

    if args.clear:
        rag.clear()
        print("Collection cleared.")

    elif args.add:
        path = Path(args.add)
        if path.is_dir():
            rag.add_directory(str(path))
        else:
            rag.add_file(str(path))
        print(f"Added to index. Total documents: {rag.collection.count()}")

    elif args.query:
        result = rag.query(args.query)
        print(f"\nAnswer: {result['answer']}")
        print(f"\nSources: {', '.join(result['sources'])}")

    elif args.interactive:
        print("RAG Interactive Mode. Type 'quit' to exit.\n")
        while True:
            question = input("Question: ").strip()
            if question.lower() in ["quit", "exit", "q"]:
                break
            if not question:
                continue
            result = rag.query(question)
            print(f"\nAnswer: {result['answer']}")
            print(f"Sources: {', '.join(result['sources'])}\n")

    else:
        parser.print_help()
