"""
Personal Knowledge Assistant - Reference Solution
Compare with your implementation to see different approaches.
"""

import anthropic
import chromadb
import argparse
from pathlib import Path

# Initialize clients
client = anthropic.Anthropic()
chroma = chromadb.PersistentClient(path="./knowledge_db")
collection = chroma.get_or_create_collection(
    name="personal_docs",
    metadata={"hnsw:space": "cosine"}
)


def chunk_document(text: str, chunk_size: int = 800, overlap: int = 200) -> list[str]:
    """Split document into overlapping chunks."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:  # Skip empty chunks
            chunks.append(chunk)

        # Move start with overlap
        start = end - overlap

    return chunks


def add_document(filepath: str):
    """Add a document to the index."""
    path = Path(filepath)

    if not path.exists():
        print(f"File not found: {filepath}")
        return

    # Read file
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print(f"Cannot read {filepath} - not a text file")
        return

    # Chunk the document
    chunks = chunk_document(text)

    if not chunks:
        print(f"No content in {filepath}")
        return

    # Generate IDs and metadata
    ids = [f"{path.name}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": path.name, "path": str(path)} for _ in chunks]

    # Add to ChromaDB
    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    print(f"Added {len(chunks)} chunks from {path.name}")


def add_directory(dirpath: str):
    """Add all documents from a directory."""
    path = Path(dirpath)
    extensions = {".txt", ".md", ".py", ".json", ".csv"}

    count = 0
    for file in path.rglob("*"):
        if file.is_file() and file.suffix in extensions:
            add_document(str(file))
            count += 1

    print(f"Processed {count} files")


def query(question: str, n_results: int = 3) -> dict:
    """Answer a question using RAG."""
    # Search for relevant chunks
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )

    if not results["documents"][0]:
        return {
            "answer": "I don't have any documents indexed yet. Add some with --add",
            "sources": []
        }

    # Build context from retrieved chunks
    context_parts = []
    sources = set()

    for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
        context_parts.append(doc)
        sources.add(metadata["source"])

    context = "\n\n---\n\n".join(context_parts)

    # Call Claude
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system="""You are a helpful assistant that answers questions based on the user's documents.
Only use information from the provided context. If the context doesn't contain the answer, say so.
Be concise but thorough. Mention which documents you found the information in.""",
        messages=[{
            "role": "user",
            "content": f"Context from my documents:\n{context}\n\nQuestion: {question}"
        }]
    )

    return {
        "answer": response.content[0].text,
        "sources": list(sources)
    }


def interactive_mode():
    """Run in interactive mode."""
    print("Personal Knowledge Assistant")
    print("Type 'quit' to exit, 'add <path>' to add documents\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            break

        if user_input.lower().startswith("add "):
            path = user_input[4:].strip()
            p = Path(path)
            if p.is_dir():
                add_directory(path)
            else:
                add_document(path)
            continue

        result = query(user_input)
        print(f"\nAssistant: {result['answer']}")
        if result["sources"]:
            print(f"(Sources: {', '.join(result['sources'])})\n")


def main():
    parser = argparse.ArgumentParser(description="Personal Knowledge Assistant")
    parser.add_argument("--add", help="Add file or directory to index")
    parser.add_argument("-q", "--query", help="Ask a question")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--clear", action="store_true", help="Clear the index")
    args = parser.parse_args()

    if args.clear:
        chroma.delete_collection("personal_docs")
        print("Index cleared.")

    elif args.add:
        path = Path(args.add)
        if path.is_dir():
            add_directory(str(path))
        else:
            add_document(str(args.add))
        print(f"\nIndex now contains {collection.count()} chunks")

    elif args.query:
        result = query(args.query)
        print(f"\nAnswer: {result['answer']}")
        if result["sources"]:
            print(f"\nSources: {', '.join(result['sources'])}")

    elif args.interactive:
        interactive_mode()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
