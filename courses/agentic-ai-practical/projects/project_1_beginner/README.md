# Project 1: Personal Knowledge Assistant

**Build a chatbot that answers questions about YOUR documents.**

## What You'll Build

A CLI tool that:
1. Indexes your documents (PDFs, markdown, text files)
2. Answers questions about them
3. Cites which document the answer came from

## Time: 2-3 hours

## Demo

```bash
$ python assistant.py --add ~/Documents/notes/
Added 47 documents to index.

$ python assistant.py -q "What did I write about Python decorators?"

Answer: In your notes from March 2024, you wrote that decorators are
"functions that wrap other functions to add behavior without modifying
the original function." You included examples of @property and @staticmethod.

Sources: python_notes.md, advanced_python.md
```

## What You'll Learn

- RAG pipeline implementation
- Document chunking strategies
- Vector database usage
- Building CLI tools

## Getting Started

1. Copy the starter code below
2. Fill in the `# TODO` sections
3. Test with your own documents
4. Deploy it (optional)

## Starter Code

```python
"""
Personal Knowledge Assistant
Your TODO items are marked below.
"""

import anthropic
import chromadb
import argparse
from pathlib import Path

# TODO 1: Initialize your clients
client = None  # anthropic.Anthropic()
chroma = None  # chromadb.PersistentClient(...)
collection = None  # chroma.get_or_create_collection(...)


def chunk_document(text: str, chunk_size: int = 800) -> list[str]:
    """Split document into chunks."""
    # TODO 2: Implement chunking with overlap
    # Hint: Use overlapping windows for better context
    pass


def add_document(filepath: str):
    """Add a document to the index."""
    # TODO 3: Read file, chunk it, add to ChromaDB
    # Hint: Store filepath in metadata for citations
    pass


def add_directory(dirpath: str):
    """Add all documents from a directory."""
    # TODO 4: Walk directory, filter by extension, add each file
    pass


def query(question: str, n_results: int = 3) -> dict:
    """Answer a question using RAG."""
    # TODO 5: Search ChromaDB, build context, call Claude
    # Return: {"answer": str, "sources": list[str]}
    pass


def main():
    parser = argparse.ArgumentParser(description="Personal Knowledge Assistant")
    parser.add_argument("--add", help="Add file or directory")
    parser.add_argument("-q", "--query", help="Ask a question")
    args = parser.parse_args()

    if args.add:
        path = Path(args.add)
        if path.is_dir():
            add_directory(str(path))
        else:
            add_document(str(path))
        print(f"Index now contains {collection.count()} chunks")

    elif args.query:
        result = query(args.query)
        print(f"\nAnswer: {result['answer']}")
        print(f"\nSources: {', '.join(result['sources'])}")


if __name__ == "__main__":
    main()
```

## Solution

Once you've tried it yourself, check [solution.py](solution.py) for a reference implementation.

## Extend It (Optional)

- Add a web interface with Streamlit
- Support PDF parsing with PyPDF2
- Add conversation memory
- Deploy to Hugging Face Spaces

## Share Your Work

Built something cool? Share it:
- Add to your GitHub portfolio
- Post on LinkedIn/Twitter
- Submit a PR to add your extension to this repo
