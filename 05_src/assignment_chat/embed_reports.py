"""
embed_reports.py — One-time script to embed Market Research PDFs into ChromaDB

This script:
1. Loads all PDF files from the Assg2/Market_reports/ directory
2. Splits them into overlapping text chunks
3. Embeds each chunk using OpenAI text-embedding-3-small
4. Stores the embeddings in a ChromaDB PersistentClient database

Run this script ONCE before starting the FinAI chatbot app.
The resulting database is stored in: 05_src/assignment_chat/chroma_db/

Usage (from Assg2 root directory):
    python 05_src/assignment_chat/embed_reports.py

Requirements:
    - OPENAI_API_KEY must be set in Assg2/.secrets
    - langchain-community must be installed (for PyPDFLoader)
    - chromadb must be installed
"""

import sys
from pathlib import Path

# Add 05_src to path
_src_dir = str(Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import os
import chromadb
from dotenv import load_dotenv
from assignment_chat.utils.embeddings import GatewayEmbeddingFunction

# Load secrets from 05_src
_src = Path(__file__).resolve().parent.parent
load_dotenv(_src / ".env")
load_dotenv(_src / ".secrets")

# Paths
_HERE = Path(__file__).resolve().parent
CHROMA_DIR = str(_HERE / "chroma_db")
REPORTS_DIR = _src.parent / "Market_reports"
COLLECTION_NAME = "market_reports"

# Chunking configuration
CHUNK_SIZE = 1000       # characters per chunk
CHUNK_OVERLAP = 200     # overlap between consecutive chunks


def load_and_chunk_pdfs():
    """Loads all PDFs from Market_reports/ and splits into text chunks."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError as e:
        raise ImportError(
            "langchain-community is required for PDF loading. "
            f"Install it with: pip install langchain-community\nError: {e}"
        )
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError as e:
            raise ImportError(
                "langchain-text-splitters is required. "
                f"Install it with: pip install langchain-text-splitters\nError: {e}"
            )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    pdf_files = sorted(REPORTS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {REPORTS_DIR}")

    print(f"Found {len(pdf_files)} PDF file(s) in {REPORTS_DIR}")

    all_chunks = []
    for pdf_file in pdf_files:
        print(f"\nLoading: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()
            print(f"  Pages loaded: {len(pages)}")
        except Exception as e:
            print(f"  ERROR loading {pdf_file.name}: {e}")
            continue

        chunks = splitter.split_documents(pages)

        # Assign unique IDs and enrich metadata
        stem = pdf_file.stem
        for i, chunk in enumerate(chunks):
            chunk.metadata["source"] = pdf_file.name
            chunk.metadata["report_name"] = stem
            # chunk_id must be unique across all PDFs
            chunk.metadata["chunk_id"] = f"{stem}_{i:04d}"

        print(f"  Chunks created: {len(chunks)}")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks across all reports: {len(all_chunks)}")
    return all_chunks


def create_chroma_collection(chunks):
    """Creates and populates a ChromaDB PersistentClient collection."""
    print(f"\nInitializing ChromaDB at: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if present (allows re-running the script)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}' (rebuilding).")

    embedding_fn = GatewayEmbeddingFunction()

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    # Add documents in batches to avoid rate limits
    BATCH_SIZE = 50
    total = len(chunks)
    print(f"\nEmbedding {total} chunks (batch size={BATCH_SIZE})...")

    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        collection.add(
            documents=[c.page_content for c in batch],
            metadatas=[c.metadata for c in batch],
            ids=[c.metadata["chunk_id"] for c in batch]
        )
        print(f"  Batch {batch_num}/{total_batches}: added {len(batch)} chunks")

    final_count = collection.count()
    print(f"\nDone! ChromaDB collection '{COLLECTION_NAME}' contains {final_count} chunks.")
    print(f"Database location: {CHROMA_DIR}")
    return collection


def verify_collection():
    """Quick verification: runs a test query against the populated collection."""
    embedding_fn = GatewayEmbeddingFunction()
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )
    results = collection.query(
        query_texts=["equity market outlook 2026"],
        n_results=2
    )
    print("\n--- Verification Query: 'equity market outlook 2026' ---")
    for i, (doc, meta) in enumerate(
        zip(results["documents"][0], results["metadatas"][0]), 1
    ):
        preview = doc[:200].encode("ascii", errors="replace").decode("ascii")
        print(f"Result {i}: [{meta.get('source', 'N/A')}] {preview}...")
    print("Verification complete.")


if __name__ == "__main__":
    print("=" * 60)
    print("FinAI — Market Reports Embedding Script")
    print("=" * 60)

    chunks = load_and_chunk_pdfs()
    collection = create_chroma_collection(chunks)
    verify_collection()

    print("\nSetup complete. You can now run the FinAI chatbot:")
    print("  cd 05_src/assignment_chat")
    print("  python app.py")
