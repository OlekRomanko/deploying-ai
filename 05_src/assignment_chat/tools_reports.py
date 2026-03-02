"""
Service 2: Semantic Search over Market Research Reports via ChromaDB

Provides semantic search over a collection of institutional market research PDFs
embedded using OpenAI text-embedding-3-small and stored in a ChromaDB
PersistentClient (file-based, no Docker required).

The ChromaDB collection must be populated first by running embed_reports.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import chromadb
from langchain.tools import tool
from dotenv import load_dotenv

from assignment_chat.utils.logger import get_logger
from assignment_chat.utils.embeddings import GatewayEmbeddingFunction

_logs = get_logger(__name__)

_root = Path(__file__).resolve().parent.parent
load_dotenv(_root / ".env")
load_dotenv(_root / ".secrets")

# ChromaDB persistent storage location (inside assignment_chat package)
_HERE = Path(__file__).resolve().parent
CHROMA_DIR = str(_HERE / "chroma_db")
COLLECTION_NAME = "market_reports"

# Cached client/collection
_collection = None


def get_collection():
    """Returns the ChromaDB collection, initializing it once."""
    global _collection
    if _collection is not None:
        return _collection

    if not Path(CHROMA_DIR).exists():
        raise RuntimeError(
            f"ChromaDB directory not found at: {CHROMA_DIR}\n"
            "Please run embed_reports.py first to create the embeddings."
        )

    embedding_fn = GatewayEmbeddingFunction()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    _collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )
    _logs.info(
        f"ChromaDB collection '{COLLECTION_NAME}' loaded with "
        f"{_collection.count()} chunks."
    )
    return _collection


@tool
def search_market_reports(query: str, n_results: int = 3) -> str:
    """
    Searches through curated institutional market research reports using semantic search.

    Use this tool when the user asks about:
    - Market outlooks and forecasts (equity, fixed income, macro)
    - Investment themes and opportunities for 2025/2026
    - Sector analysis and sector-specific outlooks
    - Risk factors and geopolitical considerations
    - Asset allocation strategies and portfolio positioning
    - Long-term investment strategies from institutional research

    The database contains reports from leading financial institutions covering
    global equity markets, sector outlooks, and market strategy.

    Args:
        query: A question or topic to search for in the market research reports.
               Examples: "equity market outlook 2026", "technology sector risks",
               "global recession probability", "interest rate impact on bonds"
        n_results: Number of relevant passages to return (default: 3, max: 5).

    Returns:
        Relevant excerpts from market research reports with source attribution.
    """
    try:
        collection = get_collection()
    except RuntimeError as e:
        _logs.error(f"Collection access failed: {e}")
        return (
            "Market reports database is not yet available. "
            "The embedding process needs to be run first. "
            "Please try the news or stock tools instead."
        )

    n = min(int(n_results), 5)
    _logs.debug(f"Searching market reports for: '{query}' (n_results={n})")

    try:
        results = collection.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        _logs.error(f"ChromaDB query failed: {e}")
        return f"Search failed: {str(e)}"

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        return f"No relevant content found in market reports for: '{query}'"

    # Format results with source attribution
    formatted = [f"## Market Research Insights: '{query}'\n"]

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), 1):
        source = meta.get("source", "Unknown report")
        page = meta.get("page", "")
        page_info = f", p.{page}" if page != "" else ""
        # Convert distance to a relevance indicator
        relevance = "High" if dist < 0.3 else "Medium" if dist < 0.5 else "Moderate"

        formatted.append(
            f"### Excerpt {i} — *{source}{page_info}* (Relevance: {relevance})\n"
            f"{doc.strip()}\n"
        )

    formatted.append(
        "\n*Sources: Institutional market research reports in the FinAI database.*"
    )
    return "\n".join(formatted)
