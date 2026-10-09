"""
PaperMind MCP Server — lets AI assistants (Claude, Cursor, ChatGPT, ...) search
your research-paper library directly, as tools, via the Model Context Protocol.

Design choice: we expose RETRIEVAL, not generation. The assistant calling us is
already a strong LLM, so we hand it the most relevant passages (with source and
page) and let it do the reasoning. That's cheaper (no second LLM call) and better
(the client's model is stronger than our local llama3.2).

Run over stdio (how Claude Desktop / Claude Code / the MCP Inspector launch it):
    .venv/bin/python app/mcp_server.py
"""

import logging
import sys
from pathlib import Path

# AI clients launch this file from an arbitrary working directory, so make the
# project root importable ourselves instead of relying on `python -m`.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp.server import MCPServer  # MCP Python SDK v2 (FastMCP was renamed to MCPServer)

from app.embed_store import get_vector_store

# IMPORTANT: over stdio, stdout IS the protocol channel between client and server.
# A stray print() corrupts the conversation, so every log line goes to stderr.
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s papermind-mcp: %(message)s",
)
log = logging.getLogger("papermind-mcp")

DATA_DIR = ROOT / "data"
MAX_K = 10  # cap results so one call can't flood the client's context window

mcp = MCPServer("PaperMind")

_store = None


def get_store():
    """Connect to pgvector once and reuse it (loading the embedding model is slow)."""
    global _store
    if _store is None:
        log.info("connecting to the vector store (first call loads the embedding model)")
        _store = get_vector_store()
    return _store


def human_page(metadata):
    """PDF loaders count pages from 0; people count from 1. Cite the page a human would see."""
    if metadata.get("page_label"):
        return str(metadata["page_label"])
    page = metadata.get("page")
    return str(page + 1) if isinstance(page, int) else "?"


@mcp.tool()
def search_papers(query: str, k: int = 5) -> list[dict]:
    """Search the user's research-paper library for passages relevant to `query`.

    Returns up to `k` passages (max 10), most relevant first. Each has the source
    file, the page number, a relevance score from 0 to 1 (higher is more relevant),
    and the passage text. Answer using these passages and cite source + page.
    If nothing relevant comes back, say the library doesn't cover it.
    """
    k = max(1, min(k, MAX_K))
    log.info("search_papers query=%r k=%d", query, k)
    results = get_store().similarity_search_with_score(query, k=k)
    return [
        {
            "source": Path(doc.metadata.get("source", "unknown")).name,
            "page": human_page(doc.metadata),
            # pgvector returns cosine DISTANCE (0 = identical); flip it to a 0-1 relevance
            "relevance": round(1 - float(distance), 3),
            "text": doc.page_content,
        }
        for doc, distance in results
    ]


@mcp.tool()
def list_papers() -> dict:
    """List the PDF papers currently in the user's library."""
    names = sorted(p.name for p in DATA_DIR.glob("*.pdf"))
    return {"count": len(names), "papers": names}


if __name__ == "__main__":
    mcp.run(transport="stdio")
