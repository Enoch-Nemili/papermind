"""
PaperMind MCP Server — lets AI assistants (Claude, Cursor, ChatGPT, ...) search and
grow your research-paper library through the Model Context Protocol.

It offers all three MCP building blocks:
  - TOOLS     (the AI decides to call them):  search_papers, list_papers, add_paper
  - RESOURCES (you/the app attach them):     papermind://library, papermind://papers/{filename}
  - PROMPTS   (you pick them like a command): answer_from_papers

Design choice: we expose RETRIEVAL, not generation. The assistant calling us is
already a strong LLM, so we hand it the most relevant passages (with source and
page) and let it reason. Cheaper (no second LLM call) and better answers.

Run it two ways:
    .venv/bin/python app/mcp_server.py           # stdio: an AI app on this machine launches it
    .venv/bin/python app/mcp_server.py --http    # Streamable HTTP: a long-running network service
                                                 # at http://127.0.0.1:8765/mcp that many clients share
"""

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field

# AI clients launch this file from an arbitrary working directory, so make the
# project root importable ourselves instead of relying on `python -m`.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp.server import MCPServer  # MCP Python SDK v2 (FastMCP was renamed to MCPServer)
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pypdf import PdfReader

from app.embed_store import get_vector_store, ingest_pdf_file

# IMPORTANT: over stdio, stdout IS the protocol channel between client and server.
# A stray print() corrupts the conversation, so every log line goes to stderr.
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s papermind-mcp: %(message)s",
)
log = logging.getLogger("papermind-mcp")

DATA_DIR = ROOT / "data"    # the library: every PDF here is indexed
INBOX_DIR = ROOT / "inbox"  # the ONLY folder add_paper is allowed to read from
MAX_K = 10                  # cap results so one call can't flood the client's context
MAX_PDF_MB = 50

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


# Typed results: the SDK turns these into each tool's output schema, so clients get
# structured, predictable data instead of a blob of JSON text.
class Passage(BaseModel):
    source: str = Field(description="PDF file the passage came from.")
    page: str = Field(description="Page number as a reader would see it.")
    relevance: float = Field(description="0 to 1; higher is more relevant.")
    text: str = Field(description="The passage itself.")


class LibraryListing(BaseModel):
    count: int
    papers: list[str]


class AddPaperResult(BaseModel):
    added: bool
    paper: str
    chunks_indexed: int | None = None
    reason: str | None = None

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


def file_inside(folder: Path, filename: str) -> Path:
    """Resolve `filename` inside `folder`, refusing anything that escapes it.

    Blocks path tricks like '../../.ssh/id_rsa' and symlinks pointing elsewhere:
    resolve() follows them, so the real location must still sit directly in `folder`.
    """
    path = (folder / filename).resolve()
    if path.parent != folder.resolve():
        raise ToolError(f"Only files directly inside {folder.name}/ are allowed.")
    return path


# ----------------------------------------------------------------- TOOLS ---

@mcp.tool(title="Search papers", annotations=READ_ONLY)
def search_papers(
    query: Annotated[str, Field(description="What to look for. Use the technical terms the papers themselves would use.")],
    k: Annotated[int, Field(ge=1, le=MAX_K, description="How many passages to return (1-10).")] = 5,
) -> list[Passage]:
    """Search the user's research-paper library for passages relevant to `query`.

    Returns passages most-relevant first, each with the source file, page number,
    a relevance score from 0 to 1, and the passage text. Answer using these passages
    and cite source + page.

    Retrieval is semantic, and it works best with the papers' own vocabulary: search
    "scaled dot-product attention queries keys values", not "how does attention work".
    If results look weak, rephrase and search again. If nothing relevant comes back,
    say the library doesn't cover it.
    """
    log.info("search_papers query=%r k=%d", query, k)
    results = get_store().similarity_search_with_score(query, k=k)
    return [
        Passage(
            source=Path(doc.metadata.get("source", "unknown")).name,
            page=human_page(doc.metadata),
            # pgvector returns cosine DISTANCE (0 = identical); flip it to a 0-1 relevance
            relevance=round(1 - float(distance), 3),
            text=doc.page_content,
        )
        for doc, distance in results
    ]


@mcp.tool(title="List papers", annotations=READ_ONLY)
def list_papers() -> LibraryListing:
    """List the PDF papers currently in the user's library."""
    names = sorted(p.name for p in DATA_DIR.glob("*.pdf"))
    return LibraryListing(count=len(names), papers=names)


@mcp.tool(
    title="Add a paper",
    annotations=ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,   # only adds; never deletes or overwrites
        idempotent_hint=True,     # adding the same file twice is a no-op
        open_world_hint=False,
    ),
)
def add_paper(
    filename: Annotated[str, Field(description="Name of a PDF the user put in PaperMind's inbox/ folder, e.g. 'mamba.pdf'.")],
) -> AddPaperResult:
    """Add a PDF from PaperMind's inbox/ folder to the searchable library.

    Only files the user has placed in inbox/ can be added; this tool cannot read
    anything else on the machine. If the file isn't there, the error lists the PDFs
    that are waiting in inbox/.
    """
    INBOX_DIR.mkdir(exist_ok=True)
    src = file_inside(INBOX_DIR, filename)

    if src.suffix.lower() != ".pdf":
        raise ToolError("Only .pdf files can be added.")
    if not src.is_file():
        waiting = sorted(p.name for p in INBOX_DIR.glob("*.pdf"))
        raise ToolError(f"{filename!r} isn't in inbox/. PDFs waiting there: {waiting or 'none'}.")
    if src.stat().st_size > MAX_PDF_MB * 1024 * 1024:
        raise ToolError(f"{src.name} is larger than {MAX_PDF_MB} MB.")
    with src.open("rb") as f:
        if f.read(5) != b"%PDF-":  # check the real file signature, not just the extension
            raise ToolError(f"{src.name} isn't a valid PDF.")

    dest = DATA_DIR / src.name
    if dest.exists():
        return AddPaperResult(added=False, paper=src.name, reason="already in the library")

    shutil.copy2(src, dest)
    try:
        chunks = ingest_pdf_file(dest)
    except Exception as exc:
        dest.unlink(missing_ok=True)  # roll back so the library and the index stay in sync
        log.exception("indexing failed for %s", src.name)
        raise ToolError(f"Couldn't index {src.name}: {exc}") from exc

    log.info("add_paper added %s (%d chunks)", src.name, chunks)
    return AddPaperResult(added=True, paper=src.name, chunks_indexed=chunks)


# ------------------------------------------------------------- RESOURCES ---

def paper_summary(path: Path) -> dict:
    reader = PdfReader(str(path))
    title = (reader.metadata.title if reader.metadata else None) or path.stem
    return {
        "file": path.name,
        "title": title,
        "pages": len(reader.pages),
        "size_kb": round(path.stat().st_size / 1024),
    }


@mcp.resource("papermind://library")
def library() -> str:
    """Catalog of every paper in the library: file, title, page count, size."""
    papers = [paper_summary(p) for p in sorted(DATA_DIR.glob("*.pdf"))]
    return json.dumps({"count": len(papers), "papers": papers}, indent=2)


@mcp.resource("papermind://papers/{filename}")
def paper_info(filename: str) -> str:
    """Details for one paper in the library (title, page count, size)."""
    path = file_inside(DATA_DIR, filename)
    if not path.is_file():
        raise ToolError(f"{filename!r} isn't in the library.")
    return json.dumps(paper_summary(path), indent=2)


# --------------------------------------------------------------- PROMPTS ---

@mcp.prompt(title="Answer from my papers")
def answer_from_papers(
    question: Annotated[str, Field(description="Your research question.")],
) -> str:
    """Answer a question using only the PaperMind library, with page-level citations."""
    return (
        f"Answer this question using ONLY my PaperMind library: {question}\n\n"
        "1. Call search_papers 1-3 times, phrasing each query in the technical "
        "vocabulary the papers would use.\n"
        "2. Answer only from the passages returned, and cite every claim as (paper, p. N).\n"
        "3. If the passages don't cover something, say the library doesn't cover it. "
        "Don't fill gaps from memory."
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PaperMind MCP server")
    parser.add_argument("--http", action="store_true",
                        help="serve over Streamable HTTP instead of stdio")
    parser.add_argument("--host", default="127.0.0.1",
                        help="interface to bind (default: localhost only, so nothing else on the network can reach it)")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if args.http:
        log.info("serving MCP over HTTP at http://%s:%d/mcp", args.host, args.port)
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
