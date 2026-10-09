"""
PaperMind API + web app.

Endpoints:
  GET  /         -> the web page (frontend/index.html)
  GET  /health   -> health check
  GET  /papers   -> list the PDFs currently in the library
  POST /ask      -> ask a question (RAG pipeline)
  POST /upload   -> upload a PDF; it's ingested into pgvector on the fly
"""

import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.embed_store import ingest_pdf_file
from app.rag import answer_question

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "index.html"

app = FastAPI(
    title="PaperMind",
    description="A RAG-powered research assistant for AI/ML papers.",
    version="0.2.0",
)


@app.get("/")
def home():
    """Serve the web page."""
    return FileResponse(FRONTEND)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/papers")
def papers():
    """List the PDFs currently in the library."""
    DATA_DIR.mkdir(exist_ok=True)
    names = sorted(p.name for p in DATA_DIR.glob("*.pdf"))
    return {"count": len(names), "papers": names}


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(request: AskRequest):
    """Answer a question using the RAG pipeline."""
    answer, sources = answer_question(request.question)
    return {
        "question": request.question,
        "answer": answer,
        "sources": [
            {
                "source": os.path.basename(doc.metadata.get("source", "unknown")),
                "page": doc.metadata.get("page"),
            }
            for doc in sources
        ],
    }


@app.post("/upload")
async def upload(file: Annotated[UploadFile, File()]):
    """Accept a PDF, save it to data/, and ingest it into the vector store."""
    # Never trust a client-supplied filename: keep only its final component, so a
    # name like "../../app/main.py" can't write outside data/ (path traversal).
    name = Path(file.filename or "").name
    if not name.lower().endswith(".pdf"):
        return {"ok": False, "error": "Only PDF files are supported."}

    content = await file.read()
    if not content.startswith(b"%PDF-"):  # check the real file signature, not just the extension
        return {"ok": False, "error": "That file isn't a valid PDF."}

    DATA_DIR.mkdir(exist_ok=True)
    dest = DATA_DIR / name
    dest.write_bytes(content)

    try:
        n_chunks = ingest_pdf_file(dest)
    except Exception as exc:  # noqa: BLE001 - API boundary: report the problem, keep serving
        dest.unlink(missing_ok=True)  # don't leave an unindexed file in the library
        return {"ok": False, "error": f"Failed to process PDF: {exc}"}

    return {"ok": True, "filename": name, "chunks_added": n_chunks}
