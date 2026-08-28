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

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.rag import answer_question
from app.embed_store import ingest_pdf_file

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
async def upload(file: UploadFile = File(...)):
    """Accept a PDF, save it to data/, and ingest it into the vector store."""
    if not file.filename.lower().endswith(".pdf"):
        return {"ok": False, "error": "Only PDF files are supported."}

    DATA_DIR.mkdir(exist_ok=True)
    dest = DATA_DIR / file.filename
    dest.write_bytes(await file.read())

    try:
        n_chunks = ingest_pdf_file(dest)
    except Exception as exc:  # keep the server alive and report the problem
        return {"ok": False, "error": f"Failed to process PDF: {exc}"}

    return {"ok": True, "filename": file.filename, "chunks_added": n_chunks}
