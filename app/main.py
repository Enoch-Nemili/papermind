"""
PaperMind API — entry point.

Endpoints:
  GET  /         -> welcome message
  GET  /health   -> simple "am I alive?" check
  POST /ask      -> ask a question about the papers (the RAG pipeline)
"""

import os

from fastapi import FastAPI
from pydantic import BaseModel

from app.rag import answer_question

app = FastAPI(
    title="PaperMind",
    description="A RAG-powered research assistant for AI/ML papers.",
    version="0.1.0",
)


@app.get("/")
def root():
    """The home endpoint."""
    return {"message": "PaperMind is running. Visit /docs to explore the API."}


@app.get("/health")
def health():
    """A health check endpoint."""
    return {"status": "ok"}


# Describes the JSON body the /ask endpoint expects: {"question": "..."}
class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(request: AskRequest):
    """
    Answer a question about the papers using the RAG pipeline:
    retrieve relevant chunks from pgvector, then let llama3.2 answer using them.
    """
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
