# PaperMind

An AI research assistant that answers questions about a library of AI/ML papers,
with citations — a production-shaped **RAG (Retrieval-Augmented Generation)** service.

## Stack
- **FastAPI** — API
- **LangChain** — RAG orchestration
- **PostgreSQL + pgvector** — vector database
- **Ollama** — local LLM (`llama3.2`) + embeddings (`nomic-embed-text`)
- **Docker + docker-compose** — containers
- **pytest + eval harness** — quality

## Status
Under construction — building step by step.

## Run it (dev)

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    uvicorn app.main:app --reload

Then open http://localhost:8000/health
