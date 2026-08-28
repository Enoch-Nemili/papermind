"""
PaperMind API — entry point.

This file defines our web application using FastAPI.
Right now it only has two simple endpoints so we can prove the server works.
We'll grow it into the full RAG (Retrieval-Augmented Generation) service step by step.
"""

from fastapi import FastAPI

# `app` is our application object. FastAPI uses it to know about all our endpoints.
app = FastAPI(
    title="PaperMind",
    description="A RAG-powered research assistant for AI/ML papers.",
    version="0.1.0",
)


@app.get("/")
def root():
    """The home endpoint. Visiting http://localhost:8000/ shows this message."""
    return {"message": "PaperMind is running. Visit /docs to explore the API."}


@app.get("/health")
def health():
    """
    A 'health check' endpoint. Real services have one of these so monitoring
    tools (and load balancers) can ask 'are you alive?' and get a quick yes.
    """
    return {"status": "ok"}
