"""
PaperMind configuration.

Picks the models + database from environment variables, with local-friendly defaults.

- Embeddings: fastembed (a small local model, no API key, no rate limits, runs anywhere).
- Chat model: Ollama locally / Gemini in the cloud (MODEL_PROVIDER).
- Database:   local Docker Postgres by default; DATABASE_URL points it at Neon in the cloud.

Secrets live in the environment, never in the code.
"""

import os

from dotenv import load_dotenv

load_dotenv()  # load a local .env file if present (git-ignored)

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "ollama")     # chat: "ollama" | "gemini"
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "fastembed")  # embeddings: "fastembed" | "gemini" | "ollama"
COLLECTION = os.getenv("COLLECTION", "papermind_papers")

GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001")
FASTEMBED_MODEL = os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")


def get_connection():
    """Database URL. Defaults to local Docker Postgres; override with DATABASE_URL."""
    url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://papermind:papermind@localhost:5432/papermind",
    )
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def get_embeddings():
    """The embedding model. Default: fastembed (local, free, no rate limits)."""
    if EMBED_PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=GEMINI_EMBED_MODEL)
    if EMBED_PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model="nomic-embed-text")
    from langchain_community.embeddings import FastEmbedEmbeddings
    return FastEmbedEmbeddings(model_name=FASTEMBED_MODEL)


def get_chat_model():
    """The answer-writing LLM: Gemini in the cloud, Ollama locally."""
    if MODEL_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=GEMINI_CHAT_MODEL)
    from langchain_ollama import ChatOllama
    return ChatOllama(model="llama3.2")
