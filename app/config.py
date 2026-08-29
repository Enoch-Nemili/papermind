"""
PaperMind configuration.

Picks the model provider and the database from ENVIRONMENT VARIABLES, so the
SAME code runs two ways:
  - Locally (default): Ollama models + local Docker Postgres. No env vars needed.
  - In the cloud: set MODEL_PROVIDER=gemini, GOOGLE_API_KEY=..., DATABASE_URL=...

Secrets live in the environment, never in the code.
"""

import os

from dotenv import load_dotenv

load_dotenv()  # load a local .env file if present (that file is git-ignored)

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "ollama")   # "ollama" or "gemini"
COLLECTION = os.getenv("COLLECTION", "papermind_papers")

# Model names are overridable via env vars, so we can change them without editing code.
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004")


def get_connection():
    """Database URL. Defaults to the local Docker Postgres; override with DATABASE_URL."""
    url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://papermind:papermind@localhost:5432/papermind",
    )
    # langchain-postgres needs the psycopg driver named in the URL scheme.
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def get_embeddings():
    """The embedding model: Gemini in the cloud, Ollama locally."""
    if MODEL_PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=GEMINI_EMBED_MODEL)
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model="nomic-embed-text")


def get_chat_model():
    """The answer-writing LLM: Gemini in the cloud, Ollama locally."""
    if MODEL_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=GEMINI_CHAT_MODEL)
    from langchain_ollama import ChatOllama
    return ChatOllama(model="llama3.2")
