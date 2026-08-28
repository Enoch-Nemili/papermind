"""
PaperMind — RAG Steps 2 & 3: EMBED and STORE.

Reuses the load + split from ingest.py, turns each chunk into a vector using
Ollama's `nomic-embed-text` model, and stores those vectors in Postgres/pgvector.

Prerequisites (both must be running):
- The database container:   docker compose up -d
- Ollama with the model:    ollama pull nomic-embed-text

Run it from the project root (venv active):
    python -m app.embed_store
"""

from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

from app.ingest import load_documents, split_documents

# How to reach the database defined in docker-compose.yml.
# Shape: postgresql+psycopg://<user>:<password>@<host>:<port>/<database>
CONNECTION = "postgresql+psycopg://papermind:papermind@localhost:5432/papermind"

# The name of the collection (think: table) that holds our paper vectors.
COLLECTION = "papermind_papers"


def sanitize(chunks):
    """
    Clean the chunk text before it goes into the database.

    Some PDFs, when their text is extracted, contain NUL (0x00) bytes -- invisible
    junk characters. PostgreSQL text columns refuse to store them, so we strip them
    out of both the text and any string metadata. This is a very common real-world
    data-cleaning step.
    """
    for chunk in chunks:
        chunk.page_content = chunk.page_content.replace("\x00", "")
        chunk.metadata = {
            key: (value.replace("\x00", "") if isinstance(value, str) else value)
            for key, value in chunk.metadata.items()
        }
    return chunks


def main():
    # 1) Prepare the chunks (the Load + Split from the previous step).
    print("Loading and splitting PDFs...")
    docs = load_documents()
    chunks = split_documents(docs)
    chunks = sanitize(chunks)  # <-- clean out forbidden characters
    print(f"Prepared {len(chunks)} chunks.")

    # 2) The embedding model -- runs locally through Ollama.
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 3) Connect to pgvector. pre_delete_collection=True means "start fresh each run"
    #    so re-running doesn't pile up duplicate copies.
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION,
        connection=CONNECTION,
        use_jsonb=True,
        pre_delete_collection=True,
    )

    # 4) Embed and store, in batches, printing progress as we go.
    print("Embedding + storing chunks (this can take a few minutes on first run)...")
    batch_size = 100
    total = len(chunks)
    for start in range(0, total, batch_size):
        batch = chunks[start:start + batch_size]
        vector_store.add_documents(batch)
        done = min(start + batch_size, total)
        print(f"  stored {done}/{total} chunks")

    print(f"Done! Stored {total} chunks in the '{COLLECTION}' collection.")


if __name__ == "__main__":
    main()
