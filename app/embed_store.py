"""
PaperMind — RAG Steps 2 & 3: EMBED and STORE.

Provides:
- main(): (re)build the whole store from every PDF in data/  ->  python -m app.embed_store
- ingest_pdf_file(path): add ONE new PDF to the store (used by the /upload endpoint)
- get_vector_store(): connect to the pgvector collection
"""

from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_community.document_loaders import PyPDFLoader

from app.ingest import load_documents, split_documents

CONNECTION = "postgresql+psycopg://papermind:papermind@localhost:5432/papermind"
COLLECTION = "papermind_papers"


def sanitize(chunks):
    """Strip NUL (0x00) bytes that PostgreSQL text columns reject (common in PDF text)."""
    for chunk in chunks:
        chunk.page_content = chunk.page_content.replace("\x00", "")
        chunk.metadata = {
            key: (value.replace("\x00", "") if isinstance(value, str) else value)
            for key, value in chunk.metadata.items()
        }
    return chunks


def get_vector_store(pre_delete=False):
    """Connect to the pgvector collection. pre_delete=True wipes it first (full rebuild)."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION,
        connection=CONNECTION,
        use_jsonb=True,
        pre_delete_collection=pre_delete,
    )


def ingest_pdf_file(path):
    """Load ONE PDF, split + clean it, and add it to the store WITHOUT wiping existing data."""
    docs = PyPDFLoader(str(path)).load()
    chunks = sanitize(split_documents(docs))
    store = get_vector_store(pre_delete=False)
    store.add_documents(chunks)
    return len(chunks)


def main():
    print("Loading and splitting PDFs...")
    docs = load_documents()
    chunks = sanitize(split_documents(docs))
    print(f"Prepared {len(chunks)} chunks.")

    store = get_vector_store(pre_delete=True)  # full rebuild

    print("Embedding + storing chunks (this can take a few minutes)...")
    batch_size = 100
    total = len(chunks)
    for start in range(0, total, batch_size):
        store.add_documents(chunks[start:start + batch_size])
        print(f"  stored {min(start + batch_size, total)}/{total} chunks")

    print(f"Done! Stored {total} chunks in the '{COLLECTION}' collection.")


if __name__ == "__main__":
    main()
