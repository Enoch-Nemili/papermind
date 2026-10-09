"""
PaperMind — EMBED and STORE.

- main(): rebuild the whole store from every PDF in data/  ->  python -m app.embed_store
- ingest_pdf_file(path): add ONE new PDF (used by /upload)
- get_vector_store(): connect to the pgvector collection (models + DB come from config)
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres import PGVector

from app.config import COLLECTION, get_connection, get_embeddings
from app.ingest import load_documents, split_documents


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
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION,
        connection=get_connection(),
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

    print("Embedding + storing chunks (local model, no rate limits)...")
    batch_size = 100
    total = len(chunks)
    for start in range(0, total, batch_size):
        store.add_documents(chunks[start:start + batch_size])
        print(f"  stored {min(start + batch_size, total)}/{total} chunks")

    print(f"Done! Stored {total} chunks in the '{COLLECTION}' collection.")


if __name__ == "__main__":
    main()
