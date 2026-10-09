"""
PaperMind — RAG Step 1: LOAD and SPLIT.

This script reads every PDF in the data/ folder, extracts the text,
and splits it into small, overlapping chunks. Later steps will turn these
chunks into vectors and store them for searching.

Run it from the project root (with your venv active):
    python -m app.ingest
"""

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.pdf_loader import load_pdf_folder

# Build the path to our data/ folder no matter where the script is run from.
# __file__ = this file's location; .parent.parent walks up from app/ to the project root.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_documents():
    """Open every PDF in data/ and return a list of documents (one per page)."""
    return load_pdf_folder(DATA_DIR)


def split_documents(documents):
    """Chop the page-documents into smaller, overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,       # aim for ~1000 characters per chunk
        chunk_overlap=200,     # repeat 150 chars between neighbours so ideas aren't cut in half
        add_start_index=True,  # record where each chunk started within its page
    )
    chunks = splitter.split_documents(documents)
    return chunks


if __name__ == "__main__":
    print(f"Loading PDFs from: {DATA_DIR}")

    docs = load_documents()
    print(f"Loaded {len(docs)} pages across your PDFs.")

    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    # Peek at one chunk so we can see exactly what we produced.
    if chunks:
        sample = chunks[0]
        print("\n----- Sample chunk -----")
        print("Source :", sample.metadata.get("source"))
        print("Page   :", sample.metadata.get("page"))
        print("Preview:\n", sample.page_content[:300], "...")
