"""
Read PDFs into LangChain Documents, one per page, with the metadata PaperMind cites.

Replaces langchain-community's PyPDFLoader (that package is being retired). Same library
(pypdf), same text extraction, same metadata keys our search results depend on:
  source (file path), total_pages, page (0-based), page_label (the printed page number).
"""

from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


def load_pdf(path):
    reader = PdfReader(str(path))
    labels = reader.page_labels
    return [
        Document(
            page_content=page.extract_text().strip(),
            metadata={
                "source": str(path),
                "total_pages": len(reader.pages),
                "page": number,
                "page_label": labels[number],
            },
        )
        for number, page in enumerate(reader.pages)
    ]


def load_pdf_folder(folder):
    """Every PDF directly inside `folder` (hidden files skipped), in a stable order."""
    docs = []
    for path in sorted(Path(folder).glob("*.pdf")):
        if not path.name.startswith("."):
            docs.extend(load_pdf(path))
    return docs
