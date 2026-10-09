"""Shared test fixtures: a throwaway library on disk, a fake vector DB, and an MCP client."""

from types import SimpleNamespace

import pytest
from mcp import Client
from pypdf import PdfWriter

import app.hybrid_search as hybrid
import app.mcp_server as srv


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def make_pdf():
    """Write a real (blank) PDF with `pages` pages and return its path."""

    def _make(path, pages=1):
        writer = PdfWriter()
        for _ in range(pages):
            writer.add_blank_page(width=72, height=72)
        with open(path, "wb") as f:
            writer.write(f)
        return path

    return _make


@pytest.fixture
def library(tmp_path, monkeypatch):
    """Point the server at an empty data/ + inbox/ in a temp dir and fake the indexer.

    Nothing touches your real library or the Neon database.
    """
    data, inbox = tmp_path / "data", tmp_path / "inbox"
    data.mkdir()
    inbox.mkdir()
    monkeypatch.setattr(srv, "DATA_DIR", data)
    monkeypatch.setattr(srv, "INBOX_DIR", inbox)

    ingested = []

    def fake_ingest(path):
        ingested.append(path)
        return 7  # pretend the PDF became 7 chunks

    monkeypatch.setattr(srv, "ingest_pdf_file", fake_ingest)
    # Keyword search hits Postgres: stub it (no matches) unless a test says otherwise.
    monkeypatch.setattr(hybrid, "ensure_fts_index", lambda: None)
    monkeypatch.setattr(hybrid, "keyword_search", lambda query, k: [])
    return SimpleNamespace(root=tmp_path, data=data, inbox=inbox, ingested=ingested)


@pytest.fixture
async def client(library):
    """A real MCP client connected to the server in-process (no subprocess, no network)."""
    async with Client(srv.mcp) as c:
        yield c
