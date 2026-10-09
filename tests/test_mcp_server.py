"""Tests for the PaperMind MCP server.

Every test talks to the server through a real MCP client (`mcp.Client`, connected
in-process), so the protocol layer is exercised too: tool listing, schema
validation and error results. The vector database is faked, so these run
anywhere, with no network and no secrets.
"""

import json

import psycopg
import pytest
from langchain_core.documents import Document

import app.hybrid_search as hybrid
import app.mcp_server as srv

pytestmark = pytest.mark.anyio


def text(result):
    return result.content[0].text


# ------------------------------------------------------------ discovery ---

async def test_exposes_three_tools_with_safety_annotations(client):
    tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == {"search_papers", "list_papers", "add_paper"}

    assert tools["search_papers"].annotations.read_only_hint is True
    assert tools["list_papers"].annotations.read_only_hint is True

    add = tools["add_paper"].annotations
    assert add.read_only_hint is False
    assert add.destructive_hint is False
    assert add.idempotent_hint is True


async def test_every_tool_declares_an_output_schema(client):
    # Regression test: tools returning a bare `dict` get NO output schema, so clients
    # receive unstructured JSON text. Typed result models fix that.
    for tool in (await client.list_tools()).tools:
        assert tool.output_schema, f"{tool.name} has no output schema"


# --------------------------------------- add_paper: the security boundary ---

@pytest.mark.parametrize("filename", ["../.env", "../../etc/passwd", "sub/../../secret.pdf"])
async def test_add_paper_refuses_paths_outside_inbox(client, library, filename):
    (library.root / ".env").write_text("SECRET_KEY=do-not-leak")
    result = await client.call_tool("add_paper", {"filename": filename})
    assert result.is_error
    assert "inbox/" in text(result)
    assert library.ingested == []


async def test_add_paper_refuses_symlink_pointing_outside_inbox(client, library, make_pdf):
    private = make_pdf(library.root / "private.pdf")
    (library.inbox / "innocent.pdf").symlink_to(private)
    result = await client.call_tool("add_paper", {"filename": "innocent.pdf"})
    assert result.is_error
    assert "inbox/" in text(result)
    assert library.ingested == []


async def test_add_paper_rejects_non_pdf_extension(client, library):
    (library.inbox / "notes.txt").write_text("hello")
    result = await client.call_tool("add_paper", {"filename": "notes.txt"})
    assert result.is_error
    assert ".pdf" in text(result)


async def test_add_paper_rejects_file_only_pretending_to_be_pdf(client, library):
    (library.inbox / "fake.pdf").write_text("this is not a pdf")
    result = await client.call_tool("add_paper", {"filename": "fake.pdf"})
    assert result.is_error
    assert "isn't a valid PDF" in text(result)
    assert not (library.data / "fake.pdf").exists()


async def test_add_paper_missing_file_lists_what_is_waiting(client, library, make_pdf):
    make_pdf(library.inbox / "mamba.pdf")
    result = await client.call_tool("add_paper", {"filename": "missing.pdf"})
    assert result.is_error
    assert "mamba.pdf" in text(result)  # the error tells the AI how to recover


# ------------------------------------------ add_paper: correct behaviour ---

async def test_add_paper_adds_once_then_is_idempotent(client, library, make_pdf):
    make_pdf(library.inbox / "mamba.pdf")

    first = await client.call_tool("add_paper", {"filename": "mamba.pdf"})
    second = await client.call_tool("add_paper", {"filename": "mamba.pdf"})

    assert first.structured_content["added"] is True
    assert first.structured_content["paper"] == "mamba.pdf"
    assert first.structured_content["chunks_indexed"] == 7
    assert second.structured_content["added"] is False
    assert second.structured_content["reason"] == "already in the library"
    assert len(library.ingested) == 1  # indexed exactly once
    assert (library.data / "mamba.pdf").exists()


async def test_add_paper_rolls_back_when_indexing_fails(client, library, make_pdf, monkeypatch):
    make_pdf(library.inbox / "mamba.pdf")

    def broken_ingest(path):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(srv, "ingest_pdf_file", broken_ingest)
    result = await client.call_tool("add_paper", {"filename": "mamba.pdf"})

    assert result.is_error
    assert "database unavailable" in text(result)
    assert not (library.data / "mamba.pdf").exists()  # library and index stay in sync


# --------------------------------------------------------- search_papers ---

class FakeStore:
    def __init__(self, hits):
        self.hits, self.calls = hits, []

    def similarity_search_with_score(self, query, k):
        self.calls.append((query, k))
        return self.hits[:k]


async def test_search_papers_returns_cited_passages(client, monkeypatch):
    attention = Document(page_content="Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V",
                         metadata={"source": "/abs/path/data/attention.pdf", "page": 3})
    front = Document(page_content="Roman-numbered front matter",
                     metadata={"source": "data/t5.pdf", "page": 1, "page_label": "xii"})
    store = FakeStore([(attention, 0.13), (front, 0.40)])
    monkeypatch.setattr(srv, "get_store", lambda: store)
    monkeypatch.setattr(hybrid, "keyword_search", lambda query, k: [(attention, 0.9)])

    result = await client.call_tool("search_papers", {"query": "scaled dot-product attention", "k": 2})
    passages = result.structured_content["result"]

    assert passages[0] == {
        "source": "attention.pdf",  # file name only, never the full local path
        "page": "4",                # 0-based loader page 3 -> the page a human sees
        "relevance": 1.0,           # ranked #1 by BOTH semantic and keyword search
        "text": "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V",
    }
    assert passages[1]["page"] == "xii"  # printed page labels win when the PDF has them
    assert passages[1]["relevance"] < 0.5  # found by semantic search only
    assert store.calls == [("scaled dot-product attention", hybrid.CANDIDATES)]


async def test_search_papers_still_works_when_keyword_search_is_down(client, monkeypatch):
    doc = Document(page_content="text", metadata={"source": "data/bert.pdf", "page": 0})
    monkeypatch.setattr(srv, "get_store", lambda: FakeStore([(doc, 0.2)]))

    def db_down(query, k):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(hybrid, "keyword_search", db_down)
    result = await client.call_tool("search_papers", {"query": "masked language model"})

    assert not result.is_error
    assert result.structured_content["result"][0]["source"] == "bert.pdf"


@pytest.mark.parametrize("k", [0, 11, 50])
async def test_search_papers_rejects_k_out_of_range(client, monkeypatch, k):
    def must_not_run():
        raise AssertionError("schema validation should reject this before any search runs")

    monkeypatch.setattr(srv, "get_store", must_not_run)
    result = await client.call_tool("search_papers", {"query": "anything", "k": k})
    assert result.is_error


# ------------------------------------------------ resources and prompts ---

async def test_list_papers_and_library_resource_agree(client, library, make_pdf):
    make_pdf(library.data / "a.pdf", pages=3)
    make_pdf(library.data / "b.pdf")

    listed = (await client.call_tool("list_papers", {})).structured_content
    assert listed == {"count": 2, "papers": ["a.pdf", "b.pdf"]}

    resource = await client.read_resource("papermind://library")
    catalog = json.loads(resource.contents[0].text)
    assert catalog["count"] == 2
    assert [p["pages"] for p in catalog["papers"]] == [3, 1]


async def test_prompt_wraps_question_and_demands_citations(client):
    prompt = await client.get_prompt("answer_from_papers", {"question": "What is LoRA?"})
    body = prompt.messages[0].content.text
    assert "What is LoRA?" in body
    assert "cite" in body.lower()
    assert "doesn't cover" in body  # tells the model not to fill gaps from memory
