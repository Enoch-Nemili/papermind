"""Retrieval-quality checks against the REAL vector database (Neon).

Skipped by default (they need DATABASE_URL and the indexed papers). Run locally:

    pytest -m integration -v

Each case is (query, the paper that must come back first). This is the seed of a
proper eval suite: when we change chunking or add hybrid search, these numbers
tell us whether retrieval actually got better.
"""

import os
from pathlib import Path

import pytest

import app.mcp_server as srv  # importing this loads .env, so DATABASE_URL is visible below

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="needs DATABASE_URL (real vector DB)"),
]

CASES = [
    ("KV cache paging memory fragmentation", "vllm-pagedattention.pdf"),
    ("scaled dot-product attention queries keys values softmax", "attention-is-all-you-need.pdf"),
    ("low-rank decomposition of weight updates for efficient fine-tuning", "lora.pdf"),
    ("masked language model bidirectional pre-training", "bert.pdf"),
    ("chain of thought prompting intermediate reasoning steps", "chain-of-thought.pdf"),
    ("retrieval-augmented generation non-parametric memory", "rag-original.pdf"),
]


def top_source(query):
    doc, _ = srv.get_store().similarity_search_with_score(query, k=1)[0]
    return Path(doc.metadata["source"]).name


@pytest.mark.parametrize(("query", "expected"), CASES)
def test_technical_query_ranks_the_right_paper_first(query, expected):
    assert top_source(query) == expected


@pytest.mark.xfail(
    reason="Known weakness: plain-English questions suffer vocabulary mismatch (measured "
    "0.65-0.69 relevance vs 0.87 for technical phrasing). Fix candidate: hybrid BM25 + vector.",
    strict=False,
)
def test_plain_english_question_ranks_the_original_paper_first():
    assert top_source("How does self-attention work?") == "attention-is-all-you-need.pdf"
