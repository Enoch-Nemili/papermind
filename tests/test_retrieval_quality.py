"""Retrieval-quality checks against the REAL vector database (Neon).

Skipped by default (they need DATABASE_URL and the indexed papers). Run locally:

    pytest -m integration -v

Each case is (query, the paper that must come back first). This is the seed of a
proper eval suite: when we change chunking or retrieval, these numbers tell us
whether retrieval actually got better.

The server uses HYBRID search (semantic + keyword, fused with RRF). The semantic-only
baseline is kept as a reference so the improvement stays visible.
"""

import os
from pathlib import Path

import pytest

import app.mcp_server as srv  # importing this loads .env, so DATABASE_URL is visible below
from app.hybrid_search import hybrid_search

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

PLAIN_ENGLISH = [
    ("How does self-attention work?", "attention-is-all-you-need.pdf"),
]


def top_semantic(query):
    doc, _ = srv.get_store().similarity_search_with_score(query, k=1)[0]
    return Path(doc.metadata["source"]).name


def top_hybrid(query):
    doc, _ = hybrid_search(srv.get_store(), query, k=1)[0]
    return Path(doc.metadata["source"]).name


@pytest.mark.parametrize(("query", "expected"), CASES)
def test_hybrid_ranks_the_right_paper_first(query, expected):
    assert top_hybrid(query) == expected


@pytest.mark.xfail(
    reason="Long-document bias: t5.pdf (67 pages, covers nearly every topic) wins #1 for many "
    "plain-English questions under any fusion weight. Measured on 36 questions in evals/; "
    "tracked as its own issue. Inspect with: python scripts/explain_search.py",
    strict=False,
)
@pytest.mark.parametrize(("query", "expected"), PLAIN_ENGLISH)
def test_hybrid_plain_english(query, expected):
    assert top_hybrid(query) == expected


@pytest.mark.parametrize(("query", "expected"), CASES)
def test_semantic_baseline_technical_queries(query, expected):
    assert top_semantic(query) == expected


@pytest.mark.xfail(
    reason="Semantic-only baseline: plain-English questions suffer vocabulary mismatch "
    "(0.65-0.69 relevance vs 0.87 for technical phrasing). Hybrid search fixes this.",
    strict=False,
)
@pytest.mark.parametrize(("query", "expected"), PLAIN_ENGLISH)
def test_semantic_baseline_plain_english(query, expected):
    assert top_semantic(query) == expected
