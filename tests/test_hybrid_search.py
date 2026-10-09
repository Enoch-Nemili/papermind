"""Unit tests for Reciprocal Rank Fusion and the hybrid search wrapper (no database needed)."""

import pytest
from langchain_core.documents import Document

import app.hybrid_search as hybrid
from app.hybrid_search import rrf_fuse


def keys(fused):
    return [key for key, _ in fused]


def test_a_chunk_both_methods_like_beats_one_method_s_favorite():
    semantic = ["a", "b", "c"]
    keyword = ["c", "d"]          # "c" is #3 semantically but #1 by keyword
    fused = rrf_fuse([semantic, keyword])
    assert keys(fused)[0] == "c"
    assert set(keys(fused)) == {"a", "b", "c", "d"}  # nothing is dropped


def test_top_of_both_lists_scores_exactly_one():
    fused = rrf_fuse([["x", "y"], ["x", "z"]])
    assert fused[0] == ("x", pytest.approx(1.0))
    assert all(0 < score <= 1 for _, score in fused)


def test_one_empty_list_normalizes_to_the_list_that_answered():
    fused = rrf_fuse([["x", "y"], []])
    assert fused[0] == ("x", pytest.approx(1.0))
    assert keys(fused) == ["x", "y"]


def test_ties_keep_semantic_order():
    fused = rrf_fuse([["s1", "s2"], ["k1", "k2"]])
    assert keys(fused) == ["s1", "k1", "s2", "k2"]


def test_nothing_in_nothing_out():
    assert rrf_fuse([[], []]) == []


class FakeStore:
    def __init__(self, docs):
        self.docs = docs

    def similarity_search_with_score(self, query, k):
        return [(doc, 0.1) for doc in self.docs[:k]]


def doc(i, text):
    return Document(id=str(i), page_content=text, metadata={"source": "p.pdf", "page": i})


def test_hybrid_merges_the_same_chunk_from_both_searches(monkeypatch):
    a, b, c = doc(1, "a"), doc(2, "b"), doc(3, "c")
    monkeypatch.setattr(hybrid, "ensure_fts_index", lambda: None)
    # keyword search returns its own Document objects for the same rows (same ids)
    monkeypatch.setattr(hybrid, "keyword_search", lambda q, k: [(doc(3, "c"), 0.9)])

    results = hybrid.hybrid_search(FakeStore([a, b, c]), "query", k=3)

    ids = [d.id for d, _ in results]
    assert ids[0] == "3"              # promoted by agreement
    assert sorted(ids) == ["1", "2", "3"]  # merged by id, not duplicated
