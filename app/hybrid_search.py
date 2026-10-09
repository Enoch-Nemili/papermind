"""
PaperMind: HYBRID search = semantic (pgvector) + keyword (Postgres full-text), fused with RRF.

Why both? Semantic search matches MEANING but can miss a plain-English question whose words
differ from the paper's ("How does self-attention work?"). Keyword search matches WORDS but
misses paraphrases ("make fine-tuning cheaper" -> LoRA). Each covers the other's blind spot.

Reciprocal Rank Fusion (RRF): every list votes 1 / (60 + rank) for each chunk it returned.
Chunks ranked well by BOTH lists win. We fuse ranks, not raw scores, because a cosine
similarity and a full-text rank aren't on the same scale, so adding them would be meaningless.
"""

import logging

import psycopg

from app.keyword_search import ensure_fts_index, keyword_search

log = logging.getLogger("papermind-mcp")

RRF_K = 60       # standard constant from the RRF paper; damps the gap between rank 1 and rank 2
CANDIDATES = 20  # how deep each method looks before fusing

_fts_ready = False


def doc_key(doc):
    """Both searches read the same table, so the row id identifies a chunk across them."""
    return doc.id or (doc.metadata.get("source"), doc.page_content)


def rrf_fuse(rankings, k=RRF_K):
    """Fuse ranked lists of keys. Returns [(key, score)] best first.

    Scores are normalized to 0-1, where 1.0 means "ranked #1 by every list that returned
    anything". Ties keep the order of the first list (semantic), since sorting is stable.
    """
    scores = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    lists_used = sum(1 for ranking in rankings if ranking)
    best_possible = lists_used / (k + 1) if lists_used else 1.0
    fused = [(key, score / best_possible) for key, score in scores.items()]
    return sorted(fused, key=lambda item: item[1], reverse=True)


def _keyword_docs(query, n):
    """Keyword hits, or [] if full-text search is unavailable: degrade, don't fail."""
    global _fts_ready
    try:
        if not _fts_ready:
            ensure_fts_index()
            _fts_ready = True
        return [doc for doc, _ in keyword_search(query, k=n)]
    except psycopg.Error as exc:
        log.warning("keyword search unavailable, falling back to semantic only: %s", exc)
        return []


def hybrid_search(store, query, k=5, candidates=CANDIDATES):
    """Return [(Document, fused_score)] for the top-k chunks from semantic + keyword search."""
    semantic = [doc for doc, _ in store.similarity_search_with_score(query, k=candidates)]
    keyword = _keyword_docs(query, candidates)

    by_key = {}
    for doc in semantic + keyword:
        by_key.setdefault(doc_key(doc), doc)

    fused = rrf_fuse([[doc_key(d) for d in semantic], [doc_key(d) for d in keyword]])
    return [(by_key[key], score) for key, score in fused[:k]]
