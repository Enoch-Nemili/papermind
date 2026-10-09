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
# Keyword votes count half as much as semantic ones. Chosen by scripts/eval_retrieval.py on
# 36 labeled questions (see evals/results.md): w=0.5 matched semantic-only on hit@1 (78%) and
# MRR (0.841 vs 0.838) and raised recall@5 from 94% to 97%. Equal weights (1.0) were worse.
KEYWORD_WEIGHT = 0.5

_fts_ready = False


def doc_key(doc):
    """Both searches read the same table, so the row id identifies a chunk across them."""
    return doc.id or (doc.metadata.get("source"), doc.page_content)


def rrf_fuse(rankings, k=RRF_K, weights=None):
    """Fuse ranked lists of keys. Returns [(key, score)] best first.

    `weights` scales each list's votes (default: all 1). Scores are normalized to 0-1, where
    1.0 means "ranked #1 by every list that returned anything". Ties keep the order of the
    first list (semantic), since sorting is stable.
    """
    weights = weights or [1.0] * len(rankings)
    scores = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
    used_weight = sum(w for ranking, w in zip(rankings, weights, strict=True) if ranking)
    best_possible = used_weight / (k + 1) if used_weight else 1.0
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


def hybrid_search(store, query, k=5, candidates=CANDIDATES, keyword_weight=None):
    """Return [(Document, fused_score)] for the top-k chunks from semantic + keyword search."""
    semantic = [doc for doc, _ in store.similarity_search_with_score(query, k=candidates)]
    keyword = _keyword_docs(query, candidates)

    by_key = {}
    for doc in semantic + keyword:
        by_key.setdefault(doc_key(doc), doc)

    weight = KEYWORD_WEIGHT if keyword_weight is None else keyword_weight
    fused = rrf_fuse([[doc_key(d) for d in semantic], [doc_key(d) for d in keyword]],
                     weights=[1.0, weight])
    return [(by_key[key], score) for key, score in fused[:k]]
