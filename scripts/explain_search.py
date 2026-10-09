"""Show WHY hybrid search ranked things the way it did: both input rankings + the fused result.

    python scripts/explain_search.py "How does self-attention work?"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.embed_store import get_vector_store
from app.hybrid_search import CANDIDATES, _keyword_docs, doc_key, hybrid_search


def label(doc):
    meta = doc.metadata
    page = meta.get("page_label") or (meta.get("page", -1) + 1)
    return f"{Path(meta.get('source', '?')).name} p.{page}"


def main():
    query = " ".join(sys.argv[1:]) or "How does self-attention work?"
    store = get_vector_store()

    semantic = [doc for doc, _ in store.similarity_search_with_score(query, k=CANDIDATES)]
    keyword = _keyword_docs(query, CANDIDATES)
    sem_rank = {doc_key(d): i for i, d in enumerate(semantic, 1)}
    kw_rank = {doc_key(d): i for i, d in enumerate(keyword, 1)}

    print(f"query: {query!r}\n")
    print("SEMANTIC top 10            KEYWORD top 10")
    for i in range(10):
        left = label(semantic[i]) if i < len(semantic) else ""
        right = label(keyword[i]) if i < len(keyword) else ""
        print(f"  {i + 1:>2}. {left:<26} {i + 1:>2}. {right}")

    print("\nFUSED top 5   (semantic rank / keyword rank; '-' = not in that list's top "
          f"{CANDIDATES})")
    for doc, score in hybrid_search(store, query, k=5):
        key = doc_key(doc)
        print(f"  {score:.3f}  {label(doc):<32} sem {sem_rank.get(key, '-'):>2} / kw {kw_rank.get(key, '-')}")


if __name__ == "__main__":
    main()
