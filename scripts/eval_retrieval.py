"""Measure retrieval quality on the eval sets (plain English + exact terms), for every search method.

    python scripts/eval_retrieval.py            # prints a table, writes evals/results.md

Each question is searched ONCE per method (semantic + keyword, top 20 each); the hybrid
variants re-fuse those same lists with different keyword weights, so the comparison is fair
and the script makes only 2 database round-trips per question.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.embed_store import get_vector_store
from app.hybrid_search import CANDIDATES, _keyword_docs, doc_key, rrf_fuse
from app.retrieval_metrics import reciprocal_rank, summarize

EVAL_SETS = {"plain English": ROOT / "evals" / "plain_english.jsonl",
             "exact terms": ROOT / "evals" / "exact_terms.jsonl"}
RESULTS = ROOT / "evals" / "results.md"
TOP = 10  # how many results each method returns for scoring
KEYWORD_WEIGHTS = [0.5, 1.0, 1.5, 2.0, 3.0]
METHODS = ["semantic", "keyword"] + [f"hybrid w={w}" for w in KEYWORD_WEIGHTS]


def paper(doc):
    return Path(doc.metadata.get("source", "?")).name


def rank_all_methods(store, query):
    """Search once per method, then re-fuse the same lists at every keyword weight."""
    semantic = [d for d, _ in store.similarity_search_with_score(query, k=CANDIDATES)]
    keyword = _keyword_docs(query, CANDIDATES)
    by_key = {doc_key(d): d for d in keyword + semantic}
    ranked = {"semantic": [paper(d) for d in semantic[:TOP]],
              "keyword": [paper(d) for d in keyword[:TOP]]}
    for w in KEYWORD_WEIGHTS:
        fused = rrf_fuse([[doc_key(d) for d in semantic], [doc_key(d) for d in keyword]],
                         weights=[1.0, w])
        ranked[f"hybrid w={w}"] = [paper(by_key[key]) for key, _ in fused[:TOP]]
    return ranked


def table(title, rows):
    lines = [f"## {title}\n", "| Method | hit@1 | recall@5 | MRR |", "|---|---|---|---|"]
    for m in METHODS:
        s = summarize(rows[m])
        lines.append(f"| {m} | {s['hit@1']:.0%} | {s['recall@5']:.0%} | {s['mrr']:.3f} |")
    return "\n".join(lines)


def main():
    store = get_vector_store()
    out = ["# Retrieval eval\n"]
    combined = {m: [] for m in METHODS}

    for name, path in EVAL_SETS.items():
        questions = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        rows = {m: [] for m in METHODS}
        misses = {m: [] for m in METHODS}
        for item in questions:
            ranked = rank_all_methods(store, item["q"])
            for m in METHODS:
                rows[m].append((ranked[m], item["paper"]))
                if reciprocal_rank(ranked[m], item["paper"]) < 1.0:
                    misses[m].append(f"- {item['q']}  ->  got {ranked[m][0] if ranked[m] else 'nothing'}")
        for m in METHODS:
            combined[m] += rows[m]
        out.append(table(f"{name} ({len(questions)} questions)", rows))
        out.append(f"\nMisses at #1, semantic ({name}):\n" + "\n".join(misses["semantic"] or ["- none"]))
        out.append(f"\nMisses at #1, hybrid w=1.0 ({name}):\n" + "\n".join(misses["hybrid w=1.0"] or ["- none"]))

    out.append(table(f"combined ({len(combined['semantic'])} questions)", combined))

    # Decision rule, fixed BEFORE looking at results: best combined MRR wins;
    # if the runner-up is within 0.02 MRR, the one with higher recall@5 wins.
    scored = sorted(METHODS, key=lambda m: summarize(combined[m])["mrr"], reverse=True)
    best, runner = scored[0], scored[1]
    sb, sr = summarize(combined[best]), summarize(combined[runner])
    if sb["mrr"] - sr["mrr"] < 0.02 and sr["recall@5"] > sb["recall@5"]:
        best = runner
    out.append(f"\n**Decision rule winner: {best}**")

    report = "\n\n".join(out) + "\n"
    print(report)
    RESULTS.write_text(report)
    print(f"wrote {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
