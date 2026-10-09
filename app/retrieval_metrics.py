"""
Standard retrieval metrics. Each takes `ranked`: the paper each retrieved chunk came from,
best first (e.g. ["t5.pdf", "attention.pdf", ...]), and the paper that SHOULD come back.

- hit@k:  1 if the right paper appears in the top k, else 0.  hit@1 = "was the #1 result right?"
- reciprocal rank: 1 / position of the first right result (1.0, 0.5, 0.33, ...; 0 if missing).
  Averaged over questions it's MRR, which rewards getting the answer near the top.
"""


def hit_at(ranked, expected, k):
    return 1.0 if expected in ranked[:k] else 0.0


def reciprocal_rank(ranked, expected):
    for position, paper in enumerate(ranked, start=1):
        if paper == expected:
            return 1.0 / position
    return 0.0


def summarize(rows):
    """rows: list of (ranked, expected). Returns averaged hit@1, hit@5 (recall@5) and MRR."""
    n = len(rows) or 1
    return {
        "hit@1": sum(hit_at(r, e, 1) for r, e in rows) / n,
        "recall@5": sum(hit_at(r, e, 5) for r, e in rows) / n,
        "mrr": sum(reciprocal_rank(r, e) for r, e in rows) / n,
    }
