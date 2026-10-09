import pytest

from app.retrieval_metrics import hit_at, reciprocal_rank, summarize

RANKED = ["t5.pdf", "attention.pdf", "t5.pdf", "bert.pdf"]


def test_hit_at():
    assert hit_at(RANKED, "t5.pdf", 1) == 1.0
    assert hit_at(RANKED, "attention.pdf", 1) == 0.0
    assert hit_at(RANKED, "attention.pdf", 2) == 1.0
    assert hit_at(RANKED, "lora.pdf", 5) == 0.0


def test_reciprocal_rank_uses_the_first_correct_result():
    assert reciprocal_rank(RANKED, "t5.pdf") == 1.0
    assert reciprocal_rank(RANKED, "attention.pdf") == 0.5
    assert reciprocal_rank(RANKED, "bert.pdf") == 0.25
    assert reciprocal_rank(RANKED, "lora.pdf") == 0.0


def test_summarize_averages_over_questions():
    scores = summarize([(RANKED, "t5.pdf"), (RANKED, "attention.pdf")])
    assert scores == {"hit@1": 0.5, "recall@5": 1.0, "mrr": pytest.approx(0.75)}
