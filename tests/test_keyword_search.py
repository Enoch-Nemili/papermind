"""Unit tests for the query builder (no database needed)."""

import pytest

from app.keyword_search import to_or_query


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("How does self-attention work?", "how | does | self | attention | work"),
        ("LoRA low-rank LoRA", "lora | low | rank"),  # lower-cased and de-duplicated
        ("KV-cache  paging!!", "kv | cache | paging"),
    ],
)
def test_builds_an_or_query(text, expected):
    assert to_or_query(text) == expected


def test_strips_anything_that_could_inject_operators_or_sql():
    q = to_or_query("attention'); DROP TABLE x; -- & ! :* <->")
    assert q == "attention | drop | table"
    assert all(ch.isalnum() or ch in " |" for ch in q)


@pytest.mark.parametrize("text", ["", "   ", "?!", "a"])
def test_returns_none_when_nothing_searchable(text):
    assert to_or_query(text) is None
