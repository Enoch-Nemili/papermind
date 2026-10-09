"""Regression test: the vector store must survive the database closing idle connections."""

from app import embed_store


def test_vector_store_reconnects_after_idle_disconnects(monkeypatch):
    captured = {}

    class FakePGVector:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(embed_store, "PGVector", FakePGVector)
    monkeypatch.setattr(embed_store, "get_embeddings", lambda: object())

    embed_store.get_vector_store()

    # Without these, a server idle longer than Neon's auto-suspend fails its next search.
    assert captured["engine_args"]["pool_pre_ping"] is True
    assert captured["engine_args"]["pool_recycle"] <= 300
