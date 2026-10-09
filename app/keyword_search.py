"""
PaperMind: KEYWORD search (Postgres full-text search) over the same chunks pgvector stores.

Vector search matches MEANING; keyword search matches WORDS. Hybrid search (issue #1)
runs both and fuses the rankings, so each covers the other's blind spots.

How it works:
- ensure_fts_index(): adds a `fts` column to langchain's embedding table. Postgres fills it
  automatically with each chunk's words reduced to their roots ("attentions" -> "attent"),
  and a GIN index makes lookups fast. Safe to run many times.
- keyword_search(query, k): finds chunks that contain the query's words, best match first.

Try it:  python -m app.keyword_search "How does self-attention work?"
"""

import re
import sys
from pathlib import Path

import psycopg
from langchain_core.documents import Document

from app.config import COLLECTION, get_connection

SETUP_SQL = [
    # A stored generated column: Postgres keeps it in sync on every insert, no app code needed.
    """
    ALTER TABLE langchain_pg_embedding
    ADD COLUMN IF NOT EXISTS fts tsvector
    GENERATED ALWAYS AS (to_tsvector('english', coalesce(document, ''))) STORED
    """,
    # GIN = an inverted index: word -> list of chunks containing it (like a book's index).
    "CREATE INDEX IF NOT EXISTS langchain_pg_embedding_fts_idx ON langchain_pg_embedding USING gin (fts)",
]

SEARCH_SQL = """
    SELECT e.id, e.document, e.cmetadata, ts_rank_cd(e.fts, q.query, 32) AS score
    FROM langchain_pg_embedding e
    JOIN langchain_pg_collection c ON c.uuid = e.collection_id,
         to_tsquery('english', %(tsquery)s) AS q(query)
    WHERE c.name = %(collection)s AND e.fts @@ q.query
    ORDER BY score DESC
    LIMIT %(k)s
"""


def db_url():
    """psycopg wants plain postgresql://; SQLAlchemy (used by langchain) wants +psycopg."""
    return get_connection().replace("postgresql+psycopg://", "postgresql://", 1)


def to_or_query(text):
    """Turn free text into a safe OR-query: "How does self-attention work?" -> "how | does | self | attention | work".

    Only letters and digits survive, so nothing the user (or the model) types can inject
    tsquery operators or SQL. OR (not AND) means a chunk needs SOME of the words, and
    chunks with more of them rank higher. Postgres drops stopwords like "how" and "does".
    """
    words = dict.fromkeys(w.lower() for w in re.findall(r"[A-Za-z0-9]+", text) if len(w) > 1)
    return " | ".join(words) or None


def ensure_fts_index():
    with psycopg.connect(db_url()) as conn:
        for sql in SETUP_SQL:
            conn.execute(sql)


def keyword_search(query, k=5):
    """Return [(Document, score)] for chunks containing the query's words, best first."""
    tsquery = to_or_query(query)
    if tsquery is None:
        return []
    with psycopg.connect(db_url()) as conn:
        rows = conn.execute(SEARCH_SQL, {"tsquery": tsquery, "collection": COLLECTION, "k": k}).fetchall()
    return [
        (Document(id=row_id, page_content=text, metadata=meta or {}), float(score))
        for row_id, text, meta, score in rows
    ]


def main():
    query = " ".join(sys.argv[1:]) or "How does self-attention work?"
    ensure_fts_index()
    print(f"query: {query!r}   ->   tsquery: {to_or_query(query)!r}\n")
    for doc, score in keyword_search(query, k=5):
        meta = doc.metadata
        page = meta.get("page_label") or (meta.get("page", -1) + 1)
        print(f"  {score:.3f}  {Path(meta.get('source', '?')).name}  p.{page}")


if __name__ == "__main__":
    main()
