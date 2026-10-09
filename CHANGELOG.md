# Changelog

All notable changes to PaperMind. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-10-09

First stable release: PaperMind as a tested, containerized MCP server with measured retrieval quality.

### Added
- **Hybrid retrieval** (#8, #10): Postgres full-text search (generated `tsvector` column + GIN index) alongside pgvector, fused with Reciprocal Rank Fusion. Keyword weight 0.5, chosen on a labeled eval.
- **Retrieval eval suite** (#10): 36 labeled questions (`evals/`), hit@1 / recall@5 / MRR metrics, and `scripts/eval_retrieval.py` comparing semantic, keyword and hybrid search. Result: 78% hit@1, 97% recall@5, 0.841 MRR.
- **Bearer-token auth for HTTP mode** (#15): `TokenVerifier` for the MCP SDK's resource-server auth; the server refuses to bind a network address without `PAPERMIND_TOKEN`.
- `scripts/explain_search.py` shows both rankings and the fused result for any query.
- CI builds the Docker image and checks that the container rejects unauthenticated requests (#16).
- Demo GIF in the README (#17).

### Changed
- Multi-stage Docker build with MCP-only dependencies: 978 MB to 836 MB (#12).
- Replaced deprecated `langchain-community` with in-repo PDF loading and a fastembed adapter, verified to produce identical text and vectors (#11).

### Fixed
- A long-running server failed its next search after Neon auto-suspended idle connections (`AdminShutdown`). The pool now pings connections before reuse and recycles them after 5 minutes (#13).

## Earlier work (pre-1.0)

- MCP server with 3 tools, 2 resources and 1 prompt; typed outputs and tool annotations.
- Security boundary for `add_paper` (path traversal, symlinks, PDF signature, rollback); tests also found and fixed a path-traversal bug in the web `/upload` endpoint.
- Streamable HTTP transport and a non-root Docker image.
- FastAPI web app with PDF upload and grounded Q&A; provider-agnostic config (fastembed / Gemini / Ollama, local Postgres or Neon).

[1.0.0]: https://github.com/Enoch-Nemili/papermind/releases/tag/v1.0.0
