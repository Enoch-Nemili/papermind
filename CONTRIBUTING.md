# Contributing to PaperMind

Thanks for your interest! Issues and pull requests are welcome.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # defaults are fully local
docker compose up -d            # local Postgres + pgvector
python -m app.embed_store       # index the PDFs in data/
```

## Before opening a pull request

```bash
ruff check app tests scripts
pytest -v                        # fast suite: no database needed
```

If your change touches **retrieval** (chunking, embeddings, search, fusion), also run the eval against an indexed library and paste the table into the PR:

```bash
pytest -m integration -v
python scripts/eval_retrieval.py
```

Retrieval changes are judged by the numbers, not by individual examples. Don't tune parameters until a single case passes; add labeled questions to `evals/` instead.

## Conventions

- One branch and one pull request per issue; reference it with `Closes #N`.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `refactor:`, `build:`, `ci:`, `test:`.
- Every bug fix comes with a test that fails without it.
- Security issues: see [SECURITY.md](SECURITY.md), not public issues.
