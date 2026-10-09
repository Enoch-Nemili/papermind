# Security Policy

## Reporting a vulnerability

Please **don't open a public issue** for security problems. Email **enoch.das@gmail.com** with:

- what you found and how to reproduce it,
- what an attacker could do with it,
- the version or commit you tested.

You'll get an acknowledgement within a few days. Fixes are released on `main` and noted in the [changelog](CHANGELOG.md).

## Supported versions

| Version | Supported |
|---------|-----------|
| 1.x     | Yes       |
| < 1.0   | No        |

## Threat model

PaperMind gives an AI model tools that touch the file system and a database, so the main risk is a model (or a prompt injected into a document) trying to do more than it should. Controls are enforced in server code, not in prompts:

| Threat | Control |
|--------|---------|
| Reading or writing files outside the library | `add_paper` only reads files directly inside `inbox/`; paths are resolved (following symlinks) and must stay there |
| Disguised or oversized files | `.pdf` extension, size limit and `%PDF-` signature checks |
| Excessive agency (OWASP LLM Top 10) | No delete tool and no arbitrary file access; search is read-only |
| Unauthenticated network access | HTTP mode requires a bearer token (32+ characters, constant-time comparison) and refuses to bind a non-loopback address without one |
| DNS rebinding | `Origin` and `Host` are validated on every HTTP request; foreign origins get 403. Configured explicitly so it also covers the Docker image's `0.0.0.0` bind |
| Query injection | Keyword queries are reduced to alphanumeric terms; all SQL is parameterized |
| Secrets in images or git | Secrets live in a git-ignored `.env` and are passed to containers at runtime; the image runs as a non-root user |

## Out of scope

- Running HTTP mode on a public network with no reverse proxy or TLS. Put PaperMind behind HTTPS (e.g. a reverse proxy) before exposing it beyond a trusted network.
- Full OAuth flows. Bearer-token auth suits self-hosted use; for OAuth, plug in a JWT `TokenVerifier` (see [`app/auth.py`](app/auth.py)).
