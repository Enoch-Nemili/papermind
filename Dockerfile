# PaperMind MCP server, containerized: serves MCP over Streamable HTTP on port 8765.
#
#   docker build -t papermind-mcp .
#   docker run --rm --env-file .env -p 127.0.0.1:8765:8765 \
#     -v "$PWD/data:/app/data" -v "$PWD/inbox:/app/inbox" papermind-mcp
#
# Two stages: the BUILDER installs packages and downloads the embedding model; the final
# image starts clean and copies only the results, so installer leftovers never ship.

# ---------------------------------------------------------------- stage 1: build ---
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# A virtualenv is one self-contained folder, which makes it easy to copy to stage 2.
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Only the MCP server's dependencies (no web app, no Gemini/Ollama SDKs).
COPY requirements-mcp.txt .
RUN pip install -r requirements-mcp.txt

# Bundle the embedding model so the first search doesn't stall on a 67 MB download.
ENV FASTEMBED_CACHE_PATH=/opt/models
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

# -------------------------------------------------------------- stage 2: runtime ---
FROM python:3.12-slim

# Run as an unprivileged user, never as root.
RUN useradd --create-home app && mkdir -p /app/data /app/inbox && chown -R app:app /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=app:app /opt/models /opt/models

ENV PATH=/opt/venv/bin:$PATH \
    FASTEMBED_CACHE_PATH=/opt/models \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --chown=app:app app/ app/
USER app

EXPOSE 8765

# Inside the container we listen on every interface; the `-p 127.0.0.1:...` in
# `docker run` decides who on the host can actually reach it (this machine only).
CMD ["python", "app/mcp_server.py", "--http", "--host", "0.0.0.0", "--port", "8765"]
