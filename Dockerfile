# PaperMind MCP server, containerized: serves MCP over Streamable HTTP on port 8765.
#
#   docker build -t papermind-mcp .
#   docker run --rm --env-file .env -p 127.0.0.1:8765:8765 \
#     -v "$PWD/data:/app/data" -v "$PWD/inbox:/app/inbox" papermind-mcp
FROM python:3.12-slim

WORKDIR /app

# Dependencies first: Docker caches this layer, so code-only changes rebuild in seconds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Run as an unprivileged user, never as root.
RUN useradd --create-home app && mkdir -p /app/data /app/inbox && chown -R app:app /app
USER app

# Bundle the embedding model so the first search doesn't stall on a 67 MB download.
ENV FASTEMBED_CACHE_PATH=/home/app/.cache/fastembed
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

COPY --chown=app:app app/ app/

ENV PYTHONUNBUFFERED=1
EXPOSE 8765

# Inside the container we listen on every interface; the `-p 127.0.0.1:...` in
# `docker run` decides who on the host can actually reach it (this machine only).
CMD ["python", "app/mcp_server.py", "--http", "--host", "0.0.0.0", "--port", "8765"]
