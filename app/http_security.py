"""
DNS-rebinding protection for HTTP mode.

A web page open in the user's browser can make the browser send requests to
http://127.0.0.1:8765 by pointing its own domain at that address ("DNS rebinding").
The MCP spec requires servers to validate the Origin header and answer an unknown
origin with 403 Forbidden.

The MCP SDK turns this check on by itself only when the server binds a loopback address.
The Docker image binds 0.0.0.0 (it has to, so the port mapping can reach it), which left
the check off. So we configure it explicitly for every bind address:

  - Host header must be localhost (any port), plus PAPERMIND_ALLOWED_HOSTS
  - Origin header, when present, must be localhost (any port), plus PAPERMIND_ALLOWED_ORIGINS

Both variables take comma-separated values, e.g. when serving behind a domain:
    PAPERMIND_ALLOWED_HOSTS=papers.example.com
    PAPERMIND_ALLOWED_ORIGINS=https://papers.example.com
"""

import os

from mcp.server.transport_security import TransportSecuritySettings

LOCAL_HOSTS = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
LOCAL_ORIGINS = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]


def _env_list(name):
    return [value.strip() for value in os.getenv(name, "").split(",") if value.strip()]


def transport_security():
    """Host/Origin checks for the Streamable HTTP transport, on for every bind address."""
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=LOCAL_HOSTS + _env_list("PAPERMIND_ALLOWED_HOSTS"),
        allowed_origins=LOCAL_ORIGINS + _env_list("PAPERMIND_ALLOWED_ORIGINS"),
    )
