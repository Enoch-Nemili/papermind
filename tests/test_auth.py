"""Bearer-token auth for HTTP mode: unit checks plus real HTTP requests through the SDK's middleware."""

import pytest
from mcp.server import MCPServer
from starlette.testclient import TestClient

import app.mcp_server as srv
from app.auth import StaticTokenVerifier, auth_options

TOKEN = "t" * 40
INITIALIZE = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {},
               "clientInfo": {"name": "test", "version": "1"}},
}
HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


@pytest.mark.anyio
async def test_verifier_accepts_only_the_exact_token():
    verifier = StaticTokenVerifier(TOKEN)
    assert (await verifier.verify_token(TOKEN)).scopes == ["papermind"]
    assert await verifier.verify_token(TOKEN[:-1]) is None
    assert await verifier.verify_token(TOKEN + "x") is None
    assert await verifier.verify_token("") is None


def test_short_tokens_are_refused():
    with pytest.raises(ValueError, match="at least 32"):
        StaticTokenVerifier("password123")


def test_no_token_means_no_auth_config():
    assert auth_options(None) == {}
    assert auth_options("") == {}


@pytest.fixture
def http():
    """A real Streamable HTTP app with PaperMind's auth wiring, driven over HTTP."""
    app = MCPServer("auth-test", **auth_options(TOKEN)).streamable_http_app()
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        yield client


@pytest.mark.parametrize("auth_header", [None, "Bearer wrong-token", f"Basic {TOKEN}"])
def test_requests_without_the_right_token_get_401(http, auth_header):
    headers = dict(HEADERS)
    if auth_header:
        headers["Authorization"] = auth_header
    response = http.post("/mcp", json=INITIALIZE, headers=headers)
    assert response.status_code == 401


def test_the_right_token_gets_through(http):
    response = http.post("/mcp", json=INITIALIZE, headers={**HEADERS, "Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 200


def test_refuses_to_serve_on_the_network_without_a_token(monkeypatch):
    monkeypatch.delenv("PAPERMIND_TOKEN", raising=False)
    monkeypatch.setattr("sys.argv", ["mcp_server.py", "--http", "--host", "0.0.0.0"])
    monkeypatch.setattr(srv.mcp, "run", lambda **kw: pytest.fail("must not start"))
    with pytest.raises(SystemExit):
        srv.main()
