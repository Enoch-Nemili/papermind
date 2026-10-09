"""DNS-rebinding protection: real HTTP requests through the SDK's transport, bound like Docker."""

import pytest
from mcp.server import MCPServer
from starlette.testclient import TestClient

import app.mcp_server as srv
from app.auth import auth_options
from app.http_security import transport_security

TOKEN = "t" * 40
INITIALIZE = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {},
               "clientInfo": {"name": "test", "version": "1"}},
}
HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {TOKEN}",
}
FOREIGN = "https://attacker.example"


def client(security):
    """The HTTP app as the Docker image runs it: bound to 0.0.0.0, token required."""
    app = MCPServer("rebinding-test", **auth_options(TOKEN)).streamable_http_app(
        host="0.0.0.0", transport_security=security
    )
    return TestClient(app, base_url="http://127.0.0.1:8765")


def post(security, **headers):
    with client(security) as http:
        return http.post("/mcp", json=INITIALIZE, headers={**HEADERS, **headers}).status_code


def test_sdk_default_leaves_a_docker_bind_unprotected():
    """The bug: on 0.0.0.0 the SDK skips Origin checks unless told otherwise."""
    assert post(None, Origin=FOREIGN) == 200


def test_foreign_origin_is_rejected():
    assert post(transport_security(), Origin=FOREIGN) == 403


@pytest.mark.parametrize("origin", ["http://127.0.0.1:8765", "http://localhost:6274"])
def test_local_origins_are_allowed(origin):
    assert post(transport_security(), Origin=origin) == 200


def test_requests_without_origin_are_allowed():
    """Non-browser clients (Claude Desktop, curl, the SDK) send no Origin header."""
    assert post(transport_security()) == 200


def test_foreign_host_header_is_rejected():
    assert post(transport_security(), Host="attacker.example:8765") == 421


def test_extra_hosts_and_origins_from_env(monkeypatch):
    monkeypatch.setenv("PAPERMIND_ALLOWED_HOSTS", "papers.example.com:8765, ")
    monkeypatch.setenv("PAPERMIND_ALLOWED_ORIGINS", "https://papers.example.com")
    security = transport_security()
    assert post(security, Host="papers.example.com:8765", Origin="https://papers.example.com") == 200
    assert post(security, Origin=FOREIGN) == 403


def test_server_starts_with_protection_on(monkeypatch):
    monkeypatch.setenv("PAPERMIND_TOKEN", TOKEN)
    monkeypatch.setattr("sys.argv", ["mcp_server.py", "--http", "--host", "0.0.0.0"])
    started = {}
    monkeypatch.setattr(srv.mcp, "run", lambda **kw: started.update(kw))
    srv.main()
    assert started["transport_security"].enable_dns_rebinding_protection
    assert started["host"] == "0.0.0.0"
