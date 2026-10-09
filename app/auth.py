"""
Bearer-token auth for PaperMind's HTTP mode.

MCP's authorization model makes the server an OAuth *resource server*: it doesn't log
anyone in, it verifies the bearer token each request carries. The SDK delegates that check
to a TokenVerifier. This one accepts a single shared secret (PAPERMIND_TOKEN), which is
the usual choice for a self-hosted tool. Moving to full OAuth later means swapping in a
verifier that validates JWTs from an authorization server (Auth0, Keycloak, ...); nothing
else changes.

stdio mode ignores all of this: there, the client launches the server as a local process.
"""

import hmac

from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings

SCOPE = "papermind"
MIN_TOKEN_LENGTH = 32
GENERATE_HINT = "python -c 'import secrets; print(secrets.token_urlsafe(32))'"


class StaticTokenVerifier:
    """Accept exactly one secret, compared in constant time so timing can't leak it."""

    def __init__(self, token):
        if len(token) < MIN_TOKEN_LENGTH:
            raise ValueError(
                f"PAPERMIND_TOKEN must be at least {MIN_TOKEN_LENGTH} characters. "
                f"Generate one with: {GENERATE_HINT}"
            )
        self._token = token.encode()

    async def verify_token(self, token):
        if hmac.compare_digest(token.encode(), self._token):
            return AccessToken(token=token, client_id="papermind-client", scopes=[SCOPE])
        return None


def auth_options(token):
    """Keyword arguments for MCPServer: bearer auth when a token is configured, else none."""
    if not token:
        return {}
    return {
        "token_verifier": StaticTokenVerifier(token),
        "auth": AuthSettings(
            # Required by the SDK's settings model; only published when the server also acts
            # as (or advertises) an OAuth authorization server, which this mode doesn't.
            issuer_url="http://localhost",
            resource_server_url=None,
            required_scopes=[SCOPE],
        ),
    }
