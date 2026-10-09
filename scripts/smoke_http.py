"""Smoke-test a running PaperMind MCP server over Streamable HTTP.

    python scripts/smoke_http.py                         # default: http://127.0.0.1:8765/mcp
    python scripts/smoke_http.py http://host:port/mcp

Sends PAPERMIND_TOKEN (from the environment or .env) as a bearer token when it's set.
"""

import os
import sys

import anyio
import httpx2
from dotenv import load_dotenv
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/mcp"


def transport():
    load_dotenv()
    token = os.getenv("PAPERMIND_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return streamable_http_client(URL, http_client=httpx2.AsyncClient(headers=headers))


async def main():
    async with Client(transport()) as client:
        tools = [t.name for t in (await client.list_tools()).tools]
        print(f"connected to {URL}")
        print(f"tools:   {tools}")

        library = (await client.call_tool("list_papers", {})).structured_content
        print(f"library: {library['count']} papers")

        result = await client.call_tool(
            "search_papers", {"query": "KV cache paging memory fragmentation", "k": 1}
        )
        top = result.structured_content["result"][0]
        print(f"search:  top hit {top['source']} p.{top['page']} (relevance {top['relevance']})")


anyio.run(main)
