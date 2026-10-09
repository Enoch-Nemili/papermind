"""Unit tests for small pure helpers (no MCP client needed)."""

import pytest
from mcp.server.mcpserver.exceptions import ToolError

import app.mcp_server as srv


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [({"page": 0}, "1"), ({"page": 9}, "10"), ({"page": 2, "page_label": "iii"}, "iii"), ({}, "?")],
)
def test_human_page(metadata, expected):
    assert srv.human_page(metadata) == expected


def test_file_inside_allows_plain_names(tmp_path):
    assert srv.file_inside(tmp_path, "paper.pdf") == (tmp_path / "paper.pdf").resolve()


@pytest.mark.parametrize("name", ["../x.pdf", "a/b.pdf", "/etc/passwd"])
def test_file_inside_blocks_escapes_and_subfolders(tmp_path, name):
    with pytest.raises(ToolError):
        srv.file_inside(tmp_path, name)
