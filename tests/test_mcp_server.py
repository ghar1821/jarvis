"""
Tests for the MCP server in jarvis/mcp/server.py.

Claude Desktop is a cloud client, so these check the one thing that matters
most: nothing private comes back through any tool. Calls go through the SDK's
in-process Client, so they exercise the real MCP path (argument validation,
the ToolError -> is_error conversion) rather than calling the functions
directly.

The store fixture comes from conftest.py (real embeddings, isolated
collection); get_store()/get_config() are monkeypatched where the chat
helpers call the process singletons, as in test_privacy_guard.py.
"""

import asyncio
import os
import subprocess
import sys

import pytest
from mcp import Client

from jarvis.core.config import Config
from jarvis.kb.store import add_texts
from jarvis.mcp.server import server


def _call(tool: str, arguments: dict | None = None):
    """Call one tool over an in-process MCP session and return (text, is_error)."""

    async def run():
        async with Client(server) as client:
            return await client.call_tool(tool, arguments or {})

    result = asyncio.run(run())
    text = "\n".join(block.text for block in result.content if block.type == "text")
    return text, result.is_error


@pytest.fixture
def kb(store, tmp_path, monkeypatch):
    """
    A vault on disk plus an index holding one public and one private note
    on the same topic, and one public paper.
    """
    (tmp_path / "projects").mkdir()
    (tmp_path / "private").mkdir()
    (tmp_path / "projects" / "quantum.md").write_text("# Quantum\nPublic overview of quantum sensing.")
    (tmp_path / "private" / "quantum.md").write_text("# Secret\nPrivate budget worries about quantum sensing.")

    config = Config(vault_path=tmp_path, private_vault_dirs=["private"])
    monkeypatch.setattr("jarvis.kb.store.get_config", lambda: config)
    monkeypatch.setattr("jarvis.mcp.server.get_config", lambda: config)
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)

    add_texts(content="Public overview of quantum sensing.",
              doc_type="note", visibility="public", source="local",
              extra_metadata={"file_path": "projects/quantum.md", "title": "Quantum public",
                              "category": "project", "status": "active"},
              store=store)
    add_texts(content="Private budget worries about quantum sensing.",
              doc_type="note", visibility="private", source="local",
              extra_metadata={"file_path": "private/quantum.md", "title": "Quantum secret",
                              "category": "salary_negotiation", "status": "offer"},
              store=store)
    add_texts(content="We present a diamond magnetometer for quantum sensing.",
              doc_type="paper", visibility="public", source="https://arxiv.org/abs/2401.00001",
              extra_metadata={"title": "Diamond magnetometry"},
              store=store)
    return tmp_path


def test_tools_are_all_read_only():
    """
    The server exposes exactly the five read tools, each annotated read-only,
    and nothing that writes or deletes.

    Expected output: {search_kb, get_document, read_note, list_documents, kb_stats}
    """
    tools = asyncio.run(server.list_tools())
    assert {tool.name for tool in tools} == {
        "search_kb", "get_document", "read_note", "list_documents", "kb_stats",
    }
    assert all(tool.annotations.read_only_hint for tool in tools)


def test_search_returns_public_hits_and_withholds_private(kb):
    """
    A query matching both notes returns only the public one, with the static
    "excluded" caveat, wrapped in the retrieved-data delimiters.
    """
    text, is_error = _call("search_kb", {"query": "quantum sensing", "kinds": ["notes"]})
    assert not is_error
    assert "Public overview" in text
    assert "budget worries" not in text
    assert "excluded from these results" in text
    assert "BEGIN RETRIEVED DATA" in text


def test_search_matching_only_private_is_an_error_without_content(kb):
    """
    Filtering down to the private note's category leaves only private matches,
    which is a hard stop: an error result carrying none of the note's text.
    """
    text, is_error = _call(
        "search_kb",
        {"query": "budget worries", "kinds": ["notes"], "category": "salary_negotiation"},
    )
    assert is_error
    assert "cannot be accessed by a cloud provider" in text
    assert "budget worries" not in text


def test_read_note_public_ok_private_refused(kb):
    """read_note opens a public note and refuses a private one with no content."""
    text, is_error = _call("read_note", {"path": "projects/quantum.md"})
    assert not is_error and "Public overview" in text

    text, is_error = _call("read_note", {"path": "private/quantum.md"})
    assert is_error
    assert "budget worries" not in text


def test_read_note_refuses_symlink_into_private(kb):
    """
    A symlink in a public folder pointing into private/ is classified by its
    resolved target, so it is refused too.
    """
    os.symlink(kb / "private" / "quantum.md", kb / "projects" / "shortcut.md")
    text, is_error = _call("read_note", {"path": "projects/shortcut.md"})
    assert is_error
    assert "budget worries" not in text


def test_get_document_reads_a_paper(kb):
    """get_document returns a public paper's stored text."""
    text, is_error = _call("get_document", {"source": "https://arxiv.org/abs/2401.00001"})
    assert not is_error
    assert "diamond magnetometer" in text


def test_get_document_refuses_the_shared_note_source(kb):
    """
    source="local" would mean every vault note at once, so it is refused with
    a pointer to read_note rather than read (or tripping the privacy stop).
    """
    text, is_error = _call("get_document", {"source": "local"})
    assert is_error
    assert "read_note" in text
    assert "Public overview" not in text and "budget worries" not in text


def test_list_documents_and_stats_hide_private_notes(kb):
    """
    Neither the note listing nor kb_stats names the private note, its path,
    or its record values.
    """
    listing, is_error = _call("list_documents", {"kind": "notes"})
    assert not is_error
    assert "Quantum public" in listing
    assert "Quantum secret" not in listing and "private/quantum.md" not in listing

    stats, is_error = _call("kb_stats")
    assert not is_error
    assert "project" in stats and "active" in stats
    assert "salary_negotiation" not in stats and "offer" not in stats


def test_importing_the_server_loads_no_store():
    """
    The handshake must be fast, so importing the module (as Claude Desktop's
    launch does) must not open the store or load the embedding model. Run in
    a fresh interpreter, since this test session has already loaded both.
    """
    # Importing the store module is itself what pulls in the embedding stack,
    # so "not imported" is the whole check: no module, no store, no model.
    probe = (
        "import sys, jarvis.mcp.server; "
        "print('jarvis.kb.store' in sys.modules, 'sentence_transformers' in sys.modules)"
    )
    output = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert output == "False False"
