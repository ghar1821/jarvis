"""
Tests for jarvis/mcp/server.py — the read-only MCP surface.

Every MCP client is treated as a cloud model, so the most important property
is that nothing private ever comes back. That holds two ways and both are
tested: private folders are never indexed (the vault here is synced through
the real mirror), and an index built by an older version that still holds a
private chunk is filtered by the visibility backstop.

Calls go through the SDK's in-process Client, so they exercise the real MCP
path — argument validation and the ToolError → is_error conversion — rather
than calling the functions directly. `open -R` is the only boundary replaced.
"""

import asyncio
import os
import subprocess
import sys

import pytest
from mcp import Client

from jarvis.core.config import Config
from jarvis.core.errors import KBCorruptionError
from jarvis.kb.mirror import refresh_vault
from jarvis.kb.store import add_texts
from jarvis.mcp.server import server

PAPER_SOURCE = "file:///papers/magnetometry.pdf"


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
    A vault synced through the real mirror (one public note, one private
    note on the same topic), one public paper, and one private chunk left
    over from an older version's index.
    """
    vault = tmp_path / "vault"
    (vault / "projects").mkdir(parents=True)
    (vault / "private").mkdir()
    (vault / ".obsidian").mkdir()
    (vault / "projects" / "quantum.md").write_text(
        "---\ntype: project\nstatus: active\n---\n# Quantum public\nPublic overview of quantum sensing."
    )
    (vault / "private" / "quantum.md").write_text("# Quantum secret\nPrivate budget worries about quantum sensing.")
    (vault / ".obsidian" / "app.json").write_text('{"secret": "settings"}')

    config = Config(vault_path=vault, private_vault_dirs=["private"])
    for module in ("jarvis.kb.store", "jarvis.kb.mirror", "jarvis.mcp.server"):
        monkeypatch.setattr(f"{module}.get_config", lambda: config)
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)

    refresh_vault(vault, store)
    add_texts(content="We present a diamond magnetometer for quantum sensing.",
              doc_type="paper", visibility="public", source=PAPER_SOURCE,
              extra_metadata={"title": "Diamond magnetometry", "authors": "N. V. Centre",
                              "doi": "10.1234/diamond", "file_path": "/papers/magnetometry.pdf"},
              store=store)
    # What an index built before private folders stopped being indexed can
    # still hold, until `kb rebuild`.
    add_texts(content="Legacy private salary negotiation about quantum sensing.",
              doc_type="note", visibility="private", source="local",
              extra_metadata={"file_path": "private/legacy.md", "title": "Legacy secret",
                              "category": "salary_negotiation", "status": "offer"},
              store=store)
    return vault


# ── Surface ────────────────────────────────────────────────────────────────────

def test_every_tool_is_read_only():
    """Seven tools, all annotated read-only, and none that adds or removes."""
    tools = asyncio.run(server.list_tools())
    assert {tool.name for tool in tools} == {
        "search_kb", "get_document", "read_note", "list_documents", "kb_stats",
        "has_paper", "reveal_in_finder",
    }
    assert all(tool.annotations.read_only_hint for tool in tools)


def test_importing_the_server_loads_no_store():
    """
    The handshake must be fast, so importing the module (as a client's launch
    does) must not import the store or the embedding stack. Run in a fresh
    interpreter, since this test session has already loaded both.
    """
    probe = (
        "import sys, jarvis.mcp.server; "
        "print('jarvis.kb.store' in sys.modules, 'sentence_transformers' in sys.modules)"
    )
    output = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert output == "False False"


# ── search_kb ──────────────────────────────────────────────────────────────────

def test_search_returns_public_hits_with_paths_and_nothing_private(kb):
    """
    The public note comes back with its absolute path, wrapped as data; the
    private note was never indexed and the legacy private chunk is filtered.
    """
    text, is_error = _call("search_kb", {"query": "quantum sensing budget salary", "n_results": 10})
    assert not is_error
    assert "Public overview" in text
    assert str(kb / "projects" / "quantum.md") in text
    assert "budget worries" not in text and "salary" not in text.lower()
    assert "BEGIN RETRIEVED DATA" in text


def test_paper_hits_show_the_doi_link_and_source(kb):
    text, _ = _call("search_kb", {"query": "diamond magnetometer", "kinds": ["papers"]})
    assert "https://doi.org/10.1234/diamond" in text
    assert PAPER_SOURCE in text


def test_record_filters_cannot_surface_the_legacy_private_chunk(kb):
    """Filtering to the private record's exact category finds nothing at all."""
    text, is_error = _call(
        "search_kb", {"query": "salary negotiation", "category": "salary_negotiation"},
    )
    assert not is_error
    assert text == "No matching documents."


def test_a_knowledge_base_error_is_relayed_verbatim(kb, monkeypatch):
    def broken_search(*args, **kwargs):
        raise KBCorruptionError("the index could not resolve a chunk id")

    monkeypatch.setattr("jarvis.kb.store.search", broken_search)
    text, is_error = _call("search_kb", {"query": "anything"})
    assert is_error
    assert "KNOWLEDGE BASE ERROR" in text and "could not resolve a chunk id" in text


# ── read_note ──────────────────────────────────────────────────────────────────

def test_read_note_by_relative_or_absolute_path(kb):
    for path in ("projects/quantum.md", str(kb / "projects" / "quantum.md")):
        text, is_error = _call("read_note", {"path": path})
        assert not is_error and "Public overview" in text


@pytest.mark.parametrize("path", ["private/quantum.md", ".obsidian/app.json"])
def test_read_note_refuses_private_and_hidden_paths(kb, path):
    text, is_error = _call("read_note", {"path": path})
    assert is_error
    assert "budget worries" not in text and "settings" not in text


def test_read_note_refuses_a_symlink_into_private(kb):
    """Judged on the resolved path, so a public-folder symlink cannot reach in."""
    os.symlink(kb / "private" / "quantum.md", kb / "projects" / "shortcut.md")
    text, is_error = _call("read_note", {"path": "projects/shortcut.md"})
    assert is_error and "budget worries" not in text


def test_read_note_refuses_paths_outside_the_vault(kb, tmp_path):
    (tmp_path / "outside.md").write_text("Not part of the vault.")
    text, is_error = _call("read_note", {"path": "../outside.md"})
    assert is_error and "outside the vault" in text


# ── get_document / list_documents / kb_stats ───────────────────────────────────

def test_get_document_reads_a_paper(kb):
    text, is_error = _call("get_document", {"source": PAPER_SOURCE})
    assert not is_error
    assert "diamond magnetometer" in text and "https://doi.org/10.1234/diamond" in text


def test_get_document_refuses_the_shared_note_source(kb):
    text, is_error = _call("get_document", {"source": "local"})
    assert is_error and "read_note" in text
    assert "Public overview" not in text


def test_listings_and_stats_name_nothing_private(kb):
    listing, _ = _call("list_documents", {"kind": "notes"})
    assert "Quantum public" in listing
    assert "Legacy secret" not in listing and "secret" not in listing.lower()

    stats, _ = _call("kb_stats")
    assert "project" in stats and "active" in stats
    assert "salary_negotiation" not in stats and "offer" not in stats


# ── has_paper ──────────────────────────────────────────────────────────────────

@pytest.fixture
def preprint(kb, store):
    """An arXiv preprint saved under its filename, so its title is just the ID."""
    add_texts(content="Sparse autoencoders for single-cell data.",
              doc_type="paper", visibility="public", source="file:///papers/2411.01234v2.pdf",
              extra_metadata={"title": "2411.01234v2", "doi": "", "arxiv_id": "2411.01234",
                              "file_path": "/papers/2411.01234v2.pdf"},
              store=store)


@pytest.mark.parametrize("arguments, matched_by", [
    # A doi.org link with the DOI in a different case from the indexed one.
    ({"doi": "https://doi.org/10.1234/DIAMOND"}, "DOI"),
    # Case and punctuation in the title are ignored.
    ({"title": "  diamond MAGNETOMETRY. "}, "title"),
    # A DOI that misses (e.g. the journal version) still matches on the title.
    ({"doi": "10.9999/journal.version", "title": "Diamond magnetometry"}, "title"),
])
def test_has_paper_finds_a_paper_by_doi_or_title(kb, arguments, matched_by):
    text, is_error = _call("has_paper", arguments)
    assert not is_error
    assert "Yes, 1 matching paper" in text
    assert "Diamond magnetometry" in text and "/papers/magnetometry.pdf" in text
    assert f"Source (for get_document): {PAPER_SOURCE}" in text
    assert f"Matched by: {matched_by}" in text
    assert "BEGIN RETRIEVED DATA" in text


@pytest.mark.parametrize("arguments", [
    {"arxiv_id": "2411.01234"},
    # Any version, and an arxiv.org link.
    {"arxiv_id": "https://arxiv.org/abs/2411.01234v3"},
    # arXiv's own DOI carries the ID.
    {"doi": "10.48550/arXiv.2411.01234"},
])
def test_has_paper_finds_a_preprint_by_arxiv_id(preprint, arguments):
    text, is_error = _call("has_paper", arguments)
    assert not is_error
    assert "/papers/2411.01234v2.pdf" in text and "Matched by: arXiv ID" in text


def test_has_paper_says_no_when_nothing_matches(preprint):
    text, is_error = _call("has_paper", {"doi": "10.5555/unknown", "arxiv_id": "2501.00001",
                                         "title": "A paper nobody saved"})
    assert not is_error
    assert text.startswith("No paper in the knowledge base matches")


def test_has_paper_never_matches_a_note(kb):
    """Only papers count: a note with the same title is not "having the paper"."""
    text, _ = _call("has_paper", {"title": "Quantum public"})
    assert text.startswith("No paper")


@pytest.mark.parametrize("arguments, expected", [
    ({}, "at least one of doi, arxiv_id or title"),
    ({"doi": "not a doi"}, "does not look like a DOI"),
    ({"arxiv_id": "abc"}, "does not look like an arXiv ID"),
])
def test_has_paper_rejects_unreadable_input(kb, arguments, expected):
    text, is_error = _call("has_paper", arguments)
    assert is_error and expected in text


# ── reveal_in_finder ───────────────────────────────────────────────────────────

@pytest.fixture
def opened(monkeypatch):
    """Record `open -R` calls instead of opening Finder windows."""
    calls = []
    monkeypatch.setattr("jarvis.mcp.server.subprocess.run",
                        lambda args, check, timeout: calls.append(args))
    monkeypatch.setattr("jarvis.mcp.server.sys.platform", "darwin")
    return calls


def test_reveal_an_indexed_note(kb, opened):
    note = kb / "projects" / "quantum.md"
    text, is_error = _call("reveal_in_finder", {"path": str(note)})
    assert not is_error
    assert opened == [["open", "-R", str(note)]]


def test_reveal_refuses_anything_not_indexed(kb, opened, tmp_path):
    """
    Only an indexed document's own path is accepted, so a path planted in a
    document cannot point Finder anywhere else — including at a private note
    that exists on disk but was never indexed.
    """
    for path in ("/etc/hosts", str(kb / "private" / "quantum.md"), "private/legacy.md"):
        text, is_error = _call("reveal_in_finder", {"path": path})
        assert is_error and "not an indexed document" in text
    assert opened == []
