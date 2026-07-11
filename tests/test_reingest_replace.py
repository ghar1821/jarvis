"""
Regression tests for the replace-on-duplicate reingest flow.

Before this fix, `confirm_duplicate` (jarvis/kb/cli.py) and `duplicate_notice`
(jarvis/chat/chat.py) deleted the old entry's chunks the moment the user opted
into a same-source duplicate re-add — before the new PDF was downloaded or
converted. A failed download/conversion then left the knowledge base with
neither the old content nor the new, silently destroying irreplaceable
annotation chunks along with it.

The fix makes both helpers pure gates (proceed/replace-source decision, no
deletion) and moves the actual `delete_by_metadata` call to run only after
the new content has been produced successfully, immediately before the add
call. These tests preseed a paper "by source" with a distinctive marker
chunk and drive both the chat and CLI reingest paths through success and
failure to prove the delete is correctly sequenced.
"""

from pathlib import Path

import pymupdf
import pytest
import requests

from jarvis.core.config import Config
from jarvis.core.errors import ConversionError
from jarvis.kb.cli import cmd_add
from jarvis.kb.store import add_texts
from jarvis.chat.chat import _add_document

MARKER_TEXT = "MARKER_CHUNK — irreplaceable annotation from the original ingest."


def _preseed_marker_chunk(source: str, title: str, store) -> None:
    """Index one full-text paper chunk under `source`, standing in for a
    previously ingested paper (including annotations, which share source)."""
    add_texts(
        content=MARKER_TEXT,
        doc_type="paper",
        visibility="public",
        source=source,
        extra_metadata={"title": title, "storage_mode": "full_text"},
        store=store,
    )


def _chunks_for_source(source: str, store) -> list[str]:
    result = store._collection.get(where={"source": {"$eq": source}}, include=["documents"])
    return result["documents"]


class _StubProvider:
    """Stands in for make_provider(): metadata inference degrades to {} (no
    fields), summarize() returns a canned string. Never hits a real LLM."""

    def complete(self, messages, max_tokens=300, context_length=None):
        return "{}"

    def summarize(self, title, source, max_tokens=2048):
        return "A canned summary."


def _fake_arxiv_download_writing(text: str):
    """A download_arxiv_pdf stand-in that writes a real tiny PDF locally."""

    def fake_download(arxiv_id: str, dest_dir: Path) -> Path:
        pdf_path = dest_dir / f"{arxiv_id.replace('/', '_')}.pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=12)
        doc.save(pdf_path)
        doc.close()
        return pdf_path

    return fake_download


def _arxiv_paper_meta() -> dict:
    return {
        "title": "Reingest Target Paper",
        "abstract": "An abstract.",
        "link": "https://arxiv.org/abs/2406.04093",
        "authors": "Ada Lovelace",
        "doi": "",
    }


# ── (a)/(b) chat._add_document — arXiv full-text reingest ─────────────────────


def test_chat_reingest_download_failure_preserves_old_chunks(store, monkeypatch):
    """
    A same-source duplicate re-add whose PDF download then fails must leave
    the old entry completely untouched — this is the data-loss window Fix 1
    closes. Before the fix, duplicate_notice deleted the old chunks the
    moment allow_duplicate=true was seen, regardless of what happened next.
    """
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    paper = _arxiv_paper_meta()
    _preseed_marker_chunk(paper["link"], paper["title"], store)

    monkeypatch.setattr("jarvis.digest.arxiv.fetch.fetch_arxiv_paper", lambda arxiv_id: paper)

    def failing_download(arxiv_id: str, dest_dir: Path) -> Path:
        raise requests.HTTPError("404 Client Error: Not Found")

    monkeypatch.setattr("jarvis.digest.arxiv.convert.download_arxiv_pdf", failing_download)

    result = _add_document(
        {"source": paper["link"], "mode": "full_text", "allow_duplicate": True},
        _StubProvider(),
    )

    # The failure surfaces as an error string (caught by _add_document's
    # outer try/except), not a raised exception.
    assert "error" in result.lower() or result.startswith("[Error")

    remaining = _chunks_for_source(paper["link"], store)
    assert MARKER_TEXT in remaining, "the old entry must survive a failed reingest"


def test_chat_reingest_success_replaces_old_chunks(store, monkeypatch):
    """
    A same-source duplicate re-add that succeeds end-to-end deletes the old
    entry exactly once and leaves only the new content behind.
    """
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    paper = _arxiv_paper_meta()
    _preseed_marker_chunk(paper["link"], paper["title"], store)

    monkeypatch.setattr("jarvis.digest.arxiv.fetch.fetch_arxiv_paper", lambda arxiv_id: paper)
    monkeypatch.setattr(
        "jarvis.digest.arxiv.convert.download_arxiv_pdf",
        _fake_arxiv_download_writing("Freshly re-downloaded full text of the paper."),
    )

    result = _add_document(
        {"source": paper["link"], "mode": "full_text", "allow_duplicate": True},
        _StubProvider(),
    )
    assert result.startswith("Added ")

    remaining = _chunks_for_source(paper["link"], store)
    assert MARKER_TEXT not in remaining, "the old marker chunk must be gone"
    assert remaining, "new content must be present"

    # Exactly one entry for the source — no leftover duplicate copies.
    all_for_source = store._collection.get(
        where={"source": {"$eq": paper["link"]}}, include=["metadatas"]
    )
    titles = {m.get("title") for m in all_for_source["metadatas"]}
    assert titles == {paper["title"]}


# ── (c) same-title-different-source leaves the other entry untouched ─────────


def test_chat_reingest_same_title_different_source_leaves_other_entry(store, tmp_path, monkeypatch):
    """
    A same-title-but-different-source "duplicate" is a genuinely separate
    entry: allow_duplicate must never delete the other source's chunks, even
    though duplicate_notice/confirm_duplicate() fires on the title match.
    """
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    shared_title = "Shared Title Paper"
    original_source = "https://arxiv.org/abs/1111.11111"
    _preseed_marker_chunk(original_source, shared_title, store)

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Second copy of a paper with the same title.", fontsize=12)
    pdf_path = tmp_path / "second_copy.pdf"
    doc.save(pdf_path)
    doc.close()

    result = _add_document(
        {
            "source": str(pdf_path), "doc_type": "paper", "visibility": "public",
            "mode": "summary", "title": shared_title, "allow_duplicate": True,
        },
        _StubProvider(),
    )
    assert result.startswith("Added ")

    # The original entry (different source) must be completely untouched.
    original_chunks = _chunks_for_source(original_source, store)
    assert MARKER_TEXT in original_chunks

    # The new entry exists as a separate document under its own source.
    new_chunks = _chunks_for_source(pdf_path.resolve().as_uri(), store)
    assert new_chunks


# ── CLI path — cmd_add local-PDF full-text reingest ────────────────────────────


def _cli_args(**overrides):
    from argparse import Namespace

    defaults = dict(
        input="", score=0, track="", title="", authors="", doi="",
        visibility="public", doc_type="paper", provider="", full_text=False,
        figures=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_cli_reingest_conversion_failure_preserves_old_chunks(store, tmp_path, monkeypatch):
    """
    `kb add <pdf> --full-text`, answering y to the duplicate prompt, must not
    delete the old entry if pdf_to_markdown then fails to convert it. This is
    the CLI counterpart of the chat-path download-failure regression above —
    before Fix 1, cli.py's confirm_duplicate deleted the old chunks
    immediately on the "y" answer, before any conversion was attempted.
    """
    pdf_path = tmp_path / "reingest_me.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Original local paper content.", fontsize=12)
    doc.save(pdf_path)
    doc.close()
    pdf_path = pdf_path.resolve()

    source = pdf_path.as_uri()
    _preseed_marker_chunk(source, pdf_path.stem, store)

    monkeypatch.setattr("jarvis.core.config.get_config", lambda: Config())
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    monkeypatch.setattr("jarvis.core.llm.make_provider", lambda provider_str: _StubProvider())
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")

    def broken_convert(path):
        raise ConversionError("conversion exploded")

    monkeypatch.setattr("jarvis.kb.convert.pdf_to_markdown", broken_convert)

    with pytest.raises(SystemExit):
        cmd_add(_cli_args(input=str(pdf_path), full_text=True))

    remaining = _chunks_for_source(source, store)
    assert MARKER_TEXT in remaining, "the old entry must survive a failed conversion"


def test_cli_reingest_success_replaces_old_chunks(store, tmp_path, monkeypatch):
    """The CLI counterpart of the successful-reingest test: a clean convert
    deletes the old entry exactly once and leaves only the new content."""
    pdf_path = tmp_path / "reingest_ok.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Freshly converted local paper content.", fontsize=12)
    doc.save(pdf_path)
    doc.close()
    pdf_path = pdf_path.resolve()

    source = pdf_path.as_uri()
    _preseed_marker_chunk(source, pdf_path.stem, store)

    monkeypatch.setattr("jarvis.core.config.get_config", lambda: Config())
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    monkeypatch.setattr("jarvis.core.llm.make_provider", lambda provider_str: _StubProvider())
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")

    cmd_add(_cli_args(input=str(pdf_path), full_text=True))

    remaining = _chunks_for_source(source, store)
    assert MARKER_TEXT not in remaining
    assert remaining
