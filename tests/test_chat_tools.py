"""
Tests for chunk-first retrieval in jarvis/chat/chat.py — the chat agent should
be able to answer from search hits and the get_document tool without falling
back to reading raw files.

Covers:
- _search_kb returns the full chunk text (previously
  truncated to 300 chars), so a long passage stays fully visible to the model.
- _get_document: pagination (15 chunks/page), the header format, the
  summary-mode honesty note, and unknown-source handling.
- _dispatch_tool wraps get_document's output in the RETRIEVED DATA markers
  and flags the session private when the local provider returns private
  content, exactly like the other retrieval tools.

Privacy hard-stops for get_document are covered separately in
test_privacy_guard.py.
"""

from pathlib import Path

import pytest

from jarvis.chat.chat import _dispatch_tool, _get_document, _search_kb
from jarvis.chat.sessions import new_session
from jarvis.kb.store import add_paper, add_texts


# ── Full-text hits (no more 300-char truncation) ────────────────────────────────

def test_search_kb_papers_returns_text_beyond_300_chars(store, monkeypatch):
    """
    A paper chunk longer than 300 characters must appear in full in
    _search_kb's output — the old behaviour truncated with "...".

    Input:  a paper summary >300 chars, indexed via add_paper
    Expected output: the full summary text is present, with no "..." elision
    """
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    long_summary = (
        "This paper introduces a graph neural network architecture for "
        "predicting protein-protein interaction networks from sequence data "
        "alone. The model combines message passing over a learned residue "
        "graph with an attention mechanism that highlights binding-site "
        "candidates. Benchmarked against three public interaction datasets, "
        "it improves F1 by twelve points over the prior state of the art "
        "while requiring an order of magnitude less training data."
    )
    assert len(long_summary) > 300
    paper = {"link": "https://arxiv.org/abs/9999.00001", "title": "GNN for PPI Prediction"}
    add_paper(paper, dense_summary=long_summary, store=store)

    result, _ = _search_kb({"kinds": ["papers"], "query": "protein interaction graph neural network"}, "ollama")
    assert long_summary in result
    assert "..." not in result


def test_search_kb_notes_returns_text_beyond_300_chars(store, monkeypatch):
    """
    Same contract for notes: full chunk text visible, no truncation.

    Input:  a note chunk >300 chars
    Expected output: full text present, no "..." elision
    """
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    long_note = (
        "Meeting notes on the wombat burrow census project: we walked the "
        "northern transect and counted fourteen active burrows, six of which "
        "showed fresh digging within the last week. Soil moisture readings "
        "were taken at each site and will be cross-referenced against the "
        "rainfall records from the regional station once they are digitised. "
        "Next visit should extend the transect another two kilometres east."
    )
    assert len(long_note) > 300
    add_texts(content=long_note, doc_type="note", visibility="public",
              source="local", extra_metadata={"file_path": "wombats.md", "title": "Wombat census"},
              store=store)

    result, _ = _search_kb({"kinds": ["notes"], "query": "wombat burrow census transect"}, "ollama")
    assert long_note in result
    assert "..." not in result


def test_search_kb_notes_includes_section_breadcrumb_when_present(store, monkeypatch):
    """
    A hit under a markdown heading carries a "Section:" line naming the
    heading breadcrumb, giving the model context beyond raw chunk text.

    Input:  a note with a "## Results" heading
    Expected output: "Section: Results" appears in the rendered hit
    """
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    add_texts(content="## Results\nThe population estimate came to roughly two hundred individuals.",
              doc_type="note", visibility="public", source="local",
              extra_metadata={"file_path": "survey.md", "title": "Survey"}, store=store)

    result, _ = _search_kb({"kinds": ["notes"], "query": "population estimate two hundred individuals"}, "ollama")
    assert "Section: Results" in result


# ── _get_document pagination ────────────────────────────────────────────────────

def _index_many_chunks(store, source: str, n: int, title: str = "Long Paper") -> None:
    """Index n distinct, individually-searchable chunks under one source."""
    for i in range(n):
        add_texts(
            content=f"Chunk number {i} discusses topic area {i} of the long paper in detail.",
            doc_type="paper", visibility="public", source=source,
            extra_metadata={"title": title},
            store=store,
        )


def test_get_document_paginates_15_per_page(store, monkeypatch):
    """
    With more than 15 chunks stored, page 1 returns exactly the first 15 and
    names the total page count; page 2 returns the remainder.

    Input:  22 chunks under one source
    Expected output: page 1 header says "page 1 of 2" and contains chunk 0 but
        not chunk 15; page 2 header says "page 2 of 2" and contains chunk 15
    """
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    source = "file:///long-paper.pdf"
    _index_many_chunks(store, source, 22)

    page1, _ = _get_document({"source": source, "page": 1}, "ollama")
    assert "page 1 of 2" in page1
    assert "chunks 1–15 of 22" in page1
    assert "Chunk number 0 " in page1
    assert "Chunk number 15 " not in page1
    assert "Call get_document(source, page=2) for more." in page1

    page2, _ = _get_document({"source": source, "page": 2}, "ollama")
    assert "page 2 of 2" in page2
    assert "chunks 16–22 of 22" in page2
    assert "Chunk number 15 " in page2
    assert "Call get_document" not in page2  # last page: no "for more" hint


def test_get_document_unknown_source_returns_not_found(store, monkeypatch):
    """
    Input:  a source that was never indexed
    Expected output: a "[No document found ...]" string, not an exception
    """
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    result, saw_private = _get_document({"source": "file:///nope.pdf"}, "ollama")
    assert "No document found" in result
    assert saw_private is False


def test_get_document_summary_mode_appends_honesty_note(store, monkeypatch):
    """
    A document stored with storage_mode="summary" gets an appended note that
    the full text isn't in the KB — the model should not claim to have read
    the whole paper from a summary.

    Input:  a chunk with storage_mode="summary"
    Expected output: the honesty note is present in the result
    """
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    source = "https://arxiv.org/abs/1111.00001"
    add_texts(content="A dense one-paragraph summary of the paper's contribution.",
              doc_type="paper", visibility="public", source=source,
              extra_metadata={"title": "Summarised Paper", "storage_mode": "summary"},
              store=store)

    result, _ = _get_document({"source": source}, "ollama")
    assert "not in the knowledge base" in result
    assert "mode='full_text'" in result


# ── _dispatch_tool wiring ────────────────────────────────────────────────────────

def test_dispatch_get_document_wraps_output_and_flags_private_session(store, monkeypatch, tmp_path):
    """
    _dispatch_tool routes "get_document" the same way as the other retrieval
    tools: output wrapped in BEGIN/END RETRIEVED DATA markers, and a private
    hit under the local provider flips the session's private flag.

    Input:  a private document, ollama provider, a fresh (non-private) session
    Expected output: wrapped text; session.private becomes True
    """
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    source = "file:///private-doc.pdf"
    add_texts(content="Confidential lab notebook entry about the pilot experiment.",
              doc_type="note", visibility="private", source=source,
              extra_metadata={"title": "Lab notebook"}, store=store)

    session = new_session("ollama")
    assert session.private is False

    result = _dispatch_tool(
        "get_document", {"source": source}, tmp_path, "ollama", provider_obj=None, session=session,
    )
    assert result.startswith("=== BEGIN RETRIEVED DATA")
    assert result.rstrip().endswith("=== END RETRIEVED DATA ===")
    assert "Confidential lab notebook entry" in result
    assert session.private is True


# ── Draft tools: the model's entire write surface ─────────────────────────────

@pytest.fixture
def drafts(tmp_path, monkeypatch):
    """An isolated drafts sandbox for the chat-tool layer."""
    from jarvis.core.config import Config
    from jarvis.drafts import workspace

    cfg = Config(drafts_dir=tmp_path / "drafts", vault_path=tmp_path / "vault")
    monkeypatch.setattr("jarvis.drafts.workspace.get_config", lambda: cfg)
    workspace._proposals.clear()
    return cfg


def test_create_draft_tool_writes_into_the_sandbox_only(drafts, tmp_path):
    """
    The agent CAN write — that is the point of drafts — but only here. Nothing
    lands in the vault.
    """
    from jarvis.chat.chat import _create_draft
    from jarvis.drafts import list_drafts

    result = _create_draft({"title": "My CV", "filename": "cv.md", "content": "# CV\n"})

    assert "Created draft" in result
    drafts_on_disk = list_drafts()
    assert [d["title"] for d in drafts_on_disk] == ["My CV"]
    # The vault was never created, let alone written to.
    assert not (tmp_path / "vault").exists()


def test_create_draft_inherits_a_private_session(drafts):
    """A draft built while private content is in play is private itself."""
    from types import SimpleNamespace

    from jarvis.chat.chat import _create_draft
    from jarvis.drafts import list_drafts

    session = SimpleNamespace(id="s1", private=True)
    _create_draft({"title": "Notes", "filename": "n.md", "content": "x\n"}, session)

    assert list_drafts()[0]["visibility"] == "private"


def test_read_draft_hard_stops_a_private_draft_under_a_cloud_provider(drafts):
    from jarvis.chat.chat import _read_draft
    from jarvis.core.errors import PrivacyError
    from jarvis.drafts import create_draft

    draft = create_draft("Private", "p.md", "sensitive\n", visibility="private")

    text, saw_private = _read_draft({"draft_id": draft["id"]}, "ollama")
    assert "sensitive" in text and saw_private is True

    with pytest.raises(PrivacyError):
        _read_draft({"draft_id": draft["id"]}, "anthropic")


def test_propose_draft_edit_writes_nothing_and_defers(drafts):
    """
    Same shape as remove_document: the tool builds the change and hands it to a
    human. A review channel that defers (the webapp) gets a "wait" message, and
    the model is told not to claim the edit happened.
    """
    from jarvis.chat.chat import _propose_draft_edit
    from jarvis.drafts import create_draft, read_draft

    draft = create_draft("Doc", "doc.md", "one\ntwo\n")

    result = _propose_draft_edit(
        {"draft_id": draft["id"], "file": "doc.md", "new_text": "one\nTWO\n",
         "rationale": "capitalise"},
        request_edit_review=lambda proposal: None,
    )

    assert read_draft(draft["id"])["text"] == "one\ntwo\n"  # untouched
    assert "Proposal only" in result
    assert "do not say the file" in result.lower()


def test_propose_draft_edit_reports_what_the_human_accepted(drafts):
    """Only the review channel's answer writes anything."""
    from jarvis.chat.chat import _propose_draft_edit
    from jarvis.drafts import apply_hunks, create_draft, read_draft

    draft = create_draft("Doc", "doc.md", "one\ntwo\n")

    def accept_everything(proposal):
        apply_hunks(proposal["token"])
        return "The user accepted every change."

    result = _propose_draft_edit(
        {"draft_id": draft["id"], "file": "doc.md", "new_text": "one\nTWO\n"},
        request_edit_review=accept_everything,
    )

    assert result == "The user accepted every change."
    assert read_draft(draft["id"])["text"] == "one\nTWO\n"


def test_an_identical_rewrite_produces_no_proposal(drafts):
    from jarvis.chat.chat import _propose_draft_edit
    from jarvis.drafts import create_draft

    draft = create_draft("Doc", "doc.md", "same\n")
    result = _propose_draft_edit(
        {"draft_id": draft["id"], "file": "doc.md", "new_text": "same\n"},
        request_edit_review=lambda p: pytest.fail("nothing should be reviewed"),
    )
    assert "identical" in result


def test_draft_tools_refuse_paths_outside_the_sandbox(drafts):
    """The containment policy reaches the tool layer, as an error not a crash."""
    from jarvis.chat.chat import _create_draft

    result = _create_draft({"title": "Escape", "filename": "../escape.md", "content": "x"})
    assert result.startswith("[create_draft error:")


def test_no_chat_tool_can_write_outside_the_drafts_root(drafts, tmp_path, monkeypatch):
    """
    The W1 guarantee, enforced the way the no-file-deletion one is: spy on the
    write primitives and assert every path a tool dispatch writes to lands
    inside the drafts sandbox.
    """
    import os as _os
    from pathlib import Path as _Path

    from jarvis.chat.chat import _create_draft

    written: list[_Path] = []
    real_replace = _os.replace

    def spy_replace(src, dst, *args, **kwargs):
        written.append(_Path(dst).resolve())
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(_os, "replace", spy_replace)

    _create_draft({"title": "Doc", "filename": "doc.md", "content": "text\n"})

    assert written
    root = (tmp_path / "drafts").resolve()
    for path in written:
        assert root in path.parents, f"{path} is outside the drafts sandbox"
