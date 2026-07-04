"""
Tests for the security-hardening layer:
- delete_local_file: the papers-only hard rule for on-disk deletion
- _remove_document: human-in-the-loop confirmation gate
- webapp: TrustedHost, session-id validation on network-facing endpoints
"""

from pathlib import Path

import pytest

from digest.kb.store import add_texts, delete_local_file
from vault_chat.chat import _remove_document, execute_remove, truncate_middle


# ── delete_local_file: papers-only rule ────────────────────────────────────────

def test_delete_local_file_removes_paper_pdf(tmp_path):
    """A paper PDF is deletable."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    deleted, msg = delete_local_file(pdf, "paper")
    assert deleted is True
    assert not pdf.exists()


def test_delete_local_file_never_deletes_notes(tmp_path):
    """
    Note files — vault .md or note-type PDFs — are refused, categorically.
    This is the 'not even by the user' rule.
    """
    note_pdf = tmp_path / "notebook.pdf"
    note_pdf.write_bytes(b"%PDF-1.4")
    deleted, msg = delete_local_file(note_pdf, "note")
    assert deleted is False
    assert note_pdf.exists()
    assert "never deleted" in msg

    note_md = tmp_path / "thoughts.md"
    note_md.write_text("# Thoughts")
    deleted, msg = delete_local_file(note_md, "note")
    assert deleted is False
    assert note_md.exists()


def test_delete_local_file_refuses_non_pdf_papers(tmp_path):
    """Even for papers, only .pdf files can be unlinked."""
    stray = tmp_path / "paper.tex"
    stray.write_text("\\documentclass{article}")
    deleted, msg = delete_local_file(stray, "paper")
    assert deleted is False
    assert stray.exists()


def test_delete_local_file_missing_file(tmp_path):
    """Missing/None files report cleanly instead of raising."""
    deleted, msg = delete_local_file(tmp_path / "gone.pdf", "paper")
    assert deleted is False
    deleted, msg = delete_local_file(None, "paper")
    assert deleted is False


# ── remove_document: human confirmation gate ───────────────────────────────────

@pytest.fixture
def indexed_paper(store, tmp_path, monkeypatch):
    """A paper with an on-disk PDF, indexed in the test store."""
    monkeypatch.setattr("digest.kb.store.get_store", lambda: store)
    pdf = tmp_path / "target.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    source = pdf.as_uri()
    add_texts(content="A paper about targeted deletion.", doc_type="paper",
              visibility="public", source=source,
              extra_metadata={"title": "Target", "file_path": str(pdf)}, store=store)
    return pdf, source


def test_unconfirmed_call_only_previews(indexed_paper, tmp_path):
    """The first call never deletes anything, with or without a confirm channel."""
    pdf, source = indexed_paper
    result = _remove_document({"source": source, "delete_file": True}, tmp_path)
    assert "Found 1 chunk(s) to remove" in result
    assert pdf.exists()


def test_confirmed_call_without_channel_refuses(indexed_paper, tmp_path):
    """
    confirmed=true with no interactive confirmation channel must refuse —
    the model's own flag can never execute a deletion.
    """
    pdf, source = indexed_paper
    result = _remove_document({"source": source, "confirmed": True, "delete_file": True}, tmp_path)
    assert "[Error" in result
    assert pdf.exists()


def test_human_decline_blocks_deletion(indexed_paper, tmp_path, store):
    """The human answering 'no' cancels everything."""
    pdf, source = indexed_paper
    result = _remove_document(
        {"source": source, "confirmed": True, "delete_file": True},
        tmp_path,
        request_confirmation=lambda description, action: False,
    )
    assert "declined" in result
    assert pdf.exists()
    remaining = store._collection.get(where={"source": {"$eq": source}}, include=[])
    assert remaining["ids"]


def test_human_approval_executes_deletion(indexed_paper, tmp_path, store):
    """The human answering 'yes' removes the chunks and the paper PDF."""
    pdf, source = indexed_paper
    result = _remove_document(
        {"source": source, "confirmed": True, "delete_file": True},
        tmp_path,
        request_confirmation=lambda description, action: True,
    )
    assert "Removed" in result
    assert not pdf.exists()
    remaining = store._collection.get(where={"source": {"$eq": source}}, include=[])
    assert remaining["ids"] == []


def test_keep_file_removal_leaves_pdf_and_says_so(indexed_paper, tmp_path, store):
    """
    A DB-only removal (delete_file=false) deletes the chunks but leaves the PDF
    on disk, and the preview + confirmation dialog both name the full path and
    say the file is KEPT.
    """
    pdf, source = indexed_paper

    # Preview must show the full path and the KEPT wording, never a bare directory.
    preview = _remove_document({"source": source, "delete_file": False}, tmp_path)
    assert str(pdf) in preview
    assert "KEPT" in preview

    captured = {}

    def spy_channel(description, action):
        captured["description"] = description
        return True  # approve

    result = _remove_document(
        {"source": source, "confirmed": True, "delete_file": False},
        tmp_path,
        request_confirmation=spy_channel,
    )
    assert "Removed" in result
    # The dialog description carried the full path and KEPT wording.
    assert str(pdf) in captured["description"]
    assert "KEPT" in captured["description"]
    # File survives; chunks are gone.
    assert pdf.exists()
    remaining = store._collection.get(where={"source": {"$eq": source}}, include=[])
    assert remaining["ids"] == []


def test_deferred_confirmation_leaves_everything_intact(indexed_paper, tmp_path, store):
    """
    A deferring channel (webapp dialog) returns None: the tool reports the
    pending dialog and nothing is touched until /confirm-action fires.
    """
    pdf, source = indexed_paper
    captured = {}

    def deferring_channel(description, action):
        captured["action"] = action
        return None

    result = _remove_document(
        {"source": source, "confirmed": True, "delete_file": True},
        tmp_path,
        request_confirmation=deferring_channel,
    )
    assert "confirmation dialog" in result
    assert pdf.exists()

    # The stored action executes correctly later (this is what /confirm-action runs).
    outcome = execute_remove(captured["action"], store)
    assert "Removed" in outcome
    assert not pdf.exists()


# ── Webapp hardening ───────────────────────────────────────────────────────────

def test_webapp_rejects_foreign_host_header():
    """
    TrustedHost blocks DNS-rebinding: a request whose Host isn't
    localhost/127.0.0.1 is refused before any endpoint runs.
    """
    from starlette.testclient import TestClient

    import webapp.app as appmod

    client = TestClient(appmod.app, base_url="http://evil.example")
    response = client.get("/info")
    assert response.status_code == 400

    ok = TestClient(appmod.app, base_url="http://127.0.0.1").get("/sessions")
    assert ok.status_code == 200


def test_webapp_session_id_traversal_rejected():
    """Path-shaped session ids are rejected by validation, not resolved."""
    from starlette.testclient import TestClient

    import webapp.app as appmod

    client = TestClient(appmod.app, base_url="http://127.0.0.1")
    assert client.delete("/sessions/..%2F..%2Fescape").status_code == 404
    assert client.post("/sessions/..%2Fx/resume").status_code == 404


# ── Tool-arg display truncation ─────────────────────────────────────────────────

def test_truncate_middle_preserves_head_and_tail():
    """
    Short values pass through untouched; long values keep head + tail with a
    single ellipsis, so a file:/// URI's filename stays visible at the tail
    (the old repr()[:40] cut it off).
    """
    assert truncate_middle("short value") == "short value"

    uri = "file:///Users/putri.g/Documents/papers/some-really-long-paper-name.pdf"
    out = truncate_middle(repr(uri))
    assert "…" in out
    assert out.count("…") == 1
    assert out.endswith("paper-name.pdf'")  # filename preserved at the tail
    assert out.startswith("'file:///Users")  # scheme preserved at the head