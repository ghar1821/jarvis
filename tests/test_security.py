"""
Tests for the security-hardening layer:
- _remove_document: human-in-the-loop confirmation gate, one-shot flow
- file deletion has been removed from the codebase entirely (no code path
  left that can delete a file on disk)
- webapp: TrustedHost, session-id validation on network-facing endpoints,
  stale confirm-dialog token guard
"""

from pathlib import Path

import pytest

from jarvis.kb.store import add_texts
from jarvis.chat.chat import _remove_document, execute_remove, truncate_middle


# ── File deletion removed wholesale ─────────────────────────────────────────

def test_delete_local_file_removed_from_store_module():
    """The physical deletion capability was removed from the codebase, not just disabled."""
    import jarvis.kb.store as store_module
    assert not hasattr(store_module, "delete_local_file")


def test_no_unlink_calls_remain_in_kb_or_chat_source():
    """Structural guarantee: no `.unlink(` call survives in the modules jarvis ships."""
    import inspect
    import jarvis.kb.store
    import jarvis.kb.cli
    import jarvis.chat.chat
    for module in (jarvis.kb.store, jarvis.kb.cli, jarvis.chat.chat):
        assert ".unlink(" not in inspect.getsource(module), module.__name__


# ── remove_document: human confirmation gate ───────────────────────────────────

@pytest.fixture
def indexed_paper(store, tmp_path, monkeypatch):
    """A paper with an on-disk PDF, indexed in the test store."""
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    pdf = tmp_path / "target.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    source = pdf.as_uri()
    add_texts(content="A paper about targeted deletion.", doc_type="paper",
              visibility="public", source=source,
              extra_metadata={"title": "Target", "file_path": str(pdf)}, store=store)
    return pdf, source


def test_no_channel_refuses(indexed_paper, tmp_path):
    """With no interactive confirmation channel, the tool refuses outright."""
    pdf, source = indexed_paper
    result = _remove_document({"source": source}, tmp_path)
    assert "[Error" in result
    assert pdf.exists()


def test_decline_blocks_deletion(indexed_paper, tmp_path, store):
    """The human answering 'no' cancels everything."""
    pdf, source = indexed_paper
    result = _remove_document(
        {"source": source}, tmp_path,
        request_confirmation=lambda d, a: False,
    )
    assert "declined" in result
    assert pdf.exists()
    remaining = store._collection.get(where={"source": {"$eq": source}}, include=[])
    assert remaining["ids"]


def test_approve_executes_and_never_touches_disk(indexed_paper, tmp_path, store):
    """The human answering 'yes' removes the DB chunks; the file always survives."""
    pdf, source = indexed_paper
    result = _remove_document(
        {"source": source}, tmp_path,
        request_confirmation=lambda d, a: True,
    )
    assert "Removed" in result
    assert pdf.exists()  # the file survives regardless of approval
    remaining = store._collection.get(where={"source": {"$eq": source}}, include=[])
    assert remaining["ids"] == []


def test_deferred_webapp_channel_leaves_pending(indexed_paper, tmp_path, store):
    """
    A deferring channel (webapp dialog) returns None: the tool reports the
    pending dialog and nothing is touched until /confirm-action fires.
    """
    pdf, source = indexed_paper
    captured = {}

    def deferring(description, action):
        captured["action"] = action
        return None

    result = _remove_document({"source": source}, tmp_path, request_confirmation=deferring)
    assert "confirmation" in result.lower()
    assert "do not call remove_document again" in result.lower()
    assert pdf.exists()

    # The stored action executes correctly later (this is what /confirm-action runs).
    outcome = execute_remove(captured["action"], store)
    assert "Removed" in outcome and "No files were touched" in outcome
    assert pdf.exists()


def test_invariant_line_shown_to_human_and_in_every_preview(indexed_paper, tmp_path):
    """
    Both the returned string and the description handed to the confirmation
    channel state the "files are never touched" invariant and name the full
    local path — never a bare directory.
    """
    pdf, source = indexed_paper
    captured = {}

    def spy(description, action):
        captured["description"] = description
        return None

    result = _remove_document({"source": source}, tmp_path, request_confirmation=spy)
    for text in (result, captured["description"]):
        assert "files on disk are never touched by jarvis" in text
        assert str(pdf) in text


# ── Webapp hardening ───────────────────────────────────────────────────────────

def test_webapp_rejects_foreign_host_header():
    """
    TrustedHost blocks DNS-rebinding: a request whose Host isn't
    localhost/127.0.0.1 is refused before any endpoint runs.
    """
    from starlette.testclient import TestClient

    import jarvis.webapp.app as appmod

    client = TestClient(appmod.app, base_url="http://evil.example")
    response = client.get("/info")
    assert response.status_code == 400

    ok = TestClient(appmod.app, base_url="http://127.0.0.1").get("/sessions")
    assert ok.status_code == 200


def test_webapp_session_id_traversal_rejected():
    """Path-shaped session ids are rejected by validation, not resolved."""
    from starlette.testclient import TestClient

    import jarvis.webapp.app as appmod

    client = TestClient(appmod.app, base_url="http://127.0.0.1")
    assert client.delete("/sessions/..%2F..%2Fescape").status_code == 404
    assert client.post("/sessions/..%2Fx/resume").status_code == 404


def test_confirm_action_requires_matching_token():
    """A token that doesn't match the pending action 409s and leaves it intact."""
    from starlette.testclient import TestClient

    import jarvis.webapp.app as appmod

    appmod._session["pending_action"] = {
        "token": "abc123",
        "action": {"ids": [], "title": "t", "doc_type": "paper", "source": "s"},
    }
    client = TestClient(appmod.app, base_url="http://127.0.0.1")
    response = client.post("/confirm-action", json={"confirmed": True, "token": "WRONG"})
    assert response.status_code == 409
    assert appmod._session["pending_action"] is not None  # not cleared on mismatch


def test_confirm_action_matching_token_executes(monkeypatch):
    """The matching token clears the pending action and executes the removal."""
    from starlette.testclient import TestClient

    import jarvis.webapp.app as appmod

    monkeypatch.setattr(appmod, "execute_remove", lambda action, store: f"Removed {action['title']}")
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: None)
    appmod._session["pending_action"] = {
        "token": "abc123",
        "action": {"ids": [], "title": "t", "doc_type": "paper", "source": "s"},
    }
    client = TestClient(appmod.app, base_url="http://127.0.0.1")
    response = client.post("/confirm-action", json={"confirmed": True, "token": "abc123"})
    assert response.status_code == 200
    assert appmod._session["pending_action"] is None


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
