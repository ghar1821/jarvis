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


def test_confirm_action_unknown_token_409s_and_leaves_dict_untouched():
    """A token that was never issued (or already popped) 409s and touches nothing else."""
    from starlette.testclient import TestClient

    import jarvis.webapp.app as appmod

    appmod._session["pending_actions"] = {
        "abc123": {
            "session_id": "some-session",
            "action": {"ids": [], "title": "t", "doc_type": "paper", "source": "s"},
        },
    }
    client = TestClient(appmod.app, base_url="http://127.0.0.1")
    response = client.post("/confirm-action", json={"confirmed": True, "token": "WRONG"})
    assert response.status_code == 409
    # the real pending token is untouched — only the unknown one was rejected
    assert "abc123" in appmod._session["pending_actions"]


def test_confirm_action_matching_token_executes(monkeypatch):
    """The matching token pops just that entry and executes the removal."""
    from starlette.testclient import TestClient

    import jarvis.webapp.app as appmod

    monkeypatch.setattr(appmod, "execute_remove", lambda action, store: f"Removed {action['title']}")
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: None)
    appmod._session["pending_actions"] = {
        "abc123": {
            "session_id": "some-session",
            "action": {"ids": [], "title": "t", "doc_type": "paper", "source": "s"},
        },
    }
    client = TestClient(appmod.app, base_url="http://127.0.0.1")
    response = client.post("/confirm-action", json={"confirmed": True, "token": "abc123"})
    assert response.status_code == 200
    assert "abc123" not in appmod._session["pending_actions"]


def test_two_pending_tokens_independently_confirmable(monkeypatch):
    """
    Stacked dialogs (e.g. a bulk removal proposing several documents in one
    turn) must each be resolvable on their own — confirming one must not
    supersede or clear the other. No session check applies here: token
    possession is the capability, regardless of which session either dialog
    belongs to.
    """
    from starlette.testclient import TestClient

    import jarvis.webapp.app as appmod

    monkeypatch.setattr(appmod, "execute_remove", lambda action, store: f"Removed {action['title']}")
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: None)
    appmod._session["pending_actions"] = {
        "token-a": {
            "session_id": "session-a",
            "action": {"ids": [], "title": "Paper A", "doc_type": "paper", "source": "a"},
        },
        "token-b": {
            "session_id": "session-b",
            "action": {"ids": [], "title": "Paper B", "doc_type": "paper", "source": "b"},
        },
    }
    client = TestClient(appmod.app, base_url="http://127.0.0.1")

    response_a = client.post("/confirm-action", json={"confirmed": True, "token": "token-a"})
    assert response_a.status_code == 200
    assert response_a.json()["result"] == "Removed Paper A"
    assert "token-b" in appmod._session["pending_actions"]  # untouched by resolving token-a

    response_b = client.post("/confirm-action", json={"confirmed": True, "token": "token-b"})
    assert response_b.status_code == 200
    assert response_b.json()["result"] == "Removed Paper B"
    assert appmod._session["pending_actions"] == {}


def test_cancel_pops_only_its_own_token():
    """Cancelling one dialog leaves any other pending dialog fully intact."""
    from starlette.testclient import TestClient

    import jarvis.webapp.app as appmod

    appmod._session["pending_actions"] = {
        "token-a": {
            "session_id": "session-a",
            "action": {"ids": [], "title": "Paper A", "doc_type": "paper", "source": "a"},
        },
        "token-b": {
            "session_id": "session-b",
            "action": {"ids": [], "title": "Paper B", "doc_type": "paper", "source": "b"},
        },
    }
    client = TestClient(appmod.app, base_url="http://127.0.0.1")

    response = client.post("/confirm-action", json={"confirmed": False, "token": "token-a"})
    assert response.status_code == 200
    assert response.json()["result"] == "Cancelled — nothing was removed."
    assert "token-a" not in appmod._session["pending_actions"]
    assert "token-b" in appmod._session["pending_actions"]


def test_new_chat_leaves_other_sessions_dialogs_pending(monkeypatch):
    """
    /sessions/new only swaps in a fresh session for the browser to look at —
    it owns no pending_actions tokens of its own, and with true parallel
    sessions a dialog belonging to any other session (including the outgoing
    one) must keep working rather than being abandoned.
    """
    from starlette.testclient import TestClient

    import jarvis.webapp.app as appmod

    monkeypatch.setattr(appmod, "execute_remove", lambda action, store: f"Removed {action['title']}")
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: None)
    appmod._session["pending_actions"] = {
        "stale-token": {"session_id": "some-other-session", "action": {"ids": [], "title": "t"}},
    }
    client = TestClient(appmod.app, base_url="http://127.0.0.1")
    response = client.post("/sessions/new")
    assert response.status_code == 200
    assert "stale-token" in appmod._session["pending_actions"]

    # Unlike the old single-session model, this token still confirms normally.
    confirm = client.post("/confirm-action", json={"confirmed": True, "token": "stale-token"})
    assert confirm.status_code == 200
    assert confirm.json()["result"] == "Removed t"


def test_resume_clears_only_the_resumed_sessions_pending_actions(monkeypatch):
    """
    Resuming session S abandons only S's own dialogs left over from before
    the swap — a dialog belonging to a different session T must survive
    untouched, since T may itself be mid-turn right now.
    """
    from starlette.testclient import TestClient

    import jarvis.webapp.app as appmod
    from jarvis.chat.sessions import new_session

    session_s = new_session(appmod.cfg.provider)
    session_s.display.append({"role": "user", "content": "hi"})

    # Stub out the load and the resume-safety check so this test exercises
    # only the endpoint's pending_actions bookkeeping, not real session I/O.
    monkeypatch.setattr(appmod, "load_session", lambda session_id: session_s)
    monkeypatch.setattr(appmod, "check_resume", lambda *a, **k: None)
    appmod._session["pending_actions"] = {
        "s-token": {"session_id": session_s.id, "action": {"ids": [], "title": "s"}},
        "t-token": {"session_id": "session-t", "action": {"ids": [], "title": "t"}},
    }
    client = TestClient(appmod.app, base_url="http://127.0.0.1")
    response = client.post(f"/sessions/{session_s.id}/resume")
    assert response.status_code == 200

    assert "s-token" not in appmod._session["pending_actions"]
    assert "t-token" in appmod._session["pending_actions"]

    confirm_s = client.post("/confirm-action", json={"confirmed": True, "token": "s-token"})
    assert confirm_s.status_code == 409


def test_turn_start_clears_only_that_sessions_tokens(monkeypatch):
    """
    Starting a new /chat turn on session A clears only A's own stale
    dialogs — a dialog belonging to session B (which might itself be
    mid-turn concurrently) must survive untouched.
    """
    from starlette.testclient import TestClient

    import jarvis.webapp.app as appmod
    from jarvis.chat.sessions import new_session

    class _StubProvider:
        def agentic_turn(self, messages, tools, dispatch_fn, system):
            return "ok"

    session_a = new_session(appmod.cfg.provider)
    appmod._session["session"] = session_a
    appmod._session["running"] = {}
    appmod._session["provider"] = _StubProvider()
    monkeypatch.setattr(appmod, "maybe_compact", lambda *a, **k: False)
    monkeypatch.setattr(appmod, "save_session", lambda *a, **k: None)
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: "the-store")
    appmod._session["pending_actions"] = {
        "a-token": {"session_id": session_a.id, "action": {"ids": [], "title": "a"}},
        "b-token": {"session_id": "session-b", "action": {"ids": [], "title": "b"}},
    }
    client = TestClient(appmod.app, base_url="http://127.0.0.1")
    response = client.post("/chat", json={"message": "hi", "session_id": session_a.id})
    assert response.status_code == 200

    assert "a-token" not in appmod._session["pending_actions"]
    assert "b-token" in appmod._session["pending_actions"]

    appmod._session["running"] = {}  # leave the shared state clean


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
