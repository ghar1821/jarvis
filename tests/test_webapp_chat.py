"""
Tests for the webapp's /chat turn lifecycle (jarvis/webapp/app.py):

- save -> turn -> save ordering, with the user's message on disk before the
  LLM call even starts (the early save added to fix "message disappears on
  session switch mid-turn")
- the busy guard: a second /chat addressed at a session already mid-turn
  409s, deleting that session 409s, and /sessions reports which session ids
  are busy
- true parallel sessions: a second, different session can run its own turn
  to completion while the first is still blocked, and each session's own
  display holds only its own exchange
- /chat is session-addressed: an unknown id 404s, and a message always lands
  on the session named in the request rather than whatever happens to be
  the shared "active" session at that instant
- error handling added for chat.log visibility: an LLMError still replies,
  logs, and saves; an uncaught exception ("crash path") does the same with an
  internal-error reply, and the SSE stream still terminates cleanly
- resuming a session that's still mid-turn installs the live registry
  object (the same one the background thread is mutating), not a stale
  disk copy — this is what makes /history correct without any reinstall
  step in run_agent's finally block

These exercise the real FastAPI app via TestClient with a fake provider
standing in for the LLM (a real agentic_turn needs a live API), and with
save_session/maybe_compact/get_store stubbed so no real session files or
ChromaDB calls happen. _session is a module-level dict shared across the
whole test process (same pattern as test_security.py), so every test sets
the fields it depends on rather than assuming a clean slate.
"""

import json
import logging
import threading

import pytest
from starlette.testclient import TestClient

import jarvis.chat.chat as chat_module
import jarvis.webapp.app as appmod
from jarvis.chat.sessions import new_session
from jarvis.core.errors import LLMError


@pytest.fixture
def isolated_log():
    """Detach chat.py's real FileHandler so tests never touch chat.log (copied from test_chat_logging.py)."""
    handlers = list(chat_module.log.handlers)
    for handler in handlers:
        chat_module.log.removeHandler(handler)
    yield
    for handler in handlers:
        chat_module.log.addHandler(handler)


class FakeProvider:
    """Stands in for a real ChatProvider — agentic_turn just runs whatever behavior a test supplies."""

    def __init__(self, behavior):
        self.behavior = behavior

    def agentic_turn(self, messages, tools, dispatch_fn, system):
        return self.behavior(messages, tools, dispatch_fn, system)


def _parse_sse(text: str) -> list[dict]:
    """Turn a fully-buffered SSE response body into the list of event dicts it carried."""
    events = []
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
    return events


@pytest.fixture
def wired_session(monkeypatch):
    """
    A real Session installed as the webapp's active session, with maybe_compact
    and get_store stubbed so run_agent's turn is free of real KB/compaction
    side effects. Tests still set appmod._session["provider"] to their own
    FakeProvider and monkeypatch appmod.save_session with their own recorder.
    """
    session = new_session(appmod.cfg.provider)
    appmod._session["session"] = session
    appmod._session["running"] = {}
    appmod._session["pending_actions"] = {}
    monkeypatch.setattr(appmod, "maybe_compact", lambda *a, **k: False)
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: "the-store")
    return session


# ── save -> turn -> save ordering ────────────────────────────────────────────


def test_early_save_persists_user_message_before_the_llm_call(wired_session, monkeypatch):
    """
    The first save_session call must already show the user's turn on disk —
    this is what keeps the question from disappearing if the browser switches
    sessions (or the process dies) before the reply lands.
    """
    save_calls = []

    def fake_save_session(session, store=None):
        save_calls.append({"display": [dict(turn) for turn in session.display], "store": store})

    monkeypatch.setattr(appmod, "save_session", fake_save_session)
    appmod._session["provider"] = FakeProvider(lambda messages, tools, dispatch_fn, system: "hello back")

    client = TestClient(appmod.app, base_url="http://127.0.0.1")
    response = client.post("/chat", json={"message": "hi there", "session_id": wired_session.id})
    assert response.status_code == 200

    assert len(save_calls) == 2
    # First save: only the user's turn exists yet, and it wasn't indexed
    # (no store= — that side effect is deliberately deferred to the final save).
    assert [t["role"] for t in save_calls[0]["display"]] == ["user"]
    assert save_calls[0]["display"][0]["content"] == "hi there"
    assert save_calls[0]["store"] is None
    # Second save: the completed turn, this time indexed via the store.
    assert [t["role"] for t in save_calls[1]["display"]] == ["user", "assistant"]
    assert save_calls[1]["store"] == "the-store"

    assert appmod._session["running"] == {}


# ── busy guard ────────────────────────────────────────────────────────────────


def test_busy_guard_blocks_second_chat_and_session_delete(wired_session):
    """
    While a turn is in flight for a session, a second /chat addressed at
    THAT session 409s, deleting that session 409s, and /sessions surfaces
    its id in the busy list.
    """
    appmod._session["running"] = {wired_session.id: wired_session}  # simulate an in-flight turn

    client = TestClient(appmod.app, base_url="http://127.0.0.1")

    chat_response = client.post(
        "/chat", json={"message": "another question", "session_id": wired_session.id}
    )
    assert chat_response.status_code == 409
    assert "still being generated" in chat_response.json()["detail"]

    delete_response = client.delete(f"/sessions/{wired_session.id}")
    assert delete_response.status_code == 409

    sessions_response = client.get("/sessions")
    assert sessions_response.json()["busy"] == [wired_session.id]

    appmod._session["running"] = {}  # leave the shared state clean


# ── crash path ──────────────────────────────────────────────────────────────


def test_uncaught_exception_still_replies_and_terminates_stream(wired_session, monkeypatch, caplog, isolated_log):
    """
    A bug in the tool loop (anything not an LLMError) must not hang the SSE
    stream forever — it should log the full traceback, persist the error
    turn (same as the LLMError path), and still deliver a reply event so
    the browser's "Working..." placeholder clears.
    """
    save_calls = []
    monkeypatch.setattr(appmod, "save_session", lambda session, store=None: save_calls.append(store))

    def behavior(messages, tools, dispatch_fn, system):
        raise RuntimeError("kaboom")

    appmod._session["provider"] = FakeProvider(behavior)

    client = TestClient(appmod.app, base_url="http://127.0.0.1")
    with caplog.at_level(logging.ERROR, logger="vault-chat"):
        response = client.post(
            "/chat", json={"message": "trigger a crash", "session_id": wired_session.id}
        )

    assert response.status_code == 200  # the stream itself completes normally
    events = _parse_sse(response.text)
    reply_events = [e for e in events if e["type"] == "reply"]
    assert len(reply_events) == 1
    assert "Internal error" in reply_events[0]["content"]
    assert "kaboom" in reply_events[0]["content"]

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert error_records, "expected an ERROR record logged for the crash"
    assert error_records[0].exc_info is not None

    # The error turn was persisted: the early save plus the save in the
    # broad except branch (both store-free).
    assert len(save_calls) == 2
    assert appmod._session["running"] == {}


# ── LLMError path ─────────────────────────────────────────────────────────────


def test_llm_error_replies_logs_and_saves(wired_session, monkeypatch, caplog, isolated_log):
    """An LLMError is an expected provider failure: log it, reply with a warning, still save."""
    save_calls = []
    monkeypatch.setattr(appmod, "save_session", lambda session, store=None: save_calls.append(store))

    def behavior(messages, tools, dispatch_fn, system):
        raise LLMError("rate limited")

    appmod._session["provider"] = FakeProvider(behavior)

    client = TestClient(appmod.app, base_url="http://127.0.0.1")
    with caplog.at_level(logging.ERROR, logger="vault-chat"):
        response = client.post("/chat", json={"message": "hi", "session_id": wired_session.id})

    events = _parse_sse(response.text)
    reply_events = [e for e in events if e["type"] == "reply"]
    assert len(reply_events) == 1
    assert reply_events[0]["content"] == "⚠️ rate limited"

    assert any(r.levelno == logging.ERROR for r in caplog.records)
    assert save_calls  # the early save plus the error-path save both fired
    assert appmod._session["running"] == {}


# ── resume of a mid-turn session ─────────────────────────────────────────────


def test_resume_of_busy_session_installs_live_object(wired_session, monkeypatch):
    """
    Resuming a session that's still mid-turn must install the SAME live
    object the background thread is mutating — not a fresh-from-disk copy —
    so /history reflects the reply the instant it lands, with no reinstall
    step needed anywhere. This replaces the old single-session model's
    reinstall hack in run_agent's finally block, which is gone now that
    resume reads straight out of the "running" registry.
    """
    monkeypatch.setattr(appmod, "save_session", lambda *a, **k: None)

    turn_started = threading.Event()
    release_turn = threading.Event()

    def blocking_behavior(messages, tools, dispatch_fn, system):
        turn_started.set()
        assert release_turn.wait(timeout=10), "test never released the blocked turn"
        return "the finished reply"

    appmod._session["provider"] = FakeProvider(blocking_behavior)

    client = TestClient(appmod.app, base_url="http://127.0.0.1")

    # TestClient.post drains the whole SSE stream before returning, so the
    # request has to run on its own thread while this one plays the browser
    # switching away and resuming.
    stream_result = {}

    def post_chat():
        stream_result["response"] = client.post(
            "/chat", json={"message": "slow question", "session_id": wired_session.id}
        )

    post_thread = threading.Thread(target=post_chat)
    post_thread.start()
    assert turn_started.wait(timeout=10), "the turn never started"

    # Simulate switching away and back to the busy session mid-turn: the
    # resume route must find it in the "running" registry and install that
    # exact object, reporting busy=True.
    assert wired_session.id in appmod._session["running"]
    resume_response = client.post(f"/sessions/{wired_session.id}/resume")
    assert resume_response.status_code == 200
    body = resume_response.json()
    assert body["busy"] is True
    assert appmod._session["session"] is wired_session
    assert appmod._session["session"] is appmod._session["running"][wired_session.id]

    # Let the turn finish and the stream drain.
    release_turn.set()
    post_thread.join(timeout=10)
    assert not post_thread.is_alive()
    assert stream_result["response"].status_code == 200

    assert appmod._session["running"] == {}
    history = client.get("/history").json()
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"] == "the finished reply"


# ── true parallel sessions ───────────────────────────────────────────────────


def test_two_parallel_turns(wired_session, monkeypatch):
    """
    Sending to session B while session A is still blocked mid-turn must not
    409 or wait — the two turns run concurrently, and each session's own
    display ends up holding only its own exchange (the bug-3/4 regression
    test: no cross-contamination between concurrently running sessions).
    """
    monkeypatch.setattr(appmod, "save_session", lambda *a, **k: None)

    session_a = wired_session  # already installed as the active session
    session_b = new_session(appmod.cfg.provider)
    monkeypatch.setattr(
        appmod, "load_session",
        lambda session_id: session_b if session_id == session_b.id else pytest.fail(session_id),
    )

    a_started = threading.Event()
    release_a = threading.Event()

    def behavior_a(messages, tools, dispatch_fn, system):
        a_started.set()
        assert release_a.wait(timeout=10), "session A's turn was never released"
        return "reply for A"

    def behavior_b(messages, tools, dispatch_fn, system):
        return "reply for B"

    client = TestClient(appmod.app, base_url="http://127.0.0.1")

    appmod._session["provider"] = FakeProvider(behavior_a)
    stream_result = {}

    def post_a():
        stream_result["a"] = client.post(
            "/chat", json={"message": "question for A", "session_id": session_a.id}
        )

    thread_a = threading.Thread(target=post_a)
    thread_a.start()
    assert a_started.wait(timeout=10), "session A's turn never started"

    # A is now blocked mid-turn (its user turn is on session_a.display, no
    # assistant reply yet). Sending to B — a different, non-active session —
    # must succeed immediately rather than 409ing on A's busy state.
    appmod._session["provider"] = FakeProvider(behavior_b)
    response_b = client.post(
        "/chat", json={"message": "question for B", "session_id": session_b.id}
    )
    assert response_b.status_code == 200

    assert [t["role"] for t in session_a.display] == ["user"]  # still blocked, no reply yet
    assert [t["role"] for t in session_b.display] == ["user", "assistant"]
    assert session_b.display[-1]["content"] == "reply for B"
    assert appmod._session["running"] == {session_a.id: session_a}  # B already popped, A still running

    release_a.set()
    thread_a.join(timeout=10)
    assert not thread_a.is_alive()
    assert stream_result["a"].status_code == 200

    assert appmod._session["running"] == {}
    assert [t["role"] for t in session_a.display] == ["user", "assistant"]
    assert session_a.display[-1]["content"] == "reply for A"
    # Neither session's display picked up the other's exchange.
    assert session_b.display == [
        {"role": "user", "content": "question for B"},
        {"role": "assistant", "content": "reply for B", "tool_calls": []},
    ]


# ── session addressing ───────────────────────────────────────────────────────


def test_chat_wrong_session_id_404s(wired_session):
    """An id that isn't the active session and has no file on disk 404s rather than being silently created or misapplied."""
    client = TestClient(appmod.app, base_url="http://127.0.0.1")
    response = client.post(
        "/chat", json={"message": "hi", "session_id": "20260101-000000-abcdef"}
    )
    assert response.status_code == 404


def test_chat_delivers_to_addressed_session_not_active(wired_session, monkeypatch):
    """
    A message addressed to a session that is NOT the currently active one
    must land on that addressed session, never on whatever _session["session"]
    happens to be at that instant — the direct pin for bug 4 (cross-session
    contamination via the old mutable-global design).
    """
    monkeypatch.setattr(appmod, "save_session", lambda *a, **k: None)
    other_session = new_session(appmod.cfg.provider)
    monkeypatch.setattr(
        appmod, "load_session",
        lambda session_id: other_session if session_id == other_session.id else pytest.fail(session_id),
    )
    appmod._session["provider"] = FakeProvider(
        lambda messages, tools, dispatch_fn, system: "reply for other"
    )

    client = TestClient(appmod.app, base_url="http://127.0.0.1")
    response = client.post(
        "/chat", json={"message": "hello", "session_id": other_session.id}
    )
    assert response.status_code == 200

    assert [t["role"] for t in other_session.display] == ["user", "assistant"]
    assert other_session.display[-1]["content"] == "reply for other"
    assert wired_session.display == []  # the active session was never touched


def test_validation_error_is_logged_and_returns_readable_detail(caplog, isolated_log):
    """
    A request FastAPI rejects at validation time (e.g. a stale browser tab
    posting the pre-parallel-sessions /chat shape without session_id) never
    reaches a route body — the exception handler must log it to the
    vault-chat logger so it is diagnosable from chat.log, and return the
    standard 422 detail list.
    """
    client = TestClient(appmod.app, base_url="http://127.0.0.1")
    with caplog.at_level(logging.ERROR, logger="vault-chat"):
        response = client.post("/chat", json={"message": "no session id here"})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail[0]["loc"] == ["body", "session_id"]

    validation_logs = [r for r in caplog.records if "request validation failed" in r.message]
    assert len(validation_logs) == 1
    assert "/chat" in validation_logs[0].message
    assert "session_id" in validation_logs[0].message
