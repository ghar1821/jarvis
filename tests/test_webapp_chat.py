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
- /chat/stop: the forceful stop. It ends the SSE stream immediately with a
  'stopped' event, cancels the worker's token, waits for the worker to
  unwind, and leaves the session with no trace of the stopped turn — so the
  very next message can be sent straight away
- the compaction status event, so a long pre-turn summarisation isn't a
  silent wait

These exercise the real FastAPI app via TestClient with a fake provider
standing in for the LLM (a real agentic_turn needs a live API), and with
save_session/maybe_compact/get_store stubbed so no real session files or
ChromaDB calls happen. _session is a module-level dict shared across the
whole test process (same pattern as test_security.py), so every test sets
the fields it depends on rather than assuming a clean slate.
"""

import json
import logging
import queue
import threading
import time

import pytest
from starlette.testclient import TestClient

import jarvis.chat.chat as chat_module
import jarvis.webapp.app as appmod
from jarvis.chat.sessions import new_session
from jarvis.core.cancel import CancelToken
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
    """
    Stands in for a real ChatProvider — agentic_turn just runs whatever
    behavior a test supplies.

    The cancel token the webapp passes in is kept on the instance rather than
    handed to the behavior, so the existing four-argument behaviors stay as
    they are. A test that needs to act like a real provider (checking the
    token where a streamed response would) reads provider.cancel.
    """

    def __init__(self, behavior):
        self.behavior = behavior
        self.cancel = None

    def agentic_turn(self, messages, tools, dispatch_fn, system, cancel=None):
        self.cancel = cancel
        return self.behavior(messages, tools, dispatch_fn, system)


def _fake_running_turn(session):
    """
    A RunningTurn standing in for an in-flight turn, for tests that only need
    the registry to look busy without actually starting a thread.
    """
    return appmod.RunningTurn(
        session=session, cancel=CancelToken(), queue=queue.Queue(), thread=None
    )


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
    appmod._session["running"] = {wired_session.id: _fake_running_turn(wired_session)}

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
    assert appmod._session["session"] is appmod._session["running"][wired_session.id].session

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
    # B already popped, A still running
    assert list(appmod._session["running"]) == [session_a.id]
    assert appmod._session["running"][session_a.id].session is session_a

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


# ── forceful stop ────────────────────────────────────────────────────────────


def test_stop_ends_the_stream_and_leaves_no_trace(wired_session, monkeypatch):
    """
    The whole point of the stop, end to end: the SSE stream terminates with a
    'stopped' event (not a reply), the session is left exactly as it was
    before the turn, nothing is indexed, and the session is free for the next
    message immediately.
    """
    save_calls = []
    monkeypatch.setattr(
        appmod, "save_session",
        lambda session, store=None: save_calls.append(store),
    )

    turn_started = threading.Event()
    provider = FakeProvider(None)

    def blocking_behavior(messages, tools, dispatch_fn, system):
        turn_started.set()
        # Stand in for a streaming provider: wait, then check the token at the
        # point a real one would (between streamed events).
        for _ in range(100):
            provider.cancel.check()
            time.sleep(0.02)
        pytest.fail("the turn was never cancelled")

    provider.behavior = blocking_behavior
    appmod._session["provider"] = provider

    client = TestClient(appmod.app, base_url="http://127.0.0.1")
    stream_result = {}

    def post_chat():
        stream_result["response"] = client.post(
            "/chat", json={"message": "a runaway question", "session_id": wired_session.id}
        )

    post_thread = threading.Thread(target=post_chat)
    post_thread.start()
    assert turn_started.wait(timeout=10), "the turn never started"

    stop_response = client.post("/chat/stop", json={"session_id": wired_session.id})
    assert stop_response.status_code == 200
    body = stop_response.json()
    assert body["stopped"] is True
    # The endpoint rolled the turn back itself rather than waiting for the
    # worker, so the session is already clean when this response is sent.
    assert body["rolled_back"] is True

    post_thread.join(timeout=10)
    assert not post_thread.is_alive(), "the SSE stream never terminated"

    events = _parse_sse(stream_result["response"].text)
    assert [e["type"] for e in events] == ["stopped"]

    # No trace: the question is gone from both lists and from turn_starts.
    assert wired_session.display == []
    assert wired_session.messages == []
    assert wired_session.turn_starts == []
    # And nothing was indexed — every save on the stop path is store-free, so
    # the abandoned turn never becomes a searchable chat exchange.
    assert save_calls and all(store is None for store in save_calls)
    assert appmod._session["running"] == {}


def test_stop_frees_the_session_for_the_next_message(wired_session, monkeypatch):
    """
    "The user can just send another request through": a stopped session must
    accept a new /chat straight away, with no 409 and no leftover history from
    the abandoned turn.
    """
    monkeypatch.setattr(appmod, "save_session", lambda *a, **k: None)

    turn_started = threading.Event()
    provider = FakeProvider(None)

    def blocking_behavior(messages, tools, dispatch_fn, system):
        turn_started.set()
        for _ in range(100):
            provider.cancel.check()
            time.sleep(0.02)
        pytest.fail("the turn was never cancelled")

    provider.behavior = blocking_behavior
    appmod._session["provider"] = provider

    client = TestClient(appmod.app, base_url="http://127.0.0.1")
    post_thread = threading.Thread(
        target=lambda: client.post(
            "/chat", json={"message": "first question", "session_id": wired_session.id}
        )
    )
    post_thread.start()
    assert turn_started.wait(timeout=10)

    client.post("/chat/stop", json={"session_id": wired_session.id})
    post_thread.join(timeout=10)

    appmod._session["provider"] = FakeProvider(
        lambda messages, tools, dispatch_fn, system: "the second reply"
    )
    second = client.post(
        "/chat", json={"message": "second question", "session_id": wired_session.id}
    )
    assert second.status_code == 200

    # Only the second exchange survives; the stopped one left nothing behind.
    assert [t["content"] for t in wired_session.display] == [
        "second question", "the second reply"
    ]


def test_stop_landing_as_the_reply_commits_leaves_the_reply_standing(wired_session, monkeypatch):
    """
    A stop and a finished reply can land in the same instant. If the reply won,
    it has already been committed and saved, and rolling the turn back would
    delete an answer the user is looking at — so the stop stands down instead.
    """
    monkeypatch.setattr(appmod, "save_session", lambda *a, **k: None)

    # A session whose turn has just been committed: the assistant reply is
    # already in display, but the turn is still registered as running.
    wired_session.turn_starts.append(0)
    wired_session.messages.append({"role": "user", "content": "a question"})
    wired_session.display.append({"role": "user", "content": "a question"})
    wired_session.display.append({"role": "assistant", "content": "the answer", "tool_calls": []})
    appmod._session["running"] = {wired_session.id: _fake_running_turn(wired_session)}

    client = TestClient(appmod.app, base_url="http://127.0.0.1")
    response = client.post("/chat/stop", json={"session_id": wired_session.id})

    assert response.status_code == 200
    assert response.json() == {"stopped": True, "rolled_back": False}
    # The committed exchange is untouched.
    assert [t["content"] for t in wired_session.display] == ["a question", "the answer"]
    assert appmod._session["running"] == {}


def test_worker_stands_down_when_a_stop_beats_its_commit(wired_session, monkeypatch):
    """
    The other side of that race: if the stop got there first, the worker must
    not commit the reply it just produced — the turn was abandoned, and the
    session may already be onto a newer one.
    """
    save_calls = []
    monkeypatch.setattr(
        appmod, "save_session", lambda session, store=None: save_calls.append(store)
    )

    provider = FakeProvider(None)

    def stopped_before_committing(messages, tools, dispatch_fn, system):
        # Stands in for a stop arriving in the window between the provider
        # returning and run_agent committing the result.
        provider.cancel.stop()
        return "a reply nobody asked for any more"

    provider.behavior = stopped_before_committing
    appmod._session["provider"] = provider

    client = TestClient(appmod.app, base_url="http://127.0.0.1")
    response = client.post("/chat", json={"message": "hi", "session_id": wired_session.id})

    # No reply event, and the reply never reached the session.
    events = _parse_sse(response.text)
    assert [e["type"] for e in events] == []
    assert all(t["content"] != "a reply nobody asked for any more" for t in wired_session.display)
    # Only the early save ran; the indexed save on the success path did not.
    assert save_calls == [None]
    assert appmod._session["running"] == {}


def test_stop_on_an_idle_session_404s(wired_session):
    """
    Stopping a session with nothing in flight is a 404, not a silent success —
    the browser can then tell the difference between "cancelled" and "there
    was nothing to cancel".
    """
    client = TestClient(appmod.app, base_url="http://127.0.0.1")
    response = client.post("/chat/stop", json={"session_id": wired_session.id})
    assert response.status_code == 404
    assert "no reply is being generated" in response.json()["detail"]


def test_stop_clears_only_its_own_sessions_pending_dialogs(wired_session, monkeypatch):
    """
    A confirmation dialog the stopped turn put up is no longer actionable, but
    another session's pending dialogs must survive — the same per-session rule
    _clear_pending_for follows everywhere else.
    """
    monkeypatch.setattr(appmod, "save_session", lambda *a, **k: None)

    turn_started = threading.Event()
    provider = FakeProvider(None)

    def blocking_behavior(messages, tools, dispatch_fn, system):
        turn_started.set()
        for _ in range(100):
            provider.cancel.check()
            time.sleep(0.02)
        pytest.fail("the turn was never cancelled")

    provider.behavior = blocking_behavior
    appmod._session["provider"] = provider

    client = TestClient(appmod.app, base_url="http://127.0.0.1")
    post_thread = threading.Thread(
        target=lambda: client.post(
            "/chat", json={"message": "hi", "session_id": wired_session.id}
        )
    )
    post_thread.start()
    assert turn_started.wait(timeout=10)

    appmod._session["pending_actions"] = {
        "mine": {"session_id": wired_session.id, "action": {"ids": [], "title": "mine"}},
        "theirs": {"session_id": "another-session", "action": {"ids": [], "title": "theirs"}},
    }

    client.post("/chat/stop", json={"session_id": wired_session.id})
    post_thread.join(timeout=10)

    assert list(appmod._session["pending_actions"]) == ["theirs"]
    appmod._session["pending_actions"] = {}  # leave the shared state clean


def test_compaction_emits_a_status_event(wired_session, monkeypatch):
    """
    Compaction is an extra LLM call before the reply even starts. The browser
    is told when it begins and when it's over, so a long summarisation shows
    as "compacting" instead of an unexplained wait.
    """
    monkeypatch.setattr(appmod, "save_session", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "needs_compaction", lambda session, cfg: True)
    monkeypatch.setattr(appmod, "maybe_compact", lambda *a, **k: True)
    appmod._session["provider"] = FakeProvider(
        lambda messages, tools, dispatch_fn, system: "compacted then answered"
    )

    client = TestClient(appmod.app, base_url="http://127.0.0.1")
    response = client.post("/chat", json={"message": "hi", "session_id": wired_session.id})

    events = _parse_sse(response.text)
    states = [e["state"] for e in events if e["type"] == "status"]
    assert states == ["compacting", "thinking"]
    assert [e["type"] for e in events][-1] == "reply"


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
