"""
Tests for vault_chat/sessions.py — persistent chat sessions.

Filesystem behaviour uses tmp_path; anything touching the vector store uses
the real per-test Chroma collection from conftest.py. The compaction test
uses a fake provider (LLM summarisation is a billed/remote boundary).
"""

import json

import pytest

from digest.config import Config
from digest.errors import PrivacyError
from digest.kb.store import add_texts, search_with_privacy_check
from vault_chat.sessions import (
    Session,
    check_resume,
    delete_session,
    estimate_tokens,
    list_sessions,
    load_session,
    mark_private,
    maybe_compact,
    new_session,
    prune_sessions,
    rename_session,
    save_session,
    set_pinned,
)


def _session_with_turns(n_turns: int = 1, provider: str = "ollama") -> Session:
    session = new_session(provider)
    for i in range(n_turns):
        session.turn_starts.append(len(session.messages))
        session.messages.append({"role": "user", "content": f"question {i}"})
        session.messages.append({"role": "assistant", "content": f"answer {i}"})
        session.display.append({"role": "user", "content": f"question {i}"})
        session.display.append({"role": "assistant", "content": f"answer {i}"})
    return session


# ── Persistence ────────────────────────────────────────────────────────────────

def test_save_load_roundtrip(tmp_path):
    """
    A saved session loads back with identical fields, a title derived from
    the first user message, and secure file modes.
    """
    session = _session_with_turns(2)
    save_session(session, sessions_dir=tmp_path)

    loaded = load_session(session.id, sessions_dir=tmp_path)
    assert loaded.messages == session.messages
    assert loaded.display == session.display
    assert loaded.turn_starts == session.turn_starts
    assert loaded.title == "question 0"
    assert loaded.provider == "ollama"

    session_file = tmp_path / f"{session.id}.json"
    assert (session_file.stat().st_mode & 0o777) == 0o600


def test_save_normalises_pydantic_messages(tmp_path):
    """
    Provider clients append pydantic objects to messages; save must
    model_dump() them so json.dumps never crashes.
    """

    class FakePydanticMessage:
        def model_dump(self, exclude_none=False):
            return {"role": "assistant", "content": "from pydantic"}

    session = _session_with_turns(1)
    session.messages.append(FakePydanticMessage())
    save_session(session, sessions_dir=tmp_path)

    loaded = load_session(session.id, sessions_dir=tmp_path)
    assert loaded.messages[-1] == {"role": "assistant", "content": "from pydantic"}


def test_empty_session_never_written(tmp_path):
    """A session with no display turns leaves no file behind."""
    session = new_session("ollama")
    save_session(session, sessions_dir=tmp_path)
    assert list(tmp_path.glob("*.json")) == []


def test_load_rejects_malicious_session_ids(tmp_path):
    """
    Session ids come from the network and become file paths — traversal and
    absolute-path shapes must be rejected before any filesystem access.
    """
    for bad_id in ("../escape", "/etc/passwd", "a/b", "..", "UPPER", ""):
        with pytest.raises(ValueError):
            load_session(bad_id, sessions_dir=tmp_path)


# ── Pin / prune ────────────────────────────────────────────────────────────────

def test_prune_keeps_newest_unpinned_and_all_pinned(tmp_path):
    """
    With keep=3: the 3 most recently updated unpinned sessions survive,
    older ones are deleted, and pinned sessions are exempt and uncounted.
    """
    ids = []
    for i in range(6):
        session = _session_with_turns(1)
        session.updated_at = f"2026-07-0{i + 1}T00:00:00+00:00"
        # Bypass save_session's updated_at stamping to control order.
        import dataclasses, os
        payload = dataclasses.asdict(session)
        (tmp_path / f"{session.id}.json").write_text(json.dumps(payload))
        ids.append(session.id)

    set_pinned(ids[0], True, sessions_dir=tmp_path)  # the OLDEST is pinned

    removed = prune_sessions(sessions_dir=tmp_path, keep=3)
    assert removed == 2  # 5 unpinned, keep 3 → 2 deleted

    remaining = {e["id"] for e in list_sessions(sessions_dir=tmp_path)}
    assert ids[0] in remaining          # pinned survives despite being oldest
    assert set(ids[3:]) <= remaining    # 3 newest unpinned survive
    assert ids[1] not in remaining and ids[2] not in remaining


def test_list_sessions_orders_pinned_first_then_newest(tmp_path):
    """The sidebar order: pinned block first, then updated_at descending."""
    a = _session_with_turns(1)
    save_session(a, sessions_dir=tmp_path)
    b = _session_with_turns(1)
    save_session(b, sessions_dir=tmp_path)
    set_pinned(a.id, True, sessions_dir=tmp_path)

    entries = list_sessions(sessions_dir=tmp_path)
    assert entries[0]["id"] == a.id and entries[0]["pinned"] is True
    assert entries[1]["id"] == b.id


# ── Privacy ────────────────────────────────────────────────────────────────────

def test_mark_private_flags_and_reindexes(tmp_path, store):
    """
    mark_private flips the flag, purges previously indexed public chunks,
    and the next save re-indexes the full history as private.
    """
    session = _session_with_turns(2)
    save_session(session, sessions_dir=tmp_path, store=store)
    assert session.indexed_exchanges == 2

    public_chunks = store._collection.get(
        where={"source": {"$eq": f"session:{session.id}"}}, include=["metadatas"]
    )
    assert all(m["visibility"] == "public" for m in public_chunks["metadatas"])

    mark_private(session, store)
    assert session.private is True
    assert session.indexed_exchanges == 0

    save_session(session, sessions_dir=tmp_path, store=store)
    reindexed = store._collection.get(
        where={"source": {"$eq": f"session:{session.id}"}}, include=["metadatas"]
    )
    assert reindexed["ids"]
    assert all(m["visibility"] == "private" for m in reindexed["metadatas"])


def test_check_resume_matrix():
    """
    private+anthropic → PrivacyError; cross-provider → ValueError;
    private+local and matching-provider public resumes pass.
    """
    private_local = _session_with_turns(1, provider="ollama")
    private_local.private = True
    with pytest.raises(PrivacyError):
        check_resume(private_local, "anthropic")
    check_resume(private_local, "ollama")  # no raise

    cloud_session = _session_with_turns(1, provider="anthropic")
    with pytest.raises(ValueError):
        check_resume(cloud_session, "ollama")
    check_resume(cloud_session, "anthropic")  # no raise


def test_check_resume_refuses_retired_llamacpp_session():
    """
    Provider matching is strict per name (only anthropic shares a family with
    itself), so a session recorded under the retired 'llamacpp' provider refuses
    to resume under 'ollama' rather than silently replaying its history.
    """
    legacy = _session_with_turns(1, provider="llamacpp")
    with pytest.raises(ValueError, match="llamacpp"):
        check_resume(legacy, "ollama")


def test_chat_history_search_respects_session_privacy(tmp_path, store):
    """
    Indexed exchanges from a private session are invisible to the cloud
    provider's chat search but visible locally.
    """
    session = _session_with_turns(0)
    session.turn_starts.append(0)
    session.messages += [
        {"role": "user", "content": "Tell me about zebrafish neurogenesis"},
        {"role": "assistant", "content": "Zebrafish neurogenesis involves..."},
    ]
    session.display += [
        {"role": "user", "content": "Tell me about zebrafish neurogenesis"},
        {"role": "assistant", "content": "Zebrafish neurogenesis involves..."},
    ]
    session.private = True
    save_session(session, sessions_dir=tmp_path, store=store)

    cloud_results, has_private = search_with_privacy_check(
        "zebrafish neurogenesis", provider="anthropic", doc_type="chat", store=store
    )
    assert cloud_results == []
    assert has_private is True

    local_results, _ = search_with_privacy_check(
        "zebrafish neurogenesis", provider="ollama", doc_type="chat", store=store
    )
    assert local_results


def test_delete_session_removes_file_and_chunks(tmp_path, store):
    """Deleting a session removes both its JSON file and its chat chunks."""
    session = _session_with_turns(1)
    save_session(session, sessions_dir=tmp_path, store=store)
    assert (tmp_path / f"{session.id}.json").exists()

    delete_session(session.id, sessions_dir=tmp_path, store=store)
    assert not (tmp_path / f"{session.id}.json").exists()
    chunks = store._collection.get(
        where={"source": {"$eq": f"session:{session.id}"}}, include=[]
    )
    assert chunks["ids"] == []


# ── Rename ───────────────────────────────────────────────────────────────────

def test_rename_session_roundtrip(tmp_path):
    """rename_session persists a new title, trimmed, and returns it."""
    session = _session_with_turns(1)
    save_session(session, sessions_dir=tmp_path)

    applied = rename_session(session.id, "  New descriptive title  ", sessions_dir=tmp_path)
    assert applied == "New descriptive title"
    assert load_session(session.id, sessions_dir=tmp_path).title == "New descriptive title"


def test_rename_session_rejects_empty_and_whitespace(tmp_path):
    """An empty or whitespace-only title is rejected."""
    session = _session_with_turns(1)
    save_session(session, sessions_dir=tmp_path)
    for bad in ("", "   ", "\t\n"):
        with pytest.raises(ValueError, match="must not be empty"):
            rename_session(session.id, bad, sessions_dir=tmp_path)


def test_rename_session_caps_length(tmp_path):
    """Titles are capped at 120 characters."""
    session = _session_with_turns(1)
    save_session(session, sessions_dir=tmp_path)
    applied = rename_session(session.id, "x" * 300, sessions_dir=tmp_path)
    assert len(applied) == 120


def test_rename_session_unknown_id(tmp_path):
    """Renaming a nonexistent session raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        rename_session("20260101-000000-abcdef", "whatever", sessions_dir=tmp_path)


def test_update_chat_title_updates_indexed_chunks(tmp_path, store):
    """
    Renaming propagates to the session's indexed chat chunks so past-conversation
    search shows the new name.
    """
    from digest.kb.store import update_chat_title

    session = _session_with_turns(0)
    session.turn_starts.append(0)
    session.messages += [
        {"role": "user", "content": "Discuss photosynthesis pathways"},
        {"role": "assistant", "content": "Photosynthesis has light and dark reactions..."},
    ]
    session.display += [
        {"role": "user", "content": "Discuss photosynthesis pathways"},
        {"role": "assistant", "content": "Photosynthesis has light and dark reactions..."},
    ]
    save_session(session, sessions_dir=tmp_path, store=store)

    updated = update_chat_title(session.id, "Photosynthesis chat", store=store)
    assert updated >= 1
    chunks = store._collection.get(
        where={"session_id": {"$eq": session.id}}, include=["metadatas"]
    )
    assert all(m["title"] == "Photosynthesis chat" for m in chunks["metadatas"])


# ── Compaction ─────────────────────────────────────────────────────────────────

class _CannedSummaryProvider:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, max_tokens=2048, context_length=None):
        self.calls += 1
        return "Canned summary of earlier conversation."


def test_maybe_compact_noop_below_threshold():
    """A short session is left untouched and the provider is never called."""
    session = _session_with_turns(8)
    provider = _CannedSummaryProvider()
    cfg = Config(compact_after_tokens=10**9, compact_keep_exchanges=2)
    assert maybe_compact(session, provider, cfg) is False
    assert provider.calls == 0


def test_maybe_compact_replaces_old_turns_with_summary():
    """
    Above the threshold, all but the last K turns collapse into a summary
    pair; the cut lands exactly on a turn boundary; the display list is
    untouched; turn_starts is rebuilt consistently.
    """
    session = _session_with_turns(10)
    display_before = list(session.display)
    provider = _CannedSummaryProvider()
    cfg = Config(compact_after_tokens=1, compact_keep_exchanges=3)

    assert maybe_compact(session, provider, cfg) is True
    assert provider.calls == 1

    # Summary pair + 3 kept turns × 2 messages each
    assert session.messages[0]["content"].startswith("[Summary of the conversation so far]")
    assert session.messages[1]["role"] == "assistant"
    assert len(session.messages) == 2 + 3 * 2
    # The first kept turn is turn 7 ("question 7") and starts right after the pair
    assert session.messages[2] == {"role": "user", "content": "question 7"}
    assert session.turn_starts == [2, 4, 6]
    # UI history is untouched
    assert session.display == display_before


def test_estimate_tokens_scales_with_content():
    """More content → higher estimate (sanity check on the heuristic)."""
    small = [{"role": "user", "content": "hi"}]
    large = [{"role": "user", "content": "hi " * 1000}]
    assert estimate_tokens(large) > estimate_tokens(small) > 0