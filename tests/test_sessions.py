"""
Tests for jarvis/chat/sessions.py — persistent chat sessions.

Filesystem behaviour uses tmp_path; anything touching the vector store uses
the real per-test Chroma collection from conftest.py. The compaction test
uses a fake provider (LLM summarisation is a billed/remote boundary).
"""

import json

import pytest

from jarvis.core.config import Config
from jarvis.core.errors import PrivacyError
from jarvis.core.transcript import message_text, text_block, user_message
from jarvis.kb.store import add_texts, search_with_privacy_check


def assistant_message(text: str) -> dict:
    """The assistant counterpart of transcript.user_message, for fixtures."""
    return {"role": "assistant", "content": [text_block(text)]}
from jarvis.chat.sessions import (
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
    record_usage,
    rename_session,
    save_session,
    session_cost_usd,
    set_pinned,
)


def _session_with_turns(n_turns: int = 1, provider: str = "ollama") -> Session:
    session = new_session(provider)
    for i in range(n_turns):
        session.turn_starts.append(len(session.messages))
        session.messages.append(user_message(f"question {i}"))
        session.messages.append(assistant_message(f"answer {i}"))
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


def test_check_resume_refuses_a_private_session_on_any_cloud_provider():
    """
    Once a session has seen private content the whole transcript is private,
    so it may only run on a local model. The rule is local-vs-cloud, not one
    vendor's name — a new cloud provider is covered without a code change.
    """
    private_local = _session_with_turns(1, provider="ollama")
    private_local.private = True

    with pytest.raises(PrivacyError):
        check_resume(private_local, "anthropic")
    with pytest.raises(PrivacyError):
        check_resume(private_local, "openrouter:openai/gpt-5")

    check_resume(private_local, "ollama")  # no raise


def test_check_resume_allows_cross_provider_resume():
    """
    v2 sessions store the neutral transcript, so any provider can read history
    any other provider wrote. The old cross-provider refusal existed only
    because the stored format was vendor-specific — it is gone with the format.
    """
    ollama_session = _session_with_turns(1, provider="ollama")
    check_resume(ollama_session, "openrouter:openai/gpt-5")  # no raise

    cloud_session = _session_with_turns(1, provider="anthropic")
    check_resume(cloud_session, "ollama")  # no raise
    check_resume(cloud_session, "openrouter:openai/gpt-5")  # no raise


def test_chat_history_search_respects_session_privacy(tmp_path, store):
    """
    Indexed exchanges from a private session are invisible to the cloud
    provider's chat search but visible locally.
    """
    session = _session_with_turns(0)
    session.turn_starts.append(0)
    session.messages += [
        user_message("Tell me about zebrafish neurogenesis"),
        assistant_message("Zebrafish neurogenesis involves..."),
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
    from jarvis.kb.store import update_chat_title

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
    assert message_text(session.messages[0]).startswith("[Summary of the conversation so far]")
    assert session.messages[1]["role"] == "assistant"
    assert len(session.messages) == 2 + 3 * 2
    # The first kept turn is turn 7 ("question 7") and starts right after the pair
    assert session.messages[2] == user_message("question 7")
    assert session.turn_starts == [2, 4, 6]
    # UI history is untouched
    assert session.display == display_before


def test_estimate_tokens_scales_with_content():
    """More content → higher estimate (sanity check on the heuristic)."""
    small = [{"role": "user", "content": "hi"}]
    large = [{"role": "user", "content": "hi " * 1000}]
    assert estimate_tokens(large) > estimate_tokens(small) > 0

# ── v1 → v2 migration and per-model cost ───────────────────────────────────────

def test_v1_anthropic_session_migrates_to_the_neutral_transcript(tmp_path):
    """
    A session written before the neutral transcript existed must still load —
    its Anthropic content blocks convert on read. The file is not rewritten
    here; the next completed turn saves it as v2 through the normal path.
    """
    payload = {
        "id": "20260101-000000-abcdef",
        "provider": "anthropic",
        "display": [{"role": "user", "content": "hi"}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "read a.md"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tu_1", "name": "read_file", "input": {"path": "a.md"}}
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "body"}],
            },
        ],
        "turn_starts": [0],
    }
    (tmp_path / "20260101-000000-abcdef.json").write_text(json.dumps(payload))

    session = load_session("20260101-000000-abcdef", sessions_dir=tmp_path)

    assert session.format_version == 2
    assert message_text(session.messages[0]) == "read a.md"
    call = session.messages[1]["content"][0]
    assert call["type"] == "tool_call" and call["name"] == "read_file"
    assert session.messages[2]["content"][0]["tool_call_id"] == "tu_1"
    # v1 turn_starts index into the old wire list, whose message count the
    # conversion does not preserve, so they are dropped rather than left wrong.
    assert session.turn_starts == []


def test_v1_migration_keeps_thinking_but_never_replays_it(tmp_path):
    """
    A v1 file never recorded which model wrote it, so a thinking block is
    preserved in the transcript yet tagged with an unknown model — and an
    opaque block only replays on an exact provider+model match, so it can
    never be sent back to a model that might not have produced it.
    """
    payload = {
        "id": "20260101-000000-aaaaaa",
        "provider": "anthropic",
        "display": [{"role": "user", "content": "hi"}],
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "s"},
                    {"type": "text", "text": "answer"},
                ],
            }
        ],
    }
    (tmp_path / "20260101-000000-aaaaaa.json").write_text(json.dumps(payload))

    session = load_session("20260101-000000-aaaaaa", sessions_dir=tmp_path)
    opaque = session.messages[0]["content"][0]
    assert opaque["type"] == "provider_opaque"
    assert opaque["model"] == ""

    from jarvis.core.transcript import to_anthropic
    wire = to_anthropic(session.messages, model="claude-sonnet-4-6")
    assert [b["type"] for b in wire[0]["content"]] == ["text"]


def test_a_v2_session_is_loaded_untouched(tmp_path):
    """Already-migrated sessions must not be converted a second time."""
    session = _session_with_turns(2)
    save_session(session, sessions_dir=tmp_path)
    reloaded = load_session(session.id, sessions_dir=tmp_path)
    assert reloaded.messages == session.messages
    assert reloaded.turn_starts == session.turn_starts


def test_new_session_splits_a_provider_model_spec():
    """
    The privacy rules key on the provider alone while switching keys on both,
    so the spec is stored as two fields and recombined on demand.
    """
    session = new_session("openrouter:anthropic/claude-sonnet-4.6")
    assert session.provider == "openrouter"
    assert session.model == "anthropic/claude-sonnet-4.6"
    assert session.model_spec == "openrouter:anthropic/claude-sonnet-4.6"

    # A spec naming no model resolves the provider's configured default, so
    # model_spec is always concrete — "ollama" alone would name no model, and
    # the header, picker, and cost key all read it.
    local = new_session("ollama")
    assert local.provider == "ollama"
    assert local.model == Config().ollama_model
    assert local.model_spec == f"ollama:{Config().ollama_model}"


def test_record_usage_accumulates_per_model_and_ignores_none():
    """
    Cost is tracked per model so a session that switched mid-conversation
    shows the split. A provider reporting nothing records nothing — a session
    with no entries shows no cost at all rather than a fabricated zero.
    """
    session = new_session("openrouter:openai/gpt-5")

    record_usage(session, "openrouter:openai/gpt-5", {"usd": 0.002, "requests": 2})
    record_usage(session, "openrouter:openai/gpt-5", {"usd": 0.001, "requests": 1})
    record_usage(session, "ollama", None)

    assert session.cost == {"openrouter:openai/gpt-5": {"usd": 0.003, "requests": 3}}
    assert session_cost_usd(session) == 0.003

    local_only = new_session("ollama")
    record_usage(local_only, "ollama", None)
    assert local_only.cost == {}
    assert session_cost_usd(local_only) == 0.0


def test_cost_survives_a_save_load_roundtrip(tmp_path):
    session = _session_with_turns(1, provider="openrouter")
    record_usage(session, "openrouter:openai/gpt-5", {"usd": 0.01, "requests": 1})
    save_session(session, sessions_dir=tmp_path)

    reloaded = load_session(session.id, sessions_dir=tmp_path)
    assert reloaded.cost == {"openrouter:openai/gpt-5": {"usd": 0.01, "requests": 1}}
