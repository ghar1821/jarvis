"""
Persistent chat sessions — save, resume, pin, prune, compact.

One JSON file per session in ~/.jarvis/sessions/. The file holds both the
provider wire-format `messages` (what the LLM sees) and the webapp `display`
list (what the human sees) — the two cannot be rebuilt from each other, and
compaction deliberately shrinks only `messages`.

Privacy model:
- A session is flagged private the moment any tool returns private content
  (only possible under the local provider). The flag never clears.
- Session exchanges are indexed into the knowledge base as doc_type="chat"
  with the session's visibility, so search_chat_history under a cloud
  provider only ever sees public sessions — the same search_with_privacy_check
  machinery that protects notes.
- Resuming a private session with the cloud provider is refused
  (check_resume): it would replay private history to Anthropic.

Retention: the 50 most recently updated unpinned sessions are kept; pinned
sessions are exempt and uncounted, deleted only explicitly.
"""

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from digest.errors import PrivacyError

SESSIONS_DIR = Path.home() / ".jarvis" / "sessions"
MAX_UNPINNED_SESSIONS = 50
TITLE_MAX_CHARS = 60


@dataclass
class Session:
    id: str
    title: str = ""
    created_at: str = ""
    updated_at: str = ""
    pinned: bool = False
    private: bool = False
    provider: str = "ollama"
    kb_only: bool = True
    messages: list = field(default_factory=list)   # provider wire format
    display: list = field(default_factory=list)    # human-facing render list
    turn_starts: list = field(default_factory=list)  # messages-index where each user turn began
    indexed_exchanges: int = 0                     # (user, assistant) pairs already in Chroma


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_session(provider: str, kb_only: bool = True) -> Session:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    session_id = f"{stamp}-{uuid.uuid4().hex[:6]}"
    return Session(id=session_id, created_at=_now(), updated_at=_now(),
                   provider=provider, kb_only=kb_only)


def _jsonable(message):
    """Provider clients may hand back pydantic objects — normalise for JSON."""
    return message.model_dump(exclude_none=True) if hasattr(message, "model_dump") else message


def _session_source(session_id: str) -> str:
    return f"session:{session_id}"


# ── Persistence ────────────────────────────────────────────────────────────────


def save_session(session: Session, sessions_dir: Path = SESSIONS_DIR, store=None) -> None:
    """
    Persist the session (atomic write, private file modes) and, when a store
    is supplied, index any new exchanges and prune old sessions. Empty
    sessions are never written. Call after every completed turn — crash-safe.
    """
    if not session.display:
        return
    if not session.title:
        first_user = next((t["content"] for t in session.display if t["role"] == "user"), "")
        session.title = first_user[:TITLE_MAX_CHARS]
    session.updated_at = _now()

    payload = asdict(session)
    payload["messages"] = [_jsonable(m) for m in session.messages]

    sessions_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(sessions_dir, 0o700)  # session files hold private chat content
    session_file = sessions_dir / f"{session.id}.json"
    tmp_file = session_file.with_suffix(".json.tmp")
    tmp_file.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.chmod(tmp_file, 0o600)
    os.replace(tmp_file, session_file)

    if store is not None:
        _index_new_exchanges(session, store)
        prune_sessions(sessions_dir=sessions_dir, store=store)


def load_session(session_id: str, sessions_dir: Path = SESSIONS_DIR) -> Session:
    """Load a session by id. Raises FileNotFoundError for unknown ids."""
    _require_valid_session_id(session_id)
    payload = json.loads((sessions_dir / f"{session_id}.json").read_text(encoding="utf-8"))
    return Session(**payload)


def list_sessions(sessions_dir: Path = SESSIONS_DIR) -> list[dict]:
    """
    Metadata for every stored session (no messages), pinned first, then by
    updated_at descending.
    """
    entries = []
    if not sessions_dir.is_dir():
        return entries
    for session_file in sessions_dir.glob("*.json"):
        try:
            payload = json.loads(session_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        entries.append({
            "id": payload.get("id", session_file.stem),
            "title": payload.get("title", ""),
            "created_at": payload.get("created_at", ""),
            "updated_at": payload.get("updated_at", ""),
            "pinned": bool(payload.get("pinned", False)),
            "private": bool(payload.get("private", False)),
            "provider": payload.get("provider", ""),
        })
    entries.sort(key=lambda e: (not e["pinned"], e["updated_at"]), reverse=False)
    # pinned first, newest first within each group
    entries.sort(key=lambda e: e["updated_at"], reverse=True)
    entries.sort(key=lambda e: not e["pinned"])
    return entries


def delete_session(session_id: str, sessions_dir: Path = SESSIONS_DIR, store=None) -> None:
    """Remove the session file and its indexed chat chunks."""
    _require_valid_session_id(session_id)
    session_file = sessions_dir / f"{session_id}.json"
    if session_file.exists():
        session_file.unlink()
    if store is not None:
        from digest.kb.store import delete_by_metadata

        delete_by_metadata("source", _session_source(session_id), store)


def set_pinned(session_id: str, pinned: bool, sessions_dir: Path = SESSIONS_DIR) -> None:
    """Flip a stored session's pinned flag in place."""
    session = load_session(session_id, sessions_dir)
    session.pinned = pinned
    session_file = sessions_dir / f"{session_id}.json"
    payload = asdict(session)
    tmp_file = session_file.with_suffix(".json.tmp")
    tmp_file.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.chmod(tmp_file, 0o600)
    os.replace(tmp_file, session_file)


def rename_session(session_id: str, title: str, sessions_dir: Path = SESSIONS_DIR) -> str:
    """
    Rename a stored session in place. The title is trimmed and capped at 120
    characters; an empty title is rejected. Returns the applied title so the
    caller can propagate it (e.g. to the in-memory active session and the
    indexed chat chunks). load_session validates the session id.
    """
    title = title.strip()[:120]
    if not title:
        raise ValueError("session title must not be empty")
    session = load_session(session_id, sessions_dir)
    session.title = title
    session_file = sessions_dir / f"{session_id}.json"
    payload = asdict(session)
    tmp_file = session_file.with_suffix(".json.tmp")
    tmp_file.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.chmod(tmp_file, 0o600)
    os.replace(tmp_file, session_file)
    return title


def prune_sessions(
    sessions_dir: Path = SESSIONS_DIR,
    keep: int = MAX_UNPINNED_SESSIONS,
    store=None,
) -> int:
    """
    Delete the oldest unpinned sessions beyond `keep`. Pinned sessions never
    count and are never deleted here. Returns the number removed.
    """
    unpinned = [e for e in list_sessions(sessions_dir) if not e["pinned"]]
    stale = sorted(unpinned, key=lambda e: e["updated_at"], reverse=True)[keep:]
    for entry in stale:
        delete_session(entry["id"], sessions_dir, store)
    return len(stale)


def _require_valid_session_id(session_id: str) -> None:
    """
    Session ids reach the code from the network (webapp endpoints) and are
    used to build file paths — restrict to the generated alphabet.
    """
    import re

    if not re.fullmatch(r"[0-9a-z-]{1,64}", session_id or ""):
        raise ValueError(f"invalid session id: {session_id!r}")


# ── Privacy ────────────────────────────────────────────────────────────────────


def mark_private(session: Session, store=None) -> None:
    """
    Flag the session private (never un-flipped) and re-classify anything
    already indexed: previously indexed public chunks are deleted, and the
    next save re-indexes the whole history as private. Over-restrictive for
    pre-flip exchanges — deliberately fail-closed.
    """
    if session.private:
        return
    session.private = True
    if session.indexed_exchanges and store is not None:
        from digest.kb.store import delete_by_metadata

        delete_by_metadata("source", _session_source(session.id), store)
        session.indexed_exchanges = 0


def check_resume(session: Session, current_provider: str) -> None:
    """
    Refuse resumes that would be unsafe or unworkable:
    - private session + cloud provider → PrivacyError (history would replay
      private content to Anthropic)
    - cross-provider resume → ValueError (Anthropic content blocks vs the
      local wire format are incompatible). The match is strict per provider
      name (only anthropic shares a family with itself), so a session recorded
      under the retired "llamacpp" provider refuses to resume under "ollama"
      rather than silently replaying an incompatible history.
    """
    if session.private and current_provider == "anthropic":
        raise PrivacyError(
            f"Session {session.id} contains private content and cannot be resumed "
            "with a cloud provider. Restart with the local provider "
            "(webapp --provider ollama)."
        )

    def family(provider: str) -> str:
        return "anthropic" if provider == "anthropic" else provider

    if family(session.provider) != family(current_provider):
        raise ValueError(
            f"Session {session.id} was recorded with the {session.provider!r} provider "
            f"and cannot be replayed under {current_provider!r} — the stored message "
            "formats are incompatible. Restart with the matching provider."
        )


# ── Chat-history indexing ──────────────────────────────────────────────────────


def _display_exchanges(display: list) -> list[tuple[str, str]]:
    """Pair user turns with the assistant reply that follows each."""
    exchanges = []
    pending_user: str | None = None
    for turn in display:
        if turn["role"] == "user":
            pending_user = turn["content"]
        elif turn["role"] == "assistant" and pending_user is not None:
            exchanges.append((pending_user, turn["content"]))
            pending_user = None
    return exchanges


def _index_new_exchanges(session: Session, store) -> None:
    """
    Index exchanges beyond indexed_exchanges into the KB as doc_type="chat".
    Built from the display list — raw tool results never get indexed (they
    would duplicate document content that is already in the store).
    """
    from digest.kb.store import add_texts

    exchanges = _display_exchanges(session.display)
    visibility = "private" if session.private else "public"
    for i, (user_text, assistant_text) in enumerate(exchanges):
        if i < session.indexed_exchanges:
            continue
        add_texts(
            content=f"User: {user_text}\n\nAssistant: {assistant_text}",
            doc_type="chat",
            visibility=visibility,
            source=_session_source(session.id),
            extra_metadata={
                "title": session.title,
                "session_id": session.id,
                "exchange_index": i,
                "session_date": session.updated_at,
            },
            store=store,
        )
    session.indexed_exchanges = len(exchanges)


# ── Compaction ─────────────────────────────────────────────────────────────────


def estimate_tokens(messages: list) -> int:
    """Crude but adequate: serialised JSON length / 4."""
    return len(json.dumps([_jsonable(m) for m in messages], default=str)) // 4


_COMPACT_PROMPT = (
    "Summarise the following conversation between a user and an assistant so "
    "the assistant can continue it later. Preserve: what the user is working "
    "on, decisions made, documents discussed (with sources), and any open "
    "questions. Be dense and factual.\n\nConversation:\n{transcript}"
)


def maybe_compact(session: Session, provider_obj, cfg) -> bool:
    """
    When the session's LLM context exceeds cfg.compact_after_tokens, replace
    everything but the last cfg.compact_keep_exchanges turns with a summary
    generated by the session's OWN provider (a private session is by
    definition local, so private history never goes to a cloud model for
    summarisation).

    Only `messages` shrinks — the display list keeps the full history for the
    UI, and chat-history indexing is display-driven so search is unaffected.
    The cut always lands on a recorded turn boundary (turn_starts), keeping
    tool_use/tool_result message structure intact.

    Returns True when compaction happened.
    """
    keep = max(1, cfg.compact_keep_exchanges)
    if len(session.turn_starts) <= keep:
        return False
    if estimate_tokens(session.messages) < cfg.compact_after_tokens:
        return False

    cut = session.turn_starts[-keep]
    old_messages = session.messages[:cut]

    transcript = json.dumps([_jsonable(m) for m in old_messages], default=str)[:60000]
    summary = provider_obj.complete(
        [{"role": "user", "content": _COMPACT_PROMPT.format(transcript=transcript)}],
        max_tokens=1024,
    )

    summary_pair = [
        {"role": "user", "content": f"[Summary of the conversation so far]\n{summary}"},
        {"role": "assistant", "content": "Understood — continuing from that summary."},
    ]
    session.messages[:] = summary_pair + session.messages[cut:]
    shift = cut - len(summary_pair)
    session.turn_starts = [start - shift for start in session.turn_starts[-keep:]]
    return True
