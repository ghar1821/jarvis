"""
Jarvis web UI — FastAPI + SSE + vanilla JS.

Single-user local application. The active conversation is a persistent
Session (jarvis/chat/sessions.py): saved to ~/.jarvis/sessions/ after every
turn, resumable from the sidebar, pruned to the 50 most recent unpinned
sessions. Refreshing the browser restores the active conversation.

Launch:
    uv run webapp
"""

import asyncio
import json
import queue
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from jarvis.core.config import get_config, reset_config, set_config_value
from jarvis.core.errors import LLMError, PrivacyError
from jarvis.chat.chat import (
    READ_SKILL_TOOL,
    TOOLS,
    USE_OWN_KNOWLEDGE_TOOL,
    _auto_refresh_vault,
    _dispatch_tool,
    _format_tool_args,
    build_system_prompt,
    execute_remove,
    log,
)
from jarvis.chat.sessions import (
    Session,
    check_resume,
    delete_session,
    list_sessions,
    load_session,
    maybe_compact,
    new_session,
    rename_session,
    save_session,
    set_pinned,
)
from jarvis.chat.skills import list_skills

_ROOT = Path(__file__).parent
cfg = get_config()
_vault = cfg.vault_path

# Single-user state — shared across browser tabs (intended for local use only).
# session      : the active (currently viewed) persistent Session — /history,
#                /config, and a plain /chat with a matching id all read/write
#                this one. It is NOT a lock: several sessions can be mid-turn
#                at once (see "running" below), and switching the active
#                session never interrupts a turn running against another one.
# kb_only      : default for brand-new sessions; when True (default), LLM
#                answers only from KB tools; when False, it may fall back to
#                training knowledge after searching the KB. /config also
#                updates the active session's own kb_only (see /config).
# running      : {session_id: live Session object} — every session currently
#                mid-turn in its own run_agent background thread. A second
#                /chat addressed at an id already in here 409s; resuming that
#                id installs this same live object (not a stale disk copy) so
#                /history reflects turns as they land; sessions_delete refuses
#                to delete an id that's in here.
_session: dict = {
    "session": None,
    "provider": None,
    "kb_only": True,
    "response_style": cfg.response_style,
    # Deletions awaiting the user's Confirm/Cancel click: {token: {"session_id",
    # "action"}}. The model can only request a removal; execution happens
    # through /confirm-action, entirely outside the LLM tool loop. Keyed by
    # token rather than a single slot so several stacked dialogs (e.g. the
    # model proposes removing more than one document in a turn) are each
    # independently confirmable — confirming or cancelling one doesn't
    # invalidate the others. The session_id lets a new turn clear only its own
    # session's dialogs (_clear_pending_for) without touching another
    # session's still-pending ones. A dialog left unclicked when its entry is
    # cleared (new turn on that session, or that session's resume) 409s if its
    # token is posted later. /confirm-action itself does not check session_id
    # — token possession is the capability, and popping is what makes a click
    # one-shot regardless of which session is currently active.
    "pending_actions": {},
    "running": {},
}


def _clear_pending_for(session_id: str) -> None:
    """
    Drop only `session_id`'s pending confirmation tokens, leaving every other
    session's dialogs (including ones still mid-turn) untouched. Event-loop-
    side callers only (/chat, resume, delete) — run_agent's background thread
    only ever inserts tokens, never clears them.
    """
    _session["pending_actions"] = {
        token: entry
        for token, entry in _session["pending_actions"].items()
        if entry["session_id"] != session_id
    }


def _build_tools(kb_only: bool) -> list[dict]:
    tools = list(TOOLS)
    if list_skills(cfg.skills_dir):
        tools.append(READ_SKILL_TOOL)
    if not kb_only:
        tools.append(USE_OWN_KNOWLEDGE_TOOL)
    return tools


@asynccontextmanager
async def lifespan(app: FastAPI):
    from jarvis.core.llm import make_provider

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _auto_refresh_vault, _vault)
    _session["provider"] = make_provider(cfg.provider)
    _session["session"] = new_session(cfg.provider, kb_only=True)
    yield


app = FastAPI(lifespan=lifespan)
# Blocks DNS-rebinding: a malicious page pointing an attacker domain at
# 127.0.0.1 gets refused because the Host header won't match.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
app.mount("/static", StaticFiles(directory=_ROOT / "static"), name="static")


@app.exception_handler(RequestValidationError)
async def log_validation_errors(request: Request, exc: RequestValidationError) -> JSONResponse:
    # A 422 is rejected before any route body runs, so without this it leaves
    # no trace in chat.log at all — a stale browser tab sending an outdated
    # request shape (e.g. /chat without session_id after an upgrade) becomes
    # undiagnosable. Log it, then return FastAPI's standard 422 shape.
    errors = jsonable_encoder(exc.errors())
    log.error(
        "request validation failed: %s %s — %s", request.method, request.url.path, errors
    )
    return JSONResponse(status_code=422, content={"detail": errors})


# Request bodies — defined before any route so FastAPI can resolve each
# parameter's type at route-registration time. A model referenced by name
# before it exists (even via a quoted forward reference) makes FastAPI treat
# the parameter as a query param instead of a JSON body.
class ChatRequest(BaseModel):
    message: str
    session_id: str


class ConfigRequest(BaseModel):
    kb_only: bool


class PinRequest(BaseModel):
    pinned: bool


class RenameRequest(BaseModel):
    title: str


class SettingsRequest(BaseModel):
    response_style: str


class ConfirmActionRequest(BaseModel):
    confirmed: bool
    token: str


class PaperMetaRequest(BaseModel):
    source: str
    title: str | None = None
    authors: str | None = None
    doi: str | None = None


class PaperRemoveRequest(BaseModel):
    source: str


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((_ROOT / "index.html").read_text())


@app.get("/info")
async def info() -> dict:
    from jarvis.core.llm import active_model

    # Provider label shown in the browser header
    label = (
        f"Anthropic · {active_model(cfg)}"
        if cfg.provider == "anthropic"
        else f"Ollama · {active_model(cfg)}"
    )
    return {
        "provider": label,
        "provider_kind": cfg.provider,
        "vault": str(_vault),
    }


@app.get("/history")
async def history() -> list:
    # Returns the display list so the browser can re-render the conversation
    # after a page refresh without re-running any LLM calls.
    session: Session = _session["session"]
    return session.display if session else []


# ── Sessions ───────────────────────────────────────────────────────────────────


@app.get("/sessions")
async def sessions_index() -> dict:
    session: Session = _session["session"]
    return {
        "active": session.id if session else None,
        "busy": list(_session["running"]),
        "sessions": list_sessions(),
    }


@app.post("/sessions/new")
async def sessions_new() -> dict:
    # The outgoing session is already persisted per-turn; just swap in a fresh one.
    # A fresh id owns no pending_actions tokens, and any dialogs left over from
    # the outgoing session (or any other session) must keep working — a click
    # on one of those now should still resolve normally, not 409. So, unlike
    # the old single-session model, there is nothing to clear here.
    _session["session"] = new_session(cfg.provider, kb_only=_session["kb_only"])
    return {"id": _session["session"].id}


@app.post("/sessions/{session_id}/resume")
async def sessions_resume(session_id: str) -> dict:
    live = _session["running"].get(session_id)
    if live is not None:
        # Mid-turn: a disk load would be stale (the background thread hasn't
        # saved yet) and check_resume is redundant — this session started
        # its turn under the current provider by construction.
        session = live
    else:
        try:
            session = load_session(session_id)
        except (FileNotFoundError, ValueError):
            raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
        try:
            check_resume(session, cfg.provider)
        except (PrivacyError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    _session["session"] = session
    _session["kb_only"] = session.kb_only
    _clear_pending_for(session_id)
    return {
        "id": session.id,
        "kb_only": session.kb_only,
        "display": session.display,
        # True when this session's own turn is still running in the
        # background thread (e.g. it was left mid-turn and is being resumed
        # again) — the frontend shows a placeholder and polls until it lands.
        "busy": session_id in _session["running"],
    }


@app.post("/sessions/{session_id}/pin")
async def sessions_pin(session_id: str, req: PinRequest) -> dict:
    try:
        set_pinned(session_id, req.pinned)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
    return {"id": session_id, "pinned": req.pinned}


@app.post("/sessions/{session_id}/rename")
async def sessions_rename(session_id: str, req: RenameRequest) -> dict:
    from jarvis.kb.store import get_store, update_chat_title

    try:
        applied_title = rename_session(session_id, req.title)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    # Keep the in-memory active session and the indexed chat chunks in step, so
    # the sidebar and search_chat_history both show the new name.
    active: Session = _session["session"]
    if active and active.id == session_id:
        active.title = applied_title
    update_chat_title(session_id, applied_title, get_store())
    return {"id": session_id, "title": applied_title}


@app.delete("/sessions/{session_id}")
async def sessions_delete(session_id: str) -> dict:
    from jarvis.kb.store import get_store

    if session_id in _session["running"]:
        raise HTTPException(
            status_code=409,
            detail="cannot delete a session while a reply is still being generated for it",
        )
    try:
        delete_session(session_id, store=get_store())
    except ValueError:
        raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
    result = {"deleted": session_id}
    _clear_pending_for(session_id)
    active: Session = _session["session"]
    if active and active.id == session_id:
        _session["session"] = new_session(cfg.provider, kb_only=_session["kb_only"])
        result["active"] = _session["session"].id
    return result


# ── Settings ───────────────────────────────────────────────────────────────────


@app.post("/config")
async def config(req: ConfigRequest) -> dict:
    # Sets the default for future new sessions AND the currently active
    # session's own flag — otherwise a running/resumed session would keep
    # using whatever kb_only it was created with, since /chat now builds its
    # tools from the resolved session rather than this global.
    _session["kb_only"] = req.kb_only
    active: Session = _session["session"]
    if active is not None:
        active.kb_only = req.kb_only
    return {"kb_only": req.kb_only}


@app.get("/settings")
async def settings_get() -> dict:
    return {"response_style": _session["response_style"]}


@app.post("/settings")
async def settings_set(req: SettingsRequest) -> dict:
    # Applies immediately to the running system prompt AND persists to
    # ~/.jarvis/config.toml (comments preserved via tomlkit).
    _session["response_style"] = req.response_style
    set_config_value("chat", "response_style", req.response_style)
    reset_config()
    return {"response_style": req.response_style}


@app.post("/confirm-action")
async def confirm_action(req: ConfirmActionRequest) -> dict:
    # The human decision point for deletions requested by the model. Each
    # dialog owns its own token, so popping it here only ever resolves that
    # one dialog — other pending confirmations from the same turn (or a
    # different session entirely) are untouched. A token that isn't in the
    # dict anymore — already resolved, or cleared by a new turn/resume on its
    # own session — 409s rather than silently doing nothing. No session check
    # here: token possession is the capability, regardless of which session
    # happens to be active in the browser right now.
    entry = _session["pending_actions"].pop(req.token, None)
    if entry is None:
        raise HTTPException(status_code=409, detail="this confirmation request was superseded")
    if not req.confirmed:
        return {"result": "Cancelled — nothing was removed."}
    from jarvis.kb.store import get_store

    result = execute_remove(entry["action"], get_store())
    return {"result": result}


# ── Papers manager ───────────────────────────────────────────────────────────


_PAPER_FIELDS = (
    "title", "authors", "doi", "source", "storage_mode",
    "visibility", "score", "track", "date_added", "chunk_count", "file_path",
)


@app.get("/papers")
async def papers_list(q: str = "") -> list[dict]:
    # list_papers already de-dupes by source and sorts most-recent-first; the
    # default limit is high enough that a single-user KB never gets truncated.
    from jarvis.kb.store import get_store, list_papers

    papers = list_papers(store=get_store())
    if q:
        needle = q.lower()
        papers = [
            p for p in papers
            if needle in " ".join(str(p.get(f, "")) for f in ("title", "authors", "doi", "source")).lower()
        ]
    return [{field: p.get(field) for field in _PAPER_FIELDS} for p in papers]


@app.post("/papers/meta")
async def papers_update_meta(req: PaperMetaRequest) -> dict:
    # Metadata-only — no re-embedding. Only the fields the caller sent are
    # changed; everything else on each chunk is left alone.
    from jarvis.kb.store import get_store, update_paper_metadata

    store = get_store()
    # Scoped to papers, mirroring /papers/remove: editing a note or digest
    # by source through this route 404s.
    existing = store._collection.get(
        where={"$and": [{"source": {"$eq": req.source}}, {"doc_type": {"$eq": "paper"}}]},
        include=[],
    )
    if not existing["ids"]:
        raise HTTPException(status_code=404, detail=f"no paper with source {req.source!r}")
    updated = update_paper_metadata(
        req.source, title=req.title, authors=req.authors, doi=req.doi, store=store
    )
    return {"source": req.source, "chunks_updated": updated}


@app.post("/papers/remove")
async def papers_remove(req: PaperRemoveRequest) -> dict:
    # Human-only by construction: no chat tool references this route, so the
    # model can never reach it. It deletes ChromaDB chunks via execute_remove
    # ONLY — same function the token-confirmed chat removal path calls — and
    # never touches a file on disk. The two-step "are you sure" confirmation
    # lives entirely in the modal UI; by the time this fires the human has
    # already seen the "Database entry only…" invariant line and clicked
    # through it themselves.
    from jarvis.kb.store import get_store

    store = get_store()
    # Scoped to papers so the route matches its name — a note or digest
    # source 404s here instead of being silently removable.
    result = store._collection.get(
        where={"$and": [{"source": {"$eq": req.source}}, {"doc_type": {"$eq": "paper"}}]},
        include=["metadatas"],
    )
    ids = result["ids"]
    if not ids:
        raise HTTPException(status_code=404, detail=f"no paper with source {req.source!r}")
    meta = result["metadatas"][0] if result["metadatas"] else {}
    action = {
        "ids": ids,
        "title": meta.get("title", "untitled"),
        "doc_type": meta.get("doc_type", "paper"),
        "source": req.source,
    }
    return {"result": execute_remove(action, store)}


# ── Chat ───────────────────────────────────────────────────────────────────────


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    if req.session_id in _session["running"]:
        raise HTTPException(
            status_code=409,
            detail="a reply is still being generated for this session — wait for it to finish",
        )

    # Resolve the session this message is actually addressed to. If it's the
    # currently active in-memory object, use it directly — this is what lets
    # a brand-new, not-yet-saved session accept its very first message (it
    # has no file on disk yet to load). Otherwise load it from disk and run
    # the same resume-safety checks /sessions/{id}/resume applies (privacy /
    # cross-provider). This makes /chat impossible to misdeliver: a message
    # always lands on the session named in the request, never on whatever
    # happens to be "active" in the shared dict at that instant.
    active: Session = _session["session"]
    if active is not None and active.id == req.session_id:
        session = active
    else:
        try:
            session = load_session(req.session_id)
        except (FileNotFoundError, ValueError):
            raise HTTPException(status_code=404, detail=f"no session {req.session_id!r}")
        try:
            check_resume(session, cfg.provider)
        except (PrivacyError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    provider = _session["provider"]
    tools = _build_tools(session.kb_only)
    # Built fresh per turn from the resolved session's own kb_only, rather
    # than a cached global — otherwise a /config change made while viewing a
    # different session (or a resumed session with its own kb_only) would be
    # silently ignored for this turn.
    system = build_system_prompt(
        kb_only=session.kb_only,
        response_style=_session["response_style"],
        skills=list_skills(cfg.skills_dir),
    )

    # Dialogs left over from a previous turn on THIS session are no longer
    # actionable once a new one starts — clear only this session's tokens so
    # a stale click 409s, without touching any other session's still-pending
    # dialogs (including ones belonging to a session mid-turn right now).
    _clear_pending_for(session.id)
    # Registered on the event loop, before the thread is spawned, so a second
    # /chat for the same id arriving before the thread has even started still
    # sees the busy guard above.
    _session["running"][session.id] = session

    # The agent runs in a background thread so the async event loop stays free
    # to serve SSE chunks. A plain queue bridges the two worlds. If the browser
    # aborts mid-stream the thread runs to completion with no consumer — fine
    # for a single-user local app; the turn still lands in the session history.
    event_queue: queue.Queue = queue.Queue()

    def run_agent() -> None:
        tool_calls_log: list[tuple[str, str]] = []
        reply = None  # set on every path below; finally always has something to send

        def request_confirmation(description: str, action: dict):
            # Store the pending deletion under a fresh token, tagged with the
            # session it belongs to, and show the dialog; returning None
            # tells the tool the decision is deferred to the human. Keying by
            # token (rather than one slot) means a second deletion proposed
            # in the same turn doesn't clobber the first — both dialogs stay
            # independently confirmable.
            token = uuid.uuid4().hex
            # A resume of this session on the main thread (its _clear_pending_for
            # call) may clear this entry right around this insert; either way the
            # token just goes stale and a later click 409s — intended
            # (fail-closed), not a bug.
            _session["pending_actions"][token] = {"session_id": session.id, "action": action}
            event_queue.put({"type": "confirm", "description": description, "token": token})
            return None

        def dispatch_fn(name: str, arguments: dict) -> str:
            arg_summary = _format_tool_args(arguments)
            # Push a tool event so the browser can show it immediately
            event_queue.put({"type": "tool", "name": name, "args": arg_summary})
            tool_calls_log.append((name, arg_summary))
            return _dispatch_tool(
                name, arguments, _vault, cfg.provider, provider,
                session=session, request_confirmation=request_confirmation,
            )

        try:
            from jarvis.kb.store import get_store

            try:
                maybe_compact(session, provider, get_config())
            except LLMError:
                pass  # compaction is best-effort; the turn itself may still work

            session.turn_starts.append(len(session.messages))
            session.messages.append({"role": "user", "content": req.message})
            session.display.append({"role": "user", "content": req.message})
            # Save right away (no store=, so no indexing/prune side effects) —
            # the question is on disk before the LLM call even starts, so it
            # survives a crash or a session switch mid-turn instead of
            # vanishing from the sidebar's history.
            save_session(session)

            reply = provider.agentic_turn(
                messages=session.messages,
                tools=tools,
                dispatch_fn=dispatch_fn,
                system=system,
            )

            session.display.append({
                "role": "assistant",
                "content": reply,
                "tool_calls": tool_calls_log,
            })
            save_session(session, store=get_store())
        except LLMError as exc:
            log.exception("chat turn failed with an LLM error")
            reply = f"⚠️ {exc}"
            session.display.append({"role": "assistant", "content": reply, "tool_calls": tool_calls_log})
            save_session(session)
        except Exception as exc:
            # Anything else is a bug, not an expected provider failure — log
            # the full traceback (an LLM would only paraphrase the message,
            # losing it) and still hand the browser a usable reply instead of
            # leaving the "Working..." placeholder stuck forever.
            log.exception("chat turn crashed unexpectedly")
            reply = f"⚠️ Internal error: {exc}"
            session.display.append({"role": "assistant", "content": reply, "tool_calls": tool_calls_log})
            save_session(session)
        finally:
            # Note: no reinstall step here. In the old single-session model,
            # resuming this same id mid-turn installed a fresh-from-disk copy
            # that this thread never wrote to, so the finished object had to
            # be swapped back in for /history to show it. Now resume installs
            # the *live registry object* (see /sessions/{id}/resume) — the
            # very same object this thread is mutating — so there is nothing
            # stale to reconcile.
            #
            # Always reaches the browser and always clears the busy flag,
            # even if the try block died before `reply` was ever assigned —
            # this is what keeps the SSE stream from hanging indefinitely.
            event_queue.put({
                "type": "reply",
                "content": reply,
                "tool_calls": tool_calls_log,
                "private": session.private,
            })
            event_queue.put(None)  # sentinel — tells the stream generator to stop
            _session["running"].pop(session.id, None)

    threading.Thread(target=run_agent, daemon=True).start()

    async def stream():
        # Poll the queue every 50 ms. Yields SSE-formatted data lines.
        while True:
            try:
                event = event_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue
            if event is None:
                return
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
