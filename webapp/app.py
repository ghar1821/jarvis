"""
Jarvis web UI — FastAPI + SSE + vanilla JS.

Single-user local application. The active conversation is a persistent
Session (vault_chat/sessions.py): saved to ~/.jarvis/sessions/ after every
turn, resumable from the sidebar, pruned to the 50 most recent unpinned
sessions. Refreshing the browser restores the active conversation.

Launch:
    uv run webapp
"""

import asyncio
import json
import queue
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from digest.config import get_config, reset_config, set_config_value
from digest.errors import LLMError, PrivacyError
from vault_chat.chat import (
    READ_SKILL_TOOL,
    TOOLS,
    USE_OWN_KNOWLEDGE_TOOL,
    _auto_refresh_vault,
    _dispatch_tool,
    _format_tool_args,
    build_system_prompt,
    execute_remove,
)
from vault_chat.sessions import (
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
from vault_chat.skills import list_skills

_ROOT = Path(__file__).parent
cfg = get_config()
_vault = cfg.vault_path

# Single-user state — shared across browser tabs (intended for local use only).
# session   : the active persistent Session (messages + display + privacy flag)
# kb_only   : when True (default), LLM answers only from KB tools; when False,
#             it may fall back to training knowledge after searching the KB
_session: dict = {
    "session": None,
    "provider": None,
    "system": None,
    "kb_only": True,
    "response_style": cfg.response_style,
    # Deletion awaiting the user's Confirm/Cancel click. The model can only
    # request a removal; execution happens through /confirm-action, entirely
    # outside the LLM tool loop.
    "pending_action": None,
}


def _rebuild_system_prompt() -> None:
    _session["system"] = build_system_prompt(
        kb_only=_session["kb_only"],
        response_style=_session["response_style"],
        skills=list_skills(cfg.skills_dir),
    )


def _build_tools() -> list[dict]:
    tools = list(TOOLS)
    if list_skills(cfg.skills_dir):
        tools.append(READ_SKILL_TOOL)
    if not _session["kb_only"]:
        tools.append(USE_OWN_KNOWLEDGE_TOOL)
    return tools


@asynccontextmanager
async def lifespan(app: FastAPI):
    from digest.llm import make_provider

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _auto_refresh_vault, _vault)
    _session["provider"] = make_provider(cfg.provider)
    _session["session"] = new_session(cfg.provider, kb_only=True)
    _rebuild_system_prompt()
    yield


app = FastAPI(lifespan=lifespan)
# Blocks DNS-rebinding: a malicious page pointing an attacker domain at
# 127.0.0.1 gets refused because the Host header won't match.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
app.mount("/static", StaticFiles(directory=_ROOT / "static"), name="static")


# Request bodies — defined before any route so FastAPI can resolve each
# parameter's type at route-registration time. A model referenced by name
# before it exists (even via a quoted forward reference) makes FastAPI treat
# the parameter as a query param instead of a JSON body.
class ChatRequest(BaseModel):
    message: str


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


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((_ROOT / "index.html").read_text())


@app.get("/info")
async def info() -> dict:
    # Provider label shown in the browser header
    label = (
        f"Anthropic · {cfg.anthropic_model}"
        if cfg.provider == "anthropic"
        else f"Ollama · {cfg.ollama_model}"
    )
    return {"provider": label, "provider_kind": cfg.provider, "vault": str(_vault)}


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
    return {"active": session.id if session else None, "sessions": list_sessions()}


@app.post("/sessions/new")
async def sessions_new() -> dict:
    # The outgoing session is already persisted per-turn; just swap in a fresh one.
    _session["session"] = new_session(cfg.provider, kb_only=_session["kb_only"])
    return {"id": _session["session"].id}


@app.post("/sessions/{session_id}/resume")
async def sessions_resume(session_id: str) -> dict:
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
    _rebuild_system_prompt()
    return {"id": session.id, "kb_only": session.kb_only, "display": session.display}


@app.post("/sessions/{session_id}/pin")
async def sessions_pin(session_id: str, req: PinRequest) -> dict:
    try:
        set_pinned(session_id, req.pinned)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
    return {"id": session_id, "pinned": req.pinned}


@app.post("/sessions/{session_id}/rename")
async def sessions_rename(session_id: str, req: RenameRequest) -> dict:
    from digest.kb.store import get_store, update_chat_title

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
    from digest.kb.store import get_store

    try:
        delete_session(session_id, store=get_store())
    except ValueError:
        raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
    result = {"deleted": session_id}
    active: Session = _session["session"]
    if active and active.id == session_id:
        _session["session"] = new_session(cfg.provider, kb_only=_session["kb_only"])
        result["active"] = _session["session"].id
    return result


# ── Settings ───────────────────────────────────────────────────────────────────


@app.post("/config")
async def config(req: ConfigRequest) -> dict:
    _session["kb_only"] = req.kb_only
    _rebuild_system_prompt()
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
    _rebuild_system_prompt()
    return {"response_style": req.response_style}


@app.post("/confirm-action")
async def confirm_action(req: ConfirmActionRequest) -> dict:
    # The human decision point for deletions requested by the model.
    action = _session.get("pending_action")
    _session["pending_action"] = None
    if action is None:
        raise HTTPException(status_code=409, detail="no action awaiting confirmation")
    if not req.confirmed:
        return {"result": "Cancelled — nothing was removed."}
    from digest.kb.store import get_store

    result = execute_remove(action, get_store())
    return {"result": result}


# ── Chat ───────────────────────────────────────────────────────────────────────


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    provider = _session["provider"]
    system = _session["system"]
    session: Session = _session["session"]
    tools = _build_tools()

    # The agent runs in a background thread so the async event loop stays free
    # to serve SSE chunks. A plain queue bridges the two worlds. If the browser
    # aborts mid-stream the thread runs to completion with no consumer — fine
    # for a single-user local app; the turn still lands in the session history.
    event_queue: queue.Queue = queue.Queue()

    def run_agent() -> None:
        tool_calls_log: list[tuple[str, str]] = []

        def request_confirmation(description: str, action: dict):
            # Store the pending deletion and show the dialog; returning None
            # tells the tool the decision is deferred to the human.
            _session["pending_action"] = action
            event_queue.put({"type": "confirm", "description": description})
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
            from digest.kb.store import get_store

            try:
                maybe_compact(session, provider, get_config())
            except LLMError:
                pass  # compaction is best-effort; the turn itself may still work

            session.turn_starts.append(len(session.messages))
            session.messages.append({"role": "user", "content": req.message})
            session.display.append({"role": "user", "content": req.message})

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
            reply = f"⚠️ {exc}"
            session.display.append({"role": "assistant", "content": reply, "tool_calls": tool_calls_log})

        event_queue.put({
            "type": "reply",
            "content": reply,
            "tool_calls": tool_calls_log,
            "private": session.private,
        })
        event_queue.put(None)  # sentinel — tells the stream generator to stop

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
