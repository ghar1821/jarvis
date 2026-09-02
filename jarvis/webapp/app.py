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
import subprocess
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi import Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from jarvis.core import transcript
from jarvis.core.config import (
    CONFIG_FILE,
    get_config,
    load_config,
    redact_secrets,
    reset_config,
    set_config_value,
)
from jarvis.core.errors import AuthenticationError, LLMError, PrivacyError
from jarvis.chat.chat import (
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
    record_usage,
    rename_session,
    save_session,
    session_cost_usd,
    set_pinned,
)

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
# providers    : {spec: ChatProvider} — clients cached by "provider:model".
#                There is no single active provider any more: each session
#                carries its own model_spec and resolves from here per turn.
# running      : {session_id: live Session object} — every session currently
#                mid-turn in its own run_agent background thread. A second
#                /chat addressed at an id already in here 409s; resuming that
#                id installs this same live object (not a stale disk copy) so
#                /history reflects turns as they land; sessions_delete refuses
#                to delete an id that's in here.
_session: dict = {
    "session": None,
    # Providers cached by "provider:model" spec. Each session resolves its own
    # from this cache per turn, so two sessions can run different models at the
    # same time and a switch costs no client rebuild.
    "providers": {},
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


def _resolve_session(session_id: str) -> "Session":
    """
    The session a request is addressed to: the live in-memory object when the
    id matches (a brand-new session has no file on disk yet), otherwise the
    one loaded from disk. 404s on an unknown id.
    """
    active: Session = _session["session"]
    if active is not None and active.id == session_id:
        return active
    running = _session["running"].get(session_id)
    if running is not None:
        return running
    try:
        return load_session(session_id)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail=f"no session {session_id!r}")


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
    if not kb_only:
        tools.append(USE_OWN_KNOWLEDGE_TOOL)
    return tools


def _live_config():
    """
    Config as the file says right now, not as it said when this process
    started — so a hand-edit of the config shows up on the next
    picker open instead of needing a restart.

    Returned as a local rather than through the process-wide singleton:
    nothing else should have its config change underneath it mid-turn. A
    malformed file falls back to what the process started with, so a typo
    empties nothing.
    """
    try:
        return load_config(CONFIG_FILE)
    except Exception as exc:
        # Falling back keeps the picker populated rather than emptying it over
        # a typo — but a config being ignored is the single most confusing
        # failure there is, so it says so.
        log.warning("could not re-read %s, using the config from startup: %s", CONFIG_FILE, exc)
        return cfg


def _provider_for(spec: str):
    """
    The cached client for one "provider:model" spec, built on first use.

    Caching matters beyond speed for Ollama, where rebuilding the client
    needlessly is wasteful, and it keeps parallel sessions on the same model
    sharing one client.
    """
    from jarvis.core.llm import make_provider

    if spec not in _session["providers"]:
        _session["providers"][spec] = make_provider(spec)
    return _session["providers"][spec]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Seed the editable prompt copies on first run, so they exist under
    # ~/.jarvis/prompts/ before anyone opens the editor looking for them.
    from jarvis.core.prompts import ensure_all

    for name in ensure_all():
        log.info("created %s from the shipped default", name)

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _auto_refresh_vault, _vault)
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


class DraftSaveRequest(BaseModel):
    draft_id: str
    file: str
    content: str
    expect_hash: str = ""


class DraftNewRequest(BaseModel):
    filename: str
    title: str = ""
    # When given, the file joins that document instead of starting a new one.
    # This is how a LaTeX project grows a chapter or a .bib rather than
    # scattering its parts across separate folders.
    draft_id: str = ""


class DraftKeepRequest(BaseModel):
    draft_id: str
    keep: bool


class PreviewRequest(BaseModel):
    draft_id: str
    file: str


class ApplyEditRequest(BaseModel):
    token: str
    indices: "list[int] | None" = None


class RevealRequest(BaseModel):
    draft_id: str
    file: str


class PromptRequest(BaseModel):
    text: str


class RestoreVersionRequest(BaseModel):
    draft_id: str
    file: str
    version: str


class ModelRequest(BaseModel):
    session_id: str
    spec: str


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


class DocumentMetaRequest(BaseModel):
    source: str
    title: str | None = None
    authors: str | None = None
    doi: str | None = None


class DocumentRemoveRequest(BaseModel):
    source: str


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((_ROOT / "index.html").read_text())


@app.get("/info")
async def info() -> dict:
    # The header shows the model the ACTIVE SESSION is on, not a process-wide
    # one — two sessions can be running different models at the same time.
    from jarvis.chat.sessions import session_cost_usd

    session: Session = _session["session"]
    return {
        "provider": session.model_spec if session else cfg.provider,
        "provider_kind": session.provider if session else cfg.provider,
        # What actually answered, when a router picked something else. Empty
        # for an ordinary model, where it would only repeat `provider`.
        "served": session.served_model if session else "",
        "cost_usd": session_cost_usd(session) if session else 0.0,
        # Per-model spend, so a router session can show which models it used.
        "cost_by_model": dict(session.cost) if session else {},
        "vault": str(_vault),
    }


@app.get("/config/summary")
async def config_summary() -> dict:
    """
    The loaded configuration, grouped for display, with secrets reduced to
    set/not set. Read fresh so an edited file shows up without a restart, the
    same as the model picker.
    """
    from jarvis.core.config import describe

    return {"sections": describe(_live_config())}


@app.get("/prompts")
async def prompts_index() -> dict:
    """Every editable prompt, with whether it has been changed from default."""
    from jarvis.core.prompts import listing

    return {"prompts": listing()}


@app.get("/prompts/{name}")
async def prompt_get(name: str) -> dict:
    """One prompt's current text, creating the copy if this is the first look."""
    from jarvis.core.prompts import PromptError, is_customised, load

    try:
        return {"name": name, "text": load(name), "customised": is_customised(name)}
    except PromptError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/prompts/{name}")
async def prompt_save(name: str, req: PromptRequest) -> dict:
    """
    Replace the user's copy. Takes effect on the next turn — the system prompt
    is rebuilt per turn and the others are read per call, so nothing caches a
    stale version until a restart.
    """
    from jarvis.core.prompts import PromptError, is_customised, save

    try:
        save(name, req.text)
    except PromptError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"name": name, "customised": is_customised(name)}


@app.post("/prompts/{name}/reset")
async def prompt_reset(name: str) -> dict:
    """Put the shipped default back, and hand back the text it restored."""
    from jarvis.core.prompts import PromptError, reset

    try:
        return {"name": name, "text": reset(name), "customised": False}
    except PromptError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/models")
async def models_index() -> dict:
    """
    The switchable catalogue for the picker, read from config only — this
    route never touches the network, so opening the UI makes no outbound
    request.

    Config is re-read per call (see _live_config), so a hand-edit shows up on
    the next open without a restart.
    """
    from jarvis.chat.models import list_catalogue

    live = _live_config()
    session: Session = _session["session"]
    current = session.model_spec if session else live.provider
    return {
        "current": current,
        # A private session may only run locally, so the picker can grey out
        # the cloud entries with a reason instead of failing on click.
        "private": bool(session and session.private),
        "models": list_catalogue(live, current),
    }


@app.post("/model")
async def model_switch(req: ModelRequest) -> dict:
    """
    Switch one session to another model, from the next turn onwards.

    Refused while that session has a turn in flight (the running turn holds a
    provider already), and refused outright for a private session moving to a
    cloud model — once private content is in the transcript, the transcript
    itself is private.
    """
    from jarvis.chat.models import apply_switch

    if req.session_id in _session["running"]:
        raise HTTPException(
            status_code=409,
            detail="a reply is still being generated for this session — wait for it to finish",
        )

    session = _resolve_session(req.session_id)
    try:
        spec = apply_switch(session, req.spec, _live_config())
    except PrivacyError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    save_session(session)
    return {"ok": True, "spec": spec}


@app.get("/history")
async def history() -> list:
    # Returns the display list so the browser can re-render the conversation
    # after a page refresh without re-running any LLM calls.
    session: Session = _session["session"]
    return session.display if session else []


# ── Drafts ─────────────────────────────────────────────────────────────────────
#
# The editor's routes. Reading and writing a draft is a HUMAN action here — the
# model reaches the sandbox only through its chat tools, and cannot reach these
# routes at all. That is also why /reveal lives here and nowhere else.


def _draft_error(exc: Exception) -> HTTPException:
    """
    Draft and render failures are user-facing messages, not 500s — "that file
    is not in the draft", or a missing LaTeX toolchain and how to install it,
    is the whole content of the error and is useless as a stack trace.
    Anything else is a real bug and propagates.
    """
    from jarvis.drafts import DraftError, RenderError

    if isinstance(exc, (DraftError, RenderError)):
        return HTTPException(status_code=400, detail=str(exc))
    raise exc


@app.get("/drafts")
async def drafts_index() -> dict:
    from jarvis.drafts import latex_available, list_drafts, pdf_export_available

    return {
        "drafts": list_drafts(),
        "retention_days": cfg.drafts_retention_days,
        # The UI hides the compile and export buttons rather than offering
        # something that can only fail on a machine without the toolchain.
        "latex": latex_available(),
        "pandoc": pdf_export_available(),
    }


@app.get("/drafts/{draft_id}/file")
async def draft_file(draft_id: str, file: str = "") -> dict:
    from jarvis.drafts import read_draft
    from jarvis.drafts.workspace import list_versions

    try:
        draft = read_draft(draft_id, file)
        draft["versions"] = list_versions(draft_id, draft["file"])
        return draft
    except Exception as exc:
        raise _draft_error(exc)


@app.post("/drafts/save")
async def draft_save(req: DraftSaveRequest) -> dict:
    """
    The human's own save. Writes straight through — a person editing their own
    file — but still through resolve_in_draft, with a snapshot taken first and
    the hash checked so a second tab cannot be clobbered.
    """
    from jarvis.drafts import save_draft_file

    try:
        return {"hash": save_draft_file(req.draft_id, req.file, req.content, req.expect_hash)}
    except Exception as exc:
        raise _draft_error(exc)


@app.post("/drafts/keep")
async def draft_keep(req: DraftKeepRequest) -> dict:
    """Exempt a draft from the retention sweep, or stop exempting it."""
    from jarvis.drafts import set_keep

    try:
        return set_keep(req.draft_id, req.keep)
    except Exception as exc:
        raise _draft_error(exc)


@app.post("/drafts/restore")
async def draft_restore(req: RestoreVersionRequest) -> dict:
    from jarvis.drafts.workspace import restore_version

    try:
        return {"hash": restore_version(req.draft_id, req.file, req.version)}
    except Exception as exc:
        raise _draft_error(exc)


@app.post("/drafts/new")
async def draft_new(req: DraftNewRequest) -> dict:
    """
    Start an empty document, or add a file to one that exists.

    A draft is a folder, not a file. Passing `draft_id` puts the new file in
    that folder, which is what keeps a LaTeX project together: `main.tex`, its
    chapters and its `.bib` compile as one document precisely because they sit
    in one draft, and compile_latex seeds its temp directory from the whole
    folder.

    Nothing special is needed to make the assistant aware of it: drafts are the
    one place it can already read and write, so a document you create here is
    immediately something it can be asked to work on.

    Visibility follows the active session, so a document started during a
    private conversation is private too — fail-closed, matching what the
    create_draft tool does.
    """
    from jarvis.drafts import add_draft_file, create_draft

    session: Session = _session["session"]
    if req.draft_id:
        # Adding to an existing document, not starting one. Visibility and
        # ownership come from the document it joins.
        try:
            add_draft_file(req.draft_id, req.filename, "")
        except Exception as exc:
            raise _draft_error(exc)
        return {"id": req.draft_id, "main_file": req.filename}
    try:
        return create_draft(
            title=req.title,
            filename=req.filename,
            content="",
            visibility="private" if (session and session.private) else "public",
            session_id=session.id if session else "",
        )
    except Exception as exc:
        raise _draft_error(exc)


@app.delete("/drafts/{draft_id}")
async def draft_delete(draft_id: str) -> dict:
    """
    Delete a draft at the user's request.

    Human-only by construction: no chat tool is named for deleting a draft, so
    the model has no way to reach this — the same reasoning that lets
    /documents/remove exist without a token flow. Anything already copied
    into the vault is untouched, because archiving copies.
    """
    from jarvis.drafts import delete_draft

    try:
        metadata = delete_draft(draft_id)
    except Exception as exc:
        raise _draft_error(exc)
    return {"deleted": draft_id, "title": metadata.get("title", "")}


@app.post("/preview")
async def preview(req: PreviewRequest) -> dict:
    """
    Render a Markdown draft for the preview pane.

    The HTML comes back as a string for the browser to put in a SANDBOXED
    iframe (`sandbox=""`, via srcdoc): a draft can hold text the model produced
    from an untrusted document, and it must never run script in the app's
    origin. Embedded HTML is also stripped at render time.
    """
    from jarvis.drafts import markdown_to_html, read_draft

    try:
        draft = read_draft(req.draft_id, req.file)
        return {"html": markdown_to_html(draft["text"])}
    except Exception as exc:
        raise _draft_error(exc)


@app.post("/compile")
async def compile_document(req: PreviewRequest) -> Response:
    """
    Compile a .tex draft and return the PDF, or the log when it failed.

    Runs in a sandboxed temp directory with shell escape off and TeX's file
    access restricted — see jarvis/drafts/render.py. Compilation involves no
    model, so it is permitted on a private draft.
    """
    from jarvis.drafts import compile_latex

    loop = asyncio.get_running_loop()
    try:
        # Blocking subprocess work belongs off the event loop.
        result = await loop.run_in_executor(None, compile_latex, req.draft_id, req.file)
    except Exception as exc:
        raise _draft_error(exc)

    if not result["ok"]:
        # A LaTeX error is a normal outcome with a log to read, not a 500.
        return JSONResponse(status_code=422, content={"detail": "compile failed", "log": result["log"]})
    return Response(
        content=result["pdf"],
        media_type="application/pdf",
        headers={"X-Compile-Log-Length": str(len(result["log"]))},
    )


@app.post("/export")
async def export_document(req: PreviewRequest) -> Response:
    """Export a Markdown draft as PDF via pandoc."""
    from jarvis.drafts import markdown_to_pdf

    loop = asyncio.get_running_loop()
    try:
        pdf = await loop.run_in_executor(None, markdown_to_pdf, req.draft_id, req.file)
    except Exception as exc:
        raise _draft_error(exc)
    return Response(content=pdf, media_type="application/pdf")


def _proposal_payload(proposal: dict) -> dict:
    """
    The shape a proposal takes on the wire. Shared by the SSE push and by
    /proposals, so a suggestion re-opened later renders identically to the way
    it first appeared — the browser has one code path for both.

    `new_text` is deliberately not included: the browser only ever needs the
    hunks, and the whole proposed file is not its business.
    """
    return {
        "token": proposal["token"],
        "draft_id": proposal["draft_id"],
        "file": proposal["file"],
        "rationale": proposal.get("rationale", ""),
        "created": proposal.get("created", ""),
        "hunks": [
            {
                "index": h["index"],
                "kind": h["kind"],
                "header": h["header"],
                "old_start": h["old_start"],
                "old_end": h["old_end"],
                "old_lines": h["old_lines"],
                "new_lines": h["new_lines"],
                # Which lines inside the hunk actually changed. Without these
                # the browser tints the whole span, context included.
                "old_spans": h["old_spans"],
                "new_spans": h["new_spans"],
            }
            for h in proposal["hunks"]
        ],
    }


@app.get("/proposals")
async def proposals_index() -> dict:
    """
    Every suggestion still waiting on a decision.

    Proposals live in memory in this process, so they do not survive a restart
    — but within one run a suggestion the user navigated away from used to be
    unreachable, with nothing in the UI to say it was still there. The editor
    reads this on load so an open file shows its pending diff again, and so a
    document with one waiting says so in the list.
    """
    from jarvis.drafts.workspace import _proposals

    return {"proposals": [_proposal_payload(p) for p in _proposals.values()]}


@app.post("/proposals/discard-all")
async def proposals_discard_all() -> dict:
    """Drop every pending suggestion at once — the 'clear these out' button."""
    from jarvis.drafts.workspace import _proposals

    count = len(_proposals)
    _proposals.clear()
    return {"discarded": count}


@app.post("/apply-edit")
async def apply_edit(req: ApplyEditRequest) -> dict:
    """
    Apply the hunks a human accepted. The only route that writes an agent's
    proposed change, and it needs a token that came from a diff someone saw.

    Same token discipline as /confirm-action: possession is the capability, the
    token is one-shot, and an unknown or stale one is refused rather than
    guessed at.
    """
    from jarvis.drafts import apply_hunks

    try:
        return apply_hunks(req.token, req.indices)
    except Exception as exc:
        raise _draft_error(exc)


@app.post("/discard-edit")
async def discard_edit(req: ApplyEditRequest) -> dict:
    """Drop a proposal the human rejected outright."""
    from jarvis.drafts.workspace import discard_proposal

    discard_proposal(req.token)
    return {"ok": True}


def _file_manager_command(path: "Path", platform: str) -> list:
    """
    The argv that shows `path` in this platform's file manager.

    Always a list, never a string, so nothing goes through a shell.

    macOS and Windows can both *reveal* a file — highlight it in a window
    without opening it. Linux has no portable equivalent: `xdg-open` on a file
    hands it to whatever application claims the extension, and opening a `.tex`
    the model wrote in an editor of the OS's choosing is not a decision this
    should make. So Linux opens the containing folder instead, which is the
    same intent minus the highlight.

    Windows wants `/select,<path>` as ONE argument. Splitting it in two, which
    this used to do, leaves explorer with a `/select,` naming nothing — it
    opens the folder and silently fails to select.
    """
    if platform == "darwin":
        return ["open", "-R", str(path)]
    if platform == "win32":
        return ["explorer", f"/select,{path}"]
    return ["xdg-open", str(path.parent)]


@app.post("/reveal")
async def reveal(req: RevealRequest) -> dict:
    """
    Show a draft file in the OS file manager, so the user can copy it wherever
    they want by hand.

    This replaced a password-gated archive flow. Jarvis now has no write path
    into the vault at all: moving a document out of the sandbox is something
    the user does in Finder, with the file manager's own confirmations, and
    there is no gate for an injected instruction to try to talk its way past.

    Human-only by construction, like /documents/remove — no chat tool
    references it. The path still goes through resolve_in_draft, so the only
    thing this can reveal is a file inside the drafts sandbox, and the command
    is a fixed argv with no shell.
    """
    from jarvis.drafts.workspace import resolve_in_draft

    try:
        path = resolve_in_draft(req.draft_id, req.file)
    except Exception as exc:
        raise _draft_error(exc)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No file {req.file!r} in that draft")

    command = _file_manager_command(path, sys.platform)

    try:
        subprocess.run(command, check=False, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not open a file manager here: {exc}. The file is at {path}",
        )
    return {"path": str(path)}


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
            # Checked against the session's OWN model, which is what it will
            # actually resume on — a private session may only ever run local.
            check_resume(session, session.model_spec)
        except (PrivacyError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    _session["session"] = session
    _session["kb_only"] = session.kb_only
    _clear_pending_for(session_id)
    return {
        "id": session.id,
        "kb_only": session.kb_only,
        "model": session.model_spec,
        "cost_usd": session_cost_usd(session),
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


_DOCUMENT_FIELDS = (
    "title", "authors", "doi", "source", "storage_mode",
    "visibility", "score", "track", "date_added", "chunk_count", "file_path",
    # Record fields, present on notes with frontmatter and absent on papers.
    "doc_type", "category", "status", "entity", "event_date", "tags",
)

# What the modal's search box matches against, across both kinds.
_SEARCHABLE_FIELDS = (
    "title", "authors", "doi", "source", "file_path",
    "category", "status", "entity", "tags",
)


@app.get("/documents")
async def documents_list(q: str = "", kind: str = "papers", category: str = "",
                         status: str = "") -> list[dict]:
    # list_documents already de-dupes and sorts most-recent-first; the default
    # limit is high enough that a single-user KB never gets truncated.
    from jarvis.kb.store import get_store, list_documents

    documents = list_documents(
        doc_type="note" if kind == "notes" else "paper",
        category=category or None,
        status=status or None,
        store=get_store(),
    )
    if q:
        needle = q.lower()
        documents = [
            d for d in documents
            if needle in " ".join(str(d.get(f, "")) for f in _SEARCHABLE_FIELDS).lower()
        ]
    return [{field: d.get(field) for field in _DOCUMENT_FIELDS} for d in documents]


@app.post("/documents/meta")
async def documents_update_meta(req: DocumentMetaRequest) -> dict:
    # Metadata-only — no re-embedding. Only the fields the caller sent are
    # changed; everything else on each chunk is left alone.
    from jarvis.kb.store import get_store, update_paper_metadata

    store = get_store()
    # Scoped to papers, mirroring /documents/remove: editing a note or digest
    # by source through this route 404s. Notes are edited in Obsidian — jarvis
    # indexes the vault, it does not own it.
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


@app.post("/documents/remove")
async def documents_remove(req: DocumentRemoveRequest) -> dict:
    # Human-only by construction: no chat tool references this route, so the
    # model can never reach it. It deletes ChromaDB chunks via execute_remove
    # ONLY — same function the token-confirmed chat removal path calls — and
    # never touches a file on disk. The two-step "are you sure" confirmation
    # lives entirely in the modal UI; by the time this fires the human has
    # already seen the "Database entry only…" invariant line and clicked
    # through it themselves.
    from jarvis.kb.store import get_store

    store = get_store()
    # Scoped to papers deliberately: a note's KB entry is derived from a file
    # in the vault, so removing it here would just be undone by the next sync.
    # A note or digest source 404s instead of appearing to work.
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
            check_resume(session, session.model_spec)
        except (PrivacyError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    # Resolved from the session's OWN model, so two sessions can be mid-turn
    # on different models at the same time.
    try:
        provider = _provider_for(session.model_spec)
    except (ValueError, AuthenticationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    tools = _build_tools(session.kb_only)
    # Built fresh per turn from the resolved session's own kb_only, rather
    # than a cached global — otherwise a /config change made while viewing a
    # different session (or a resumed session with its own kb_only) would be
    # silently ignored for this turn.
    system = build_system_prompt(
        kb_only=session.kb_only,
        response_style=_session["response_style"],
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

        def request_edit_review(proposal: dict):
            """
            Push the diff to the browser and defer. Returning None tells the
            tool the decision belongs to the human — the same contract
            request_confirmation uses for deletions. Only /apply-edit, clicked
            outside the LLM loop, writes anything.
            """
            event_queue.put({"type": "edit_proposal", **_proposal_payload(proposal)})
            return None

        def dispatch_fn(name: str, arguments: dict) -> str:
            arg_summary = _format_tool_args(arguments)
            # Push a tool event so the browser can show it immediately
            event_queue.put({"type": "tool", "name": name, "args": arg_summary})
            tool_calls_log.append((name, arg_summary))
            return _dispatch_tool(
                name, arguments, _vault, session.provider, provider,
                session=session, request_confirmation=request_confirmation,
                request_edit_review=request_edit_review,
            )

        try:
            from jarvis.kb.store import get_store

            try:
                maybe_compact(session, provider, get_config())
            except LLMError as exc:
                # Best-effort: the turn still works uncompacted. But if this
                # keeps failing the context grows without bound and turns get
                # slower and dearer for no visible reason.
                log.warning("compaction failed, continuing uncompacted: %s", exc)

            session.turn_starts.append(len(session.messages))
            session.messages.append(transcript.user_message(req.message))
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
            # Second layer. The provider already scrubs its own messages, but
            # this reply is written to the session file and indexed as a chat
            # chunk, so anything reaching it gets checked again.
            reply = redact_secrets(f"⚠️ {exc}")
            session.display.append({"role": "assistant", "content": reply, "tool_calls": tool_calls_log})
            save_session(session)
        except Exception as exc:
            # Anything else is a bug, not an expected provider failure — log
            # the full traceback (an LLM would only paraphrase the message,
            # losing it) and still hand the browser a usable reply instead of
            # leaving the "Working..." placeholder stuck forever.
            log.exception("chat turn crashed unexpectedly")
            reply = redact_secrets(f"⚠️ Internal error: {exc}")
            session.display.append({"role": "assistant", "content": reply, "tool_calls": tool_calls_log})
            save_session(session)
        finally:
            # Spend is recorded even when the turn failed part-way — those
            # requests were still billed. Only OpenRouter reports a figure;
            # the others return None and nothing is recorded.
            record_usage(session, session.model_spec, provider.pop_usage())
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
                "model": session.model_spec,
                "served": session.served_model,
                "cost_usd": session_cost_usd(session),
                "cost_by_model": dict(session.cost),
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
