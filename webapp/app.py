"""
Seshat web UI — FastAPI + SSE + vanilla JS.

Single-user local application. All state is kept in memory for the lifetime
of the server process. Refreshing the browser restores the conversation from
the in-memory display list.

Launch:
    uv run webapp
"""

import asyncio
import json
import queue
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from digest.config import get_config
from digest.errors import LLMError
from digest.llm import make_provider
from vault_chat.chat import TOOLS, USE_OWN_KNOWLEDGE_TOOL, _auto_refresh_vault, _dispatch_tool, build_system_prompt

_ROOT = Path(__file__).parent
cfg = get_config()
_vault = cfg.vault_path

# Single-user session — shared across browser tabs (intended for local use only).
# messages  : full API history passed to the LLM, including internal tool turns
# display   : user + assistant turns sent to the browser for rendering
# kb_only   : when True (default), LLM answers only from KB tools; when False,
#             it may fall back to training knowledge after searching the KB
_session: dict = {
    "messages": [],
    "display": [],
    "provider": None,
    "system": None,
    "kb_only": True,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _auto_refresh_vault, _vault)
    _session["provider"] = make_provider(cfg.provider)
    _session["system"] = build_system_prompt(kb_only=True)
    yield


app = FastAPI(lifespan=lifespan)


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
    return {"provider": label, "vault": str(_vault)}


@app.get("/history")
async def history() -> list:
    # Returns the display list so the browser can re-render the conversation
    # after a page refresh without re-running any LLM calls.
    return _session["display"]


class ChatRequest(BaseModel):
    message: str


class ConfigRequest(BaseModel):
    kb_only: bool


@app.post("/config")
async def config(req: ConfigRequest) -> dict:
    _session["kb_only"] = req.kb_only
    _session["system"] = build_system_prompt(kb_only=req.kb_only)
    return {"kb_only": req.kb_only}


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    provider = _session["provider"]
    system = _session["system"]
    messages = _session["messages"]
    # Snapshot kb_only at request time so a mid-stream toggle doesn't change it
    kb_only = _session["kb_only"]
    tools = TOOLS if kb_only else TOOLS + [USE_OWN_KNOWLEDGE_TOOL]

    messages.append({"role": "user", "content": req.message})
    _session["display"].append({"role": "user", "content": req.message})

    # The agent runs in a background thread so the async event loop stays free
    # to serve SSE chunks. A plain queue bridges the two worlds.
    event_queue: queue.Queue = queue.Queue()

    def run_agent() -> None:
        tool_calls_log: list[tuple[str, str]] = []

        def dispatch_fn(name: str, arguments: dict) -> str:
            arg_summary = ", ".join(f"{k}={repr(v)[:40]}" for k, v in arguments.items())
            # Push a tool event so the browser can show it immediately
            event_queue.put({"type": "tool", "name": name, "args": arg_summary})
            tool_calls_log.append((name, arg_summary))
            return _dispatch_tool(name, arguments, _vault, cfg.provider, provider)

        try:
            reply = provider.agentic_turn(
                messages=messages,
                tools=tools,
                dispatch_fn=dispatch_fn,
                system=system,
            )
        except LLMError as exc:
            reply = f"⚠️ {exc}"

        _session["display"].append({
            "role": "assistant",
            "content": reply,
            "tool_calls": tool_calls_log,
        })
        event_queue.put({"type": "reply", "content": reply, "tool_calls": tool_calls_log})
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
