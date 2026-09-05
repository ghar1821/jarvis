"""
LLM provider abstraction.

Three concrete adapters satisfy the ChatProvider protocol:
  OllamaProvider     — local model served by Ollama (http://localhost:11434)
  AnthropicProvider  — Anthropic Claude directly (API key)
  OpenRouterProvider — any model reachable through OpenRouter (API key)

All implement:
  complete(messages, max_tokens, context_length)  — single-shot text completion
  summarize(title, source)                         — paper summary, PDF-aware
  agentic_turn(messages, tools, dispatch_fn, system) — full tool-calling loop
  describe_image(image_bytes, context)             — caption a PDF figure
  pop_usage()                                      — spend since the last call

Use make_provider() to construct the right adapter from a spec string, with an
optional model after a colon:
  make_provider("ollama")                            → config ollama_model
  make_provider("anthropic")                         → config anthropic_model
  make_provider("openrouter:anthropic/claude-sonnet-4.6")

agentic_turn() speaks the provider-neutral transcript (jarvis/core/transcript.py)
rather than any one vendor's wire format, which is what lets a conversation
switch models mid-flight. Each adapter converts neutral → wire on the way in and
converts only the messages it generated back on the way out, so history written
by a different provider is never round-tripped through a lossy conversion.

Every method takes an optional cancel token (jarvis/core/cancel.py) so a turn
can be stopped while it is in flight. Ollama and Anthropic send their requests
streamed, through each adapter's single _request() helper — not to show tokens
as they arrive (replies are still delivered whole) but because a stream can be
checked between events and closed part-way, and closing the connection is the
only "stop generating" signal either service has. Because a turn's work stays
in the local `wire` list until commit(), a cancelled turn needs no unwinding:
nothing was ever published to the neutral transcript.

Ollama must be running (the macOS login-item app or `ollama serve`). For full
functionality the configured model needs tool-calling and vision support —
figure captioning and vision-based summaries depend on the vision capability.
"""

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

from . import transcript
from .errors import AuthenticationError, LLMError, PrivacyError, TurnCancelled

if TYPE_CHECKING:
    from .cancel import CancelToken
    from .config import Config

# The one local provider. Everything else sends content to someone else's
# machine, which is the only distinction the privacy model cares about.
LOCAL_PROVIDERS = frozenset({"ollama"})


def is_cloud_provider(provider_str: str) -> bool:
    """
    True when this provider sends content off the machine.

    Private vault notes, private sessions, and private drafts are gated on
    this one predicate rather than on any particular vendor name, so adding a
    provider cannot accidentally open a hole in the privacy model.
    """
    return provider_str.split(":", 1)[0] not in LOCAL_PROVIDERS

# ── Protocol ───────────────────────────────────────────────────────────────────


@runtime_checkable
class ChatProvider(Protocol):
    """
    Every method takes an optional cancel token. When one is supplied and the
    human stops the turn, the method raises TurnCancelled instead of returning.
    Callers that can't be interrupted (the digest pipeline, the sync daemon,
    the kb CLI) simply leave it None.
    """

    def complete(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
        context_length: int | None = None,
        cancel: "CancelToken | None" = None,
    ) -> str:
        """Single-shot text completion. A system message may be included in messages."""
        ...

    def summarize(
        self,
        title: str,
        source: "str | Path",
        max_tokens: int = 2048,
        cancel: "CancelToken | None" = None,
    ) -> str:
        """
        Generate a dense paper summary.

        source: plain text (abstract) or a Path to a PDF file.
        """
        ...

    def agentic_turn(
        self,
        messages: list[dict],
        tools: list[dict],
        dispatch_fn: Callable[[str, dict], str],
        system: str = "",
        cancel: "CancelToken | None" = None,
    ) -> str:
        """
        Run a full agentic turn including tool dispatch loop.

        messages is a provider-neutral transcript (see core/transcript.py) and
        is modified in place — the assistant turn and any tool calls/results
        are appended in neutral form, never in this provider's wire format.
        dispatch_fn(tool_name, arguments) -> result_string
        Returns the final text reply.

        A cancelled turn appends nothing at all. Each adapter accumulates the
        turn in a local wire list and publishes it to `messages` in one commit()
        at the return points, so raising TurnCancelled anywhere in between
        leaves the transcript exactly as it was found.
        """
        ...

    def describe_image(
        self,
        image_bytes: bytes,
        context: str,
        cancel: "CancelToken | None" = None,
    ) -> str:
        """
        Caption one image (a figure lifted from a PDF) so it can be indexed as
        searchable text. context is free text — usually the document title —
        that helps the model ground the description.
        """
        ...

    def pop_usage(self) -> "dict | None":
        """
        Spend recorded since the last call, then reset: {"usd", "requests"}.

        None when this provider does not report a cost jarvis can stand behind
        — a local model has none, and a provider that only reports token counts
        would need a price table jarvis deliberately does not keep.
        """
        ...


# ── Prompt loading ─────────────────────────────────────────────────────────────


# Shared by both providers' describe_image(), so figure captions read the same
# regardless of which model produced them. {context} is the document title.
_FIGURE_CAPTION_PROMPT = (
    "This image is a figure from a research paper or document titled "
    "\"{context}\". Describe what the figure shows in 2-4 dense, factual "
    "sentences a researcher could later search for: name the kind of figure "
    "(plot, diagram, micrograph, table, schematic), its axes or components, "
    "and the main result or relationship it conveys. Do not add generic "
    "commentary or caveats."
)


def _get_summary_prompt() -> str:
    # Not cached: the user can edit this from the UI mid-session, and a stale
    # module global would keep sending the old wording until a restart.
    from .prompts import load as _load_prompt

    return _load_prompt("paper_summary")


def _tool_arguments(raw) -> dict:
    """
    Normalise the arguments of one tool call to a plain dict.

    Ollama hands arguments back as a mapping already (not a JSON string like
    the OpenAI wire format), so this is usually just a copy; the parse is there
    for model variants that return a string anyway.
    """
    if isinstance(raw, str):
        try:
            return json.loads(raw or "{}")
        except json.JSONDecodeError:
            return {}
    return dict(raw or {})


# ── Ollama adapter ─────────────────────────────────────────────────────────────


class OllamaProvider:
    """
    Talks to a local Ollama server (http://localhost:11434 by default).

    Ollama keeps the model resident across the CLI, webapp, and sync daemon,
    and honours a per-request context window (num_ctx), so complete() can pass
    the caller's requested context straight through. Tool calling and vision
    both depend on the configured model supporting them.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import ollama

            self._client = ollama.Client()
        return self._client

    def _request(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        options: dict | None = None,
        cancel: "CancelToken | None" = None,
    ) -> dict:
        """
        Make one streamed chat request and return the assembled assistant
        message as a plain dict — role, content, and tool_calls when the model
        asked for any.

        Assembling the message ourselves (rather than handing back ollama's
        pydantic object) keeps the wire history JSON-serialisable for free, and
        streaming gives the cancel token somewhere to act: it is checked
        between chunks, and closing the generator on the way out drops the HTTP
        connection, which is how Ollama is told to stop generating.
        """
        client = self._get_client()
        content_parts: list[str] = []
        tool_calls: list[dict] = []
        try:
            stream = client.chat(
                model=self.model,
                messages=messages,
                tools=tools,
                options=options or {},
                stream=True,
            )
            try:
                for chunk in stream:
                    if cancel is not None:
                        cancel.check()
                    message = chunk["message"]
                    content_parts.append(message.get("content") or "")
                    for call in message.get("tool_calls") or []:
                        tool_calls.append({
                            "function": {
                                "name": call.function.name,
                                "arguments": _tool_arguments(call.function.arguments),
                            }
                        })
            finally:
                # Closing the generator unwinds ollama's streaming context and
                # drops the connection. On a cancel that is the kill signal; on
                # normal completion the generator is already exhausted and this
                # is a no-op.
                close_stream = getattr(stream, "close", None)
                if close_stream is not None:
                    close_stream()
        except (TurnCancelled, KeyboardInterrupt):
            raise
        except Exception as exc:
            raise LLMError(_redact(f"Ollama request failed: {exc}")) from exc

        assembled: dict = {"role": "assistant", "content": "".join(content_parts)}
        if tool_calls:
            assembled["tool_calls"] = tool_calls
        return assembled

    def complete(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
        context_length: int | None = None,
        cancel: "CancelToken | None" = None,
    ) -> str:
        # Ollama honours a per-request context window; only set it when asked.
        options = {"num_ctx": context_length} if context_length else {}
        return self._request(messages, options=options, cancel=cancel)["content"]

    def summarize(
        self,
        title: str,
        source: "str | Path",
        max_tokens: int = 2048,
        cancel: "CancelToken | None" = None,
    ) -> str:
        prompt = _get_summary_prompt().replace("{title}", title)
        if isinstance(source, Path):
            # Ollama has no document-input API, and the conversion is cheap
            # (pymupdf4llm, no ML models), so feed it the markdown text.
            from jarvis.kb.convert import pdf_to_markdown

            text = pdf_to_markdown(source)
        else:
            text = source
        messages = [{"role": "user", "content": f"{prompt}\n\nAbstract/text:\n{text}"}]
        return self.complete(messages, max_tokens=max_tokens, cancel=cancel)

    def agentic_turn(
        self,
        messages: list[dict],
        tools: list[dict],
        dispatch_fn: Callable[[str, dict], str],
        system: str = "",
        cancel: "CancelToken | None" = None,
    ) -> str:
        wire = transcript.to_ollama(messages, model=self.model)
        # Everything from here on is appended to the wire list; only this tail
        # is converted back, so history from another provider stays untouched.
        turn_start = len(wire)

        def commit() -> None:
            messages.extend(transcript.from_ollama(wire[turn_start:], model=self.model))

        while True:
            # Checked before the request so a turn stopped between iterations
            # never sends another one. Every raise below leaves `messages`
            # untouched, because commit() is only reached on a return.
            if cancel is not None:
                cancel.check()

            full = ([{"role": "system", "content": system}] + wire) if system else wire
            message = self._request(full, tools=tools, cancel=cancel)

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                reply = message["content"]
                wire.append({"role": "assistant", "content": reply})
                commit()
                return reply

            wire.append(message)
            for call in tool_calls:
                if cancel is not None:
                    cancel.check()
                name = call["function"]["name"]
                try:
                    result = dispatch_fn(name, dict(call["function"]["arguments"]))
                except PrivacyError as exc:
                    # Drop the assistant message we just added — with its tool
                    # calls unanswered it would leave the transcript invalid
                    # for the next turn — and stop without another LLM call.
                    wire.pop()
                    commit()
                    return str(exc)
                wire.append({"role": "tool", "tool_name": name, "content": result})

    def describe_image(
        self,
        image_bytes: bytes,
        context: str,
        cancel: "CancelToken | None" = None,
    ) -> str:
        prompt = _FIGURE_CAPTION_PROMPT.format(context=context or "untitled document")
        message = self._request(
            [{"role": "user", "content": prompt, "images": [image_bytes]}],
            cancel=cancel,
        )
        return message["content"]

    def pop_usage(self) -> "dict | None":
        """Nothing to report: the model runs on the user's own hardware."""
        return None


# ── Anthropic adapter ──────────────────────────────────────────────────────────


def _convert_tools_to_anthropic(tools: list[dict]) -> list[dict]:
    return [
        {
            "name": t["function"]["name"],
            "description": t["function"]["description"],
            "input_schema": t["function"]["parameters"],
        }
        for t in tools
    ]


def _block_to_dict(block) -> dict:
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return block.model_dump()


class AnthropicProvider:
    def __init__(self, model: str) -> None:
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        import os

        import anthropic

        from .config import get_config
        api_key = os.environ.get("ANTHROPIC_API_KEY") or get_config().anthropic_api_key
        if api_key:
            self._client = anthropic.Anthropic(api_key=api_key)
            return self._client

        raise AuthenticationError(
            "No Anthropic credentials found.\n"
            "  Set ANTHROPIC_API_KEY env var or add api_key to [auth] in ~/.jarvis/config.toml"
        )

    def _request(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        max_tokens: int = 2048,
        cancel: "CancelToken | None" = None,
    ):
        """
        Make one streamed message request and return the finished Message.

        get_final_message() reassembles exactly what messages.create() used to
        return (content blocks plus stop_reason), so callers are unaffected by
        the streaming. What streaming buys is the loop below: the cancel token
        is checked between events, and raising out of the `with` closes the
        HTTP response, which is how Anthropic is told to stop generating (the
        Messages API has no cancel endpoint — disconnecting is the signal).
        """
        client = self._get_client()
        request: dict = {"model": self.model, "max_tokens": max_tokens, "messages": messages}
        if system:
            request["system"] = system
        if tools:
            request["tools"] = tools
        try:
            with client.messages.stream(**request) as stream:
                for _ in stream:
                    if cancel is not None:
                        cancel.check()
                return stream.get_final_message()
        except (TurnCancelled, KeyboardInterrupt):
            raise
        except Exception as exc:
            raise LLMError(_redact(f"Anthropic request failed: {exc}")) from exc

    def _reply_text(self, response) -> str:
        """The first text block of a response, or empty when it has none."""
        return next((b.text for b in response.content if b.type == "text"), "")

    def complete(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
        context_length: int | None = None,  # unused for Anthropic; accepted for interface compatibility
        cancel: "CancelToken | None" = None,
    ) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        non_system = [m for m in messages if m["role"] != "system"]
        response = self._request(
            non_system, system=system, max_tokens=max_tokens, cancel=cancel
        )
        return self._reply_text(response)

    def summarize(
        self,
        title: str,
        source: "str | Path",
        max_tokens: int = 2048,
        cancel: "CancelToken | None" = None,
    ) -> str:
        prompt = _get_summary_prompt().replace("{title}", title)
        if isinstance(source, Path):
            pdf_b64 = base64.b64encode(source.read_bytes()).decode()
            content: list[dict] = [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_b64,
                    },
                },
                {"type": "text", "text": prompt},
            ]
        else:
            content = [{"type": "text", "text": f"{prompt}\n\nAbstract/text:\n{source}"}]

        messages = [{"role": "user", "content": content}]
        response = self._request(messages, max_tokens=max_tokens, cancel=cancel)
        return self._reply_text(response)

    def agentic_turn(
        self,
        messages: list[dict],
        tools: list[dict],
        dispatch_fn: Callable[[str, dict], str],
        system: str = "",
        cancel: "CancelToken | None" = None,
    ) -> str:
        anthropic_tools = _convert_tools_to_anthropic(tools)
        wire = transcript.to_anthropic(messages, model=self.model)
        # Everything from here on is appended to the wire list; only this tail
        # is converted back, so history from another provider stays untouched.
        turn_start = len(wire)

        def commit() -> None:
            messages.extend(transcript.from_anthropic(wire[turn_start:], model=self.model))

        while True:
            # Checked before the request so a turn stopped between iterations
            # never sends another one. Every raise below leaves `messages`
            # untouched, because commit() is only reached on a return.
            if cancel is not None:
                cancel.check()

            response = self._request(
                wire, system=system, tools=anthropic_tools, max_tokens=4096, cancel=cancel
            )

            if response.stop_reason == "tool_use":
                wire.append({"role": "assistant", "content": [_block_to_dict(b) for b in response.content]})
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    if cancel is not None:
                        cancel.check()
                    try:
                        result = dispatch_fn(block.name, block.input)
                    except PrivacyError as exc:
                        # Drop the assistant message we just added — with its
                        # tool calls unanswered it would leave the transcript
                        # invalid — and stop without another LLM call.
                        wire.pop()
                        commit()
                        return str(exc)
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
                # All results for one assistant turn go in a SINGLE user
                # message; separate messages are a 400 from this API.
                wire.append({"role": "user", "content": tool_results})
                continue

            reply = self._reply_text(response)
            if response.stop_reason == "end_turn":
                wire.append({"role": "assistant", "content": [_block_to_dict(b) for b in response.content]})
            else:
                wire.append({"role": "assistant", "content": reply})
            commit()
            return reply

    def describe_image(
        self,
        image_bytes: bytes,
        context: str,
        cancel: "CancelToken | None" = None,
    ) -> str:
        prompt = _FIGURE_CAPTION_PROMPT.format(context=context or "untitled document")
        # PDF figures are extracted as PNG bytes (see jarvis/kb/images.py), so
        # the media type is fixed.
        image_b64 = base64.b64encode(image_bytes).decode()
        content = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": image_b64},
            },
            {"type": "text", "text": prompt},
        ]
        response = self._request(
            [{"role": "user", "content": content}], max_tokens=1024, cancel=cancel
        )
        return self._reply_text(response)

    def pop_usage(self) -> "dict | None":
        """
        No cost reported. The API returns token counts, but turning those into
        a number would need a per-model price table that ages silently — see
        the [openrouter] path for the one provider that reports real spend.
        """
        return None


# ── OpenRouter adapter ─────────────────────────────────────────────────────────


def _redact(text: str) -> str:
    """
    Strip any configured API key out of an error message.

    An SDK can quote the credential it failed with, and the caller writes that
    text to the user's screen, the chat log, and the saved session — so it has
    to be scrubbed here, at the point the exception becomes a message, rather
    than at each of the places it lands.
    """
    from .config import redact_secrets

    return redact_secrets(text)


def _openrouter_extra_body(cfg: "Config") -> dict:
    """
    Provider-routing preferences sent with every OpenRouter request.

    OpenRouter is a broker: without these, a request can land at any upstream
    inference provider it has a route for, including ones that train on what
    they are sent. The defaults are the strict ones (see [openrouter] in
    config.toml) and the user can loosen them deliberately.
    """
    routing: dict = {
        "data_collection": cfg.openrouter_data_collection,
        "allow_fallbacks": cfg.openrouter_allow_fallbacks,
    }
    if cfg.openrouter_only:
        routing["only"] = list(cfg.openrouter_only)
    return {
        "provider": routing,
        # Ask for real spend on every response — this is the only provider
        # jarvis can report a cost for without inventing one.
        "usage": {"include": True},
    }


@dataclass
class _StreamedToolCall:
    """One tool call reassembled from an OpenAI-style stream."""

    id: str = ""
    name: str = ""
    # Arguments arrive as JSON split across chunks, so they accumulate as text
    # and are parsed once the whole call has landed.
    arguments: str = ""


@dataclass
class _StreamedCompletion:
    """
    One OpenRouter request's result, reassembled from its stream.

    Deliberately not shaped like the SDK's response object: the callers below
    read these four fields directly, which is easier to follow than mimicking
    `choices[0].message` and lets the cost and served-model fields sit where
    they are actually used.
    """

    content: str = ""
    tool_calls: list = field(default_factory=list)
    cost: "float | None" = None
    model: str = ""


def _accumulate_openai_stream(stream, cancel: "CancelToken | None") -> _StreamedCompletion:
    """
    Fold an OpenAI-style chat completion stream into one result.

    Content arrives as deltas to concatenate. Tool calls arrive piecemeal too
    and are keyed by `index`: the id and function name come once, while the
    arguments JSON is split across as many chunks as it takes, so each field is
    only overwritten when the chunk actually carries it. Usage and the served
    model ride on a final chunk that has no choices at all.

    The cancel token is checked per chunk; raising here unwinds through the
    caller's `finally`, which closes the stream and drops the connection.
    """
    completion = _StreamedCompletion()
    calls_by_index: dict = {}

    for chunk in stream:
        if cancel is not None:
            cancel.check()

        if getattr(chunk, "model", ""):
            completion.model = chunk.model
        usage = getattr(chunk, "usage", None)
        if usage is not None and getattr(usage, "cost", None) is not None:
            completion.cost = float(usage.cost)

        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue  # the usage-only chunk that closes the stream
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue

        if getattr(delta, "content", None):
            completion.content += delta.content

        for part in getattr(delta, "tool_calls", None) or []:
            index = getattr(part, "index", 0) or 0
            call = calls_by_index.setdefault(index, _StreamedToolCall())
            if getattr(part, "id", None):
                call.id = part.id
            function = getattr(part, "function", None)
            if function is not None:
                if getattr(function, "name", None):
                    call.name = function.name
                if getattr(function, "arguments", None):
                    call.arguments += function.arguments

    completion.tool_calls = [calls_by_index[i] for i in sorted(calls_by_index)]
    return completion


class OpenRouterProvider:
    """
    Any model reachable through OpenRouter, spoken over the OpenAI wire format.

    Two deliberate omissions: the optional HTTP-Referer / X-Title headers
    (they exist to list your app on OpenRouter's public leaderboards, so
    sending them would be telemetry by another name), and PDF upload —
    summarize() converts locally with pymupdf4llm and sends text, so nothing
    leaves the machine that the user's own converter did not produce.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = None
        self._usd = 0.0
        self._requests = 0
        # What OpenRouter actually served. Only interesting for a router model
        # like `openrouter/auto`, where the configured name is a request and
        # the response names the model that answered it.
        self._served = ""

    def _get_client(self):
        if self._client is not None:
            return self._client

        import os

        from openai import OpenAI

        from .config import get_config
        api_key = os.environ.get("OPENROUTER_API_KEY") or get_config().openrouter_api_key
        if not api_key:
            raise AuthenticationError(
                "No OpenRouter credentials found.\n"
                "  Set OPENROUTER_API_KEY env var or add openrouter_api_key to "
                "[auth] in ~/.jarvis/config.toml"
            )
        self._client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        return self._client

    def _create(
        self,
        cancel: "CancelToken | None" = None,
        **kwargs,
    ) -> _StreamedCompletion:
        """
        One place for the shared request options, cost accounting, and errors.

        The request is streamed for the same reason the other two adapters
        stream: it is the only way to interrupt one. The token is checked
        before the request and again between chunks, and closing the stream on
        the way out drops the connection, which is what stops the upstream
        model generating.

        Cost still arrives: `_openrouter_extra_body` already asks for usage
        accounting, and OpenRouter puts it in the final SSE chunk. The
        OpenAI-standard `stream_options={"include_usage": True}` would ask for
        the same thing a second way, so it is deliberately not sent — a
        redundant parameter buys nothing and is one more thing an upstream
        provider could reject outright.
        """
        from .config import get_config

        if cancel is not None:
            cancel.check()
        client = self._get_client()
        try:
            stream = client.chat.completions.create(
                model=self.model,
                extra_body=_openrouter_extra_body(get_config()),
                stream=True,
                **kwargs,
            )
            try:
                completion = _accumulate_openai_stream(stream, cancel)
            finally:
                # On a cancel this is the kill signal; on normal completion the
                # stream is already exhausted and closing is a no-op.
                close_stream = getattr(stream, "close", None)
                if close_stream is not None:
                    close_stream()
        except (TurnCancelled, KeyboardInterrupt):
            raise
        except Exception as exc:
            raise LLMError(_redact(f"OpenRouter request failed: {exc}")) from exc
        self._record_usage(completion)
        return completion

    def _record_usage(self, completion: _StreamedCompletion) -> None:
        """Accumulate the credits OpenRouter reports, and note who answered."""
        self._requests += 1
        if completion.cost is not None:
            self._usd += completion.cost
        if completion.model:
            self._served = completion.model

    def pop_usage(self) -> "dict | None":
        """
        Spend since the last call, then reset.

        `model` is the spec that actually ran, which is the same thing you
        asked for unless you asked for a router. With `openrouter/auto` the
        request names the router and the response names the model, so keying
        cost by this is what makes a session's spend break down by the models
        that really answered.
        """
        if not self._requests:
            return None
        usage = {"usd": round(self._usd, 6), "requests": self._requests}
        if self._served:
            usage["model"] = f"openrouter:{self._served}"
        self._usd = 0.0
        self._requests = 0
        return usage

    def complete(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
        context_length: int | None = None,  # Ollama-only knob; accepted for interface compatibility
        cancel: "CancelToken | None" = None,
    ) -> str:
        return self._create(messages=messages, max_tokens=max_tokens, cancel=cancel).content

    def summarize(
        self,
        title: str,
        source: "str | Path",
        max_tokens: int = 2048,
        cancel: "CancelToken | None" = None,
    ) -> str:
        prompt = _get_summary_prompt().replace("{title}", title)
        if isinstance(source, Path):
            # Convert locally rather than uploading the file: the conversion is
            # cheap (pymupdf4llm, no ML models) and it keeps the request to
            # plain text.
            from jarvis.kb.convert import pdf_to_markdown

            text = pdf_to_markdown(source)
        else:
            text = source
        messages = [{"role": "user", "content": f"{prompt}\n\nAbstract/text:\n{text}"}]
        return self.complete(messages, max_tokens=max_tokens, cancel=cancel)

    def agentic_turn(
        self,
        messages: list[dict],
        tools: list[dict],
        dispatch_fn: Callable[[str, dict], str],
        system: str = "",
        cancel: "CancelToken | None" = None,
    ) -> str:
        wire = transcript.to_openai(messages, provider="openrouter", model=self.model)
        # Everything from here on is appended to the wire list; only this tail
        # is converted back, so history from another provider stays untouched.
        turn_start = len(wire)

        def commit() -> None:
            messages.extend(
                transcript.from_openai(
                    wire[turn_start:], provider="openrouter", model=self.model
                )
            )

        while True:
            # Checked before the request so a turn stopped between iterations
            # never sends another one. Every raise below leaves `messages`
            # untouched, because commit() is only reached on a return.
            full = ([{"role": "system", "content": system}] + wire) if system else wire
            completion = self._create(messages=full, tools=tools or None, cancel=cancel)
            tool_calls = completion.tool_calls

            if not tool_calls:
                reply = completion.content
                wire.append({"role": "assistant", "content": reply})
                commit()
                return reply

            wire.append(
                {
                    "role": "assistant",
                    "content": completion.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": call.arguments},
                        }
                        for call in tool_calls
                    ],
                }
            )
            for call in tool_calls:
                if cancel is not None:
                    cancel.check()
                arguments = transcript._arguments_to_dict(call.arguments)
                try:
                    result = dispatch_fn(call.name, arguments)
                except PrivacyError as exc:
                    # Drop the assistant message we just added — with its tool
                    # calls unanswered it would leave the transcript invalid —
                    # and stop without another LLM call.
                    wire.pop()
                    commit()
                    return str(exc)
                # The OpenAI wire wants one tool message per call, keyed by id.
                wire.append({"role": "tool", "tool_call_id": call.id, "content": result})

    def describe_image(
        self,
        image_bytes: bytes,
        context: str,
        cancel: "CancelToken | None" = None,
    ) -> str:
        prompt = _FIGURE_CAPTION_PROMPT.format(context=context or "untitled document")
        # PDF figures are extracted as PNG bytes (see jarvis/kb/images.py), so
        # the media type is fixed.
        image_b64 = base64.b64encode(image_bytes).decode()
        response = self._create(
            cancel=cancel,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return response.content


# ── Factory ───────────────────────────────────────────────────────────────────


PROVIDERS = ("ollama", "anthropic", "openrouter")


def split_spec(spec: str) -> tuple[str, str]:
    """
    Split "provider" or "provider:model" into its two parts.

    The model half may itself contain slashes and dots (OpenRouter names look
    like "anthropic/claude-sonnet-4.6"), so only the first colon separates.
    """
    provider, _, model = spec.partition(":")
    return provider, model


# Which Config field holds each provider's default model.
_MODEL_FIELD = {
    "anthropic": "anthropic_model",
    "ollama": "ollama_model",
    "openrouter": "openrouter_model",
}


def default_model(provider: str, cfg: "Config") -> str:
    """The configured model for a provider when a spec names no model."""
    return getattr(cfg, _MODEL_FIELD[provider], "")


def make_provider(
    spec: str = "ollama",
    model: str | None = None,
) -> "OllamaProvider | AnthropicProvider | OpenRouterProvider":
    """
    Construct a ChatProvider from a spec string.

    spec:
      "ollama"                 → OllamaProvider with config ollama_model
      "anthropic"              → AnthropicProvider with config anthropic_model
      "openrouter"             → OpenRouterProvider with config openrouter_model
      "<provider>:<model>"     → any of the above with that model, which is how
                                 a session records the model it is running on

    An explicit `model` argument wins over a model named in the spec.
    """
    from .config import get_config

    cfg = get_config()
    provider, spec_model = split_spec(spec)
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider spec: {spec!r} (expected one of {', '.join(PROVIDERS)})"
        )

    chosen = model or spec_model or default_model(provider, cfg)
    if not chosen:
        raise ValueError(
            f"No model configured for provider {provider!r} — name one in the spec "
            f"('{provider}:<model>') or set {provider}_model in ~/.jarvis/config.toml"
        )

    if provider == "anthropic":
        return AnthropicProvider(model=chosen)
    if provider == "openrouter":
        return OpenRouterProvider(model=chosen)
    return OllamaProvider(model=chosen)


def active_model(cfg: "Config") -> str:
    """Return the model name jarvis will actually use for cfg.provider."""
    provider, spec_model = split_spec(cfg.provider)
    if provider not in PROVIDERS:
        return spec_model
    return spec_model or default_model(provider, cfg)
