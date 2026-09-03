"""
LLM provider abstraction.

Two concrete adapters satisfy the ChatProvider protocol:
  OllamaProvider    — local model served by Ollama (http://localhost:11434)
  AnthropicProvider — Anthropic Claude (API key)

Both implement:
  complete(messages, max_tokens, context_length)  — single-shot text completion
  summarize(title, source)                         — paper summary, PDF-aware
  agentic_turn(messages, tools, dispatch_fn, system) — full tool-calling loop
  describe_image(image_bytes, context)             — caption a PDF figure

Every request goes out streamed, through each adapter's single _request()
helper. Streaming is not about showing tokens as they arrive (replies are still
delivered whole) — it is what makes a turn interruptible. A blocking
non-streaming call offers no moment to bail out, whereas a stream can be
checked between events and closed part-way, and closing the connection is what
tells the cloud or Ollama server to stop generating. Every method therefore
accepts an optional cancel token; see jarvis/core/cancel.py.

Use make_provider() to construct the right adapter from a spec string:
  make_provider("ollama")     → OllamaProvider using config ollama_model
  make_provider("anthropic")  → AnthropicProvider using config anthropic_model

Ollama must be running (the macOS login-item app or `ollama serve`). For full
functionality the configured model needs tool-calling and vision support —
figure captioning and vision-based summaries depend on the vision capability.
"""

import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

from .errors import AuthenticationError, LLMError, PrivacyError, TurnCancelled

if TYPE_CHECKING:
    from .cancel import CancelToken
    from .config import Config

# ── Protocol ───────────────────────────────────────────────────────────────────


@runtime_checkable
class ChatProvider(Protocol):
    """
    Every method takes an optional cancel token. When one is supplied and the
    human stops the turn, the method raises TurnCancelled instead of returning
    — see each adapter's _request() for where the checks sit. Callers that
    can't be interrupted (the digest pipeline, the sync daemon, the kb CLI)
    simply leave it None.
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

        messages is appended to as the loop runs (tool calls and results), so
        callers that may need to abandon a turn should pass a working copy and
        commit it to the session only once this returns — a cancelled turn must
        leave no half-finished exchange behind. What this method guarantees is
        that once the turn is cancelled, messages is never appended to again:
        the cancel checks sit before each request and before each append, so an
        assistant tool_use block is never left without its matching tool_result.

        dispatch_fn(tool_name, arguments) -> result_string
        Returns the final text reply.
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


# ── Prompt loading ─────────────────────────────────────────────────────────────

_SUMMARY_PROMPT: str | None = None

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
    global _SUMMARY_PROMPT
    if _SUMMARY_PROMPT is None:
        _SUMMARY_PROMPT = (
            Path(__file__).parent.parent / "kb" / "prompts" / "paper_summary.md"
        ).read_text()
    return _SUMMARY_PROMPT


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
        pydantic object) keeps session history JSON-serialisable for free, and
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
            raise LLMError(f"Ollama request failed: {exc}") from exc

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
        message = self._request(messages, options=options, cancel=cancel)
        return message["content"]

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
        while True:
            # Checked before the request so a turn stopped between iterations
            # never sends another one.
            if cancel is not None:
                cancel.check()

            full = ([{"role": "system", "content": system}] + messages) if system else messages
            message = self._request(full, tools=tools, cancel=cancel)

            # Checked again before anything is appended: a stopped turn must
            # leave `messages` exactly as it found it, or the next turn replays
            # an assistant tool_use with no matching result.
            if cancel is not None:
                cancel.check()

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                reply = message["content"]
                messages.append({"role": "assistant", "content": reply})
                return reply

            messages.append(message)
            results = []
            try:
                for call in tool_calls:
                    if cancel is not None:
                        cancel.check()
                    name = call["function"]["name"]
                    result = dispatch_fn(name, dict(call["function"]["arguments"]))
                    results.append({"role": "tool", "tool_name": name, "content": result})
            except PrivacyError as exc:
                # Remove the assistant message we just added so the
                # conversation history stays in a valid state for future turns.
                messages.pop()
                return str(exc)
            except (TurnCancelled, KeyboardInterrupt):
                # Same reasoning as the PrivacyError path above — an assistant
                # tool call left without its result would poison the next turn.
                messages.pop()
                raise
            messages.extend(results)

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
        HTTP response, which is how Anthropic is told to stop generating
        (the Messages API has no cancel endpoint — disconnecting is the signal).
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
            raise LLMError(f"Anthropic request failed: {exc}") from exc

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
        while True:
            # Checked before the request so a turn stopped between iterations
            # never sends another one.
            if cancel is not None:
                cancel.check()

            response = self._request(
                messages, system=system, tools=anthropic_tools, max_tokens=4096, cancel=cancel
            )

            # Checked again before anything is appended. This is the one that
            # protects the wire format: an assistant tool_use message appended
            # without its matching tool_result bundle makes the next turn 400.
            if cancel is not None:
                cancel.check()

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": [_block_to_dict(b) for b in response.content]})
                tool_results = []
                try:
                    for block in response.content:
                        if block.type != "tool_use":
                            continue
                        if cancel is not None:
                            cancel.check()
                        result = dispatch_fn(block.name, block.input)
                        tool_results.append(
                            {"type": "tool_result", "tool_use_id": block.id, "content": result}
                        )
                except PrivacyError as exc:
                    # Remove the assistant message we just added so the
                    # conversation history stays in a valid state for future turns.
                    messages.pop()
                    return str(exc)
                except (TurnCancelled, KeyboardInterrupt):
                    # Same reasoning as the PrivacyError path above: the
                    # tool_use blocks we appended would never get their
                    # tool_result bundle, so drop them before unwinding.
                    messages.pop()
                    raise
                messages.append({"role": "user", "content": tool_results})
                continue

            reply = self._reply_text(response)
            if response.stop_reason == "end_turn":
                messages.append({"role": "assistant", "content": [_block_to_dict(b) for b in response.content]})
            else:
                messages.append({"role": "assistant", "content": reply})
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


# ── Factory ───────────────────────────────────────────────────────────────────


def make_provider(
    spec: str = "ollama",
    model: str | None = None,
) -> "OllamaProvider | AnthropicProvider":
    """
    Construct a ChatProvider from a spec string.

    spec:
      "anthropic" → AnthropicProvider with config anthropic_model (or model override)
      "ollama"    → OllamaProvider with config ollama_model (or model override)
    """
    from .config import get_config

    cfg = get_config()

    if spec == "anthropic":
        return AnthropicProvider(model=model or cfg.anthropic_model)
    if spec == "ollama":
        return OllamaProvider(model=model or cfg.ollama_model)
    raise ValueError(f"Unknown provider spec: {spec!r} (expected 'ollama' or 'anthropic')")


def active_model(cfg: "Config") -> str:
    """Return the model name jarvis will actually use for cfg.provider."""
    return cfg.anthropic_model if cfg.provider == "anthropic" else cfg.ollama_model
