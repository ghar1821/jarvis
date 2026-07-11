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
from typing import Callable, Protocol, runtime_checkable

from .errors import AuthenticationError, LLMError, PrivacyError

# ── Protocol ───────────────────────────────────────────────────────────────────


@runtime_checkable
class ChatProvider(Protocol):
    def complete(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
        context_length: int | None = None,
    ) -> str:
        """Single-shot text completion. A system message may be included in messages."""
        ...

    def summarize(self, title: str, source: "str | Path", max_tokens: int = 2048) -> str:
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
    ) -> str:
        """
        Run a full agentic turn including tool dispatch loop.

        messages is modified in place (tool calls and results are appended).
        dispatch_fn(tool_name, arguments) -> result_string
        Returns the final text reply.
        """
        ...

    def describe_image(self, image_bytes: bytes, context: str) -> str:
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


def _message_to_dict(message) -> dict:
    """Normalise an ollama pydantic Message to a JSON-serialisable dict."""
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    return message


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

    def complete(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
        context_length: int | None = None,
    ) -> str:
        client = self._get_client()
        # Ollama honours a per-request context window; only set it when asked.
        options = {"num_ctx": context_length} if context_length else {}
        try:
            response = client.chat(model=self.model, messages=messages, options=options)
            return response["message"]["content"] or ""
        except Exception as exc:
            raise LLMError(f"Ollama complete failed: {exc}") from exc

    def summarize(self, title: str, source: "str | Path", max_tokens: int = 2048) -> str:
        prompt = _get_summary_prompt().replace("{title}", title)
        if isinstance(source, Path):
            # Ollama has no document-input API, and the conversion is cheap
            # (pymupdf4llm, no ML models), so feed it the markdown text.
            from jarvis.kb.convert import pdf_to_markdown

            text = pdf_to_markdown(source)
        else:
            text = source
        messages = [{"role": "user", "content": f"{prompt}\n\nAbstract/text:\n{text}"}]
        return self.complete(messages, max_tokens=max_tokens)

    def agentic_turn(
        self,
        messages: list[dict],
        tools: list[dict],
        dispatch_fn: Callable[[str, dict], str],
        system: str = "",
    ) -> str:
        client = self._get_client()
        while True:
            full = ([{"role": "system", "content": system}] + messages) if system else messages
            try:
                response = client.chat(model=self.model, messages=full, tools=tools)
            except Exception as exc:
                raise LLMError(f"Ollama agentic turn failed: {exc}") from exc

            message = response["message"]
            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                reply = message.get("content") or ""
                messages.append({"role": "assistant", "content": reply})
                return reply

            # The ollama client returns a pydantic Message — normalise it to a
            # plain dict so session history stays JSON-serialisable.
            messages.append(_message_to_dict(message))
            for tc in tool_calls:
                # Ollama hands back arguments as a mapping already (not a JSON
                # string like the OpenAI wire format), so use it directly; only
                # parse if some model variant ever returns a string.
                arguments = tc.function.arguments
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                try:
                    result = dispatch_fn(tc.function.name, dict(arguments))
                except PrivacyError as exc:
                    # Remove the assistant message we just added so the
                    # conversation history stays in a valid state for future turns.
                    messages.pop()
                    return str(exc)
                messages.append(
                    {"role": "tool", "tool_name": tc.function.name, "content": result}
                )

    def describe_image(self, image_bytes: bytes, context: str) -> str:
        client = self._get_client()
        prompt = _FIGURE_CAPTION_PROMPT.format(context=context or "untitled document")
        try:
            response = client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt, "images": [image_bytes]}],
            )
            return response["message"]["content"] or ""
        except Exception as exc:
            raise LLMError(f"Ollama describe_image failed: {exc}") from exc


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

    def complete(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
        context_length: int | None = None,  # unused for Anthropic; accepted for interface compatibility
    ) -> str:
        client = self._get_client()
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        non_system = [m for m in messages if m["role"] != "system"]
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=non_system,
            )
            return next((b.text for b in response.content if b.type == "text"), "")
        except Exception as exc:
            raise LLMError(f"Anthropic complete failed: {exc}") from exc

    def summarize(self, title: str, source: "str | Path", max_tokens: int = 2048) -> str:
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
        client = self._get_client()
        try:
            response = client.messages.create(
                model=self.model, max_tokens=max_tokens, messages=messages
            )
            return next((b.text for b in response.content if b.type == "text"), "")
        except Exception as exc:
            raise LLMError(f"Anthropic summarize failed: {exc}") from exc

    def agentic_turn(
        self,
        messages: list[dict],
        tools: list[dict],
        dispatch_fn: Callable[[str, dict], str],
        system: str = "",
    ) -> str:
        client = self._get_client()
        anthropic_tools = _convert_tools_to_anthropic(tools)
        while True:
            try:
                response = client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system,
                    messages=messages,
                    tools=anthropic_tools,
                )
            except Exception as exc:
                raise LLMError(f"Anthropic agentic turn failed: {exc}") from exc

            if response.stop_reason == "end_turn":
                reply = next((b.text for b in response.content if b.type == "text"), "")
                messages.append({"role": "assistant", "content": [_block_to_dict(b) for b in response.content]})
                return reply

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": [_block_to_dict(b) for b in response.content]})
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    try:
                        result = dispatch_fn(block.name, block.input)
                    except PrivacyError as exc:
                        # Remove the assistant message we just added so the
                        # conversation history stays in a valid state for future turns.
                        messages.pop()
                        return str(exc)
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
                messages.append({"role": "user", "content": tool_results})

            else:
                reply = next((b.text for b in response.content if b.type == "text"), "")
                messages.append({"role": "assistant", "content": reply})
                return reply

    def describe_image(self, image_bytes: bytes, context: str) -> str:
        client = self._get_client()
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
        try:
            response = client.messages.create(
                model=self.model, max_tokens=1024,
                messages=[{"role": "user", "content": content}],
            )
            return next((b.text for b in response.content if b.type == "text"), "")
        except Exception as exc:
            raise LLMError(f"Anthropic describe_image failed: {exc}") from exc


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
