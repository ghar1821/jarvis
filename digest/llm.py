"""
LLM provider abstraction.

Two concrete adapters satisfy the ChatProvider protocol:
  OllamaProvider   — local Ollama model
  AnthropicProvider — Anthropic Claude (API key or OAuth token)

Both implement:
  complete(messages, max_tokens, context_length)  — single-shot text completion
  summarize(title, source)                         — paper summary, PDF-aware
  agentic_turn(messages, tools, dispatch_fn, system) — full tool-calling loop

Use make_provider() to construct the right adapter from a spec string:
  make_provider("ollama")        → OllamaProvider using config ollama_model
  make_provider("anthropic")     → AnthropicProvider using config anthropic_model
  make_provider("gemma4:27b")    → OllamaProvider with that specific model
"""

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from .errors import AuthenticationError, LLMError

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
        PDF files are sent directly to the model — no pre-conversion needed,
        provided the model supports document input.
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


# ── Prompt loading ─────────────────────────────────────────────────────────────

_SUMMARY_PROMPT: str | None = None


def _get_summary_prompt() -> str:
    global _SUMMARY_PROMPT
    if _SUMMARY_PROMPT is None:
        _SUMMARY_PROMPT = (
            Path(__file__).parent / "prompts" / "paper_summary.md"
        ).read_text()
    return _SUMMARY_PROMPT


# ── Ollama adapter ─────────────────────────────────────────────────────────────


class OllamaProvider:
    def __init__(self, model: str, options: dict | None = None) -> None:
        self.model = model
        self.options = options or {}

    def complete(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
        context_length: int | None = None,
    ) -> str:
        import ollama

        opts = dict(self.options)
        if context_length:
            opts["num_ctx"] = context_length

        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options=opts or None,
            )
            return response["message"]["content"]
        except Exception as exc:
            raise LLMError(f"Ollama complete failed: {exc}") from exc

    def summarize(self, title: str, source: "str | Path", max_tokens: int = 2048) -> str:
        prompt = _get_summary_prompt().replace("{title}", title)
        if isinstance(source, Path):
            pdf_b64 = base64.b64encode(source.read_bytes()).decode()
            messages: list[dict] = [{"role": "user", "content": prompt, "images": [pdf_b64]}]
        else:
            messages = [{"role": "user", "content": f"{prompt}\n\nAbstract/text:\n{source}"}]
        return self.complete(messages, max_tokens=max_tokens)

    def agentic_turn(
        self,
        messages: list[dict],
        tools: list[dict],
        dispatch_fn: Callable[[str, dict], str],
        system: str = "",
    ) -> str:
        import ollama

        client = ollama.Client()
        while True:
            full = ([{"role": "system", "content": system}] + messages) if system else messages
            try:
                response = client.chat(model=self.model, messages=full, tools=tools, format=None)
            except ollama.ResponseError as exc:
                raise LLMError(f"Ollama agentic turn failed: {exc}") from exc

            message = response["message"]
            if not message.get("tool_calls"):
                reply = message["content"]
                messages.append({"role": "assistant", "content": reply})
                return reply

            messages.append(message)
            for tc in message["tool_calls"]:
                fn = tc["function"]
                result = dispatch_fn(fn["name"], fn["arguments"])
                messages.append({"role": "tool", "content": result})


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
    def __init__(self, model: str, auth_file: Path | None = None) -> None:
        self.model = model
        self._auth_file = auth_file or Path.home() / ".paper_digest" / "auth.json"
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        import os

        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            self._client = anthropic.Anthropic(api_key=api_key)
            return self._client

        if self._auth_file.exists():
            auth = json.loads(self._auth_file.read_text())
            token = auth.get("access_token", "")
            expires_at = auth.get("expires_at", "")
            if expires_at:
                from datetime import datetime, timezone

                try:
                    if datetime.now(timezone.utc) > datetime.fromisoformat(expires_at):
                        token = self._refresh_token(auth)
                except ValueError:
                    pass
            if token:
                self._client = anthropic.Anthropic(auth_token=token)
                return self._client

        raise AuthenticationError(
            "No Anthropic credentials found.\n"
            "  Option 1: export ANTHROPIC_API_KEY=sk-ant-...\n"
            "  Option 2: paper-rag auth login"
        )

    def _refresh_token(self, auth: dict) -> str:
        import os

        import requests

        from .config import get_config

        cfg = get_config()
        resp = requests.post(
            cfg.oauth_token_url,
            data={
                "grant_type": "refresh_token",
                "refresh_token": auth.get("refresh_token", ""),
                "client_id": cfg.oauth_client_id or os.environ.get("ANTHROPIC_OAUTH_CLIENT_ID", ""),
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        resp.raise_for_status()
        token_data = resp.json()

        from datetime import datetime, timezone

        expires_in = token_data.get("expires_in", 3600)
        auth["access_token"] = token_data["access_token"]
        auth["expires_at"] = datetime.fromtimestamp(
            datetime.now(timezone.utc).timestamp() + expires_in, tz=timezone.utc
        ).isoformat()
        if token_data.get("refresh_token"):
            auth["refresh_token"] = token_data["refresh_token"]
        self._auth_file.write_text(json.dumps(auth, indent=2))
        return token_data["access_token"]

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
                    result = dispatch_fn(block.name, block.input)
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
                messages.append({"role": "user", "content": tool_results})

            else:
                reply = next((b.text for b in response.content if b.type == "text"), "")
                messages.append({"role": "assistant", "content": reply})
                return reply


# ── Factory ───────────────────────────────────────────────────────────────────


def make_provider(
    spec: str = "ollama",
    model: str | None = None,
    options: dict | None = None,
) -> "OllamaProvider | AnthropicProvider":
    """
    Construct a ChatProvider from a spec string.

    spec:
      "anthropic"      → AnthropicProvider with config anthropic_model (or model override)
      "ollama"         → OllamaProvider with config ollama_model (or model override)
      "<model name>"   → OllamaProvider with that specific model (e.g. "gemma4:27b")
    """
    from .config import get_config

    cfg = get_config()

    if spec == "anthropic":
        return AnthropicProvider(model=model or cfg.anthropic_model, auth_file=cfg.auth_file)

    # "ollama" or a bare model name
    ollama_model = model or (cfg.ollama_model if spec == "ollama" else spec)
    return OllamaProvider(model=ollama_model, options=options or {})
