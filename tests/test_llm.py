"""
Tests for jarvis/core/llm.py — provider adapters.

Integration tests (marked, skipped by default) verify that live services are
reachable: the Anthropic API (key check, no tokens) and the local Ollama server.

Unit tests exercise the agentic_turn tool loop with the LLM client mocked at
the API boundary — the one place CLAUDE.md sanctions mocking, since real calls
bill per token / need a running model server. They pin down the PrivacyError
contract both providers must honour: return the error text, restore message
history exactly, and make no further LLM call.

Running
-------
    uv run pytest -m integration          # integration tests only
    uv run pytest -m "not integration"    # unit tests only (default CI run)
    uv run pytest                         # all tests
"""

import json
from types import SimpleNamespace

import pytest

from jarvis.core.errors import PrivacyError
from jarvis.core.llm import AnthropicProvider, OllamaProvider, make_provider

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }
]


# ── Integration: connectivity ──────────────────────────────────────────────────


@pytest.mark.integration
def test_anthropic_client_initialises():
    """
    AnthropicProvider._get_client() returns a client when a valid API key is
    available. Raises AuthenticationError if no key is found in the env var or
    config file.
    """
    from jarvis.core.config import get_config

    cfg = get_config()
    provider = AnthropicProvider(model=cfg.anthropic_model)
    assert provider._get_client() is not None


@pytest.mark.integration
def test_anthropic_models_list_confirms_auth():
    """
    client.models.list() makes a real API call (GET /v1/models) that validates
    the API key without generating output or consuming tokens.
    """
    from jarvis.core.config import get_config

    cfg = get_config()
    provider = AnthropicProvider(model=cfg.anthropic_model)
    models = list(provider._get_client().models.list())
    assert len(models) > 0


@pytest.mark.integration
def test_ollama_is_reachable():
    """
    ollama.list() makes a real call to the local Ollama server, confirming it
    is running and has at least one model pulled. No inference, no cost.

    Input:  running Ollama server at http://localhost:11434
    Expected output: a non-empty model list
    """
    import ollama

    models = ollama.list()["models"]
    assert len(models) > 0


# ── Unit: factory ──────────────────────────────────────────────────────────────


def test_make_provider_dispatches_on_spec():
    """
    make_provider maps 'ollama' and 'anthropic' to their adapters and rejects
    anything else.
    """
    assert isinstance(make_provider("ollama"), OllamaProvider)
    assert isinstance(make_provider("anthropic"), AnthropicProvider)
    with pytest.raises(ValueError, match="Unknown provider"):
        make_provider("llamacpp")


# ── Unit: OllamaProvider.agentic_turn ──────────────────────────────────────────


class _OllamaMessage:
    """
    Minimal stand-in for the ollama client's pydantic Message object. Ollama
    hands tool_calls back as objects whose function.arguments is already a
    dict (not a JSON string), and the message normalises via model_dump().
    """

    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def get(self, key, default=None):
        return getattr(self, key, default)

    def model_dump(self, exclude_none=False):
        dumped = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            dumped["tool_calls"] = [
                {"function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in self.tool_calls
            ]
        return dumped


def _ollama_tool_call(name: str, arguments: dict):
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


class _FakeOllamaClient:
    """Replays a scripted sequence of responses; records every request."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        message = self._responses.pop(0)
        return {"message": message}


def test_ollama_agentic_turn_dispatches_tools_with_dict_args():
    """
    Ollama returns tool arguments as a mapping already, so they reach dispatch
    as a dict without any JSON parsing; results feed back as role='tool'
    messages and the final text is returned.

    Input:  scripted tool_use response then a text response
    Expected output: dispatch got dict args; reply text returned; history has
            the assistant tool-call dict + tool result + final assistant text
    """
    provider = OllamaProvider(model="test-model")
    provider._client = _FakeOllamaClient(
        [
            _OllamaMessage(tool_calls=[_ollama_tool_call("read_file", {"path": "notes.md"})]),
            _OllamaMessage(content="Here is the summary."),
        ]
    )

    dispatched = []

    def dispatch_fn(name, args):
        dispatched.append((name, args))
        return "file contents"

    messages = [{"role": "user", "content": "read notes.md"}]
    reply = provider.agentic_turn(messages, TOOLS, dispatch_fn, system="be helpful")

    assert reply == "Here is the summary."
    assert dispatched == [("read_file", {"path": "notes.md"})]
    assert messages[1]["tool_calls"][0]["function"]["name"] == "read_file"
    assert messages[2] == {"role": "tool", "tool_name": "read_file", "content": "file contents"}
    assert messages[3] == {"role": "assistant", "content": "Here is the summary."}


def test_ollama_agentic_turn_privacy_error_stops_cleanly():
    """
    PrivacyError from a tool must end the turn: the error text is the reply,
    the orphaned assistant tool-call message is popped so history is exactly
    the original user turn, and no second LLM call happens.

    Input:  one tool_use response; dispatch raises PrivacyError
    Expected output: reply == error text; messages unchanged; one API call
    """
    provider = OllamaProvider(model="test-model")
    client = _FakeOllamaClient(
        [_OllamaMessage(tool_calls=[_ollama_tool_call("read_file", {"path": "private/x.md"})])]
    )
    provider._client = client

    def dispatch_fn(name, args):
        raise PrivacyError("blocked: private content")

    messages = [{"role": "user", "content": "read my private note"}]
    reply = provider.agentic_turn(messages, TOOLS, dispatch_fn)

    assert reply == "blocked: private content"
    assert messages == [{"role": "user", "content": "read my private note"}]
    assert len(client.calls) == 1


# ── Unit: AnthropicProvider.agentic_turn ───────────────────────────────────────


def _anthropic_tool_use_response():
    block = SimpleNamespace(
        type="tool_use", id="tu_1", name="read_file", input={"path": "private/x.md"}
    )
    return SimpleNamespace(stop_reason="tool_use", content=[block])


class _FakeAnthropicClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


def test_anthropic_agentic_turn_privacy_error_stops_cleanly():
    """
    Same PrivacyError contract for the Anthropic loop: error text as reply,
    history restored to the original user turn, exactly one API call.
    """
    provider = AnthropicProvider(model="claude-test")
    client = _FakeAnthropicClient([_anthropic_tool_use_response()])
    provider._client = client

    def dispatch_fn(name, args):
        raise PrivacyError("blocked: private content")

    messages = [{"role": "user", "content": "read my private note"}]
    reply = provider.agentic_turn(messages, TOOLS, dispatch_fn, system="be helpful")

    assert reply == "blocked: private content"
    assert messages == [{"role": "user", "content": "read my private note"}]
    assert len(client.calls) == 1


def test_anthropic_agentic_turn_bundles_tool_results_in_one_user_message():
    """
    All tool_result blocks from one assistant turn must land in a single
    role='user' message (separate messages cause a 400 from the API).

    Input:  a tool_use response with two calls, then an end_turn response
    Expected output: one user message whose content holds both tool_result blocks
    """
    block_a = SimpleNamespace(type="tool_use", id="tu_a", name="read_file", input={"path": "a.md"})
    block_b = SimpleNamespace(type="tool_use", id="tu_b", name="read_file", input={"path": "b.md"})
    text_block = SimpleNamespace(type="text", text="Done.")
    responses = [
        SimpleNamespace(stop_reason="tool_use", content=[block_a, block_b]),
        SimpleNamespace(stop_reason="end_turn", content=[text_block]),
    ]
    provider = AnthropicProvider(model="claude-test")
    provider._client = _FakeAnthropicClient(responses)

    messages = [{"role": "user", "content": "read both files"}]
    reply = provider.agentic_turn(messages, TOOLS, lambda name, args: f"contents of {args['path']}")

    assert reply == "Done."
    tool_result_message = messages[2]
    assert tool_result_message["role"] == "user"
    assert [b["tool_use_id"] for b in tool_result_message["content"]] == ["tu_a", "tu_b"]