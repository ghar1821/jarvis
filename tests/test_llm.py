"""
Tests for jarvis/core/llm.py — provider adapters.

Integration tests (marked, skipped by default) verify that live services are
reachable: the Anthropic API (key check, no tokens) and the local Ollama server.

Unit tests exercise the agentic_turn tool loop with the LLM client mocked at
the API boundary — the one place CLAUDE.md sanctions mocking, since real calls
bill per token / need a running model server. They pin down the PrivacyError
contract both providers must honour: return the error text, restore message
history exactly, and make no further LLM call.

Both providers stream their requests, so the fakes below are streams rather
than single responses. They record whether the stream was closed, because
closing it is what drops the connection and tells the server to stop
generating — the mechanism the forceful stop depends on.

Running
-------
    uv run pytest -m integration          # integration tests only
    uv run pytest -m "not integration"    # unit tests only (default CI run)
    uv run pytest                         # all tests
"""

import json
from types import SimpleNamespace

import pytest

from jarvis.core.cancel import CancelToken
from jarvis.core.errors import PrivacyError, TurnCancelled
from jarvis.core.llm import AnthropicProvider, OllamaProvider, active_model, make_provider

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


def test_active_model_picks_provider_appropriate_field():
    """
    active_model(cfg) consolidates the "which model are we actually using"
    conditional: anthropic_model under the anthropic provider, ollama_model
    otherwise. A bare SimpleNamespace stands in for Config since active_model
    only reads the three attributes it needs.
    """
    anthropic_cfg = SimpleNamespace(
        provider="anthropic", anthropic_model="claude-sonnet-4-6", ollama_model="qwen3-vl:30b"
    )
    ollama_cfg = SimpleNamespace(
        provider="ollama", anthropic_model="claude-sonnet-4-6", ollama_model="qwen3-vl:30b"
    )
    assert active_model(anthropic_cfg) == "claude-sonnet-4-6"
    assert active_model(ollama_cfg) == "qwen3-vl:30b"


# ── Unit: OllamaProvider.agentic_turn ──────────────────────────────────────────


class _OllamaMessage:
    """
    Minimal stand-in for the ollama client's pydantic Message object. Ollama
    hands tool_calls back as objects whose function.arguments is already a
    dict (not a JSON string), and supports .get() for the plain fields.
    """

    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def get(self, key, default=None):
        return getattr(self, key, default)


def _ollama_tool_call(name: str, arguments: dict):
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


class _FakeOllamaStream:
    """One streamed response: a sequence of chunk messages, closeable."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.closed = False

    def __iter__(self):
        for message in self._chunks:
            yield {"message": message}

    def close(self):
        self.closed = True


class _FakeOllamaClient:
    """
    Replays a scripted sequence of streamed responses — one list of chunk
    messages per request — and records every request and stream handed out.
    """

    def __init__(self, turns):
        self._turns = list(turns)
        self.calls: list[dict] = []
        self.streams: list[_FakeOllamaStream] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        stream = _FakeOllamaStream(self._turns.pop(0))
        self.streams.append(stream)
        return stream


class _StopAfterChunks(CancelToken):
    """
    A token that stops itself once it has been checked `allowed` times —
    stands in for the human hitting stop while a response streams in.
    """

    def __init__(self, allowed: int):
        super().__init__()
        self._remaining = allowed

    def check(self):
        if self._remaining <= 0:
            self.stop()
        self._remaining -= 1
        super().check()


def test_ollama_agentic_turn_dispatches_tools_with_dict_args():
    """
    Ollama returns tool arguments as a mapping already, so they reach dispatch
    as a dict without any JSON parsing; results feed back as role='tool'
    messages and the final text is returned. Content split across chunks is
    reassembled into one reply.

    Input:  scripted tool_use response then a two-chunk text response
    Expected output: dispatch got dict args; reply text returned; history has
            the assistant tool-call dict + tool result + final assistant text
    """
    provider = OllamaProvider(model="test-model")
    provider._client = _FakeOllamaClient(
        [
            [_OllamaMessage(tool_calls=[_ollama_tool_call("read_file", {"path": "notes.md"})])],
            [_OllamaMessage(content="Here is "), _OllamaMessage(content="the summary.")],
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
        [[_OllamaMessage(tool_calls=[_ollama_tool_call("read_file", {"path": "private/x.md"})])]]
    )
    provider._client = client

    def dispatch_fn(name, args):
        raise PrivacyError("blocked: private content")

    messages = [{"role": "user", "content": "read my private note"}]
    reply = provider.agentic_turn(messages, TOOLS, dispatch_fn)

    assert reply == "blocked: private content"
    assert messages == [{"role": "user", "content": "read my private note"}]
    assert len(client.calls) == 1


# ── Unit: cancellation ─────────────────────────────────────────────────────────


def test_ollama_cancel_before_request_sends_nothing():
    """
    A turn stopped before the loop's first iteration must never reach the
    server — this is the "if the request has not been sent, don't send it" half
    of the contract.

    Input:  an already-stopped token
    Expected output: TurnCancelled; zero requests made; messages untouched
    """
    provider = OllamaProvider(model="test-model")
    client = _FakeOllamaClient([])
    provider._client = client
    cancel = CancelToken()
    cancel.stop()

    messages = [{"role": "user", "content": "hello"}]
    with pytest.raises(TurnCancelled):
        provider.agentic_turn(messages, TOOLS, lambda n, a: "", cancel=cancel)

    assert client.calls == []
    assert messages == [{"role": "user", "content": "hello"}]


def test_ollama_cancel_mid_stream_closes_the_connection():
    """
    Stopping while the response is streaming closes the stream — that is the
    kill signal Ollama sees — and leaves the message history untouched, so the
    turn can simply be discarded.

    Input:  a three-chunk response, token stops after the first chunk
    Expected output: TurnCancelled; stream closed; messages unchanged
    """
    provider = OllamaProvider(model="test-model")
    client = _FakeOllamaClient([[
        _OllamaMessage(content="Here "),
        _OllamaMessage(content="is "),
        _OllamaMessage(content="the answer."),
    ]])
    provider._client = client

    messages = [{"role": "user", "content": "hello"}]
    with pytest.raises(TurnCancelled):
        provider.agentic_turn(
            messages, TOOLS, lambda n, a: "", cancel=_StopAfterChunks(1)
        )

    assert client.streams[0].closed, "the stream must be closed so Ollama stops generating"
    assert messages == [{"role": "user", "content": "hello"}]


def test_ollama_cancel_during_tool_dispatch_leaves_history_valid():
    """
    A stop that lands while a tool is running must not leave the assistant's
    tool-call message behind without its result — the same invariant the
    PrivacyError path protects, since a dangling tool call poisons the next
    turn.

    Input:  a tool_use response; dispatch raises TurnCancelled
    Expected output: TurnCancelled propagates; messages back to the user turn
    """
    provider = OllamaProvider(model="test-model")
    client = _FakeOllamaClient(
        [[_OllamaMessage(tool_calls=[_ollama_tool_call("read_file", {"path": "a.md"})])]]
    )
    provider._client = client

    def dispatch_fn(name, args):
        raise TurnCancelled("Stopped.")

    messages = [{"role": "user", "content": "read a.md"}]
    with pytest.raises(TurnCancelled):
        provider.agentic_turn(messages, TOOLS, dispatch_fn, cancel=CancelToken())

    assert messages == [{"role": "user", "content": "read a.md"}]
    assert len(client.calls) == 1


# ── Unit: AnthropicProvider.agentic_turn ───────────────────────────────────────


def _anthropic_tool_use_response():
    block = SimpleNamespace(
        type="tool_use", id="tu_1", name="read_file", input={"path": "private/x.md"}
    )
    return SimpleNamespace(stop_reason="tool_use", content=[block])


class _FakeAnthropicStream:
    """
    Stands in for the SDK's MessageStream: a context manager that yields raw
    events as it goes and hands back the assembled Message at the end. Records
    whether __exit__ ran, since closing the stream is what tells Anthropic to
    stop generating.
    """

    def __init__(self, response, event_count=3):
        self._response = response
        self._event_count = event_count
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.closed = True
        return False

    def __iter__(self):
        for index in range(self._event_count):
            yield SimpleNamespace(type=f"event_{index}")

    def get_final_message(self):
        return self._response


class _FakeAnthropicClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.streams: list[_FakeAnthropicStream] = []
        self.messages = SimpleNamespace(stream=self._stream)

    def _stream(self, **kwargs):
        self.calls.append(kwargs)
        stream = _FakeAnthropicStream(self._responses.pop(0))
        self.streams.append(stream)
        return stream


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


def test_anthropic_cancel_before_request_sends_nothing():
    """
    Same contract as the Ollama case: a turn already stopped must not reach the
    API at all, so a stop can never cost money it didn't have to.
    """
    provider = AnthropicProvider(model="claude-test")
    client = _FakeAnthropicClient([])
    provider._client = client
    cancel = CancelToken()
    cancel.stop()

    messages = [{"role": "user", "content": "hello"}]
    with pytest.raises(TurnCancelled):
        provider.agentic_turn(messages, TOOLS, lambda n, a: "", cancel=cancel)

    assert client.calls == []
    assert messages == [{"role": "user", "content": "hello"}]


def test_anthropic_cancel_mid_stream_closes_the_connection():
    """
    Stopping mid-response must exit the streaming context manager, which is
    what closes the HTTP response and stops generation server-side (the
    Messages API has no cancel endpoint). History stays untouched.
    """
    text_block = SimpleNamespace(type="text", text="A long answer.")
    provider = AnthropicProvider(model="claude-test")
    client = _FakeAnthropicClient([SimpleNamespace(stop_reason="end_turn", content=[text_block])])
    provider._client = client

    messages = [{"role": "user", "content": "hello"}]
    with pytest.raises(TurnCancelled):
        provider.agentic_turn(
            messages, TOOLS, lambda n, a: "", cancel=_StopAfterChunks(1)
        )

    assert client.streams[0].closed, "the stream must close so Anthropic stops generating"
    assert messages == [{"role": "user", "content": "hello"}]


def test_anthropic_cancel_during_tool_dispatch_pops_the_assistant_message():
    """
    The Anthropic wire format is strict: an assistant tool_use message must be
    followed by a user message bundling every tool_result. A stop mid-dispatch
    must therefore drop the assistant message, or the next turn 400s.
    """
    provider = AnthropicProvider(model="claude-test")
    client = _FakeAnthropicClient([_anthropic_tool_use_response()])
    provider._client = client

    def dispatch_fn(name, args):
        raise TurnCancelled("Stopped.")

    messages = [{"role": "user", "content": "read a file"}]
    with pytest.raises(TurnCancelled):
        provider.agentic_turn(messages, TOOLS, dispatch_fn, cancel=CancelToken())

    assert messages == [{"role": "user", "content": "read a file"}]
    assert len(client.calls) == 1