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
from jarvis.core.llm import (
    AnthropicProvider,
    OllamaProvider,
    OpenRouterProvider,
    active_model,
    is_cloud_provider,
    make_provider,
    split_spec,
)
from jarvis.core.transcript import message_text, user_message

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
    make_provider maps each provider name to its adapter and rejects anything
    else. A "provider:model" spec names the model inline, which is how a
    session records the model it is running on.
    """
    assert isinstance(make_provider("ollama"), OllamaProvider)
    assert isinstance(make_provider("anthropic"), AnthropicProvider)

    router = make_provider("openrouter:openai/gpt-5")
    assert isinstance(router, OpenRouterProvider)
    assert router.model == "openai/gpt-5"

    with pytest.raises(ValueError, match="Unknown provider"):
        make_provider("llamacpp")


def test_split_spec_keeps_slashes_and_dots_in_the_model():
    """OpenRouter model names carry both, so only the first colon splits."""
    assert split_spec("openrouter:anthropic/claude-sonnet-4.6") == (
        "openrouter",
        "anthropic/claude-sonnet-4.6",
    )
    assert split_spec("ollama") == ("ollama", "")


def test_make_provider_refuses_openrouter_without_a_model():
    """
    There is no sensible default model for a broker fronting hundreds of them,
    so jarvis asks rather than guessing.
    """
    with pytest.raises(ValueError, match="No model configured"):
        make_provider("openrouter")


def test_is_cloud_provider_is_local_vs_everything_else():
    """
    The privacy model keys on one predicate, not on a vendor name, so adding a
    provider cannot quietly open a hole. A "provider:model" spec is classified
    by its provider half.
    """
    assert is_cloud_provider("ollama") is False
    assert is_cloud_provider("ollama:qwen3-vl:30b") is False
    assert is_cloud_provider("anthropic") is True
    assert is_cloud_provider("openrouter") is True
    assert is_cloud_provider("openrouter:openai/gpt-5") is True
    # An unknown name must fail closed — treated as cloud, never as local.
    assert is_cloud_provider("some-new-vendor") is True


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
    as a dict without any JSON parsing. The wire sent to Ollama uses its
    role="tool" / tool_name shape, while the history appended to `messages` is
    the provider-neutral transcript — that split is what lets the next turn run
    on a different model. Content split across chunks is reassembled into one
    reply.

    Input:  scripted tool_use response then a two-chunk text response
    Expected output: dispatch got dict args; reply text returned; neutral
            history has the tool_call block, its tool_result, and the reply
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

    messages = [user_message("read notes.md")]
    reply = provider.agentic_turn(messages, TOOLS, dispatch_fn, system="be helpful")

    assert reply == "Here is the summary."
    assert dispatched == [("read_file", {"path": "notes.md"})]

    call = messages[1]["content"][0]
    assert call["type"] == "tool_call" and call["name"] == "read_file"
    result = messages[2]["content"][0]
    assert result["type"] == "tool_result"
    assert result["tool_call_id"] == call["id"]
    assert result["content"] == "file contents"
    assert message_text(messages[3]) == "Here is the summary."

    # The wire Ollama actually saw keys the result by tool name, not by id.
    sent = provider._client.calls[-1]["messages"]
    assert {"role": "tool", "tool_name": "read_file", "content": "file contents"} in sent


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

    messages = [user_message("read my private note")]
    reply = provider.agentic_turn(messages, TOOLS, dispatch_fn)

    assert reply == "blocked: private content"
    assert messages == [user_message("read my private note")]
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

    messages = [user_message("hello")]
    with pytest.raises(TurnCancelled):
        provider.agentic_turn(messages, TOOLS, lambda n, a: "", cancel=cancel)

    assert client.calls == []
    assert messages == [user_message("hello")]


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

    messages = [user_message("hello")]
    with pytest.raises(TurnCancelled):
        provider.agentic_turn(
            messages, TOOLS, lambda n, a: "", cancel=_StopAfterChunks(1)
        )

    assert client.streams[0].closed, "the stream must be closed so Ollama stops generating"
    assert messages == [user_message("hello")]


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

    messages = [user_message("read a.md")]
    with pytest.raises(TurnCancelled):
        provider.agentic_turn(messages, TOOLS, dispatch_fn, cancel=CancelToken())

    assert messages == [user_message("read a.md")]
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

    messages = [user_message("read my private note")]
    reply = provider.agentic_turn(messages, TOOLS, dispatch_fn, system="be helpful")

    assert reply == "blocked: private content"
    assert messages == [user_message("read my private note")]
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

    messages = [user_message("read both files")]
    reply = provider.agentic_turn(messages, TOOLS, lambda name, args: f"contents of {args['path']}")

    assert reply == "Done."
    # Neutral history keeps both results together in one user message...
    tool_result_message = messages[2]
    assert tool_result_message["role"] == "user"
    assert [b["tool_call_id"] for b in tool_result_message["content"]] == ["tu_a", "tu_b"]

    # ...and so does the wire actually sent to the API, which is the rule that
    # matters: separate messages are a 400.
    sent = provider._client.calls[-1]["messages"]
    result_messages = [
        m for m in sent
        if m["role"] == "user" and any(b.get("type") == "tool_result" for b in m["content"])
    ]
    assert len(result_messages) == 1
    assert [b["tool_use_id"] for b in result_messages[0]["content"]] == ["tu_a", "tu_b"]

# ── Unit: OpenRouterProvider ───────────────────────────────────────────────────


def _openai_tool_call(call_id: str, name: str, arguments: str):
    """One tool call, for _openai_response to break into stream deltas."""
    return (call_id, name, arguments)


def _openai_chunk(content=None, tool_calls=None, cost=None, model=None):
    """
    One streamed chunk. A chunk carrying usage has no choices at all, which is
    how the real API sends the final one.
    """
    chunk = SimpleNamespace(choices=[], usage=None, model=model or "")
    if content is not None or tool_calls is not None:
        chunk.choices = [
            SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=tool_calls))
        ]
    if cost is not None:
        chunk.usage = SimpleNamespace(cost=cost)
    return chunk


def _tool_call_delta(index, call_id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index, id=call_id, type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _openai_response(content=None, tool_calls=None, cost=None, model=None):
    """
    One streamed response, as the chunk sequence the SDK would yield.

    Content is split across two chunks and every tool call's arguments across
    two more, with the id and name arriving in a chunk of their own — that is
    how the real API drives it, and handing the accumulator everything at once
    would test nothing. `model` is what OpenRouter says answered, the
    interesting half of a router response where it isn't what was asked for.
    """
    chunks = []
    if content:
        half = len(content) // 2
        chunks.append(_openai_chunk(content=content[:half]))
        chunks.append(_openai_chunk(content=content[half:]))
    for index, (call_id, name, arguments) in enumerate(tool_calls or []):
        chunks.append(
            _openai_chunk(tool_calls=[_tool_call_delta(index, call_id=call_id, name=name)])
        )
        half = len(arguments) // 2
        chunks.append(_openai_chunk(tool_calls=[_tool_call_delta(index, arguments=arguments[:half])]))
        chunks.append(_openai_chunk(tool_calls=[_tool_call_delta(index, arguments=arguments[half:])]))
    # The closing chunk carries usage and the served model, and no choices.
    chunks.append(_openai_chunk(cost=cost, model=model))
    return chunks


class _FakeOpenAIStream:
    """One streamed response: a sequence of chunks, closeable."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.closed = False

    def __iter__(self):
        yield from self._chunks

    def close(self):
        self.closed = True


class _FakeOpenAIClient:
    """
    Replays scripted streamed responses — one chunk list per request — and
    records every request and stream handed out.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.streams: list[_FakeOpenAIStream] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        stream = _FakeOpenAIStream(self._responses.pop(0))
        self.streams.append(stream)
        return stream


def test_openrouter_agentic_turn_runs_the_tool_loop():
    """
    OpenAI-shaped tool calling: arguments arrive as a JSON string and are
    parsed before dispatch, each result goes back as its own role="tool"
    message keyed by call id, and the neutral history records the pairing.
    """
    provider = OpenRouterProvider(model="openai/gpt-5")
    provider._client = _FakeOpenAIClient([
        _openai_response(tool_calls=[_openai_tool_call("c1", "read_file", '{"path": "notes.md"}')]),
        _openai_response(content="Here is the summary."),
    ])

    dispatched = []

    def dispatch_fn(name, args):
        dispatched.append((name, args))
        return "file contents"

    messages = [user_message("read notes.md")]
    reply = provider.agentic_turn(messages, TOOLS, dispatch_fn, system="be helpful")

    assert reply == "Here is the summary."
    assert dispatched == [("read_file", {"path": "notes.md"})]

    call = messages[1]["content"][0]
    result = messages[2]["content"][0]
    assert call["type"] == "tool_call" and call["id"] == "c1"
    assert result["tool_call_id"] == "c1" and result["content"] == "file contents"
    assert message_text(messages[3]) == "Here is the summary."

    sent = provider._client.calls[-1]["messages"]
    assert {"role": "tool", "tool_call_id": "c1", "content": "file contents"} in sent


def test_openrouter_privacy_error_stops_cleanly():
    """Same PrivacyError contract as the other two adapters."""
    provider = OpenRouterProvider(model="openai/gpt-5")
    client = _FakeOpenAIClient([
        _openai_response(tool_calls=[_openai_tool_call("c1", "read_file", '{"path": "private/x.md"}')]),
    ])
    provider._client = client

    def dispatch_fn(name, args):
        raise PrivacyError("blocked: private content")

    messages = [user_message("read my private note")]
    reply = provider.agentic_turn(messages, TOOLS, dispatch_fn)

    assert reply == "blocked: private content"
    assert messages == [user_message("read my private note")]
    assert len(client.calls) == 1


def test_openrouter_streams_and_still_reports_cost():
    """
    The request must be streamed — that is the only way it can be interrupted —
    and the cost must survive the switch, since it now arrives on the stream's
    final chunk rather than on a whole response. A session's spend silently
    reading as zero is the failure this guards.

    Only OpenRouter's own `usage: {include: true}` is sent; the OpenAI-standard
    `stream_options` would ask for the same thing twice.
    """
    provider = OpenRouterProvider(model="openai/gpt-5")
    provider._client = _FakeOpenAIClient([_openai_response(content="hi", cost=0.01)])

    provider.complete([{"role": "user", "content": "hello"}])

    sent = provider._client.calls[-1]
    assert sent["stream"] is True
    assert "stream_options" not in sent
    assert sent["extra_body"]["usage"] == {"include": True}
    assert provider.pop_usage()["usd"] == 0.01


def test_openrouter_reassembles_two_tool_calls_split_across_chunks():
    """
    A turn asking for two tools at once must come back as two calls, each with
    its own id and its arguments JSON rejoined from the fragments it arrived
    in — the accumulator keys on the delta `index`, so interleaved fragments
    must not bleed between calls.
    """
    provider = OpenRouterProvider(model="openai/gpt-5")
    provider._client = _FakeOpenAIClient([
        _openai_response(tool_calls=[
            _openai_tool_call("c1", "read_file", '{"path": "a.md"}'),
            _openai_tool_call("c2", "read_file", '{"path": "b.md"}'),
        ]),
        _openai_response(content="Read both."),
    ])

    dispatched = []
    reply = provider.agentic_turn(
        [user_message("read both files")], TOOLS,
        lambda name, args: dispatched.append((name, args)) or "contents",
    )

    assert reply == "Read both."
    assert dispatched == [
        ("read_file", {"path": "a.md"}),
        ("read_file", {"path": "b.md"}),
    ]
    # Each result goes back keyed by its own call id.
    sent = provider._client.calls[-1]["messages"]
    tool_messages = [m for m in sent if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["c1", "c2"]


def test_openrouter_cancel_before_request_sends_nothing():
    """
    Same contract as the other two adapters: a turn already stopped must not
    reach the API at all, so a stop can never cost money it didn't have to.
    """
    provider = OpenRouterProvider(model="openai/gpt-5")
    client = _FakeOpenAIClient([])
    provider._client = client
    cancel = CancelToken()
    cancel.stop()

    messages = [user_message("hello")]
    with pytest.raises(TurnCancelled):
        provider.agentic_turn(messages, TOOLS, lambda n, a: "", cancel=cancel)

    assert client.calls == []
    assert messages == [user_message("hello")]


def test_openrouter_cancel_mid_stream_closes_the_connection():
    """
    Stopping mid-response closes the stream, which drops the connection and is
    what tells the upstream model to stop generating. The neutral transcript is
    left untouched, so the abandoned turn leaves no trace.
    """
    provider = OpenRouterProvider(model="openai/gpt-5")
    client = _FakeOpenAIClient([_openai_response(content="A long answer about soil.")])
    provider._client = client

    messages = [user_message("hello")]
    with pytest.raises(TurnCancelled):
        provider.agentic_turn(
            messages, TOOLS, lambda n, a: "", cancel=_StopAfterChunks(1)
        )

    assert client.streams[0].closed, "the stream must close so the model stops generating"
    assert messages == [user_message("hello")]


def test_openrouter_cancel_during_tool_dispatch_leaves_history_valid():
    """
    A stop landing while a tool runs must not publish the assistant's tool call
    without its result — the transcript would be invalid for the next turn.
    Nothing is committed until the turn returns, so there is nothing to undo.
    """
    provider = OpenRouterProvider(model="openai/gpt-5")
    client = _FakeOpenAIClient([
        _openai_response(tool_calls=[_openai_tool_call("c1", "read_file", '{"path": "a.md"}')]),
    ])
    provider._client = client

    def dispatch_fn(name, args):
        raise TurnCancelled("Stopped.")

    messages = [user_message("read a.md")]
    with pytest.raises(TurnCancelled):
        provider.agentic_turn(messages, TOOLS, dispatch_fn, cancel=CancelToken())

    assert messages == [user_message("read a.md")]
    assert len(client.calls) == 1



def test_openrouter_sends_the_routing_hardening_on_every_request():
    """
    OpenRouter is a broker, so each request must carry the strict routing
    preferences — otherwise it may pick an upstream provider that trains on
    what it is sent. Usage accounting is requested too; it is the only cost
    figure jarvis reports.
    """
    provider = OpenRouterProvider(model="openai/gpt-5")
    provider._client = _FakeOpenAIClient([_openai_response(content="hi")])

    provider.agentic_turn([user_message("hello")], TOOLS, lambda n, a: "")

    extra = provider._client.calls[0]["extra_body"]
    assert extra["provider"]["data_collection"] == "deny"
    assert extra["provider"]["allow_fallbacks"] is False
    assert extra["usage"] == {"include": True}


def test_openrouter_omits_leaderboard_headers():
    """
    The optional HTTP-Referer / X-Title headers exist to list the app on
    OpenRouter's public leaderboards — sending them would be telemetry by
    another name, so no request may carry them.
    """
    provider = OpenRouterProvider(model="openai/gpt-5")
    provider._client = _FakeOpenAIClient([_openai_response(content="hi")])

    provider.agentic_turn([user_message("hello")], TOOLS, lambda n, a: "")

    request = provider._client.calls[0]
    headers = request.get("extra_headers") or {}
    assert "HTTP-Referer" not in headers
    assert "X-Title" not in headers


def test_openrouter_accumulates_reported_cost_and_pop_resets():
    """
    Cost comes from what OpenRouter reports per request, summed across the
    whole turn (a tool loop is several requests). pop_usage() drains it so the
    caller can add exactly one turn's spend to the session.
    """
    provider = OpenRouterProvider(model="openai/gpt-5")
    provider._client = _FakeOpenAIClient([
        _openai_response(tool_calls=[_openai_tool_call("c1", "read_file", "{}")], cost=0.001),
        _openai_response(content="done", cost=0.002),
    ])

    provider.agentic_turn([user_message("go")], TOOLS, lambda n, a: "result")

    usage = provider.pop_usage()
    assert usage == {"usd": 0.003, "requests": 2}
    assert provider.pop_usage() is None  # drained


def test_openrouter_reports_no_cost_when_none_came_back():
    """
    A response without a cost figure contributes nothing — jarvis never
    invents a number. The request is still counted.
    """
    provider = OpenRouterProvider(model="openai/gpt-5")
    provider._client = _FakeOpenAIClient([_openai_response(content="hi")])

    provider.agentic_turn([user_message("hello")], TOOLS, lambda n, a: "")

    assert provider.pop_usage() == {"usd": 0.0, "requests": 1}


def test_local_and_anthropic_providers_report_no_cost():
    """
    Ollama runs on the user's own hardware, and turning Anthropic's token
    counts into dollars would need a price table that ages silently. Both
    report nothing rather than something unreliable.
    """
    assert OllamaProvider(model="m").pop_usage() is None
    assert AnthropicProvider(model="m").pop_usage() is None


def test_openrouter_reports_which_model_actually_answered():
    """
    `openrouter/auto` is a router: the request names it, the response names
    the model that really ran. Without carrying that back, a session on auto
    can only ever report "openrouter/auto" and the spend piles up under a
    name that never served a token.
    """
    provider = OpenRouterProvider(model="openrouter/auto")
    provider._client = _FakeOpenAIClient([
        _openai_response(content="hi", cost=0.002, model="anthropic/claude-sonnet-4.6"),
    ])

    provider.agentic_turn([user_message("go")], TOOLS, lambda n, a: "result")

    usage = provider.pop_usage()
    assert usage["model"] == "openrouter:anthropic/claude-sonnet-4.6"
    assert usage["usd"] == 0.002
    # The request still asked for the router — only the reporting resolves.
    assert provider._client.calls[0]["model"] == "openrouter/auto"


def test_openrouter_usage_names_no_model_when_the_response_does_not():
    """
    A response with no model field must not invent one; the caller then keys
    the spend by whatever was requested, exactly as before.
    """
    provider = OpenRouterProvider(model="openai/gpt-5")
    provider._client = _FakeOpenAIClient([_openai_response(content="hi", cost=0.001)])

    provider.agentic_turn([user_message("go")], TOOLS, lambda n, a: "result")

    assert "model" not in provider.pop_usage()

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

    messages = [user_message("hello")]
    with pytest.raises(TurnCancelled):
        provider.agentic_turn(messages, TOOLS, lambda n, a: "", cancel=cancel)

    assert client.calls == []
    assert messages == [user_message("hello")]


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

    messages = [user_message("hello")]
    with pytest.raises(TurnCancelled):
        provider.agentic_turn(
            messages, TOOLS, lambda n, a: "", cancel=_StopAfterChunks(1)
        )

    assert client.streams[0].closed, "the stream must close so Anthropic stops generating"
    assert messages == [user_message("hello")]


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

    messages = [user_message("read a file")]
    with pytest.raises(TurnCancelled):
        provider.agentic_turn(messages, TOOLS, dispatch_fn, cancel=CancelToken())

    assert messages == [user_message("read a file")]
    assert len(client.calls) == 1
