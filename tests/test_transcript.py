"""
Tests for jarvis/core/transcript.py — the provider-neutral transcript.

The whole point of this module is that a conversation can move between
providers, so the tests are mostly round-trips: neutral → wire → neutral must
come back unchanged for everything the schema models, and the provider-specific
material that it *cannot* model (Anthropic thinking blocks, OpenAI reasoning
fields) must survive a return to the same model while being dropped on the way
to a different one.

No provider client is involved — these are pure data transformations.
"""

import json

from jarvis.core.transcript import (
    from_anthropic,
    from_ollama,
    from_openai,
    message_text,
    opaque_block,
    text_block,
    tool_call_block,
    tool_result_block,
    to_anthropic,
    to_ollama,
    to_openai,
    user_message,
)


def _conversation() -> list[dict]:
    """A transcript exercising every block type the schema models."""
    return [
        user_message("what did I say about hybrid retrieval?"),
        {
            "role": "assistant",
            "content": [
                text_block("Let me search."),
                tool_call_block("call_1", "search_kb", {"query": "hybrid retrieval"}),
            ],
        },
        {
            "role": "user",
            "content": [tool_result_block("call_1", "BM25 fused by RRF.")],
        },
        {"role": "assistant", "content": [text_block("You called it reciprocal rank fusion.")]},
    ]


# ── round trips ────────────────────────────────────────────────────────────────

def test_anthropic_round_trip_is_identity():
    """Neutral → Anthropic → neutral returns exactly what went in."""
    original = _conversation()
    assert from_anthropic(to_anthropic(original, model="m"), model="m") == original


def test_openai_round_trip_is_identity():
    """
    Neutral → OpenAI → neutral survives the split of tool results into their
    own role:"tool" messages and their re-folding back into one user message.
    """
    original = _conversation()
    wire = to_openai(original, provider="openrouter", model="m")
    assert from_openai(wire, provider="openrouter", model="m") == original


def test_ollama_round_trip_preserves_structure():
    """
    Ollama carries no tool-call ids, so the ids are re-synthesised rather than
    preserved. Everything else — roles, text, tool names, arguments, results,
    and the call↔result pairing — must survive.
    """
    original = _conversation()
    wire = to_ollama(original, model="m")
    restored = from_ollama(wire, model="m")

    assert [m["role"] for m in restored] == [m["role"] for m in original]
    call = restored[1]["content"][1]
    result = restored[2]["content"][0]
    assert call["name"] == "search_kb"
    assert call["arguments"] == {"query": "hybrid retrieval"}
    assert result["content"] == "BM25 fused by RRF."
    # The pairing is what the ids exist for.
    assert result["tool_call_id"] == call["id"]


# ── wire-format specifics ──────────────────────────────────────────────────────

def test_openai_emits_one_tool_message_per_result():
    """
    Anthropic bundles tool results into a single user message; the OpenAI wire
    requires one role:"tool" message each. The neutral form is the same either
    way — the adapter absorbs the difference.
    """
    messages = [
        {
            "role": "assistant",
            "content": [
                tool_call_block("a", "one", {}),
                tool_call_block("b", "two", {}),
            ],
        },
        {
            "role": "user",
            "content": [tool_result_block("a", "first"), tool_result_block("b", "second")],
        },
    ]
    wire = to_openai(messages, provider="openrouter", model="m")

    assistant = wire[0]
    assert len(assistant["tool_calls"]) == 2
    # Arguments go over the wire as a JSON string, not a mapping.
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {}

    tool_messages = [m for m in wire if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["a", "b"]
    assert [m["content"] for m in tool_messages] == ["first", "second"]


def test_anthropic_bundles_tool_results_into_one_user_message():
    """The 400-error rule: all results for one assistant turn, one message."""
    messages = [
        {
            "role": "user",
            "content": [tool_result_block("a", "first"), tool_result_block("b", "second")],
        }
    ]
    wire = to_anthropic(messages, model="m")
    assert len(wire) == 1
    assert wire[0]["role"] == "user"
    assert [b["type"] for b in wire[0]["content"]] == ["tool_result", "tool_result"]
    assert wire[0]["content"][0]["tool_use_id"] == "a"


def test_ollama_tool_results_are_keyed_by_name():
    """Ollama identifies a result by tool name, looked up from its call."""
    messages = [
        {"role": "assistant", "content": [tool_call_block("x1", "search_kb", {})]},
        {"role": "user", "content": [tool_result_block("x1", "hit")]},
    ]
    wire = to_ollama(messages, model="m")
    tool_message = next(m for m in wire if m["role"] == "tool")
    assert tool_message["tool_name"] == "search_kb"
    assert tool_message["content"] == "hit"


def test_openai_parses_json_string_arguments():
    """OpenAI sends arguments as a JSON string; neutral holds a dict."""
    wire = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "search_kb", "arguments": '{"query": "rrf"}'},
                }
            ],
        }
    ]
    neutral = from_openai(wire, provider="openrouter", model="m")
    assert neutral[0]["content"][0]["arguments"] == {"query": "rrf"}


def test_openai_tolerates_malformed_arguments():
    """A model that emits broken JSON must not take the conversation down."""
    wire = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{oops"}}
            ],
        }
    ]
    neutral = from_openai(wire, provider="openrouter", model="m")
    assert neutral[0]["content"][0]["arguments"] == {}


# ── opaque blocks: same model replays, other models drop ───────────────────────

def test_anthropic_thinking_replays_to_the_same_model():
    """
    A thinking block must be echoed back verbatim when the conversation
    continues on the model that produced it.
    """
    thinking = {"type": "thinking", "thinking": "step one", "signature": "sig"}
    wire = [{"role": "assistant", "content": [thinking, {"type": "text", "text": "hi"}]}]

    neutral = from_anthropic(wire, model="claude-x")
    assert neutral[0]["content"][0]["type"] == "provider_opaque"

    assert to_anthropic(neutral, model="claude-x")[0]["content"][0] == thinking


def test_anthropic_thinking_is_dropped_for_a_different_model():
    """
    Switching models mid-conversation must not replay one vendor's internal
    state to another — the neutral text survives, the opaque block does not.
    """
    wire = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "step one", "signature": "sig"},
                {"type": "text", "text": "hi"},
            ],
        }
    ]
    neutral = from_anthropic(wire, model="claude-x")

    same_family_other_model = to_anthropic(neutral, model="claude-y")
    assert [b["type"] for b in same_family_other_model[0]["content"]] == ["text"]

    other_provider = to_openai(neutral, provider="openrouter", model="claude-x")
    assert other_provider[0]["content"] == "hi"
    assert "thinking" not in json.dumps(other_provider)


def test_openai_extra_message_fields_round_trip_to_the_same_model():
    """
    Reasoning traces and other provider-specific message fields are preserved
    for the same provider+model and dropped for anything else.
    """
    wire = [{"role": "assistant", "content": "hi", "reasoning": "because"}]
    neutral = from_openai(wire, provider="openrouter", model="m")

    same = to_openai(neutral, provider="openrouter", model="m")
    assert same[0]["reasoning"] == "because"

    different_model = to_openai(neutral, provider="openrouter", model="other")
    assert "reasoning" not in different_model[0]

    different_provider = to_anthropic(neutral, model="m")
    assert "because" not in json.dumps(different_provider)


def test_opaque_block_from_another_provider_never_reaches_the_wire():
    """The guard is provider AND model, not model alone."""
    neutral = [
        {
            "role": "assistant",
            "content": [
                text_block("hi"),
                opaque_block("anthropic", "m", {"type": "thinking", "thinking": "secret"}),
            ],
        }
    ]
    assert "secret" not in json.dumps(to_openai(neutral, provider="openrouter", model="m"))
    assert "secret" not in json.dumps(to_ollama(neutral, model="m"))
    assert "secret" in json.dumps(to_anthropic(neutral, model="m"))


# ── helpers ────────────────────────────────────────────────────────────────────

def test_message_text_joins_text_blocks_only():
    message = {
        "role": "assistant",
        "content": [text_block("a"), tool_call_block("i", "n", {}), text_block("b")],
    }
    assert message_text(message) == "ab"


def test_empty_messages_are_not_emitted():
    """
    A neutral message holding only tool results produces tool messages on the
    OpenAI wire and no empty user message alongside them.
    """
    messages = [{"role": "user", "content": [tool_result_block("a", "r")]}]
    wire = to_openai(messages, provider="openrouter", model="m")
    assert [m["role"] for m in wire] == ["tool"]
