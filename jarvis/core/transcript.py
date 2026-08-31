"""
Provider-neutral conversation transcript.

Sessions used to store whatever wire format the active provider happened to
speak, which is exactly why a conversation could never move between providers:
an Anthropic transcript is a list of typed content blocks, an OpenAI one is
flat strings plus a `tool_calls` array, and Ollama is a third shape again. This
module defines one format they all convert to and from, so the session file is
readable by whichever model the user switches to next.

The schema is flat dicts, matching the project's "flat over nested" preference:

    message : {"role": "user" | "assistant", "content": [block, ...]}

    blocks:
      {"type": "text",        "text": str}
      {"type": "tool_call",   "id": str, "name": str, "arguments": dict}
      {"type": "tool_result", "tool_call_id": str, "content": str,
                              "is_error": bool}
      {"type": "provider_opaque", "provider": str, "model": str, "data": dict}

Tool results live in a *user* message, which is Anthropic's requirement and
harmless everywhere else — the OpenAI and Ollama adapters split them back out
into their own `role: "tool"` messages on the way to the wire.

`provider_opaque` carries anything the neutral schema cannot express, most
importantly Anthropic `thinking` blocks: they must be echoed back verbatim when
the conversation continues on the same Anthropic model, and are meaningless to
any other model. Each opaque block records the provider and model it came from,
and `to_*` emits it only on an exact match — otherwise it is dropped, which is
the correct behaviour for a cross-model switch.

Usage inside a provider's agentic_turn:

    wire = to_anthropic(messages, model=self.model)
    start = len(wire)
    ... run the tool loop, appending to `wire` ...
    messages.extend(from_anthropic(wire[start:], model=self.model))

Only the newly generated wire messages are converted back, so history written
by some other provider is never round-tripped through a lossy conversion.
"""

import json
import uuid

# ── Neutral block constructors ─────────────────────────────────────────────────


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def tool_call_block(call_id: str, name: str, arguments: dict) -> dict:
    return {"type": "tool_call", "id": call_id, "name": name, "arguments": arguments}


def tool_result_block(tool_call_id: str, content: str, is_error: bool = False) -> dict:
    return {
        "type": "tool_result",
        "tool_call_id": tool_call_id,
        "content": content,
        "is_error": is_error,
    }


def opaque_block(provider: str, model: str, data: dict) -> dict:
    return {"type": "provider_opaque", "provider": provider, "model": model, "data": data}


def user_message(text: str) -> dict:
    """The common case: a plain user turn."""
    return {"role": "user", "content": [text_block(text)]}


def message_text(message: dict) -> str:
    """Concatenate the text blocks of one neutral message."""
    return "".join(
        block["text"] for block in message.get("content", []) if block.get("type") == "text"
    )


def _blocks(message: dict, block_type: str) -> list[dict]:
    return [b for b in message.get("content", []) if b.get("type") == block_type]


def _matching_opaque(message: dict, provider: str, model: str) -> list[dict]:
    """Opaque blocks that belong to this exact provider+model, in order."""
    return [
        b
        for b in message.get("content", [])
        if b.get("type") == "provider_opaque"
        and b.get("provider") == provider
        and b.get("model") == model
    ]


def _tool_name_by_call_id(messages: list[dict]) -> dict:
    """
    Map tool_call id → tool name across the whole transcript.

    Ollama identifies a tool result by name rather than by call id, so the
    adapter has to look back at the call that produced it.
    """
    names = {}
    for message in messages:
        for block in _blocks(message, "tool_call"):
            names[block["id"]] = block["name"]
    return names


def _arguments_to_dict(arguments) -> dict:
    """Tool arguments arrive as a mapping (Ollama) or a JSON string (OpenAI)."""
    if isinstance(arguments, dict):
        return arguments
    try:
        return json.loads(arguments or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}


# ── Anthropic ──────────────────────────────────────────────────────────────────

def to_anthropic(messages: list[dict], model: str) -> list[dict]:
    """Neutral → Anthropic wire messages."""
    wire = []
    for message in messages:
        content = []
        for block in message.get("content", []):
            kind = block.get("type")
            if kind == "text":
                content.append({"type": "text", "text": block["text"]})
            elif kind == "tool_call":
                content.append(
                    {
                        "type": "tool_use",
                        "id": block["id"],
                        "name": block["name"],
                        "input": block.get("arguments", {}),
                    }
                )
            elif kind == "tool_result":
                result = {
                    "type": "tool_result",
                    "tool_use_id": block["tool_call_id"],
                    "content": block.get("content", ""),
                }
                if block.get("is_error"):
                    result["is_error"] = True
                content.append(result)
            elif kind == "provider_opaque":
                # Only replay a thinking block to the model that produced it.
                if block.get("provider") == "anthropic" and block.get("model") == model:
                    content.append(block["data"])
        if content:
            wire.append({"role": message["role"], "content": content})
    return wire


def from_anthropic(wire: list[dict], model: str) -> list[dict]:
    """
    Anthropic wire messages → neutral.

    Block types the neutral schema models directly (text, tool_use,
    tool_result) convert; anything else the API returns — thinking,
    redacted_thinking, and whatever Anthropic adds next — is preserved as an
    opaque block rather than silently dropped.
    """
    messages = []
    for message in wire:
        content = message.get("content", "")
        if isinstance(content, str):
            messages.append({"role": message["role"], "content": [text_block(content)]})
            continue

        blocks = []
        for block in content:
            kind = block.get("type")
            if kind == "text":
                blocks.append(text_block(block.get("text", "")))
            elif kind == "tool_use":
                blocks.append(
                    tool_call_block(block["id"], block["name"], block.get("input", {}))
                )
            elif kind == "tool_result":
                blocks.append(
                    tool_result_block(
                        block["tool_use_id"],
                        block.get("content", ""),
                        bool(block.get("is_error", False)),
                    )
                )
            else:
                blocks.append(opaque_block("anthropic", model, block))
        messages.append({"role": message["role"], "content": blocks})
    return messages


# ── OpenAI shape (OpenAI, OpenRouter) ──────────────────────────────────────────

# Message keys the neutral schema reconstructs on its own. Anything else on an
# assistant message (reasoning traces, provider annotations) is kept in an
# opaque block so it can be replayed to the same model.
_OPENAI_KNOWN_FIELDS = {"role", "content", "tool_calls", "refusal", "annotations"}


def to_openai(messages: list[dict], provider: str, model: str) -> list[dict]:
    """
    Neutral → OpenAI chat-completions messages.

    Tool results become their own `role: "tool"` messages (the wire format's
    requirement), so one neutral message can expand into several.
    """
    wire = []
    for message in messages:
        role = message["role"]
        tool_results = _blocks(message, "tool_result")
        for result in tool_results:
            wire.append(
                {
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": result.get("content", ""),
                }
            )

        text = message_text(message)
        tool_calls = [
            {
                "id": block["id"],
                "type": "function",
                "function": {
                    "name": block["name"],
                    "arguments": json.dumps(block.get("arguments", {})),
                },
            }
            for block in _blocks(message, "tool_call")
        ]
        if not text and not tool_calls:
            continue

        entry = {"role": role, "content": text}
        if tool_calls:
            entry["tool_calls"] = tool_calls
        # Replay provider-specific fields (reasoning and friends) only to the
        # model that produced them.
        for opaque in _matching_opaque(message, provider, model):
            entry.update(opaque["data"])
        wire.append(entry)
    return wire


def from_openai(wire: list[dict], provider: str, model: str) -> list[dict]:
    """
    OpenAI chat-completions messages → neutral.

    Consecutive `role: "tool"` messages are folded back into a single neutral
    user message of tool_result blocks, which is the shape Anthropic needs and
    the one this schema standardises on.
    """
    messages = []
    pending_results: list[dict] = []

    def flush_results():
        if pending_results:
            messages.append({"role": "user", "content": list(pending_results)})
            pending_results.clear()

    for message in wire:
        role = message.get("role")
        if role == "tool":
            pending_results.append(
                tool_result_block(
                    message.get("tool_call_id", ""),
                    message.get("content", "") or "",
                )
            )
            continue

        flush_results()
        blocks = []
        if message.get("content"):
            blocks.append(text_block(message["content"]))
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            blocks.append(
                tool_call_block(
                    call.get("id", ""),
                    function.get("name", ""),
                    _arguments_to_dict(function.get("arguments")),
                )
            )
        extra = {k: v for k, v in message.items() if k not in _OPENAI_KNOWN_FIELDS}
        if extra:
            blocks.append(opaque_block(provider, model, extra))
        messages.append({"role": role, "content": blocks})

    flush_results()
    return messages


# ── Ollama ─────────────────────────────────────────────────────────────────────


def to_ollama(messages: list[dict], model: str) -> list[dict]:
    """
    Neutral → Ollama chat messages.

    Ollama keys a tool result by tool *name* rather than by call id, so the
    name is looked up from the call that produced it.
    """
    names = _tool_name_by_call_id(messages)
    wire = []
    for message in messages:
        for result in _blocks(message, "tool_result"):
            wire.append(
                {
                    "role": "tool",
                    "tool_name": names.get(result["tool_call_id"], ""),
                    "content": result.get("content", ""),
                }
            )

        text = message_text(message)
        tool_calls = [
            {"function": {"name": block["name"], "arguments": block.get("arguments", {})}}
            for block in _blocks(message, "tool_call")
        ]
        if not text and not tool_calls:
            continue

        entry = {"role": message["role"], "content": text}
        if tool_calls:
            entry["tool_calls"] = tool_calls
        for opaque in _matching_opaque(message, "ollama", model):
            entry.update(opaque["data"])
        wire.append(entry)
    return wire


def from_ollama(wire: list[dict], model: str, id_prefix: "str | None" = None) -> list[dict]:
    """
    Ollama chat messages → neutral.

    Ollama's tool calls carry no id of their own, so one is synthesised per
    call. The ids exist so the neutral schema (and any provider it later
    converts to) can pair a result with its call, and they must stay unique
    across the whole transcript — one turn's ids must never collide with an
    earlier turn's, or a tool result would be attributed to the wrong call.
    The default prefix is therefore random per conversion.
    """
    if id_prefix is None:
        id_prefix = f"call{uuid.uuid4().hex[:8]}"
    messages = []
    pending_results: list[dict] = []
    counter = 0
    # Results arrive in call order, so a queue of ids pairs them up.
    awaiting_ids: list[str] = []

    def flush_results():
        if pending_results:
            messages.append({"role": "user", "content": list(pending_results)})
            pending_results.clear()

    for message in wire:
        role = message.get("role")
        if role == "tool":
            call_id = awaiting_ids.pop(0) if awaiting_ids else f"{id_prefix}_orphan"
            pending_results.append(tool_result_block(call_id, message.get("content", "") or ""))
            continue

        flush_results()
        blocks = []
        if message.get("content"):
            blocks.append(text_block(message["content"]))
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            call_id = f"{id_prefix}_{counter}"
            counter += 1
            awaiting_ids.append(call_id)
            blocks.append(
                tool_call_block(
                    call_id,
                    function.get("name", ""),
                    _arguments_to_dict(function.get("arguments")),
                )
            )
        extra = {
            k: v
            for k, v in message.items()
            if k not in {"role", "content", "tool_calls", "tool_name", "images"}
        }
        if extra:
            blocks.append(opaque_block("ollama", model, extra))
        messages.append({"role": role, "content": blocks})

    flush_results()
    return messages
