"""LLM-based filtering and scoring of papers."""

import json
from pathlib import Path

from .errors import LLMError, with_retries
from .llm import ChatProvider

# 192k tokens to fit ~490 abstracts in one call
_SCORING_CONTEXT_LENGTH = 196608


def filter_and_score(
    papers: list[dict],
    provider: ChatProvider,
    max_results: int,
    prompt_path: Path,
) -> dict:
    """
    Ask the LLM to filter and score a list of papers.

    provider: any ChatProvider (Ollama or Anthropic).
    Returns the parsed JSON object from the model response.
    """
    abstracts_text = ""
    for i, p in enumerate(papers):
        abstracts_text += (
            f"\n[{i}] TITLE: {p['title']}\n"
            f"AUTHORS: {p['authors']}\n"
            f"SOURCE: {p['source']} | {p['published']}\n"
            f"ABSTRACT: {p['abstract'][:500]}\n"
        )

    prompt_template = prompt_path.read_text()
    prompt = (
        prompt_template
        .replace("{num_papers}", str(len(papers)))
        .replace("{max_results}", str(max_results))
        .replace("{abstracts_text}", abstracts_text)
    )

    @with_retries(max_attempts=3, backoff=2.0, exceptions=(LLMError, ValueError))
    def _call() -> dict:
        raw = provider.complete(
            [{"role": "user", "content": prompt}],
            max_tokens=8192,
            context_length=_SCORING_CONTEXT_LENGTH,
        )
        return _parse_json(raw)

    return _call()


def _parse_json(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    if not cleaned:
        raise ValueError("Model returned empty content.")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise
