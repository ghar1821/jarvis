"""
Tests for jarvis/kb/metadata.py — automatic metadata inference for local PDFs.

infer_pdf_metadata() reads a PDF's first pages and asks the active provider to
extract title/authors/DOI in one call; a DOI regex runs first since it is
cheap and exact when a DOI is actually printed on the page.
resolve_pdf_metadata() is the policy layer every add-path shares: explicit
overrides win outright, then the private-note-under-Anthropic privacy guard,
then inference for whatever is still unset.

PDFs are generated in-test via pymupdf (same pattern as
tests/test_daemon.py::_make_pdf) rather than mocked — cheap and real.
"""

from pathlib import Path

import pymupdf
import pytest

from jarvis.kb.metadata import infer_pdf_metadata, resolve_pdf_metadata


def _make_pdf(path: Path, text: str) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=10)
    doc.save(path)
    doc.close()
    return path


class _StubProvider:
    """Captures the prompt it was called with and returns a canned response."""

    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    def complete(self, messages, max_tokens=300, context_length=None):
        self.prompts.append(messages[0]["content"])
        return self.response


class _ExplodingProvider:
    """Fails the test if inference is ever actually attempted."""

    def complete(self, messages, max_tokens=300, context_length=None):
        raise AssertionError("provider.complete() must not be called")


class _RaisingProvider:
    """Simulates an LLM call that fails outright (timeout, malformed response, etc)."""

    def complete(self, messages, max_tokens=300, context_length=None):
        raise RuntimeError("simulated LLM failure")


# ── infer_pdf_metadata ───────────────────────────────────────────────────────────

def test_doi_regex_found_without_llm_guess(tmp_path):
    """
    A DOI printed on the page is picked up by the cheap regex, and the LLM
    prompt is NOT asked to guess one (the "Also extract the DOI" instruction
    is only added when the regex misses).
    """
    pdf = _make_pdf(tmp_path / "with_doi.pdf", "A Paper\nDOI: 10.1234/example.5678\nSome body text.")
    provider = _StubProvider('{"title": "A Paper", "authors": "Ada Lovelace"}')

    result = infer_pdf_metadata(pdf, provider)

    assert result["doi"] == "10.1234/example.5678"
    assert "Also extract the DOI" not in provider.prompts[0]
    assert result["title"] == "A Paper"
    assert result["authors"] == "Ada Lovelace"


def test_llm_guesses_doi_when_regex_finds_none(tmp_path):
    """
    When no DOI appears in the extracted text, the prompt asks the LLM to
    find one, and a DOI in the LLM's JSON response is used.
    """
    pdf = _make_pdf(tmp_path / "no_doi.pdf", "A Paper\nNo DOI printed anywhere here.")
    provider = _StubProvider('{"title": "A Paper", "authors": "Ada Lovelace", "doi": "10.9999/guessed"}')

    result = infer_pdf_metadata(pdf, provider)

    assert "Also extract the DOI" in provider.prompts[0]
    assert result["doi"] == "10.9999/guessed"


def test_llm_failure_degrades_to_empty_dict(tmp_path):
    """
    Inference is best-effort: a raising provider must not propagate — title
    and authors are simply left unset (doi falls back to "" since the regex
    found nothing either).
    """
    pdf = _make_pdf(tmp_path / "plain.pdf", "A Paper\nNo DOI printed anywhere here.")
    result = infer_pdf_metadata(pdf, _RaisingProvider())

    assert "title" not in result
    assert "authors" not in result
    assert result["doi"] == ""


# ── resolve_pdf_metadata ─────────────────────────────────────────────────────────

def test_explicit_overrides_always_win(tmp_path):
    """
    When title, authors, and DOI are all given explicitly, inference is
    skipped entirely — the provider must never be called.
    """
    pdf = _make_pdf(tmp_path / "overridden.pdf", "Irrelevant PDF text.")

    result = resolve_pdf_metadata(
        pdf, _ExplodingProvider(), "ollama", doc_type="paper", visibility="public",
        title_override="My Title", authors_override="My Authors", doi_override="10.1/mine",
    )

    assert result == {
        "title": "My Title", "authors": "My Authors", "doi": "10.1/mine", "meta_inferred": False,
    }


def test_resolve_skips_inference_for_private_note_under_anthropic(tmp_path):
    """
    A private note's text must never reach a cloud provider — inference is
    skipped with a visible warning, not silently attempted and then blocked.
    """
    pdf = _make_pdf(tmp_path / "private.pdf", "Confidential lab notebook entry.")

    result = resolve_pdf_metadata(
        pdf, _ExplodingProvider(), "anthropic", doc_type="note", visibility="private",
    )

    assert result["meta_inferred"] is False
    assert result["title"] == ""


def test_resolve_runs_inference_for_public_paper_under_anthropic(tmp_path):
    """
    Papers are always public, so inference under Anthropic is allowed even
    though the same doc_type/visibility combination would be blocked for a note.
    """
    pdf = _make_pdf(tmp_path / "public_paper.pdf", "A Public Paper\nSome body text.")
    provider = _StubProvider('{"title": "A Public Paper", "authors": "Grace Hopper"}')

    result = resolve_pdf_metadata(
        pdf, provider, "anthropic", doc_type="paper", visibility="public",
    )

    assert result["title"] == "A Public Paper"
    assert result["authors"] == "Grace Hopper"
    assert result["meta_inferred"] is True
