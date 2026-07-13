"""
Tests for jarvis/kb/metadata.py — automatic metadata inference for local PDFs.

infer_pdf_metadata() reads a PDF's first pages and asks the active provider to
extract title/authors/DOI in one call; a DOI regex runs first since it is
cheap and exact when a DOI is actually printed on the page.
resolve_pdf_metadata() is the policy layer every add-path shares: explicit
overrides win outright, then inference for whatever is still unset. Local
PDFs are always public papers, so there is no private-note guard here — that
machinery lives entirely with vault notes instead.

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
        pdf, _ExplodingProvider(),
        title_override="My Title", authors_override="My Authors", doi_override="10.1/mine",
    )

    assert result == {"title": "My Title", "authors": "My Authors", "doi": "10.1/mine"}


def test_resolve_runs_inference_for_public_paper_under_anthropic(tmp_path):
    """
    Local PDFs are always public papers, so inference is always allowed —
    there is no private-note guard to skip it, even under Anthropic.
    """
    pdf = _make_pdf(tmp_path / "public_paper.pdf", "A Public Paper\nSome body text.")
    provider = _StubProvider('{"title": "A Public Paper", "authors": "Grace Hopper"}')

    result = resolve_pdf_metadata(pdf, provider)

    assert result["title"] == "A Public Paper"
    assert result["authors"] == "Grace Hopper"
