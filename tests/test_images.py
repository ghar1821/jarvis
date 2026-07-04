"""
Tests for digest/kb/images.py (figure extraction) and store.add_figures
(vision captioning + indexing).

Figure extraction runs on real generated PDFs (PyMuPDF, cheap, no mocking).
Captioning is the LLM boundary CLAUDE.md sanctions mocking at, so add_figures
is exercised with a fake provider that returns canned captions and counts its
calls — that lets us assert the private+anthropic skip really bypasses the
model, and that a per-figure failure doesn't abort the whole ingest.
"""

import pymupdf
import pytest

from digest.config import reset_config
from digest.kb.images import extract_figures
from digest.kb.store import add_figures, delete_by_metadata


def _png_bytes(width: int, height: int, color=(200, 30, 30)) -> bytes:
    """Build a solid-colour PNG of the given size, entirely in-memory."""
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), False)
    pixmap.set_rect(pixmap.irect, color)
    return pixmap.tobytes("png")


def _pdf_with_images(path, image_specs) -> None:
    """
    Write a one-page-per-image PDF. image_specs is a list of (width, height);
    each image is inserted on its own page so page numbers are predictable.
    Each image gets a distinct colour so PyMuPDF assigns it its own xref
    (identical image bytes would be deduplicated to a single xref).
    """
    doc = pymupdf.open()
    for index, (width, height) in enumerate(image_specs):
        page = doc.new_page()
        color = (30 + index * 40, 60, 200 - index * 30)
        page.insert_image(pymupdf.Rect(0, 0, width, height), stream=_png_bytes(width, height, color))
    doc.save(str(path))
    doc.close()


class _FakeProvider:
    """Canned describe_image; records every call for assertion."""

    def __init__(self, captions=None, fail_on=None):
        self._captions = list(captions or [])
        self._fail_on = set(fail_on or [])
        self.calls = 0

    def describe_image(self, image_bytes: bytes, context: str) -> str:
        from digest.errors import LLMError

        current = self.calls
        self.calls += 1
        if current in self._fail_on:
            raise LLMError("vision model choked")
        if self._captions:
            return self._captions[current % len(self._captions)]
        return f"caption {current}"


# ── extract_figures ────────────────────────────────────────────────────────────

def test_extract_figures_keeps_large_and_drops_tiny(tmp_path):
    """
    A large embedded image is returned with its 1-indexed page; a tiny decoy
    below min_pixels is filtered out.
    """
    pdf = tmp_path / "figs.pdf"
    _pdf_with_images(pdf, [(300, 300), (8, 8)])  # 90000 px kept, 64 px dropped

    figures = extract_figures(pdf, max_figures=20, min_pixels=40000)
    assert len(figures) == 1
    assert figures[0]["page"] == 1
    assert figures[0]["image_bytes"][:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic


def test_extract_figures_honours_max(tmp_path):
    """max_figures caps how many candidates come back."""
    pdf = tmp_path / "many.pdf"
    _pdf_with_images(pdf, [(300, 300), (300, 300), (300, 300)])

    figures = extract_figures(pdf, max_figures=2, min_pixels=40000)
    assert len(figures) == 2


# ── add_figures ────────────────────────────────────────────────────────────────

def test_add_figures_indexes_caption_chunks(tmp_path, store):
    """
    Each figure becomes one chunk: page_content prefixed [FIGURE p.N], metadata
    annotation_kind="figure" with the page number.
    """
    pdf = tmp_path / "paper.pdf"
    _pdf_with_images(pdf, [(300, 300)])
    provider = _FakeProvider(captions=["A scatter plot of X vs Y."])

    ids = add_figures(
        pdf, doc_type="paper", visibility="public", source=pdf.as_uri(),
        provider_obj=provider, provider_str="ollama", title="My Paper",
        file_path=str(pdf), store=store,
    )
    assert len(ids) == 1
    assert provider.calls == 1

    stored = store._collection.get(where={"annotation_kind": {"$eq": "figure"}}, include=["metadatas", "documents"])
    assert stored["documents"][0].startswith("[FIGURE p.1]")
    assert stored["metadatas"][0]["annotation_kind"] == "figure"
    assert stored["metadatas"][0]["page"] == 1


def test_add_figures_private_anthropic_is_skipped(tmp_path, store, capsys):
    """
    A private note under the cloud provider skips captioning entirely: no
    chunk written, the vision model is never called, and a warning is printed.
    """
    pdf = tmp_path / "secret.pdf"
    _pdf_with_images(pdf, [(300, 300)])
    provider = _FakeProvider(captions=["should never run"])

    ids = add_figures(
        pdf, doc_type="note", visibility="private", source=pdf.as_uri(),
        provider_obj=provider, provider_str="anthropic", title="Secret",
        file_path=str(pdf), store=store,
    )
    assert ids == []
    assert provider.calls == 0
    assert "skipping figure captioning" in capsys.readouterr().out


def test_add_figures_tolerates_per_figure_failure(tmp_path, store):
    """A caption failure on one figure skips only that figure, not the ingest."""
    pdf = tmp_path / "three.pdf"
    _pdf_with_images(pdf, [(300, 300), (300, 300), (300, 300)])
    # Fail on the middle figure (index 1); the other two succeed.
    provider = _FakeProvider(captions=["ok"], fail_on=[1])

    ids = add_figures(
        pdf, doc_type="paper", visibility="public", source=pdf.as_uri(),
        provider_obj=provider, provider_str="ollama", title="Paper",
        file_path=str(pdf), store=store,
    )
    assert len(ids) == 2
    assert provider.calls == 3


def test_add_figures_respects_kill_switch(tmp_path, store, monkeypatch):
    """figure_captions=false disables captioning without touching the provider."""
    monkeypatch.setenv("OLLAMA_MODEL", "unused")  # keep env clean-ish
    pdf = tmp_path / "off.pdf"
    _pdf_with_images(pdf, [(300, 300)])
    provider = _FakeProvider(captions=["nope"])

    # Turn the kill-switch off by monkeypatching the loaded config.
    import digest.kb.store as store_mod
    real_get_config = store_mod.get_config

    class _Cfg:
        figure_captions = False
        figure_max_per_doc = 20
        figure_min_pixels = 40000

    monkeypatch.setattr(store_mod, "get_config", lambda: _Cfg())
    try:
        ids = add_figures(
            pdf, doc_type="paper", visibility="public", source=pdf.as_uri(),
            provider_obj=provider, provider_str="ollama", title="Paper",
            file_path=str(pdf), store=store,
        )
    finally:
        monkeypatch.setattr(store_mod, "get_config", real_get_config)
    assert ids == []
    assert provider.calls == 0


def test_delete_by_source_sweeps_figures(tmp_path, store):
    """Figure chunks share source with the body, so a source delete removes them."""
    pdf = tmp_path / "paper.pdf"
    _pdf_with_images(pdf, [(300, 300)])
    provider = _FakeProvider(captions=["a plot"])

    add_figures(
        pdf, doc_type="paper", visibility="public", source=pdf.as_uri(),
        provider_obj=provider, provider_str="ollama", title="Paper",
        file_path=str(pdf), store=store,
    )
    assert store._collection.get(where={"annotation_kind": {"$eq": "figure"}})["ids"]

    delete_by_metadata("source", pdf.as_uri(), store)
    assert store._collection.get(where={"annotation_kind": {"$eq": "figure"}})["ids"] == []
