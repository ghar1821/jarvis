"""
Tests for jarvis/kb/metadata.py — title, authors and DOI without an LLM.

The order is DOI regex → Crossref → the PDF's embedded metadata → filename,
first answer per field wins, and nothing ever raises. PDFs are generated
in-test with pymupdf (cheap and real); Crossref is the one HTTP boundary, so
requests.get is replaced with a canned response there and nowhere else.
"""

from pathlib import Path

import pymupdf
import requests

from jarvis.kb import metadata
from jarvis.kb.metadata import crossref_lookup, embedded_metadata, extract_doi, resolve_pdf_metadata


def _make_pdf(path: Path, text: str, title: str = "", author: str = "") -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=10)
    if title or author:
        doc.set_metadata({"title": title, "author": author})
    doc.save(path)
    doc.close()
    return path


class _Response:
    """Just enough of requests.Response for crossref_lookup."""

    def __init__(self, payload: dict, status: int = 200):
        self.payload = payload
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"{self.status}")

    def json(self):
        return self.payload


_CROSSREF_WORK = {
    "message": {
        "title": ["Deep  Phenotyping\n of Immune Cells"],
        "author": [
            {"given": "Ada", "family": "Lovelace"},
            {"family": "Turing"},
            {},
        ],
    }
}


# ── extract_doi ────────────────────────────────────────────────────────────────

def test_doi_found_and_trailing_punctuation_stripped(tmp_path):
    """A DOI printed mid-sentence loses the punctuation it drags along."""
    pdf = _make_pdf(tmp_path / "p.pdf", "Published as doi:10.1038/s41586-020-1234-5).")
    assert extract_doi(pdf) == "10.1038/s41586-020-1234-5"


def test_no_doi_gives_empty_string(tmp_path):
    pdf = _make_pdf(tmp_path / "p.pdf", "A preprint with no identifier at all.")
    assert extract_doi(pdf) == ""


# ── crossref_lookup ────────────────────────────────────────────────────────────

def test_crossref_maps_title_and_authors(monkeypatch):
    """
    Whitespace in the title is collapsed, authors are "Given Family" joined by
    commas, and an author with no name at all is skipped.
    """
    requested = {}

    def fake_get(url, headers, timeout):
        requested.update(url=url, headers=headers)
        return _Response(_CROSSREF_WORK)

    monkeypatch.setattr(metadata.requests, "get", fake_get)

    result = crossref_lookup("10.1/abc")

    assert result == {"title": "Deep Phenotyping of Immune Cells", "authors": "Ada Lovelace, Turing"}
    assert requested["url"].endswith("/works/10.1/abc")


def test_crossref_request_carries_nothing_identifying(monkeypatch):
    """
    Crossref's polite pool asks for a contact email; jarvis never sends one.
    Only the DOI leaves the machine.
    """
    seen_headers = {}

    def fake_get(url, headers, timeout):
        seen_headers.update(headers)
        return _Response(_CROSSREF_WORK)

    monkeypatch.setattr(metadata.requests, "get", fake_get)
    crossref_lookup("10.1/abc")

    assert "@" not in " ".join(seen_headers.values())
    assert "mailto" not in " ".join(seen_headers.values()).lower()


def test_crossref_failure_gives_empty_dict(monkeypatch):
    """A network error or a 404 is logged and falls through, never raised."""
    def failing_get(url, headers, timeout):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(metadata.requests, "get", failing_get)
    assert crossref_lookup("10.1/abc") == {}

    monkeypatch.setattr(metadata.requests, "get", lambda url, headers, timeout: _Response({}, 404))
    assert crossref_lookup("10.1/abc") == {}


# ── embedded_metadata ──────────────────────────────────────────────────────────

def test_embedded_metadata_read_and_placeholders_skipped(tmp_path):
    """A real title is used; an authoring tool's "untitled" placeholder is not."""
    good = _make_pdf(tmp_path / "good.pdf", "Body.", title="Spatial Atlas", author="R. Franklin")
    junk = _make_pdf(tmp_path / "junk.pdf", "Body.", title="Untitled")

    assert embedded_metadata(good) == {"title": "Spatial Atlas", "authors": "R. Franklin"}
    assert embedded_metadata(junk) == {}


# ── resolve_pdf_metadata ───────────────────────────────────────────────────────

def test_crossref_wins_over_embedded_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(metadata, "crossref_lookup",
                        lambda doi: {"title": "Canonical Title", "authors": "A. Author"})
    pdf = _make_pdf(tmp_path / "p.pdf", "doi 10.1234/xyz", title="Embedded Title", author="E. Author")

    assert resolve_pdf_metadata(pdf) == {
        "title": "Canonical Title", "authors": "A. Author", "doi": "10.1234/xyz",
    }


def test_embedded_metadata_used_without_a_doi(tmp_path, monkeypatch):
    """No DOI means no Crossref call at all."""
    def must_not_call(doi):
        raise AssertionError("Crossref must not be called without a DOI")

    monkeypatch.setattr(metadata, "crossref_lookup", must_not_call)
    pdf = _make_pdf(tmp_path / "p.pdf", "No identifier.", title="Embedded Title", author="E. Author")

    assert resolve_pdf_metadata(pdf) == {"title": "Embedded Title", "authors": "E. Author", "doi": ""}


def test_filename_is_the_title_of_last_resort(tmp_path, monkeypatch):
    monkeypatch.setattr(metadata, "crossref_lookup", lambda doi: {})
    pdf = _make_pdf(tmp_path / "smith2024-cytometry.pdf", "doi 10.1234/xyz")

    assert resolve_pdf_metadata(pdf) == {"title": "smith2024-cytometry", "authors": "", "doi": "10.1234/xyz"}


def test_an_unreadable_file_still_resolves(tmp_path):
    """A corrupt PDF must not stop indexing; it falls back to its filename."""
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf at all")

    assert resolve_pdf_metadata(broken) == {"title": "broken", "authors": "", "doi": ""}
