"""
Tests for jarvis/kb/metadata.py — title, authors and DOI without an LLM.

The order is identifiers by regex (DOI, arXiv ID) → Crossref → arXiv → the
PDF's embedded metadata → filename, first answer per field wins, and nothing
ever raises. PDFs are generated in-test with pymupdf (cheap and real).
Crossref and arXiv are the HTTP boundaries, so only they are replaced with
canned responses.
"""

from pathlib import Path

import arxiv
import pymupdf
import requests

from jarvis.kb import metadata
from jarvis.kb.metadata import (
    arxiv_id_from_doi,
    arxiv_lookup,
    crossref_lookup,
    embedded_metadata,
    extract_arxiv_id,
    extract_doi,
    resolve_pdf_metadata,
)


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
        "title": "Canonical Title", "authors": "A. Author", "doi": "10.1234/xyz", "arxiv_id": "",
    }


def test_embedded_metadata_used_without_a_doi(tmp_path, monkeypatch):
    """No DOI means no Crossref call at all."""
    def must_not_call(doi):
        raise AssertionError("Crossref must not be called without a DOI")

    monkeypatch.setattr(metadata, "crossref_lookup", must_not_call)
    pdf = _make_pdf(tmp_path / "p.pdf", "No identifier.", title="Embedded Title", author="E. Author")

    assert resolve_pdf_metadata(pdf) == {"title": "Embedded Title", "authors": "E. Author", "doi": "", "arxiv_id": ""}


def test_filename_is_the_title_of_last_resort(tmp_path, monkeypatch):
    monkeypatch.setattr(metadata, "crossref_lookup", lambda doi: {})
    pdf = _make_pdf(tmp_path / "smith2024-cytometry.pdf", "doi 10.1234/xyz")

    assert resolve_pdf_metadata(pdf) == {
        "title": "smith2024-cytometry", "authors": "", "doi": "10.1234/xyz", "arxiv_id": "",
    }


def test_an_unreadable_file_still_resolves(tmp_path):
    """A corrupt PDF must not stop indexing; it falls back to its filename."""
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf at all")

    assert resolve_pdf_metadata(broken) == {"title": "broken", "authors": "", "doi": "", "arxiv_id": ""}


# ── arXiv IDs ──────────────────────────────────────────────────────────────────
# arXiv preprints rarely print a DOI, so the arXiv ID is what lets
# has_paper recognise one already in the papers folder.

def test_arxiv_stamp_found_without_its_version(tmp_path):
    pdf = _make_pdf(tmp_path / "p.pdf", "arXiv:2411.01234v2  [cs.LG]  3 Nov 2024")
    assert extract_arxiv_id(pdf) == "2411.01234"


def test_arxiv_stamp_preferred_over_a_cited_preprint(tmp_path):
    """A first page can cite other preprints; the margin stamp names this one."""
    pdf = _make_pdf(
        tmp_path / "p.pdf",
        "Building on arXiv:2301.00001, we ...\narXiv:2411.01234v1 [q-bio.QM] 1 Nov 2024",
    )
    assert extract_arxiv_id(pdf) == "2411.01234"


def test_old_style_arxiv_id_found(tmp_path):
    pdf = _make_pdf(tmp_path / "p.pdf", "arXiv:hep-th/9901001v3 [hep-th]")
    assert extract_arxiv_id(pdf) == "hep-th/9901001"


def test_no_arxiv_id_gives_empty_string(tmp_path):
    pdf = _make_pdf(tmp_path / "p.pdf", "doi: 10.1038/s41586-020-1234-5")
    assert extract_arxiv_id(pdf) == ""


def test_arxiv_id_read_from_an_arxiv_doi():
    assert arxiv_id_from_doi("10.48550/arXiv.2411.01234") == "2411.01234"
    assert arxiv_id_from_doi("10.1038/s41586-020-1234-5") == ""


def test_arxiv_lookup_maps_title_and_authors(monkeypatch):
    """Only the ID goes to arXiv; whitespace in the title is collapsed."""
    requested = []

    def fake_results(search):
        requested.append(search.id_list)
        return iter([arxiv.Result(
            entry_id="http://arxiv.org/abs/2411.01234v2",
            title="Sparse  Autoencoders\n for Cells",
            authors=[arxiv.Result.Author("Ada Lovelace"), arxiv.Result.Author("Alan Turing")],
        )])

    monkeypatch.setattr(metadata._arxiv_client, "results", fake_results)

    assert arxiv_lookup("2411.01234") == {
        "title": "Sparse Autoencoders for Cells", "authors": "Ada Lovelace, Alan Turing",
    }
    assert requested == [["2411.01234"]]


def test_arxiv_lookup_failure_gives_empty_dict(monkeypatch):
    def failing_results(search):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(metadata._arxiv_client, "results", failing_results)
    assert arxiv_lookup("2411.01234") == {}


def test_arxiv_title_used_when_there_is_no_doi(tmp_path, monkeypatch):
    """An arXiv preprint saved as "2411.01234v2.pdf" gets its real title, not the filename."""
    monkeypatch.setattr(metadata, "arxiv_lookup",
                        lambda arxiv_id: {"title": "Sparse Autoencoders", "authors": "A. Lovelace"})
    pdf = _make_pdf(tmp_path / "2411.01234v2.pdf", "arXiv:2411.01234v2 [cs.LG] 3 Nov 2024")

    assert resolve_pdf_metadata(pdf) == {
        "title": "Sparse Autoencoders", "authors": "A. Lovelace", "doi": "", "arxiv_id": "2411.01234",
    }


def test_journal_title_wins_over_the_preprint_title(tmp_path, monkeypatch):
    """A published paper with both identifiers keeps Crossref's title; arXiv isn't asked."""
    def must_not_call(arxiv_id):
        raise AssertionError("arXiv must not be asked when Crossref has a title")

    monkeypatch.setattr(metadata, "crossref_lookup", lambda doi: {"title": "Journal Title"})
    monkeypatch.setattr(metadata, "arxiv_lookup", must_not_call)
    pdf = _make_pdf(tmp_path / "p.pdf", "arXiv:2411.01234v2 [cs.LG]\ndoi 10.1234/xyz")

    result = resolve_pdf_metadata(pdf)
    assert result["title"] == "Journal Title"
    assert result["arxiv_id"] == "2411.01234"
