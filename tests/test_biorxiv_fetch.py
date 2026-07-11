"""
Tests for jarvis/digest/biorxiv/fetch.py — bioRxiv retrieval over the details API.

The bioRxiv API is a real network boundary (live HTTP to api.biorxiv.org), so
these tests stub `requests.get` with fake responses shaped exactly like the
real payload observed live: a top-level {"messages": [...], "collection": [...]}
with plain-keyed records (title/authors/abstract/doi/date/category, no rel_
prefix). The retry behaviour under test is ours (the @with_retries layer that
treats an empty first page as transient).

Retries use monkeypatched time.sleep so no real waiting happens.
"""

import pytest

import jarvis.digest.biorxiv.fetch as fetch_mod
import jarvis.core.errors
from jarvis.digest.biorxiv.fetch import fetch_biorxiv, fetch_biorxiv_keywords
from jarvis.core.errors import FetchError


def _record(title="A Preprint", abstract="An abstract.", doi="10.1101/2026.01.01.123456",
            authors="Lovelace, A.; Turing, A.", date="2026-07-01", category="bioinformatics"):
    """One bioRxiv record in the API's real (plain-key) shape."""
    return {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "doi": doi,
        "date": date,
        "category": category,
        "version": "1",
        "type": "new results",
    }


class _FakeResponse:
    def __init__(self, collection):
        self._payload = {
            "messages": [{"status": "ok", "count": len(collection), "total": str(len(collection))}],
            "collection": collection,
        }

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _fake_get_factory(pages):
    """
    Build a requests.get stub that serves successive pages (each a list of
    records) keyed by the cursor in the URL. Records every call.
    """
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append({"url": url, "params": params})
        cursor = int(url.rstrip("/json").rsplit("/", 1)[-1])
        index = cursor // 30
        collection = pages[index] if index < len(pages) else []
        return _FakeResponse(collection)

    return fake_get, calls


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Skip real backoff sleeps in every test in this module."""
    monkeypatch.setattr(jarvis.core.errors.time, "sleep", lambda _: None)


# ── Field mapping ──────────────────────────────────────────────────────────────

def test_record_maps_to_paper_shape(monkeypatch):
    """
    A record maps to the shared paper dict: doi → doi.org link, date →
    published, and the source label carries the category.
    """
    fake_get, _ = _fake_get_factory([[_record()]])
    monkeypatch.setattr(fetch_mod.requests, "get", fake_get)

    papers = fetch_biorxiv("bioinformatics", max_results=5)
    assert len(papers) == 1
    p = papers[0]
    assert p["title"] == "A Preprint"
    assert p["abstract"] == "An abstract."
    assert p["authors"] == "Lovelace, A.; Turing, A."
    assert p["link"] == "https://doi.org/10.1101/2026.01.01.123456"
    assert p["published"] == "2026-07-01"
    assert p["source"] == "bioRxiv:bioinformatics"


# ── Pagination ───────────────────────────────────────────────────────────────

def test_fetch_paginates_across_cursors(monkeypatch):
    """
    A full first page (30 records) triggers a second request; a short second
    page ends paging. The category query param is passed through.
    """
    first_page = [_record(doi=f"10.1101/p{i}", title=f"Paper {i}") for i in range(30)]
    second_page = [_record(doi="10.1101/last", title="Last")]
    fake_get, calls = _fake_get_factory([first_page, second_page])
    monkeypatch.setattr(fetch_mod.requests, "get", fake_get)

    papers = fetch_biorxiv("bioinformatics", max_results=100)
    assert len(papers) == 31
    assert len(calls) == 2
    assert calls[0]["params"] == {"category": "bioinformatics"}
    assert "/0/json" in calls[0]["url"]
    assert "/30/json" in calls[1]["url"]


def test_fetch_stops_at_max_results(monkeypatch):
    """max_results caps the returned list even when more records are available."""
    page = [_record(doi=f"10.1101/p{i}", title=f"Paper {i}") for i in range(30)]
    fake_get, calls = _fake_get_factory([page, page])
    monkeypatch.setattr(fetch_mod.requests, "get", fake_get)

    papers = fetch_biorxiv("bioinformatics", max_results=10)
    assert len(papers) == 10
    # One page of 30 already covers 10, so no second request is made.
    assert len(calls) == 1


# ── Empty window / retry ───────────────────────────────────────────────────────

def test_empty_first_page_is_retried_then_raises(monkeypatch):
    """An empty first page is transient; after max_attempts it raises FetchError."""
    call_count = {"n": 0}

    def always_empty(url, params=None, timeout=None):
        call_count["n"] += 1
        return _FakeResponse([])

    monkeypatch.setattr(fetch_mod.requests, "get", always_empty)
    with pytest.raises(FetchError, match="no records"):
        fetch_biorxiv("bioinformatics", max_results=5)
    assert call_count["n"] == 4  # max_attempts


# ── Keyword matching ───────────────────────────────────────────────────────────

def test_keyword_matches_title_and_abstract_case_insensitive(monkeypatch):
    """
    A keyword matches against title or abstract, case-insensitively; a paper
    matching no keyword is excluded. The source label carries the keyword.
    """
    records = [
        _record(doi="10.1101/a", title="Spatial Transcriptomics of the brain", abstract="x", category="genomics"),
        _record(doi="10.1101/b", title="Unrelated study", abstract="Uses CYTOMETRY heavily", category="genomics"),
        _record(doi="10.1101/c", title="Nothing relevant", abstract="plain text", category="genomics"),
    ]
    fake_get, _ = _fake_get_factory([records])
    monkeypatch.setattr(fetch_mod.requests, "get", fake_get)

    papers = fetch_biorxiv_keywords(["cytometry", "spatial transcriptomics"], max_results=10)
    dois = {p["link"] for p in papers}
    assert dois == {"https://doi.org/10.1101/a", "https://doi.org/10.1101/b"}
    # First matching keyword wins as the source tag.
    by_doi = {p["link"]: p["source"] for p in papers}
    assert by_doi["https://doi.org/10.1101/a"] == "bioRxiv:spatial transcriptomics"
    assert by_doi["https://doi.org/10.1101/b"] == "bioRxiv:cytometry"


def test_keyword_dedupes_by_doi(monkeypatch):
    """A paper matching two keywords is returned once (DOI-deduped)."""
    records = [
        _record(doi="10.1101/dup", title="Spatial transcriptomics with cytometry", abstract="x", category="genomics"),
    ]
    fake_get, _ = _fake_get_factory([records])
    monkeypatch.setattr(fetch_mod.requests, "get", fake_get)

    papers = fetch_biorxiv_keywords(["cytometry", "spatial transcriptomics"], max_results=10)
    assert len(papers) == 1


def test_keyword_empty_first_page_raises(monkeypatch):
    """The keyword fetch shares the empty-window retry path."""
    monkeypatch.setattr(fetch_mod.requests, "get", lambda url, params=None, timeout=None: _FakeResponse([]))
    with pytest.raises(FetchError, match="no records"):
        fetch_biorxiv_keywords(["cytometry"], max_results=5)
