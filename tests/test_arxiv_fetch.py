"""
Tests for jarvis/digest/arxiv/fetch.py — arXiv retrieval via the `arxiv` package.

The arxiv.Client is a genuine network boundary (live HTTP to export.arxiv.org),
so these tests stub `_client.results` and construct arxiv.Result objects
directly. The retry behaviour under test is ours (the @with_retries layer that
treats empty feeds as transient), not the library's internal paging retries.

Retries in these tests use monkeypatched time.sleep so no real waiting happens.
"""

from datetime import datetime, timezone

import arxiv
import pytest

import jarvis.digest.arxiv.fetch as fetch_mod
import jarvis.core.errors
from jarvis.digest.arxiv.fetch import _to_paper, deduplicate, fetch_arxiv, fetch_arxiv_paper
from jarvis.core.errors import FetchError


def _make_result(title="A Paper", summary="An abstract.", primary_category="cs.LG", doi=""):
    result = arxiv.Result(
        entry_id="http://arxiv.org/abs/2301.07041v1",
        title=title,
        summary=summary,
        authors=[arxiv.Result.Author("Ada Lovelace"), arxiv.Result.Author("Alan Turing")],
        published=datetime(2023, 1, 17, tzinfo=timezone.utc),
        updated=datetime(2023, 1, 17, tzinfo=timezone.utc),
        primary_category=primary_category,
        categories=[primary_category],
        doi=doi,
    )
    return result


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Skip real backoff sleeps in every test in this module."""
    monkeypatch.setattr(jarvis.core.errors.time, "sleep", lambda _: None)


def test_to_paper_maps_result_fields():
    """
    _to_paper converts an arxiv.Result into the pipeline's paper dict shape.

    Input:  a Result with title (containing a newline), summary, two authors,
            and a published date
    Expected output: dict with cleaned title, abstract, link, joined authors,
            ISO date, and the given source label
    """
    result = _make_result(title="A\nPaper", summary="  An abstract.  ")
    paper = _to_paper(result, "arXiv:cs.LG")

    assert paper["title"] == "A Paper"
    assert paper["abstract"] == "An abstract."
    assert paper["link"] == "http://arxiv.org/abs/2301.07041v1"
    assert paper["authors"] == "Ada Lovelace, Alan Turing"
    assert paper["published"] == "2023-01-17"
    assert paper["source"] == "arXiv:cs.LG"
    assert paper["doi"] == ""  # arxiv.Result.doi defaults to "" when the API has none


def test_to_paper_surfaces_doi_when_arxiv_result_has_one():
    """
    Some arXiv entries carry a DOI (e.g. once published in a journal) —
    _to_paper must pass it straight through into the paper dict.
    """
    result = _make_result(doi="10.1234/example.doi")
    paper = _to_paper(result, "arXiv:cs.LG")
    assert paper["doi"] == "10.1234/example.doi"


def test_fetch_arxiv_returns_papers(monkeypatch):
    """
    A search yielding results maps each into a paper dict.

    Input:  client stub returning two Results
    Expected output: two paper dicts labelled with the category
    """
    monkeypatch.setattr(
        fetch_mod._client, "results", lambda search: iter([_make_result(), _make_result(title="B")])
    )
    papers = fetch_arxiv("cs.LG", 2)
    assert len(papers) == 2
    assert papers[0]["source"] == "arXiv:cs.LG"


def test_fetch_arxiv_empty_feed_is_retried_then_raises(monkeypatch):
    """
    arXiv sometimes returns an empty feed with HTTP 200. A persistently empty
    feed must exhaust the retries and surface as FetchError, not silently
    yield zero papers.

    Input:  client stub always returning no results
    Expected output: FetchError after exactly 4 attempts (max_attempts)
    """
    calls = 0

    def empty_results(search):
        nonlocal calls
        calls += 1
        return iter([])

    monkeypatch.setattr(fetch_mod._client, "results", empty_results)
    with pytest.raises(FetchError, match="no entries"):
        fetch_arxiv("cs.LG", 5)
    assert calls == 4


def test_fetch_arxiv_recovers_when_empty_feed_is_transient(monkeypatch):
    """
    An empty feed on the first attempt followed by a populated one must
    succeed via the retry layer.

    Input:  client stub returning [] once, then one Result
    Expected output: one paper, two client calls
    """
    calls = 0

    def flaky_results(search):
        nonlocal calls
        calls += 1
        if calls == 1:
            return iter([])
        return iter([_make_result()])

    monkeypatch.setattr(fetch_mod._client, "results", flaky_results)
    papers = fetch_arxiv("cs.LG", 5)
    assert len(papers) == 1
    assert calls == 2


def test_fetch_arxiv_wraps_library_errors_as_fetch_error(monkeypatch):
    """
    arxiv.ArxivError from the client must be wrapped into FetchError so
    callers only ever handle the domain exception.

    Input:  client stub raising arxiv.UnexpectedEmptyPageError on every call
    Expected output: FetchError naming the category
    """

    def broken_results(search):
        raise arxiv.UnexpectedEmptyPageError("http://example", 3, None)

    monkeypatch.setattr(fetch_mod._client, "results", broken_results)
    with pytest.raises(FetchError, match="cs.LG"):
        fetch_arxiv("cs.LG", 5)


def test_fetch_arxiv_paper_by_id(monkeypatch):
    """
    Single-paper fetch strips the version suffix and derives source from the
    primary category.

    Input:  '2301.07041v2', client stub returning one cs.CL Result
    Expected output: paper dict with source 'arXiv:cs.CL'; the id_list the
            client saw contains the version-stripped id
    """
    seen_searches = []

    def capture_results(search):
        seen_searches.append(search)
        return iter([_make_result(primary_category="cs.CL")])

    monkeypatch.setattr(fetch_mod._client, "results", capture_results)
    paper = fetch_arxiv_paper("2301.07041v2")
    assert paper["source"] == "arXiv:cs.CL"
    assert seen_searches[0].id_list == ["2301.07041"]


def test_fetch_arxiv_paper_unknown_id_raises(monkeypatch):
    """
    No entry for the given ID surfaces as FetchError after retries.

    Input:  client stub returning no results
    Expected output: FetchError mentioning the ID
    """
    monkeypatch.setattr(fetch_mod._client, "results", lambda search: iter([]))
    with pytest.raises(FetchError, match="2301.99999"):
        fetch_arxiv_paper("2301.99999")


def test_deduplicate_by_title_case_insensitive():
    """
    Papers with the same title (ignoring case/whitespace) collapse to one.

    Input:  three papers, two sharing a title modulo case
    Expected output: two papers, first occurrence kept
    """
    papers = [
        {"title": "Attention Is All You Need"},
        {"title": "attention is all you need "},
        {"title": "Different"},
    ]
    unique = deduplicate(papers)
    assert len(unique) == 2
    assert unique[0]["title"] == "Attention Is All You Need"
