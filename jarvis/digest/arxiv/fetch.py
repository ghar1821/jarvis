"""Fetch and deduplicate papers from arXiv.

Uses the `arxiv` package (lukasschwab/arxiv.py), whose Client exists to work
around the arXiv API's known flakiness: it pages requests, retries responses
that come back empty despite HTTP 200, and enforces the 3-second courtesy
delay arXiv's terms of use ask for. The @with_retries layer on top covers
whole-search failures (connection resets, persistent empty feeds).
"""

import re

import arxiv
import requests

from jarvis.core.errors import FetchError, with_retries

# One shared client so the courtesy delay applies across successive category
# fetches, not just within a single paged search.
_client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)


def _to_paper(result: arxiv.Result, source_label: str) -> dict:
    return {
        "title": result.title.strip().replace("\n", " "),
        "abstract": result.summary.strip(),
        "link": result.entry_id,
        "authors": ", ".join(a.name for a in result.authors),
        "published": result.published.date().isoformat(),
        "source": source_label,
        "doi": result.doi or "",
    }


@with_retries(max_attempts=4, backoff=3.0, exceptions=(FetchError,))
def fetch_arxiv(cat: str, max_results: int) -> list[dict]:
    """Fetch the most recent papers from a single arXiv category."""
    search = arxiv.Search(
        query=f"cat:{cat}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    try:
        results = list(_client.results(search))
    except (arxiv.ArxivError, requests.RequestException) as exc:
        raise FetchError(f"arXiv fetch failed for {cat}: {exc}") from exc
    # arXiv sometimes returns a valid-but-empty feed with HTTP 200; a recent
    # category is never genuinely empty, so treat this as retryable.
    if not results:
        raise FetchError(f"arXiv returned no entries for {cat} (transient empty feed?)")
    return [_to_paper(r, f"arXiv:{cat}") for r in results]


@with_retries(max_attempts=4, backoff=3.0, exceptions=(FetchError,))
def fetch_arxiv_paper(arxiv_id: str) -> dict:
    """
    Fetch metadata for a single paper by arXiv ID (e.g. '2301.07041').

    Returns a paper dict in the same format as fetch_arxiv().
    The 'source' field is derived from the entry's primary category
    (e.g. 'arXiv:cs.LG'), not from the ID prefix.
    """
    clean_id = re.sub(r"v\d+$", "", arxiv_id)
    search = arxiv.Search(id_list=[clean_id])
    try:
        results = list(_client.results(search))
    except (arxiv.ArxivError, requests.RequestException) as exc:
        raise FetchError(f"arXiv fetch failed for ID {arxiv_id}: {exc}") from exc
    if not results:
        raise FetchError(f"No paper found for arXiv ID: {arxiv_id}")
    result = results[0]
    source = f"arXiv:{result.primary_category or 'unknown'}"
    return _to_paper(result, source)


def deduplicate(papers: list[dict]) -> list[dict]:
    """Remove duplicate papers by title (case-insensitive)."""
    seen: set[str] = set()
    unique = []
    for p in papers:
        key = p["title"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique
