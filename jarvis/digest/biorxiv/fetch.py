"""Fetch and map recent preprints from the bioRxiv API.

The bioRxiv "details" endpoint returns preprints posted in a date window, 30
records per page, walked with a numeric cursor:

    https://api.biorxiv.org/details/biorxiv/{start}/{end}/{cursor}/json

A response carries a `messages` block (status + `total` count) and a
`collection` list of records. Each record has plain keys — `title`, `authors`
(semicolon-separated), `abstract`, `doi`, `date`, `category`. Only real
bioRxiv categories can be filtered server-side (an optional `category` query
param); topics bioRxiv has no category for are matched client-side over the
recent window instead. The @with_retries layer covers whole-window failures
(connection resets, empty responses).
"""

from datetime import date, timedelta

import requests

from jarvis.core.errors import FetchError, with_retries

_BASE_URL = "https://api.biorxiv.org/details/biorxiv"
_PAGE_SIZE = 30  # fixed by the bioRxiv API
_TIMEOUT = 20


def _record_to_paper(record: dict, source_label: str) -> dict:
    """Map one bioRxiv record to the shared paper dict shape."""
    doi = str(record.get("doi", "")).strip()
    return {
        "title": str(record.get("title", "")).strip().replace("\n", " "),
        "abstract": str(record.get("abstract", "")).strip(),
        "link": f"https://doi.org/{doi}" if doi else "",
        "authors": str(record.get("authors", "")).strip(),
        "published": str(record.get("date", "")).strip(),
        "source": source_label,
    }


def _window(days: int) -> tuple[str, str]:
    """Return (start, end) ISO dates for the last `days` up to today."""
    today = date.today()
    start = today - timedelta(days=days)
    return start.isoformat(), today.isoformat()


def _fetch_window(start: str, end: str, max_results: int, category: str = "") -> list[dict]:
    """
    Walk the cursor-paginated window and return the raw records (dicts),
    stopping once max_results are collected or the window is exhausted.
    Raises FetchError on network failure or an unusable first page.
    """
    records: list[dict] = []
    cursor = 0
    while len(records) < max_results:
        url = f"{_BASE_URL}/{start}/{end}/{cursor}/json"
        params = {"category": category} if category else None
        try:
            response = requests.get(url, params=params, timeout=_TIMEOUT)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise FetchError(f"bioRxiv fetch failed ({start}..{end}, cursor {cursor}): {exc}") from exc

        batch = payload.get("collection") or []
        # A recent window is never genuinely empty on the first page; treat an
        # empty or malformed first page as a retryable transient failure.
        if not batch and cursor == 0:
            raise FetchError(f"bioRxiv returned no records for {start}..{end} (transient empty window?)")
        records.extend(batch)
        # A short page means the window is exhausted — stop paging.
        if len(batch) < _PAGE_SIZE:
            break
        cursor += _PAGE_SIZE

    return records[:max_results]


@with_retries(max_attempts=4, backoff=3.0, exceptions=(FetchError,))
def fetch_biorxiv(category: str, max_results: int, days: int = 7) -> list[dict]:
    """
    Fetch recent preprints for one server-side bioRxiv category.

    Only real bioRxiv categories (e.g. "bioinformatics") work here — the API
    filters server-side. Returns paper dicts with source "bioRxiv:{category}".
    """
    start, end = _window(days)
    records = _fetch_window(start, end, max_results, category=category)
    return [_record_to_paper(r, f"bioRxiv:{category}") for r in records]


@with_retries(max_attempts=4, backoff=3.0, exceptions=(FetchError,))
def fetch_biorxiv_keywords(keywords: list[str], max_results: int, days: int = 7) -> list[dict]:
    """
    Fetch a recent uncategorised window and keep preprints whose title or
    abstract contains any of the keywords (case-insensitive). Covers topics
    bioRxiv has no category for (cytometry, spatial, scRNA-seq). Each kept
    paper is tagged with the first keyword it matched, and duplicates (a paper
    matched under two keywords, or the same DOI twice) are removed by DOI.
    """
    start, end = _window(days)
    # Pull a wide window uncategorised, then filter locally. max_results here
    # bounds how many keyword matches we return, not how many records we scan,
    # so scan a generous multiple of the window before giving up.
    records = _fetch_window(start, end, max_results * _PAGE_SIZE, category="")

    lowered = [kw.lower() for kw in keywords]
    matched: list[dict] = []
    seen_dois: set[str] = set()
    for record in records:
        haystack = f"{record.get('title', '')} {record.get('abstract', '')}".lower()
        keyword = next((kw for kw, low in zip(keywords, lowered) if low in haystack), None)
        if keyword is None:
            continue
        doi = str(record.get("doi", "")).strip()
        if doi and doi in seen_dois:
            continue
        seen_dois.add(doi)
        matched.append(_record_to_paper(record, f"bioRxiv:{keyword}"))
        if len(matched) >= max_results:
            break

    return matched
