"""Fetch and deduplicate papers from arXiv."""

import re
import xml.etree.ElementTree as ET

import requests

from .errors import FetchError, with_retries

_NS = {"atom": "http://www.w3.org/2005/Atom"}
_CATEGORY_RE = re.compile(r"\d{4}\.\d{4,5}")


def _parse_entries(xml_text: str, source_label: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall("atom:entry", _NS):
        papers.append(
            {
                "title": entry.find("atom:title", _NS).text.strip().replace("\n", " "),
                "abstract": entry.find("atom:summary", _NS).text.strip(),
                "link": entry.find("atom:id", _NS).text.strip(),
                "authors": ", ".join(
                    a.find("atom:name", _NS).text
                    for a in entry.findall("atom:author", _NS)
                ),
                "published": entry.find("atom:published", _NS).text[:10],
                "source": source_label,
            }
        )
    return papers


@with_retries(max_attempts=5, backoff=2.0, exceptions=(requests.RequestException,))
def fetch_arxiv(cat: str, max_results: int) -> list[dict]:
    """Fetch the most recent papers from a single arXiv category."""
    url = (
        f"https://export.arxiv.org/api/query"
        f"?search_query=cat:{cat}"
        f"&sortBy=submittedDate&sortOrder=descending"
        f"&max_results={max_results}"
    )
    resp = requests.get(url, timeout=60, headers={"User-Agent": "paper-digest/1.0"})
    if resp.status_code == 429:
        raise requests.RequestException(f"Rate limited by arXiv (429)")
    resp.raise_for_status()
    return _parse_entries(resp.text, f"arXiv:{cat}")


@with_retries(max_attempts=5, backoff=2.0, exceptions=(requests.RequestException,))
def fetch_arxiv_paper(arxiv_id: str) -> dict:
    """
    Fetch metadata for a single paper by arXiv ID (e.g. '2301.07041').

    Returns a paper dict in the same format as fetch_arxiv().
    The 'source' field is derived from the arXiv category tag in the entry
    (e.g. 'arXiv:cs.LG'), not from the ID prefix.
    """
    clean_id = re.sub(r"v\d+$", "", arxiv_id)
    url = f"https://export.arxiv.org/api/query?id_list={clean_id}"
    resp = requests.get(url, timeout=60, headers={"User-Agent": "paper-digest/1.0"})
    if resp.status_code == 429:
        raise requests.RequestException("Rate limited by arXiv (429)")
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    entry = root.find("atom:entry", _NS)
    if entry is None:
        raise FetchError(f"No paper found for arXiv ID: {arxiv_id}")

    # Use the primary category tag for a correct 'source' label (e.g. "arXiv:cs.LG")
    primary_cat = entry.find("arxiv:primary_category", {"arxiv": "http://arxiv.org/schemas/atom"})
    if primary_cat is not None:
        source = f"arXiv:{primary_cat.get('term', 'unknown')}"
    else:
        # Fallback: first category element
        cat_el = entry.find("atom:category", _NS)
        source = f"arXiv:{cat_el.get('term', 'unknown')}" if cat_el is not None else "arXiv:unknown"

    return {
        "title": entry.find("atom:title", _NS).text.strip().replace("\n", " "),
        "abstract": entry.find("atom:summary", _NS).text.strip(),
        "link": entry.find("atom:id", _NS).text.strip(),
        "authors": ", ".join(
            a.find("atom:name", _NS).text for a in entry.findall("atom:author", _NS)
        ),
        "published": entry.find("atom:published", _NS).text[:10],
        "source": source,
    }


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
