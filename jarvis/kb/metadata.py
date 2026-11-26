"""
Title, authors and DOI for a PDF in the papers folder, without an LLM.

A PDF arrives with nothing but its bytes, so resolve_pdf_metadata() works
down a list of sources and keeps the first answer for each field:

  1. DOI — a regex over the first pages. Most published papers print it.
  2. Crossref — given a DOI, its public API returns the canonical title and
     author list. Only the DOI is sent, with a generic User-Agent.
  3. The PDF's own embedded metadata (title/author), which publishers and
     LaTeX often fill in, though sometimes with junk like "untitled".
  4. The filename, as the title of last resort.

Every step is best-effort: a failure is logged and the next source is tried,
so a paper is always indexed, at worst under its filename.
"""

import re
from pathlib import Path

import pymupdf
import requests

from jarvis.core.logs import get_logger

log = get_logger(__name__)

_DOI_RE = re.compile(r"10\.\d{4,9}/\S+")

_CROSSREF_URL = "https://api.crossref.org/works/{doi}"
# Generic on purpose: Crossref's "polite pool" asks for a contact email, and
# jarvis never sends anything that identifies its user.
_CROSSREF_HEADERS = {"User-Agent": "jarvis-kb/1.0 (personal knowledge base)"}
_CROSSREF_TIMEOUT = 10

# Embedded titles that are really placeholders left by an authoring tool.
_JUNK_TITLES = {"", "untitled", "microsoft word", "title", "paper"}


def extract_doi(pdf_path: Path, max_pages: int = 2) -> str:
    """The first DOI printed on the first pages, or "" if there is none."""
    doc = pymupdf.open(pdf_path)
    try:
        excerpt = "\n".join(page.get_text() for page in doc[: min(max_pages, len(doc))])
    finally:
        doc.close()
    match = _DOI_RE.search(excerpt)
    # A DOI printed mid-sentence often drags trailing punctuation with it.
    return match.group(0).rstrip(").,;") if match else ""


def crossref_lookup(doi: str) -> dict:
    """
    {title, authors} for a DOI from Crossref, or {} on any failure.

    Authors are joined as "Given Family, Given Family" — the same shape the
    arXiv fetcher produces, so author-name search behaves the same for both.
    """
    try:
        response = requests.get(
            _CROSSREF_URL.format(doi=doi),
            headers=_CROSSREF_HEADERS,
            timeout=_CROSSREF_TIMEOUT,
        )
        response.raise_for_status()
        work = response.json().get("message", {})
    except (requests.RequestException, ValueError) as exc:
        log.warning("Crossref lookup failed for %s: %s", doi, exc)
        return {}

    result: dict = {}
    titles = work.get("title") or []
    if titles and str(titles[0]).strip():
        result["title"] = " ".join(str(titles[0]).split())
    names = [
        " ".join(part for part in (author.get("given", ""), author.get("family", "")) if part)
        for author in work.get("author") or []
    ]
    names = [name for name in names if name]
    if names:
        result["authors"] = ", ".join(names)
    return result


def embedded_metadata(pdf_path: Path) -> dict:
    """{title, authors} from the PDF's own metadata dictionary, skipping placeholders."""
    doc = pymupdf.open(pdf_path)
    try:
        info = doc.metadata or {}
    finally:
        doc.close()
    result: dict = {}
    title = " ".join(str(info.get("title") or "").split())
    if title.lower() not in _JUNK_TITLES:
        result["title"] = title
    author = " ".join(str(info.get("author") or "").split())
    if author:
        result["authors"] = author
    return result


def resolve_pdf_metadata(pdf_path: Path) -> dict:
    """
    Resolve {"title", "authors", "doi"} for one PDF. Never raises.

    Title always ends up non-empty (the filename stem at worst); authors and
    doi are "" when nothing found them.
    """
    try:
        doi = extract_doi(pdf_path)
    except Exception as exc:
        log.warning("could not read %s to look for a DOI: %s", pdf_path.name, exc)
        doi = ""

    from_crossref = crossref_lookup(doi) if doi else {}

    try:
        from_pdf = embedded_metadata(pdf_path)
    except Exception as exc:
        log.warning("could not read embedded metadata from %s: %s", pdf_path.name, exc)
        from_pdf = {}

    return {
        "title": from_crossref.get("title") or from_pdf.get("title") or pdf_path.stem,
        "authors": from_crossref.get("authors") or from_pdf.get("authors", ""),
        "doi": doi,
    }
