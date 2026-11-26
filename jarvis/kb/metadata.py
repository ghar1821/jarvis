"""
Title, authors, DOI and arXiv ID for a PDF in the papers folder, without an LLM.

A PDF arrives with nothing but its bytes, so resolve_pdf_metadata() works
down a list of sources and keeps the first answer for each field:

  1. Identifiers, found by regex in the first pages. Most published papers
     print a DOI. arXiv preprints usually print no DOI, but arXiv stamps
     their ID (e.g. "arXiv:2411.01234v2 [cs.LG]") in the first page's margin.
  2. Crossref: given a DOI, its public API returns the canonical title and
     author list. Only the DOI is sent, with a generic User-Agent.
  3. arXiv: given an arXiv ID and no Crossref answer, the arXiv API returns
     the title and authors. Only the ID is sent.
  4. The PDF's own embedded metadata (title/author), which publishers and
     LaTeX often fill in, though sometimes with junk like "untitled".
  5. The filename, as the title of last resort.

The arXiv ID is also what lets `fetch_recent_papers` recognise an arXiv
paper that is already in the papers folder, since the arXiv feed rarely has
a DOI to match on.

Every step is best-effort: a failure is logged and the next source is tried,
so a paper is always indexed, at worst under its filename.
"""

import re
from pathlib import Path

import arxiv
import pymupdf
import requests

from jarvis.core.logs import get_logger

log = get_logger(__name__)

_DOI_RE = re.compile(r"10\.\d{4,9}/\S+")

# arXiv IDs in both formats: the current "2411.01234" and the pre-2007
# "hep-th/9901001". The version suffix ("v2") is matched but not kept, so
# every version of a preprint counts as the same paper.
_ARXIV_ID = r"(\d{4}\.\d{4,5}|[a-z][a-z\-]*(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?"
# The margin stamp arXiv adds to every PDF: the ID followed by its category.
_ARXIV_STAMP_RE = re.compile(rf"arXiv:\s*{_ARXIV_ID}\s*\[[\w.\-]+\]", re.IGNORECASE)
# Any "arXiv:<id>" mention. Used only when there is no stamp, because a first
# page can also cite other preprints.
_ARXIV_MENTION_RE = re.compile(rf"arXiv:\s*{_ARXIV_ID}", re.IGNORECASE)
# arXiv's own DOIs (10.48550/arXiv.<id>) carry the ID too.
_ARXIV_DOI_RE = re.compile(rf"^10\.48550/arXiv\.{_ARXIV_ID}$", re.IGNORECASE)

# Only for title and author lookups while indexing, so retries are few: a
# failed lookup just falls through to the next source.
_arxiv_client = arxiv.Client(page_size=1, delay_seconds=3.0, num_retries=2)

_CROSSREF_URL = "https://api.crossref.org/works/{doi}"
# Generic on purpose: Crossref's "polite pool" asks for a contact email, and
# jarvis never sends anything that identifies its user.
_CROSSREF_HEADERS = {"User-Agent": "jarvis-kb/1.0 (personal knowledge base)"}
_CROSSREF_TIMEOUT = 10

# Embedded titles that are really placeholders left by an authoring tool.
_JUNK_TITLES = {"", "untitled", "microsoft word", "title", "paper"}


def _first_pages_text(pdf_path: Path, max_pages: int) -> str:
    """The text of the first max_pages pages of a PDF."""
    doc = pymupdf.open(pdf_path)
    try:
        return "\n".join(page.get_text() for page in doc[: min(max_pages, len(doc))])
    finally:
        doc.close()


def extract_doi(pdf_path: Path, max_pages: int = 2) -> str:
    """The first DOI printed on the first pages, or "" if there is none."""
    match = _DOI_RE.search(_first_pages_text(pdf_path, max_pages))
    # A DOI printed mid-sentence often drags trailing punctuation with it.
    return match.group(0).rstrip(").,;]") if match else ""


def extract_arxiv_id(pdf_path: Path) -> str:
    """
    The arXiv ID printed on the first page, without its version, or "".

    Only the first page is read, since that is where arXiv puts its stamp.
    The stamp is preferred over other "arXiv:" mentions, which may be
    citations of other preprints.
    """
    first_page = _first_pages_text(pdf_path, max_pages=1)
    match = _ARXIV_STAMP_RE.search(first_page) or _ARXIV_MENTION_RE.search(first_page)
    return match.group(1) if match else ""


def arxiv_id_from_doi(doi: str) -> str:
    """The arXiv ID inside an arXiv DOI (10.48550/arXiv.<id>), or "" for any other DOI."""
    match = _ARXIV_DOI_RE.match(doi)
    return match.group(1) if match else ""


def arxiv_lookup(arxiv_id: str) -> dict:
    """
    {title, authors} for an arXiv ID from the arXiv API, or {} on any failure.

    Authors are joined as "Given Family, Given Family", the same shape as
    crossref_lookup, so author-name search behaves the same for both.
    """
    try:
        results = list(_arxiv_client.results(arxiv.Search(id_list=[arxiv_id])))
    except (arxiv.ArxivError, requests.RequestException) as exc:
        log.warning("arXiv lookup failed for %s: %s", arxiv_id, exc)
        return {}
    if not results:
        log.warning("arXiv has no paper with the ID %s", arxiv_id)
        return {}
    paper = results[0]
    result: dict = {}
    title = " ".join(paper.title.split())
    if title:
        result["title"] = title
    names = [author.name for author in paper.authors if author.name]
    if names:
        result["authors"] = ", ".join(names)
    return result


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
    Resolve {"title", "authors", "doi", "arxiv_id"} for one PDF. Never raises.

    Title always ends up non-empty (the filename stem at worst). Authors, doi
    and arxiv_id are "" when nothing found them.
    """
    try:
        doi = extract_doi(pdf_path)
        arxiv_id = arxiv_id_from_doi(doi) or extract_arxiv_id(pdf_path)
    except Exception as exc:
        log.warning("could not read %s to look for a DOI or arXiv ID: %s", pdf_path.name, exc)
        doi, arxiv_id = "", ""

    from_crossref = crossref_lookup(doi) if doi else {}
    # arXiv is asked only when Crossref had no title, so a published paper
    # keeps its journal title rather than the preprint's.
    from_arxiv = arxiv_lookup(arxiv_id) if arxiv_id and not from_crossref.get("title") else {}

    try:
        from_pdf = embedded_metadata(pdf_path)
    except Exception as exc:
        log.warning("could not read embedded metadata from %s: %s", pdf_path.name, exc)
        from_pdf = {}

    return {
        "title": (from_crossref.get("title") or from_arxiv.get("title")
                  or from_pdf.get("title") or pdf_path.stem),
        "authors": (from_crossref.get("authors") or from_arxiv.get("authors")
                    or from_pdf.get("authors", "")),
        "doi": doi,
        "arxiv_id": arxiv_id,
    }
