"""
Automatic metadata inference for locally-added PDFs.

Local PDFs arrive with nothing but a filename, so infer_pdf_metadata() reads
the first couple of pages and asks the active provider to extract a title
and author list. A DOI is looked for with a regex first — cheap and exact
when present — and the LLM is only asked to guess one when the regex misses.
resolve_pdf_metadata() is the entry point every add-path calls: it layers
explicit user overrides, the private-note privacy guard, and inference, so
the three call sites (kb add, chat add_document, daemon ingest_pdf) share
one policy instead of three copies of it.
"""

import json
import re
from pathlib import Path

import pymupdf

_DOI_RE = re.compile(r"10\.\d{4,9}/\S+")

_EXTRACTION_PROMPT = (
    "Below is text extracted from the first pages of a PDF document. Extract "
    "the paper's title and author list.{doi_instruction} Respond with ONLY a "
    'JSON object of the form {{"title": "...", "authors": "...", "doi": "..."}} '
    "(omit a field — empty string — if you can't find it). No other text."
    "\n\n---\n{excerpt}\n---"
)


def _extract_excerpt(pdf_path: Path, max_pages: int = 2) -> str:
    doc = pymupdf.open(pdf_path)
    try:
        return "\n".join(page.get_text() for page in doc[: min(max_pages, len(doc))])
    finally:
        doc.close()


def _parse_json_object(raw: str) -> dict:
    """
    Models often wrap JSON in commentary or code fences despite instructions
    not to — pull out the first {...} substring rather than requiring the
    whole response to be valid JSON.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def infer_pdf_metadata(pdf_path: Path, provider) -> dict:
    """
    Infer {title, authors, doi} from a local PDF's first pages via one small
    provider.complete() call. Returns only the keys it filled in — always
    includes "doi" (possibly ""). Degrades to {} on any LLM failure —
    inference is best-effort, never fatal to the add.
    """
    excerpt = _extract_excerpt(pdf_path)
    doi_match = _DOI_RE.search(excerpt)
    regex_doi = doi_match.group(0).rstrip(").,;") if doi_match else ""

    doi_instruction = "" if regex_doi else " Also extract the DOI if one appears in the text."
    prompt = _EXTRACTION_PROMPT.format(doi_instruction=doi_instruction, excerpt=excerpt[:4000])
    try:
        raw = provider.complete([{"role": "user", "content": prompt}], max_tokens=300)
        parsed = _parse_json_object(raw)
    except Exception:
        parsed = {}

    result: dict = {}
    if parsed.get("title"):
        result["title"] = str(parsed["title"]).strip()
    if parsed.get("authors"):
        result["authors"] = str(parsed["authors"]).strip()
    result["doi"] = regex_doi or str(parsed.get("doi", "")).strip()
    return result


def resolve_pdf_metadata(
    pdf_path: Path,
    provider,
    provider_str: str,
    doc_type: str,
    visibility: str,
    title_override: str = "",
    authors_override: str = "",
    doi_override: str = "",
) -> dict:
    """
    Resolve title/authors/doi for one local-PDF add, applying in order:
      1. explicit overrides — always win, skip inference entirely if all three given
      2. the privacy guard — a private note's text must never reach a cloud
         provider, so inference is skipped under Anthropic with a visible
         warning, leaving the caller to fall back to the filename stem
      3. automatic inference for whichever fields are still unset

    Returns {"title", "authors", "doi", "meta_inferred"}; title/authors/doi
    default to "" (caller falls back to the filename stem for title).
    meta_inferred is True whenever inference ran (not skipped by overrides or
    the privacy guard) — even if it came back empty, since an unfilled field
    falling back to the filename stem is exactly the case that most needs
    human review.
    """
    if title_override and authors_override and doi_override:
        return {
            "title": title_override, "authors": authors_override,
            "doi": doi_override, "meta_inferred": False,
        }

    if doc_type == "note" and visibility == "private" and provider_str == "anthropic":
        print(
            "  ⚠️  skipping metadata inference — a private note's text must "
            "not reach a cloud provider (switch to the local model to infer "
            "it, or use --title/--authors/--doi to set it manually)",
            flush=True,
        )
        inferred: dict = {}
    else:
        inferred = infer_pdf_metadata(pdf_path, provider)

    return {
        "title": title_override or inferred.get("title", ""),
        "authors": authors_override or inferred.get("authors", ""),
        "doi": doi_override or inferred.get("doi", ""),
        "meta_inferred": bool(inferred) and not (title_override and authors_override and doi_override),
    }
