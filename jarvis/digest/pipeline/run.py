"""Main pipeline: fetch → deduplicate → score → write digest → index knowledge base."""

import tempfile
from datetime import datetime
from pathlib import Path

import requests

from jarvis.core.config import get_config
from jarvis.core.errors import ConversionError
from jarvis.core.llm import active_model, make_provider

from ..arxiv.convert import download_arxiv_pdf, parse_arxiv_url
from ..arxiv.fetch import deduplicate, fetch_arxiv
from ..biorxiv.fetch import fetch_biorxiv, fetch_biorxiv_keywords
from .format import format_digest
from .score import filter_and_score

PROMPT_PATH = Path(__file__).parent / "prompts" / "prompt_filter_score.md"


def _summary_fallback(paper: dict, selected: dict, store) -> str:
    """
    Index a must-read paper as a summary entry built from the scoring run's
    own summary+why text — zero extra LLM calls. Used when full text can't be
    fetched (bioRxiv DOI link, download or conversion failure).
    """
    from jarvis.kb.store import add_paper

    dense_summary = "\n\n".join(
        filter(None, [selected.get("summary", ""), selected.get("why", "")])
    )
    add_paper(
        paper=paper,
        dense_summary=dense_summary,
        score=selected.get("score", 0),
        track=selected.get("track", ""),
        store=store,
        storage_mode="summary",
    )
    return "added_summary"


def ingest_full_text_paper(paper: dict, selected: dict, provider, store) -> str:
    """
    Index one must-read (score >= 9) paper with its full text.

    Returns one of:
      "skipped"         — already in the knowledge base (source URL or title)
      "added_full_text" — arXiv PDF downloaded, converted, and fully chunked
      "added_summary"   — fell back to a summary entry: the link is not an
                          arXiv URL (bioRxiv DOI links have no derivable PDF
                          URL), or the download/conversion failed

    No summarize() call is made anywhere in this path — the fallback reuses
    the scoring run's summary+why text. A single 404 or bad PDF must never
    fail the digest job, so fetch failures warn visibly and fall back.
    """
    from jarvis.kb.convert import pdf_to_markdown
    from jarvis.kb.store import _source_exists, _title_exists, add_annotations, add_figures, add_texts

    if _source_exists(paper.get("link", ""), store) or _title_exists(paper.get("title", ""), store):
        return "skipped"

    arxiv_id = parse_arxiv_url(paper.get("link", ""))
    if not arxiv_id:
        # bioRxiv papers link via doi.org, which doesn't lead to a versioned
        # PDF URL we could download. A download_biorxiv_pdf helper is future
        # work; until then bioRxiv must-reads get the summary entry.
        return _summary_fallback(paper, selected, store)

    cfg = get_config()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = download_arxiv_pdf(arxiv_id, Path(tmp))
            full_text = pdf_to_markdown(pdf_path)
            # Annotations first, matching the other ingest paths (a fresh
            # arXiv download has none, but the order costs nothing). Figures
            # follow the config default (off unless figure_captions is on).
            add_annotations(
                pdf_path, doc_type="paper", visibility="public",
                source=paper["link"], title=paper.get("title", ""), store=store,
            )
            add_figures(
                pdf_path, doc_type="paper", visibility="public",
                source=paper["link"], provider_obj=provider,
                provider_str=cfg.provider, title=paper.get("title", ""),
                store=store, enabled=None,
            )
    except (requests.RequestException, ConversionError, OSError, RuntimeError) as exc:
        # OSError covers a disk-full download; RuntimeError (and its
        # subclass fitz.FileDataError) covers pymupdf choking on a corrupt
        # PDF. Neither should ever abort the whole indexing pass over one
        # bad paper — fall back to the summary entry like every other
        # download/convert failure.
        print(f"  ⚠️  full-text fetch failed for {paper.get('title', '?')!r}: {exc} "
              "— falling back to a summary entry", flush=True)
        return _summary_fallback(paper, selected, store)

    authors = paper.get("authors", "")
    title = paper.get("title", "")
    add_texts(
        content=full_text,
        doc_type="paper",
        visibility="public",
        source=paper["link"],
        extra_metadata={
            "title": title,
            "authors": authors,
            "doi": paper.get("doi", ""),
            "score": int(selected.get("score", 0)),
            "track": str(selected.get("track", "")),
            "storage_mode": "full_text",
        },
        embed_header=(f"{title} — {authors}" if authors else title),
        store=store,
    )
    return "added_full_text"


def index_scored_papers(selected: list[dict], papers: list[dict], provider, store) -> None:
    """
    Index the scored papers by tier:
      score >= 9   — full text (summary fallback when the PDF can't be fetched)
      8 <= s < 9   — summary entry from the scoring run's own text (no LLM call)
      score < 8    — not indexed per-paper; still discoverable through the
                     indexed digest document itself (see index_digest_file)
    """
    from jarvis.kb.store import add_papers_batch

    must_reads = [s for s in selected if s["score"] >= 9]
    if must_reads:
        outcome_counts = {"added_full_text": 0, "added_summary": 0, "skipped": 0}
        for s in must_reads:
            outcome = ingest_full_text_paper(papers[s["index"]], s, provider, store)
            outcome_counts[outcome] += 1
        print(
            f"  score >= 9: {outcome_counts['added_full_text']} full text, "
            f"{outcome_counts['added_summary']} summary fallback, "
            f"{outcome_counts['skipped']} already in knowledge base",
            flush=True,
        )
    else:
        print("  No papers scored >= 9 this run", flush=True)

    worth_reading = [s for s in selected if 8 <= s["score"] < 9]
    if worth_reading:
        entries = [(papers[s["index"]], s) for s in worth_reading]
        added, skipped = add_papers_batch(entries, store)
        print(
            f"  score 8-8.9: {added} summaries added, {skipped} already in knowledge base",
            flush=True,
        )


def index_digest_file(digest_text: str, output_path: Path, today: datetime, store) -> list[str]:
    """
    Index the weekly digest .md itself as doc_type="digest", so every paper it
    mentions — including the < 8 tier that never gets its own entry — is
    searchable, with a file:// link back to the digest on disk.

    A dedicated doc_type (not "note") is deliberate: refresh_vault deletes
    note entries whose vault-relative path no longer exists, and an absolute
    digest path would look exactly like that and get wiped on the next sync.
    """
    from jarvis.kb.store import add_texts

    resolved = output_path.resolve()
    return add_texts(
        content=digest_text,
        doc_type="digest",
        visibility="public",
        source=resolved.as_uri(),
        extra_metadata={
            "title": f"Paper Digest — {today:%Y-%m-%d}",
            "file_path": str(resolved),
            "storage_mode": "full_text",
        },
        store=store,
    )


def main() -> None:
    cfg = get_config()
    today = datetime.today()
    datetime_str = today.strftime("%Y-%m-%d_%H-%M")

    provider = make_provider(cfg.provider)

    print("Fetching arXiv...", flush=True)
    all_papers = []
    for cat, n in cfg.arxiv_cats:
        print(f"  {cat} ({n})", flush=True)
        all_papers.extend(fetch_arxiv(cat, n))

    print("Fetching bioRxiv...", flush=True)
    for cat, n in cfg.biorxiv_cats:
        print(f"  category {cat} ({n})", flush=True)
        all_papers.extend(fetch_biorxiv(cat, n, days=cfg.biorxiv_days))
    for keyword, n in cfg.biorxiv_keywords:
        print(f"  keyword {keyword!r} ({n})", flush=True)
        all_papers.extend(fetch_biorxiv_keywords([keyword], n, days=cfg.biorxiv_days))

    print(f"Deduplicating {len(all_papers)} papers...", flush=True)
    all_papers = deduplicate(all_papers)
    print(f"  {len(all_papers)} unique papers", flush=True)

    print("Asking LLM to filter and score...", flush=True)
    result = filter_and_score(all_papers, provider, cfg.max_results, PROMPT_PATH)
    selected = result["selected"]
    print(f"  {len(selected)} papers selected", flush=True)

    print("Writing digest...", flush=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = cfg.output_dir / f"digest-{datetime_str}.md"
    model_label = active_model(cfg)
    digest = format_digest(selected, all_papers, model_label, today, datetime_str)
    output_path.write_text(digest)
    print(f"  Written to {output_path}", flush=True)

    from jarvis.kb.store import get_store

    store = get_store()

    print("Indexing digest document...", flush=True)
    index_digest_file(digest, output_path, today, store)

    print("Adding high-score papers to knowledge base...", flush=True)
    index_scored_papers(selected, all_papers, provider, store)


if __name__ == "__main__":
    main()
