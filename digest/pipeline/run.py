"""Main pipeline: fetch → deduplicate → score → write digest → index knowledge base."""

from datetime import datetime
from pathlib import Path

from ..config import get_config
from ..arxiv.fetch import deduplicate, fetch_arxiv
from ..biorxiv.fetch import fetch_biorxiv, fetch_biorxiv_keywords
from ..llm import make_provider
from .format import format_digest
from .score import filter_and_score

PROMPT_PATH = Path(__file__).parent / "prompts" / "prompt_filter_score.md"


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
    model_label = cfg.anthropic_model if cfg.provider == "anthropic" else cfg.ollama_model
    digest = format_digest(selected, all_papers, model_label, today, datetime_str)
    output_path.write_text(digest)
    print(f"  Written to {output_path}", flush=True)

    print("Adding high-score papers to knowledge base...", flush=True)
    from ..kb.store import add_papers_batch, get_store

    must_reads = [s for s in selected if s["score"] >= 9]
    if must_reads:
        entries = [(all_papers[s["index"]], s) for s in must_reads]
        added, skipped = add_papers_batch(entries, get_store())
        print(
            f"  {added} added, {skipped} already in knowledge base (score >= 9)",
            flush=True,
        )
    else:
        print("  No papers scored >= 9 this run", flush=True)


if __name__ == "__main__":
    main()
