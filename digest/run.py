"""Main pipeline: fetch → deduplicate → score → write digest → index RAG."""

from datetime import datetime
from pathlib import Path

from .config import get_config
from .fetch import deduplicate, fetch_arxiv
from .format import download_must_reads, format_digest
from .llm import make_provider
from .score import filter_and_score

PROMPT_PATH = Path(__file__).parent / "prompts" / "prompt_filter_score.md"


def main() -> None:
    cfg = get_config()
    today = datetime.today()
    datetime_str = today.strftime("%Y-%m-%d_%H-%M")

    # Use a large context window for scoring so all abstracts fit in one call
    provider = make_provider(cfg.provider, options={"num_ctx": 196608})

    print("Fetching arXiv...", flush=True)
    all_papers = []
    for cat, n in cfg.arxiv_cats:
        print(f"  {cat} ({n})", flush=True)
        all_papers.extend(fetch_arxiv(cat, n))

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
    digest = format_digest(selected, all_papers, cfg.ollama_model, today, datetime_str)
    output_path.write_text(digest)
    print(f"  Written to {output_path}", flush=True)

    # download_must_reads(selected, all_papers, cfg.output_dir, datetime_str)

    print("Adding high-score papers to RAG...", flush=True)
    from .rag import add_papers_batch, get_papers_collection

    must_reads = [s for s in selected if s["score"] >= 9]
    if must_reads:
        entries = [(all_papers[s["index"]], s) for s in must_reads]
        added = add_papers_batch(entries, get_papers_collection())
        print(f"  {len(added)} papers added to RAG (score >= 9)", flush=True)
    else:
        print("  No papers scored >= 9 this run", flush=True)


if __name__ == "__main__":
    main()
