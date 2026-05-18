"""Main pipeline: fetch → deduplicate → score → write digest → download must-reads."""

from datetime import datetime
from pathlib import Path

from .fetch import deduplicate, fetch_arxiv
from .format import download_must_reads, format_digest
from .score import filter_and_score

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL = "gemma4:26b"
OUTPUT_DIR = Path("~/Documents/papers/digest").expanduser()
MAX_RESULTS = 10

ARXIV_CATS = [
    ("cs.LG", 150),
    ("cs.AI", 80),
    ("cs.NE", 50),
    ("cs.CV", 80),
    ("cs.CL", 80),
    ("cs.MA", 50),
]

PROMPT_PATH = Path(__file__).parent / "prompt_filter_score.md"

# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    today = datetime.today()
    datetime_str = today.strftime("%Y-%m-%d_%H-%M")

    print("Fetching arXiv...", flush=True)
    all_papers = []
    for cat, n in ARXIV_CATS:
        print(f"  {cat} ({n})", flush=True)
        all_papers.extend(fetch_arxiv(cat, n))

    print(f"Deduplicating {len(all_papers)} papers...", flush=True)
    all_papers = deduplicate(all_papers)
    print(f"  {len(all_papers)} unique papers", flush=True)

    print("Asking LLM to filter and score...", flush=True)
    result = filter_and_score(all_papers, MODEL, MAX_RESULTS, PROMPT_PATH)
    selected = result["selected"]
    print(f"  {len(selected)} papers selected", flush=True)

    print("Writing digest...", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"digest-{datetime_str}.md"
    digest = format_digest(selected, all_papers, MODEL, today, datetime_str)
    output_path.write_text(digest)
    print(f"  Written to {output_path}", flush=True)

    # download_must_reads(selected, all_papers, OUTPUT_DIR, datetime_str)


if __name__ == "__main__":
    main()
