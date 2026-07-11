"""`kb add-digest` implementation — imports papers from digest Markdown file(s)."""

import argparse
import re
import sys
from pathlib import Path

from jarvis.kb.store import add_paper, get_store


def _parse_digest_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n---\n", text)
    return [p for block in blocks if block.strip().startswith("###")
            for p in [_parse_paper_block(block.strip())] if p]


def _parse_paper_block(block: str) -> dict | None:
    lines = block.split("\n")
    title_match = re.match(r"^###\s+(.+?)(?:\s*🤖⚠️)?$", lines[0].strip())
    if not title_match:
        return None
    title = title_match.group(1).strip()

    def _field(pattern: str) -> str:
        m = re.search(pattern, block, re.MULTILINE)
        return m.group(1).strip().rstrip("  ") if m else ""

    track = _field(r"\*\*Track:\*\*\s*(.+?)$")
    authors = _field(r"\*\*Authors:\*\*\s*(.+?)$")
    source_line = _field(r"\*\*Source:\*\*\s*(.+?)$")
    parts = [p.strip() for p in source_line.split("·")]
    source = parts[0] if parts else ""
    link = parts[1] if len(parts) > 1 else ""
    published_m = re.search(r"Published\s+(\S+)", parts[2]) if len(parts) > 2 else None
    published = published_m.group(1) if published_m else ""
    score_m = re.search(r"\*\*Relevance:\*\*\s*(\d+)/10", block)
    score = int(score_m.group(1)) if score_m else 0
    why_m = re.search(r"\*\*Why this digest:\*\*\s*\n(.+?)(?=\n\*\*|\Z)", block, re.DOTALL)
    why = why_m.group(1).strip() if why_m else ""
    summary_m = re.search(r"\*\*Summary:\*\*\s*\n(.+?)(?=\n\*\*|\Z)", block, re.DOTALL)
    summary = summary_m.group(1).strip() if summary_m else ""

    return {"title": title, "authors": authors, "link": link, "published": published,
            "source": source, "track": track, "score": score, "why": why, "summary": summary}


def cmd_add_digest(args: argparse.Namespace) -> None:
    path = Path(args.path).expanduser()
    if not path.exists():
        print(f"Error: path does not exist: {path}", file=sys.stderr)
        sys.exit(1)
    files = sorted(path.glob("*.md")) if path.is_dir() else [path]
    if not files:
        print("No .md files found.", file=sys.stderr)
        sys.exit(1)

    store = get_store()
    total_added = total_skipped = total_files = 0

    for f in files:
        papers = _parse_digest_file(f)
        if not papers:
            continue
        total_files += 1
        added = skipped = 0
        for p in papers:
            if p["score"] < args.min_score:
                skipped += 1
                continue
            paper = {k: p[k] for k in ("title", "authors", "link", "published", "source")}
            dense_summary = "\n\n".join(filter(None, [p["summary"], p["why"]]))
            add_paper(paper=paper, dense_summary=dense_summary,
                      score=p["score"], track=p["track"], store=store)
            added += 1
        total_added += added
        total_skipped += skipped
        print(f"  {f.name}: +{added} added, {skipped} below score threshold")

    print(f"\nTotal: {total_added} papers added from {total_files} file(s) "
          f"({total_skipped} skipped, score < {args.min_score})")
