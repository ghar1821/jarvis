"""
kb — local knowledge base manager.

Manages a local vector database of research papers and Obsidian vault notes
that vault-chat draws on during conversations.

Subcommands:
  add <url|path>    Add a paper by arXiv URL or local PDF path
  add-digest <path> Import papers from digest Markdown file(s)
  list              List indexed papers
  stats             Show document and chunk counts
  remove <source>   Remove a document by source URL
  clear             Delete all documents (prompts for confirmation)

  index-vault       Full (re)index of Obsidian vault
  refresh-vault     Incremental update of vault index

Usage examples:
  uv run kb add https://arxiv.org/abs/2406.04093 --score 9 --track "Track 1"
  uv run kb add paper.pdf --provider anthropic
  uv run kb add-digest ~/Documents/papers/digest/
  uv run kb list
  uv run kb stats
  uv run kb remove https://arxiv.org/abs/2301.07041
  uv run kb index-vault
  uv run kb refresh-vault
"""

import argparse
import re
import sys
from pathlib import Path


# ── Add ───────────────────────────────────────────────────────────────────────


def cmd_add(args: argparse.Namespace) -> None:
    from ..config import get_config
    from ..arxiv.convert import parse_arxiv_url
    from ..arxiv.fetch import fetch_arxiv_paper
    from ..llm import make_provider
    from .store import add_paper, add_texts, get_store

    cfg = get_config()
    store = get_store()
    _provider = None

    def get_provider():
        nonlocal _provider
        if _provider is None:
            _provider = make_provider(args.provider or cfg.provider)
        return _provider
    input_str: str = args.input

    if input_str.startswith("http://") or input_str.startswith("https://"):
        arxiv_id = parse_arxiv_url(input_str)
        if not arxiv_id:
            print(f"Error: could not parse arXiv ID from URL: {input_str}", file=sys.stderr)
            sys.exit(1)
        print(f"Fetching metadata for arXiv:{arxiv_id}...")
        paper = fetch_arxiv_paper(arxiv_id)
        print(f"  Title: {paper['title']}")

        if args.full_text:
            import tempfile
            from ..arxiv.convert import convert_pdf, download_arxiv_pdf
            print("Downloading PDF...")
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                pdf_path_dl = download_arxiv_pdf(arxiv_id, tmp_path)
                print("Converting to Markdown (this may take a moment)...")
                convert_pdf(pdf_path_dl, tmp_path)
                md_path = tmp_path / f"{pdf_path_dl.stem}.md"
                if not md_path.exists():
                    print("Error: PDF conversion produced no output.", file=sys.stderr)
                    sys.exit(1)
                full_text = md_path.read_text(encoding="utf-8")
            print("Chunking and indexing full text...")
            from .store import _source_exists
            if _source_exists(paper.get("link", ""), store):
                print(f"Already in knowledge base: {paper['link']}")
            else:
                ids = add_texts(
                    content=full_text,
                    doc_type="paper",
                    visibility="public",
                    source=paper["link"],
                    extra_metadata={
                        "title": paper.get("title", ""),
                        "authors": paper.get("authors", ""),
                        "score": int(args.score),
                        "track": str(args.track),
                        "storage_mode": "full_text",
                    },
                    store=store,
                )
                print(f"Added (full text, {len(ids)} chunks): {paper['link']}")
        else:
            print("Generating summary...")
            summary = get_provider().summarize(paper["title"], paper["abstract"])
            add_paper(paper=paper, dense_summary=summary, score=args.score,
                      track=args.track, store=store, storage_mode="summary")
            print(f"Added (summary): {paper['link']}")

    elif Path(input_str).exists() and Path(input_str).suffix.lower() == ".pdf":
        pdf_path = Path(input_str).resolve()
        title = args.title or pdf_path.stem
        visibility = args.visibility
        doc_type = args.doc_type

        if doc_type == "note":
            # Notes are always indexed as full text; content_hash tracked for refresh
            from ..arxiv.convert import convert_pdf
            import hashlib as _hashlib
            import tempfile
            print(f"Converting PDF note to Markdown ({visibility}): {pdf_path.name}...")
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                convert_pdf(pdf_path, tmp_path)
                md_path = tmp_path / f"{pdf_path.stem}.md"
                if not md_path.exists():
                    print("Error: PDF conversion produced no output.", file=sys.stderr)
                    sys.exit(1)
                full_text = md_path.read_text(encoding="utf-8")
            content_hash = _hashlib.sha256(pdf_path.read_bytes()).hexdigest()
            ids = add_texts(
                content=full_text,
                doc_type="note",
                visibility=visibility,
                source=pdf_path.as_uri(),
                extra_metadata={
                    "title": title,
                    "file_path": str(pdf_path),
                    "content_hash": content_hash,
                    "storage_mode": "full_text",
                },
                store=store,
            )
            print(f"Added note (full text, {len(ids)} chunks): {pdf_path.name} ({visibility})")

        elif args.full_text:
            from ..arxiv.convert import convert_pdf
            import tempfile
            print(f"Converting PDF to Markdown ({visibility}): {pdf_path.name}...")
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                convert_pdf(pdf_path, tmp_path)
                md_path = tmp_path / f"{pdf_path.stem}.md"
                if not md_path.exists():
                    print("Error: PDF conversion produced no output.", file=sys.stderr)
                    sys.exit(1)
                full_text = md_path.read_text(encoding="utf-8")
            ids = add_texts(
                content=full_text,
                doc_type="paper",
                visibility=visibility,
                source=pdf_path.as_uri(),
                extra_metadata={"title": title, "file_path": str(pdf_path), "storage_mode": "full_text"},
                store=store,
            )
            print(f"Added paper (full text, {len(ids)} chunks): {pdf_path.name} ({visibility})")
        else:
            print(f"Generating summary from PDF ({visibility}): {pdf_path.name}...")
            summary = get_provider().summarize(title, pdf_path)
            add_texts(
                content=f"{title}\n\n{summary}",
                doc_type="paper",
                visibility=visibility,
                source=pdf_path.as_uri(),
                extra_metadata={"title": title, "file_path": str(pdf_path), "storage_mode": "summary"},
                store=store,
            )
            print(f"Added paper (summary): {pdf_path.name} ({visibility})")

    else:
        print(f"Error: '{input_str}' is not a valid arXiv URL or PDF path.", file=sys.stderr)
        sys.exit(1)


# ── Add-digest ────────────────────────────────────────────────────────────────


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
    from .store import add_paper, get_store

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


# ── List / stats / remove / clear ─────────────────────────────────────────────


def cmd_list(args: argparse.Namespace) -> None:
    from .store import get_store, list_papers

    papers = list_papers(limit=args.limit, store=get_store())
    if not papers:
        print("No papers in knowledge base.")
        return
    for p in papers:
        chunks = p.get("chunk_count", "?")
        mode = p.get("storage_mode", "summary" if chunks in ("?", 1, 2) else "full_text")
        print(f"[{p.get('score', '?')}/10] {p.get('title', 'untitled')}  [{mode}, {chunks} chunks]")
        print(f"  {p.get('source', 'no source')}  ·  {p.get('date_added', 'N/A')[:10]}")
        print()


def cmd_stats() -> None:
    from .store import count, count_unique_documents, get_store

    store = get_store()
    total_chunks = count(store)
    papers = count_unique_documents("paper", "source", store)
    notes = count_unique_documents("note", "file_path", store)
    print(f"Documents:  {papers} papers · {notes} notes")
    print(f"Chunks:     {total_chunks} total")


def _resolve_local_file(source: str, meta: dict) -> "Path | None":
    """
    Return the local filesystem path for a document, or None if no local file exists.
    - file:/// URI  → the PDF path encoded in the URI
    - vault note    → vault_path / file_path from metadata
    - http(s) URL   → None (arXiv papers have no local file)
    """
    from urllib.parse import urlparse
    if source.startswith("file:///"):
        return Path(urlparse(source).path)
    if meta.get("file_path"):
        from ..config import get_config
        return get_config().vault_path / meta["file_path"]
    return None


def cmd_remove(args: argparse.Namespace) -> None:
    from .store import get_store

    store = get_store()
    result = store._collection.get(
        where={"source": {"$eq": args.source}}, include=["metadatas"]
    )
    ids = result["ids"]
    if not ids:
        print(f"No documents found with source: {args.source}")
        return

    meta = result["metadatas"][0] if result["metadatas"] else {}
    title = meta.get("title", "untitled")
    doc_type = meta.get("doc_type", "document")
    local_file = _resolve_local_file(args.source, meta)

    print(f"  Title:  {title}")
    print(f"  Type:   {doc_type}")
    print(f"  Source: {args.source}")
    print(f"  Chunks: {len(ids)}")
    if args.delete_file:
        if local_file and local_file.exists():
            print(f"  File:   {local_file}  ← will be PERMANENTLY DELETED")
        else:
            print("  File:   no local file found (only database entry will be removed)")
    else:
        print("  Note:   database entry only — no files will be deleted")

    confirm = input("Confirm? [y/N] ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    store.delete(ids)
    print(f"Removed \"{title}\" ({len(ids)} chunk(s)) from the knowledge base.")
    if args.delete_file:
        if local_file and local_file.exists():
            local_file.unlink()
            print(f"Deleted file: {local_file}")
        else:
            print("No local file found — database entry only was removed.")


def cmd_clear(args: argparse.Namespace) -> None:
    from .store import count, get_store

    store = get_store()
    n = count(store)
    if n == 0:
        print("Knowledge base is already empty.")
        return
    print(f"This will delete {n} chunks from the database.")
    print("No files will be deleted — only the database index is affected.")
    confirm = input("Type 'yes' to confirm: ").strip().lower()
    if confirm != "yes":
        print("Cancelled.")
        return
    ids = store._collection.get(include=[])["ids"]
    store.delete(ids)
    print(f"Deleted {n} chunks.")


# ── Vault index ────────────────────────────────────────────────────────────────


def cmd_index_vault(args: argparse.Namespace) -> None:
    from ..config import get_config
    from .store import get_store, refresh_vault

    cfg = get_config()
    vault = Path(args.vault_path).expanduser() if args.vault_path else cfg.vault_path
    if not vault.exists():
        print(f"Error: vault path does not exist: {vault}", file=sys.stderr)
        sys.exit(1)

    store = get_store()
    if args.force:
        print("Clearing existing vault index...", flush=True)
        try:
            result = store._collection.get(
                where={"doc_type": {"$eq": "note"}}, include=[]
            )
            if result["ids"]:
                store.delete(result["ids"])
                print(f"  Cleared {len(result['ids'])} chunks", flush=True)
        except Exception:
            pass

    print(f"Indexing vault: {vault}", flush=True)
    added, updated, deleted = refresh_vault(vault, store)
    print(f"Done — +{added} new, ~{updated} changed, -{deleted} removed")


def cmd_update_path(args: argparse.Namespace) -> None:
    from .store import get_store, update_file_path

    new_path = Path(args.new_path).expanduser().resolve()
    if not new_path.exists():
        print(f"Warning: new path does not exist: {new_path}", file=sys.stderr)
    n = update_file_path(args.source, str(new_path), get_store())
    if n == 0:
        print(f"No documents found with source: {args.source}")
    else:
        print(f"Updated {n} chunk(s) — new path: {new_path}")


def cmd_refresh_vault(args: argparse.Namespace) -> None:
    from ..config import get_config
    from .store import get_store, refresh_vault

    cfg = get_config()
    vault = Path(args.vault_path).expanduser() if args.vault_path else cfg.vault_path
    if not vault.exists():
        print(f"Error: vault path does not exist: {vault}", file=sys.stderr)
        sys.exit(1)

    added, updated, deleted = refresh_vault(vault, get_store())
    if added + updated + deleted == 0:
        print("Vault index is up to date.")
    else:
        print(f"Vault refreshed — +{added} new, ~{updated} changed, -{deleted} removed")


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    from ..config import get_config
    cfg = get_config()

    parser = argparse.ArgumentParser(
        prog="kb",
        description="Manage the local knowledge base (papers + vault notes).",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # add
    p_add = sub.add_parser("add", help="Add a paper (arXiv URL) or local PDF")
    p_add.add_argument("input", help="arXiv URL or local PDF path")
    p_add.add_argument("--score", type=int, default=0)
    p_add.add_argument("--track", default="")
    p_add.add_argument("--title", default="", help="Override title (for local PDFs)")
    p_add.add_argument(
        "--visibility", default="public", choices=["public", "private"],
        help="Visibility for local PDFs (arXiv papers are always public)",
    )
    p_add.add_argument(
        "--doc-type", default="paper", choices=["paper", "note"], dest="doc_type",
        help="Type for local PDFs: 'paper' (research paper) or 'note' (always indexed as full text with hash tracking)",
    )
    p_add.add_argument(
        "--provider", default="",
        help=f"'anthropic' or Ollama model name (default: {cfg.provider})",
    )
    p_add.add_argument(
        "--full-text", action="store_true", dest="full_text",
        help="Download PDF and index the full paper text instead of an LLM-generated summary",
    )

    # add-digest
    p_adig = sub.add_parser("add-digest", help="Import papers from digest Markdown file(s)")
    p_adig.add_argument("path", help="Digest .md file or directory of digest files")
    p_adig.add_argument("--min-score", type=int, default=9, dest="min_score",
                        help="Only import papers with score >= N (default: 0)")

    # list / stats / remove / clear
    p_list = sub.add_parser("list", help="List indexed papers")
    p_list.add_argument("--limit", type=int, default=20)
    sub.add_parser("stats", help="Show document and chunk counts")
    p_remove = sub.add_parser("remove", help="Remove a document by source URL")
    p_remove.add_argument("source", help="Source URL of the document to remove")
    p_remove.add_argument(
        "--delete-file", action="store_true", dest="delete_file",
        help="Also delete the local file (for vault notes and local PDFs only)",
    )
    sub.add_parser("clear", help="Delete all documents (prompts for confirmation)")

    # update-path
    p_upd = sub.add_parser("update-path", help="Update the file path for a local document")
    p_upd.add_argument("source", help="Current source URL of the document (file:/// URI or arXiv URL)")
    p_upd.add_argument("new_path", help="New filesystem path to the file")

    # index-vault / refresh-vault
    p_idx = sub.add_parser("index-vault", help="(Re)index the Obsidian vault")
    p_idx.add_argument("--vault-path", default="")
    p_idx.add_argument("--force", action="store_true", help="Clear existing vault index first")
    p_ref = sub.add_parser("refresh-vault", help="Incrementally update vault index")
    p_ref.add_argument("--vault-path", default="")

    args = parser.parse_args()
    dispatch = {
        "add":           lambda: cmd_add(args),
        "add-digest":    lambda: cmd_add_digest(args),
        "list":          lambda: cmd_list(args),
        "stats":         cmd_stats,
        "remove":        lambda: cmd_remove(args),
        "clear":         lambda: cmd_clear(args),
        "update-path":   lambda: cmd_update_path(args),
        "index-vault":   lambda: cmd_index_vault(args),
        "refresh-vault": lambda: cmd_refresh_vault(args),
    }
    dispatch[args.command]()


if __name__ == "__main__":
    main()
