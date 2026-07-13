"""
kb — local knowledge base manager.

Manages a local vector database of research papers and Obsidian vault notes
that vault-chat draws on during conversations.

Subcommands:
  add <url|path>    Add a paper by arXiv URL or local PDF path
  add-digest <path> Import papers from digest Markdown file(s)
  list              List indexed papers
  stats             Show document and chunk counts
  remove <source>   Remove a document by source URL (database entry only — never touches files)
  clear             Delete all documents (prompts for confirmation)
  set-meta <source> Set verified title/authors/doi

  index-vault       Incrementally update vault index; --force clears first
  reindex           Re-embed all chunks with the configured embed_model
  doctor            Diagnose knowledge base health (embed model, corruption)
  sync-status       Show jarvis-sync daemon health and last job outcomes

Usage examples:
  uv run kb add https://arxiv.org/abs/2406.04093 --score 9 --track "Track 1"
  uv run kb add https://arxiv.org/abs/2406.04093 --full-text --figures
  uv run kb add paper.pdf --provider anthropic
  uv run kb add-digest ~/Documents/papers/digest/
  uv run kb list
  uv run kb stats
  uv run kb remove https://arxiv.org/abs/2301.07041
  uv run kb set-meta https://arxiv.org/abs/2301.07041 --authors "Ada Lovelace"
  uv run kb index-vault
  uv run kb index-vault --force
  uv run kb reindex
  uv run kb doctor
"""

import argparse
import sys
from pathlib import Path

from jarvis.digest.import_digest import cmd_add_digest
from jarvis.sync.status import cmd_sync_status


# ── Add ───────────────────────────────────────────────────────────────────────


def cmd_add(args: argparse.Namespace) -> None:
    from jarvis.core.config import get_config
    from jarvis.digest.arxiv.convert import parse_arxiv_url
    from jarvis.digest.arxiv.fetch import fetch_arxiv_paper
    from jarvis.core.llm import make_provider
    from .store import (
        _source_exists, _title_exists, add_annotations, add_figures,
        add_paper, add_texts, delete_by_metadata, get_store,
    )

    cfg = get_config()
    store = get_store()
    _provider = None
    # --figures forces captioning for this one document; None leaves it to
    # cfg.figure_captions (off by default).
    figures_enabled = True if args.figures else None

    def get_provider():
        nonlocal _provider
        if _provider is None:
            _provider = make_provider(args.provider or cfg.provider)
        return _provider

    def confirm_duplicate(source: str, title: str) -> tuple[bool, str | None]:
        """
        Return (proceed, replace_source).

        proceed=False means the user declined — the caller must abort without
        touching the store. A paper can arrive twice via different sources
        (arXiv + bioRxiv), so we check both the source URL and the title.

        replace_source is set to `source` only when the duplicate matched by
        SOURCE (a same-title-but-different-source duplicate is a genuinely
        separate entry and must never trigger a delete). This function only
        gates the decision — it does NOT delete anything. The caller deletes
        the old chunks (body, annotations, figures — they all share source)
        itself, and only once the new content has actually been produced
        (PDF downloaded and converted, or summary generated). Deleting here,
        before that work even starts, would wipe the old entry — including
        irreplaceable annotation chunks — even if the download or conversion
        then fails.
        """
        same_source = _source_exists(source, store)
        if not (same_source or _title_exists(title, store)):
            return True, None
        print(f"Already in the knowledge base: \"{title}\" ({source})")
        if input("Add anyway? [y/N] ").strip().lower() != "y":
            return False, None
        return True, (source if same_source else None)

    input_str: str = args.input

    if input_str.startswith("http://") or input_str.startswith("https://"):
        arxiv_id = parse_arxiv_url(input_str)
        if not arxiv_id:
            print(f"Error: could not parse arXiv ID from URL: {input_str}", file=sys.stderr)
            sys.exit(1)
        print(f"Fetching metadata for arXiv:{arxiv_id}...")
        paper = fetch_arxiv_paper(arxiv_id)
        print(f"  Title: {paper['title']}")

        proceed, replace_source = confirm_duplicate(paper.get("link", ""), paper.get("title", ""))
        if not proceed:
            print("Cancelled.")
            return

        if args.full_text:
            import tempfile
            from jarvis.digest.arxiv.convert import download_arxiv_pdf
            from jarvis.core.errors import ConversionError
            from .convert import pdf_to_markdown
            print("Downloading PDF...")
            with tempfile.TemporaryDirectory() as tmp:
                pdf_path_dl = download_arxiv_pdf(arxiv_id, Path(tmp))
                print("Converting to Markdown...")
                try:
                    full_text = pdf_to_markdown(pdf_path_dl)
                except ConversionError as exc:
                    print(f"Error: {exc}", file=sys.stderr)
                    sys.exit(1)
                if replace_source:
                    deleted = delete_by_metadata("source", replace_source, store)
                    print(f"  Replacing existing entry — {deleted} old chunk(s) removed")
                annotation_ids = add_annotations(
                    pdf_path_dl, doc_type="paper", visibility="public",
                    source=paper["link"], title=paper.get("title", ""), store=store,
                )
                figure_ids = add_figures(
                    pdf_path_dl, doc_type="paper", visibility="public",
                    source=paper["link"], provider_obj=get_provider(),
                    provider_str=(args.provider or cfg.provider),
                    title=paper.get("title", ""), store=store,
                    enabled=figures_enabled,
                )
            print("Chunking and indexing full text...")
            authors = paper.get("authors", "")
            embed_header = f"{paper['title']} — {authors}" if authors else paper["title"]
            ids = add_texts(
                content=full_text,
                doc_type="paper",
                visibility="public",
                source=paper["link"],
                extra_metadata={
                    "title": paper.get("title", ""),
                    "authors": authors,
                    "doi": paper.get("doi", ""),
                    "score": int(args.score),
                    "track": str(args.track),
                    "storage_mode": "full_text",
                },
                store=store,
                embed_header=embed_header,
            )
            print(f"Added (full text, {len(ids)} chunks): {paper['link']}")
            if annotation_ids:
                print(f"  {len(annotation_ids)} annotation(s) indexed")
            if figure_ids:
                print(f"  {len(figure_ids)} figure(s) captioned")
        else:
            print("Generating summary...")
            summary = get_provider().summarize(paper["title"], paper["abstract"])
            if replace_source:
                deleted = delete_by_metadata("source", replace_source, store)
                print(f"  Replacing existing entry — {deleted} old chunk(s) removed")
            # allow_duplicate: confirm_duplicate already gated this — the user
            # either has no duplicate or explicitly chose to add anyway.
            add_paper(paper=paper, dense_summary=summary, score=args.score,
                      track=args.track, store=store, storage_mode="summary",
                      allow_duplicate=True)
            print(f"Added (summary): {paper['link']}")

    elif Path(input_str).exists() and Path(input_str).suffix.lower() == ".pdf":
        # Local PDFs are always public papers — notes come exclusively from
        # the Obsidian vault (.md files), so there is no visibility/doc_type
        # choice to make here.
        pdf_path = Path(input_str).resolve()

        from .metadata import resolve_pdf_metadata

        meta = resolve_pdf_metadata(
            pdf_path, get_provider(),
            title_override=args.title, authors_override=args.authors, doi_override=args.doi,
        )
        title = meta["title"] or pdf_path.stem
        authors, doi = meta["authors"], meta["doi"]

        proceed, replace_source = confirm_duplicate(pdf_path.as_uri(), title)
        if not proceed:
            print("Cancelled.")
            return

        def index_annotations() -> None:
            # Highlights, typed notes, and captioned figures each become their
            # own chunks, regardless of whether the body was stored as summary
            # or full text.
            annotation_ids = add_annotations(
                pdf_path, doc_type="paper", visibility="public",
                source=pdf_path.as_uri(), title=title,
                file_path=str(pdf_path), store=store,
            )
            if annotation_ids:
                print(f"  {len(annotation_ids)} annotation(s) indexed")
            figure_ids = add_figures(
                pdf_path, doc_type="paper", visibility="public",
                source=pdf_path.as_uri(), provider_obj=get_provider(),
                provider_str=(args.provider or cfg.provider),
                title=title, file_path=str(pdf_path), store=store,
                enabled=figures_enabled,
            )
            if figure_ids:
                print(f"  {len(figure_ids)} figure(s) captioned")

        if args.full_text:
            from jarvis.core.errors import ConversionError
            from .convert import pdf_to_markdown
            print(f"Converting PDF to Markdown: {pdf_path.name}...")
            try:
                full_text = pdf_to_markdown(pdf_path)
            except ConversionError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(1)
            if replace_source:
                deleted = delete_by_metadata("source", replace_source, store)
                print(f"  Replacing existing entry — {deleted} old chunk(s) removed")
            extra_metadata = {"title": title, "file_path": str(pdf_path),
                               "storage_mode": "full_text", "authors": authors, "doi": doi}
            ids = add_texts(
                content=full_text,
                doc_type="paper",
                visibility="public",
                source=pdf_path.as_uri(),
                extra_metadata=extra_metadata,
                store=store,
                embed_header=(f"{title} — {authors}" if authors else title),
            )
            print(f"Added paper (full text, {len(ids)} chunks): {pdf_path.name}")
            index_annotations()
        else:
            print(f"Generating summary from PDF: {pdf_path.name}...")
            summary = get_provider().summarize(title, pdf_path)
            if replace_source:
                deleted = delete_by_metadata("source", replace_source, store)
                print(f"  Replacing existing entry — {deleted} old chunk(s) removed")
            extra_metadata = {"title": title, "file_path": str(pdf_path),
                               "storage_mode": "summary", "authors": authors, "doi": doi}
            add_texts(
                content=f"{title}\n\n{summary}",
                doc_type="paper",
                visibility="public",
                source=pdf_path.as_uri(),
                extra_metadata=extra_metadata,
                store=store,
                embed_header=(f"{title} — {authors}" if authors else title),
            )
            print(f"Added paper (summary): {pdf_path.name}")
            index_annotations()

    else:
        print(f"Error: '{input_str}' is not a valid arXiv URL or PDF path.", file=sys.stderr)
        sys.exit(1)


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
        if p.get("authors"):
            print(f"  Authors: {p['authors']}")
        if p.get("doi"):
            print(f"  DOI: {p['doi']}")
        print()


def cmd_stats() -> None:
    from .store import count, count_unique_documents, get_store

    store = get_store()
    total_chunks = count(store)
    papers = count_unique_documents("paper", "source", store)
    notes = count_unique_documents("note", "file_path", store)
    digests = count_unique_documents("digest", "source", store)
    print(f"Documents:  {papers} papers · {notes} notes · {digests} digests")
    print(f"Chunks:     {total_chunks} total")

    # Consistency check for the papers-are-always-public invariant. Entries
    # added before the invariant existed could still be private; surface them
    # rather than silently migrating.
    try:
        stray = store._collection.get(
            where={"$and": [{"doc_type": {"$eq": "paper"}}, {"visibility": {"$eq": "private"}}]},
            include=["metadatas"],
        )
        private_sources = sorted({m.get("source", "?") for m in stray["metadatas"]})
        if private_sources:
            print(
                f"\n⚠️  {len(private_sources)} paper(s) are marked private, but papers "
                "must always be public.\n   Move its content into the vault as a note, "
                "or make it public (kb remove, then kb add to re-add as a public paper):"
            )
            for src in private_sources:
                print(f"   - {src}")
    except Exception:
        pass


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
        from jarvis.core.config import get_config
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

    local_file_str = str(local_file) if local_file else "no local file"

    print(f"  Title:  {title}")
    print(f"  Type:   {doc_type}")
    print(f"  Source: {args.source}")
    print(f"  Chunks: {len(ids)}")
    print(f"  Database entry only — files on disk are never touched by jarvis: {local_file_str}")

    confirm = input("Confirm? [y/N] ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    store.delete(ids)
    print(f"Removed \"{title}\" ({len(ids)} chunk(s)) from the knowledge base. No files were touched.")


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
    from jarvis.core.config import get_config
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
            ids_to_delete = result["ids"]
            if ids_to_delete:
                store.delete(ids_to_delete)
                print(f"  Cleared {len(ids_to_delete)} chunks", flush=True)
        except Exception:
            pass

    print(f"Indexing vault: {vault}", flush=True)
    added, updated, deleted = refresh_vault(vault, store)
    print(f"Done — +{added} new, ~{updated} changed, -{deleted} removed")


def _migrated_chunk_text(text: str, metadata: dict) -> str:
    """
    Backfill the title/authors embed-header onto a legacy paper chunk that
    predates it, so author-name and acronym queries can match papers indexed
    before add_texts() started prepending a header to every chunk.

    Only paper body chunks are touched — annotation chunks (identified by a
    present annotation_kind key) and note chunks (doc_type != "paper") are
    left exactly as stored, since the header only makes sense on paper text.
    Idempotent: if the text already starts with the title, it is returned
    unchanged, so running the migration twice never double-prepends.
    """
    if metadata.get("doc_type") != "paper" or metadata.get("annotation_kind"):
        return text
    title = metadata.get("title", "")
    if not title or text.startswith(title):
        return text
    authors = metadata.get("authors", "")
    header = f"{title} — {authors}" if authors else title
    return f"{header}\n{text}"


def cmd_reindex(args: argparse.Namespace) -> None:
    """
    Re-embed every stored chunk with the currently configured embedding model.

    The chunk texts are already stored in ChromaDB, so this needs no LLM calls
    and no re-summarising or re-downloading — it only recomputes vectors. Used
    after changing embed_model / query_prefix. Work happens in a temporary
    collection that is swapped in only once fully built, so an interrupted run
    never leaves the knowledge base half-migrated.
    """
    import chromadb

    from jarvis.core.config import get_config
    from .store import COLLECTION_NAME, build_embeddings

    cfg = get_config()
    reindex_name = f"{COLLECTION_NAME}_reindex"

    # Read the old collection directly, bypassing get_store()'s model-mismatch
    # guard — the mismatch is exactly what we are here to resolve.
    client = chromadb.PersistentClient(path=str(cfg.rag_dir))
    try:
        old_collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        print(f"No '{COLLECTION_NAME}' collection found — nothing to reindex.")
        return

    stored = old_collection.get(include=["documents", "metadatas"])
    ids = stored["ids"]
    documents = stored["documents"]
    metadatas = stored["metadatas"]
    if not ids:
        print("Knowledge base is empty — nothing to reindex.")
        return

    print(f"Reindexing {len(ids)} chunks with '{cfg.embed_model}'...", flush=True)
    embeddings = build_embeddings(cfg.embed_model, cfg.query_prefix)

    # Start from a clean temp collection in case a previous run was interrupted.
    try:
        client.delete_collection(reindex_name)
    except Exception:
        pass
    new_collection = client.create_collection(
        reindex_name,
        metadata={"embed_model": cfg.embed_model, "query_prefix": cfg.query_prefix},
    )

    batch_size = 100
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        batch_metas = metadatas[start:end]
        # Backfill the title/authors embed-header on legacy paper chunks that
        # predate it (see _migrated_chunk_text) before embedding, and store
        # the migrated text — not the original — so the fix persists.
        batch_docs = [
            _migrated_chunk_text(doc_text, meta)
            for doc_text, meta in zip(documents[start:end], batch_metas)
        ]
        vectors = embeddings.embed_documents(batch_docs)
        new_collection.add(
            ids=ids[start:end],
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=vectors,
        )
        print(f"  {min(end, len(ids))}/{len(ids)} chunks", flush=True)

    # Swap: drop the old collection, then rename the rebuilt one into its place.
    client.delete_collection(COLLECTION_NAME)
    new_collection.modify(name=COLLECTION_NAME)
    print(f"Done — reindexed {len(ids)} chunks with '{cfg.embed_model}'.")
    print(
        "NOTE: the swap gives the collection a new identity, so any jarvis "
        "process that was already running (webapp, jarvis-sync, vault-chat) "
        "now holds a stale handle — restart those processes before using them."
    )


def cmd_doctor() -> None:
    """
    Diagnose knowledge base health: open the store (exercises the
    embed-model guard), count chunks, then probe a real search (exercises
    corruption detection). Exits non-zero on any failure so this is
    scriptable. No automatic startup probe elsewhere — this is opt-in so a
    healthy launch never pays the cost.

    Note: on a badly corrupted store, even count() can hard-segfault the
    process (a Rust-side ChromaDB crash, uncatchable in Python). If this
    command dies abruptly with no output beyond "Checking knowledge base...",
    that abrupt death is itself the diagnosis — run `kb reindex` blind.

    Once the store is confirmed healthy, also checks for legacy PDF notes
    (see _check_legacy_pdf_notes) — a one-time migration for entries added
    before local PDFs became always-public papers.
    """
    from jarvis.core.errors import KBCorruptionError, RAGError
    from .store import count, get_store, search

    print("Checking knowledge base...")
    try:
        store = get_store()
    except RAGError as exc:
        print(f"✗ Failed to open store: {exc}", file=sys.stderr)
        sys.exit(1)
    print("✓ Store opened (embedding model matches)")

    try:
        n = count(store)
    except Exception as exc:
        print(f"✗ Failed to count chunks: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"✓ {n} chunk(s) indexed")

    if n == 0:
        print("Knowledge base is empty — nothing to search-probe.")
        return

    try:
        search("diagnostic probe query", n_results=1, store=store, rerank=False)
    except KBCorruptionError as exc:
        print(f"✗ Search index is corrupted:\n  {exc}", file=sys.stderr)
        sys.exit(1)
    except RAGError as exc:
        print(f"✗ Search failed: {exc}", file=sys.stderr)
        sys.exit(1)
    print("✓ Search probe succeeded\n\nKnowledge base is healthy.")

    _check_legacy_pdf_notes(store)


def _check_legacy_pdf_notes(store) -> None:
    """
    One-time migration check: local PDFs are now always public papers — notes
    come exclusively from the Obsidian vault. Entries added before that
    decision may still carry doc_type="note" with an absolute PDF file_path.

    Public ones are reclassified in place with a single y/N prompt (doc_type
    flips to "paper"; content_hash/storage_mode/file_path are untouched, so
    the result has the same shape a daemon-ingested paper carries). Private
    ones are NEVER silently made public — they are only listed, with
    resolution options, and `kb doctor` keeps reporting them until resolved.
    """
    from .store import find_pdf_notes, reclassify_notes_as_papers

    pdf_notes = find_pdf_notes(store)
    if not pdf_notes:
        return

    public = [n for n in pdf_notes if n["visibility"] != "private"]
    private = [n for n in pdf_notes if n["visibility"] == "private"]

    if public:
        print(
            f"\n⚠️  {len(public)} legacy PDF note(s) found — local PDFs are always "
            "papers now; notes come only from the vault."
        )
        for n in public:
            print(f"   - {n['title']}  ({n['source']}, {n['chunk_count']} chunk(s))")
        answer = input(f"Reclassify {len(public)} document(s) as papers? [y/N] ").strip().lower()
        if answer == "y":
            n_chunks = reclassify_notes_as_papers([n["source"] for n in public], store)
            print(f"  Reclassified {len(public)} document(s) ({n_chunks} chunk(s)) as papers.")
        else:
            print("  Skipped — run `kb doctor` again to reclassify later.")

    if private:
        print(
            f"\n⚠️  {len(private)} private legacy PDF note(s) found. Papers are "
            "always public, so these are never silently reclassified — resolve "
            "each one, then re-run `kb doctor`:"
        )
        for n in private:
            print(f"   - {n['title']}  ({n['source']}, {n['chunk_count']} chunk(s))")
        print(
            "     Resolve by either: `kb remove <source>` then re-add the PDF as "
            "a public paper, or move its content into the vault as a private .md note."
        )


def cmd_set_meta(args: argparse.Namespace) -> None:
    from .store import get_store, update_paper_metadata

    if args.title is None and args.authors is None and args.doi is None:
        print("Error: at least one of --title/--authors/--doi is required.", file=sys.stderr)
        sys.exit(1)
    n = update_paper_metadata(
        args.source, title=args.title, authors=args.authors, doi=args.doi, store=get_store(),
    )
    if n == 0:
        print(f"No documents found with source: {args.source}")
    else:
        print(f"Updated {n} chunk(s) — metadata verified.")


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


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    from jarvis.core.config import get_config
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
    p_add.add_argument("--authors", default="", help="Override authors (for local PDFs)")
    p_add.add_argument("--doi", default="", help="Override DOI (for local PDFs)")
    p_add.add_argument(
        "--provider", default="",
        help=f"'anthropic' or 'ollama' (default: {cfg.provider})",
    )
    p_add.add_argument(
        "--full-text", action="store_true", dest="full_text",
        help="Download PDF and index the full paper text instead of an LLM-generated summary",
    )
    p_add.add_argument(
        "--figures", action="store_true",
        help="Caption and index this document's figures even though figure_captions "
             "is off by default (answering y to the duplicate prompt replaces the old entry)",
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
    sub.add_parser("clear", help="Delete all documents (prompts for confirmation)")

    # set-meta
    p_setmeta = sub.add_parser("set-meta", help="Set verified title/authors/doi")
    p_setmeta.add_argument("source")
    p_setmeta.add_argument("--title", default=None)
    p_setmeta.add_argument("--authors", default=None)
    p_setmeta.add_argument("--doi", default=None)

    # update-path
    p_upd = sub.add_parser("update-path", help="Update the file path for a local document")
    p_upd.add_argument("source", help="Current source URL of the document (file:/// URI or arXiv URL)")
    p_upd.add_argument("new_path", help="New filesystem path to the file")

    # index-vault
    p_idx = sub.add_parser("index-vault", help="(Re)index the Obsidian vault")
    p_idx.add_argument("--vault-path", default="")
    p_idx.add_argument("--force", action="store_true", help="Clear existing vault note index first")

    # reindex
    sub.add_parser("reindex", help="Re-embed all chunks with the configured embed_model (no LLM calls)")

    # doctor
    sub.add_parser("doctor", help="Diagnose knowledge base health (embed model, corruption)")

    # sync-status
    sub.add_parser("sync-status", help="Show jarvis-sync daemon health and last job outcomes")

    args = parser.parse_args()
    dispatch = {
        "add":         lambda: cmd_add(args),
        "add-digest":  lambda: cmd_add_digest(args),
        "list":        lambda: cmd_list(args),
        "stats":       cmd_stats,
        "remove":      lambda: cmd_remove(args),
        "clear":       lambda: cmd_clear(args),
        "set-meta":    lambda: cmd_set_meta(args),
        "update-path": lambda: cmd_update_path(args),
        "index-vault": lambda: cmd_index_vault(args),
        "reindex":     lambda: cmd_reindex(args),
        "doctor":      cmd_doctor,
        "sync-status": cmd_sync_status,
    }
    dispatch[args.command]()


if __name__ == "__main__":
    main()
