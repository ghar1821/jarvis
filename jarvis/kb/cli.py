"""
kb — local knowledge base manager.

Manages a local vector database of research papers and Obsidian vault notes
that the webapp agent draws on during conversations.

Subcommands:
  add <url|path>    Add a paper by arXiv URL or local PDF path
  add-digest <path> Import papers from digest Markdown file(s)
  list              List indexed papers (--notes lists vault notes/records)
  stats             Show document and chunk counts
  remove <source>   Remove a document by source URL (database entry only — never touches files)
  clear             Delete all documents (prompts for confirmation)
  set-meta <source> Set verified title/authors/doi

  index-vault       Incrementally update vault index; --force clears first
  reindex           Re-embed all chunks with the configured embed_model
  doctor            Diagnose knowledge base health (embed model, corruption)
  models            List the switchable models your config offers
  schema            Show which metadata keys/values exist (your record ontology)
  drafts            List drafts; --prune removes stale ones
  sync-status       Show jarvis-sync daemon health and last job outcomes

Usage examples:
  uv run kb add https://arxiv.org/abs/2406.04093 --score 9 --track "Track 1"
  uv run kb add https://arxiv.org/abs/2406.04093 --full-text --figures
  uv run kb add paper.pdf --provider anthropic
  uv run kb add-digest ~/Documents/papers/digest/
  uv run kb list
  uv run kb list --notes --category job_application --status rejected
  uv run kb schema
  uv run kb schema status
  uv run kb drafts
  uv run kb drafts --prune --dry-run
  uv run kb stats
  uv run kb remove https://arxiv.org/abs/2301.07041
  uv run kb set-meta https://arxiv.org/abs/2301.07041 --authors "Ada Lovelace"
  uv run kb index-vault
  uv run kb index-vault --force
  uv run kb reindex
  uv run kb doctor
  uv run kb models
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
    """List indexed papers, or vault notes/records with --notes."""
    from .store import get_store, list_documents

    doc_type = "note" if args.notes else "paper"
    documents = list_documents(
        limit=args.limit,
        doc_type=doc_type,
        category=args.category,
        status=args.status,
        entity=args.entity,
        store=get_store(),
    )
    if not documents:
        print(f"No matching {'notes' if args.notes else 'papers'} in knowledge base.")
        return

    for d in documents:
        chunks = d.get("chunk_count", "?")
        if doc_type == "paper":
            mode = d.get("storage_mode", "summary" if chunks in ("?", 1, 2) else "full_text")
            print(f"[{d.get('score', '?')}/10] {d.get('title', 'untitled')}  [{mode}, {chunks} chunks]")
            print(f"  {d.get('source', 'no source')}  ·  {d.get('date_added', 'N/A')[:10]}")
            if d.get("authors"):
                print(f"  Authors: {d['authors']}")
            if d.get("doi"):
                print(f"  DOI: {d['doi']}")
        else:
            record = " · ".join(
                str(d[field]) for field in ("category", "entity", "status") if d.get(field)
            )
            print(f"{d.get('title', 'untitled')}  [{chunks} chunks]")
            print(f"  {d.get('file_path', 'unknown')}  ·  {d.get('date_added', 'N/A')[:10]}")
            if record:
                print(f"  {record}")
            if d.get("event_date"):
                print(f"  Date: {d['event_date']}")
            if d.get("tags"):
                print(f"  Tags: {d['tags'].strip('|').replace('|', ', ')}")
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
    except Exception as exc:
        # This check is advisory, so a failure should not take `kb stats`
        # down — but it must say it did not run, or a silent skip reads as
        # "you have no legacy private papers".
        print(f"\n⚠️  Could not check for legacy private papers: {exc}", file=sys.stderr)


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
        except Exception as exc:
            # --force promises a clean rebuild. Indexing on top of a store we
            # failed to clear gives the opposite — the stale chunks the user
            # asked to be rid of, silently kept. Stop instead.
            print(f"\n✗ Could not clear the existing vault index: {exc}", file=sys.stderr)
            print("  --force cannot rebuild cleanly on top of this. Run "
                  "`uv run kb doctor` to diagnose the store first.", file=sys.stderr)
            sys.exit(1)

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


def _chunks_from_sqlite(rag_dir: Path) -> tuple[list[str], list[str], list[dict]]:
    """
    Read every stored chunk straight out of ChromaDB's SQLite file, bypassing
    the collection API entirely. Returns (ids, documents, metadatas).

    This exists because the normal reindex path cannot always run. Chunk text
    and metadata live in SQLite, but vectors live in HNSW files, and a damaged
    HNSW header makes hnswlib try to map an absurd allocation — the kernel
    kills the process with SIGBUS before any Python exception can be raised.
    At that point `collection.get()` is unusable, and it is the very call the
    ordinary reindex depends on. Reading the tables directly sidesteps the
    broken index and recovers everything, because nothing of value was ever
    stored in it: the vectors are derived data and this rebuilds them.
    """
    import sqlite3

    from jarvis.core.errors import RAGError

    from .store import COLLECTION_NAME

    database = rag_dir / "chroma.sqlite3"
    if not database.exists():
        raise RAGError(f"No ChromaDB file at {database}")

    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        # The live collection's METADATA segment owns the rows we want; an
        # orphaned collection from an earlier reindex may still be present.
        segment = connection.execute(
            """
            SELECT s.id FROM segments s
            JOIN collections c ON c.id = s.collection
            WHERE c.name = ? AND s.scope = 'METADATA'
            """,
            (COLLECTION_NAME,),
        ).fetchone()
        if segment is None:
            raise RAGError(f"No '{COLLECTION_NAME}' collection in {database}")

        rows = connection.execute(
            """
            SELECT e.embedding_id, m.key,
                   m.string_value, m.int_value, m.float_value, m.bool_value
            FROM embeddings e
            JOIN embedding_metadata m ON m.id = e.id
            WHERE e.segment_id = ?
            ORDER BY e.id
            """,
            (segment[0],),
        ).fetchall()
    finally:
        connection.close()

    # One row per (chunk, metadata key); fold them back into per-chunk dicts.
    documents: dict[str, str] = {}
    metadatas: dict[str, dict] = {}
    order: list[str] = []
    for chunk_id, key, string_value, int_value, float_value, bool_value in rows:
        if chunk_id not in metadatas:
            metadatas[chunk_id] = {}
            order.append(chunk_id)
        if key == "chroma:document":
            documents[chunk_id] = string_value or ""
            continue
        for value in (string_value, int_value, float_value, bool_value):
            if value is not None:
                metadatas[chunk_id][key] = value
                break

    # A chunk with no text cannot be re-embedded into anything useful.
    ids = [chunk_id for chunk_id in order if documents.get(chunk_id)]
    return ids, [documents[i] for i in ids], [metadatas[i] for i in ids]


def cmd_reindex(args: "argparse.Namespace | None" = None) -> None:
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

    client = chromadb.PersistentClient(path=str(cfg.rag_dir))

    # Tolerate being called without parsed args (tests, and any programmatic
    # caller that just wants the ordinary rebuild).
    if getattr(args, "from_storage", False):
        # Recovery path: the vector index is too damaged to read through, so
        # take the chunks from SQLite instead. See _chunks_from_sqlite.
        print("Reading chunks directly from storage (bypassing the vector index)...", flush=True)
        ids, documents, metadatas = _chunks_from_sqlite(cfg.rag_dir)
    else:
        # Read the old collection directly, bypassing get_store()'s
        # model-mismatch guard — the mismatch is exactly what we are here to
        # resolve.
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
        "process that was already running (the webapp or jarvis-sync) "
        "now holds a stale handle — restart those processes before using them."
    )


def cmd_drafts(args: argparse.Namespace) -> None:
    """
    List drafts, or sweep the stale ones.

    Drafts are scratch space and expire, so the list always shows how long each
    has left — a removal should never be a surprise. `--prune --dry-run` prints
    exactly what a sweep would take without touching anything.
    """
    from jarvis.core.config import get_config
    from jarvis.drafts import list_drafts, prune_drafts

    retention = get_config().drafts_retention_days

    if args.prune:
        removed = prune_drafts(dry_run=args.dry_run)
        if not removed:
            print("Nothing to remove." if not args.dry_run else "Nothing would be removed.")
            return
        verb = "Would remove" if args.dry_run else "Removed"
        for draft in removed:
            print(f"{verb} {draft['id']}  {draft['title']!r}  "
                  f"(untouched for {draft['age_days']:.1f} days)")
        return

    drafts = list_drafts()
    if not drafts:
        print("No drafts yet.")
        return
    for draft in drafts:
        files = ", ".join(draft.get("files", []))
        if draft.get("keep"):
            expiry = "kept (never expires)"
        elif retention <= 0:
            expiry = "retention disabled"
        else:
            remaining = retention - draft.get("age_days", 0)
            expiry = (
                f"expires in {remaining:.1f} days" if remaining > 0 else "expires on the next sweep"
            )
        print(f"{draft['title']}  ({draft['id']})")
        print(f"  files: {files}")
        print(f"  {draft.get('visibility', 'public')} · {expiry}")
        print()


def cmd_schema(args: argparse.Namespace) -> None:
    """
    Show the metadata keys present in the store, or one key's distinct values.

    This exists to make your own ontology visible. Jarvis enforces no
    vocabulary for record types or statuses, which means a typo
    ("stauts: rejected") becomes its own key that silently never matches a
    filter — listing what is actually there is how you catch that.
    """
    from .store import get_store, metadata_key_counts, metadata_value_counts

    store = get_store()
    if args.key:
        values = metadata_value_counts(args.key, store)
        if not values:
            print(f"No chunks carry the metadata key {args.key!r}.")
            return
        print(f"{args.key}:")
        for value, count in sorted(values.items(), key=lambda pair: -pair[1]):
            print(f"  {count:6d}  {value}")
        return

    counts = metadata_key_counts(store)
    if not counts:
        print("Knowledge base is empty.")
        return
    print("Metadata keys (chunk counts). Run `kb schema <key>` to see its values.\n")
    for key, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])):
        marker = "  (custom frontmatter)" if key.startswith("x_") else ""
        print(f"  {count:6d}  {key}{marker}")


def cmd_models(args: argparse.Namespace) -> None:
    """
    Show the switchable model catalogue as the config defines it.

    Makes no network call, so it doubles as the quickest check that the config
    file is being read at all, and which providers have credentials.
    """
    from jarvis.chat.models import list_catalogue
    from jarvis.core.config import get_config

    cfg = get_config()
    entries = list_catalogue(cfg)
    if not entries:
        print("No models configured. Add them under [models] in ~/.jarvis/config.toml,")
        return
    for entry in entries:
        where = "local" if entry["local"] else "cloud"
        note = "" if entry["available"] else "  (no API key)"
        print(f"  {entry['spec']}  [{where}]{note}")


# Everything that touches the vector index runs here, in a child process.
# Reading a damaged index does not raise — it terminates the process by signal
# (SIGBUS from a corrupt HNSW header, SIGSEGV from a bad pointer), which no
# Python handler can intercept. Running it in a child is what turns "the
# command vanished with no output" into a diagnosis the user can act on.
_DOCTOR_PROBE = """
import json
from jarvis.core.errors import KBCorruptionError, RAGError
from jarvis.kb.store import count, get_store, search

result = {"opened": False, "count": None, "search_ok": None, "error": None}
try:
    store = get_store()
    result["opened"] = True
    result["count"] = count(store)
    if result["count"]:
        try:
            search("diagnostic probe query", n_results=1, store=store, rerank=False)
            result["search_ok"] = True
        except (KBCorruptionError, RAGError) as exc:
            result["search_ok"] = False
            result["error"] = str(exc)
except (RAGError, Exception) as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"
print("DOCTOR_RESULT " + json.dumps(result))
"""

_INDEX_UNREADABLE = """\
✗ The vector index is damaged beyond reading.
  Probing it killed a child process outright (signal {signal}), which means the
  index file itself is unreadable — not a Python-level error that could be
  caught and reported.

  Your documents are almost certainly fine. Chunk text and metadata live in
  chroma.sqlite3; only the vectors live in the damaged index, and those are
  derived data that can be rebuilt.

  Recover with:
      uv run kb reindex --from-storage

  That reads the chunks straight from SQLite, re-embeds them, and swaps in a
  fresh index. No LLM calls, nothing re-downloaded."""


def _run_doctor_probe() -> "dict | None":
    """Run the probe in a child. Returns its result, or None if the child died."""
    import json
    import subprocess

    completed = subprocess.run(
        [sys.executable, "-c", _DOCTOR_PROBE], capture_output=True, text=True
    )
    for line in completed.stdout.splitlines():
        if line.startswith("DOCTOR_RESULT "):
            return json.loads(line[len("DOCTOR_RESULT "):])
    # No result line: the child died before it could report. A negative
    # returncode is the signal that killed it.
    print(f"   (probe exited with code {completed.returncode})", file=sys.stderr)
    if completed.stderr.strip():
        print("   " + completed.stderr.strip().splitlines()[-1], file=sys.stderr)
    return None


def cmd_doctor() -> None:
    """
    Diagnose knowledge base health: open the store (exercises the embed-model
    guard), count chunks, then probe a real search (exercises corruption
    detection). Exits non-zero on any failure so this is scriptable.

    The probe runs in a subprocess — see _DOCTOR_PROBE for why that is not
    optional. Once the store is confirmed healthy, this also checks for legacy
    PDF notes (see _check_legacy_pdf_notes), a one-time migration for entries
    added before local PDFs became always-public papers.
    """
    print("Checking knowledge base...", flush=True)
    result = _run_doctor_probe()

    if result is None:
        signal_name = "SIGBUS/SIGSEGV"
        print(_INDEX_UNREADABLE.format(signal=signal_name), file=sys.stderr)
        sys.exit(1)

    if not result["opened"]:
        print(f"✗ Failed to open store: {result['error']}", file=sys.stderr)
        sys.exit(1)
    print("✓ Store opened (embedding model matches)")

    print(f"✓ {result['count']} chunk(s) indexed")
    if not result["count"]:
        print("Knowledge base is empty — nothing to search-probe.")
        return

    if result["search_ok"] is False:
        print(f"✗ Search failed:\n  {result['error']}", file=sys.stderr)
        sys.exit(1)
    print("✓ Search probe succeeded\n\nKnowledge base is healthy.")

    from .store import get_store

    _check_legacy_pdf_notes(get_store())


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
    p_list = sub.add_parser("list", help="List indexed papers, or notes/records with --notes")
    p_list.add_argument("--limit", type=int, default=20)
    p_list.add_argument("--notes", action="store_true", help="List vault notes/records instead of papers")
    p_list.add_argument("--category", help="Filter notes by record type, e.g. job_application")
    p_list.add_argument("--status", help="Filter notes by status, e.g. rejected")
    p_list.add_argument("--entity", help="Filter notes by organisation or person")
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
    p_reindex = sub.add_parser(
        "reindex", help="Re-embed all chunks with the configured embed_model (no LLM calls)"
    )
    p_reindex.add_argument(
        "--from-storage",
        action="store_true",
        dest="from_storage",
        help=(
            "Read chunks straight from SQLite instead of through the vector index. "
            "Use when the index is so damaged that kb doctor dies without output."
        ),
    )

    # doctor
    sub.add_parser("doctor", help="Diagnose knowledge base health (embed model, corruption)")

    # models
    p_models = sub.add_parser("models", help="List the switchable models your config offers")

    # drafts
    p_drafts = sub.add_parser("drafts", help="List drafts; --prune sweeps stale ones")
    p_drafts.add_argument("--prune", action="store_true", help="Remove drafts past the retention window")
    p_drafts.add_argument("--dry-run", action="store_true", dest="dry_run",
                          help="With --prune: print what would be removed, remove nothing")

    # schema
    p_schema = sub.add_parser("schema", help="Show which metadata keys and values exist in the store")
    p_schema.add_argument("key", nargs="?", help="Show the distinct values of one key")

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
        "models":      lambda: cmd_models(args),
        "schema":      lambda: cmd_schema(args),
        "drafts":      lambda: cmd_drafts(args),
        "sync-status": cmd_sync_status,
    }
    dispatch[args.command]()


if __name__ == "__main__":
    main()
