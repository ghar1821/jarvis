"""
Knowledge base using LangChain + ChromaDB.

Single unified collection with a flat document schema. All content —
papers, vault notes, local PDFs — is chunked and stored as LangChain
Documents with consistent metadata.

Document schema
---------------
  page_content : str   — chunked text (embedded for similarity search)
  metadata:
    date_added  : str  — ISO timestamp of when the chunk was indexed
    doc_type    : str  — "paper" | "note" | "chat" (past chat exchanges) |
                         "digest" (indexed weekly digest .md files)
    visibility  : str  — "public" | "private" (papers are always public)
    source      : str  — arXiv/DOI URL for papers, file:/// URI for local
                         PDFs and digest files, "local" for vault notes,
                         "session:<id>" for chat exchanges
    title       : str  — display title (optional)
    authors     : str  — comma-separated authors, papers only (optional)
    doi         : str  — DOI for papers, regex/LLM-inferred for local PDFs (optional)
    score       : int  — relevance score 0-10, papers only (optional)
    track       : str  — research track label, papers only (optional)
    file_path   : str  — vault-relative path for notes, absolute for PDFs (optional)
    content_hash: str  — SHA-256 of full file, used for change detection
    chunk_index : int  — position of this chunk within its source document
    section     : str  — markdown header breadcrumb ("H1 › H2"), "" if none
    storage_mode: str  — "summary" | "full_text" (optional)
    modified_at : str  — ISO mtime of the source file, vault notes only (optional)

  PDF annotation and figure chunks (see add_annotations / add_figures)
  additionally carry:
    annotation_kind : str — "highlight" | "comment" | "figure" (absent on body chunks)
    page            : int — 1-indexed PDF page the annotation/figure came from
    note_text       : str — the user's typed comment, "" if none (always "" for figures)
  They share source/file_path/doc_type/visibility with the parent PDF's body
  chunks, so every existing delete path sweeps them along automatically.
  Figure chunks store a vision-model caption prefixed "[FIGURE p.N]".

Privacy model
-------------
  "public"  — accessible to all providers (local Ollama and Anthropic)
  "private" — accessible to the local model only

  search_with_privacy_check() enforces this at query time:
  - cloud provider  → searches public docs only; reports whether private
                       docs also matched so the user can be warned
  - local provider  → searches all docs without restriction
"""

import fcntl
import hashlib
import re
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from jarvis.core.logs import get_logger

log = get_logger(__name__)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from jarvis.core.config import get_config
from jarvis.core.llm import is_cloud_provider
from jarvis.core.errors import KBCorruptionError, LLMError, RAGError

COLLECTION_NAME = "knowledge_base"

_embeddings: HuggingFaceEmbeddings | None = None
_reranker: CrossEncoder | None = None
_store: Chroma | None = None

# Tracks write-lock nesting per thread so composite operations (refresh_vault
# calling add_texts, etc.) don't deadlock on their own flock.
_write_lock_state = threading.local()


@contextmanager
def _kb_write_lock():
    """
    Cross-process advisory lock serialising ChromaDB writes.

    The daemon, webapp, and CLI all open the same PersistentClient directory,
    and Chroma's SQLite backend is not safe for concurrent multi-process
    writers. Every write path takes this flock on <rag_dir>/.write.lock;
    reads stay unlocked (SQLite WAL handles concurrent readers).
    """
    if getattr(_write_lock_state, "depth", 0):
        _write_lock_state.depth += 1
        try:
            yield
        finally:
            _write_lock_state.depth -= 1
        return

    lock_path = get_config().rag_dir / ".write.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        _write_lock_state.depth = 1
        try:
            yield
        finally:
            _write_lock_state.depth = 0
            fcntl.flock(lock_fd, fcntl.LOCK_UN)


def _is_stale_view_error(exc: Exception) -> bool:
    """
    Whether this failure is consistent with an out-of-date view of the index
    rather than damage to it — i.e. worth reopening and retrying once.

    Both signatures arise when another process rebuilt or rewrote the index
    underneath this one: a deleted chunk id still referenced by this process's
    view, or the whole collection replaced by `kb reindex`.
    """
    text = str(exc)
    return "Error finding id" in text or ("does not exist" in text and "Collection" in text)


def _diagnose_kb_error(exc: Exception, fallback_message: str) -> RAGError:
    """
    Map a raw ChromaDB failure to the most actionable error we can raise.

    Two failure signatures get a specific diagnosis instead of a generic
    RAGError, both raised as KBCorruptionError so the chat tools relay the
    message verbatim and never retry (per the KBCorruptionError contract —
    both describe persistent state a retry would only hide):

    - "Error finding id": a stale HNSW reference to a deleted chunk — real
      on-disk corruption; the fix is `kb reindex`.
    - "Collection [...] does not exist": NOT corruption — `kb reindex` swapped
      in a rebuilt collection (new UUID) while this process was running, so
      the process-wide store singleton holds a handle to the deleted one.
      The fix is simply restarting the process.
    """
    text = str(exc)
    if "Error finding id" in text:
        return KBCorruptionError(
            "The knowledge-base search hit a chunk id that is no longer there. "
            "This search was already retried once against a freshly reopened "
            "index, so it is not simply a stale handle left over from another "
            "process writing (the usual cause, which that retry fixes). "
            "The index itself is likely damaged: run `uv run kb reindex` to "
            "rebuild it from the chunk texts already stored — nothing is lost. "
            "If even that cannot run, use `uv run kb reindex --from-storage`."
        )
    if "does not exist" in text and "Collection" in text:
        return KBCorruptionError(
            "The knowledge-base collection was rebuilt (by `kb reindex`) while "
            "this process was running, and reopening it did not help. Fix: "
            "restart this process (the webapp or jarvis-sync) — "
            "nothing is lost; the rebuilt knowledge base is intact on disk."
        )
    return RAGError(fallback_message)


# ── Singletons ────────────────────────────────────────────────────────────────


def build_embeddings(model_name: str, query_prefix: str = "") -> HuggingFaceEmbeddings:
    """
    Construct a HuggingFace embedding model.

    Embeddings are L2-normalised so cosine similarity and inner product agree.
    When query_prefix is set (BGE-style models are trained with an asymmetric
    instruction prefix), it is prepended to queries only — never to documents.
    """
    query_encode_kwargs = {"normalize_embeddings": True}
    if query_prefix:
        query_encode_kwargs["prompt"] = query_prefix
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs=query_encode_kwargs,
    )


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        cfg = get_config()
        print("Loading embedding model...", flush=True)
        _embeddings = build_embeddings(cfg.embed_model, cfg.query_prefix)
    return _embeddings


def _get_reranker() -> CrossEncoder | None:
    """
    Return the cross-encoder used to re-rank search candidates, or None when
    re-ranking is disabled (empty rerank_model in config).
    """
    global _reranker
    cfg = get_config()
    if not cfg.rerank_model:
        return None
    if _reranker is None:
        print("Loading reranker...", flush=True)
        _reranker = CrossEncoder(cfg.rerank_model)
    return _reranker


def get_store(rag_dir: Path | None = None) -> Chroma:
    """Return the process-wide Chroma vector store singleton."""
    global _store
    if _store is None:
        d = rag_dir or get_config().rag_dir
        d.mkdir(parents=True, exist_ok=True)
        cfg = get_config()
        _store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=_get_embeddings(),
            persist_directory=str(d),
            collection_metadata={
                "embed_model": cfg.embed_model,
                "query_prefix": cfg.query_prefix,
            },
        )
        _check_embedding_model_matches(_store, cfg.embed_model)
    return _store


def reset_store() -> None:
    """
    Drop the process-wide store handle so the next get_store() re-opens it.

    Needed because a long-running reader (the webapp, a chat session) holds an
    open view of the index, and another process — usually the sync daemon —
    writes to it on a schedule. After such a write the reader's view refers to
    chunk ids that no longer exist, and a search fails with "Error finding id".
    Nothing is damaged; the handle is simply out of date.
    """
    global _store
    _store = None


def _check_embedding_model_matches(store: Chroma, embed_model: str) -> None:
    """
    Guard against silently mixing embedding spaces.

    Chroma records collection_metadata only when the collection is first
    created, so a collection built with a different model keeps its original
    embed_model tag. If a non-empty collection was built with a different model
    than the config now names, retrieval would compare vectors from two
    incompatible spaces and return garbage — so we fail loudly instead.
    """
    if store._collection.count() == 0:
        return
    recorded = (store._collection.metadata or {}).get("embed_model")
    if recorded != embed_model:
        raise RAGError(
            f"Embedding model mismatch: the knowledge base was built with "
            f"'{recorded or 'unknown'}' but config specifies '{embed_model}'. "
            f"Run 'uv run kb reindex' to re-embed the knowledge base with the new model."
        )


def _splitter() -> RecursiveCharacterTextSplitter:
    cfg = get_config()
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )


# Headers we split markdown on, paired with the metadata key each level is stored under.
_MARKDOWN_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def _split_markdown(content: str) -> list[tuple[str, str]]:
    """
    Split content into chunks that respect markdown section boundaries.

    First break the text on markdown headers, then split each section further
    with the recursive character splitter so long sections still obey the chunk
    size. Every chunk is paired with a breadcrumb of the headers above it, e.g.
    "CRISPR screens › Results", so a chunk carries the context of where it came
    from. Content with no headers (such as paper summaries) passes through as a
    single section with an empty breadcrumb — behaviour identical to before.

    Returns a list of (chunk_text, breadcrumb) pairs.
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_MARKDOWN_HEADERS,
        strip_headers=False,
    )
    sections = header_splitter.split_text(content)

    recursive = _splitter()
    chunks: list[tuple[str, str]] = []
    for section in sections:
        # Build a "H1 › H2 › H3" breadcrumb from whichever header levels are present.
        breadcrumb = " › ".join(
            section.metadata[key] for _, key in _MARKDOWN_HEADERS if key in section.metadata
        )
        for chunk_text in recursive.split_text(section.page_content):
            chunks.append((chunk_text, breadcrumb))
    return chunks


# ── Core write operations ─────────────────────────────────────────────────────
#
# Each add_* function below splits into two halves: a build_* function that
# turns its input into Documents without touching the database, and one shared
# commit_documents() that performs the only write. Keeping the halves separate
# is what lets a caller assemble a whole document — body, annotations, figure
# captions — and commit it in a single atomic write. An ingest that is stopped
# or fails part-way then leaves nothing behind at all, instead of a half-indexed
# document. See _add_document in jarvis/chat/chat.py for the staged caller.


def commit_documents(
    documents: list[Document],
    store: Chroma | None = None,
    replace_source: str = "",
) -> list[str]:
    """
    Write staged documents to the knowledge base. Returns the new chunk IDs.

    All of it happens inside one write lock: when replace_source is set, the
    old entry's chunks are deleted and the new ones added together, so a
    re-ingest is never observable as "old entry gone, new entry not yet here".

    Cancellation is deliberately not checked in here. This is the point of no
    return — an interrupted Chroma write is exactly the corruption the staging
    above exists to prevent, so once we reach this function the write always
    runs to completion.
    """
    s = store or get_store()
    try:
        with _kb_write_lock():
            if replace_source:
                existing = s._collection.get(where={"source": {"$eq": replace_source}})
                if existing["ids"]:
                    s.delete(existing["ids"])
            if not documents:
                return []
            return s.add_documents(documents)
    except Exception as exc:
        raise _diagnose_kb_error(exc, f"Failed to add documents: {exc}") from exc


def build_text_documents(
    content: str,
    doc_type: str,
    visibility: str,
    source: str,
    extra_metadata: dict | None = None,
    embed_header: str = "",
) -> list[Document]:
    """
    Chunk content into Documents ready for commit_documents(). Touches nothing.

    embed_header, when set, is prepended to the EMBEDDED text of every chunk
    (not just the first) — an author-name or title-word query must be able to
    match any chunk of a long paper, not only its opening one. Metadata is
    untouched by this; only what gets embedded changes. Vault notes never
    pass embed_header.
    """
    chunks = _split_markdown(content)
    if not chunks:
        return []
    # Caller-supplied metadata first; the four fields every privacy and delete
    # path keys on go last, so nothing passed in can overwrite them.
    base_metadata = {
        **(extra_metadata or {}),
        "date_added": datetime.now(timezone.utc).isoformat(),
        "doc_type": doc_type,
        "visibility": visibility,
        "source": source,
    }
    # Each chunk gets its own metadata dict (never a shared reference) carrying
    # its position and section breadcrumb. When a chunk sits under a heading, we
    # also prepend the breadcrumb to the embedded text so a search for the note
    # topic plus the section topic can match the chunk.
    documents = []
    for index, (chunk_text, breadcrumb) in enumerate(chunks):
        metadata = {**base_metadata, "chunk_index": index, "section": breadcrumb}
        body = f"{breadcrumb}\n{chunk_text}" if breadcrumb else chunk_text
        page_content = f"{embed_header}\n{body}" if embed_header else body
        documents.append(Document(page_content=page_content, metadata=metadata))
    return documents


def add_texts(
    content: str,
    doc_type: str,
    visibility: str,
    source: str,
    extra_metadata: dict | None = None,
    store: Chroma | None = None,
    embed_header: str = "",
) -> list[str]:
    """Chunk content and add all chunks to the knowledge base. Returns chunk IDs."""
    documents = build_text_documents(
        content, doc_type, visibility, source,
        extra_metadata=extra_metadata, embed_header=embed_header,
    )
    if not documents:
        return []
    return commit_documents(documents, store)


def build_annotation_documents(
    pdf_path: Path,
    doc_type: str,
    visibility: str,
    source: str,
    title: str = "",
    file_path: str = "",
) -> list[Document]:
    """
    Extract highlights/typed comments from a PDF as Documents, one per
    annotation, ready for commit_documents(). Returns [] when the PDF has none.

    The embedded text is prefixed "[HIGHLIGHT p.N]" or "[USER NOTE p.N]" so
    retrieval (and the chat agent reading the results) can tell user-marked
    passages apart from ordinary body prose. Metadata mirrors the parent
    PDF's body chunks (same source/file_path/doc_type/visibility) so deletes
    and re-ingests sweep annotations together with the body.
    """
    from .annotations import extract_annotations

    extracted = extract_annotations(pdf_path)
    if not extracted:
        return []

    documents = []
    for index, ann in enumerate(extracted):
        if ann["kind"] == "highlight":
            page_content = f"[HIGHLIGHT p.{ann['page']}] {ann['text']}".strip()
            if ann["note_text"]:
                page_content += f"\nUser note: {ann['note_text']}"
        else:
            page_content = f"[USER NOTE p.{ann['page']}] {ann['note_text']}"
        metadata = {
            "date_added": datetime.now(timezone.utc).isoformat(),
            "doc_type": doc_type,
            "visibility": visibility,
            "source": source,
            "title": title,
            "annotation_kind": ann["kind"],
            "page": ann["page"],
            "note_text": ann["note_text"],
            "chunk_index": index,
            "section": "",
        }
        if file_path:
            metadata["file_path"] = file_path
        documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


def add_annotations(
    pdf_path: Path,
    doc_type: str,
    visibility: str,
    source: str,
    title: str = "",
    file_path: str = "",
    store: Chroma | None = None,
) -> list[str]:
    """
    Extract a PDF's highlights/typed comments and index each as its own
    searchable chunk. Returns the chunk IDs ([] when the PDF has none).
    """
    documents = build_annotation_documents(
        pdf_path, doc_type, visibility, source, title=title, file_path=file_path
    )
    if not documents:
        return []
    return commit_documents(documents, store)


def build_figure_documents(
    pdf_path: Path,
    doc_type: str,
    visibility: str,
    source: str,
    provider_obj,
    provider_str: str,
    title: str = "",
    file_path: str = "",
    *,
    enabled: bool | None = None,
    cancel=None,
) -> list[Document]:
    """
    Caption a PDF's embedded figures with the active provider's vision model
    and build one Document per figure, ready for commit_documents(). Returns []
    when there are no figures, captioning is disabled, or the privacy guard
    blocks it.

    This is the slowest build step by far — one vision call per figure — so
    cancel is checked before each one. Nothing has been written at this point,
    so a stop here simply throws the captions away.

    enabled=None follows cfg.figure_captions (off by default); enabled=True
    forces captioning for this one document — the per-document opt-in behind
    `kb add --figures` and the chat tool's with_figures. It never overrides
    the privacy guard below, only the config kill-switch.

    The embedded text is prefixed "[FIGURE p.N]" so retrieval can tell captions
    apart from body prose. Metadata mirrors the parent PDF's body chunks (same
    source/file_path/doc_type/visibility, annotation_kind="figure") so deletes
    and re-ingests sweep figures together with the body and annotations.

    Privacy guard: images of a private note must never reach a cloud model, so
    when visibility is "private" and the provider is Anthropic, captioning is
    skipped entirely with a visible warning and nothing is written. Papers are
    always public, so paper figures caption under either provider.

    Per-figure captioning failures warn and skip that one figure — a single bad
    image never aborts the ingest.
    """
    cfg = get_config()
    captions_on = cfg.figure_captions if enabled is None else enabled
    if not captions_on:
        return []

    if visibility == "private" and is_cloud_provider(provider_str):
        print(
            "  ⚠️  skipping figure captioning — images of a private note must not "
            "reach a cloud provider (switch to the local model to caption them)",
            flush=True,
        )
        return []

    from .images import extract_figures

    figures = extract_figures(
        pdf_path,
        max_figures=cfg.figure_max_per_doc,
        min_pixels=cfg.figure_min_pixels,
    )
    if not figures:
        return []

    documents = []
    for index, figure in enumerate(figures):
        if cancel is not None:
            cancel.check()
        try:
            caption = provider_obj.describe_image(
                figure["image_bytes"], context=title, cancel=cancel
            )
        except LLMError as exc:
            print(f"  ⚠️  figure caption failed (p.{figure['page']}): {exc}", flush=True)
            continue
        page_content = f"[FIGURE p.{figure['page']}] {caption}".strip()
        metadata = {
            "date_added": datetime.now(timezone.utc).isoformat(),
            "doc_type": doc_type,
            "visibility": visibility,
            "source": source,
            "title": title,
            "annotation_kind": "figure",
            "page": figure["page"],
            "note_text": "",
            "chunk_index": index,
            "section": "",
        }
        if file_path:
            metadata["file_path"] = file_path
        documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


def add_figures(
    pdf_path: Path,
    doc_type: str,
    visibility: str,
    source: str,
    provider_obj,
    provider_str: str,
    title: str = "",
    file_path: str = "",
    store: Chroma | None = None,
    *,
    enabled: bool | None = None,
) -> list[str]:
    """
    Caption a PDF's embedded figures and index one chunk per figure. Returns
    the chunk IDs ([] when there is nothing to caption).
    """
    documents = build_figure_documents(
        pdf_path, doc_type, visibility, source, provider_obj, provider_str,
        title=title, file_path=file_path, enabled=enabled,
    )
    if not documents:
        return []
    return commit_documents(documents, store)


def _source_exists(source: str, store: Chroma) -> bool:
    """Return True if any chunks with this source URL are already indexed."""
    if not source:
        return False
    try:
        result = store._collection.get(where={"source": {"$eq": source}}, include=[])
        return len(result["ids"]) > 0
    except Exception as exc:
        # False means "not a duplicate", so the caller goes ahead and adds.
        # On a broken store that quietly produces a second copy of a paper
        # you already have.
        log.warning("duplicate check by source failed, treating as new: %s", exc)
        return False


def _normalise_title(title: str) -> str:
    """Lowercase and collapse whitespace so near-identical titles compare equal."""
    return re.sub(r"\s+", " ", title.strip().lower())


def _title_exists(title: str, store: Chroma) -> bool:
    """
    Return True if a paper with this title is already indexed, matching on the
    normalised title. A paper can now arrive via two sources (arXiv + bioRxiv)
    with different URLs, so source-URL dedup alone is no longer enough. This
    scans stored paper metadata — fine at personal-KB scale.
    """
    if not title:
        return False
    target = _normalise_title(title)
    try:
        result = store._collection.get(where={"doc_type": {"$eq": "paper"}}, include=["metadatas"])
        return any(_normalise_title(m.get("title", "")) == target for m in result["metadatas"])
    except Exception as exc:
        log.warning("duplicate check by title failed, treating as new: %s", exc)
        return False


def build_paper_documents(
    paper: dict,
    dense_summary: str,
    score: int = 0,
    track: str = "",
    storage_mode: str = "summary",
) -> list[Document]:
    """
    Assemble a paper's summary into Documents ready for commit_documents().

    The chunked text leads with title, source URL and authors so a chunk can
    match on any of them, then carries the dense summary.
    """
    source = paper.get("link", "")
    header_lines = [paper.get("title", ""), source]
    if paper.get("authors"):
        header_lines.append(paper["authors"])
    content = "\n".join(header_lines) + f"\n\n{dense_summary}"
    return build_text_documents(
        content=content,
        doc_type="paper",
        visibility="public",
        source=source,
        extra_metadata={
            "title": paper.get("title", ""),
            "authors": paper.get("authors", ""),
            "doi": paper.get("doi", ""),
            "score": int(score),
            "track": str(track),
            "storage_mode": storage_mode,
        },
    )


def add_paper(
    paper: dict,
    dense_summary: str,
    score: int = 0,
    track: str = "",
    store: Chroma | None = None,
    storage_mode: str = "summary",
    allow_duplicate: bool = False,
) -> list[str]:
    """
    Add a paper to the knowledge base. Papers are always public. Skips (returns
    []) if a paper with the same source URL or the same title is already
    indexed, unless allow_duplicate is set (the user explicitly chose to add it
    anyway).
    """
    s = store or get_store()
    source = paper.get("link", "")
    if not allow_duplicate and (_source_exists(source, s) or _title_exists(paper.get("title", ""), s)):
        return []
    documents = build_paper_documents(
        paper, dense_summary, score=score, track=track, storage_mode=storage_mode
    )
    if not documents:
        return []
    return commit_documents(documents, s)


def add_papers_batch(
    entries: list[tuple[dict, dict]],
    store: Chroma | None = None,
) -> tuple[int, int]:
    """
    Batch-add papers from a digest scoring run.
    Reuses existing summary+why fields — no extra LLM call.
    Returns (added, skipped): skipped papers were already in the KB (matched by
    source URL or title), so add_paper returned no chunk ids for them.
    """
    s = store or get_store()
    added = skipped = 0
    for paper, selected in entries:
        summary = "\n\n".join(
            filter(None, [selected.get("summary", ""), selected.get("why", "")])
        )
        ids = add_paper(
            paper=paper,
            dense_summary=summary,
            score=selected.get("score", 0),
            track=selected.get("track", ""),
            store=s,
        )
        if ids:
            added += 1
        else:
            skipped += 1
    return added, skipped


def delete_by_metadata(
    key: str,
    value: str,
    store: Chroma | None = None,
) -> int:
    """Delete all chunks matching a metadata key=value pair. Returns count deleted."""
    s = store or get_store()
    try:
        with _kb_write_lock():
            result = s._collection.get(where={key: {"$eq": value}})
            ids = result["ids"]
            if ids:
                s.delete(ids)
            return len(ids)
    except Exception as exc:
        raise RAGError(f"Delete failed: {exc}") from exc


def update_paper_metadata(
    source: str,
    title: str | None = None,
    authors: str | None = None,
    doi: str | None = None,
    store: Chroma | None = None,
) -> int:
    """
    Apply user-verified title/authors/doi to every chunk of a paper, no
    re-embedding. Only the fields the caller passes (not None) are changed.
    Returns the chunk count updated.
    """
    s = store or get_store()
    try:
        result = s._collection.get(where={"source": {"$eq": source}}, include=["metadatas"])
    except Exception as exc:
        raise RAGError(f"Failed to look up source: {exc}") from exc

    ids = result["ids"]
    if not ids:
        return 0

    updated_metadatas = []
    for meta in result["metadatas"]:
        new_meta = dict(meta)
        if title is not None:
            new_meta["title"] = title
        if authors is not None:
            new_meta["authors"] = authors
        if doi is not None:
            new_meta["doi"] = doi
        updated_metadatas.append(new_meta)

    try:
        with _kb_write_lock():
            s._collection.update(ids=ids, metadatas=updated_metadatas)
    except Exception as exc:
        raise RAGError(f"Failed to update metadata: {exc}") from exc

    return len(ids)


# ── Legacy PDF-note migration (kb doctor) ──────────────────────────────────────
#
# Local PDFs are now always public papers — notes come exclusively from the
# Obsidian vault (.md files). Entries added before that decision may still
# carry doc_type="note" with an absolute PDF file_path; these two helpers let
# `kb doctor` find and reclassify them.


def find_pdf_notes(store: Chroma | None = None) -> list[dict]:
    """
    Find leftover doc_type="note" chunks whose file_path is a local PDF path.
    Returns one summary dict per distinct source, sorted by source:
    {"source", "title", "visibility", "chunk_count"}.
    """
    s = store or get_store()
    result = s._collection.get(where={"doc_type": {"$eq": "note"}}, include=["metadatas"])
    by_source: dict[str, dict] = {}
    for meta in result["metadatas"]:
        file_path = meta.get("file_path", "")
        if not file_path.lower().endswith(".pdf"):
            continue
        source = meta.get("source", file_path)
        entry = by_source.setdefault(source, {
            "source": source,
            "title": meta.get("title", "untitled"),
            "visibility": meta.get("visibility", "public"),
            "chunk_count": 0,
        })
        entry["chunk_count"] += 1
    return sorted(by_source.values(), key=lambda entry: entry["source"])


def reclassify_notes_as_papers(sources: list[str], store: Chroma | None = None) -> int:
    """
    Flip doc_type from "note" to "paper" for every chunk belonging to the
    given sources — every other field (content_hash, storage_mode, file_path,
    ...) is left untouched, so the result has the same shape a
    daemon-ingested paper carries. Private chunks are skipped outright —
    papers are always public, so that invariant holds here by construction
    rather than by caller discipline. Returns the number of chunks updated.
    """
    if not sources:
        return 0
    s = store or get_store()
    result = s._collection.get(where={"source": {"$in": sources}}, include=["metadatas"])
    ids_to_update = []
    updated_metadatas = []
    for chunk_id, meta in zip(result["ids"], result["metadatas"]):
        if meta.get("visibility") == "private":
            continue
        new_meta = dict(meta)
        new_meta["doc_type"] = "paper"
        ids_to_update.append(chunk_id)
        updated_metadatas.append(new_meta)
    if not ids_to_update:
        return 0
    with _kb_write_lock():
        s._collection.update(ids=ids_to_update, metadatas=updated_metadatas)
    return len(ids_to_update)


# ── Search ────────────────────────────────────────────────────────────────────


def _reciprocal_rank_fusion(dense_ids: list[str], sparse_ids: list[str], c: int = 60) -> list[str]:
    """
    Fuse two chunk-id rankings: each id's score is the sum of 1/(c + rank)
    across whichever ranking(s) it appears in (rank is 1-based; absence from
    a ranking contributes 0 for that ranking). c=60 is the standard RRF
    damping constant — it keeps a low rank in one list from dominating.
    Returns ids sorted by fused score, descending.
    """
    scores: dict[str, float] = {}
    for ranking in (dense_ids, sparse_ids):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (c + rank)
    return sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)


def _hybrid_search(query: str, fetch_k: int, filter_dict: dict | None, s: Chroma) -> list[Document]:
    """
    Combine dense (embedding) and sparse (BM25) retrieval over the SAME
    privacy-filtered candidate pool, fused by reciprocal rank fusion.

    Privacy holds by construction: the where filter is applied when fetching
    the pool, and the dense query below uses the identical filter — so no id
    outside the filtered pool can ever be scored or returned.
    """
    get_kwargs: dict = {"include": ["documents", "metadatas"]}
    if filter_dict:
        get_kwargs["where"] = filter_dict
    pool = s._collection.get(**get_kwargs)
    pool_ids = pool["ids"]
    if not pool_ids:
        return []
    pool_lookup = {
        doc_id: Document(page_content=text, metadata=meta)
        for doc_id, text, meta in zip(pool_ids, pool["documents"], pool["metadatas"])
    }

    # Sparse ranking: BM25 rebuilt fresh per query over the SAME filtered
    # pool — cheap at personal-KB scale (tens of ms for thousands of chunks).
    tokenized_pool = [doc.lower().split() for doc in pool["documents"]]
    bm25 = BM25Okapi(tokenized_pool)
    scores = bm25.get_scores(query.lower().split())
    order = sorted(range(len(pool_ids)), key=lambda i: scores[i], reverse=True)
    sparse_ids = [pool_ids[i] for i in order[:fetch_k]]

    # Dense ranking: vector query through the identical where filter, so its
    # results are guaranteed to be a subset of the same privacy-filtered pool.
    query_vector = s._embedding_function.embed_query(query)
    query_kwargs: dict = {"query_embeddings": [query_vector], "n_results": fetch_k, "include": []}
    if filter_dict:
        query_kwargs["where"] = filter_dict
    dense_ids = s._collection.query(**query_kwargs)["ids"][0]

    fused_ids = _reciprocal_rank_fusion(dense_ids, sparse_ids)[:fetch_k]
    return [pool_lookup[doc_id] for doc_id in fused_ids if doc_id in pool_lookup]


def search(
    query: str,
    n_results: int = 5,
    visibility: str | None = None,
    doc_type: str | list[str] | None = None,
    annotation_kind: str | None = None,
    store: Chroma | None = None,
    rerank: bool = True,
    category: str | None = None,
    status: str | None = None,
    entity: str | None = None,
    fields: dict | None = None,
    tags: list[str] | None = None,
) -> list[Document]:
    """
    Semantic search with optional metadata filters. doc_type accepts a single
    type or a list of types (e.g. ["paper", "digest"], matched with $in).

    Record filters — category/status/entity, plus `fields` for any custom
    frontmatter key (e.g. {"x_venue": "NeurIPS"}) — fold into the SAME `where`
    clause as the visibility filter, so they can only ever narrow the pool the
    privacy filter already produced. Privacy therefore holds by construction
    here, not by a second check.

    `tags` is the one exception: ChromaDB has no substring or list-contains
    operator, so tag filtering happens on the returned metadata AFTER
    re-ranking. That means a tag filter can only shrink the result set it is
    handed — ask for a larger n_results when filtering by tag.

    When cfg.hybrid is set (the default), candidates come from _hybrid_search
    (dense + BM25 fused by reciprocal rank fusion) instead of plain
    similarity_search — this is what lets a query for an author's name or an
    acronym match a chunk that the embedding model alone would miss.

    When a reranker is configured and rerank=True, this fetches a wider pool of
    candidates (rerank_top_n) with the embedding model, then re-orders them with
    a cross-encoder and returns the top n_results. The cross-encoder scores each
    (query, chunk) pair jointly, which is far more accurate than the bi-encoder's
    independent embeddings — the right chunk is much more likely to land in the
    top few. Filters are applied by ChromaDB before re-ranking, so re-ranking
    never widens visibility.
    """
    s = store or get_store()
    conditions = []
    if visibility:
        conditions.append({"visibility": {"$eq": visibility}})
    if doc_type:
        # A list of types becomes an $in filter; both the hybrid and the plain
        # branch below receive the same filter_dict, so the widening applies
        # to whichever retrieval path runs.
        if isinstance(doc_type, str):
            conditions.append({"doc_type": {"$eq": doc_type}})
        else:
            conditions.append({"doc_type": {"$in": list(doc_type)}})
    if annotation_kind:
        conditions.append({"annotation_kind": {"$eq": annotation_kind}})
    for field, value in (
        ("category", category), ("status", status), ("entity", entity),
        *(fields or {}).items(),
    ):
        if value:
            conditions.append({field: {"$eq": value}})

    filter_dict = None
    if len(conditions) == 1:
        filter_dict = conditions[0]
    elif len(conditions) > 1:
        filter_dict = {"$and": conditions}

    reranker = _get_reranker() if rerank else None
    fetch_k = max(n_results, get_config().rerank_top_n) if reranker else n_results

    cfg = get_config()

    def run(active_store):
        if cfg.hybrid:
            return _hybrid_search(query, fetch_k, filter_dict, active_store)
        return active_store.similarity_search(query, k=fetch_k, filter=filter_dict)

    try:
        candidates = run(s)
    except Exception as exc:
        # A long-running reader (webapp, chat session) holds an open view of
        # the index while the sync daemon writes to it on a schedule. After a
        # write, the reader's view names ids that no longer exist and the
        # search fails — but nothing is damaged, the handle is just out of
        # date. Reopen and try once more. Only a second failure is treated as
        # real corruption, so a persistent problem still surfaces rather than
        # being retried away.
        if store is not None or not _is_stale_view_error(exc):
            raise _diagnose_kb_error(exc, f"Search failed: {exc}") from exc
        reset_store()
        try:
            candidates = run(get_store())
        except Exception as retry_exc:
            raise _diagnose_kb_error(retry_exc, f"Search failed: {retry_exc}") from retry_exc

    if reranker is not None and len(candidates) > n_results:
        scores = reranker.predict([(query, doc.page_content) for doc in candidates])
        candidates = [
            doc for _, doc in
            sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
        ]

    if tags:
        from .frontmatter import has_tag

        candidates = [
            doc for doc in candidates
            if all(has_tag(doc.metadata.get("tags", ""), tag) for tag in tags)
        ]

    return candidates[:n_results]


def search_with_privacy_check(
    query: str,
    provider: str,
    n_results: int = 5,
    doc_type: str | list[str] | None = None,
    store: Chroma | None = None,
    **filters,
) -> tuple[list[Document], bool]:
    """
    Search with provider-aware privacy handling.

    Returns (results, has_private_hits).

    For cloud providers (anything but ollama):
      - Returns public docs only
      - has_private_hits=True if private docs also matched (so the caller
        can warn the user that results may be incomplete)

    For local providers (Ollama):
      - Returns all docs regardless of visibility
      - has_private_hits is always False

    `filters` (category/status/entity/fields/tags) pass straight through to
    search(), where they narrow the already-privacy-filtered pool. They are
    applied to the private-hit probe too, so the "some matches were excluded"
    caveat reflects the same query the user actually asked.
    """
    s = store or get_store()
    if is_cloud_provider(provider):
        results = search(query, n_results=n_results, visibility="public",
                         doc_type=doc_type, store=s, **filters)
        try:
            # A cheap existence probe — order doesn't matter, so skip re-ranking.
            # The retrieved private document stays in local process memory only;
            # nothing beyond len(...) is used, and the probe runs before any
            # cloud request, so private content cannot leak through this path.
            private_check = search(query, n_results=1, visibility="private",
                                   doc_type=doc_type, store=s, rerank=False, **filters)
            has_private = len(private_check) > 0
        except RAGError as exc:
            # The public results above are already correctly filtered, so no
            # private content leaks — but the "some matches were excluded"
            # caveat is silently dropped, which is worth knowing about.
            log.warning("private-content probe failed, caveat suppressed: %s", exc)
            has_private = False
        return results, has_private
    else:
        return search(query, n_results=n_results, doc_type=doc_type, store=s, **filters), False


def metadata_key_counts(store: Chroma | None = None) -> dict[str, int]:
    """
    How many chunks carry each metadata key, for `kb schema`.

    The point is to make the user's own ontology visible: which record types,
    statuses, and custom `x_` fields actually exist in the store, so a typo
    ("stauts: rejected") shows up as its own key rather than silently never
    matching a filter.
    """
    s = store or get_store()
    counts: dict[str, int] = {}
    try:
        result = s._collection.get(include=["metadatas"])
    except Exception as exc:
        raise _diagnose_kb_error(exc, f"Failed to read metadata: {exc}") from exc
    for meta in result["metadatas"] or []:
        for key in meta:
            counts[key] = counts.get(key, 0) + 1
    return counts


def metadata_value_counts(key: str, store: Chroma | None = None) -> dict[str, int]:
    """Distinct values of one metadata key with their chunk counts."""
    s = store or get_store()
    counts: dict[str, int] = {}
    try:
        result = s._collection.get(include=["metadatas"])
    except Exception as exc:
        raise _diagnose_kb_error(exc, f"Failed to read metadata: {exc}") from exc
    for meta in result["metadatas"] or []:
        if key in meta:
            value = str(meta[key])
            counts[value] = counts.get(value, 0) + 1
    return counts


# ── Stats ─────────────────────────────────────────────────────────────────────


def count(store: Chroma | None = None) -> int:
    """Total number of chunks in the knowledge base."""
    s = store or get_store()
    return s._collection.count()


def count_unique_documents(
    doc_type: str,
    id_key: str,
    store: Chroma | None = None,
) -> int:
    """Count unique documents of a given type, de-duplicated by id_key metadata."""
    s = store or get_store()
    try:
        result = s._collection.get(
            where={"doc_type": {"$eq": doc_type}},
            include=["metadatas"],
        )
        return len({m.get(id_key) for m in result["metadatas"] if m.get(id_key)})
    except Exception as exc:
        # 0 reads as "your knowledge base is empty", which is alarming and
        # wrong when the real answer is "the count could not be read".
        log.warning("could not count %s documents: %s", doc_type, exc)
        return 0


def update_file_path(source: str, new_path: str, store: Chroma | None = None) -> int:
    """
    Update the file_path metadata (and source URI for local files) for all chunks
    matching the given source. Returns the number of chunks updated.
    """
    s = store or get_store()
    try:
        result = s._collection.get(
            where={"source": {"$eq": source}}, include=["metadatas", "documents", "embeddings"]
        )
    except Exception as exc:
        raise RAGError(f"Failed to look up source: {exc}") from exc

    ids = result["ids"]
    if not ids:
        return 0

    new_path_str = str(Path(new_path).expanduser().resolve())
    new_source = Path(new_path_str).as_uri() if source.startswith("file:///") else source

    updated_metadatas = []
    for meta in result["metadatas"]:
        updated_metadatas.append({**meta, "file_path": new_path_str, "source": new_source})

    try:
        with _kb_write_lock():
            s._collection.update(ids=ids, metadatas=updated_metadatas)
    except Exception as exc:
        raise RAGError(f"Failed to update metadata: {exc}") from exc

    return len(ids)


def get_document_chunks(source: str, store: Chroma | None = None) -> list[Document]:
    """
    Fetch every chunk belonging to one document, in reading order.

    Body chunks come first (sorted by chunk_index), followed by annotation
    and figure chunks (identified by the presence of annotation_kind
    metadata, also sorted by chunk_index) — this lets a caller page through
    the source text before hitting highlights/figure captions. Returns []
    for an unknown source rather than raising, mirroring update_file_path.
    """
    s = store or get_store()
    try:
        result = s._collection.get(
            where={"source": {"$eq": source}}, include=["documents", "metadatas"]
        )
    except Exception as exc:
        raise RAGError(f"Failed to look up source: {exc}") from exc

    chunks = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(result["documents"], result["metadatas"])
    ]
    body_chunks = sorted(
        (doc for doc in chunks if "annotation_kind" not in doc.metadata),
        key=lambda doc: doc.metadata.get("chunk_index", 0),
    )
    annotation_chunks = sorted(
        (doc for doc in chunks if "annotation_kind" in doc.metadata),
        key=lambda doc: doc.metadata.get("chunk_index", 0),
    )
    return body_chunks + annotation_chunks


def update_metadata_fields(
    file_path: str,
    fields: dict,
    store: Chroma | None = None,
) -> int:
    """
    Set metadata fields on every chunk of one file, without touching content
    or re-embedding. Returns the number of chunks updated.

    This is the cheap half of refresh_vault: when only jarvis's interpretation
    of a file has moved on — not the file itself — there is nothing to
    re-embed, and rewriting the vectors would be pure cost plus a window where
    the note is missing from the index.
    """
    s = store or get_store()
    try:
        result = s._collection.get(
            where={"file_path": {"$eq": file_path}}, include=["metadatas"]
        )
    except Exception as exc:
        raise RAGError(f"Failed to look up file_path: {exc}") from exc

    ids = result["ids"]
    if not ids:
        return 0

    updated_metadatas = [{**m, **fields} for m in result["metadatas"]]
    try:
        with _kb_write_lock():
            s._collection.update(ids=ids, metadatas=updated_metadatas)
    except Exception as exc:
        raise RAGError(f"Failed to update metadata: {exc}") from exc

    return len(ids)


def update_visibility(file_path: str, new_visibility: str, store: Chroma | None = None) -> int:
    """
    Update the visibility metadata for all chunks of a vault note, without
    touching content or re-embedding. Used by refresh_vault when a file's
    classification changes (e.g. private_vault_dirs edited in config) even
    though its content and path did not.
    """
    return update_metadata_fields(file_path, {"visibility": new_visibility}, store)


def update_chat_title(session_id: str, new_title: str, store: Chroma | None = None) -> int:
    """
    Update the title metadata for all indexed chat chunks of one session, so a
    renamed session shows its new name in search_chat_history results. Metadata
    only — no content change, no re-embedding. Returns the chunk count updated.
    """
    s = store or get_store()
    try:
        result = s._collection.get(
            where={"session_id": {"$eq": session_id}}, include=["metadatas"]
        )
    except Exception as exc:
        raise RAGError(f"Failed to look up session chunks: {exc}") from exc

    ids = result["ids"]
    if not ids:
        return 0

    updated_metadatas = [{**m, "title": new_title} for m in result["metadatas"]]
    try:
        with _kb_write_lock():
            s._collection.update(ids=ids, metadatas=updated_metadatas)
    except Exception as exc:
        raise RAGError(f"Failed to update chat title: {exc}") from exc

    return len(ids)


def list_documents(
    limit: int = 10_000,
    doc_type: str | list[str] = "paper",
    category: str | None = None,
    status: str | None = None,
    entity: str | None = None,
    store: Chroma | None = None,
) -> list[dict]:
    """
    De-duplicated list of indexed documents as metadata dicts, most recently
    added first, with a chunk_count per document.

    Papers are keyed by `source`; notes share `source="local"` and are keyed by
    `file_path` instead, so listing notes returns one entry per file rather
    than one entry for the entire vault.

    The default limit returns every document in a single-user KB in one call;
    callers wanting a short preview pass their own smaller limit.
    """
    s = store or get_store()
    conditions: list[dict] = []
    if isinstance(doc_type, str):
        conditions.append({"doc_type": {"$eq": doc_type}})
    else:
        conditions.append({"doc_type": {"$in": list(doc_type)}})
    for field, value in (("category", category), ("status", status), ("entity", entity)):
        if value:
            conditions.append({field: {"$eq": value}})
    where = conditions[0] if len(conditions) == 1 else {"$and": conditions}

    try:
        result = s._collection.get(where=where, include=["metadatas"])
    except Exception as exc:
        raise RAGError(f"List failed: {exc}") from exc

    chunk_counts: dict[str, int] = {}
    first_meta: dict[str, dict] = {}
    for meta in result["metadatas"]:
        key = meta.get("file_path") if meta.get("doc_type") == "note" else meta.get("source")
        if not key:
            continue
        chunk_counts[key] = chunk_counts.get(key, 0) + 1
        if key not in first_meta:
            first_meta[key] = meta

    documents = [{**meta, "chunk_count": chunk_counts[key]} for key, meta in first_meta.items()]
    documents.sort(key=lambda d: d.get("date_added", ""), reverse=True)
    return documents[:limit]


# ── Vault indexing ────────────────────────────────────────────────────────────


def get_visibility(file_path: Path, vault_root: Path) -> str:
    """Determine document visibility from vault folder structure."""
    try:
        parts = file_path.relative_to(vault_root).parts
        if parts and parts[0] in get_config().private_vault_dirs:
            return "private"
    except ValueError:
        pass
    return "public"


def index_vault_file(
    file_path: Path,
    vault_root: Path,
    store: Chroma | None = None,
) -> list[str]:
    """
    Chunk and index a single vault .md file. Returns list of chunk IDs.

    YAML frontmatter, when present, becomes filterable metadata (see
    jarvis/kb/frontmatter.py) — that is what lets a note be a job application
    with an outcome or a manuscript with a venue rather than just prose. The
    frontmatter block itself is stripped before chunking (it is structure, not
    content the user would want retrieved), but the record header derived from
    it is embedded into every chunk so a query naming the record's type,
    entity, or status can match any part of it.
    """
    from .frontmatter import META_SCHEMA, parse_frontmatter, record_header

    content = file_path.read_text(encoding="utf-8", errors="replace")
    rel_path = str(file_path.relative_to(vault_root))
    visibility = get_visibility(file_path, vault_root)
    # Hashed over the WHOLE file, frontmatter included — editing only the
    # status of a record must still count as a change.
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    modified_at = datetime.fromtimestamp(
        file_path.stat().st_mtime, tz=timezone.utc
    ).isoformat()

    record_metadata, body = parse_frontmatter(content, file_label=rel_path)
    title_match = re.search(r"^#\s+(.+)", body, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else file_path.stem

    return add_texts(
        content=body,
        doc_type="note",
        visibility=visibility,
        source="local",
        extra_metadata={
            # Record fields FIRST, jarvis's own keys after, so the note's own
            # frontmatter can never win a collision. parse_frontmatter already
            # namespaces user keys under x_, but ordering makes the guarantee
            # structural rather than dependent on that table staying correct.
            **record_metadata,
            "file_path": rel_path,
            "title": title,
            "content_hash": content_hash,
            "modified_at": modified_at,
            # Stamped on every note, frontmatter or not, so refresh_vault can
            # tell "indexed under the current schema" from "needs a backfill".
            "meta_schema": META_SCHEMA,
        },
        embed_header=record_header(record_metadata),
        store=store,
    )


def refresh_vault(
    vault_root: Path,
    store: Chroma | None = None,
) -> tuple[int, int, int]:
    """
    Incrementally sync the vault index with the filesystem.
    Indexes new/changed files, removes chunks for deleted files.
    Returns (added, updated, deleted) file counts.
    Safe to call on an empty collection.
    """
    s = store or get_store()

    from .frontmatter import META_SCHEMA

    # Build map of currently indexed notes:
    #   file_path → (content_hash, visibility, meta_schema)
    try:
        result = s._collection.get(
            where={"doc_type": {"$eq": "note"}},
            include=["metadatas"],
        )
        indexed: dict[str, tuple[str, str, int]] = {}
        for meta in result["metadatas"]:
            fp = meta.get("file_path", "")
            if fp and fp not in indexed:
                indexed[fp] = (
                    meta.get("content_hash", ""),
                    meta.get("visibility", "public"),
                    int(meta.get("meta_schema", 0)),
                )
    except Exception as exc:
        # Falling back to an empty map would look like "nothing is indexed":
        # refresh_vault would re-index the entire vault and treat every
        # existing note as new. A read failure here is a real fault, so it
        # surfaces instead of quietly changing what the sync does.
        log.error("refresh_vault: could not read the indexed note map: %s", exc)
        raise _diagnose_kb_error(exc, f"Failed to read the vault index: {exc}") from exc

    # Scan current vault files
    current: dict[str, tuple[Path, str]] = {}
    for md_file in vault_root.rglob("*.md"):
        try:
            text = md_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = str(md_file.relative_to(vault_root))
        current[rel] = (md_file, hashlib.sha256(text.encode()).hexdigest())

    added = updated = deleted = 0

    for rel_path, (file_path, file_hash) in current.items():
        stored = indexed.get(rel_path)
        if stored is None:
            index_vault_file(file_path, vault_root, s)
            added += 1
            continue
        stored_hash, stored_visibility, stored_schema = stored
        if stored_hash != file_hash:
            # The file itself changed: its text must be re-chunked and
            # re-embedded, so the delete-and-re-add is unavoidable.
            delete_by_metadata("file_path", rel_path, s)
            index_vault_file(file_path, vault_root, s)
            updated += 1
        elif stored_schema < META_SCHEMA:
            # The file is untouched; only jarvis's reading of it moved on.
            # Whether that needs a re-embed depends on one thing: does the note
            # have a frontmatter block? If it does, both the indexed body (the
            # block is now stripped from it) and the embedded record header
            # change, so it has to be re-indexed. If it does not — the common
            # case for a plain note — nothing about its vectors would differ,
            # so stamp the marker and move on. That keeps a schema migration
            # from re-embedding an entire vault to record that most of it had
            # nothing to record.
            from .frontmatter import split_frontmatter

            raw, _ = split_frontmatter(
                file_path.read_text(encoding="utf-8", errors="replace")
            )
            if raw.strip():
                delete_by_metadata("file_path", rel_path, s)
                index_vault_file(file_path, vault_root, s)
            else:
                update_metadata_fields(rel_path, {"meta_schema": META_SCHEMA}, s)
            updated += 1
        else:
            # Content unchanged, but the classification rule may have changed
            # (private_vault_dirs config edits). Stale visibility would let a
            # cloud provider keep seeing a note that is now private.
            current_visibility = get_visibility(file_path, vault_root)
            if current_visibility != stored_visibility:
                update_visibility(rel_path, current_visibility, s)
                updated += 1

    for rel_path in indexed:
        if rel_path not in current:
            delete_by_metadata("file_path", rel_path, s)
            deleted += 1

    return added, updated, deleted
