"""
Knowledge base storage and retrieval, using LangChain + ChromaDB.

One collection with a flat document schema. What goes into it — which files,
and when — is decided by jarvis/kb/mirror.py; this module only stores chunks
and searches them.

Document schema
---------------
  page_content : str   — chunked text (embedded for similarity search)
  metadata:
    date_added  : str  — ISO timestamp of when the chunk was indexed
    doc_type    : str  — "paper" | "note"  ("chat"/"digest" only on indexes
                         built by older versions, until `kb rebuild`)
    visibility  : str  — always "public" for anything indexed now; private
                         vault folders are never indexed
    source      : str  — file:/// URI for papers, "local" for vault notes
    title       : str  — display title
    authors     : str  — comma-separated authors, papers only (optional)
    doi         : str  — DOI, papers only (optional)
    file_path   : str  — vault-relative path for notes, absolute for papers
    content_hash: str  — SHA-256 of the whole file, for change and move detection
    chunk_index : int  — position of this chunk within its source document
    section     : str  — markdown header breadcrumb ("H1 › H2"), "" if none
    modified_at : str  — ISO mtime of the source file, vault notes only

  PDF annotation chunks (see build_annotation_documents) additionally carry:
    annotation_kind : str — "highlight" | "comment" (absent on body chunks)
    page            : int — 1-indexed PDF page the annotation came from
    note_text       : str — the user's typed comment, "" if none
  They share source/file_path/doc_type/visibility with the paper's body
  chunks, so every delete path sweeps them along automatically.
"""

import fcntl
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from chromadb.config import Settings
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
from jarvis.core.errors import KBCorruptionError, RAGError

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
    Cross-process advisory lock serialising direct writes to the index files.

    Narrower than it used to be. Normally the knowledge-base server owns the
    index and serialises everything itself, so this lock has nothing to do.
    It still matters on the direct path — one-shot `kb` commands opening the
    directory themselves because the server is not running — where two of them
    could otherwise write to the same SQLite file at once.

    Reads stay unlocked. Note that WAL only ever covered half the store: the
    chunk text and metadata live in SQLite, the vectors in a separate segment
    it knows nothing about. Relying on that distinction is what produced stale
    readers, and is why the server exists.
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


def _diagnose_kb_error(exc: Exception, fallback_message: str) -> RAGError:
    """
    Map a raw ChromaDB failure to the most actionable error we can raise.

    Three signatures get a specific diagnosis instead of a generic RAGError:

    - The server not answering. Everything else is unreachable too, so say so
      rather than letting a connection error surface as a search failure.

    - "Error finding id": the index could not resolve a chunk id. This used to
      be diagnosed as on-disk corruption, which was wrong — it was a process
      holding a cached copy of the vector index while another process rewrote
      it, and it went away the moment that reader was restarted. The server
      owns the index now, so a reader cannot hold a stale copy at all, and a
      one-shot command on the direct path opened the index moments ago. If it
      still appears, the index genuinely cannot resolve its own ids, and
      `kb rebuild` (from the files) is the answer.

    - "Collection [...] does not exist": survives the move to a server. A
      client resolves a collection to an id once, and `kb rebuild` swaps in a
      rebuilt collection with a new id, so a process that was already
      connected still names the old one. Restarting it is still the fix.
    """
    text = str(exc)
    if _looks_like_server_down(exc):
        return RAGError(f"{_SERVER_DOWN_HELP}\n  ({exc})")
    if "Error finding id" in text:
        return KBCorruptionError(
            "The knowledge-base index could not resolve a chunk id it holds a "
            "reference to. Run `uv run kb doctor` to check the index; "
            "`uv run kb rebuild` recreates it from your vault and papers "
            "folders — nothing is lost, every entry comes from a file."
        )
    if "does not exist" in text and "Collection" in text:
        return KBCorruptionError(
            "The knowledge-base collection was rebuilt (by `kb rebuild`) while "
            "this process was running, so it still refers to the collection "
            "that was replaced. Fix: restart this process (your MCP client, "
            "or `kb serve`) — nothing is lost; the rebuilt index is intact."
        )
    return RAGError(fallback_message)


def _looks_like_server_down(exc: Exception) -> bool:
    """
    Whether this failure is the server being unreachable rather than a problem
    with the data.

    Matched on the exception text because the httpx/chromadb error types differ
    by transport and version. The markers are deliberately specific: "connect"
    on its own would also swallow "connection reset by peer", which is a
    different fault and deserves its own message.
    """
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "connection refused",
            "failed to connect",
            "could not connect",
            "cannot connect",
            "max retries",
        )
    )


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


def _chroma_settings() -> Settings:
    """
    Chroma client settings, with its usage reporting switched off.

    Built fresh per client rather than shared: constructing an HttpClient
    stamps the server transport into the Settings object it is handed, so a
    reused instance would then tell a PersistentClient to go looking for a
    server too — which is exactly the fallback path that must not do that.

    Telemetry is a constructor argument rather than an environment variable,
    which is why it lives here and not in jarvis/__init__.py. In the version
    pinned today Chroma's reporting class is already a no-op stub, but the
    default is True, so setting it stops an upgrade quietly re-enabling it.
    """
    return Settings(anonymized_telemetry=False)

_SERVER_DOWN_HELP = (
    "The knowledge-base server is not running.\n"
    "  Start it with:  uv run kb serve\n"
    "  It needs to stay running (tmux or its own terminal)."
)


def _server_client(port: int):
    """
    Connect to the knowledge-base server, or raise if it is not answering.

    Bound to 127.0.0.1 by both sides: Chroma's server has no authentication, so reachability from outside this
    machine is the thing that must never become configurable.
    """
    import chromadb

    client = chromadb.HttpClient(host="127.0.0.1", port=port, settings=_chroma_settings())
    client.heartbeat()  # cheap round-trip; raises when nothing is listening
    return client


# Set once by the entry points that are one-shot commands rather than
# long-lived services (the kb CLI). A flag rather than an argument
# threaded through every call site: get_store() is called from a dozen places
# per command, and one missed kwarg would mean a command that dies whenever
# the server is down — precisely the case the fallback exists for.
_allow_direct = False


def allow_direct_index_access() -> None:
    """
    Let this process open the index files itself if the server is not running.

    Only for commands that exit in seconds. A long-lived process must never
    call this: holding the index open across another process's write is the
    fault the server was introduced to remove.
    """
    global _allow_direct
    _allow_direct = True


def open_index_client():
    """
    Return (client, via_server): a raw Chroma client for the index.

    get_store() wraps one fixed collection; `kb rebuild` needs the client
    itself so it can build a second collection beside the live one and swap
    it in. Same routing rule as get_store(): through the server when it is up,
    straight to the files only for a one-shot command and only when it is down.
    """
    import chromadb

    cfg = get_config()
    try:
        return _server_client(cfg.server_port), True
    except Exception as exc:
        if not _allow_direct:
            raise RAGError(f"{_SERVER_DOWN_HELP}\n  ({exc})") from exc
    cfg.rag_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(cfg.rag_dir, 0o700)
    return chromadb.PersistentClient(path=str(cfg.rag_dir), settings=_chroma_settings()), False


def open_collection(client, name: str) -> Chroma:
    """A LangChain store over one named collection, tagged with the embedding model."""
    cfg = get_config()
    return Chroma(
        client=client,
        collection_name=name,
        embedding_function=_get_embeddings(),
        collection_metadata={"embed_model": cfg.embed_model, "query_prefix": cfg.query_prefix},
    )


def forget_store() -> None:
    """Drop the cached store, so the next get_store() resolves the collection afresh."""
    global _store
    _store = None


def get_store(rag_dir: Path | None = None, allow_direct: bool = False) -> Chroma:
    """
    Return the process-wide Chroma vector store singleton.

    One process owns the index — the server started by `uv run kb serve` —
    and everything else talks to it. That is what stops a long-lived reader
    holding a cached copy of the vector index while another process rewrites
    it underneath (see docs/DESIGN.md, "Who owns the index").

    allow_direct=True opens the directory in-process instead when the server
    is not running. It is for one-shot `kb` commands only: they exit in
    seconds, so they cannot go stale, and it keeps `kb doctor` working at
    exactly the moment the server is the thing that is broken. Long-lived
    readers (`kb serve`'s sync, jarvis-mcp) leave it False and fail loudly instead.
    """
    global _store
    if _store is None:
        d = rag_dir or get_config().rag_dir
        d.mkdir(parents=True, exist_ok=True)
        # The index holds the full text of your notes and papers, so it is
        # owner-only. chmod every time rather than only on create: an index
        # that predates this is fixed on open.
        try:
            os.chmod(d, 0o700)
        except OSError as exc:
            # Someone else owning the directory shouldn't stop jarvis working,
            # but it does mean note text is readable by whoever can reach it —
            # worth a line rather than a silent pass.
            log.warning("could not restrict %s to owner-only access: %s", d, exc)
        cfg = get_config()
        shared: dict = {
            "collection_name": COLLECTION_NAME,
            "embedding_function": _get_embeddings(),
            "collection_metadata": {
                "embed_model": cfg.embed_model,
                "query_prefix": cfg.query_prefix,
            },
        }
        try:
            _store = Chroma(client=_server_client(cfg.server_port), **shared)
        except Exception as exc:
            if not (allow_direct or _allow_direct):
                raise RAGError(f"{_SERVER_DOWN_HELP}\n  ({exc})") from exc
            # Fall back to owning the files ourselves for the length of this
            # command. Safe only because the server is demonstrably not up:
            # two writers on one SQLite file is what the write lock guards.
            _store = Chroma(
                client_settings=_chroma_settings(), persist_directory=str(d), **shared
            )
        _check_embedding_model_matches(_store, cfg.embed_model)
    return _store


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
            f"Run 'uv run kb rebuild' to re-index your folders with the new model."
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
    from. Content with no headers passes through as a single section with an
    empty breadcrumb.

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
# is what lets a caller assemble a whole document — body and annotations — and
# commit it in a single atomic write. An ingest that fails part-way then leaves
# nothing behind at all, instead of a half-indexed document. See
# index_paper_file in jarvis/kb/mirror.py for the staged caller.


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
        # No reopen-and-retry here any more. It existed because a reader could
        # hold a cached copy of the vector index that another process had
        # rewritten, and the retry never actually ran (every caller passes
        # store=). The server owns the index now, so that state cannot arise.
        raise _diagnose_kb_error(exc, f"Search failed: {exc}") from exc

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


def metadata_value_counts(
    key: str,
    store: Chroma | None = None,
    visibility: str | None = None,
) -> dict[str, int]:
    """
    Distinct values of one metadata key with their chunk counts.

    Pass visibility="public" when the answer goes to a cloud model, since a
    private note's category or status says something about it on its own.
    """
    s = store or get_store()
    counts: dict[str, int] = {}
    # ChromaDB rejects an empty where dict, so only pass one when filtering.
    where_kwargs = {"where": {"visibility": {"$eq": visibility}}} if visibility else {}
    try:
        result = s._collection.get(include=["metadatas"], **where_kwargs)
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
    chunks (identified by the presence of annotation_kind metadata, also
    sorted by chunk_index) — this lets a caller page through the source text
    before hitting the highlights. Returns []
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


def list_documents(
    limit: int = 10_000,
    doc_type: str | list[str] = "paper",
    category: str | None = None,
    status: str | None = None,
    entity: str | None = None,
    store: Chroma | None = None,
    visibility: str | None = None,
) -> list[dict]:
    """
    De-duplicated list of indexed documents as metadata dicts, most recently
    added first, with a chunk_count per document.

    Papers are keyed by `source`; notes share `source="local"` and are keyed by
    `file_path` instead, so listing notes returns one entry per file rather
    than one entry for the entire vault.

    The default limit returns every document in a single-user KB in one call;
    callers wanting a short preview pass their own smaller limit.

    `visibility` narrows the listing in the same where-clause as the other
    filters. A cloud caller passes "public": a private note's title and path
    are private information too, not just its text.
    """
    s = store or get_store()
    conditions: list[dict] = []
    if isinstance(doc_type, str):
        conditions.append({"doc_type": {"$eq": doc_type}})
    else:
        conditions.append({"doc_type": {"$in": list(doc_type)}})
    for field, value in (
        ("category", category), ("status", status), ("entity", entity), ("visibility", visibility),
    ):
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
