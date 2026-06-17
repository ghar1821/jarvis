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
    doc_type    : str  — "paper" | "note" | "pdf"
    visibility  : str  — "public" | "private"
    source      : str  — arXiv/DOI URL for papers, "local" for notes/PDFs
    title       : str  — display title (optional)
    authors     : str  — comma-separated authors, papers only (optional)
    score       : int  — relevance score 0-10, papers only (optional)
    track       : str  — research track label, papers only (optional)
    file_path   : str  — vault-relative path, notes/PDFs only (optional)
    content_hash: str  — SHA-256 of full file, used for change detection

Privacy model
-------------
  "public"  — accessible to all providers (Ollama and Anthropic)
  "private" — accessible to local Ollama only

  search_with_privacy_check() enforces this at query time:
  - cloud provider  → searches public docs only; reports whether private
                       docs also matched so the user can be warned
  - local provider  → searches all docs without restriction
"""

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import get_config
from ..errors import RAGError

COLLECTION_NAME = "knowledge_base"

_embeddings: HuggingFaceEmbeddings | None = None
_store: Chroma | None = None


# ── Singletons ────────────────────────────────────────────────────────────────


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        print("Loading embedding model...", flush=True)
        _embeddings = HuggingFaceEmbeddings(model_name=get_config().embed_model)
    return _embeddings


def get_store(rag_dir: Path | None = None) -> Chroma:
    """Return the process-wide Chroma vector store singleton."""
    global _store
    if _store is None:
        d = rag_dir or get_config().rag_dir
        d.mkdir(parents=True, exist_ok=True)
        _store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=_get_embeddings(),
            persist_directory=str(d),
        )
    return _store


def _splitter() -> RecursiveCharacterTextSplitter:
    cfg = get_config()
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )


# ── Core write operations ─────────────────────────────────────────────────────


def add_texts(
    content: str,
    doc_type: str,
    visibility: str,
    source: str,
    extra_metadata: dict | None = None,
    store: Chroma | None = None,
) -> list[str]:
    """Chunk content and add all chunks to the knowledge base. Returns chunk IDs."""
    chunks = _splitter().split_text(content)
    if not chunks:
        return []
    metadata = {
        "date_added": datetime.now(timezone.utc).isoformat(),
        "doc_type": doc_type,
        "visibility": visibility,
        "source": source,
        **(extra_metadata or {}),
    }
    documents = [Document(page_content=chunk, metadata=metadata) for chunk in chunks]
    s = store or get_store()
    try:
        return s.add_documents(documents)
    except Exception as exc:
        raise RAGError(f"Failed to add documents: {exc}") from exc


def _source_exists(source: str, store: Chroma) -> bool:
    """Return True if any chunks with this source URL are already indexed."""
    if not source:
        return False
    try:
        result = store._collection.get(where={"source": {"$eq": source}}, include=[])
        return len(result["ids"]) > 0
    except Exception:
        return False


def add_paper(
    paper: dict,
    dense_summary: str,
    score: int = 0,
    track: str = "",
    store: Chroma | None = None,
    storage_mode: str = "summary",
) -> list[str]:
    """Add a paper to the knowledge base. Papers are always public. Skips if already indexed."""
    s = store or get_store()
    source = paper.get("link", "")
    if _source_exists(source, s):
        return []
    content = f"{paper.get('title', '')}\n{source}\n\n{dense_summary}"
    return add_texts(
        content=content,
        doc_type="paper",
        visibility="public",
        source=source,
        extra_metadata={
            "title": paper.get("title", ""),
            "authors": paper.get("authors", ""),
            "score": int(score),
            "track": str(track),
            "storage_mode": storage_mode,
        },
        store=s,
    )


def add_papers_batch(
    entries: list[tuple[dict, dict]],
    store: Chroma | None = None,
) -> int:
    """
    Batch-add papers from a digest scoring run.
    Reuses existing summary+why fields — no extra LLM call.
    Returns count of papers added.
    """
    s = store or get_store()
    count = 0
    for paper, selected in entries:
        summary = "\n\n".join(
            filter(None, [selected.get("summary", ""), selected.get("why", "")])
        )
        add_paper(
            paper=paper,
            dense_summary=summary,
            score=selected.get("score", 0),
            track=selected.get("track", ""),
            store=s,
        )
        count += 1
    return count


def delete_by_metadata(
    key: str,
    value: str,
    store: Chroma | None = None,
) -> int:
    """Delete all chunks matching a metadata key=value pair. Returns count deleted."""
    s = store or get_store()
    try:
        result = s._collection.get(where={key: {"$eq": value}})
        ids = result["ids"]
        if ids:
            s.delete(ids)
        return len(ids)
    except Exception as exc:
        raise RAGError(f"Delete failed: {exc}") from exc


# ── Search ────────────────────────────────────────────────────────────────────


def search(
    query: str,
    n_results: int = 5,
    visibility: str | None = None,
    doc_type: str | None = None,
    store: Chroma | None = None,
) -> list[Document]:
    """Semantic search with optional metadata filters."""
    s = store or get_store()
    conditions = []
    if visibility:
        conditions.append({"visibility": {"$eq": visibility}})
    if doc_type:
        conditions.append({"doc_type": {"$eq": doc_type}})

    filter_dict = None
    if len(conditions) == 1:
        filter_dict = conditions[0]
    elif len(conditions) > 1:
        filter_dict = {"$and": conditions}

    try:
        return s.similarity_search(query, k=n_results, filter=filter_dict)
    except Exception as exc:
        raise RAGError(f"Search failed: {exc}") from exc


def search_with_privacy_check(
    query: str,
    provider: str,
    n_results: int = 5,
    doc_type: str | None = None,
    store: Chroma | None = None,
) -> tuple[list[Document], bool]:
    """
    Search with provider-aware privacy handling.

    Returns (results, has_private_hits).

    For cloud providers (Anthropic):
      - Returns public docs only
      - has_private_hits=True if private docs also matched (so the caller
        can warn the user that results may be incomplete)

    For local providers (Ollama):
      - Returns all docs regardless of visibility
      - has_private_hits is always False
    """
    s = store or get_store()
    if provider == "anthropic":
        results = search(query, n_results=n_results, visibility="public",
                         doc_type=doc_type, store=s)
        try:
            private_check = search(query, n_results=1, visibility="private",
                                   doc_type=doc_type, store=s)
            has_private = len(private_check) > 0
        except RAGError:
            has_private = False
        return results, has_private
    else:
        return search(query, n_results=n_results, doc_type=doc_type, store=s), False


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
    except Exception:
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
        s._collection.update(ids=ids, metadatas=updated_metadatas)
    except Exception as exc:
        raise RAGError(f"Failed to update metadata: {exc}") from exc

    return len(ids)


def list_papers(
    limit: int = 50,
    store: Chroma | None = None,
) -> list[dict]:
    """Return de-duplicated list of indexed papers as metadata dicts."""
    s = store or get_store()
    try:
        result = s._collection.get(
            where={"doc_type": {"$eq": "paper"}},
            include=["metadatas"],
        )
    except Exception as exc:
        raise RAGError(f"List failed: {exc}") from exc

    chunk_counts: dict[str, int] = {}
    first_meta: dict[str, dict] = {}
    for meta in result["metadatas"]:
        src = meta.get("source", "")
        if not src:
            continue
        chunk_counts[src] = chunk_counts.get(src, 0) + 1
        if src not in first_meta:
            first_meta[src] = meta

    papers = []
    for src, meta in list(first_meta.items())[:limit]:
        papers.append({**meta, "chunk_count": chunk_counts[src]})
    return papers


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
    """Chunk and index a single vault .md file. Returns list of chunk IDs."""
    content = file_path.read_text(encoding="utf-8", errors="replace")
    rel_path = str(file_path.relative_to(vault_root))
    visibility = get_visibility(file_path, vault_root)
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    modified_at = datetime.fromtimestamp(
        file_path.stat().st_mtime, tz=timezone.utc
    ).isoformat()
    title_match = re.search(r"^#\s+(.+)", content, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else file_path.stem

    return add_texts(
        content=content,
        doc_type="note",
        visibility=visibility,
        source="local",
        extra_metadata={
            "file_path": rel_path,
            "title": title,
            "content_hash": content_hash,
            "modified_at": modified_at,
        },
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

    # Build map of currently indexed notes: file_path → content_hash
    try:
        result = s._collection.get(
            where={"doc_type": {"$eq": "note"}},
            include=["metadatas"],
        )
        indexed: dict[str, str] = {}
        for meta in result["metadatas"]:
            fp = meta.get("file_path", "")
            if fp and fp not in indexed:
                indexed[fp] = meta.get("content_hash", "")
    except Exception:
        indexed = {}

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
        stored_hash = indexed.get(rel_path)
        if stored_hash is None:
            index_vault_file(file_path, vault_root, s)
            added += 1
        elif stored_hash != file_hash:
            delete_by_metadata("file_path", rel_path, s)
            index_vault_file(file_path, vault_root, s)
            updated += 1

    for rel_path in indexed:
        if rel_path not in current:
            delete_by_metadata("file_path", rel_path, s)
            deleted += 1

    # ── Phase 2: PDF notes (absolute file paths, doc_type="note") ────────────
    # Local PDFs added as notes — not scanned from vault_root.
    # Always full_text. Missing file: warn, leave DB unchanged.
    # Changed file: delete old chunks, re-convert, re-index, then temp dir cleaned up.
    try:
        pdf_result = s._collection.get(
            where={"doc_type": {"$eq": "note"}},
            include=["metadatas"],
        )
        pdf_notes: dict[str, dict] = {}
        for meta in pdf_result["metadatas"]:
            fp = meta.get("file_path", "")
            if fp and fp.endswith(".pdf") and fp not in pdf_notes:
                pdf_notes[fp] = meta
    except Exception:
        pdf_notes = {}

    for abs_path_str, meta in pdf_notes.items():
        pdf_path = Path(abs_path_str)
        if not pdf_path.exists():
            print(f"  ⚠️  PDF note not found (keeping DB entry): {abs_path_str}", flush=True)
            continue
        current_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        if current_hash == meta.get("content_hash", ""):
            continue
        import tempfile
        from ..arxiv.convert import convert_pdf
        print(f"  PDF note changed, re-indexing: {pdf_path.name}", flush=True)
        delete_by_metadata("file_path", abs_path_str, s)
        with tempfile.TemporaryDirectory() as tmp:
            # temp dir (and converted .md) deleted automatically on exit
            tmp_path = Path(tmp)
            convert_pdf(pdf_path, tmp_path)
            md_path = tmp_path / f"{pdf_path.stem}.md"
            if md_path.exists():
                add_texts(
                    content=md_path.read_text(encoding="utf-8"),
                    doc_type="note",
                    visibility=meta.get("visibility", "public"),
                    source=pdf_path.as_uri(),
                    extra_metadata={
                        "title": meta.get("title", pdf_path.stem),
                        "file_path": abs_path_str,
                        "content_hash": current_hash,
                        "storage_mode": "full_text",
                    },
                    store=s,
                )
                updated += 1

    return added, updated, deleted
