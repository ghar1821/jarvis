"""
Local RAG database for arXiv papers and Obsidian vault notes.

Two ChromaDB collections:
  papers      — one document per paper (title + LLM-generated dense summary)
  vault_notes — Obsidian vault .md files chunked into overlapping windows

Storage: configured via get_config().rag_dir (default ~/.paper_digest/rag/)
"""

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from .config import get_config
from .errors import RAGError

# ── Collection names ───────────────────────────────────────────────────────────

PAPERS_COLLECTION = "papers"
VAULT_COLLECTION = "vault_notes"

# ── Shared embedding function (loaded once per process) ───────────────────────

_embed_fn: SentenceTransformerEmbeddingFunction | None = None


def _get_embed_fn() -> SentenceTransformerEmbeddingFunction:
    global _embed_fn
    if _embed_fn is None:
        print("Loading embedding model...", flush=True)
        _embed_fn = SentenceTransformerEmbeddingFunction(get_config().embed_model)
    return _embed_fn


# ── Papers collection ──────────────────────────────────────────────────────────


def get_papers_collection(rag_dir: Path | None = None) -> chromadb.Collection:
    d = (rag_dir or get_config().rag_dir)
    d.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(d))
    return client.get_or_create_collection(PAPERS_COLLECTION, embedding_function=_get_embed_fn())


def _stable_paper_id(paper: dict) -> str:
    """Derive a stable, version-independent doc ID from the paper's arXiv link."""
    from .convert import parse_arxiv_url

    arxiv_id = parse_arxiv_url(paper.get("link", ""))
    if arxiv_id:
        return re.sub(r"v\d+$", "", arxiv_id)
    return hashlib.sha256(
        paper.get("link", paper.get("title", "")).encode()
    ).hexdigest()[:16]


def add_paper(
    paper: dict,
    dense_summary: str,
    score: int = 0,
    track: str = "",
    collection: chromadb.Collection | None = None,
) -> str:
    """Add or update a paper in the RAG database. Returns the doc_id."""
    if collection is None:
        collection = get_papers_collection()
    doc_id = _stable_paper_id(paper)
    metadata: dict[str, str | int | float] = {
        "title": paper.get("title", ""),
        "authors": paper.get("authors", ""),
        "link": paper.get("link", ""),
        "published": paper.get("published", ""),
        "source": paper.get("source", ""),
        "score": int(score),
        "track": str(track),
        "added_at": datetime.now(timezone.utc).isoformat(),
    }
    collection.upsert(
        ids=[doc_id],
        documents=[f"{paper['title']}\n\n{dense_summary}"],
        metadatas=[metadata],
    )
    return doc_id


def add_papers_batch(
    entries: list[tuple[dict, dict]],
    collection: chromadb.Collection | None = None,
) -> list[str]:
    """
    Batch-add papers from a digest scoring run.

    entries: list of (paper_dict, selected_entry_dict) pairs.
    Digest papers already have summary+why from scoring — no extra LLM call needed.
    """
    if collection is None:
        collection = get_papers_collection()
    return [
        add_paper(
            paper=paper,
            dense_summary="\n\n".join(filter(None, [s.get("summary", ""), s.get("why", "")])),
            score=s.get("score", 0),
            track=s.get("track", ""),
            collection=collection,
        )
        for paper, s in entries
    ]


def generate_paper_summary(
    title: str,
    source: "str | Path",
    provider: "Any",  # ChatProvider — avoid circular import at module level
) -> str:
    """
    Generate a dense summary of a paper using the configured LLM provider.

    source: plain text (abstract) or a Path to a PDF file.
    """
    return provider.summarize(title, source)


def retrieve_papers(
    query: str,
    n_results: int = 5,
    score_min: int | None = None,
    track: str | None = None,
    collection: chromadb.Collection | None = None,
) -> list[dict]:
    """
    Semantic search over papers.

    Returns a list of result dicts including doc_id, document, distance, and metadata.
    Raises RAGError if the collection cannot be queried.
    """
    if collection is None:
        collection = get_papers_collection()

    total = collection.count()
    if total == 0:
        return []

    where: dict[str, Any] | None = None
    conditions: list[dict] = []
    if score_min is not None:
        conditions.append({"score": {"$gte": score_min}})
    if track is not None:
        conditions.append({"track": {"$eq": track}})
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    kwargs: dict[str, Any] = {
        "query_texts": [query],
        "n_results": min(n_results, total),
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        kwargs["where"] = where

    try:
        results = collection.query(**kwargs)
    except Exception as exc:
        raise RAGError(f"Papers query failed: {exc}") from exc

    return [
        {
            "doc_id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "distance": results["distances"][0][i],
            **meta,
        }
        for i, meta in enumerate(results["metadatas"][0])
    ]


def remove_paper(doc_id: str, collection: chromadb.Collection | None = None) -> None:
    if collection is None:
        collection = get_papers_collection()
    collection.delete(ids=[doc_id])


def count(collection: chromadb.Collection | None = None) -> int:
    if collection is None:
        collection = get_papers_collection()
    return collection.count()


def list_papers(
    limit: int = 50,
    offset: int = 0,
    collection: chromadb.Collection | None = None,
) -> list[dict]:
    if collection is None:
        collection = get_papers_collection()
    result = collection.get(limit=limit, offset=offset, include=["metadatas"])
    return [{"doc_id": result["ids"][i], **meta} for i, meta in enumerate(result["metadatas"])]


# ── Vault collection ───────────────────────────────────────────────────────────


def get_vault_collection(rag_dir: Path | None = None) -> chromadb.Collection:
    d = (rag_dir or get_config().rag_dir)
    d.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(d))
    return client.get_or_create_collection(VAULT_COLLECTION, embedding_function=_get_embed_fn())


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks


def index_vault_file(
    file_path: Path,
    vault_root: Path,
    collection: chromadb.Collection,
) -> list[str]:
    """Chunk and upsert a single vault .md file. Returns list of chunk doc_ids."""
    cfg = get_config()
    content = file_path.read_text(encoding="utf-8", errors="replace")
    rel_path = str(file_path.relative_to(vault_root))

    title_match = re.search(r"^#\s+(.+)", content, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else file_path.stem

    content_hash = hashlib.sha256(content.encode()).hexdigest()
    modified_at = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc).isoformat()
    chunks = _chunk_text(content, cfg.chunk_size, cfg.chunk_overlap)
    rel_hash = hashlib.sha256(rel_path.encode()).hexdigest()[:12]

    ids = [f"{rel_hash}_{i}" for i in range(len(chunks))]
    metadatas: list[dict[str, str | int | float]] = [
        {
            "file_path": rel_path,
            "title": title,
            "modified_at": modified_at,
            "content_hash": content_hash,
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]
    collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
    return ids


def _delete_vault_file_chunks(rel_path: str, collection: chromadb.Collection) -> int:
    result = collection.get(where={"file_path": {"$eq": rel_path}}, include=[])
    ids_to_delete = result["ids"]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
    return len(ids_to_delete)


def refresh_vault(
    vault_root: Path,
    collection: chromadb.Collection | None = None,
) -> tuple[int, int, int]:
    """
    Incrementally sync the vault index with the filesystem.
    Indexes new/changed files, removes chunks for deleted files.
    Returns (added, updated, deleted) file counts.
    Safe to call on an empty (not-yet-indexed) collection.
    """
    if collection is None:
        collection = get_vault_collection()

    current: dict[str, tuple[Path, str]] = {}
    for md_file in vault_root.rglob("*.md"):
        try:
            text = md_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = str(md_file.relative_to(vault_root))
        current[rel] = (md_file, hashlib.sha256(text.encode()).hexdigest())

    all_indexed = collection.get(include=["metadatas"])
    indexed_hashes: dict[str, str] = {}
    for meta in all_indexed["metadatas"]:
        fp = meta.get("file_path", "")
        if fp and fp not in indexed_hashes:
            indexed_hashes[fp] = meta.get("content_hash", "")

    added = updated = deleted = 0

    for rel_path, (file_path, file_hash) in current.items():
        stored = indexed_hashes.get(rel_path)
        if stored is None:
            index_vault_file(file_path, vault_root, collection)
            added += 1
        elif stored != file_hash:
            _delete_vault_file_chunks(rel_path, collection)
            index_vault_file(file_path, vault_root, collection)
            updated += 1

    for rel_path in indexed_hashes:
        if rel_path not in current:
            _delete_vault_file_chunks(rel_path, collection)
            deleted += 1

    return added, updated, deleted


def search_vault(
    query: str,
    n_results: int = 5,
    collection: chromadb.Collection | None = None,
) -> list[dict]:
    """
    Semantic search over vault notes.

    Returns list of matching chunks with file_path, title, chunk text, distance.
    Raises RAGError if the collection cannot be queried.
    """
    if collection is None:
        collection = get_vault_collection()

    total = collection.count()
    if total == 0:
        return []

    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, total),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        raise RAGError(f"Vault query failed: {exc}") from exc

    return [
        {
            "file_path": meta.get("file_path", ""),
            "title": meta.get("title", ""),
            "chunk": results["documents"][0][i],
            "distance": results["distances"][0][i],
        }
        for i, meta in enumerate(results["metadatas"][0])
    ]


def count_vault(collection: chromadb.Collection | None = None) -> int:
    if collection is None:
        collection = get_vault_collection()
    return collection.count()
