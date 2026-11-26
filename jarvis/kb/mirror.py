"""
The index as a mirror of two folders you manage yourself.

Everything in the knowledge base comes from a file: vault notes (`.md`
outside the private folders) and the PDFs under `papers_dir`. Adding,
removing, moving or renaming a file is how you change the index — jarvis has
no add or remove command, and never writes to either folder. Each sync pass
brings the index back in line with the files:

  new file         → indexed
  changed file     → re-indexed (its old chunks replaced in one commit)
  deleted file     → its chunks removed
  moved PDF        → recognised by content hash and re-pointed, no re-embed

Because every entry is derived from a file, `rebuild_index()` can always
recreate the whole index from disk.

Two safety rules hold throughout:

- Private vault folders are never read. Every MCP client sends what it
  retrieves to a cloud model, so a private note is kept out of the index
  entirely rather than filtered at query time.
- A folder that is missing (an unmounted drive, a moved vault) never empties
  the index. Its sync is skipped with a warning, deletions included.
"""

import hashlib
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_chroma import Chroma

from jarvis.core.config import Config, get_config
from jarvis.core.errors import ConversionError
from jarvis.core.logs import get_logger

from .store import (
    COLLECTION_NAME,
    _diagnose_kb_error,
    build_annotation_documents,
    build_text_documents,
    commit_documents,
    delete_by_metadata,
    forget_store,
    get_store,
    open_collection,
    open_index_client,
    update_file_path,
    update_metadata_fields,
)

log = get_logger(__name__)


# ── What counts as a source file ──────────────────────────────────────────────


def get_visibility(file_path: Path, vault_root: Path) -> str:
    """
    "private" when the file sits under a top-level private vault folder.

    Top-level only, by contract: `research/private/` is not private. The
    caller should pass a resolved path, so a symlink in a public folder that
    points into `private/` is classified by where it really leads.
    """
    try:
        parts = file_path.relative_to(vault_root).parts
        if parts and parts[0] in get_config().private_vault_dirs:
            return "private"
    except ValueError:
        pass
    return "public"


def _is_hidden(path: Path, root: Path) -> bool:
    """
    True for anything under a dot-folder or named like a temp/lock file.

    Skips `.obsidian/` and `.trash/` (Obsidian's own bin of deleted notes,
    which would otherwise keep "deleted" notes searchable), editor lock files,
    and partially-written cloud downloads.
    """
    relative_parts = path.relative_to(root).parts
    name = path.name
    return (
        any(part.startswith(".") for part in relative_parts)
        or name.startswith("~$")
        or name.endswith(".icloud")
    )


def _modified_at(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wait_for_stable(
    path: Path,
    checks: int = 2,
    interval: float = 1.0,
    timeout: float = 30.0,
) -> bool:
    """
    Poll size+mtime until the file has been unchanged for `checks` consecutive
    polls — cloud-sync clients and slow copies write PDFs incrementally.
    Returns False on timeout or if the file disappears, and the caller leaves
    the file for the next pass rather than indexing half a PDF.
    """
    deadline = time.monotonic() + timeout
    stable_polls = 0
    last_signature = None
    while time.monotonic() < deadline:
        try:
            stat = path.stat()
        except OSError:
            return False
        signature = (stat.st_size, stat.st_mtime)
        if signature == last_signature and stat.st_size > 0:
            stable_polls += 1
            if stable_polls >= checks:
                return True
        else:
            stable_polls = 0
        last_signature = signature
        time.sleep(interval)
    return False


# ── Vault notes ───────────────────────────────────────────────────────────────


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
    # Hashed over the WHOLE file, frontmatter included — editing only the
    # status of a record must still count as a change.
    content_hash = hashlib.sha256(content.encode()).hexdigest()

    record_metadata, body = parse_frontmatter(content, file_label=rel_path)
    title_match = re.search(r"^#\s+(.+)", body, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else file_path.stem

    documents = build_text_documents(
        content=body,
        doc_type="note",
        # Only public notes are ever indexed; the field stays so the search
        # backstop (visibility="public") has something to match.
        visibility="public",
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
            "modified_at": _modified_at(file_path),
            # Stamped on every note, frontmatter or not, so refresh_vault can
            # tell "indexed under the current schema" from "needs a backfill".
            "meta_schema": META_SCHEMA,
        },
        embed_header=record_header(record_metadata),
    )
    return commit_documents(documents, store) if documents else []


def refresh_vault(
    vault_root: Path,
    store: Chroma | None = None,
) -> tuple[int, int, int]:
    """
    Bring the indexed notes in line with the vault's public .md files.

    Returns (added, updated, deleted) file counts. A note that moves into a
    private folder simply stops being a source, so it is deleted from the
    index on the next pass — no separate reclassification step exists.
    """
    s = store or get_store()

    if not vault_root.is_dir():
        log.warning("vault sync skipped: %s is not a folder (nothing deleted)", vault_root)
        return 0, 0, 0

    from .frontmatter import META_SCHEMA, split_frontmatter

    # file_path → (content_hash, meta_schema) for every note already indexed.
    try:
        result = s._collection.get(where={"doc_type": {"$eq": "note"}}, include=["metadatas"])
    except Exception as exc:
        # Falling back to an empty map would look like "nothing is indexed"
        # and re-index the whole vault. A read failure is a real fault.
        log.error("refresh_vault: could not read the indexed note map: %s", exc)
        raise _diagnose_kb_error(exc, f"Failed to read the vault index: {exc}") from exc
    indexed: dict[str, tuple[str, int]] = {}
    for meta in result["metadatas"]:
        rel_path = meta.get("file_path", "")
        if rel_path and rel_path not in indexed:
            indexed[rel_path] = (meta.get("content_hash", ""), int(meta.get("meta_schema", 0)))

    # The public .md files on disk right now.
    vault_root = vault_root.resolve()
    current: dict[str, tuple[Path, str]] = {}
    for md_file in vault_root.rglob("*.md"):
        if _is_hidden(md_file, vault_root):
            continue
        if get_visibility(md_file.resolve(), vault_root) == "private":
            continue
        try:
            text = md_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        current[str(md_file.relative_to(vault_root))] = (
            md_file, hashlib.sha256(text.encode()).hexdigest()
        )

    added = updated = deleted = 0
    for rel_path, (file_path, file_hash) in current.items():
        stored = indexed.get(rel_path)
        if stored is None:
            index_vault_file(file_path, vault_root, s)
            added += 1
            continue
        stored_hash, stored_schema = stored
        if stored_hash != file_hash:
            # The file itself changed, so its text must be re-chunked and
            # re-embedded.
            delete_by_metadata("file_path", rel_path, s)
            index_vault_file(file_path, vault_root, s)
            updated += 1
        elif stored_schema < META_SCHEMA:
            # The file is untouched; only jarvis's reading of it moved on. A
            # note with a frontmatter block gets a different body and record
            # header under the new schema, so it is re-indexed. A plain note
            # would embed identically, so just stamp the marker — a schema
            # migration should not re-embed a whole vault for nothing.
            raw, _ = split_frontmatter(file_path.read_text(encoding="utf-8", errors="replace"))
            if raw.strip():
                delete_by_metadata("file_path", rel_path, s)
                index_vault_file(file_path, vault_root, s)
            else:
                update_metadata_fields(rel_path, {"meta_schema": META_SCHEMA}, s)
            updated += 1

    for rel_path in indexed:
        if rel_path not in current:
            delete_by_metadata("file_path", rel_path, s)
            deleted += 1

    return added, updated, deleted


# ── Papers ────────────────────────────────────────────────────────────────────


def index_paper_file(pdf_path: Path, store: Chroma | None = None) -> int:
    """
    Index one PDF from the papers folder as a full-text paper, replacing any
    existing entry for the same file in the same commit. Returns chunk count.

    Title, authors and DOI come from jarvis/kb/metadata.py (DOI regex,
    Crossref, embedded metadata, filename) — no LLM. Highlights and typed
    notes are indexed alongside the body, and are kept even when the body
    cannot be converted (a scanned PDF with no text layer).
    """
    from .convert import pdf_to_markdown
    from .metadata import resolve_pdf_metadata

    # absolute(), not resolve(): the entry is keyed on where the file sits in
    # papers_dir. Resolving a symlink would record a path outside the folder,
    # and the next pass would treat the paper as missing and re-index it.
    resolved = pdf_path.absolute()
    source = resolved.as_uri()
    meta = resolve_pdf_metadata(resolved)
    title, authors, doi = meta["title"], meta["authors"], meta["doi"]
    # Carried on every chunk — body and annotations — so the sync can read
    # them back from whichever chunk it sees first.
    file_metadata = {
        "title": title,
        "authors": authors,
        "doi": doi,
        "file_path": str(resolved),
        "content_hash": _file_hash(resolved),
        "modified_at": _modified_at(resolved),
    }

    documents = build_annotation_documents(
        resolved, doc_type="paper", visibility="public", source=source,
        title=title, file_path=str(resolved),
    )
    for document in documents:
        document.metadata.update(file_metadata)

    try:
        body = pdf_to_markdown(resolved)
    except ConversionError as exc:
        log.warning("papers: %s", exc)
        body = ""
    if body:
        documents += build_text_documents(
            content=body,
            doc_type="paper",
            visibility="public",
            source=source,
            extra_metadata=file_metadata,
            # Title and authors on every chunk, so an author-name or title
            # query can match any part of a long paper.
            embed_header=f"{title} — {authors}" if authors else title,
        )

    if not documents:
        log.warning("papers: nothing indexable in %s (no text, no annotations)", resolved.name)
    commit_documents(documents, store, replace_source=source)
    log.info('papers: indexed "%s" — authors: %s; doi: %s; file: %s',
             title, authors or "unknown", doi or "unknown", resolved)
    return len(documents)


def refresh_papers(papers_dir: Path | None, store: Chroma | None = None) -> dict[str, int]:
    """
    Bring the indexed papers in line with the PDFs under papers_dir.

    Returns counts: {"added", "updated", "moved", "deleted"}. Only entries
    whose file lives under papers_dir take part — anything indexed from
    elsewhere by an older version is left alone (`kb doctor` reports those).
    """
    counts = {"added": 0, "updated": 0, "moved": 0, "deleted": 0}
    if papers_dir is None:
        return counts
    if not papers_dir.is_dir():
        log.warning("papers sync skipped: %s is not a folder (nothing deleted)", papers_dir)
        return counts
    papers_dir = papers_dir.resolve()
    s = store or get_store()

    # source → {file_path, content_hash, modified_at} for papers under papers_dir.
    try:
        result = s._collection.get(where={"doc_type": {"$eq": "paper"}}, include=["metadatas"])
    except Exception as exc:
        raise _diagnose_kb_error(exc, f"Failed to read the papers index: {exc}") from exc
    indexed: dict[str, dict] = {}
    for meta in result["metadatas"]:
        file_path = meta.get("file_path", "")
        if not file_path or not Path(file_path).is_relative_to(papers_dir):
            continue
        entry = indexed.setdefault(meta["source"], {"file_path": file_path})
        for key in ("content_hash", "modified_at"):
            if meta.get(key) and not entry.get(key):
                entry[key] = meta[key]
    indexed_by_path = {entry["file_path"]: source for source, entry in indexed.items()}

    # The PDFs on disk right now. An iCloud-evicted file shows up only as a
    # ".name.pdf.icloud" placeholder; it still exists as far as the index is
    # concerned, so its entry is kept rather than deleted and re-added.
    current: list[Path] = []
    evicted: set[str] = set()
    for path in sorted(papers_dir.rglob("*")):
        if path.name.startswith(".") and path.name.endswith(".pdf.icloud"):
            evicted.add(str(path.with_name(path.name[1:-len(".icloud")])))
            continue
        if path.suffix.lower() == ".pdf" and path.is_file() and not _is_hidden(path, papers_dir):
            current.append(path)
    current_paths = {str(path) for path in current}

    # Entries whose file has gone: either moved (matched by hash below) or deleted.
    vanished = {
        source: entry for source, entry in indexed.items()
        if entry["file_path"] not in current_paths and entry["file_path"] not in evicted
    }
    vanished_by_hash = {entry.get("content_hash"): source for source, entry in vanished.items()}

    for path in current:
        source = indexed_by_path.get(str(path))
        if source is not None:
            # Already indexed at this path. An unchanged mtime means unchanged
            # bytes, which skips hashing hundreds of PDFs every pass.
            entry = indexed[source]
            if entry.get("modified_at") == _modified_at(path):
                continue
            if not wait_for_stable(path):
                log.warning("papers: %s still changing — leaving it for the next pass", path.name)
                continue
            if _file_hash(path) == entry.get("content_hash"):
                # Touched but not changed (copied, re-synced): refresh the mtime only.
                update_metadata_fields(str(path), {"modified_at": _modified_at(path)}, s)
                continue
            index_paper_file(path, s)
            counts["updated"] += 1
            continue

        if not wait_for_stable(path):
            log.warning("papers: %s still changing — leaving it for the next pass", path.name)
            continue
        old_source = vanished_by_hash.pop(_file_hash(path), None)
        if old_source is not None:
            # Same bytes, new place: re-point the entry instead of re-embedding.
            update_file_path(old_source, str(path), s)
            update_metadata_fields(str(path), {"modified_at": _modified_at(path)}, s)
            del vanished[old_source]
            log.info("papers: moved %s → %s", Path(indexed[old_source]["file_path"]).name, path)
            counts["moved"] += 1
            continue
        try:
            index_paper_file(path, s)
        except Exception:
            # One unreadable PDF must not stop the rest of the pass.
            log.exception("papers: failed to index %s", path)
            continue
        counts["added"] += 1

    for source, entry in vanished.items():
        delete_by_metadata("source", source, s)
        log.info("papers: removed %s (file deleted)", entry["file_path"])
        counts["deleted"] += 1

    return counts


# ── Whole-index operations ────────────────────────────────────────────────────


def sync_all(cfg: Config | None = None, store: Chroma | None = None) -> dict[str, int]:
    """One mirror pass over both folders. Returns combined counts for display."""
    cfg = cfg or get_config()
    notes_added, notes_updated, notes_deleted = refresh_vault(cfg.vault_path, store)
    papers = refresh_papers(cfg.papers_dir, store)
    return {
        "notes_added": notes_added,
        "notes_updated": notes_updated,
        "notes_deleted": notes_deleted,
        **{f"papers_{key}": value for key, value in papers.items()},
    }


def rebuild_index(cfg: Config | None = None) -> dict[str, int]:
    """
    Recreate the whole index from the two folders and swap it in.

    The new collection is built beside the live one and only replaces it once
    complete, so an interrupted rebuild leaves the old index exactly as it
    was. It never reads the old collection, which is why this is also the fix
    for an index too damaged to open. Anything not backed by a file (entries
    from older versions: arXiv URLs, digests, chat history) is gone afterwards.
    """
    cfg = cfg or get_config()
    client, _via_server = open_index_client()
    rebuild_name = f"{COLLECTION_NAME}_rebuild"

    # Start clean in case a previous rebuild was interrupted.
    try:
        client.delete_collection(rebuild_name)
    except Exception:
        pass
    new_store = open_collection(client, rebuild_name)
    counts = sync_all(cfg, new_store)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # no live collection yet — a first build
    new_store._collection.modify(name=COLLECTION_NAME)
    forget_store()
    return counts
