"""
Unified knowledge base agent — query and manage via natural language.

Handles both retrieval (find papers, search notes, read files) and
management (add papers, remove documents, list contents, refresh vault).
The LLM plans and executes tool calls; each call is shown in the terminal
so the user can see every step.

Provider (set via CHAT_PROVIDER env var or config):
  ollama     — local Ollama, full access (public + private documents)
  anthropic  — Anthropic Claude, public documents only; raises PrivacyError on any
               private content hit, which terminates the tool loop immediately
               (prompt-injection defence — private content never reaches the model)

Auth for Anthropic:
  Option 1: export ANTHROPIC_API_KEY=sk-ant-...
  Option 2: add api_key to [auth] in ~/.seshat/config.toml
"""

import sys
from pathlib import Path

from digest.config import get_config
from digest.errors import LLMError, PrivacyError
from digest.llm import make_provider

# ── Tool definitions ───────────────────────────────────────────────────────────

TOOLS = [
    # ── Query tools ──────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "retrieve_papers",
            "description": (
                "Search the knowledge base for research papers. "
                "Use for questions about papers and scientific topics. "
                "Always search before answering."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "n_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": (
                "Semantically search vault notes and local documents. "
                "Use to discover relevant files before reading them with read_file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "n_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the complete, ordered content of one vault file. "
                "Use this when search_notes has identified a specific file and you need the "
                "whole document — not just the matching chunks — to give a coherent answer. "
                "Do not use for discovery; use search_notes for that."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path within the vault"},
                },
                "required": ["path"],
            },
        },
    },
    # ── Management tools ──────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "add_document",
            "description": (
                "Add a paper or document to the knowledge base. "
                "Source can be an arXiv URL or an absolute path to a local PDF file.\n"
                "For arXiv URLs: always stored as 'paper'. Ask the user whether they want "
                "summary (default, fast) or full_text (paragraph-level retrieval) mode.\n"
                "For local PDFs: ALWAYS ask the user whether it is a 'paper' or a 'note' before calling.\n"
                "  doc_type='paper': research paper — supports summary or full_text mode.\n"
                "  doc_type='note': personal note — always indexed as full text; "
                "content hash tracked so refresh_vault detects changes automatically.\n"
                "Also ask for visibility (public/private) for local PDFs. "
                "Narrate each step as you go."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "arXiv URL (https://arxiv.org/abs/...) or absolute path to a local PDF file",
                    },
                    "doc_type": {
                        "type": "string",
                        "enum": ["paper", "note"],
                        "description": "For local PDFs only: 'paper' (research paper) or 'note' (personal note, always full text with hash tracking). arXiv URLs are always 'paper'.",
                        "default": "paper",
                    },
                    "score": {"type": "integer", "description": "Relevance score 0-10", "default": 0},
                    "track": {"type": "string", "description": "Research track label", "default": ""},
                    "mode": {
                        "type": "string",
                        "enum": ["summary", "full_text"],
                        "description": "For papers only: summary (LLM-generated) or full_text (full PDF chunked). Notes are always full_text.",
                        "default": "summary",
                    },
                    "visibility": {
                        "type": "string",
                        "enum": ["public", "private"],
                        "description": "Visibility for local PDFs. arXiv papers are always public.",
                        "default": "public",
                    },
                    "title": {
                        "type": "string",
                        "description": "Override title (for local PDFs without a clear title)",
                        "default": "",
                    },
                },
                "required": ["source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_document",
            "description": (
                "Remove a document from the knowledge base. Two-step process: "
                "call WITHOUT confirmed first — it shows exactly what will be removed and asks for user confirmation. "
                "Only call with confirmed=true after the user has explicitly approved. "
                "Never pass confirmed=true on the first call. "
                "Set delete_file=true if the user wants the actual file deleted too (vault notes and local PDFs only — arXiv papers have no local file)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Source URL of the document"},
                    "confirmed": {
                        "type": "boolean",
                        "description": "Set to true only after the user has confirmed removal",
                        "default": False,
                    },
                    "delete_file": {
                        "type": "boolean",
                        "description": "Also delete the local file (vault notes and local PDFs only)",
                        "default": False,
                    },
                },
                "required": ["source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_papers",
            "description": "List papers currently indexed in the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max papers to show (default 10)", "default": 10},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kb_stats",
            "description": "Show counts of papers, notes, and total chunks in the knowledge base.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_file_path",
            "description": (
                "Update the stored file path for a local document (PDF or vault note) "
                "when the file has been moved or renamed. Updates both the file_path "
                "metadata and the source URI for all chunks of that document."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Current source URL of the document (file:/// URI). Use list_papers or search to find it.",
                    },
                    "new_path": {
                        "type": "string",
                        "description": "New filesystem path to the file (absolute or ~ expanded).",
                    },
                },
                "required": ["source", "new_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "index_vault",
            "description": (
                "Build or rebuild the vault index from scratch. "
                "Use this for the initial setup or when you want a clean re-index. "
                "Set force=true to clear the existing index first; omit it for a safe incremental run."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "force": {
                        "type": "boolean",
                        "description": "Clear the existing vault index before re-indexing",
                        "default": False,
                    }
                },
                "required": [],
            },
        },
    },
]

# The use_own_knowledge tool is only included in the tools list when the
# kb_only toggle is OFF. It acts as an explicit signal — the LLM must call it
# before drawing on its training knowledge, giving the UI something concrete
# to display to the user.
USE_OWN_KNOWLEDGE_TOOL = {
    "type": "function",
    "function": {
        "name": "use_own_knowledge",
        "description": (
            "Call this before answering from your training knowledge, when all "
            "knowledge base searches returned no relevant results. This signals "
            "to the user that the answer comes from your training data, not their documents."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

# ── System prompt ──────────────────────────────────────────────────────────────

_DEFAULT_SYSTEM = """\
You are a knowledgeable assistant that can both query and manage a local \
knowledge base of research papers and Obsidian vault notes.

Querying workflow:
1. Search first — use search_notes and/or retrieve_papers before reading anything.
2. Read for detail — use read_file only after search has identified a relevant file.
3. Never call read_file speculatively.

Management:
- To add a paper or PDF: call add_document with an arXiv URL or local file path. \
Ask the user whether they want summary or full_text mode if not specified. Narrate each step.
- To remove a document: call remove_document without confirmed first to preview, \
then confirm with the user before calling with confirmed=true.
- To inspect the knowledge base: use list_papers or kb_stats.
- To index or update the vault: call index_vault (incremental by default; force=true for a clean rebuild).
- To update the path of a moved or renamed local file: call update_file_path with the old source URL and the new path. Use list_papers or search_notes to find the source URL first.

Always include the source URL when discussing a paper.\
"""

# Appended to the base prompt depending on the knowledge source mode.
_KB_ONLY_ADDENDUM = (
    "\nKnowledge source restriction: You MUST answer ONLY from information "
    "retrieved using the tools above. Do NOT draw on your training knowledge "
    "to fill gaps or speculate. If the tools return no relevant results, say "
    "so clearly and stop."
)

_OWN_KNOWLEDGE_ADDENDUM = (
    "\nKnowledge source preference: Always search the knowledge base first. "
    "If all searches return no relevant results, you may draw on your training "
    "knowledge — but you MUST call use_own_knowledge() first to inform the user "
    "before doing so."
)


def build_system_prompt(kb_only: bool = True) -> str:
    """
    Build the agent system prompt.

    Override the base prompt by creating ~/.seshat/system_prompt.md.
    Falls back to the built-in default.

    kb_only=True  (default): LLM may only answer from KB tool results.
    kb_only=False: LLM searches KB first, falls back to training knowledge.
    """
    from pathlib import Path as _Path
    override = _Path.home() / ".seshat" / "system_prompt.md"
    base = override.read_text(encoding="utf-8").rstrip() if override.exists() else _DEFAULT_SYSTEM
    return base + (_KB_ONLY_ADDENDUM if kb_only else _OWN_KNOWLEDGE_ADDENDUM)


# ── Vault helpers ──────────────────────────────────────────────────────────────


def read_file(vault: Path, rel_path: str, provider_str: str = "ollama") -> str:
    target = (vault / rel_path).resolve()
    try:
        target.relative_to(vault.resolve())
    except ValueError:
        return f"[Error: '{rel_path}' is outside the vault]"
    if not target.exists() or not target.is_file():
        return f"[Error: file not found: '{rel_path}']"
    if provider_str == "anthropic":
        cfg = get_config()
        if Path(rel_path).parts and Path(rel_path).parts[0] in cfg.private_vault_dirs:
            # Hard stop — do not return the path or any hint about content;
            # private notes may contain adversarial text designed to manipulate the model.
            raise PrivacyError(
                f"'{rel_path}' is in a private vault directory and cannot be read by a "
                "cloud provider. Switch to Ollama to access private notes."
            )
    return target.read_text(encoding="utf-8")


# ── Tool implementations ───────────────────────────────────────────────────────


def _retrieve_papers(args: dict, provider_str: str) -> str:
    try:
        from digest.kb.store import get_store, search_with_privacy_check

        results, has_private = search_with_privacy_check(
            query=args["query"],
            provider=provider_str,
            n_results=min(int(args.get("n_results", 5)), 20),
            doc_type="paper",
            store=get_store(),
        )
    except Exception as exc:
        return f"[retrieve_papers error: {exc}]"

    # Query matched private content only — hard stop to prevent further probing.
    if has_private and not results:
        raise PrivacyError(
            "This query matched papers that are private and cannot be accessed by a "
            "cloud provider. Switch to Ollama to access private documents."
        )
    if not results:
        return "[No papers found.]"
    lines = [f"Found {len(results)} paper(s):\n"]
    for i, doc in enumerate(results, 1):
        m = doc.metadata
        lines.append(
            f"{i}. [{m.get('score', '?')}/10 · {m.get('track', '')}] {m.get('title', 'untitled')}\n"
            f"   {m.get('source', '')}\n"
            f"   {doc.page_content[:300].replace(chr(10), ' ')}...\n"
        )
    return "\n".join(lines)


def _search_notes(args: dict, provider_str: str) -> str:
    try:
        from digest.kb.store import get_store, search_with_privacy_check

        results, has_private = search_with_privacy_check(
            query=args["query"],
            provider=provider_str,
            n_results=min(int(args.get("n_results", 5)), 20),
            doc_type="note",
            store=get_store(),
        )
    except Exception as exc:
        return f"[search_notes error: {exc}]"

    # Query matched private notes only — hard stop to prevent further probing.
    if has_private and not results:
        raise PrivacyError(
            "This query matched notes that are private and cannot be accessed by a "
            "cloud provider. Switch to Ollama to access private notes."
        )
    if not results:
        return "[No notes found. Run 'kb index-vault' if vault is not yet indexed.]"
    lines = [f"Found {len(results)} note chunk(s):\n"]
    for i, doc in enumerate(results, 1):
        m = doc.metadata
        lines.append(
            f"{i}. {m.get('title', 'untitled')}  ({m.get('file_path', 'unknown')})\n"
            f"   {doc.page_content[:300].replace(chr(10), ' ')}...\n"
        )
    return "\n".join(lines)


def _add_document(args: dict, provider_obj) -> str:
    """
    Add a paper or local PDF document to the knowledge base.

    source: arXiv URL  → fetch metadata from API, then summary or full-text
    source: local path → read PDF directly, then summary (LLM reads PDF) or full-text (marker-pdf)

    mode="summary"   (default): LLM generates dense summary → chunk
    mode="full_text": convert PDF to Markdown → chunk full text
    """
    try:
        from pathlib import Path as _Path
        from digest.kb.store import add_paper, add_texts, get_store, _source_exists

        source = args.get("source", "")
        score = int(args.get("score", 0))
        track = str(args.get("track", ""))
        mode = args.get("mode", "summary")
        visibility = args.get("visibility", "public")
        title_override = args.get("title", "")
        store = get_store()

        # ── arXiv URL ─────────────────────────────────────────────────────────
        if source.startswith("http://") or source.startswith("https://"):
            from digest.arxiv.convert import parse_arxiv_url, download_arxiv_pdf, convert_pdf
            from digest.arxiv.fetch import fetch_arxiv_paper

            arxiv_id = parse_arxiv_url(source)
            if not arxiv_id:
                return f"[Error: could not parse arXiv ID from: {source}]"

            print(f"  Fetching metadata for arXiv:{arxiv_id}...", flush=True)
            paper = fetch_arxiv_paper(arxiv_id)
            print(f"  Title: {paper['title']}", flush=True)

            if _source_exists(paper.get("link", ""), store):
                return f"Already in knowledge base: \"{paper['title']}\""

            if mode == "full_text":
                import tempfile
                print("  Downloading PDF...", flush=True)
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_path = _Path(tmp)
                    pdf_path = download_arxiv_pdf(arxiv_id, tmp_path)
                    print("  Converting to Markdown (this may take a moment)...", flush=True)
                    convert_pdf(pdf_path, tmp_path)
                    md_path = tmp_path / f"{pdf_path.stem}.md"
                    if not md_path.exists():
                        return "[Error: PDF conversion produced no output]"
                    content = md_path.read_text(encoding="utf-8")
                print("  Chunking and indexing full text...", flush=True)
                ids = add_texts(
                    content=content, doc_type="paper", visibility="public",
                    source=paper["link"],
                    extra_metadata={"title": paper.get("title", ""),
                                    "authors": paper.get("authors", ""),
                                    "score": score, "track": track},
                    store=store,
                )
            else:
                print("  Generating summary...", flush=True)
                summary = provider_obj.summarize(paper["title"], paper["abstract"])
                ids = add_paper(paper=paper, dense_summary=summary,
                                score=score, track=track, store=store)

            return (
                f"Added \"{paper['title']}\" ({mode}, {len(ids)} chunk(s)).\n"
                f"  Source: {paper['link']}  ·  Score: {score}/10  ·  Track: {track or '(none)'}"
            )

        # ── Local PDF ─────────────────────────────────────────────────────────
        pdf_path = _Path(source).expanduser().resolve()
        if not pdf_path.exists():
            return f"[Error: file not found: {source}]"
        if pdf_path.suffix.lower() != ".pdf":
            return f"[Error: only PDF files are supported for local paths: {source}]"

        title = title_override or pdf_path.stem
        file_source = pdf_path.as_uri()
        doc_type = args.get("doc_type", "paper")

        if _source_exists(file_source, store):
            return f"Already in knowledge base: \"{title}\""

        if doc_type == "note":
            # Notes are always full text with content_hash for change tracking
            from digest.arxiv.convert import convert_pdf
            import hashlib as _hashlib
            import tempfile
            print(f"  Converting PDF note {pdf_path.name} to Markdown...", flush=True)
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = _Path(tmp)
                convert_pdf(pdf_path, tmp_path)
                md_path = tmp_path / f"{pdf_path.stem}.md"
                if not md_path.exists():
                    return "[Error: PDF conversion produced no output]"
                content = md_path.read_text(encoding="utf-8")
            content_hash = _hashlib.sha256(pdf_path.read_bytes()).hexdigest()
            print("  Chunking and indexing...", flush=True)
            ids = add_texts(
                content=content, doc_type="note", visibility=visibility,
                source=file_source,
                extra_metadata={
                    "title": title, "file_path": str(pdf_path),
                    "content_hash": content_hash, "storage_mode": "full_text",
                },
                store=store,
            )
            return (
                f"Added note \"{title}\" (full text, {visibility}, {len(ids)} chunk(s)).\n"
                f"  Source: {file_source}\n"
                f"  Hash tracked — refresh_vault will detect changes automatically."
            )

        if mode == "full_text":
            from digest.arxiv.convert import convert_pdf
            import tempfile
            print(f"  Converting {pdf_path.name} to Markdown...", flush=True)
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = _Path(tmp)
                convert_pdf(pdf_path, tmp_path)
                md_path = tmp_path / f"{pdf_path.stem}.md"
                if not md_path.exists():
                    return "[Error: PDF conversion produced no output]"
                content = md_path.read_text(encoding="utf-8")
            print("  Chunking and indexing full text...", flush=True)
            ids = add_texts(
                content=content, doc_type="paper", visibility=visibility,
                source=file_source,
                extra_metadata={"title": title, "file_path": str(pdf_path),
                                "score": score, "track": track, "storage_mode": "full_text"},
                store=store,
            )
        else:
            print(f"  Generating summary from {pdf_path.name}...", flush=True)
            summary = provider_obj.summarize(title, pdf_path)
            ids = add_texts(
                content=f"{title}\n\n{summary}", doc_type="paper", visibility=visibility,
                source=file_source,
                extra_metadata={"title": title, "file_path": str(pdf_path),
                                "score": score, "track": track, "storage_mode": "summary"},
                store=store,
            )

        return (
            f"Added paper \"{title}\" ({mode}, {visibility}, {len(ids)} chunk(s)).\n"
            f"  Source: {file_source}"
        )
    except Exception as exc:
        return f"[add_document error: {exc}]"


def _resolve_local_file(source: str, meta: dict, vault: Path) -> "Path | None":
    """Return the local filesystem path for a document, or None if no local file exists."""
    from urllib.parse import urlparse
    if source.startswith("file:///"):
        return Path(urlparse(source).path)
    if meta.get("file_path"):
        return vault / meta["file_path"]
    return None


def _remove_document(args: dict, vault: Path) -> str:
    try:
        from digest.kb.store import get_store

        source = args.get("source", "")
        confirmed = bool(args.get("confirmed", False))
        delete_file = bool(args.get("delete_file", False))
        if not source:
            return "[Error: source URL is required]"

        store = get_store()
        result = store._collection.get(
            where={"source": {"$eq": source}}, include=["metadatas"]
        )
        ids = result["ids"]
        if not ids:
            return f"No documents found with source: {source}"

        meta = result["metadatas"][0] if result["metadatas"] else {}
        title = meta.get("title", "untitled")
        doc_type = meta.get("doc_type", "document")
        local_file = _resolve_local_file(source, meta, vault)

        if not confirmed:
            lines = [
                f"Found {len(ids)} chunk(s) to remove:",
                f"  Title:  {title}",
                f"  Type:   {doc_type}",
                f"  Source: {source}",
            ]
            if delete_file:
                if local_file and local_file.exists():
                    lines.append(f"  File:   {local_file}  ← will be PERMANENTLY DELETED")
                else:
                    lines.append("  File:   no local file found (database entry only will be removed)")
            else:
                lines.append("  Note:   database entry only — no files will be deleted")
            lines.append("\nAsk the user to confirm, then call remove_document again with confirmed=true.")
            return "\n".join(lines)

        # Confirmed — execute
        store.delete(ids)
        msg = f"Removed \"{title}\" ({len(ids)} chunk(s)) from the knowledge base."
        if delete_file:
            if local_file and local_file.exists():
                local_file.unlink()
                msg += f"\nDeleted file: {local_file}"
            else:
                msg += "\nNo local file found — database entry only was removed."
        else:
            msg += "\nNo files were deleted."
        return msg
    except Exception as exc:
        return f"[remove_document error: {exc}]"


def _list_papers(args: dict) -> str:
    try:
        from digest.kb.store import get_store, list_papers

        limit = min(int(args.get("limit", 10)), 50)
        papers = list_papers(limit=limit, store=get_store())
        if not papers:
            return "[No papers in knowledge base.]"
        lines = [f"{len(papers)} paper(s):\n"]
        for p in papers:
            lines.append(
                f"• [{p.get('score', '?')}/10] {p.get('title', 'untitled')}\n"
                f"  {p.get('source', 'no source')}"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"[list_papers error: {exc}]"


def _kb_stats() -> str:
    try:
        from digest.kb.store import count, count_unique_documents, get_store

        store = get_store()
        papers = count_unique_documents("paper", "source", store)
        notes = count_unique_documents("note", "file_path", store)
        chunks = count(store)
        return (
            f"Knowledge base:\n"
            f"  {papers} papers · {notes} notes\n"
            f"  {chunks} total chunks"
        )
    except Exception as exc:
        return f"[kb_stats error: {exc}]"


def _update_file_path(args: dict) -> str:
    try:
        from digest.kb.store import get_store, update_file_path

        source = args.get("source", "")
        new_path = args.get("new_path", "")
        if not source or not new_path:
            return "[Error: both source and new_path are required]"
        n = update_file_path(source, new_path, get_store())
        if n == 0:
            return f"No documents found with source: {source}"
        resolved = str(Path(new_path).expanduser().resolve())
        return f"Updated {n} chunk(s) — new path: {resolved}"
    except Exception as exc:
        return f"[update_file_path error: {exc}]"


def _use_own_knowledge() -> str:
    return "Understood. Proceeding to answer from training knowledge."


def _index_vault_tool(vault: Path, force: bool = False) -> str:
    try:
        from digest.kb.store import get_store, refresh_vault

        store = get_store()
        if force:
            print("  Clearing existing vault index...", flush=True)
            try:
                result = store._collection.get(
                    where={"doc_type": {"$eq": "note"}}, include=["metadatas"]
                )
                # Only delete vault .md notes (relative paths).
                # PDF notes (absolute paths ending in .pdf) are left untouched.
                ids_to_delete = [
                    id_ for id_, meta in zip(result["ids"], result["metadatas"])
                    if not meta.get("file_path", "").endswith(".pdf")
                ]
                if ids_to_delete:
                    store.delete(ids_to_delete)
                    print(f"  Cleared {len(ids_to_delete)} chunks", flush=True)
            except Exception:
                pass

        print(f"  Indexing vault: {vault}", flush=True)
        added, updated, deleted = refresh_vault(vault, store)
        return f"Vault indexed: +{added} new, ~{updated} changed, -{deleted} removed"
    except Exception as exc:
        return f"[index_vault error: {exc}]"


def _dispatch_tool(
    name: str,
    arguments: dict,
    vault: Path,
    provider_str: str,
    provider_obj,
) -> str:
    arg_summary = ", ".join(f"{k}={repr(v)[:40]}" for k, v in arguments.items())
    print(f"  → {name}({arg_summary})", flush=True)

    if name == "read_file":
        return read_file(vault, arguments.get("path", ""), provider_str)
    if name == "retrieve_papers":
        return _retrieve_papers(arguments, provider_str)
    if name == "search_notes":
        return _search_notes(arguments, provider_str)
    if name == "add_document":
        return _add_document(arguments, provider_obj)
    if name == "remove_document":
        return _remove_document(arguments, vault)
    if name == "list_papers":
        return _list_papers(arguments)
    if name == "kb_stats":
        return _kb_stats()
    if name == "update_file_path":
        return _update_file_path(arguments)
    if name == "index_vault":
        return _index_vault_tool(vault, bool(arguments.get("force", False)))
    if name == "use_own_knowledge":
        return _use_own_knowledge()
    return f"[Error: unknown tool '{name}']"


# ── Vault auto-refresh ─────────────────────────────────────────────────────────


def _auto_refresh_vault(vault: Path) -> None:
    try:
        from digest.kb.store import get_store, refresh_vault

        store = get_store()
        try:
            result = store._collection.get(where={"doc_type": {"$eq": "note"}}, include=[])
            if not result["ids"]:
                print("Vault not yet indexed — run: kb index-vault", flush=True)
                return
        except Exception:
            return
        added, updated, deleted = refresh_vault(vault, store)
        if added + updated + deleted > 0:
            print(
                f"Vault index refreshed: +{added} new, ~{updated} changed, -{deleted} removed",
                flush=True,
            )
    except Exception as exc:
        print(f"Warning: vault index refresh failed: {exc}", flush=True)


# ── Session ────────────────────────────────────────────────────────────────────


def run_session(vault: Path, kb_only: bool = True) -> None:
    cfg = get_config()
    provider = make_provider(cfg.provider)
    system_prompt = build_system_prompt(kb_only=kb_only)
    tools = TOOLS if kb_only else TOOLS + [USE_OWN_KNOWLEDGE_TOOL]
    messages: list[dict] = []

    provider_label = (
        f"Anthropic ({cfg.anthropic_model})"
        if cfg.provider == "anthropic"
        else f"Ollama ({cfg.ollama_model})"
    )
    print(f"Vault chat ready. Provider: {provider_label}  Vault: {vault}")
    print("Type your question and press Enter. Ctrl-C or Ctrl-D to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        try:
            reply = provider.agentic_turn(
                messages=messages,
                tools=tools,
                dispatch_fn=lambda name, args: _dispatch_tool(
                    name, args, vault, cfg.provider, provider
                ),
                system=system_prompt,
            )
        except LLMError as exc:
            print(f"[LLM error: {exc}]")
            messages.pop()
            continue

        print(f"\nAssistant: {reply}\n")


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    cfg = get_config()
    parser = argparse.ArgumentParser(
        prog="vault-chat",
        description="Knowledge base agent — query and manage via natural language.",
    )
    parser.add_argument(
        "vault",
        nargs="?",
        help=f"Path to the vault root (default from config: {cfg.vault_path})",
    )
    parser.add_argument(
        "--no-db-only",
        dest="kb_only",
        action="store_false",
        default=True,
        help="Allow the LLM to fall back to its training knowledge when the database has no results.",
    )
    args = parser.parse_args()

    vault = Path(args.vault).expanduser() if args.vault else cfg.vault_path
    if not vault.exists():
        print(f"Error: vault path does not exist: {vault}", file=sys.stderr)
        sys.exit(1)

    _auto_refresh_vault(vault)
    run_session(vault, kb_only=args.kb_only)


if __name__ == "__main__":
    main()
