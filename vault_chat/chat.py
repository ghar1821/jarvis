"""
CLI chat tool connecting the Obsidian vault and paper RAG to an LLM.

Provider is selected from config (default: ollama). Override with CHAT_PROVIDER env var
or via ~/.paper_digest/config.toml → [chat] provider = "anthropic".

Auth for Anthropic:
  Option 1: export ANTHROPIC_API_KEY=sk-ant-...
  Option 2: paper-rag auth login  (browser OAuth → ~/.paper_digest/auth.json)
"""

import sys
from pathlib import Path

from digest.config import get_config
from digest.errors import LLMError
from digest.llm import make_provider

# ── Tool definitions ───────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the full contents of a file from the Obsidian vault. "
                "Use this whenever you need the actual content of a note, the glossary, "
                "the reading list, common cross-paper themes, or any other vault file "
                "before answering. Always read relevant files before answering — never "
                "answer from memory alone."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file within the vault, e.g. 'to-read.md' or 'notes/paper.md'",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_papers",
            "description": (
                "Search the local paper RAG database for papers relevant to a query. "
                "Use this to find specific papers, look up details about papers added from "
                "high-scoring digest runs, or explore research topics. "
                "Returns up to n_results papers with title, authors, score, track, and a summary excerpt."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language query"},
                    "n_results": {"type": "integer", "description": "Number of results (default 5, max 20)", "default": 5},
                    "score_min": {"type": "integer", "description": "Minimum relevance score (1-10)"},
                    "track": {"type": "string", "description": "Track filter: 'Track 1' or 'Track 2'"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_vault",
            "description": (
                "Semantically search Obsidian vault notes to discover relevant files. "
                "Use this to find which notes exist on a topic before reading them in full "
                "with read_file. Returns matching chunks with file path and title."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language query"},
                    "n_results": {"type": "integer", "description": "Number of results (default 5)", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_paper",
            "description": (
                "Remove a paper from the RAG database by its document ID. "
                "Always call retrieve_papers first to find the paper and confirm with the "
                "user before removing. Never remove without explicit confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "Document ID from retrieve_papers results"},
                },
                "required": ["doc_id"],
            },
        },
    },
]

# ── Vault helpers ──────────────────────────────────────────────────────────────


def build_file_index(vault: Path) -> str:
    lines = ["Available files in vault:"]
    for path in sorted(vault.rglob("*.md")):
        lines.append(f"  {path.relative_to(vault)}")
    return "\n".join(lines)


def read_file(vault: Path, rel_path: str) -> str:
    target = (vault / rel_path).resolve()
    try:
        target.relative_to(vault.resolve())
    except ValueError:
        return f"[Error: '{rel_path}' is outside the vault]"
    if not target.exists() or not target.is_file():
        return f"[Error: file not found: '{rel_path}']"
    return target.read_text(encoding="utf-8")


def build_system_prompt(vault: Path) -> str:
    skill_path = vault / "system" / "SKILL.md"
    base = (
        skill_path.read_text(encoding="utf-8").rstrip()
        if skill_path.exists()
        else (
            "You are a knowledgeable assistant with access to an Obsidian vault "
            "and a local database of curated research papers. "
            "Use retrieve_papers to search indexed papers, search_vault to discover "
            "relevant vault notes, and read_file to read specific files in full."
        )
    )
    return f"{base}\n\n{build_file_index(vault)}"


# ── Tool dispatch ──────────────────────────────────────────────────────────────


def _dispatch_tool(name: str, arguments: dict, vault: Path) -> str:
    if name == "read_file":
        return read_file(vault, arguments.get("path", ""))
    if name == "retrieve_papers":
        return _retrieve_papers(arguments)
    if name == "search_vault":
        return _search_vault(arguments)
    if name == "remove_paper":
        return _remove_paper(arguments)
    return f"[Error: unknown tool '{name}']"


def _retrieve_papers(args: dict) -> str:
    try:
        from digest.rag import RAGError, get_papers_collection, retrieve_papers

        results = retrieve_papers(
            query=args["query"],
            n_results=min(int(args.get("n_results", 5)), 20),
            score_min=args.get("score_min"),
            track=args.get("track"),
            collection=get_papers_collection(),
        )
        if not results:
            return "[No papers found. The RAG database may be empty — run 'paper-rag add' or wait for a digest run.]"
        lines = [f"Found {len(results)} paper(s):\n"]
        for i, r in enumerate(results, 1):
            lines.append(
                f"{i}. [{r.get('score', '?')}/10 · {r.get('track', '')}] {r['title']}\n"
                f"   Authors: {r.get('authors', 'N/A')}\n"
                f"   Published: {r.get('published', 'N/A')}  |  {r.get('link', '')}\n"
                f"   Doc ID: {r['doc_id']}\n"
                f"   Summary: {r.get('document', '')[:300].replace(chr(10), ' ')}...\n"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"[RAG papers error: {exc}]"


def _search_vault(args: dict) -> str:
    try:
        from digest.rag import get_vault_collection, search_vault

        results = search_vault(
            query=args["query"],
            n_results=min(int(args.get("n_results", 5)), 20),
            collection=get_vault_collection(),
        )
        if not results:
            return "[No vault notes found. Run 'paper-rag index-vault' to index your vault.]"
        lines = [f"Found {len(results)} matching note chunk(s):\n"]
        for i, r in enumerate(results, 1):
            lines.append(
                f"{i}. {r['title']}  ({r['file_path']})\n"
                f"   Excerpt: {r['chunk'][:300].replace(chr(10), ' ')}...\n"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"[RAG vault error: {exc}]"


def _remove_paper(args: dict) -> str:
    try:
        from digest.rag import count, get_papers_collection, remove_paper

        col = get_papers_collection()
        before = count(col)
        remove_paper(args["doc_id"], col)
        after = count(col)
        if before != after:
            return f"Removed paper (doc_id: {args['doc_id']}). {after} papers remain in RAG."
        return f"No paper found with doc_id: {args['doc_id']}"
    except Exception as exc:
        return f"[Remove error: {exc}]"


# ── Vault auto-refresh ─────────────────────────────────────────────────────────


def _auto_refresh_vault(vault: Path) -> None:
    try:
        from digest.rag import count_vault, get_vault_collection, refresh_vault

        col = get_vault_collection()
        if count_vault(col) == 0:
            print("Vault not yet indexed — run: paper-rag index-vault", flush=True)
            return
        added, updated, deleted = refresh_vault(vault, col)
        if added + updated + deleted > 0:
            print(
                f"Vault index refreshed: +{added} new, ~{updated} changed, -{deleted} removed",
                flush=True,
            )
    except Exception as exc:
        print(f"Warning: vault index refresh failed: {exc}", flush=True)


# ── Session loop ───────────────────────────────────────────────────────────────


def run_session(vault: Path) -> None:
    cfg = get_config()
    provider = make_provider(cfg.provider)
    system_prompt = build_system_prompt(vault)
    messages: list[dict] = []

    provider_label = f"Anthropic ({cfg.anthropic_model})" if cfg.provider == "anthropic" else f"Ollama ({cfg.ollama_model})"
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
                tools=TOOLS,
                dispatch_fn=lambda name, args: _dispatch_tool(name, args, vault),
                system=system_prompt,
            )
        except LLMError as exc:
            print(f"[LLM error: {exc}]")
            messages.pop()  # Remove the failed user message so the user can retry
            continue

        print(f"\nAssistant: {reply}\n")


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    cfg = get_config()
    vault = (
        Path(sys.argv[1]).expanduser()
        if len(sys.argv) > 1
        else cfg.vault_path
    )

    if not vault.exists():
        print(f"Error: vault path does not exist: {vault}", file=sys.stderr)
        sys.exit(1)

    _auto_refresh_vault(vault)
    run_session(vault)


if __name__ == "__main__":
    main()
