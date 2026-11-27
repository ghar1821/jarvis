"""
kb — the jarvis command line.

The index mirrors two folders you manage yourself: your Obsidian vault and
your papers folder. There is no add or
remove command — save a file there to add it, delete the file to remove it.
MCP clients reach the index through jarvis-mcp, which they start themselves.

Subcommands:
  serve           Run the index server and keep it in sync (leave running)
  sync            Run one sync pass now
  rebuild         Recreate the whole index from the two folders
  list            List indexed papers (--notes for vault notes/records)
  stats           Show document and chunk counts
  schema [key]    Show which metadata keys/values exist (your record ontology)
  status          Show whether `kb serve` is running and how its last sync went
  doctor          Diagnose index health

Examples:
  uv run kb serve
  uv run kb list --notes --category job_application --status rejected
  uv run kb schema status
  uv run kb doctor
"""

import argparse
import sys

from jarvis.core.errors import ConfigError
from jarvis.kb.store import allow_direct_index_access


# ── Sync and rebuild ──────────────────────────────────────────────────────────


def _print_counts(counts: dict[str, int]) -> None:
    print(
        f"  notes:  +{counts['notes_added']} new, ~{counts['notes_updated']} changed, "
        f"-{counts['notes_deleted']} removed\n"
        f"  papers: +{counts['papers_added']} new, ~{counts['papers_updated']} changed, "
        f"→{counts['papers_moved']} moved, -{counts['papers_deleted']} removed"
    )


def cmd_sync() -> None:
    from jarvis.core.config import get_config
    from .mirror import sync_all

    cfg = get_config()
    print(f"Syncing {cfg.vault_path} and {cfg.papers_dir or '(no papers_dir set)'}...", flush=True)
    _print_counts(sync_all(cfg))


def cmd_rebuild() -> None:
    """Rebuild from files after a confirmation that says exactly what is lost."""
    from jarvis.core.config import get_config
    from .mirror import rebuild_index

    cfg = get_config()
    print(
        "This rebuilds the index from scratch, from:\n"
        f"  vault:  {cfg.vault_path}  (every .md file in it)\n"
        f"  papers: {cfg.papers_dir or '— none (no papers_dir set)'}\n"
        "Only the index changes; no file anywhere is touched."
    )
    if input("Rebuild? [y/N] ").strip().lower() != "y":
        print("Cancelled.")
        return
    counts = rebuild_index(cfg)
    print("Done:")
    _print_counts(counts)
    print(
        "\nThe rebuilt index is a new collection, so restart `kb serve` (if it is\n"
        "running) and your MCP client before using them."
    )


# ── Inspection ────────────────────────────────────────────────────────────────


def cmd_list(args: argparse.Namespace) -> None:
    """List indexed papers, or vault notes/records with --notes."""
    from jarvis.core.config import get_config
    from .frontmatter import display_tags
    from .metadata import doi_url
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
        print(f"No matching {'notes' if args.notes else 'papers'} in the knowledge base.")
        return

    vault = get_config().vault_path
    for d in documents:
        print(f"{d.get('title', 'untitled')}  [{d.get('chunk_count', '?')} chunks]")
        if doc_type == "paper":
            print(f"  {d.get('file_path') or d.get('source', 'no source')}")
            if d.get("authors"):
                print(f"  Authors: {d['authors']}")
            if d.get("doi"):
                print(f"  DOI: {doi_url(d['doi'])}")
        else:
            print(f"  {vault / d.get('file_path', '')}")
            record = " · ".join(
                str(d[field]) for field in ("category", "entity", "status") if d.get(field)
            )
            if record:
                print(f"  {record}")
            if d.get("event_date"):
                print(f"  Date: {d['event_date']}")
            if d.get("tags"):
                print(f"  Tags: {display_tags(d['tags'])}")
        print()


def cmd_stats() -> None:
    from .store import count, count_unique_documents, get_store

    store = get_store()
    papers = count_unique_documents("paper", "source", store)
    notes = count_unique_documents("note", "file_path", store)
    print(f"Documents:  {papers} papers · {notes} notes")
    print(f"Chunks:     {count(store)} total")


def cmd_schema(args: argparse.Namespace) -> None:
    """
    Show the metadata keys present in the store, or one key's distinct values.

    This exists to make your own ontology visible. jarvis enforces no
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
        for value, chunk_count in sorted(values.items(), key=lambda pair: -pair[1]):
            print(f"  {chunk_count:6d}  {value}")
        return

    counts = metadata_key_counts(store)
    if not counts:
        print("Knowledge base is empty.")
        return
    print("Metadata keys (chunk counts). Run `kb schema <key>` to see its values.\n")
    for key, chunk_count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])):
        marker = "  (custom frontmatter)" if key.startswith("x_") else ""
        print(f"  {chunk_count:6d}  {key}{marker}")


# ── Doctor ────────────────────────────────────────────────────────────────────

# Everything that touches the vector index runs here, in a child process.
# Reading a damaged index does not raise — it terminates the process by signal
# (SIGBUS from a corrupt HNSW header, SIGSEGV from a bad pointer), which no
# Python handler can intercept. Running it in a child is what turns "the
# command vanished with no output" into a diagnosis the user can act on.
_DOCTOR_PROBE = """
import json
from jarvis.core.config import get_config
from jarvis.core.errors import RAGError
from jarvis.kb.store import _server_client, allow_direct_index_access, count, get_store, search

result = {"opened": False, "count": None, "search_ok": None, "error": None,
          "via_server": None}
try:
    try:
        _server_client(get_config().server_port)
        result["via_server"] = True
    except Exception:
        result["via_server"] = False
    # Doctor has to work when the server is the broken thing.
    allow_direct_index_access()
    store = get_store()
    result["opened"] = True
    result["count"] = count(store)
    if result["count"]:
        try:
            search("diagnostic probe query", n_results=1, store=store, rerank=False)
            result["search_ok"] = True
        except RAGError as exc:
            result["search_ok"] = False
            result["error"] = str(exc)
except Exception as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"
print("DOCTOR_RESULT " + json.dumps(result))
"""

_INDEX_UNREADABLE = """\
✗ The vector index is damaged beyond reading.
  Probing it killed a child process outright (signal {signal}), which means the
  index file itself is unreadable — not a Python-level error that could be
  caught and reported.

  Nothing is lost: every entry comes from a file in your vault or papers
  folder. Recover with:
      uv run kb rebuild

  That builds a fresh index from the files and swaps it in, without ever
  reading the damaged one."""


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
    Diagnose index health: open the store (exercises the version and
    embedding-model checks), count chunks, then probe a real search (exercises
    corruption detection). Exits non-zero on any failure so this is scriptable.
    """
    print("Checking knowledge base...", flush=True)
    result = _run_doctor_probe()

    if result is None:
        print(_INDEX_UNREADABLE.format(signal="SIGBUS/SIGSEGV"), file=sys.stderr)
        sys.exit(1)

    if not result["opened"]:
        print(f"✗ Failed to open store: {result['error']}", file=sys.stderr)
        sys.exit(1)
    print("✓ Store opened (built by this version, embedding model matches)")
    if result.get("via_server"):
        print("✓ kb serve is running (jarvis-mcp uses it)")
    elif result.get("via_server") is False:
        print("• kb serve is NOT running — checked the index files directly.\n"
              "  jarvis-mcp needs it: uv run kb serve")

    print(f"✓ {result['count']} chunk(s) indexed")
    if not result["count"]:
        print("Knowledge base is empty — run `uv run kb sync` to index your folders.")
        return

    if result["search_ok"] is False:
        print(f"✗ Search failed:\n  {result['error']}", file=sys.stderr)
        sys.exit(1)
    print("✓ Search probe succeeded")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="kb",
        description="jarvis: mirror your vault and papers into a searchable index for MCP clients.",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    sub.add_parser("serve", help="Run the index server and keep it in sync (leave running)")
    sub.add_parser("sync", help="Run one sync pass now")
    sub.add_parser("rebuild", help="Recreate the whole index from the vault and papers folders")
    p_list = sub.add_parser("list", help="List indexed papers, or notes/records with --notes")
    p_list.add_argument("--limit", type=int, default=20)
    p_list.add_argument("--notes", action="store_true", help="List vault notes/records instead of papers")
    p_list.add_argument("--category", help="Filter notes by record type, e.g. job_application")
    p_list.add_argument("--status", help="Filter notes by status, e.g. rejected")
    p_list.add_argument("--entity", help="Filter notes by organisation or person")
    sub.add_parser("stats", help="Show document and chunk counts")
    p_schema = sub.add_parser("schema", help="Show which metadata keys and values exist in the store")
    p_schema.add_argument("key", nargs="?", help="Show the distinct values of one key")
    sub.add_parser("status", help="Show whether kb serve is running and how its last sync went")
    sub.add_parser("doctor", help="Diagnose index health")

    args = parser.parse_args()

    if args.command == "serve":
        # Long-lived: it runs the server itself and must never fall back to
        # opening the index files directly.
        from .serve import serve

        serve()
        return

    # Every other command is one-shot: it does its work and exits. That is
    # what makes it safe to open the index files directly when the server is
    # not running — and it keeps `kb doctor` usable at exactly the moment the
    # server is the thing that is broken.
    allow_direct_index_access()

    from .serve import print_status

    dispatch = {
        "sync":    cmd_sync,
        "rebuild": cmd_rebuild,
        "list":    lambda: cmd_list(args),
        "stats":   cmd_stats,
        "schema":  lambda: cmd_schema(args),
        "status":  print_status,
        "doctor":  cmd_doctor,
    }
    try:
        dispatch[args.command]()
    except ConfigError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
