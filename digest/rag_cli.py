"""
paper-rag CLI — manage the local RAG database.

Subcommands:
  auth login        Browser OAuth PKCE flow → ~/.paper_digest/auth.json
  auth status       Show active auth method

  add <url|path>    Add a paper by arXiv URL or local PDF path
  query <query>     Semantic search over papers
  list              List papers in the database
  stats             Show counts for papers and vault notes
  remove <doc_id>   Remove a paper by ID

  index-vault       Full (re)index of Obsidian vault
  refresh-vault     Incremental update of vault index

Usage examples:
  paper-rag auth login
  paper-rag add https://arxiv.org/abs/2406.04093 --score 9 --track "Track 1"
  paper-rag add paper.pdf --provider anthropic
  paper-rag query "sparse autoencoders" --n 5 --score-min 8
  paper-rag list --limit 20
  paper-rag stats
  paper-rag remove 2301.07041
  paper-rag index-vault --vault-path ~/vault
  paper-rag refresh-vault
"""

import argparse
import base64
import hashlib
import json
import os
import sys
import urllib.parse
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

CALLBACK_TIMEOUT = 120  # seconds to wait for browser callback


# ── OAuth PKCE helpers ─────────────────────────────────────────────────────────


def _generate_pkce() -> tuple[str, str]:
    code_verifier = base64.urlsafe_b64encode(os.urandom(32)).rstrip(b"=").decode()
    digest = hashlib.sha256(code_verifier.encode()).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return code_verifier, code_challenge


def _wait_for_oauth_callback() -> str | None:
    auth_code: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if "code" in params:
                auth_code.append(params["code"][0])
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(
                    b"<html><body><h2>Authenticated!</h2>"
                    b"<p>You can close this tab and return to the terminal.</p>"
                    b"</body></html>"
                )
            else:
                error = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get("error", ["unknown"])[0]
                self.send_response(400)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(f"<html><body><h2>Auth error: {error}</h2></body></html>".encode())

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("localhost", 8080), Handler)
    server.timeout = CALLBACK_TIMEOUT
    thread = Thread(target=server.handle_request, daemon=True)
    thread.start()
    thread.join(timeout=CALLBACK_TIMEOUT + 2)
    server.server_close()
    return auth_code[0] if auth_code else None


def _save_auth(token_data: dict, auth_file: Path) -> None:
    auth_file.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    record = {
        "access_token": token_data["access_token"],
        "refresh_token": token_data.get("refresh_token", ""),
        "expires_at": datetime.fromtimestamp(
            now.timestamp() + token_data.get("expires_in", 3600), tz=timezone.utc
        ).isoformat(),
        "saved_at": now.isoformat(),
    }
    auth_file.write_text(json.dumps(record, indent=2))
    auth_file.chmod(0o600)


# ── Auth subcommands ───────────────────────────────────────────────────────────


def cmd_auth_login() -> None:
    from .config import get_config

    cfg = get_config()
    if not cfg.oauth_client_id:
        print("Error: oauth_client_id is not configured.", file=sys.stderr)
        print("  Set ANTHROPIC_OAUTH_CLIENT_ID env var, or add to ~/.paper_digest/config.toml:", file=sys.stderr)
        print("    [auth]", file=sys.stderr)
        print("    oauth_client_id = \"your-client-id\"", file=sys.stderr)
        print("  Confirm OAuth app credentials from https://docs.anthropic.com", file=sys.stderr)
        sys.exit(1)

    import requests

    code_verifier, code_challenge = _generate_pkce()
    state = base64.urlsafe_b64encode(os.urandom(16)).decode()
    params = urllib.parse.urlencode({
        "response_type": "code",
        "client_id": cfg.oauth_client_id,
        "redirect_uri": "http://localhost:8080/callback",
        "scope": "api",
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    })
    auth_url = f"{cfg.oauth_auth_url}?{params}"
    print("Opening browser for claude.ai authentication...")
    print(f"  URL: {auth_url}\n")
    webbrowser.open(auth_url)
    print(f"Waiting for callback (up to {CALLBACK_TIMEOUT}s)...")

    code = _wait_for_oauth_callback()
    if not code:
        print("Error: timed out waiting for OAuth callback.", file=sys.stderr)
        sys.exit(1)

    print("Exchanging code for token...")
    resp = requests.post(
        cfg.oauth_token_url,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": "http://localhost:8080/callback",
            "client_id": cfg.oauth_client_id,
            "code_verifier": code_verifier,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    resp.raise_for_status()
    _save_auth(resp.json(), cfg.auth_file)
    print(f"Authenticated successfully. Credentials saved to {cfg.auth_file}")


def cmd_auth_status() -> None:
    from .config import get_config

    cfg = get_config()
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("Active: ANTHROPIC_API_KEY environment variable")
        return
    if cfg.auth_file.exists():
        auth = json.loads(cfg.auth_file.read_text())
        try:
            exp = datetime.fromisoformat(auth.get("expires_at", ""))
            remaining = exp - datetime.now(timezone.utc)
            if remaining.total_seconds() > 0:
                print(f"Active: claude.ai OAuth token (expires in ~{int(remaining.total_seconds() / 60)} min)")
            else:
                print("Stored OAuth token is expired. Run: paper-rag auth login")
        except ValueError:
            print(f"Stored auth file: {cfg.auth_file} (expiry unknown)")
        return
    print("No credentials configured.")
    print("  Option 1: export ANTHROPIC_API_KEY=sk-ant-...")
    print("  Option 2: paper-rag auth login  (browser OAuth)")


# ── Paper add ─────────────────────────────────────────────────────────────────


def cmd_add(args: argparse.Namespace) -> None:
    from .config import get_config
    from .convert import parse_arxiv_url
    from .fetch import fetch_arxiv_paper
    from .llm import make_provider
    from .rag import add_paper, get_papers_collection

    cfg = get_config()
    provider = make_provider(args.provider or cfg.provider)
    collection = get_papers_collection()
    input_str: str = args.input

    if input_str.startswith("http://") or input_str.startswith("https://"):
        arxiv_id = parse_arxiv_url(input_str)
        if not arxiv_id:
            print(f"Error: could not parse arXiv ID from URL: {input_str}", file=sys.stderr)
            sys.exit(1)
        print(f"Fetching metadata for arXiv:{arxiv_id}...")
        paper = fetch_arxiv_paper(arxiv_id)
        print(f"  Title: {paper['title']}")
        print("Generating summary...")
        summary = provider.summarize(paper["title"], paper["abstract"])

    elif Path(input_str).exists() and Path(input_str).suffix.lower() == ".pdf":
        pdf_path = Path(input_str)
        title = args.title or pdf_path.stem
        paper = {
            "title": title,
            "abstract": "",
            "link": pdf_path.resolve().as_uri(),
            "authors": "",
            "published": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "source": "local",
        }
        print(f"Generating summary from PDF: {pdf_path.name}...")
        summary = provider.summarize(title, pdf_path)
        paper["abstract"] = summary[:500]

    else:
        print(f"Error: '{input_str}' is not a valid arXiv URL or PDF path.", file=sys.stderr)
        sys.exit(1)

    doc_id = add_paper(paper=paper, dense_summary=summary, score=args.score, track=args.track, collection=collection)
    print(f"Added to RAG (doc_id: {doc_id})")


# ── Query / list / stats / remove ─────────────────────────────────────────────


def cmd_query(args: argparse.Namespace) -> None:
    from .rag import get_papers_collection, retrieve_papers

    results = retrieve_papers(
        query=args.query, n_results=args.n,
        score_min=args.score_min, track=args.track,
        collection=get_papers_collection(),
    )
    if not results:
        print("No results found.")
        return
    print(f"Found {len(results)} result(s):\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r.get('score', '?')}/10 · {r.get('track', '')}] {r['title']}")
        print(f"   Authors:   {r.get('authors', 'N/A')}")
        print(f"   Published: {r.get('published', 'N/A')}  |  {r.get('link', '')}")
        print(f"   Doc ID:    {r['doc_id']}")
        preview = r.get("document", "")[:200].replace("\n", " ")
        if preview:
            print(f"   Preview:   {preview}...")
        print()


def cmd_list(args: argparse.Namespace) -> None:
    from .rag import get_papers_collection, list_papers

    papers = list_papers(limit=args.limit, collection=get_papers_collection())
    if not papers:
        print("No papers in RAG database.")
        return
    print(f"{'Doc ID':<20} {'Score':>5}  {'Published':<12}  Title")
    print("-" * 80)
    for p in papers:
        print(f"{p['doc_id']:<20} {str(p.get('score', '?')):>5}  {p.get('published', 'N/A'):<12}  {p.get('title', '')[:50]}")


def cmd_stats() -> None:
    from .rag import count, count_vault, get_papers_collection, get_vault_collection

    print(f"Papers in RAG:      {count(get_papers_collection())}")
    print(f"Vault note chunks:  {count_vault(get_vault_collection())}")


def cmd_remove(args: argparse.Namespace) -> None:
    from .rag import count, get_papers_collection, remove_paper

    col = get_papers_collection()
    before = count(col)
    remove_paper(args.doc_id, col)
    if count(col) != before:
        print(f"Removed {args.doc_id}")
    else:
        print(f"No paper found with doc_id: {args.doc_id}")


# ── Vault index ────────────────────────────────────────────────────────────────


def cmd_index_vault(args: argparse.Namespace) -> None:
    from .config import get_config
    from .rag import get_vault_collection, refresh_vault

    cfg = get_config()
    vault = Path(args.vault_path).expanduser() if args.vault_path else cfg.vault_path
    if not vault.exists():
        print(f"Error: vault path does not exist: {vault}", file=sys.stderr)
        sys.exit(1)

    col = get_vault_collection()
    if args.force:
        print("Clearing existing vault index...", flush=True)
        existing = col.get(include=[])
        if existing["ids"]:
            col.delete(ids=existing["ids"])

    print(f"Indexing vault: {vault}", flush=True)
    added, updated, deleted = refresh_vault(vault, col)
    print(f"Done — +{added} new, ~{updated} changed, -{deleted} removed")


def cmd_refresh_vault(args: argparse.Namespace) -> None:
    from .config import get_config
    from .rag import get_vault_collection, refresh_vault

    cfg = get_config()
    vault = Path(args.vault_path).expanduser() if args.vault_path else cfg.vault_path
    if not vault.exists():
        print(f"Error: vault path does not exist: {vault}", file=sys.stderr)
        sys.exit(1)

    added, updated, deleted = refresh_vault(vault, get_vault_collection())
    if added + updated + deleted == 0:
        print("Vault index is up to date.")
    else:
        print(f"Vault refreshed — +{added} new, ~{updated} changed, -{deleted} removed")


# ── CLI entry point ────────────────────────────────────────────────────────────


def main() -> None:
    from .config import get_config

    cfg = get_config()
    parser = argparse.ArgumentParser(prog="paper-rag", description="Manage the local RAG database.")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # auth
    p_auth = sub.add_parser("auth", help="Manage authentication")
    auth_sub = p_auth.add_subparsers(dest="auth_command", metavar="<subcommand>")
    auth_sub.required = True
    auth_sub.add_parser("login", help="Browser OAuth PKCE login with claude.ai account")
    auth_sub.add_parser("status", help="Show active auth method")

    # add
    p_add = sub.add_parser("add", help="Add a paper by arXiv URL or local PDF")
    p_add.add_argument("input", help="arXiv URL or local PDF path")
    p_add.add_argument("--score", type=int, default=0)
    p_add.add_argument("--track", default="")
    p_add.add_argument("--title", default="", help="Override title (for local PDFs)")
    p_add.add_argument(
        "--provider", default="",
        help=f"'anthropic' or Ollama model name (default: from config, currently {cfg.provider})",
    )

    # query
    p_query = sub.add_parser("query", help="Semantic search over papers")
    p_query.add_argument("query")
    p_query.add_argument("--n", type=int, default=5)
    p_query.add_argument("--score-min", type=int, default=None, dest="score_min")
    p_query.add_argument("--track", default=None)

    # list / stats / remove
    p_list = sub.add_parser("list", help="List papers in the database")
    p_list.add_argument("--limit", type=int, default=20)
    sub.add_parser("stats", help="Show paper and vault note counts")
    p_remove = sub.add_parser("remove", help="Remove a paper by doc ID")
    p_remove.add_argument("doc_id")

    # index-vault / refresh-vault
    p_idx = sub.add_parser("index-vault", help="(Re)index the Obsidian vault")
    p_idx.add_argument("--vault-path", default="")
    p_idx.add_argument("--force", action="store_true", help="Clear existing index first")
    p_ref = sub.add_parser("refresh-vault", help="Incrementally update vault index")
    p_ref.add_argument("--vault-path", default="")

    args = parser.parse_args()
    dispatch = {
        "auth":          lambda: (cmd_auth_login() if args.auth_command == "login" else cmd_auth_status()),
        "add":           lambda: cmd_add(args),
        "query":         lambda: cmd_query(args),
        "list":          lambda: cmd_list(args),
        "stats":         cmd_stats,
        "remove":        lambda: cmd_remove(args),
        "index-vault":   lambda: cmd_index_vault(args),
        "refresh-vault": lambda: cmd_refresh_vault(args),
    }
    dispatch[args.command]()


if __name__ == "__main__":
    main()
