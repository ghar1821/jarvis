"""
`kb serve` — the one long-running jarvis process.

It does two things:

1. Runs the Chroma server (`chroma run`, bound to 127.0.0.1) as a child
   process. That server owns the index; jarvis-mcp and every other reader
   connect to it, so no process ever holds a stale private copy of the
   vectors while another rewrites them.
2. Mirrors the vault and papers folder into the index every `sync_minutes`
   (see jarvis/kb/mirror.py), plus once at start-up.

It runs in the foreground (tmux, its own terminal, or a login item) and
Ctrl-C stops both. The outcome of every sync is written to
~/.jarvis/state/serve_status.json, which `kb status` reads.

A plain loop rather than a scheduler library: there is one periodic job, and
`Event.wait(timeout)` is both the sleep and the shutdown signal.
"""

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from jarvis.core.config import Config, format_describe, get_config

STATUS_FILE = Path.home() / ".jarvis" / "state" / "serve_status.json"
LOG_DIR = Path.home() / ".jarvis" / "logs"

log = logging.getLogger("jarvis.serve")


# ── Status file ───────────────────────────────────────────────────────────────


def read_status(status_file: Path = STATUS_FILE) -> dict:
    """The last recorded status, or {} when serve has never run or the file is unreadable."""
    try:
        return json.loads(status_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_status(status: dict, status_file: Path = STATUS_FILE) -> None:
    """Atomically write the status file (temp file + os.replace)."""
    status_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = status_file.with_suffix(".json.tmp")
    tmp_file.write_text(json.dumps(status, indent=2), encoding="utf-8")
    os.replace(tmp_file, status_file)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── One sync pass ─────────────────────────────────────────────────────────────


def run_sync(status_file: Path = STATUS_FILE, store=None) -> bool:
    """
    One mirror pass over both folders, with its outcome recorded. Never raises:
    a failing pass is logged and retried at the next interval, and one bad
    file must not take the server down with it.
    """
    from jarvis.kb.mirror import sync_all

    status = read_status(status_file)
    try:
        counts = sync_all(store=store)
        changed = {key: value for key, value in counts.items() if value}
        log.info("sync: %s", ", ".join(f"{k} {v}" for k, v in changed.items()) or "no changes")
        status["last_sync"] = {"at": _now(), "ok": True, "counts": counts}
        ok = True
    except Exception as exc:
        log.exception("sync failed")
        status["last_sync"] = {"at": _now(), "ok": False, "error": str(exc)}
        ok = False
        # Reconnect on the next pass rather than reuse a handle that may be
        # the cause — after `kb rebuild` it names a collection that is gone.
        from jarvis.kb.store import forget_store

        forget_store()
    write_status(status, status_file)
    return ok


# ── The Chroma child process ──────────────────────────────────────────────────


def _start_chroma(cfg: Config) -> subprocess.Popen:
    """Launch `chroma run` on 127.0.0.1, its own output going to a log file."""
    chroma = shutil.which("chroma")
    if chroma is None:
        raise RuntimeError("The 'chroma' command is not on PATH. It ships with chromadb — try: uv sync")
    cfg.rag_dir.mkdir(parents=True, exist_ok=True)
    # Owner-only: the index holds the full text of your notes and papers.
    os.chmod(cfg.rag_dir, 0o700)
    chroma_log = open(LOG_DIR / "chroma.log", "a", encoding="utf-8")
    return subprocess.Popen(
        [
            chroma, "run",
            "--path", str(cfg.rag_dir),
            # Never a routable address: Chroma's server has no authentication.
            "--host", "127.0.0.1",
            "--port", str(cfg.server_port),
        ],
        stdout=chroma_log,
        stderr=subprocess.STDOUT,
    )


def _wait_until_answering(child: subprocess.Popen, port: int, timeout: float = 60.0) -> None:
    """Block until the server answers a heartbeat; raise if it dies or never does."""
    from jarvis.kb.store import _server_client

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(
                f"chroma exited with code {child.returncode} — is another `kb serve` "
                f"already using port {port}? See {LOG_DIR / 'chroma.log'}"
            )
        try:
            _server_client(port)
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"chroma did not start answering on port {port} within {timeout:.0f}s")


def _stop_chroma(child: subprocess.Popen) -> None:
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()


# ── Main ──────────────────────────────────────────────────────────────────────


def serve() -> None:
    """Run the index server and the periodic sync until Ctrl-C."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        # File first so it is always captured; stderr so a foreground run
        # shows the same lines live.
        handlers=[logging.FileHandler(LOG_DIR / "serve.log"), logging.StreamHandler(sys.stderr)],
    )
    # The Chroma client logs every HTTP request to the local server at INFO,
    # which buries the sync lines that actually say something.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    cfg = get_config()
    print("jarvis — kb serve\n" + format_describe(cfg) + "\n", flush=True)
    if cfg.sync_minutes < 1:
        print(f"✗ [kb] sync_minutes must be at least 1, got {cfg.sync_minutes}", file=sys.stderr)
        sys.exit(1)
    for label, folder in (("vault_path", cfg.vault_path), ("papers_dir", cfg.papers_dir)):
        if folder is not None and not folder.is_dir():
            # Not fatal: an unmounted drive comes back. Its sync just waits.
            log.warning("%s %s is not a folder — it will be skipped until it exists", label, folder)

    try:
        child = _start_chroma(cfg)
    except RuntimeError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        sys.exit(1)

    stop = threading.Event()

    def _request_stop(signum, frame):
        log.info("received signal %d, shutting down", signum)
        stop.set()

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    try:
        _wait_until_answering(child, cfg.server_port)
        log.info("index server on 127.0.0.1:%d (index: %s)", cfg.server_port, cfg.rag_dir)

        # Open the store now: an embedding-model mismatch should stop
        # start-up loudly, not fail every sync pass.
        from jarvis.kb.store import get_store

        get_store()
        write_status({"pid": os.getpid(), "started_at": _now()})
        log.info("syncing every %d min — Ctrl-C to stop", cfg.sync_minutes)

        while not stop.is_set():
            if child.poll() is not None:
                # Without its server nothing else can work; stop loudly rather
                # than keep syncing into a connection that is gone.
                log.error("chroma exited unexpectedly (code %s) — stopping", child.returncode)
                break
            run_sync()
            stop.wait(cfg.sync_minutes * 60)
    except Exception as exc:
        log.error("cannot serve: %s", exc)
        print(f"\n✗ kb serve stopped: {exc}\n", file=sys.stderr)
        _stop_chroma(child)
        sys.exit(1)
    _stop_chroma(child)
    log.info("stopped")


# ── `kb status` ───────────────────────────────────────────────────────────────


def print_status(status_file: Path = STATUS_FILE) -> None:
    """Report whether `kb serve` is running and how its last sync went."""
    status = read_status(status_file)
    if not status:
        print("kb serve has never run. Start it with: uv run kb serve")
        return
    pid = status.get("pid")
    try:
        os.kill(int(pid), 0)
        state = f"running (pid {pid})"
    except (OSError, TypeError, ValueError):
        state = "NOT RUNNING — start it with: uv run kb serve"
    print(f"kb serve:  {state}, started {status.get('started_at', '?')}")

    last_sync = status.get("last_sync")
    if not last_sync:
        print("Last sync: none yet")
    elif last_sync.get("ok"):
        changed = {k: v for k, v in last_sync.get("counts", {}).items() if v}
        summary = ", ".join(f"{k} {v}" for k, v in changed.items()) or "no changes"
        print(f"Last sync: {last_sync['at']} — {summary}")
    else:
        print(f"Last sync: {last_sync['at']} — ⚠️  FAILED: {last_sync.get('error', '?')}")

    log_file = LOG_DIR / "serve.log"
    print(f"\nLog: {log_file}")
    if log_file.exists():
        for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-5:]:
            print(f"  {line}")
