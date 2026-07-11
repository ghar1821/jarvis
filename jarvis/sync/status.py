"""`kb sync-status` implementation — reports jarvis-sync daemon health."""

import os
from pathlib import Path

from .daemon import STATUS_FILE, read_status


def cmd_sync_status() -> None:
    """Report jarvis-sync daemon liveness and per-job outcomes."""
    log_file = Path.home() / ".jarvis" / "logs" / "sync.log"

    if not STATUS_FILE.exists():
        print("Daemon has never run. Start it with: uv run jarvis-sync")
        return

    status = read_status()
    daemon = status.get("daemon", {})
    pid = daemon.get("pid")
    alive = False
    if pid:
        try:
            os.kill(int(pid), 0)
            alive = True
        except (OSError, ValueError):
            alive = False
    state = f"running (pid {pid})" if alive else "NOT RUNNING"
    print(f"Daemon:   {state}, started {daemon.get('started_at', '?')}")

    for job in ("digest", "vault_refresh", "pdf_ingest"):
        entry = status.get("jobs", {}).get(job)
        if not entry:
            print(f"{job:14s} never run")
            continue
        line = f"{job:14s} last run {entry.get('last_run', '?')}"
        if entry.get("last_success"):
            line += f" · last success {entry['last_success']}"
        if entry.get("last_error"):
            line += f"\n{'':14s} ⚠️  {entry['last_error']}"
        print(line)

    print(f"\nLog: {log_file}")
    if log_file.exists():
        tail = log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-5:]
        for line in tail:
            print(f"  {line}")
