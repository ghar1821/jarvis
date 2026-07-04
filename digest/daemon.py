"""
jarvis-sync — supervised background daemon.

One long-running process, kept alive by launchd (KeepAlive), owning three jobs:

  1. Weekly arXiv digest (APScheduler cron trigger). Runs missed while asleep
     fire on wake via misfire handling; runs missed while powered off are
     caught up at daemon start via a persistent last-success stamp.
  2. PDF inbox watcher (watchdog). New/changed PDFs in cfg.pdf_watch_dir are
     indexed full-text as public papers, with annotations. The folder is an
     inbox, not a mirror — removing a file never deletes its KB entry.
  3. Periodic Obsidian vault refresh — the existing hash-based incremental
     sync in refresh_vault(), on an interval.

Every job body catches its own exceptions and records the outcome in
~/.jarvis/state/sync_status.json (read by `kb sync-status`); one failing job
never takes the daemon down. Fatal setup problems (bad config, embedding-model
mismatch) exit non-zero so launchd restarts visibly.

The daemon does not manage other daemons: if the provider is local and
Ollama is down, the digest job fails with a pointer to the docs.
"""

import hashlib
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import Config, get_config, warn_if_config_readable

STATE_DIR = Path.home() / ".jarvis" / "state"
STATUS_FILE = STATE_DIR / "sync_status.json"

VALID_DIGEST_DAYS = {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}

log = logging.getLogger("jarvis-sync")


# ── Status file ────────────────────────────────────────────────────────────────


def read_status(status_file: Path = STATUS_FILE) -> dict:
    """Read the daemon status file; empty skeleton if missing or unreadable."""
    try:
        return json.loads(status_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"daemon": {}, "jobs": {}}


def write_status(status: dict, status_file: Path = STATUS_FILE) -> None:
    """Atomically write the daemon status file (temp file + os.replace)."""
    status_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = status_file.with_suffix(".json.tmp")
    tmp_file.write_text(json.dumps(status, indent=2), encoding="utf-8")
    os.replace(tmp_file, status_file)


def record_job_status(
    job: str,
    ok: bool,
    error: str = "",
    status_file: Path = STATUS_FILE,
) -> None:
    """Record one job run's outcome under jobs.<job> in the status file."""
    now = datetime.now(timezone.utc).isoformat()
    status = read_status(status_file)
    entry = status.setdefault("jobs", {}).setdefault(job, {})
    entry["last_run"] = now
    if ok:
        entry["last_success"] = now
        entry["last_error"] = ""
    else:
        entry["last_error"] = f"{now}: {error}"
    write_status(status, status_file)


# ── Config validation ──────────────────────────────────────────────────────────


def _validate_sync_config(cfg: Config) -> list[str]:
    """Return a list of fatal config problems (empty = valid)."""
    problems = []
    if cfg.pdf_watch_dir is not None and not cfg.pdf_watch_dir.is_dir():
        # Refuse to silently mkdir — a typo'd path would watch the wrong place.
        problems.append(
            f"pdf_watch_dir does not exist: {cfg.pdf_watch_dir} "
            "(create it, or remove the key to disable the watcher)"
        )
    if cfg.digest_day not in VALID_DIGEST_DAYS:
        problems.append(f"digest_day must be one of {sorted(VALID_DIGEST_DAYS)}, got {cfg.digest_day!r}")
    if not 0 <= cfg.digest_hour <= 23:
        problems.append(f"digest_hour must be 0-23, got {cfg.digest_hour}")
    if cfg.vault_refresh_minutes < 1:
        problems.append(f"vault_refresh_minutes must be >= 1, got {cfg.vault_refresh_minutes}")
    return problems


# ── Digest job ─────────────────────────────────────────────────────────────────


def digest_is_overdue(trigger, last_success: "datetime | None", now: datetime) -> bool:
    """
    True when a scheduled digest fire time passed since the last success —
    i.e. the machine was off (daemon not running) across a schedule boundary.
    On the very first start there is no baseline, so wait for the next slot
    rather than surprise-running immediately.
    """
    if last_success is None:
        return False
    next_after_last = trigger.get_next_fire_time(None, last_success)
    return next_after_last is not None and next_after_last <= now


def _require_local_llm(cfg: Config) -> None:
    """Fail the digest early, with clear guidance, if Ollama is down."""
    import requests

    from .errors import FetchError

    health_url = "http://localhost:11434/api/tags"
    try:
        response = requests.get(health_url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise FetchError(
            f"Ollama is not reachable at {health_url}: {exc}. "
            "Start it (the login-item app or `ollama serve`) or set "
            "provider = \"anthropic\"."
        ) from exc


def run_digest_job(status_file: Path = STATUS_FILE) -> bool:
    """Run the weekly digest; record the outcome. Never raises."""
    log.info("digest: starting")
    try:
        cfg = get_config()
        if cfg.provider != "anthropic":
            _require_local_llm(cfg)
        from .pipeline.run import main as run_digest

        run_digest()
        record_job_status("digest", ok=True, status_file=status_file)
        log.info("digest: done")
        return True
    except Exception as exc:
        log.exception("digest run failed")
        record_job_status("digest", ok=False, error=str(exc), status_file=status_file)
        return False


# ── Vault refresh job ──────────────────────────────────────────────────────────


def run_vault_refresh_job(status_file: Path = STATUS_FILE) -> None:
    """Incremental vault sync; record the outcome. Never raises."""
    try:
        from .kb.store import get_store, refresh_vault

        cfg = get_config()
        added, updated, deleted = refresh_vault(cfg.vault_path, get_store())
        if added + updated + deleted:
            log.info("vault refresh: +%d new, ~%d changed, -%d removed", added, updated, deleted)
        record_job_status("vault_refresh", ok=True, status_file=status_file)
    except Exception as exc:
        log.exception("vault refresh failed")
        record_job_status("vault_refresh", ok=False, error=str(exc), status_file=status_file)


# ── PDF inbox watcher ──────────────────────────────────────────────────────────


def wait_for_stable(
    path: Path,
    checks: int = 3,
    interval: float = 2.0,
    timeout: float = 600.0,
) -> bool:
    """
    Poll size+mtime until the file has been unchanged for `checks` consecutive
    polls — cloud-sync clients and slow copies write PDFs incrementally.
    Returns False on timeout or if the file disappears.
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


def _should_skip(path: Path) -> bool:
    """Ignore dotfiles, cloud placeholders, and lock/temp artifacts."""
    name = path.name
    return name.startswith(".") or name.startswith("~$") or name.endswith(".icloud")


def ingest_pdf(pdf_path: Path, store=None) -> str:
    """
    Index one inbox PDF as a public full-text paper (with annotations).
    Returns "added", "updated", or "skipped" (already indexed, unchanged).
    """
    from .errors import ConversionError
    from .kb.convert import pdf_to_markdown
    from .kb.store import add_annotations, add_figures, add_texts, delete_by_metadata, get_store

    s = store if store is not None else get_store()
    resolved = pdf_path.resolve()
    source = resolved.as_uri()
    file_hash = hashlib.sha256(resolved.read_bytes()).hexdigest()

    existing = s._collection.get(where={"source": {"$eq": source}}, include=["metadatas"])
    outcome = "added"
    if existing["ids"]:
        stored_hashes = {m.get("content_hash", "") for m in existing["metadatas"]}
        if file_hash in stored_hashes:
            return "skipped"
        # Bytes changed (e.g. new annotations saved into the file) — replace.
        delete_by_metadata("source", source, s)
        outcome = "updated"

    # Annotations first: a scanned PDF whose body can't convert still keeps
    # its highlights.
    add_annotations(
        resolved, doc_type="paper", visibility="public",
        source=source, title=resolved.stem, file_path=str(resolved), store=s,
    )
    # Caption figures with the configured provider. Inbox PDFs are always
    # public papers, so this never trips the private-note privacy guard.
    _caption_figures(resolved, source, resolved.stem, str(resolved), s)
    try:
        full_text = pdf_to_markdown(resolved)
    except ConversionError as exc:
        log.warning("inbox: %s", exc)
        return outcome
    add_texts(
        content=full_text,
        doc_type="paper",
        visibility="public",
        source=source,
        extra_metadata={
            "title": resolved.stem,
            "file_path": str(resolved),
            "content_hash": file_hash,
            "storage_mode": "full_text",
        },
        store=s,
    )
    return outcome


def _caption_figures(pdf_path: Path, source: str, title: str, file_path: str, store) -> None:
    """
    Caption a PDF's figures, building the provider lazily — only when the PDF
    actually has qualifying figures, so ordinary text-only inbox PDFs never pay
    for a provider construction. Never raises: a captioning failure must not
    take down an ingest.
    """
    from .config import get_config
    from .kb.images import extract_figures
    from .kb.store import add_figures

    cfg = get_config()
    if not cfg.figure_captions:
        return
    try:
        if not extract_figures(pdf_path, max_figures=1, min_pixels=cfg.figure_min_pixels):
            return  # no figures — skip provider construction entirely
        from .llm import make_provider

        provider = make_provider(cfg.provider)
        figure_ids = add_figures(
            pdf_path, doc_type="paper", visibility="public", source=source,
            provider_obj=provider, provider_str=cfg.provider,
            title=title, file_path=file_path, store=store,
        )
        if figure_ids:
            log.info("inbox: %s — %d figure(s) captioned", pdf_path.name, len(figure_ids))
    except Exception:
        log.exception("inbox: figure captioning failed for %s", pdf_path.name)


def scan_watch_dir(watch_dir: Path, work_queue: "queue.Queue[Path]") -> int:
    """Queue every PDF already sitting in the inbox (dedup makes this idempotent)."""
    queued = 0
    for pdf in sorted(watch_dir.glob("*.pdf")):
        if not _should_skip(pdf):
            work_queue.put(pdf)
            queued += 1
    return queued


def _make_pdf_event_handler(work_queue: "queue.Queue[Path]"):
    from watchdog.events import PatternMatchingEventHandler

    class PdfEventHandler(PatternMatchingEventHandler):
        """Queues new/renamed PDFs. Cloud clients write to a temp name then
        rename, so on_moved matters as much as on_created."""

        def __init__(self):
            super().__init__(patterns=["*.pdf"], ignore_directories=True)

        def on_created(self, event):
            self._enqueue(event.src_path)

        def on_moved(self, event):
            self._enqueue(event.dest_path)

        def _enqueue(self, raw_path: str):
            path = Path(raw_path)
            if not _should_skip(path):
                log.info("inbox: noticed %s", path.name)
                work_queue.put(path)

    return PdfEventHandler()


def _ingest_worker(
    work_queue: "queue.Queue[Path]",
    stop_event: threading.Event,
    status_file: Path = STATUS_FILE,
) -> None:
    """Serialises PDF ingestion — one conversion at a time, forever."""
    while not stop_event.is_set():
        try:
            pdf_path = work_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        try:
            if not pdf_path.exists() or not wait_for_stable(pdf_path):
                log.warning("inbox: %s never stabilised, skipping", pdf_path.name)
                continue
            outcome = ingest_pdf(pdf_path)
            log.info("inbox: %s — %s", pdf_path.name, outcome)
            record_job_status("pdf_ingest", ok=True, status_file=status_file)
        except Exception as exc:
            log.exception("inbox: failed to ingest %s", pdf_path.name)
            record_job_status("pdf_ingest", ok=False, error=f"{pdf_path.name}: {exc}", status_file=status_file)
        finally:
            work_queue.task_done()


# ── Scheduler ────────────────────────────────────────────────────────────────


def _build_scheduler(cfg: Config):
    """
    Build the blocking scheduler with the digest and vault-refresh jobs.

    No timezone argument is passed on purpose. APScheduler resolves the real
    system zone through its tzlocal dependency when none is given. The earlier
    code passed timezone="local", and that literal string was handed straight
    to ZoneInfo, which has no zone named "local" — so the daemon raised
    ZoneInfoNotFoundError at startup and launchd restarted it in a loop.
    Pulling construction into this helper also lets the tests build the
    scheduler without running the whole daemon.
    """
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger

    scheduler = BlockingScheduler()
    digest_trigger = CronTrigger(day_of_week=cfg.digest_day, hour=cfg.digest_hour)
    scheduler.add_job(
        run_digest_job,
        digest_trigger,
        id="digest",
        coalesce=True,
        misfire_grace_time=3600,
        max_instances=1,
    )
    scheduler.add_job(
        run_vault_refresh_job,
        IntervalTrigger(minutes=cfg.vault_refresh_minutes),
        id="vault_refresh",
        coalesce=True,
        max_instances=1,
    )
    return scheduler


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    warn_if_config_readable()
    cfg = get_config()
    problems = _validate_sync_config(cfg)
    if problems:
        for problem in problems:
            log.error("config: %s", problem)
        sys.exit(1)

    status = read_status()
    status["daemon"] = {
        "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    write_status(status)

    # Load the store (and embedding model) up front: the first inbox event
    # shouldn't stall for a model download, and an embedding-model mismatch
    # should kill the daemon loudly at startup, not mid-job.
    log.info("loading knowledge base (embedding model)...")
    from .kb.store import get_store

    get_store()
    log.info("knowledge base ready")

    scheduler = _build_scheduler(cfg)
    digest_trigger = scheduler.get_job("digest").trigger

    # Catch-up: if the Mac was powered off across the scheduled slot, the
    # cron trigger alone would silently wait a whole week.
    last_success_raw = read_status().get("jobs", {}).get("digest", {}).get("last_success")
    last_success = datetime.fromisoformat(last_success_raw) if last_success_raw else None
    if digest_is_overdue(digest_trigger, last_success, datetime.now(last_success.tzinfo if last_success else timezone.utc)):
        log.info("digest: overdue (missed while powered off) — running now")
        run_digest_job()

    # PDF inbox watcher + serialised ingest worker.
    stop_event = threading.Event()
    observer = None
    if cfg.pdf_watch_dir is not None:
        work_queue: "queue.Queue[Path]" = queue.Queue()
        worker = threading.Thread(
            target=_ingest_worker, args=(work_queue, stop_event), daemon=True
        )
        worker.start()

        queued = scan_watch_dir(cfg.pdf_watch_dir, work_queue)
        if queued:
            log.info("inbox: %d existing PDF(s) queued from startup sweep", queued)

        from watchdog.observers import Observer

        observer = Observer()
        observer.schedule(_make_pdf_event_handler(work_queue), str(cfg.pdf_watch_dir))
        observer.start()
        log.info("watching PDF inbox: %s", cfg.pdf_watch_dir)
    else:
        log.info("pdf_watch_dir not set — inbox watcher disabled")

    def _shutdown(signum, frame):
        log.info("received signal %d, shutting down", signum)
        stop_event.set()
        if observer is not None:
            observer.stop()
        scheduler.shutdown(wait=False)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    log.info(
        "scheduler starting (digest %s %02d:00, vault refresh every %d min)",
        cfg.digest_day, cfg.digest_hour, cfg.vault_refresh_minutes,
    )
    run_vault_refresh_job()  # sync once at startup rather than waiting a full interval
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        stop_event.set()
        if observer is not None:
            observer.stop()
            observer.join(timeout=5)
    log.info("stopped")


if __name__ == "__main__":
    main()
