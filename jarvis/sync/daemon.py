"""
jarvis-sync — supervised background daemon.

Run directly in a terminal with `uv run jarvis-sync` — it stays in the
foreground and owns four APScheduler jobs (everything is scheduled; there is
no filesystem-event watcher and no worker thread):

  1. Weekly arXiv digest (cron trigger). Runs missed while asleep fire on
     wake via misfire handling.
  2. Digest catch-up (every 6 hours, and once at startup). Re-checks the
     persistent last-success stamp against the cron schedule and runs the
     digest if a slot was missed while the machine was powered off — so a
     missed Monday no longer waits until the next Monday or a restart. A
     non-blocking lock keeps the cron job and the catch-up job from ever
     running the digest twice at once.
  3. Periodic PDF inbox scan (every pdf_watch_minutes). New/changed PDFs in
     cfg.pdf_watch_dir are indexed full-text as public papers, with
     annotations; byte-hash dedup makes the scan idempotent, so saving a
     highlight costs at most one re-ingest per interval rather than one per
     save. The folder is an inbox, not a mirror — removing a file never
     deletes its KB entry.
  4. Periodic Obsidian vault refresh — the existing hash-based incremental
     sync in refresh_vault(), on an interval.

Every job body catches its own exceptions and records the outcome in
~/.jarvis/state/sync_status.json (read by `kb sync-status`); one failing job
never takes the daemon down. Fatal setup problems (bad config, embedding-model
mismatch) exit non-zero (see the message printed to the log for why).

Logging goes to ~/.jarvis/logs/sync.log by default, plus stderr so a
foreground run shows the same messages live. Restart-on-crash is up to
whatever keeps the process running (your terminal, a process manager, etc.)
— the daemon itself makes no assumptions about that.

The daemon does not manage other daemons: if the provider is local and
Ollama is down, the digest job fails with a pointer to the docs.
"""

import hashlib
import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from jarvis.core.config import Config, get_config, warn_if_config_readable
from jarvis.core.llm import active_model

STATE_DIR = Path.home() / ".jarvis" / "state"
STATUS_FILE = STATE_DIR / "sync_status.json"
LOG_FILE = Path.home() / ".jarvis" / "logs" / "sync.log"

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
    if cfg.pdf_watch_minutes < 1:
        problems.append(f"pdf_watch_minutes must be >= 1, got {cfg.pdf_watch_minutes}")
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

    from jarvis.core.errors import FetchError

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


# The weekly cron job and the 6-hourly catch-up job are separate APScheduler
# ids, so max_instances=1 cannot stop them overlapping — this lock does.
_digest_run_lock = threading.Lock()


def run_digest_job(status_file: Path = STATUS_FILE) -> bool:
    """Run the weekly digest; record the outcome. Never raises."""
    if not _digest_run_lock.acquire(blocking=False):
        log.info("digest: another digest run is already in progress — skipping")
        return False
    try:
        log.info("digest: starting")
        try:
            cfg = get_config()
            if cfg.provider != "anthropic":
                _require_local_llm(cfg)
            log.info("digest: using %s (model %s)", cfg.provider, active_model(cfg))
            from jarvis.digest.pipeline.run import main as run_digest

            run_digest()
            record_job_status("digest", ok=True, status_file=status_file)
            log.info("digest: done")
            return True
        except Exception as exc:
            log.exception("digest run failed")
            record_job_status("digest", ok=False, error=str(exc), status_file=status_file)
            return False
    finally:
        _digest_run_lock.release()


def run_digest_catchup_job(trigger, status_file: Path = STATUS_FILE) -> bool:
    """
    Run the digest now if a scheduled slot passed since the last success.

    Scheduled every 6 hours (and called once at daemon start), so a run
    missed while the machine was powered off happens within hours instead of
    waiting for the next weekly slot or a daemon restart. Re-reads the status
    file on every call — the stamp moves whenever any digest run succeeds.
    Returns whether the digest was actually fired.
    """
    last_success_raw = (
        read_status(status_file).get("jobs", {}).get("digest", {}).get("last_success")
    )
    last_success = datetime.fromisoformat(last_success_raw) if last_success_raw else None
    now = datetime.now(last_success.tzinfo if last_success else timezone.utc)
    if not digest_is_overdue(trigger, last_success, now):
        return False
    log.info("digest: overdue (missed a scheduled slot) — running now")
    run_digest_job(status_file=status_file)
    return True


# ── Vault refresh job ──────────────────────────────────────────────────────────


def run_vault_refresh_job(status_file: Path = STATUS_FILE) -> None:
    """Incremental vault sync; record the outcome. Never raises."""
    try:
        from jarvis.kb.store import get_store, refresh_vault

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

    Title/authors/DOI are auto-inferred from the PDF's first pages — inbox
    PDFs are always public papers, so cloud inference is always allowed here
    (no privacy guard can fire). A provider is now built unconditionally for
    this (previously only lazily for figure captioning); _caption_figures
    reuses it so make_provider is never called twice per ingest.
    """
    from jarvis.core.errors import ConversionError
    from jarvis.kb.convert import pdf_to_markdown
    from jarvis.kb.metadata import resolve_pdf_metadata
    from jarvis.kb.store import add_annotations, add_figures, add_texts, delete_by_metadata, get_store
    from jarvis.core.llm import make_provider

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

    cfg = get_config()
    provider = make_provider(cfg.provider)
    log.info(
        "inbox: inferring metadata for %s using %s (model %s)",
        pdf_path.name, cfg.provider, active_model(cfg),
    )
    meta = resolve_pdf_metadata(resolved, provider)
    title = meta["title"] or resolved.stem
    authors, doi = meta["authors"], meta["doi"]

    # Annotations first: a scanned PDF whose body can't convert still keeps
    # its highlights.
    add_annotations(
        resolved, doc_type="paper", visibility="public",
        source=source, title=title, file_path=str(resolved), store=s,
    )
    # Caption figures, reusing the provider built above. Inbox PDFs are
    # always public papers, so this never trips the private-note privacy guard.
    _caption_figures(resolved, source, title, str(resolved), s, provider_obj=provider)
    try:
        full_text = pdf_to_markdown(resolved)
    except ConversionError as exc:
        log.warning("inbox: %s", exc)
        return outcome
    extra_metadata = {
        "title": title,
        "file_path": str(resolved),
        "content_hash": file_hash,
        "storage_mode": "full_text",
        "authors": authors,
        "doi": doi,
    }
    add_texts(
        content=full_text,
        doc_type="paper",
        visibility="public",
        source=source,
        extra_metadata=extra_metadata,
        embed_header=(f"{title} — {authors}" if authors else title),
        store=s,
    )
    log.info(
        'inbox: %s "%s" — authors: %s; doi: %s; file: %s',
        outcome, title, authors or "unknown", doi or "unknown", pdf_path.name,
    )
    return outcome


def _caption_figures(
    pdf_path: Path, source: str, title: str, file_path: str, store, provider_obj=None,
) -> None:
    """
    Caption a PDF's figures. When provider_obj is given, reuse it instead of
    constructing a fresh one (ingest_pdf now always builds a provider for
    metadata inference, so this avoids a second make_provider() call). When
    not given, build one lazily — only when the PDF actually has qualifying
    figures, so ordinary text-only callers never pay for a provider
    construction. Never raises: a captioning failure must not take down an ingest.
    """
    from jarvis.core.config import get_config
    from jarvis.kb.images import extract_figures
    from jarvis.kb.store import add_figures

    cfg = get_config()
    if not cfg.figure_captions:
        return
    try:
        if not extract_figures(pdf_path, max_figures=1, min_pixels=cfg.figure_min_pixels):
            return  # no figures — skip provider construction entirely
        provider = provider_obj
        if provider is None:
            from jarvis.core.llm import make_provider

            provider = make_provider(cfg.provider)
        log.info(
            "inbox: captioning figures for %s using %s (model %s)",
            pdf_path.name, cfg.provider, active_model(cfg),
        )
        figure_ids = add_figures(
            pdf_path, doc_type="paper", visibility="public", source=source,
            provider_obj=provider, provider_str=cfg.provider,
            title=title, file_path=file_path, store=store,
        )
        if figure_ids:
            log.info("inbox: %s — %d figure(s) captioned", pdf_path.name, len(figure_ids))
    except Exception:
        log.exception("inbox: figure captioning failed for %s", pdf_path.name)


def scan_watch_dir(watch_dir: Path) -> list[Path]:
    """The PDFs currently sitting in the inbox, sorted, minus artifacts."""
    return [pdf for pdf in sorted(watch_dir.glob("*.pdf")) if not _should_skip(pdf)]


def run_pdf_scan_job(
    watch_dir: Path | None = None,
    store=None,
    status_file: Path = STATUS_FILE,
) -> None:
    """
    One periodic inbox sweep: ingest every PDF in the watch dir, serially.

    The byte-hash dedup in ingest_pdf makes the sweep idempotent — unchanged
    files are skipped, so a scan of a quiet inbox costs no LLM tokens. Each
    file is checked for stability first (cloud-sync clients write PDFs
    incrementally); a file still changing is left for the next scan rather
    than waited on for long. Per-file failures are recorded and don't stop
    the rest of the sweep. No-op when no watch dir is configured.

    watch_dir and store default from config/get_store so the daemon needs no
    arguments while tests can inject isolated ones.
    """
    if watch_dir is None:
        watch_dir = get_config().pdf_watch_dir
    if watch_dir is None:
        return
    if store is None:
        from jarvis.kb.store import get_store

        store = get_store()

    for pdf_path in scan_watch_dir(watch_dir):
        if not wait_for_stable(pdf_path, checks=2, interval=1.0, timeout=30):
            log.warning("inbox: %s still changing — leaving it for the next scan", pdf_path.name)
            continue
        try:
            outcome = ingest_pdf(pdf_path, store)
            if outcome != "skipped":
                log.info("inbox: %s — %s", pdf_path.name, outcome)
            record_job_status("pdf_ingest", ok=True, status_file=status_file)
        except Exception as exc:
            log.exception("inbox: failed to ingest %s", pdf_path.name)
            record_job_status(
                "pdf_ingest", ok=False, error=f"{pdf_path.name}: {exc}", status_file=status_file
            )


# ── Scheduler ────────────────────────────────────────────────────────────────


def _build_scheduler(cfg: Config):
    """
    Build the blocking scheduler with all four jobs: digest (weekly cron),
    digest catch-up (6-hourly overdue re-check), vault refresh (interval),
    and — only when a watch dir is configured — the PDF inbox scan (interval).

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
    # Overdue re-check every 6 hours, so a digest slot missed while powered
    # off fires within hours of the machine coming back — not at the next
    # weekly slot or the next daemon restart. The digest lock keeps this job
    # and the cron job from overlapping (separate ids, so max_instances=1
    # alone would not).
    scheduler.add_job(
        run_digest_catchup_job,
        IntervalTrigger(hours=6),
        args=[digest_trigger],
        id="digest_catchup",
        coalesce=True,
        max_instances=1,
    )
    scheduler.add_job(
        run_vault_refresh_job,
        IntervalTrigger(minutes=cfg.vault_refresh_minutes),
        id="vault_refresh",
        coalesce=True,
        max_instances=1,
    )
    if cfg.pdf_watch_dir is not None:
        scheduler.add_job(
            run_pdf_scan_job,
            IntervalTrigger(minutes=cfg.pdf_watch_minutes),
            id="pdf_scan",
            coalesce=True,
            max_instances=1,
        )
    return scheduler


def _log_next_run_times(scheduler) -> None:
    """
    Log one line per job stating when it will next fire. job.next_run_time
    is None until BlockingScheduler.start() actually begins running the
    loop, so ask each job's trigger directly for its first fire time —
    this is what answers "when will it run next" right from startup.
    """
    now = datetime.now(timezone.utc).astimezone()
    for job in scheduler.get_jobs():
        next_fire = job.trigger.get_next_fire_time(None, now)
        log.info("job %s: next run at %s", job.id, next_fire)


def _log_job_outcome(scheduler, event) -> None:
    """Log a job's completion (or failure) and its next scheduled run."""
    job = scheduler.get_job(event.job_id)
    next_fire = (
        job.trigger.get_next_fire_time(None, datetime.now(timezone.utc).astimezone())
        if job is not None
        else None
    )
    if event.exception:
        log.error(
            "job %s failed: %s — next run at %s", event.job_id, event.exception, next_fire
        )
    else:
        log.info("job %s finished — next run at %s", event.job_id, next_fire)


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        # File first so it's always captured; stderr too so a foreground
        # `uv run jarvis-sync` shows the same messages live.
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stderr)],
    )
    # APScheduler's own module logger is unconfigured and propagates to root
    # at INFO, which floods the log with "Added job ... to job store default"
    # noise on every startup. Our own per-job next-run lines below cover the
    # useful information, so quiet APScheduler down to warnings and worse.
    logging.getLogger("apscheduler").setLevel(logging.WARNING)

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

    log.info(
        "LLM provider: %s (model %s) · embedding model: %s",
        cfg.provider, active_model(cfg), cfg.embed_model,
    )

    # Load the store (and embedding model) up front: the first inbox event
    # shouldn't stall for a model download, and an embedding-model mismatch
    # should kill the daemon loudly at startup, not mid-job.
    log.info("loading knowledge base (embedding model)...")
    from jarvis.kb.store import get_store

    get_store()
    log.info("knowledge base ready")

    from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED

    scheduler = _build_scheduler(cfg)
    digest_trigger = scheduler.get_job("digest").trigger

    _log_next_run_times(scheduler)
    scheduler.add_listener(
        lambda event: _log_job_outcome(scheduler, event), EVENT_JOB_EXECUTED | EVENT_JOB_ERROR
    )

    # Catch-up at start: if the Mac was powered off across the scheduled
    # slot, the cron trigger alone would silently wait a whole week. The same
    # function also runs every 6 hours from the scheduler — one code path.
    run_digest_catchup_job(digest_trigger)

    if cfg.pdf_watch_dir is not None:
        log.info(
            "scanning PDF inbox %s every %d min", cfg.pdf_watch_dir, cfg.pdf_watch_minutes
        )
    else:
        log.info("pdf_watch_dir not set — inbox scan disabled")

    def _shutdown(signum, frame):
        log.info("received signal %d, shutting down", signum)
        scheduler.shutdown(wait=False)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    log.info(
        "scheduler starting (digest %s %02d:00, vault refresh every %d min)",
        cfg.digest_day, cfg.digest_hour, cfg.vault_refresh_minutes,
    )
    # Run each periodic job once at startup rather than waiting a full interval.
    run_vault_refresh_job()
    run_pdf_scan_job()
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass
    log.info("stopped")


if __name__ == "__main__":
    main()
