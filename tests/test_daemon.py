"""
Tests for jarvis/sync/daemon.py — the jarvis-sync background daemon.

The daemon's scheduler loop is process-runtime plumbing; what's tested here
are the pure decision functions and job bodies: catch-up detection (including
the scheduled catch-up job and the digest double-fire lock), file-stability
polling, the periodic inbox scan and its ingestion/dedup, config validation,
and the status file. PDF fixtures are generated in-test (cheap, real
conversion — no mocking needed since marker-pdf is gone).
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pymupdf
import pytest
from apscheduler.triggers.cron import CronTrigger

import jarvis.sync.daemon as daemon_module
from jarvis.core.config import Config
from jarvis.sync.daemon import (
    _build_scheduler,
    _validate_sync_config,
    digest_is_overdue,
    ingest_pdf,
    read_status,
    record_job_status,
    run_digest_catchup_job,
    run_digest_job,
    run_pdf_scan_job,
    scan_watch_dir,
    wait_for_stable,
    write_status,
)

WEEKLY_MON_2AM = CronTrigger(day_of_week="mon", hour=2, timezone="UTC")


def _make_pdf(path: Path, text: str) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    doc.save(path)
    doc.close()
    return path


# ── digest_is_overdue ──────────────────────────────────────────────────────────

def test_digest_not_overdue_on_first_ever_start():
    """
    With no prior success there is no baseline — wait for the next scheduled
    slot instead of surprise-running at first launch.
    """
    now = datetime(2026, 7, 3, 12, 0, tzinfo=timezone.utc)
    assert digest_is_overdue(WEEKLY_MON_2AM, None, now) is False


def test_digest_not_overdue_within_the_week():
    """
    A success 2 days ago with a weekly schedule means the next fire time is
    still in the future — not overdue.

    Input:  last success Tue 2026-06-30 03:00, now Thu 2026-07-02
    Expected output: False (next fire is Mon 2026-07-06 02:00)
    """
    last_success = datetime(2026, 6, 30, 3, 0, tzinfo=timezone.utc)
    now = datetime(2026, 7, 2, 12, 0, tzinfo=timezone.utc)
    assert digest_is_overdue(WEEKLY_MON_2AM, last_success, now) is False


def test_digest_overdue_after_missed_slot():
    """
    A success 8+ days ago means a Monday slot passed while the daemon was
    down — overdue.

    Input:  last success Mon 2026-06-22 03:00, now Fri 2026-07-03
    Expected output: True (Mon 2026-06-29 02:00 was missed)
    """
    last_success = datetime(2026, 6, 22, 3, 0, tzinfo=timezone.utc)
    now = datetime(2026, 7, 3, 12, 0, tzinfo=timezone.utc)
    assert digest_is_overdue(WEEKLY_MON_2AM, last_success, now) is True


def test_digest_boundary_not_overdue_just_before_fire_time():
    """
    One minute before the next scheduled slot is not overdue.

    Input:  last success Mon 2026-06-29 02:30, now Mon 2026-07-06 01:59
    Expected output: False
    """
    last_success = datetime(2026, 6, 29, 2, 30, tzinfo=timezone.utc)
    now = datetime(2026, 7, 6, 1, 59, tzinfo=timezone.utc)
    assert digest_is_overdue(WEEKLY_MON_2AM, last_success, now) is False


# ── wait_for_stable ────────────────────────────────────────────────────────────

def test_wait_for_stable_returns_true_for_settled_file(tmp_path):
    """A file that stops changing is declared stable after `checks` polls."""
    f = tmp_path / "settled.pdf"
    f.write_bytes(b"final content")
    assert wait_for_stable(f, checks=2, interval=0.01, timeout=5.0) is True


def test_wait_for_stable_times_out_on_growing_file(tmp_path, monkeypatch):
    """
    A file that keeps growing between polls never reaches the stability
    threshold and times out.
    """
    f = tmp_path / "growing.pdf"
    f.write_bytes(b"x")

    original_stat = Path.stat
    counter = {"n": 0}

    def growing_stat(self, **kwargs):
        result = original_stat(self, **kwargs)
        if self == f:
            counter["n"] += 1
            f.write_bytes(b"x" * (counter["n"] + 1))
        return result

    monkeypatch.setattr(Path, "stat", growing_stat)
    assert wait_for_stable(f, checks=3, interval=0.01, timeout=0.2) is False


def test_wait_for_stable_returns_false_for_vanished_file(tmp_path):
    """A file deleted mid-wait returns False instead of raising."""
    f = tmp_path / "gone.pdf"
    assert wait_for_stable(f, checks=2, interval=0.01, timeout=0.1) is False


# ── ingest_pdf ─────────────────────────────────────────────────────────────────

class _StubMetadataProvider:
    """
    Every ingest now calls make_provider(...).complete(...) once for metadata
    inference, even for a PDF with no figures — this stub keeps that call off
    a real Ollama/Anthropic endpoint. An empty JSON response degrades
    inference to {} (title falls back to the filename stem), preserving the
    pre-metadata-inference assertions unchanged.
    """

    def complete(self, messages, max_tokens=300, context_length=None):
        return "{}"


def test_ingest_pdf_adds_then_skips_then_updates(store, tmp_path, monkeypatch):
    """
    First ingest indexes the PDF as a public full-text paper; a second call
    with unchanged bytes is a no-op; changing the file (as saving new
    annotations does) replaces the old chunks.
    """
    monkeypatch.setattr("jarvis.core.llm.make_provider", lambda provider_str: _StubMetadataProvider())
    pdf = _make_pdf(tmp_path / "inbox_paper.pdf", "A study of reproducible pipelines.")

    assert ingest_pdf(pdf, store) == "added"
    stored = store._collection.get(
        where={"source": {"$eq": pdf.resolve().as_uri()}}, include=["metadatas"]
    )
    assert stored["ids"]
    assert all(m["doc_type"] == "paper" for m in stored["metadatas"])
    assert all(m["visibility"] == "public" for m in stored["metadatas"])
    body_meta = [m for m in stored["metadatas"] if m.get("content_hash")]
    assert body_meta and body_meta[0]["storage_mode"] == "full_text"

    assert ingest_pdf(pdf, store) == "skipped"

    # Simulate the user saving a highlight into the file: bytes change.
    doc = pymupdf.open(pdf)
    page = doc[0]
    page.add_highlight_annot(page.search_for("reproducible pipelines", quads=True))
    doc.saveIncr()
    doc.close()

    assert ingest_pdf(pdf, store) == "updated"
    highlights = store._collection.get(
        where={"annotation_kind": {"$eq": "highlight"}}, include=["documents"]
    )
    assert any("reproducible pipelines" in d for d in highlights["documents"])


def test_ingest_pdf_populates_inferred_title_and_authors(store, tmp_path, monkeypatch):
    """
    A provider that actually returns metadata gets its title/authors stored
    on the body chunk — inbox PDFs are always public papers, so inference is
    never blocked by the privacy guard.
    """
    class _RealisticStubProvider:
        def complete(self, messages, max_tokens=300, context_length=None):
            return '{"title": "Reproducible Pipelines at Scale", "authors": "Ada Lovelace", "doi": ""}'

    monkeypatch.setattr("jarvis.core.llm.make_provider", lambda provider_str: _RealisticStubProvider())
    pdf = _make_pdf(tmp_path / "inferred_paper.pdf", "A study of reproducible pipelines.")

    assert ingest_pdf(pdf, store) == "added"
    stored = store._collection.get(
        where={"source": {"$eq": pdf.resolve().as_uri()}}, include=["metadatas"]
    )
    body_meta = [m for m in stored["metadatas"] if m.get("content_hash")]
    assert body_meta
    assert body_meta[0]["title"] == "Reproducible Pipelines at Scale"
    assert body_meta[0]["authors"] == "Ada Lovelace"


def test_ingest_pdf_logs_metadata_inference_model(store, tmp_path, monkeypatch, caplog):
    """
    ingest_pdf logs which provider+model performed metadata inference — once
    per add/update, not on a skipped (unchanged) file, since no LLM call
    happens on the skip path.
    """
    monkeypatch.setattr("jarvis.core.llm.make_provider", lambda provider_str: _StubMetadataProvider())
    monkeypatch.setattr(
        daemon_module, "get_config", lambda: Config(provider="anthropic", anthropic_model="claude-test-9")
    )
    pdf = _make_pdf(tmp_path / "logged_paper.pdf", "A study of reproducible pipelines.")

    with caplog.at_level("INFO", logger="jarvis-sync"):
        assert ingest_pdf(pdf, store) == "added"

    matching = [r for r in caplog.records if "inferring metadata" in r.message]
    assert len(matching) == 1
    assert "anthropic" in matching[0].message
    assert "claude-test-9" in matching[0].message

    caplog.clear()
    with caplog.at_level("INFO", logger="jarvis-sync"):
        assert ingest_pdf(pdf, store) == "skipped"

    assert [r for r in caplog.records if "inferring metadata" in r.message] == []


def test_ingest_pdf_logs_stored_metadata_on_add(store, tmp_path, monkeypatch, caplog):
    """
    After a successful add, ingest_pdf logs one line with the stored
    title/authors/doi and the source filename, so the sync log shows exactly
    what metadata ended up in the KB without a separate lookup.
    """
    class _RealisticStubProvider:
        def complete(self, messages, max_tokens=300, context_length=None):
            return '{"title": "Reproducible Pipelines at Scale", "authors": "Ada Lovelace", "doi": "10.1/repro"}'

    monkeypatch.setattr("jarvis.core.llm.make_provider", lambda provider_str: _RealisticStubProvider())
    pdf = _make_pdf(tmp_path / "metadata_logged.pdf", "A study of reproducible pipelines.")

    with caplog.at_level("INFO", logger="jarvis-sync"):
        assert ingest_pdf(pdf, store) == "added"

    matching = [r for r in caplog.records if "Reproducible Pipelines at Scale" in r.message]
    assert len(matching) == 1
    assert "Ada Lovelace" in matching[0].message
    assert "10.1/repro" in matching[0].message
    assert pdf.name in matching[0].message


def test_scan_watch_dir_lists_pdfs_and_skips_artifacts(tmp_path):
    """
    The inbox scan returns the real PDFs, sorted, and ignores dotfiles,
    cloud placeholders, and non-PDF files.
    """
    _make_pdf(tmp_path / "two.pdf", "two")
    _make_pdf(tmp_path / "one.pdf", "one")
    (tmp_path / ".hidden.pdf").write_bytes(b"x")
    (tmp_path / "syncing.pdf.icloud").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("not a pdf")

    scanned = scan_watch_dir(tmp_path)
    assert [p.name for p in scanned] == ["one.pdf", "two.pdf"]


# ── run_pdf_scan_job ───────────────────────────────────────────────────────────

@pytest.fixture
def fast_stability(monkeypatch):
    """
    Skip the real 2-second stability wait — these tests write complete PDFs
    up front, so polling their mtime twice adds nothing but wall-clock time.
    """
    monkeypatch.setattr(daemon_module, "wait_for_stable", lambda *a, **kw: True)


def test_run_pdf_scan_job_ingests_then_skips_and_records_status(
    store, tmp_path, monkeypatch, fast_stability
):
    """
    First scan ingests the inbox PDF and records a pdf_ingest success; a
    second scan of the unchanged inbox is a token-free no-op (hash dedup)
    that still records a healthy status.
    """
    monkeypatch.setattr("jarvis.core.llm.make_provider", lambda provider_str: _StubMetadataProvider())
    watch_dir = tmp_path / "inbox"
    watch_dir.mkdir()
    status_file = tmp_path / "state" / "sync_status.json"
    pdf = _make_pdf(watch_dir / "scanned.pdf", "A paper about periodic inbox scans.")

    run_pdf_scan_job(watch_dir=watch_dir, store=store, status_file=status_file)
    stored = store._collection.get(
        where={"source": {"$eq": pdf.resolve().as_uri()}}, include=["metadatas"]
    )
    assert stored["ids"], "the scanned PDF must be indexed"
    job_status = read_status(status_file)["jobs"]["pdf_ingest"]
    assert job_status["last_success"]
    assert job_status["last_error"] == ""

    # Second sweep: unchanged bytes → skipped, chunk count unchanged.
    chunk_count = len(stored["ids"])
    run_pdf_scan_job(watch_dir=watch_dir, store=store, status_file=status_file)
    stored_again = store._collection.get(
        where={"source": {"$eq": pdf.resolve().as_uri()}}, include=[]
    )
    assert len(stored_again["ids"]) == chunk_count


def test_run_pdf_scan_job_records_failure_and_continues(
    store, tmp_path, monkeypatch, fast_stability
):
    """A per-file ingest failure is recorded in the status file, not raised."""
    def exploding_ingest(path, store=None):
        raise RuntimeError("conversion exploded")

    monkeypatch.setattr(daemon_module, "ingest_pdf", exploding_ingest)
    watch_dir = tmp_path / "inbox"
    watch_dir.mkdir()
    status_file = tmp_path / "state" / "sync_status.json"
    _make_pdf(watch_dir / "bad.pdf", "will fail")

    run_pdf_scan_job(watch_dir=watch_dir, store=store, status_file=status_file)

    job_status = read_status(status_file)["jobs"]["pdf_ingest"]
    assert "conversion exploded" in job_status["last_error"]


def test_run_pdf_scan_job_skips_unstable_file_until_next_cycle(
    store, tmp_path, monkeypatch
):
    """
    A file still being written (never stabilises) is skipped this cycle
    without being indexed and without recording a failure.
    """
    monkeypatch.setattr(daemon_module, "wait_for_stable", lambda *a, **kw: False)
    watch_dir = tmp_path / "inbox"
    watch_dir.mkdir()
    status_file = tmp_path / "state" / "sync_status.json"
    pdf = _make_pdf(watch_dir / "mid-copy.pdf", "still syncing")

    run_pdf_scan_job(watch_dir=watch_dir, store=store, status_file=status_file)

    stored = store._collection.get(where={"source": {"$eq": pdf.resolve().as_uri()}}, include=[])
    assert stored["ids"] == []
    assert read_status(status_file)["jobs"] == {}


def test_run_pdf_scan_job_noop_without_watch_dir(store, tmp_path, monkeypatch):
    """With no watch dir configured the job returns without touching anything."""
    monkeypatch.setattr(daemon_module, "get_config", lambda: Config(pdf_watch_dir=None))
    status_file = tmp_path / "state" / "sync_status.json"
    run_pdf_scan_job(store=store, status_file=status_file)
    assert not status_file.exists()


# ── run_digest_catchup_job ─────────────────────────────────────────────────────

@pytest.fixture
def digest_recorder(monkeypatch):
    """Record run_digest_job invocations instead of running a real digest."""
    calls = []

    def fake_run_digest_job(status_file=None):
        calls.append(status_file)
        return True

    monkeypatch.setattr(daemon_module, "run_digest_job", fake_run_digest_job)
    return calls


def _status_file_with_last_success(tmp_path: Path, last_success: datetime) -> Path:
    status_file = tmp_path / "state" / "sync_status.json"
    write_status(
        {"daemon": {}, "jobs": {"digest": {"last_success": last_success.isoformat()}}},
        status_file,
    )
    return status_file


def test_catchup_does_not_fire_after_fresh_success(tmp_path, digest_recorder):
    """A success within the current week means no slot was missed."""
    status_file = _status_file_with_last_success(tmp_path, datetime.now(timezone.utc))
    fired = run_digest_catchup_job(WEEKLY_MON_2AM, status_file=status_file)
    assert fired is False
    assert digest_recorder == []


def test_catchup_fires_when_last_success_is_stale(tmp_path, digest_recorder):
    """
    A success 8+ days ago means a weekly slot passed while the daemon was
    down — the catch-up job must run the digest now.
    """
    stale = datetime.now(timezone.utc) - timedelta(days=8)
    status_file = _status_file_with_last_success(tmp_path, stale)
    fired = run_digest_catchup_job(WEEKLY_MON_2AM, status_file=status_file)
    assert fired is True
    assert digest_recorder == [status_file]


def test_catchup_does_not_fire_without_baseline(tmp_path, digest_recorder):
    """
    No recorded success (first ever start) → wait for the next scheduled
    slot instead of surprise-running immediately.
    """
    missing_status = tmp_path / "state" / "sync_status.json"
    fired = run_digest_catchup_job(WEEKLY_MON_2AM, status_file=missing_status)
    assert fired is False
    assert digest_recorder == []


def test_catchup_rereads_status_file_rather_than_caching(tmp_path, digest_recorder):
    """
    run_digest_catchup_job must read the status file fresh on every call, not
    cache the last_success it saw the first time. Drive it twice against the
    SAME status-file path: first with a stale last_success (fires), then
    overwrite the file with a fresh last_success and call again (must not
    fire) — proving the second call actually re-read the file instead of
    reusing an in-memory value from the first call.
    """
    stale = datetime.now(timezone.utc) - timedelta(days=8)
    status_file = _status_file_with_last_success(tmp_path, stale)

    first_fired = run_digest_catchup_job(WEEKLY_MON_2AM, status_file=status_file)
    assert first_fired is True
    assert digest_recorder == [status_file]

    # Simulate the digest run's own status write (the real run_digest_job is
    # stubbed here, so we write the fresh timestamp ourselves) and call again
    # against the same path.
    _status_file_with_last_success(tmp_path, datetime.now(timezone.utc))
    second_fired = run_digest_catchup_job(WEEKLY_MON_2AM, status_file=status_file)
    assert second_fired is False
    assert digest_recorder == [status_file]  # unchanged — no second fire


def test_run_digest_job_returns_early_when_lock_already_held(monkeypatch):
    """
    The cron job and the catch-up job are separate APScheduler ids, so
    max_instances=1 can't stop them overlapping — the module lock must.
    While one digest run holds the lock, a second call returns False without
    doing any work (get_config would be its first action).
    """
    def config_must_not_be_read():
        raise AssertionError("run_digest_job must not proceed while the lock is held")

    monkeypatch.setattr(daemon_module, "get_config", config_must_not_be_read)

    acquired = daemon_module._digest_run_lock.acquire(blocking=False)
    assert acquired, "test setup: the digest lock should have been free"
    try:
        assert run_digest_job() is False
    finally:
        daemon_module._digest_run_lock.release()


def test_run_digest_job_releases_lock_so_a_second_call_runs(tmp_path, monkeypatch):
    """
    The `finally: _digest_run_lock.release()` in run_digest_job must fire on
    the normal success path, not just on exceptions — otherwise the very
    first successful digest run would wedge every later run (cron, catch-up,
    manual) behind a lock nobody ever frees. Run the job to completion once
    with the pipeline stubbed out, then confirm a second call also runs
    (not skipped) by checking the stubbed pipeline main was invoked twice.
    """
    calls = []

    def fake_pipeline_main():
        calls.append(1)

    monkeypatch.setattr(daemon_module, "get_config", lambda: Config(provider="anthropic"))
    monkeypatch.setattr("jarvis.digest.pipeline.run.main", fake_pipeline_main)
    status_file = tmp_path / "state" / "sync_status.json"

    assert run_digest_job(status_file=status_file) is True
    assert run_digest_job(status_file=status_file) is True
    assert calls == [1, 1], "pipeline main must run once per call — lock was released between them"


def test_run_digest_job_logs_provider_and_model(tmp_path, monkeypatch, caplog):
    """
    run_digest_job logs which provider+model the digest will use, once per
    invocation, before handing off to the pipeline — so `jarvis-sync.log`
    always shows which model produced a given digest without re-reading config.
    """
    monkeypatch.setattr(
        daemon_module, "get_config", lambda: Config(provider="anthropic", anthropic_model="claude-test-9")
    )
    monkeypatch.setattr("jarvis.digest.pipeline.run.main", lambda: None)
    status_file = tmp_path / "state" / "sync_status.json"

    with caplog.at_level("INFO", logger="jarvis-sync"):
        assert run_digest_job(status_file=status_file) is True

    matching = [r for r in caplog.records if "digest: using" in r.message]
    assert len(matching) == 1
    assert "anthropic" in matching[0].message
    assert "claude-test-9" in matching[0].message


# ── config validation ──────────────────────────────────────────────────────────

def test_validate_sync_config_accepts_defaults(tmp_path):
    """Defaults (watcher disabled) are valid; an existing watch dir is valid."""
    assert _validate_sync_config(Config()) == []
    assert _validate_sync_config(Config(pdf_watch_dir=tmp_path)) == []


def test_validate_sync_config_rejects_bad_values(tmp_path):
    """
    Nonexistent watch dir, bad day token, out-of-range hour, and sub-minute
    refresh interval are each reported.
    """
    cfg = Config(
        pdf_watch_dir=tmp_path / "no-such-dir",
        digest_day="funday",
        digest_hour=99,
        vault_refresh_minutes=0,
        pdf_watch_minutes=0,
    )
    problems = _validate_sync_config(cfg)
    assert len(problems) == 5
    assert any("pdf_watch_dir" in p for p in problems)
    assert any("digest_day" in p for p in problems)
    assert any("digest_hour" in p for p in problems)
    assert any("vault_refresh_minutes" in p for p in problems)
    assert any("pdf_watch_minutes" in p for p in problems)


# ── scheduler construction ───────────────────────────────────────────────────

def test_build_scheduler_holds_core_jobs_without_watch_dir():
    """
    Regression for the launchd crash-loop: BlockingScheduler(timezone="local")
    handed the literal string "local" to ZoneInfo and raised at construction.
    _build_scheduler must build cleanly and register the digest, catch-up,
    and vault-refresh jobs; the pdf_scan job only exists when a watch dir is
    configured. Its timezone must be a real zone object, not the string "local".
    """
    from apscheduler.schedulers.blocking import BlockingScheduler

    # Construction alone reproduced the original crash, so nothing needs to
    # start the scheduler here.
    scheduler = _build_scheduler(Config())
    assert isinstance(scheduler, BlockingScheduler)
    assert scheduler.get_job("digest") is not None
    assert scheduler.get_job("digest_catchup") is not None
    assert scheduler.get_job("vault_refresh") is not None
    assert scheduler.get_job("pdf_scan") is None  # no watch dir configured
    # The bug was a bare string; a resolved zone is never a str.
    assert not isinstance(scheduler.timezone, str)


def test_build_scheduler_adds_pdf_scan_when_watch_dir_set(tmp_path):
    """A configured watch dir adds the periodic pdf_scan job."""
    scheduler = _build_scheduler(Config(pdf_watch_dir=tmp_path))
    assert scheduler.get_job("pdf_scan") is not None


# ── job logging ────────────────────────────────────────────────────────────────

def test_log_next_run_times_logs_one_line_per_job(caplog):
    """
    Startup should state, per job, when it will next fire — computed via the
    trigger directly since job.next_run_time is still None before
    BlockingScheduler.start() actually runs.
    """
    scheduler = _build_scheduler(Config())

    with caplog.at_level("INFO", logger="jarvis-sync"):
        daemon_module._log_next_run_times(scheduler)

    logged_ids = {
        job.id for job in scheduler.get_jobs()
        if any(f"job {job.id}: next run at" in r.message for r in caplog.records)
    }
    assert logged_ids == {job.id for job in scheduler.get_jobs()}


def test_log_job_outcome_reports_next_run_after_success_and_error(caplog):
    """A finished job (success or error) logs its next scheduled run time."""
    from types import SimpleNamespace

    scheduler = _build_scheduler(Config())
    ok_event = SimpleNamespace(job_id="vault_refresh", exception=None)
    error_event = SimpleNamespace(job_id="digest", exception=RuntimeError("boom"))

    with caplog.at_level("INFO", logger="jarvis-sync"):
        daemon_module._log_job_outcome(scheduler, ok_event)
    assert any(
        r.levelname == "INFO" and "job vault_refresh finished — next run at" in r.message
        for r in caplog.records
    )

    caplog.clear()
    with caplog.at_level("INFO", logger="jarvis-sync"):
        daemon_module._log_job_outcome(scheduler, error_event)
    assert any(
        r.levelname == "ERROR"
        and "job digest failed: boom — next run at" in r.message
        for r in caplog.records
    )


# ── status file ────────────────────────────────────────────────────────────────

def test_status_file_roundtrip_and_job_recording(tmp_path):
    """
    write/read roundtrip preserves content; record_job_status tracks last_run,
    last_success, and last_error correctly across a success and a failure.
    """
    status_file = tmp_path / "state" / "sync_status.json"

    write_status({"daemon": {"pid": 123}, "jobs": {}}, status_file)
    assert read_status(status_file)["daemon"]["pid"] == 123

    record_job_status("digest", ok=True, status_file=status_file)
    digest = read_status(status_file)["jobs"]["digest"]
    assert digest["last_run"] == digest["last_success"]
    assert digest["last_error"] == ""

    record_job_status("digest", ok=False, error="boom", status_file=status_file)
    digest = read_status(status_file)["jobs"]["digest"]
    assert "boom" in digest["last_error"]
    assert digest["last_success"]  # previous success is preserved

    # No stray temp file left behind by the atomic write.
    assert list(status_file.parent.glob("*.tmp")) == []


def test_read_status_missing_or_corrupt_file(tmp_path):
    """Missing or corrupt status files yield the empty skeleton, not a crash."""
    assert read_status(tmp_path / "absent.json") == {"daemon": {}, "jobs": {}}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert read_status(bad) == {"daemon": {}, "jobs": {}}