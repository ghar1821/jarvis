"""
Tests for digest/daemon.py — the jarvis-sync background daemon.

The daemon's scheduler loop and watchdog observer are process-runtime
plumbing; what's tested here are the pure decision functions and job bodies:
catch-up detection, file-stability polling, inbox ingestion/dedup, config
validation, and the status file. PDF fixtures are generated in-test (cheap,
real conversion — no mocking needed since marker-pdf is gone).
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pymupdf
import pytest
from apscheduler.triggers.cron import CronTrigger

from digest.config import Config
from digest.daemon import (
    _build_scheduler,
    _validate_sync_config,
    digest_is_overdue,
    ingest_pdf,
    read_status,
    record_job_status,
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

def test_ingest_pdf_adds_then_skips_then_updates(store, tmp_path):
    """
    First ingest indexes the PDF as a public full-text paper; a second call
    with unchanged bytes is a no-op; changing the file (as saving new
    annotations does) replaces the old chunks.
    """
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


def test_scan_watch_dir_queues_pdfs_and_skips_artifacts(tmp_path):
    """
    The startup sweep queues real PDFs and ignores dotfiles and cloud
    placeholders.
    """
    import queue

    _make_pdf(tmp_path / "one.pdf", "one")
    _make_pdf(tmp_path / "two.pdf", "two")
    (tmp_path / ".hidden.pdf").write_bytes(b"x")
    (tmp_path / "syncing.pdf.icloud").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("not a pdf")

    q: "queue.Queue[Path]" = queue.Queue()
    assert scan_watch_dir(tmp_path, q) == 2
    names = {q.get().name, q.get().name}
    assert names == {"one.pdf", "two.pdf"}


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
    )
    problems = _validate_sync_config(cfg)
    assert len(problems) == 4
    assert any("pdf_watch_dir" in p for p in problems)
    assert any("digest_day" in p for p in problems)
    assert any("digest_hour" in p for p in problems)
    assert any("vault_refresh_minutes" in p for p in problems)


# ── scheduler construction ───────────────────────────────────────────────────

def test_build_scheduler_holds_both_jobs():
    """
    Regression for the launchd crash-loop: BlockingScheduler(timezone="local")
    handed the literal string "local" to ZoneInfo and raised at construction.
    _build_scheduler must build cleanly and register both jobs, and its
    timezone must be a real zone object rather than the string "local".
    """
    from apscheduler.schedulers.blocking import BlockingScheduler

    # Construction alone reproduced the original crash, so nothing needs to
    # start the scheduler here.
    scheduler = _build_scheduler(Config())
    assert isinstance(scheduler, BlockingScheduler)
    assert scheduler.get_job("digest") is not None
    assert scheduler.get_job("vault_refresh") is not None
    # The bug was a bare string; a resolved zone is never a str.
    assert not isinstance(scheduler.timezone, str)


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