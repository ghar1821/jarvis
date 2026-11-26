"""
Tests for jarvis/kb/serve.py — `kb serve`'s sync loop bookkeeping and `kb status`.

The loop itself (start Chroma, sync, wait) is a thin wrapper verified by
running it; what is tested here is everything a user relies on to know
whether it is working: the outcome of each sync pass is recorded, a failing
pass is recorded rather than raised, and `kb status` reports both honestly.
"""

import os

import pytest

from jarvis.kb import serve
from jarvis.kb.serve import _wait_until_answering, print_status, read_status, run_sync, write_status


def test_status_file_round_trips_and_tolerates_garbage(tmp_path):
    status_file = tmp_path / "serve_status.json"
    assert read_status(status_file) == {}

    write_status({"pid": 1, "started_at": "now"}, status_file)
    assert read_status(status_file) == {"pid": 1, "started_at": "now"}

    status_file.write_text("{not json")
    assert read_status(status_file) == {}


def test_a_successful_sync_records_its_counts(tmp_path, monkeypatch):
    status_file = tmp_path / "serve_status.json"
    write_status({"pid": 1}, status_file)
    monkeypatch.setattr("jarvis.kb.mirror.sync_all",
                        lambda store=None: {"notes_added": 2, "papers_deleted": 1})

    assert run_sync(status_file) is True

    last_sync = read_status(status_file)["last_sync"]
    assert last_sync["ok"] is True
    assert last_sync["counts"] == {"notes_added": 2, "papers_deleted": 1}
    assert read_status(status_file)["pid"] == 1  # the rest of the file survives


def test_a_failed_sync_is_recorded_not_raised_and_reconnects_next_time(tmp_path, monkeypatch):
    """
    One bad pass must not take the server down. The error is kept for
    `kb status`, and the cached store is dropped so the next pass reconnects
    (after `kb rebuild` the old handle names a collection that is gone).
    """
    status_file = tmp_path / "serve_status.json"
    forgotten = []

    def broken_sync(store=None):
        raise RuntimeError("Collection [abc] does not exist")

    monkeypatch.setattr("jarvis.kb.mirror.sync_all", broken_sync)
    monkeypatch.setattr("jarvis.kb.store.forget_store", lambda: forgotten.append(True))

    assert run_sync(status_file) is False

    last_sync = read_status(status_file)["last_sync"]
    assert last_sync["ok"] is False and "does not exist" in last_sync["error"]
    assert forgotten == [True]


def test_status_before_serve_has_ever_run(tmp_path, capsys):
    print_status(tmp_path / "missing.json")
    assert "never run" in capsys.readouterr().out


def test_status_reports_a_running_server_and_its_last_sync(tmp_path, monkeypatch, capsys):
    status_file = tmp_path / "serve_status.json"
    write_status({
        "pid": os.getpid(),  # this test process: certainly alive
        "started_at": "2026-11-26T09:00:00+00:00",
        "last_sync": {"at": "2026-11-26T09:30:00+00:00", "ok": True,
                      "counts": {"notes_added": 0, "papers_added": 3}},
    }, status_file)
    monkeypatch.setattr(serve, "LOG_DIR", tmp_path)

    print_status(status_file)

    out = capsys.readouterr().out
    assert f"running (pid {os.getpid()})" in out
    assert "papers_added 3" in out and "notes_added" not in out


def test_status_says_when_the_server_has_stopped_and_the_last_sync_failed(
    tmp_path, monkeypatch, capsys
):
    status_file = tmp_path / "serve_status.json"
    write_status({
        "pid": 999_999_999,  # no such process
        "started_at": "earlier",
        "last_sync": {"at": "earlier", "ok": False, "error": "vault unreadable"},
    }, status_file)
    monkeypatch.setattr(serve, "LOG_DIR", tmp_path)

    print_status(status_file)

    out = capsys.readouterr().out
    assert "NOT RUNNING" in out and "uv run kb serve" in out
    assert "FAILED: vault unreadable" in out


class _DeadChild:
    """A Chroma child that exited at once — e.g. the port was already taken."""

    returncode = 1

    def poll(self):
        return self.returncode


def test_start_up_explains_a_chroma_that_exits_immediately():
    """The usual cause is a second `kb serve` on the same port; say so."""
    with pytest.raises(RuntimeError, match="already using port 8321"):
        _wait_until_answering(_DeadChild(), 8321, timeout=2)
