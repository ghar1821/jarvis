"""
Tests for jarvis/kb/cli.py — `kb doctor` and `kb rebuild`'s confirmation.

The doctor's probe runs in a subprocess, because reading a damaged vector
index kills the process by signal rather than raising. `_run_doctor_probe` is
therefore the seam these tests patch: it stands in for "what the child
reported", and returning None stands in for "the child died". Nothing here
opens the user's real knowledge base.
"""

from pathlib import Path

import pytest

from jarvis.core.config import Config
import jarvis.kb.cli as cli
from jarvis.kb.cli import cmd_doctor


def _healthy(via_server=True, count=5):
    return {"opened": True, "count": count, "search_ok": True, "error": None,
            "via_server": via_server}


# ── Doctor: health ─────────────────────────────────────────────────────────────

def test_cmd_doctor_reports_healthy_store(monkeypatch, capsys):
    """A store that opens, counts and search-probes cleanly reports healthy."""
    monkeypatch.setattr(cli, "_run_doctor_probe", _healthy)

    cmd_doctor()

    out = capsys.readouterr().out
    assert "Store opened" in out
    assert "5 chunk(s) indexed" in out
    assert "Search probe succeeded" in out


def test_cmd_doctor_survives_an_index_that_kills_the_probe(monkeypatch, capsys):
    """
    A damaged HNSW index terminates the child by signal, so there is no
    exception to catch. Doctor must survive, explain, and name the recovery —
    which is a rebuild from the files, since every entry comes from one.
    """
    monkeypatch.setattr(cli, "_run_doctor_probe", lambda: None)

    with pytest.raises(SystemExit) as exc_info:
        cmd_doctor()

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "damaged beyond reading" in err
    assert "kb rebuild" in err
    # It must also say nothing is lost, or it reads like total data loss.
    assert "Nothing is lost" in err


def test_cmd_doctor_exits_nonzero_on_corruption(monkeypatch, capsys):
    """A corrupted index reports the diagnosis and exits non-zero, scriptably."""
    monkeypatch.setattr(
        cli, "_run_doctor_probe",
        lambda: {"opened": True, "count": 5, "search_ok": False, "error": "run `uv run kb rebuild`"},
    )

    with pytest.raises(SystemExit) as exc_info:
        cmd_doctor()

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "Search failed" in err and "kb rebuild" in err


def test_cmd_doctor_exits_nonzero_when_store_fails_to_open(monkeypatch, capsys):
    """A RAGError opening the store (e.g. embed-model mismatch) also exits non-zero."""
    monkeypatch.setattr(
        cli, "_run_doctor_probe",
        lambda: {"opened": False, "count": None, "search_ok": None,
                 "error": "RAGError: Embedding model mismatch"},
    )

    with pytest.raises(SystemExit) as exc_info:
        cmd_doctor()

    assert exc_info.value.code == 1
    assert "Failed to open store" in capsys.readouterr().err


def test_cmd_doctor_reports_empty_store_without_search_probe(monkeypatch, capsys):
    """An empty store is healthy by definition, and the fix is a sync."""
    monkeypatch.setattr(cli, "_run_doctor_probe", lambda: _healthy(count=0))

    cmd_doctor()

    out = capsys.readouterr().out
    assert "0 chunk(s) indexed" in out
    assert "kb sync" in out


def test_doctor_says_when_kb_serve_is_not_running(monkeypatch, capsys):
    """jarvis-mcp needs `kb serve`; the CLI does not. Doctor says which case this is."""
    monkeypatch.setattr(cli, "_run_doctor_probe", lambda: _healthy(via_server=False))

    cmd_doctor()

    out = capsys.readouterr().out
    assert "kb serve is NOT running" in out
    assert "uv run kb serve" in out


def test_doctor_confirms_kb_serve_when_it_is_up(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_run_doctor_probe", _healthy)

    cmd_doctor()

    out = capsys.readouterr().out
    assert "kb serve is running" in out
    assert "NOT running" not in out


# ── Rebuild confirmation ───────────────────────────────────────────────────────

def test_rebuild_does_nothing_unless_confirmed(monkeypatch, capsys):
    """The rebuild prompt defaults to no, and a no touches nothing."""
    def must_not_rebuild(cfg):
        raise AssertionError("rebuild_index must not run without a yes")

    monkeypatch.setattr("jarvis.kb.mirror.rebuild_index", must_not_rebuild)
    monkeypatch.setattr("jarvis.core.config.get_config", lambda: Config(papers_dir=Path("/p")))
    monkeypatch.setattr("builtins.input", lambda prompt: "")

    cli.cmd_rebuild()

    out = capsys.readouterr().out
    assert "no file anywhere is touched" in out
    assert "Cancelled." in out
