"""
Tests for tool-call failure logging in vault_chat/chat.py.

Every tool wrapper catches Exception broadly and returns a short string for
the LLM to relay — but LLMs paraphrase rather than quote, so without a log
the real exception and its traceback would be unrecoverable after the fact.
These tests confirm the log.exception() call in each wrapper's except block
actually fires. The module attaches its own FileHandler at import time, so
it's still live during tests even though we assert via caplog — the
isolated_log fixture detaches it for the duration of each test so these
runs never append to the user's real ~/.jarvis/logs/chat.log.
"""

import logging

import pytest

import vault_chat.chat as chat_module
from vault_chat.chat import _kb_stats, _list_papers


@pytest.fixture
def isolated_log():
    """Detach chat.py's real FileHandler so tests never touch chat.log."""
    handlers = list(chat_module.log.handlers)
    for handler in handlers:
        chat_module.log.removeHandler(handler)
    yield
    for handler in handlers:
        chat_module.log.addHandler(handler)


def test_kb_stats_failure_is_logged_with_traceback(monkeypatch, caplog, isolated_log):
    """
    A tool that raises must log the exception (with traceback) before
    returning its short error string to the LLM.

    Input: get_store() raises RuntimeError inside _kb_stats
    Expected output: an ERROR record naming the tool, with a traceback
            attached; the usual short error string is still returned
    """
    def broken_get_store():
        raise RuntimeError("simulated database failure")

    monkeypatch.setattr("digest.kb.store.get_store", broken_get_store)

    with caplog.at_level(logging.ERROR, logger="vault-chat"):
        result = _kb_stats()

    assert result == "[kb_stats error: simulated database failure]"
    assert len(caplog.records) == 1
    assert "kb_stats tool failed" in caplog.records[0].message
    assert caplog.records[0].exc_info is not None


def test_list_papers_failure_is_logged(monkeypatch, caplog, isolated_log):
    """Same contract on a second tool, to confirm this isn't a one-off wire-up."""
    def broken_list_papers(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("digest.kb.store.get_store", lambda: None)
    monkeypatch.setattr("digest.kb.store.list_papers", broken_list_papers)

    with caplog.at_level(logging.ERROR, logger="vault-chat"):
        result = _list_papers({})

    assert result == "[list_papers error: boom]"
    assert any("list_papers tool failed" in r.message for r in caplog.records)
    assert caplog.records[0].exc_info is not None
