"""
Tests for tool-call failure logging in jarvis/chat/chat.py.

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

import jarvis.chat.chat as chat_module
from jarvis.core.errors import KBCorruptionError
from jarvis.chat.chat import _kb_stats, _list_documents, _search_chat_history, _search_kb


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

    monkeypatch.setattr("jarvis.kb.store.get_store", broken_get_store)

    with caplog.at_level(logging.ERROR, logger="jarvis.chat"):
        result = _kb_stats()

    assert result == "[kb_stats error: simulated database failure]"
    assert len(caplog.records) == 1
    assert "kb_stats tool failed" in caplog.records[0].message
    assert caplog.records[0].exc_info is not None


def test_list_documents_failure_is_logged(monkeypatch, caplog, isolated_log):
    """Same contract on a second tool, to confirm this isn't a one-off wire-up."""
    def broken_list_documents(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: None)
    monkeypatch.setattr("jarvis.kb.store.list_documents", broken_list_documents)

    with caplog.at_level(logging.ERROR, logger="jarvis.chat"):
        result = _list_documents({})

    assert result == "[list_documents error: boom]"
    assert any("list_documents tool failed" in r.message for r in caplog.records)
    assert caplog.records[0].exc_info is not None


# ── KBCorruptionError relay ──────────────────────────────────────────────────────
#
# A corrupted ChromaDB index (KBCorruptionError, see jarvis/core/errors.py) must be
# relayed to the LLM verbatim rather than folded into the generic "[<tool>
# error: ...]" string an LLM would paraphrase away. log.exception must still
# fire first, exactly like the generic-failure path above.

def _broken_search(*args, **kwargs):
    raise KBCorruptionError(
        "The knowledge base index is corrupted. Fix: run `uv run kb reindex`."
    )


def test_search_kb_relays_corruption_error_verbatim(monkeypatch, caplog, isolated_log):
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: None)
    monkeypatch.setattr("jarvis.kb.store.search_with_privacy_check", _broken_search)

    with caplog.at_level(logging.ERROR, logger="jarvis.chat"):
        result, saw_private = _search_kb({"query": "anything"}, "ollama")

    assert result.startswith("[KNOWLEDGE BASE ERROR")
    assert "run `uv run kb reindex`" in result
    assert saw_private is False
    assert any("search_kb tool failed" in r.message for r in caplog.records)
    assert caplog.records[0].exc_info is not None


def test_search_kb_notes_relays_corruption_error_verbatim(monkeypatch, caplog, isolated_log):
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: None)
    monkeypatch.setattr("jarvis.kb.store.search_with_privacy_check", _broken_search)

    with caplog.at_level(logging.ERROR, logger="jarvis.chat"):
        result, saw_private = _search_kb({"kinds": ["notes"], "query": "anything"}, "ollama")

    assert result.startswith("[KNOWLEDGE BASE ERROR")
    assert "run `uv run kb reindex`" in result
    assert saw_private is False
    assert any("search_kb tool failed" in r.message for r in caplog.records)
    assert caplog.records[0].exc_info is not None


def test_search_chat_history_relays_corruption_error_verbatim(monkeypatch, caplog, isolated_log):
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: None)
    monkeypatch.setattr("jarvis.kb.store.search_with_privacy_check", _broken_search)

    with caplog.at_level(logging.ERROR, logger="jarvis.chat"):
        result = _search_chat_history({"query": "anything"}, "ollama")

    assert result.startswith("[KNOWLEDGE BASE ERROR")
    assert "run `uv run kb reindex`" in result
    assert any("search_chat_history tool failed" in r.message for r in caplog.records)
    assert caplog.records[0].exc_info is not None


# ── nothing swallows an error silently ───────────────────────────────────────
#
# A caught exception with no handler is indistinguishable from no exception at
# all. These pin the cases where swallowing would change behaviour without
# saying so.


def test_a_failed_duplicate_check_is_logged_rather_than_read_as_new(caplog):
    """
    `_source_exists` returning False means "go ahead and add". On a broken
    store that quietly produces a second copy of a paper you already have.
    """
    from jarvis.kb.store import _source_exists

    class Broken:
        _collection = property(lambda self: (_ for _ in ()).throw(RuntimeError("store is broken")))

    with caplog.at_level(logging.WARNING, logger="jarvis.kb.store"):
        assert _source_exists("https://arxiv.org/abs/1", Broken()) is False

    assert "duplicate check" in caplog.text
    assert "store is broken" in caplog.text, "the real error must be in the log"


def test_an_unreadable_document_count_says_so_rather_than_reporting_zero(caplog):
    """0 reads as "your knowledge base is empty", which is alarming and wrong."""
    from jarvis.kb.store import count_unique_documents

    class Broken:
        _collection = property(lambda self: (_ for _ in ()).throw(RuntimeError("cannot read")))

    with caplog.at_level(logging.WARNING, logger="jarvis.kb.store"):
        assert count_unique_documents("paper", "source", store=Broken()) == 0

    assert "cannot read" in caplog.text


def test_refresh_vault_refuses_to_run_on_an_unreadable_index(store, tmp_path):
    """
    Falling back to "nothing is indexed" would re-index the whole vault and
    treat every existing note as new. This has to fail rather than silently
    do something else.
    """
    from jarvis.core.errors import RAGError
    from jarvis.kb.store import refresh_vault

    class Broken:
        _collection = property(lambda self: (_ for _ in ()).throw(RuntimeError("index unreadable")))

    (tmp_path / "note.md").write_text("# a note\n")

    with pytest.raises(RAGError) as caught:
        refresh_vault(tmp_path, Broken())

    assert "index unreadable" in str(caught.value)


def test_an_unreadable_session_file_is_named_in_the_log(tmp_path, caplog):
    """A conversation missing from the sidebar should be explainable."""
    from jarvis.chat.sessions import list_sessions

    (tmp_path / "20260101-000000-abcdef.json").write_text("{ not json")

    with caplog.at_level(logging.WARNING, logger="jarvis.chat.sessions"):
        assert list_sessions(sessions_dir=tmp_path) == []

    assert "20260101-000000-abcdef.json" in caplog.text
