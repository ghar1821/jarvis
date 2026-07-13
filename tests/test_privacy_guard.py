"""
Tests for the chat-layer privacy enforcement in jarvis/chat/chat.py.

These cover the guards that sit between the LLM's tool calls and the data:
- read_file: vault containment, private-dir hard stop, symlink resolution
- _search_notes: the "private matches excluded" caveat and hard stop
- _get_document: privacy mirrors read_file's behaviour

The store fixture comes from conftest.py (real embeddings, isolated
collection). get_store()/get_config() are monkeypatched where the chat
helpers call the process singletons.
"""

import os
from pathlib import Path

import pytest

from jarvis.core.config import Config
from jarvis.core.errors import PrivacyError
from jarvis.kb.store import add_texts
from jarvis.chat.chat import _get_document, _search_notes, read_file


@pytest.fixture
def vault(tmp_path, monkeypatch):
    """Vault with a public and a private note; private_vault_dirs=['private']."""
    (tmp_path / "public").mkdir()
    (tmp_path / "private").mkdir()
    (tmp_path / "public" / "open.md").write_text("# Open\nPublic content.")
    (tmp_path / "private" / "secret.md").write_text("# Secret\nPrivate content.")
    monkeypatch.setattr(
        "jarvis.kb.store.get_config",
        lambda: Config(private_vault_dirs=["private"]),
    )
    return tmp_path


# ── read_file ──────────────────────────────────────────────────────────────────

def test_read_file_public_note_ok_for_both_providers(vault):
    """
    A public note is readable regardless of provider, and is not flagged
    private.

    Input:  public/open.md, local and anthropic providers
    Expected output: (content, saw_private=False) both times
    """
    content, saw_private = read_file(vault, "public/open.md", "ollama")
    assert "Public content" in content and saw_private is False
    content, saw_private = read_file(vault, "public/open.md", "anthropic")
    assert "Public content" in content and saw_private is False


def test_read_file_private_note_blocked_for_cloud_only(vault):
    """
    A private note raises PrivacyError for the cloud provider but is readable
    locally — where it reports saw_private=True so the session gets flagged.

    Input:  private/secret.md
    Expected output: PrivacyError (anthropic); (content, True) locally
    """
    content, saw_private = read_file(vault, "private/secret.md", "ollama")
    assert "Private content" in content and saw_private is True
    with pytest.raises(PrivacyError):
        read_file(vault, "private/secret.md", "anthropic")


def test_read_file_blocks_symlink_into_private_dir(vault):
    """
    A symlink placed in a public folder that resolves into a private folder
    must be classified by its RESOLVED location — the historical bypass this
    guards against.

    Input:  public/link.md → private/secret.md, anthropic provider
    Expected output: PrivacyError; local provider still reads it (flagged private)
    """
    os.symlink(vault / "private" / "secret.md", vault / "public" / "link.md")

    with pytest.raises(PrivacyError):
        read_file(vault, "public/link.md", "anthropic")
    content, saw_private = read_file(vault, "public/link.md", "ollama")
    assert "Private content" in content and saw_private is True


def test_read_file_blocks_path_escape(vault):
    """
    Paths resolving outside the vault are refused with an error string, not
    file content.

    Input:  ../../etc/hosts style traversal
    Expected output: 'outside the vault' error string
    """
    result, saw_private = read_file(vault, "../../../../etc/hosts", "anthropic")
    assert "outside the vault" in result
    assert saw_private is False


# ── _search_notes caveat ───────────────────────────────────────────────────────

def test_search_notes_appends_caveat_when_private_matches_excluded(store, monkeypatch):
    """
    When a cloud search returns public hits but private notes also matched,
    the result must carry the static incomplete-results caveat — and no
    private content.

    Input:  one public and one private note about the same topic, anthropic
    Expected output: public hit + caveat string; private text absent
    """
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    add_texts(content="Public overview of the quantum sensing project.",
              doc_type="note", visibility="public", source="local",
              extra_metadata={"file_path": "projects/quantum.md", "title": "Quantum"},
              store=store)
    add_texts(content="Private budget worries about the quantum sensing project.",
              doc_type="note", visibility="private", source="local",
              extra_metadata={"file_path": "private/quantum.md", "title": "Quantum private"},
              store=store)

    result, saw_private = _search_notes({"query": "quantum sensing project"}, "anthropic")
    assert "Public overview" in result
    assert "excluded from these results" in result
    assert "budget worries" not in result
    # On the cloud path private docs never appear in results, so the session
    # flag must not flip.
    assert saw_private is False


def test_search_notes_hard_stops_when_only_private_matches(store, monkeypatch):
    """
    A cloud query matching only private notes raises PrivacyError instead of
    returning anything.

    Input:  a single private note, anthropic provider
    Expected output: PrivacyError
    """
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    add_texts(content="Private thoughts on the reorganisation.",
              doc_type="note", visibility="private", source="local",
              extra_metadata={"file_path": "private/reorg.md"}, store=store)

    with pytest.raises(PrivacyError):
        _search_notes({"query": "reorganisation thoughts"}, "anthropic")


def test_search_notes_local_provider_gets_no_caveat(store, monkeypatch):
    """
    The local provider sees everything, so no caveat is ever appended — and
    the private hit is reported so the session gets flagged.

    Input:  public + private notes, local provider
    Expected output: both hits, no caveat text, saw_private=True
    """
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    add_texts(content="Public note about conference travel.", doc_type="note",
              visibility="public", source="local",
              extra_metadata={"file_path": "travel.md"}, store=store)
    add_texts(content="Private note about conference travel budget.", doc_type="note",
              visibility="private", source="local",
              extra_metadata={"file_path": "private/travel.md"}, store=store)

    result, saw_private = _search_notes({"query": "conference travel"}, "ollama")
    assert "excluded from these results" not in result
    assert saw_private is True


# ── _get_document privacy ───────────────────────────────────────────────────────

def test_get_document_public_doc_fine_under_anthropic(store, monkeypatch):
    """
    A public document reads fine under the cloud provider — get_document
    mirrors read_file's privacy behaviour, not a blanket cloud restriction.

    Input:  a public note's source, anthropic provider
    Expected output: content returned, saw_private=False
    """
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    add_texts(content="Public overview of the quantum sensing project.",
              doc_type="note", visibility="public", source="local-public-doc",
              extra_metadata={"file_path": "projects/quantum.md", "title": "Quantum"},
              store=store)

    result, saw_private = _get_document({"source": "local-public-doc"}, "anthropic")
    assert "Public overview" in result
    assert saw_private is False


def test_get_document_private_source_hard_stops_under_anthropic_with_no_leak(store, monkeypatch):
    """
    A private document's source raises PrivacyError before any content —
    even a hint of title or length — reaches the cloud provider.

    Input:  a private note's source, anthropic provider
    Expected output: PrivacyError whose message contains no document content
    """
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    add_texts(content="Private budget worries about the quantum sensing project.",
              doc_type="note", visibility="private", source="local-private-doc",
              extra_metadata={"file_path": "private/quantum.md", "title": "Quantum private"},
              store=store)

    with pytest.raises(PrivacyError) as exc_info:
        _get_document({"source": "local-private-doc"}, "anthropic")
    assert "budget worries" not in str(exc_info.value)
    assert "Quantum private" not in str(exc_info.value)


def test_get_document_private_source_readable_locally(store, monkeypatch):
    """
    The local provider can read a private document in full — and the call
    reports saw_private=True so the session gets flagged.

    Input:  a private note's source, ollama provider
    Expected output: content returned, saw_private=True
    """
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    add_texts(content="Private budget worries about the quantum sensing project.",
              doc_type="note", visibility="private", source="local-private-doc-2",
              extra_metadata={"file_path": "private/quantum.md", "title": "Quantum private"},
              store=store)

    result, saw_private = _get_document({"source": "local-private-doc-2"}, "ollama")
    assert "budget worries" in result
    assert saw_private is True

