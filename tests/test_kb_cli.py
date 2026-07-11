"""
Tests for jarvis/kb/cli.py's `kb doctor` and `kb reindex` subcommands.

get_store()/search()/get_config() are monkeypatched to the isolated test
store (or a stub) so this never touches the real CLI parser's live
~/.jarvis config — per the project rule that tests must never open the
user's actual knowledge base.
"""

import chromadb
import pytest

from jarvis.core.config import Config
from jarvis.core.errors import KBCorruptionError, RAGError
from jarvis.kb.cli import _migrated_chunk_text, cmd_doctor, cmd_reindex


def test_cmd_doctor_reports_healthy_store(store, monkeypatch, capsys):
    """A store that opens, counts, and search-probes cleanly reports healthy."""
    from jarvis.kb.store import add_texts

    add_texts(content="A healthy document.", doc_type="note",
              visibility="public", source="doctor-ok", store=store)
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)

    cmd_doctor()

    out = capsys.readouterr().out
    assert "Store opened" in out
    assert "chunk(s) indexed" in out
    assert "Knowledge base is healthy." in out


def test_cmd_doctor_exits_nonzero_on_corruption(monkeypatch, capsys):
    """A corrupted index reports the diagnosis and exits non-zero, scriptably."""
    class _StubStore:
        pass

    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: _StubStore())
    monkeypatch.setattr("jarvis.kb.store.count", lambda store: 5)

    def broken_search(*args, **kwargs):
        raise KBCorruptionError("run `uv run kb reindex`")

    monkeypatch.setattr("jarvis.kb.store.search", broken_search)

    with pytest.raises(SystemExit) as exc_info:
        cmd_doctor()

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "corrupted" in err
    assert "kb reindex" in err


def test_cmd_doctor_exits_nonzero_when_store_fails_to_open(monkeypatch, capsys):
    """A RAGError opening the store (e.g. embed-model mismatch) also exits non-zero."""
    def broken_get_store():
        raise RAGError("Embedding model mismatch")

    monkeypatch.setattr("jarvis.kb.store.get_store", broken_get_store)

    with pytest.raises(SystemExit) as exc_info:
        cmd_doctor()

    assert exc_info.value.code == 1
    assert "Failed to open store" in capsys.readouterr().err


def test_cmd_doctor_reports_empty_store_without_search_probe(monkeypatch, capsys):
    """An empty store is healthy by definition — no search probe is attempted."""
    class _StubStore:
        pass

    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: _StubStore())
    monkeypatch.setattr("jarvis.kb.store.count", lambda store: 0)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("search() must not run against an empty store")

    monkeypatch.setattr("jarvis.kb.store.search", fail_if_called)

    cmd_doctor()

    out = capsys.readouterr().out
    assert "empty" in out.lower()


# ── `kb reindex` embed-header migration ─────────────────────────────────────
#
# `kb reindex` re-embeds every stored chunk with the currently configured
# embedding model. It must also backfill the title/authors embed-header onto
# legacy paper chunks that predate it, so old papers become matchable by
# author-name/acronym queries too — see _migrated_chunk_text.


def test_migrated_chunk_text_prepends_header_to_headerless_paper_chunk():
    """A header-less paper chunk gets 'title — authors' prepended."""
    metadata = {"doc_type": "paper", "title": "Attention Is All You Need", "authors": "Vaswani et al."}
    text = "The dominant sequence transduction models are based on..."

    migrated = _migrated_chunk_text(text, metadata)

    assert migrated.startswith("Attention Is All You Need — Vaswani et al.\n")
    assert text in migrated


def test_migrated_chunk_text_is_idempotent():
    """Running the migration twice must not double-prepend the header."""
    metadata = {"doc_type": "paper", "title": "Attention Is All You Need", "authors": "Vaswani et al."}
    text = "The dominant sequence transduction models are based on..."

    once = _migrated_chunk_text(text, metadata)
    twice = _migrated_chunk_text(once, metadata)

    assert once == twice


def test_migrated_chunk_text_leaves_annotation_and_note_chunks_untouched():
    """Annotation chunks and note chunks are never given a header."""
    annotation_text = "Figure 2: architecture diagram."
    annotation_meta = {
        "doc_type": "paper", "annotation_kind": "figure",
        "title": "Attention Is All You Need", "authors": "Vaswani et al.",
    }
    assert _migrated_chunk_text(annotation_text, annotation_meta) == annotation_text

    note_text = "Meeting notes from Tuesday."
    note_meta = {"doc_type": "note", "title": "Meeting notes from Tuesday.", "authors": ""}
    assert _migrated_chunk_text(note_text, note_meta) == note_text


def test_migrated_chunk_text_no_authors_uses_title_only_header():
    """No authors on record — header is just the title, no trailing dash."""
    metadata = {"doc_type": "paper", "title": "Untitled Draft", "authors": ""}
    text = "Some legacy chunk text."

    migrated = _migrated_chunk_text(text, metadata)

    assert migrated == "Untitled Draft\nSome legacy chunk text."


def test_cmd_reindex_backfills_embed_header_end_to_end(tmp_path, monkeypatch, embeddings):
    """
    Full pass through cmd_reindex against an isolated, temporary ChromaDB
    directory (never ~/.jarvis): a legacy header-less paper chunk comes out
    the other side with the header prepended, while an annotation chunk and
    a note chunk are left exactly as they were.
    """
    from jarvis.kb.store import COLLECTION_NAME

    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection(COLLECTION_NAME)
    collection.add(
        ids=["paper-chunk", "annotation-chunk", "note-chunk"],
        documents=[
            "The dominant sequence transduction models are based on...",
            "Figure 2: architecture diagram.",
            "Meeting notes from Tuesday.",
        ],
        metadatas=[
            {"doc_type": "paper", "title": "Attention Is All You Need", "authors": "Vaswani et al."},
            {"doc_type": "paper", "annotation_kind": "figure",
             "title": "Attention Is All You Need", "authors": "Vaswani et al."},
            {"doc_type": "note", "title": "Meeting notes from Tuesday.", "authors": ""},
        ],
        embeddings=[[0.0] * 384, [0.0] * 384, [0.0] * 384],
    )

    cfg = Config(rag_dir=tmp_path)
    monkeypatch.setattr("jarvis.core.config.get_config", lambda: cfg)
    monkeypatch.setattr("jarvis.kb.store.build_embeddings", lambda model, prefix: embeddings)

    cmd_reindex(None)

    reindexed = client.get_collection(COLLECTION_NAME)
    stored = reindexed.get(ids=["paper-chunk", "annotation-chunk", "note-chunk"], include=["documents"])
    by_id = dict(zip(stored["ids"], stored["documents"]))

    assert by_id["paper-chunk"].startswith("Attention Is All You Need — Vaswani et al.\n")
    assert by_id["annotation-chunk"] == "Figure 2: architecture diagram."
    assert by_id["note-chunk"] == "Meeting notes from Tuesday."

    # A second reindex pass must be a no-op on the already-migrated text.
    cmd_reindex(None)
    reindexed_again = client.get_collection(COLLECTION_NAME)
    stored_again = reindexed_again.get(ids=["paper-chunk"], include=["documents"])
    assert stored_again["documents"][0] == by_id["paper-chunk"]
