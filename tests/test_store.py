"""
Tests for digest/kb/store.py — knowledge base operations.

All tests receive an isolated ChromaDB collection via the `store` fixture
(see conftest.py) and pass it explicitly using the store= parameter that
every KB function accepts. The process-wide get_store() singleton is never
touched here.

Real HuggingFace embeddings are used throughout — see conftest.py for the
rationale. Semantically distinct content is used so similarity search returns
meaningful results.

Sections
--------
1. add_texts / add_paper     — indexing and idempotency
2. search                    — visibility filter correctness
3. search_with_privacy_check — cloud vs local provider access control
4. delete_by_metadata        — chunk removal
5. list_papers               — deduplication and chunk count
6. update_file_path          — in-place metadata update
7. refresh_vault             — incremental vault sync (add / update / delete / PDF notes)
"""

import uuid
from pathlib import Path

import pytest
from langchain_chroma import Chroma

from digest.config import Config
from digest.errors import RAGError
from digest.kb.store import (
    _check_embedding_model_matches,
    add_paper,
    add_texts,
    count,
    delete_by_metadata,
    list_papers,
    refresh_vault,
    search,
    search_with_privacy_check,
    update_file_path,
)

# Same gitignored store directory the shared fixtures use (see conftest.py).
TEST_CHROMA_DIR = Path(__file__).parent / ".chroma"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _paper(n: int) -> dict:
    """Minimal paper dict accepted by add_paper()."""
    return {
        "link": f"https://arxiv.org/abs/2301.{n:05d}",
        "title": f"Test Paper {n}",
        "authors": "Test Author",
    }


# ── 1. Indexing ────────────────────────────────────────────────────────────────

def test_add_texts_returns_ids_and_increases_count(store):
    """
    add_texts() chunks the content and stores each chunk in the collection.
    The returned list of IDs should be non-empty, and the collection count
    should rise by exactly the number of IDs returned.

    Input:  a single sentence, doc_type="note", visibility="public"
    Expected output:
        len(ids) > 0
        count increases by len(ids)
    """
    before = count(store=store)
    ids = add_texts(
        content="Transformer architectures use self-attention to capture long-range dependencies.",
        doc_type="note",
        visibility="public",
        source="local",
        store=store,
    )
    assert len(ids) > 0
    assert count(store=store) == before + len(ids)


def test_add_paper_is_idempotent(store):
    """
    Calling add_paper() twice with the same source URL must not create duplicate
    entries. The second call returns an empty list and the chunk count is unchanged.

    Input:  the same paper dict passed to add_paper() twice
    Expected output:
        first call  → returns non-empty list of IDs
        second call → returns []
        count after second call == count after first call
    """
    paper = _paper(1)
    first = add_paper(paper, dense_summary="A study on attention mechanisms.", store=store)
    assert len(first) > 0

    count_after_first = count(store=store)
    second = add_paper(paper, dense_summary="A study on attention mechanisms.", store=store)
    assert second == []
    assert count(store=store) == count_after_first


# ── 2. Search ──────────────────────────────────────────────────────────────────

def test_search_visibility_filter_excludes_private_docs(store):
    """
    search() with visibility="public" must not return private documents even
    when the private document is semantically identical to the query.

    Both documents have the same content so without the filter either could rank
    highest — the filter must do the exclusion, not relevance scoring.

    Input:
        doc A — visibility="public",  source="public-doc"
        doc B — visibility="private", source="private-doc"
        query — text matching both documents
    Expected output:
        results contain "public-doc"
        results do not contain "private-doc"
    """
    content = "Gradient descent iteratively adjusts weights to minimise the loss function."
    add_texts(content=content, doc_type="note", visibility="public",
              source="public-doc", store=store)
    add_texts(content=content, doc_type="note", visibility="private",
              source="private-doc", store=store)

    results = search(
        "gradient descent weight optimisation",
        n_results=10,
        visibility="public",
        store=store,
    )
    sources = {doc.metadata["source"] for doc in results}
    assert "public-doc" in sources
    assert "private-doc" not in sources


# ── 3. Privacy check ───────────────────────────────────────────────────────────

def test_privacy_check_cloud_provider_sees_public_only_and_reports_private_hit(store):
    """
    search_with_privacy_check() for a cloud provider ("anthropic") must return
    only public documents and set has_private_hits=True when private documents
    also matched the query.

    Input:
        doc A — visibility="public",  source="pub"
        doc B — visibility="private", source="priv"
        provider = "anthropic"
    Expected output:
        results contain "pub", not "priv"
        has_private_hits == True
    """
    content = "Contrastive learning creates representations without explicit labels."
    add_texts(content=content, doc_type="note", visibility="public",
              source="pub", store=store)
    add_texts(content=content, doc_type="note", visibility="private",
              source="priv", store=store)

    results, has_private = search_with_privacy_check(
        "contrastive learning representations",
        provider="anthropic",
        n_results=10,
        store=store,
    )
    sources = {doc.metadata["source"] for doc in results}
    assert "pub" in sources
    assert "priv" not in sources
    assert has_private is True


def test_privacy_check_local_provider_sees_all_docs(store):
    """
    search_with_privacy_check() for a local provider ("ollama") must return all
    documents regardless of visibility, and always set has_private_hits=False.

    Input:
        doc A — visibility="public",  source="pub2"
        doc B — visibility="private", source="priv2"
        provider = "ollama"
    Expected output:
        results contain both "pub2" and "priv2"
        has_private_hits == False
    """
    content = "Diffusion models learn to reverse a gradual noising process."
    add_texts(content=content, doc_type="note", visibility="public",
              source="pub2", store=store)
    add_texts(content=content, doc_type="note", visibility="private",
              source="priv2", store=store)

    results, has_private = search_with_privacy_check(
        "diffusion models denoising",
        provider="ollama",
        n_results=10,
        store=store,
    )
    sources = {doc.metadata["source"] for doc in results}
    assert "pub2" in sources
    assert "priv2" in sources
    assert has_private is False


# ── 4. Deletion ────────────────────────────────────────────────────────────────

def test_delete_by_metadata_removes_only_matching_chunks(store):
    """
    delete_by_metadata() removes every chunk whose metadata key matches the
    given value and returns the count of deleted chunks. Documents with a
    different value for that key are not affected.

    Input:
        doc A — source="to-delete"
        doc B — source="to-keep"
        delete by source == "to-delete"
    Expected output:
        deleted > 0
        count drops by exactly `deleted`
        searching for doc B still returns a result; doc A is gone
    """
    add_texts(
        content="Reinforcement learning maximises a cumulative reward signal.",
        doc_type="note", visibility="public", source="to-delete", store=store,
    )
    add_texts(
        content="Supervised learning requires labelled examples for training.",
        doc_type="note", visibility="public", source="to-keep", store=store,
    )

    before = count(store=store)
    deleted = delete_by_metadata("source", "to-delete", store=store)
    assert deleted > 0
    assert count(store=store) == before - deleted

    remaining = search("supervised labelled training", n_results=5, store=store)
    sources = {doc.metadata["source"] for doc in remaining}
    assert "to-keep" in sources
    assert "to-delete" not in sources


# ── 5. Listing ─────────────────────────────────────────────────────────────────

def test_list_papers_deduplicates_and_reports_chunk_count(store):
    """
    list_papers() returns one entry per unique source URL regardless of how many
    chunks that document was split into. The entry includes a chunk_count field
    that matches the actual number of stored chunks.

    Input:  a long summary that the splitter divides into multiple chunks
    Expected output:
        exactly one entry for that source in list_papers()
        entry["chunk_count"] == number of IDs returned by add_paper()
    """
    paper = _paper(99)
    # Repeat a sentence to exceed the default chunk_size and force multiple chunks
    long_summary = "Self-attention computes pairwise interactions between all tokens. " * 200
    ids = add_paper(paper, dense_summary=long_summary, store=store)
    assert len(ids) > 1, "summary should have produced multiple chunks"

    papers = list_papers(store=store)
    matches = [p for p in papers if p["source"] == paper["link"]]
    assert len(matches) == 1
    assert matches[0]["chunk_count"] == len(ids)


# ── 6. Path update ─────────────────────────────────────────────────────────────

def test_update_file_path_updates_metadata_and_source_uri(store, tmp_path):
    """
    update_file_path() rewrites the file_path metadata field and, when the
    source is a file:/// URI, the source field too — across every chunk that
    belongs to that document. No re-embedding is performed.

    Input:
        doc indexed with source="file:///old/paper.pdf", file_path="/old/paper.pdf"
        call update_file_path(old_source, new_path)
    Expected output:
        all chunks have file_path == new_path
        all chunks have source == new_path as a file:/// URI
        return value == number of updated chunks
    """
    old_source = "file:///old/paper.pdf"
    new_path = str(tmp_path / "paper.pdf")
    expected_source = Path(new_path).as_uri()

    add_texts(
        content="Sparse autoencoders decompose activations into interpretable features.",
        doc_type="note",
        visibility="public",
        source=old_source,
        extra_metadata={"file_path": "/old/paper.pdf"},
        store=store,
    )

    n = update_file_path(old_source, new_path, store=store)
    assert n > 0

    result = store._collection.get(
        where={"source": {"$eq": expected_source}},
        include=["metadatas"],
    )
    assert len(result["ids"]) == n
    for meta in result["metadatas"]:
        assert meta["file_path"] == new_path
        assert meta["source"] == expected_source


def test_update_file_path_returns_zero_for_unknown_source(store):
    """
    When no chunks match the given source, update_file_path returns 0 without
    raising an exception.

    Input:  a source string that was never indexed
    Expected output: 0
    """
    n = update_file_path("file:///does/not/exist.pdf", "/new/path.pdf", store=store)
    assert n == 0


# ── 7. Vault sync ──────────────────────────────────────────────────────────────

def test_refresh_vault_adds_new_files(tmp_path, store):
    """
    On first run against a vault, refresh_vault indexes every .md file it finds.

    Input:  two .md files in the vault root, empty store
    Expected output: (added=2, updated=0, deleted=0)
    """
    (tmp_path / "note_a.md").write_text("# Note A\nContent about machine learning.")
    (tmp_path / "note_b.md").write_text("# Note B\nContent about bioinformatics.")

    added, updated, deleted = refresh_vault(vault_root=tmp_path, store=store)
    assert added == 2
    assert updated == 0
    assert deleted == 0


def test_refresh_vault_reindexes_changed_files(tmp_path, store):
    """
    On a subsequent run, refresh_vault detects files whose SHA-256 hash has
    changed and re-indexes them. Unchanged files are skipped.

    Input:  vault indexed once; one file is then modified
    Expected output: second run returns (added=0, updated=1, deleted=0)
    """
    note = tmp_path / "note.md"
    note.write_text("# Note\nOriginal content about neural networks.")
    refresh_vault(vault_root=tmp_path, store=store)

    note.write_text("# Note\nCompletely rewritten — now about genomics instead.")
    added, updated, deleted = refresh_vault(vault_root=tmp_path, store=store)
    assert added == 0
    assert updated == 1
    assert deleted == 0


def test_refresh_vault_removes_deleted_files(tmp_path, store):
    """
    When a previously indexed .md file no longer exists on disk, refresh_vault
    removes its chunks from the store.

    Input:  vault indexed once; one file is then deleted from disk
    Expected output: second run returns (added=0, updated=0, deleted=1)
    """
    note = tmp_path / "note.md"
    note.write_text("# Note\nContent that will be removed from the vault.")
    refresh_vault(vault_root=tmp_path, store=store)

    note.unlink()
    added, updated, deleted = refresh_vault(vault_root=tmp_path, store=store)
    assert added == 0
    assert updated == 0
    assert deleted == 1


def test_refresh_vault_preserves_pdf_notes(tmp_path, store):
    """
    PDF notes (doc_type="note", file_path is an absolute .pdf path) must NOT
    be deleted by the vault .md scan in Phase 1 of refresh_vault.

    Before the bug fix, Phase 1 built the `indexed` dict from ALL doc_type="note"
    entries, including PDF notes whose file_path is an absolute path like
    "/tmp/.../paper.pdf". The deletion sweep then checked whether each indexed
    path appeared in `current` (relative .md paths from vault_root.rglob). An
    absolute path never matched, so every PDF note was silently deleted on every
    refresh_vault call.

    Input:  a PDF note in the store; a vault directory with no .md files
    Expected output: refresh_vault returns deleted=0; the PDF note remains present
    """
    # Simulate a PDF note as stored by 'kb add --doc-type note'.
    # The content_hash must match the actual file so Phase 2 skips re-indexing —
    # we only want to test Phase 1's deletion behaviour here.
    import hashlib
    fake_pdf = tmp_path / "paper.pdf"
    pdf_bytes = b"%PDF fake content"
    fake_pdf.write_bytes(pdf_bytes)
    content_hash = hashlib.sha256(pdf_bytes).hexdigest()
    add_texts(
        content="This is the converted text of a PDF research note.",
        doc_type="note",
        visibility="public",
        source=fake_pdf.as_uri(),
        extra_metadata={
            "title": "My PDF Note",
            "file_path": str(fake_pdf),  # absolute path — this is the trigger for the bug
            "content_hash": content_hash,
            "storage_mode": "full_text",
        },
        store=store,
    )

    # Run refresh_vault against an empty vault — nothing on disk, so Phase 1's
    # `current` dict is empty. The PDF note must survive the deletion sweep.
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()

    added, updated, deleted = refresh_vault(vault_root=vault_dir, store=store)
    assert deleted == 0, "Phase 1 must not delete PDF notes — their absolute paths are never in current"

    # Confirm the PDF note is still retrievable from the store
    result = store._collection.get(
        where={"doc_type": {"$eq": "note"}}, include=["metadatas"]
    )
    stored_paths = [m.get("file_path", "") for m in result["metadatas"]]
    assert str(fake_pdf) in stored_paths


# ── 8. Embedding-model guard ────────────────────────────────────────────────────

def _tagged_store(embeddings, embed_model_tag: str) -> Chroma:
    """A fresh isolated collection tagged with a given embed_model in its metadata."""
    TEST_CHROMA_DIR.mkdir(exist_ok=True)
    return Chroma(
        collection_name=f"test_{uuid.uuid4().hex[:8]}",
        embedding_function=embeddings,
        persist_directory=str(TEST_CHROMA_DIR),
        collection_metadata={"embed_model": embed_model_tag},
    )


def test_embedding_guard_raises_on_model_mismatch(embeddings):
    """
    A non-empty collection built with one embedding model must not be searched
    with a different model — the vectors live in incompatible spaces. The guard
    raises RAGError so the mismatch fails loudly instead of returning garbage.
    """
    store = _tagged_store(embeddings, "some-old-model")
    add_texts(content="Attention mechanisms weight input tokens.", doc_type="note",
              visibility="public", source="d1", store=store)
    try:
        with pytest.raises(RAGError):
            _check_embedding_model_matches(store, "BAAI/bge-small-en-v1.5")
    finally:
        store.delete_collection()


def test_embedding_guard_allows_matching_model(embeddings):
    """A collection tagged with the model in use passes the guard silently."""
    store = _tagged_store(embeddings, "BAAI/bge-small-en-v1.5")
    add_texts(content="Attention mechanisms weight input tokens.", doc_type="note",
              visibility="public", source="d1", store=store)
    try:
        _check_embedding_model_matches(store, "BAAI/bge-small-en-v1.5")  # no raise
    finally:
        store.delete_collection()


def test_embedding_guard_ignores_empty_collection(embeddings):
    """An empty collection has no vectors to be incompatible, so the guard is a no-op."""
    store = _tagged_store(embeddings, "some-old-model")
    try:
        _check_embedding_model_matches(store, "BAAI/bge-small-en-v1.5")  # no raise
    finally:
        store.delete_collection()


# ── 9. Re-ranking ────────────────────────────────────────────────────────────

@pytest.fixture
def rerank_on(monkeypatch):
    """Force default retrieval config (re-ranking enabled) regardless of ~/.seshat/config.toml."""
    monkeypatch.setattr("digest.kb.store.get_config", lambda: Config())


def test_reranked_search_still_excludes_private_docs(store, rerank_on):
    """
    Re-ranking runs after the ChromaDB visibility filter, so it must never
    surface a private document — the privacy invariant holds through the
    re-rank path just as it does for plain similarity search.
    """
    content = "Backpropagation computes gradients through the chain rule."
    add_texts(content=content, doc_type="note", visibility="public",
              source="pub-rr", store=store)
    add_texts(content=content, doc_type="note", visibility="private",
              source="priv-rr", store=store)

    results = search("backpropagation chain rule gradients", n_results=10,
                     visibility="public", store=store)
    sources = {doc.metadata["source"] for doc in results}
    assert "pub-rr" in sources
    assert "priv-rr" not in sources


def test_search_rerank_false_skips_reranker(store, monkeypatch):
    """
    rerank=False must return plain similarity results without invoking the
    cross-encoder at all (used for the cheap private-existence probe).
    """
    def fail_if_called():
        raise AssertionError("reranker must not be loaded when rerank=False")

    monkeypatch.setattr("digest.kb.store._get_reranker", fail_if_called)
    add_texts(content="Recurrent networks process sequences step by step.",
              doc_type="note", visibility="public", source="seq", store=store)
    results = search("recurrent sequence model", n_results=3, store=store, rerank=False)
    assert any(doc.metadata["source"] == "seq" for doc in results)


# ── 10. Chunk metadata ─────────────────────────────────────────────────────────

def test_chunks_get_distinct_index_and_section_breadcrumb(store):
    """
    Markdown-aware chunking gives every chunk its own metadata: a monotonic
    chunk_index (proving the metadata dict is not shared by reference across
    chunks) and a section breadcrumb built from the markdown headers above it.
    """
    content = (
        "# Research log\n\n"
        "## Methods\nWe designed a guide RNA library targeting 500 genes.\n\n"
        "## Results\nTwelve genes reduced proliferation when knocked out.\n"
    )
    add_texts(content=content, doc_type="note", visibility="public",
              source="note-sections", store=store)

    stored = store._collection.get(
        where={"source": {"$eq": "note-sections"}}, include=["metadatas"]
    )
    metadatas = stored["metadatas"]
    indices = sorted(m["chunk_index"] for m in metadatas)
    assert indices == list(range(len(metadatas)))  # 0..n-1, all distinct

    sections = " ".join(m["section"] for m in metadatas)
    assert "Methods" in sections
    assert "Results" in sections


def test_headerless_content_has_empty_section_and_no_breadcrumb(store):
    """
    Content without markdown headers (e.g. a paper summary) passes through as a
    single unlabelled chunk — empty section, and no breadcrumb prepended to the
    embedded text.
    """
    add_texts(content="Plain prose with no markdown headers whatsoever.",
              doc_type="note", visibility="public", source="plain", store=store)

    stored = store._collection.get(
        where={"source": {"$eq": "plain"}}, include=["metadatas", "documents"]
    )
    assert stored["metadatas"][0]["section"] == ""
    assert stored["metadatas"][0]["chunk_index"] == 0
    assert stored["documents"][0].startswith("Plain prose")
