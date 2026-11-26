"""
Tests for jarvis/kb/store.py — storing chunks and searching them.

All tests receive an isolated ChromaDB collection via the `store` fixture
(see conftest.py) and pass it explicitly using the store= parameter that
every KB function accepts. The process-wide get_store() singleton is never
touched here, except by the tests of get_store() itself.

Real HuggingFace embeddings are used throughout — see conftest.py for the
rationale. Semantically distinct content is used so similarity search returns
meaningful results. Which files get indexed, and when, is tested in
test_mirror.py.
"""

import uuid
from pathlib import Path

import pytest
from langchain_chroma import Chroma

from jarvis.core.config import Config
from jarvis.core.errors import KBCorruptionError, RAGError
from jarvis.kb.store import (
    _check_embedding_model_matches,
    _reciprocal_rank_fusion,
    add_annotations,
    add_texts,
    build_text_documents,
    commit_documents,
    count,
    delete_by_metadata,
    get_document_chunks,
    list_documents,
    search,
    update_file_path,
)

# Same gitignored store directory the shared fixtures use (see conftest.py).
TEST_CHROMA_DIR = Path(__file__).parent / ".chroma"


# ── 0. Staged writes ───────────────────────────────────────────────────────────
#
# Building Documents and committing them are separate steps so a caller can
# assemble a whole document — body, annotations, figure captions — and write it
# in one go. An ingest abandoned before the commit must leave nothing behind.


def test_building_documents_writes_nothing(store):
    """
    The build half must not touch the database at all. This is what makes an
    abandoned ingest safe: whatever was built is simply dropped.
    """
    before = count(store)
    documents = build_text_documents(
        content="Staged but never committed.",
        doc_type="paper", visibility="public", source="file:///staged.pdf",
    )
    assert documents, "the build step should have produced chunks"
    assert count(store) == before


def test_commit_documents_replaces_and_adds_together(store):
    """
    A re-ingest passes replace_source, and the delete of the old entry plus
    the add of the new one happen inside one write — so the entry is never
    observable as "old gone, new not yet here", and a failure before the
    commit leaves the old entry intact.
    """
    source = "file:///paper.pdf"
    add_texts(
        content="The original body text.",
        doc_type="paper", visibility="public", source=source,
        extra_metadata={"title": "Original"},
        store=store,
    )

    replacement = build_text_documents(
        content="The replacement body text.",
        doc_type="paper", visibility="public", source=source,
        extra_metadata={"title": "Replacement"},
    )
    ids = commit_documents(replacement, store, replace_source=source)
    assert ids

    surviving = store._collection.get(
        where={"source": {"$eq": source}}, include=["documents", "metadatas"]
    )
    assert all("original" not in text.lower() for text in surviving["documents"])
    assert {m["title"] for m in surviving["metadatas"]} == {"Replacement"}


def test_commit_documents_with_nothing_staged_still_replaces(store):
    """
    An empty commit is not a no-op when replace_source is set — a document
    whose new content chunked to nothing must still supersede the old entry
    rather than silently leaving stale chunks behind.
    """
    source = "file:///empty.pdf"
    add_texts(
        content="Stale content to be superseded.",
        doc_type="paper", visibility="public", source=source,
        store=store,
    )
    assert commit_documents([], store, replace_source=source) == []
    assert store._collection.get(where={"source": {"$eq": source}}, include=[])["ids"] == []


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

def test_list_documents_deduplicates_and_reports_chunk_count(store):
    """
    list_documents() returns one entry per unique source URL regardless of how many
    chunks that document was split into. The entry includes a chunk_count field
    that matches the actual number of stored chunks.

    Input:  a long summary that the splitter divides into multiple chunks
    Expected output:
        exactly one entry for that source in list_documents()
        entry["chunk_count"] == number of IDs returned by add_texts()
    """
    source = "file:///papers/attention.pdf"
    # Repeat a sentence to exceed the default chunk_size and force multiple chunks
    long_text = "Self-attention computes pairwise interactions between all tokens. " * 200
    ids = add_texts(content=long_text, doc_type="paper", visibility="public",
                    source=source, extra_metadata={"title": "Attention"}, store=store)
    assert len(ids) > 1, "the text should have produced multiple chunks"

    papers = list_documents(store=store)
    matches = [p for p in papers if p["source"] == source]
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


def test_get_document_chunks_orders_body_before_annotations(store):
    """
    get_document_chunks returns body chunks first (sorted by chunk_index),
    then annotation/figure chunks — identified by the annotation_kind
    metadata key, which body chunks never carry.

    Input:  one document with 3 markdown sections plus a highlight and a
        figure caption, all sharing the same source
    Expected output: the 3 body chunks (in section order) followed by the
        2 annotation chunks
    """
    source = "file:///doc.pdf"
    content = (
        "## Intro\nFirst section content about wombats and their burrows.\n\n"
        "## Method\nSecond section content about survey methodology used.\n\n"
        "## Results\nThird section content about population estimates found.\n"
    )
    add_texts(content=content, doc_type="paper", visibility="public",
              source=source, store=store)
    add_texts(content="[HIGHLIGHT p.1] a key finding", doc_type="paper",
              visibility="public", source=source,
              extra_metadata={"annotation_kind": "highlight", "page": 1, "note_text": ""},
              store=store)
    add_texts(content="[FIGURE p.2] a captioned figure", doc_type="paper",
              visibility="public", source=source,
              extra_metadata={"annotation_kind": "figure", "page": 2, "note_text": ""},
              store=store)

    chunks = get_document_chunks(source, store=store)

    assert len(chunks) == 5
    body, annotations = chunks[:3], chunks[3:]
    assert all("annotation_kind" not in doc.metadata for doc in body)
    assert [doc.metadata["chunk_index"] for doc in body] == [0, 1, 2]
    assert {doc.metadata["annotation_kind"] for doc in annotations} == {"highlight", "figure"}


def test_get_document_chunks_unknown_source_returns_empty_list(store):
    """
    get_document_chunks returns [] for a source with no matching chunks,
    mirroring update_file_path's "0 for unknown" behaviour rather than raising.

    Input:  a source string that was never indexed
    Expected output: []
    """
    assert get_document_chunks("file:///does/not/exist.pdf", store=store) == []


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
    """Force default retrieval config (re-ranking enabled) regardless of ~/.jarvis/config.toml."""
    monkeypatch.setattr("jarvis.kb.store.get_config", lambda: Config())


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

    monkeypatch.setattr("jarvis.kb.store._get_reranker", fail_if_called)
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


# ── 11. PDF annotations ────────────────────────────────────────────────────────

def _annotated_pdf(tmp_path: Path) -> Path:
    """PDF with one highlighted-and-commented passage and one sticky note."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Mitochondria are the powerhouse of the cell.", fontsize=12)
    highlight = page.add_highlight_annot(page.search_for("powerhouse of the cell", quads=True))
    highlight.set_info(content="key claim")
    highlight.update()
    page.add_text_annot((72, 150), "Check the original 1957 reference")
    pdf_path = tmp_path / "annotated.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_add_annotations_indexes_highlights_and_comments(store, tmp_path):
    """
    add_annotations turns each highlight/comment into its own chunk with
    annotation metadata and the [HIGHLIGHT]/[USER NOTE] page prefix embedded.
    """
    pdf_path = _annotated_pdf(tmp_path)
    ids = add_annotations(
        pdf_path, doc_type="paper", visibility="public",
        source="file:///annotated.pdf", title="Annotated", store=store,
    )
    assert len(ids) == 2

    highlights = store._collection.get(
        where={"annotation_kind": {"$eq": "highlight"}},
        include=["metadatas", "documents"],
    )
    assert len(highlights["ids"]) == 1
    assert highlights["documents"][0].startswith("[HIGHLIGHT p.1]")
    assert "powerhouse of the cell" in highlights["documents"][0]
    assert "User note: key claim" in highlights["documents"][0]
    assert highlights["metadatas"][0]["page"] == 1

    comments = store._collection.get(
        where={"annotation_kind": {"$eq": "comment"}}, include=["documents"]
    )
    assert comments["documents"][0] == "[USER NOTE p.1] Check the original 1957 reference"


def test_add_annotations_unannotated_pdf_is_noop(store, tmp_path):
    """A PDF without annotations adds nothing and returns []."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Nothing marked here.", fontsize=12)
    pdf_path = tmp_path / "plain.pdf"
    doc.save(pdf_path)
    doc.close()

    before = count(store)
    assert add_annotations(
        pdf_path, doc_type="paper", visibility="public",
        source="file:///plain.pdf", store=store,
    ) == []
    assert count(store) == before


def test_delete_by_source_sweeps_body_and_annotations(store, tmp_path):
    """
    Annotation chunks share the parent PDF's source, so deleting by source
    removes body text and annotations together.
    """
    pdf_path = _annotated_pdf(tmp_path)
    source = "file:///swept.pdf"
    add_texts(content="Body text of the swept paper.", doc_type="paper",
              visibility="public", source=source, store=store)
    add_annotations(pdf_path, doc_type="paper", visibility="public",
                    source=source, store=store)

    deleted = delete_by_metadata("source", source, store)
    assert deleted == 3  # 1 body chunk + 2 annotation chunks
    remaining = store._collection.get(where={"source": {"$eq": source}}, include=[])
    assert remaining["ids"] == []


def test_search_annotation_kind_filter(store, tmp_path):
    """
    search(annotation_kind="highlight") returns only highlight chunks even
    when body chunks about the same topic exist.
    """
    pdf_path = _annotated_pdf(tmp_path)
    add_texts(content="Long discussion about cellular energy production.",
              doc_type="paper", visibility="public", source="file:///cells.pdf", store=store)
    add_annotations(pdf_path, doc_type="paper", visibility="public",
                    source="file:///cells.pdf", store=store)

    results = search("cell energy powerhouse", n_results=5,
                     annotation_kind="highlight", store=store, rerank=False)
    assert results
    assert all(doc.metadata["annotation_kind"] == "highlight" for doc in results)


# ── 12. Corruption detection ────────────────────────────────────────────────────

def test_search_raises_kb_corruption_error_on_id_lookup_failure(store, monkeypatch):
    """
    A ChromaDB "Error finding id" failure (a stale HNSW reference to a deleted
    chunk) must surface as KBCorruptionError naming `kb rebuild` as the fix —
    not a generic RAGError the LLM would paraphrase into something useless.
    """
    monkeypatch.setattr("jarvis.kb.store.get_config", lambda: Config(hybrid=False))

    def broken_similarity_search(*args, **kwargs):
        raise Exception("chromadb.errors.InternalError: Error finding id abcd1234")

    monkeypatch.setattr(store, "similarity_search", broken_similarity_search)

    with pytest.raises(KBCorruptionError, match="kb rebuild"):
        search("anything", store=store, rerank=False)


def test_search_other_failures_still_raise_plain_rag_error(store, monkeypatch):
    """
    Any other similarity_search failure stays a plain RAGError — only the
    stable "Error finding id" substring gets the corruption diagnosis.
    """
    monkeypatch.setattr("jarvis.kb.store.get_config", lambda: Config(hybrid=False))

    def broken_similarity_search(*args, **kwargs):
        raise Exception("connection reset by peer")

    monkeypatch.setattr(store, "similarity_search", broken_similarity_search)

    with pytest.raises(RAGError) as exc_info:
        search("anything", store=store, rerank=False)
    assert not isinstance(exc_info.value, KBCorruptionError)


def test_search_raises_kb_corruption_error_on_hybrid_path(store, monkeypatch):
    """
    The corruption diagnosis must also fire on the hybrid (default) path, not
    just the plain similarity_search branch — hybrid=True is the default
    config, so this is the path most users actually hit.
    """
    monkeypatch.setattr("jarvis.kb.store.get_config", lambda: Config(hybrid=True))
    # The hybrid path only reaches the dense query() call if the candidate
    # pool is non-empty, so seed one chunk first.
    add_texts(content="Graph neural networks aggregate messages over edges.",
              doc_type="note", visibility="public", source="hybrid-corruption-seed", store=store)

    def broken_query(*args, **kwargs):
        raise Exception("chromadb.errors.InternalError: Error finding id abcd1234")

    monkeypatch.setattr(store._collection, "query", broken_query)

    with pytest.raises(KBCorruptionError, match="kb rebuild"):
        search("anything", store=store, rerank=False)


def test_search_diagnoses_stale_collection_handle_after_rebuild(store, monkeypatch):
    """
    `kb rebuild` swaps in a rebuilt collection with a new UUID; a process that
    was already running still holds the old handle, and every operation then
    fails with "Collection [...] does not exist". That must surface as a
    diagnosis naming the fix (restart the process) — not as a generic
    RAGError, and definitely not as advice to run a rebuild again.
    """
    monkeypatch.setattr("jarvis.kb.store.get_config", lambda: Config(hybrid=False))

    def stale_similarity_search(*args, **kwargs):
        raise Exception(
            "Error getting collection: Collection "
            "[6e8c78c5-ace7-4ab3-a40e-e3e0ac3bd8fb] does not exist."
        )

    monkeypatch.setattr(store, "similarity_search", stale_similarity_search)

    with pytest.raises(KBCorruptionError, match="restart this process"):
        search("anything", store=store, rerank=False)


def test_add_texts_diagnoses_stale_collection_handle_after_rebuild(store, monkeypatch):
    """The same stale-handle diagnosis must fire on the write path — that is
    where a long-lived process hits it when it next indexes something."""

    def stale_add_documents(*args, **kwargs):
        raise Exception(
            "Failed to add documents: Error getting collection: Collection "
            "[6e8c78c5-ace7-4ab3-a40e-e3e0ac3bd8fb] does not exist."
        )

    monkeypatch.setattr(store, "add_documents", stale_add_documents)

    with pytest.raises(KBCorruptionError, match="restart this process"):
        add_texts(content="anything", doc_type="note", visibility="public",
                  source="stale-handle-test", store=store)


# ── 13. Embed header ─────────────────────────────────────────────────────────────

def test_embed_header_is_prepended_to_every_chunk(store):
    """
    embed_header must be part of the EMBEDDED text of every chunk, not just
    the first — an author-name query has to be able to match any chunk of a
    long paper, since the author's name appears in no body chunk otherwise.
    """
    long_content = "Self-attention computes pairwise interactions between all tokens. " * 200
    add_texts(
        content=long_content, doc_type="paper", visibility="public",
        source="header-test", store=store,
        embed_header="A Long Paper — Jane Doe, John Smith",
    )
    stored = store._collection.get(where={"source": {"$eq": "header-test"}}, include=["documents"])
    assert len(stored["documents"]) > 1, "content should have produced multiple chunks"
    assert all(doc.startswith("A Long Paper — Jane Doe, John Smith") for doc in stored["documents"])


# ── 14. Hybrid BM25 + RRF retrieval ───────────────────────────────────────────────

def test_hybrid_disabled_reproduces_plain_similarity_search(store, monkeypatch):
    """hybrid=False takes the untouched similarity_search branch, byte for byte."""
    monkeypatch.setattr("jarvis.kb.store.get_config", lambda: Config(hybrid=False))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("_hybrid_search must not be called when hybrid=False")

    monkeypatch.setattr("jarvis.kb.store._hybrid_search", fail_if_called)
    add_texts(content="Recurrent networks process sequences step by step.",
              doc_type="note", visibility="public", source="seq-hybrid-off", store=store)
    results = search("recurrent sequence model", n_results=3, store=store, rerank=False)
    assert any(doc.metadata["source"] == "seq-hybrid-off" for doc in results)


def test_hybrid_enabled_uses_hybrid_search_and_returns_expected_doc(store, monkeypatch):
    """hybrid=True (the default) routes through _hybrid_search and still finds the doc."""
    from jarvis.kb import store as store_module

    monkeypatch.setattr("jarvis.kb.store.get_config", lambda: Config(hybrid=True))
    calls = {"n": 0}
    original = store_module._hybrid_search

    def spy(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr("jarvis.kb.store._hybrid_search", spy)
    add_texts(content="Diffusion models learn to reverse a gradual noising process.",
              doc_type="note", visibility="public", source="diffusion-hybrid-on", store=store)
    results = search("diffusion denoising model", n_results=3, store=store, rerank=False)
    assert calls["n"] == 1
    assert any(doc.metadata["source"] == "diffusion-hybrid-on" for doc in results)


def test_reciprocal_rank_fusion_favours_ids_ranked_high_in_both_lists():
    """A direct unit test of the fusion math, independent of any store."""
    dense = ["a", "b", "c", "d"]
    sparse = ["b", "a", "d", "c"]
    fused = _reciprocal_rank_fusion(dense, sparse)
    # "a" and "b" are ranked 1st/2nd in both lists, so they must both land
    # ahead of "c" and "d", which rank lower in both.
    assert set(fused[:2]) == {"a", "b"}
    assert set(fused[2:]) == {"c", "d"}


# ── 16. doc_type list filter (papers + digests) ────────────────────────────────

def test_search_doc_type_list_returns_papers_and_digests(store):
    """
    search() with a doc_type LIST must return documents of every listed type
    and still exclude the rest — this is what keeps digest documents indexed
    by older versions findable under the "papers" kind until a rebuild.

    All three documents share the same content, so the doc_type filter (not
    relevance) must do the discrimination.
    """
    content = "Spatial transcriptomics maps gene expression onto tissue coordinates."
    add_texts(content=content, doc_type="paper", visibility="public",
              source="paper-src", store=store)
    add_texts(content=content, doc_type="digest", visibility="public",
              source="file:///digests/digest-2026-07-06.md", store=store)
    add_texts(content=content, doc_type="note", visibility="public",
              source="note-src", store=store)

    results = search("spatial transcriptomics tissue", n_results=10,
                     doc_type=["paper", "digest"], store=store, rerank=False)
    types = {doc.metadata["doc_type"] for doc in results}
    assert types == {"paper", "digest"}

    # A plain string still narrows to that single type.
    only_papers = search("spatial transcriptomics tissue", n_results=10,
                         doc_type="paper", store=store, rerank=False)
    assert {doc.metadata["doc_type"] for doc in only_papers} == {"paper"}


# ── Who owns the index ─────────────────────────────────────────────────────────
#
# The knowledge-base server owns the index; everything else connects to it.
# These cover the two things that decision has to get right: a long-lived
# process must refuse to run without the server rather than quietly opening
# the files itself, and a one-shot command must still work when it is down.


def test_get_store_refuses_to_open_directly_when_the_server_is_down(monkeypatch, tmp_path):
    """
    jarvis-mcp and kb serve's sync must fail loudly, naming the fix. Falling
    back to opening the files would recreate the bug the server exists to
    remove: a long-lived reader caching a copy of the vector index while
    another process rewrites it.
    """
    from jarvis.kb import store as store_module

    monkeypatch.setattr(store_module, "_store", None)
    monkeypatch.setattr(store_module, "get_config", lambda: Config(rag_dir=tmp_path))

    def no_server(port):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(store_module, "_server_client", no_server)

    with pytest.raises(RAGError, match="uv run kb serve"):
        store_module.get_store()


def test_get_store_falls_back_to_the_files_for_one_shot_commands(monkeypatch, tmp_path):
    """
    `kb doctor` and friends have to work when the server is the broken thing,
    so they opt into opening the directory themselves. Safe because they exit
    in seconds — a process that short-lived cannot go stale.
    """
    from jarvis.kb import store as store_module

    monkeypatch.setattr(store_module, "_store", None)
    monkeypatch.setattr(store_module, "get_config", lambda: Config(rag_dir=tmp_path))

    def no_server(port):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(store_module, "_server_client", no_server)

    opened = store_module.get_store(allow_direct=True)
    assert opened.add_documents is not None  # a usable store, not an error
    store_module._store = None


def test_the_index_directory_is_owner_only(monkeypatch, tmp_path):
    """
    The index holds the full text of private notes, so it gets the same 0700
    treatment as sessions and drafts. Applied on every open, not just at
    creation, so an index that predates this is fixed by using it.
    """
    import os
    import stat

    from jarvis.kb import store as store_module

    rag_dir = tmp_path / "rag"
    rag_dir.mkdir()
    os.chmod(rag_dir, 0o755)

    monkeypatch.setattr(store_module, "_store", None)
    monkeypatch.setattr(store_module, "get_config", lambda: Config(rag_dir=rag_dir))
    monkeypatch.setattr(store_module, "_server_client",
                        lambda port: (_ for _ in ()).throw(ConnectionError("connection refused")))

    store_module.get_store(allow_direct=True)
    store_module._store = None

    assert stat.S_IMODE(rag_dir.stat().st_mode) == 0o700


def test_a_connection_reset_is_not_reported_as_a_stopped_server(store, monkeypatch):
    """
    The server-down message names a fix ("start it"), so it must not swallow
    unrelated connection failures. "connection reset by peer" contains the
    substring "connect" — matching on that alone would misdiagnose it.
    """
    monkeypatch.setattr("jarvis.kb.store.get_config", lambda: Config(hybrid=False))

    def broken_similarity_search(*args, **kwargs):
        raise Exception("connection reset by peer")

    monkeypatch.setattr(store, "similarity_search", broken_similarity_search)

    with pytest.raises(RAGError) as caught:
        search("anything", store=store, rerank=False)
    assert "uv run kb serve" not in str(caught.value)


def test_telemetry_is_disabled_before_any_library_is_imported():
    """
    Importing jarvis must switch off third-party reporting. HuggingFace reads
    its variable at import time, so setting it later would be too late.
    """
    import os

    import jarvis  # noqa: F401  (the import is the thing under test)

    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_the_chroma_client_is_built_with_telemetry_off():
    """Chroma's knob is a constructor argument, not an environment variable."""
    from jarvis.kb.store import _chroma_settings

    assert _chroma_settings().anonymized_telemetry is False
