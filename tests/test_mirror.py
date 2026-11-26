"""
Tests for jarvis/kb/mirror.py — the index as a mirror of two folders.

The vault and the papers folder are the only sources: adding, changing,
moving or deleting a file is how the index changes. These tests drive the
sync against real files in tmp_path and a real isolated collection (the
`store` fixture), and check both what lands in the index and what must never
land there (private notes, hidden folders) or never be deleted (everything,
when a folder is missing).

Sections
--------
1. Vault notes: add / change / delete, private and hidden folders
2. Records: frontmatter indexing, filtered search, schema migration
3. Papers: add / skip / change / move / delete, safety rules, metadata
4. rebuild_index
"""

from pathlib import Path

import pymupdf
import pytest

from jarvis.core.config import Config
from jarvis.kb.mirror import (
    index_vault_file,
    rebuild_index,
    refresh_papers,
    refresh_vault,
    sync_all,
)
from jarvis.kb.store import search


def _write_record(vault, rel_path, frontmatter, body):
    path = vault / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\n{frontmatter}\n---\n\n{body}\n", encoding="utf-8")
    return path


def _write_pdf(path: Path, text: str, highlight: str = "") -> Path:
    """A one-page PDF with the given text, optionally with one phrase highlighted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=11)
    if highlight:
        page.add_highlight_annot(page.search_for(highlight, quads=True))
    doc.save(path)
    doc.close()
    return path


def _chunks(store, **where):
    """(ids, metadatas) of the chunks matching one metadata key=value."""
    (key, value), = where.items()
    result = store._collection.get(where={key: {"$eq": value}}, include=["metadatas"])
    return result["ids"], result["metadatas"]


@pytest.fixture(autouse=True)
def _no_network_metadata(monkeypatch):
    """Crossref and arXiv are HTTP boundaries; tests that want them patch them themselves."""
    monkeypatch.setattr("jarvis.kb.metadata.crossref_lookup", lambda doi: {})
    monkeypatch.setattr("jarvis.kb.metadata.arxiv_lookup", lambda arxiv_id: {})


@pytest.fixture(autouse=True)
def _files_are_settled(request, monkeypatch):
    """
    Files written by a test are complete, so skip the ~2s settle poll per new
    PDF. The one test of the poll itself opts out with @pytest.mark.real_settle.
    """
    if "real_settle" not in request.keywords:
        monkeypatch.setattr("jarvis.kb.mirror.wait_for_stable", lambda path: True)


# ── 1. Vault notes ─────────────────────────────────────────────────────────────

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


def test_private_folders_are_never_indexed(tmp_path, store):
    """
    A note under a top-level private folder never reaches the index — not
    even filtered: every MCP client is a cloud model, so the safe place for a
    private note is outside the index altogether.

    Input:  public/open.md and private/secret.md
    Expected output: only open.md is indexed; no chunk mentions the secret
    """
    (tmp_path / "public").mkdir()
    (tmp_path / "private").mkdir()
    (tmp_path / "public" / "open.md").write_text("# Open\nShared project notes.")
    (tmp_path / "private" / "secret.md").write_text("# Secret\nSalary negotiation notes.")

    added, _, _ = refresh_vault(tmp_path, store)

    assert added == 1
    texts = store._collection.get(include=["documents"])["documents"]
    assert not any("Salary" in text for text in texts)


def test_a_note_moved_into_a_private_folder_leaves_the_index(tmp_path, store):
    """
    Moving a note into private/ is how you make it private, so the next pass
    must remove it — there is no separate reclassification step.
    """
    note = tmp_path / "plans.md"
    note.write_text("# Plans\nThings I would rather keep to myself.")
    refresh_vault(tmp_path, store)
    assert _chunks(store, file_path="plans.md")[0]

    (tmp_path / "private").mkdir()
    note.rename(tmp_path / "private" / "plans.md")
    added, updated, deleted = refresh_vault(tmp_path, store)

    assert (added, updated, deleted) == (0, 0, 1)
    assert store._collection.count() == 0


def test_hidden_folders_are_skipped(tmp_path, store):
    """
    Obsidian keeps deleted notes in .trash/ and its own settings in
    .obsidian/; neither is a note the user wants searchable.
    """
    (tmp_path / ".trash").mkdir()
    (tmp_path / ".trash" / "old.md").write_text("# Old\nA note I deleted.")
    (tmp_path / "kept.md").write_text("# Kept\nA note I kept.")

    added, _, _ = refresh_vault(tmp_path, store)

    assert added == 1
    assert _chunks(store, file_path="kept.md")[0]


def test_a_missing_vault_deletes_nothing(tmp_path, store):
    """
    An unmounted drive or a moved vault must never empty the index: the pass
    is skipped, deletions included.
    """
    (tmp_path / "note.md").write_text("# Note\nStill here.")
    refresh_vault(tmp_path, store)

    result = refresh_vault(tmp_path / "gone", store)

    assert result == (0, 0, 0)
    assert store._collection.count() > 0


# ── 2. Records: frontmatter indexing, filtered search, schema migration ───────

def test_vault_note_frontmatter_becomes_filterable_metadata(store, tmp_path):
    """
    A record's frontmatter lands on every chunk as flat metadata, and the
    frontmatter block itself is stripped from the indexed text.
    """
    vault = tmp_path / "vault"
    path = _write_record(
        vault, "records/acme.md",
        "type: job_application\nentity: Acme Bio\nstatus: rejected\n"
        "date: 2026-05-02\ntags: [remote, senior]\nvenue: NeurIPS",
        "# Senior Bioinformatician\n\nRejected after the technical screen.",
    )

    index_vault_file(path, vault, store)

    metadatas = store._collection.get(include=["metadatas"])["metadatas"]
    assert metadatas
    for meta in metadatas:
        assert meta["category"] == "job_application"
        assert meta["entity"] == "Acme Bio"
        assert meta["status"] == "rejected"
        assert meta["event_date"] == "2026-05-02"
        assert meta["tags"] == "|remote|senior|"
        assert meta["x_venue"] == "NeurIPS"
        # Still a plain note as far as every existing privacy/delete path is
        # concerned — records ride on doc_type, they don't replace it.
        assert meta["doc_type"] == "note"
        assert meta["visibility"] == "public"

    documents = store._collection.get(include=["documents"])["documents"]
    assert not any("type: job_application" in d for d in documents)


def test_search_filters_narrow_by_record_fields(store, tmp_path):
    """
    category/status/entity filters select records; they fold into the same
    where-clause as the privacy filter, so they can only ever narrow.
    """
    vault = tmp_path / "vault"
    index_vault_file(
        _write_record(vault, "a.md", "type: job_application\nentity: Acme Bio\nstatus: rejected",
                      "# Acme\n\nBioinformatics role, technical screen."),
        vault, store)
    index_vault_file(
        _write_record(vault, "b.md", "type: job_application\nentity: Beta Labs\nstatus: applied",
                      "# Beta\n\nBioinformatics role, waiting to hear back."),
        vault, store)
    index_vault_file(
        _write_record(vault, "c.md", "type: manuscript\nstatus: drafting",
                      "# Paper\n\nBioinformatics methods manuscript."),
        vault, store)

    rejected = search("bioinformatics role", n_results=10, category="job_application",
                      status="rejected", store=store)
    assert {d.metadata["entity"] for d in rejected} == {"Acme Bio"}

    applications = search("bioinformatics", n_results=10, category="job_application", store=store)
    assert {d.metadata["entity"] for d in applications} == {"Acme Bio", "Beta Labs"}

    manuscripts = search("bioinformatics", n_results=10, category="manuscript", store=store)
    assert all(d.metadata["category"] == "manuscript" for d in manuscripts)


def test_search_filters_on_a_custom_frontmatter_field(store, tmp_path):
    """A key jarvis has never heard of is filterable without any code change."""
    vault = tmp_path / "vault"
    index_vault_file(
        _write_record(vault, "a.md", "type: manuscript\nvenue: NeurIPS", "# A\n\nDraft about retrieval."),
        vault, store)
    index_vault_file(
        _write_record(vault, "b.md", "type: manuscript\nvenue: ICML", "# B\n\nDraft about retrieval."),
        vault, store)

    results = search("retrieval draft", n_results=10, fields={"x_venue": "NeurIPS"}, store=store)
    assert results
    assert all(d.metadata["x_venue"] == "NeurIPS" for d in results)


def test_tag_filter_requires_every_tag(store, tmp_path):
    vault = tmp_path / "vault"
    index_vault_file(
        _write_record(vault, "a.md", "type: job_application\ntags: [remote, senior]",
                      "# A\n\nA role with flexible hours."),
        vault, store)
    index_vault_file(
        _write_record(vault, "b.md", "type: job_application\ntags: [remote]",
                      "# B\n\nA role with flexible hours."),
        vault, store)

    both = search("flexible role", n_results=10, tags=["remote", "senior"], store=store)
    assert {d.metadata["file_path"] for d in both} == {"a.md"}

    either = search("flexible role", n_results=10, tags=["remote"], store=store)
    assert {d.metadata["file_path"] for d in either} == {"a.md", "b.md"}


def test_record_header_is_embedded_so_status_queries_match(store, tmp_path):
    """
    The record header rides in every chunk's embedded text, so a query phrased
    in terms of the record's identity finds it even though the body never uses
    those words. This is the reuse of the papers' embed_header mechanism.
    """
    vault = tmp_path / "vault"
    index_vault_file(
        _write_record(vault, "acme.md", "type: job_application\nentity: Acme Bio\nstatus: rejected",
                      "# Senior Bioinformatician\n\nThey wanted more Rust experience."),
        vault, store)

    results = search("rejected job application", n_results=5, store=store)
    assert any(d.metadata.get("entity") == "Acme Bio" for d in results)


def test_refresh_vault_backfills_notes_indexed_under_an_older_schema(store, tmp_path):
    """
    A note whose meta_schema is behind is treated as changed, so the backfill
    happens on the next daemon sweep with no user action and no migration
    script — even though its content hash is unchanged.
    """
    from jarvis.kb.frontmatter import META_SCHEMA

    vault = tmp_path / "vault"
    path = _write_record(vault, "a.md", "type: job_application\nstatus: rejected", "# A\n\nBody.")
    index_vault_file(path, vault, store)

    # Stamp an older schema version on the stored chunks, standing in for a
    # note indexed before record metadata existed. (Chroma's update merges
    # metadata keys rather than replacing the dict, so the version marker is
    # what the migration actually keys on — which is the point of having one.)
    stored = store._collection.get(include=["metadatas"])
    store._collection.update(
        ids=stored["ids"],
        metadatas=[{**meta, "meta_schema": 0} for meta in stored["metadatas"]],
    )

    added, updated, deleted = refresh_vault(vault, store)

    assert (added, updated, deleted) == (0, 1, 0)
    for meta in store._collection.get(include=["metadatas"])["metadatas"]:
        assert meta["category"] == "job_application"
        assert meta["meta_schema"] == META_SCHEMA


def test_schema_migration_does_not_re_embed_a_note_without_frontmatter(store, tmp_path):
    """
    The common case: a plain note gains nothing from a re-index except the
    schema marker, so it must be stamped in place. Re-embedding an entire
    vault to record that most of it had nothing to record is pure cost — and
    a delete-then-re-add leaves a window where the note is missing.

    Proven by watching the chunk ids: a metadata-only update keeps them, a
    delete-and-re-add cannot.
    """
    vault = tmp_path / "vault"
    plain = vault / "plain.md"
    plain.parent.mkdir(parents=True, exist_ok=True)
    plain.write_text("# Plain\n\nNo frontmatter here.\n", encoding="utf-8")
    index_vault_file(plain, vault, store)

    before = store._collection.get(include=[])["ids"]
    stored = store._collection.get(include=["metadatas"])
    store._collection.update(
        ids=stored["ids"],
        metadatas=[{**m, "meta_schema": 0} for m in stored["metadatas"]],
    )

    assert refresh_vault(vault, store) == (0, 1, 0)

    after = store._collection.get(include=["metadatas"])
    assert sorted(after["ids"]) == sorted(before), "chunks were re-created, not stamped"
    assert all(m["meta_schema"] == 1 for m in after["metadatas"])


def test_schema_migration_re_indexes_a_note_that_has_frontmatter(store, tmp_path):
    """
    A note WITH frontmatter genuinely changes on re-index — the block is now
    stripped from the indexed body and the record header joins the embedded
    text — so this one has to be rebuilt rather than stamped.
    """
    vault = tmp_path / "vault"
    path = _write_record(vault, "a.md", "type: manuscript\nstatus: drafting", "# A\n\nBody.")
    index_vault_file(path, vault, store)

    stored = store._collection.get(include=["metadatas"])
    store._collection.update(
        ids=stored["ids"],
        metadatas=[
            {**{k: v for k, v in m.items()}, "meta_schema": 0, "category": "stale"}
            for m in stored["metadatas"]
        ],
    )

    assert refresh_vault(vault, store) == (0, 1, 0)

    for meta in store._collection.get(include=["metadatas"])["metadatas"]:
        assert meta["category"] == "manuscript"   # re-read from the file
        assert meta["meta_schema"] == 1
    for document in store._collection.get(include=["documents"])["documents"]:
        assert "type: manuscript" not in document   # block stripped from the body


def test_refresh_vault_leaves_current_schema_notes_alone(store, tmp_path):
    """
    Every note carries meta_schema, frontmatter or not — otherwise a plain note
    would look perpetually out of date and re-index on every single sweep.
    """
    vault = tmp_path / "vault"
    plain = vault / "plain.md"
    plain.parent.mkdir(parents=True, exist_ok=True)
    plain.write_text("# Plain\n\nNo frontmatter here.\n", encoding="utf-8")
    refresh_vault(vault, store)

    assert refresh_vault(vault, store) == (0, 0, 0)


def test_metadata_key_and_value_counts_expose_the_user_ontology(store, tmp_path):
    """
    `kb schema` exists so a typo in a status ("stauts") shows up as its own key
    rather than silently never matching a filter.
    """
    from jarvis.kb.store import metadata_key_counts, metadata_value_counts

    vault = tmp_path / "vault"
    index_vault_file(
        _write_record(vault, "a.md", "type: job_application\nstatus: rejected", "# A\n\nBody."),
        vault, store)
    index_vault_file(
        _write_record(vault, "b.md", "type: job_application\nstauts: applied", "# B\n\nBody."),
        vault, store)

    keys = metadata_key_counts(store)
    assert keys["category"] >= 2
    assert "x_stauts" in keys  # the typo is visible, not silently ignored

    statuses = metadata_value_counts("status", store)
    assert set(statuses) == {"rejected"}




# ── 3. Papers ──────────────────────────────────────────────────────────────────

def test_papers_are_added_then_skipped_when_unchanged(tmp_path, store):
    """
    A new PDF is indexed full-text under its file:// URI; a second pass over
    the same unchanged file does nothing — not even a re-embed.
    """
    papers = tmp_path / "papers"
    pdf = _write_pdf(papers / "graphs.pdf", "Graph neural networks for cell type annotation.")

    assert refresh_papers(papers, store)["added"] == 1
    ids_before, metadatas = _chunks(store, source=pdf.as_uri())
    assert ids_before and metadatas[0]["doc_type"] == "paper"
    assert metadatas[0]["file_path"] == str(pdf)

    assert refresh_papers(papers, store) == {"added": 0, "updated": 0, "moved": 0, "deleted": 0}
    assert _chunks(store, source=pdf.as_uri())[0] == ids_before


def test_a_changed_paper_is_reindexed(tmp_path, store):
    """New bytes at the same path (e.g. new highlights saved) replace the entry."""
    papers = tmp_path / "papers"
    pdf = _write_pdf(papers / "p.pdf", "Original text about protein folding.")
    refresh_papers(papers, store)

    _write_pdf(pdf, "Rewritten text about protein folding.", highlight="protein folding")
    counts = refresh_papers(papers, store)

    assert counts["updated"] == 1
    _, metadatas = _chunks(store, source=pdf.as_uri())
    assert any(m.get("annotation_kind") == "highlight" for m in metadatas)


def test_a_moved_paper_is_repointed_without_re_embedding(tmp_path, store):
    """
    Organising papers into subfolders must not cost a re-index: the moved
    file is recognised by its content hash, and its chunks keep their ids.
    """
    papers = tmp_path / "papers"
    pdf = _write_pdf(papers / "inbox.pdf", "Single-cell RNA sequencing atlas of the lung.")
    refresh_papers(papers, store)
    ids_before, _ = _chunks(store, source=pdf.as_uri())

    destination = papers / "lung" / "atlas.pdf"
    destination.parent.mkdir()
    pdf.rename(destination)
    counts = refresh_papers(papers, store)

    assert counts == {"added": 0, "updated": 0, "moved": 1, "deleted": 0}
    ids_after, metadatas = _chunks(store, source=destination.as_uri())
    assert sorted(ids_after) == sorted(ids_before)
    assert all(m["file_path"] == str(destination) for m in metadatas)
    # The next pass sees nothing to do at the new path.
    assert refresh_papers(papers, store)["moved"] == 0


def test_a_file_still_being_written_waits_for_the_next_pass(tmp_path, store, monkeypatch):
    """A PDF that is still changing (a sync client mid-download) is left alone."""
    monkeypatch.setattr("jarvis.kb.mirror.wait_for_stable", lambda path: False)
    papers = tmp_path / "papers"
    _write_pdf(papers / "partial.pdf", "Half a paper.")

    assert refresh_papers(papers, store)["added"] == 0
    assert store._collection.count() == 0


@pytest.mark.real_settle
def test_wait_for_stable_settles_and_gives_up(tmp_path):
    """The poll returns True for a settled file and False for a missing one."""
    from jarvis.kb.mirror import wait_for_stable

    settled = tmp_path / "settled.pdf"
    settled.write_bytes(b"%PDF-1.4 complete")
    assert wait_for_stable(settled, checks=1, interval=0.05, timeout=2) is True
    assert wait_for_stable(tmp_path / "missing.pdf", checks=1, interval=0.05, timeout=1) is False


def test_a_deleted_paper_leaves_the_index(tmp_path, store):
    """Deleting the PDF is how a paper is removed."""
    papers = tmp_path / "papers"
    pdf = _write_pdf(papers / "p.pdf", "A paper about microbiome diversity.")
    refresh_papers(papers, store)

    pdf.unlink()

    assert refresh_papers(papers, store)["deleted"] == 1
    assert store._collection.count() == 0


def test_a_missing_papers_folder_deletes_nothing(tmp_path, store):
    """An unmounted papers folder must leave every paper in place."""
    papers = tmp_path / "papers"
    _write_pdf(papers / "p.pdf", "A paper about tissue imaging.")
    refresh_papers(papers, store)
    before = store._collection.count()

    counts = refresh_papers(tmp_path / "unmounted", store)

    assert counts == {"added": 0, "updated": 0, "moved": 0, "deleted": 0}
    assert store._collection.count() == before


def test_an_icloud_evicted_paper_keeps_its_entry(tmp_path, store):
    """
    iCloud's optimised storage replaces a file with a ".name.pdf.icloud"
    placeholder. The paper still exists, so its entry must survive.
    """
    papers = tmp_path / "papers"
    pdf = _write_pdf(papers / "cloud.pdf", "A paper about spatial proteomics.")
    refresh_papers(papers, store)

    pdf.unlink()
    (papers / ".cloud.pdf.icloud").write_bytes(b"placeholder")

    assert refresh_papers(papers, store)["deleted"] == 0
    assert _chunks(store, source=pdf.as_uri())[0]


def test_papers_indexed_from_elsewhere_are_left_alone(tmp_path, store):
    """
    Entries added by older versions from outside papers_dir are not this
    sync's to delete; `kb doctor` reports them and `kb rebuild` drops them.
    """
    from jarvis.kb.store import add_texts

    add_texts(content="A paper added from the Downloads folder.", doc_type="paper",
              visibility="public", source="file:///Users/x/Downloads/old.pdf",
              extra_metadata={"file_path": "/Users/x/Downloads/old.pdf"}, store=store)
    papers = tmp_path / "papers"
    papers.mkdir()

    assert refresh_papers(papers, store)["deleted"] == 0
    assert _chunks(store, source="file:///Users/x/Downloads/old.pdf")[0]


def test_highlights_survive_a_pdf_with_no_text_layer(tmp_path, store):
    """
    A scanned PDF has no text to convert, but the user's typed notes on it
    are still worth indexing on their own.
    """
    papers = tmp_path / "papers"
    papers.mkdir()
    pdf = papers / "scan.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.add_text_annot((72, 72), "Key figure for the grant renewal.")
    doc.save(pdf)
    doc.close()

    assert refresh_papers(papers, store)["added"] == 1
    _, metadatas = _chunks(store, source=pdf.as_uri())
    assert [m["annotation_kind"] for m in metadatas] == ["comment"]


def test_paper_metadata_comes_from_crossref_when_a_doi_is_printed(tmp_path, store, monkeypatch):
    """
    A DOI on the first page is looked up on Crossref for the canonical title
    and authors, and every chunk carries them.
    """
    looked_up = []

    def fake_crossref(doi):
        looked_up.append(doi)
        return {"title": "Deep Phenotyping of Immune Cells", "authors": "Ada Lovelace, Alan Turing"}

    monkeypatch.setattr("jarvis.kb.metadata.crossref_lookup", fake_crossref)
    papers = tmp_path / "papers"
    pdf = _write_pdf(papers / "x.pdf", "doi: 10.1038/s41586-020-1234-5. Cytometry results.")

    refresh_papers(papers, store)

    assert looked_up == ["10.1038/s41586-020-1234-5"]
    _, metadatas = _chunks(store, source=pdf.as_uri())
    assert {m["title"] for m in metadatas} == {"Deep Phenotyping of Immune Cells"}
    assert {m["doi"] for m in metadatas} == {"10.1038/s41586-020-1234-5"}


def test_arxiv_preprint_records_its_id_and_real_title(tmp_path, store, monkeypatch):
    """
    An arXiv PDF with no DOI is stored with its arXiv ID, and its title comes
    from arXiv instead of the filename.
    """
    monkeypatch.setattr("jarvis.kb.metadata.arxiv_lookup",
                        lambda arxiv_id: {"title": "Sparse Autoencoders", "authors": "A. Lovelace"})
    papers = tmp_path / "papers"
    pdf = _write_pdf(papers / "2411.01234v2.pdf", "arXiv:2411.01234v2 [cs.LG] 3 Nov 2024")

    refresh_papers(papers, store)

    _, metadatas = _chunks(store, source=pdf.as_uri())
    assert {m["arxiv_id"] for m in metadatas} == {"2411.01234"}
    assert {m["title"] for m in metadatas} == {"Sparse Autoencoders"}


def test_papers_indexed_before_arxiv_ids_are_upgraded_once(tmp_path, store, monkeypatch):
    """
    A paper indexed by an older version has no arxiv_id. The next sync
    re-indexes it if its PDF has an arXiv ID, and otherwise stores an empty
    one, so the PDF is not read again on later passes.
    """
    papers = tmp_path / "papers"
    preprint = _write_pdf(papers / "2411.01234v2.pdf", "arXiv:2411.01234v2 [cs.LG] 3 Nov 2024")
    journal = _write_pdf(papers / "journal.pdf", "doi: 10.1038/s41586-020-1234-5. Results.")
    refresh_papers(papers, store)

    # Make both look like they came from the older version.
    result = store._collection.get(include=["metadatas"])
    # Chroma merges metadata on update, and a None value deletes the key.
    old_metadatas = [{**m, "arxiv_id": None} for m in result["metadatas"]]
    store._collection.update(ids=result["ids"], metadatas=old_metadatas)
    assert all("arxiv_id" not in m for m in store._collection.get(include=["metadatas"])["metadatas"])

    monkeypatch.setattr("jarvis.kb.metadata.arxiv_lookup",
                        lambda arxiv_id: {"title": "Sparse Autoencoders", "authors": "A. Lovelace"})
    assert refresh_papers(papers, store)["updated"] == 1

    _, preprint_metadatas = _chunks(store, source=preprint.as_uri())
    assert {m["arxiv_id"] for m in preprint_metadatas} == {"2411.01234"}
    assert {m["title"] for m in preprint_metadatas} == {"Sparse Autoencoders"}
    _, journal_metadatas = _chunks(store, source=journal.as_uri())
    assert {m["arxiv_id"] for m in journal_metadatas} == {""}

    # Already upgraded: the next pass reads neither PDF.
    def must_not_read(pdf_path, *args, **kwargs):
        raise AssertionError(f"{pdf_path} was read again")

    monkeypatch.setattr("jarvis.kb.metadata.extract_arxiv_id", must_not_read)
    assert refresh_papers(papers, store)["updated"] == 0


def test_sync_all_covers_both_folders(tmp_path, store):
    """One pass reports notes and papers together."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "n.md").write_text("# N\nA note.")
    _write_pdf(tmp_path / "papers" / "p.pdf", "A paper.")

    counts = sync_all(Config(vault_path=vault, papers_dir=tmp_path / "papers"), store)

    assert counts["notes_added"] == 1 and counts["papers_added"] == 1


# ── 4. rebuild_index ───────────────────────────────────────────────────────────

def test_rebuild_keeps_only_what_the_folders_back(tmp_path, monkeypatch):
    """
    A rebuild recreates the index from the two folders and swaps it in. An
    entry with no file behind it (an older version's arXiv-URL paper) is gone
    afterwards; the notes and papers on disk are all there.

    Runs on the direct path against a throwaway rag_dir, with the server
    forced "down" so it can never touch a real running index.
    """
    import chromadb

    from jarvis.kb import store as store_module
    from jarvis.kb.store import COLLECTION_NAME, add_texts, open_collection

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "kept.md").write_text("# Kept\nA note backed by a file.")
    papers = tmp_path / "papers"
    _write_pdf(papers / "p.pdf", "A paper backed by a file.")
    config = Config(rag_dir=tmp_path / "rag", vault_path=vault, papers_dir=papers)

    monkeypatch.setattr(store_module, "get_config", lambda: config)
    monkeypatch.setattr("jarvis.kb.mirror.get_config", lambda: config)
    monkeypatch.setattr(store_module, "_server_client",
                        lambda port: (_ for _ in ()).throw(ConnectionError("connection refused")))
    monkeypatch.setattr(store_module, "_allow_direct", True)
    monkeypatch.setattr(store_module, "_store", None)

    # The live collection holds one legacy entry with no file behind it.
    client = chromadb.PersistentClient(path=str(config.rag_dir), settings=store_module._chroma_settings())
    add_texts(content="An arXiv paper added by URL long ago.", doc_type="paper",
              visibility="public", source="https://arxiv.org/abs/2301.00001",
              store=open_collection(client, COLLECTION_NAME))

    counts = rebuild_index(config)

    assert counts["notes_added"] == 1 and counts["papers_added"] == 1
    rebuilt = client.get_collection(COLLECTION_NAME).get(include=["metadatas"])
    sources = {m["source"] for m in rebuilt["metadatas"]}
    assert "https://arxiv.org/abs/2301.00001" not in sources
    assert (papers / "p.pdf").as_uri() in sources and "local" in sources
    assert [c.name for c in client.list_collections()] == [COLLECTION_NAME]
