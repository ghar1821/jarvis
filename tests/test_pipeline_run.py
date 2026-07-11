"""
Tests for jarvis/digest/pipeline/run.py — the digest→knowledge-base indexing tiers.

The fetch/score halves of the pipeline need live LLM responses and stay
untested (see docs/TESTING.md); what's covered here is everything after
scoring: the score-tier routing (>=9 full text, [8,9) summary, <8 not indexed
per-paper), the full-text ingest with its dedup and fallbacks, and the digest
document indexing. The arXiv PDF download is mocked at the network boundary —
the fake writes a real (tiny) PDF via PyMuPDF so conversion, chunking, and
embedding all run for real against the isolated store fixture.
"""

from datetime import datetime
from pathlib import Path

import pymupdf
import pytest
import requests

import jarvis.digest.pipeline.run as run_module
from jarvis.core.config import Config
from jarvis.kb.store import add_paper
from jarvis.digest.pipeline.run import index_digest_file, index_scored_papers, ingest_full_text_paper


class _NoCallProvider:
    """The tiered indexing path must make zero LLM calls — this proves it."""

    def summarize(self, *args, **kwargs):
        raise AssertionError("summarize() must never be called by the digest indexing tiers")

    def complete(self, *args, **kwargs):
        raise AssertionError("complete() must never be called by the digest indexing tiers")

    def describe_image(self, *args, **kwargs):
        raise AssertionError("describe_image() must not run with figure captions off")


def _fake_arxiv_download(text: str):
    """A download_arxiv_pdf stand-in that writes a real tiny PDF locally."""

    def fake_download(arxiv_id: str, dest_dir: Path) -> Path:
        pdf_path = dest_dir / f"{arxiv_id.replace('/', '_')}.pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=12)
        doc.save(pdf_path)
        doc.close()
        return pdf_path

    return fake_download


@pytest.fixture
def pipeline_config(tmp_path, monkeypatch):
    """
    Pin both the pipeline's and the store's config to clean defaults
    (figure captions off, rag paths inside tmp) regardless of what the
    machine's real ~/.jarvis/config.toml says.
    """
    cfg = Config(rag_dir=tmp_path / "rag")
    monkeypatch.setattr("jarvis.digest.pipeline.run.get_config", lambda: cfg)
    monkeypatch.setattr("jarvis.kb.store.get_config", lambda: cfg)
    return cfg


def _arxiv_paper() -> dict:
    return {
        "link": "https://arxiv.org/abs/2406.04093",
        "title": "Scaling Sparse Autoencoders",
        "authors": "Ada Lovelace, Alan Turing",
        "doi": "",
    }


def _selected(index: int = 0, score: float = 9) -> dict:
    return {
        "index": index,
        "score": score,
        "track": "Track 1",
        "summary": "A dense summary from the scoring run.",
        "why": "Highly relevant to interpretability work.",
    }


# ── ingest_full_text_paper ─────────────────────────────────────────────────────

def test_arxiv_must_read_is_indexed_full_text(store, monkeypatch, pipeline_config):
    """
    A score>=9 arXiv paper gets its PDF downloaded, converted, and chunked as
    full text — score/track/storage_mode metadata set, embed header (title +
    authors) prepended, and no LLM call anywhere.
    """
    monkeypatch.setattr(
        run_module, "download_arxiv_pdf",
        _fake_arxiv_download("Sparse autoencoders decompose activations into features."),
    )
    paper = _arxiv_paper()

    outcome = ingest_full_text_paper(paper, _selected(), _NoCallProvider(), store)
    assert outcome == "added_full_text"

    stored = store._collection.get(
        where={"source": {"$eq": paper["link"]}}, include=["metadatas", "documents"]
    )
    assert stored["ids"]
    body_meta = stored["metadatas"][0]
    assert body_meta["storage_mode"] == "full_text"
    assert body_meta["score"] == 9
    assert body_meta["track"] == "Track 1"
    assert body_meta["authors"] == "Ada Lovelace, Alan Turing"
    assert all(
        doc.startswith("Scaling Sparse Autoencoders — Ada Lovelace, Alan Turing")
        for doc in stored["documents"]
    )


def test_biorxiv_doi_link_falls_back_to_summary(store, pipeline_config):
    """
    A bioRxiv paper links via doi.org, which parse_arxiv_url can't turn into
    a PDF download — it gets a summary entry built from the scoring run's own
    text, still with zero LLM calls.
    """
    paper = {
        "link": "https://doi.org/10.1101/2026.06.30.662001",
        "title": "Cytometry Panel Design With Transformers",
        "authors": "Grace Hopper",
    }

    outcome = ingest_full_text_paper(paper, _selected(), _NoCallProvider(), store)
    assert outcome == "added_summary"

    stored = store._collection.get(
        where={"source": {"$eq": paper["link"]}}, include=["metadatas", "documents"]
    )
    assert stored["metadatas"][0]["storage_mode"] == "summary"
    assert any("dense summary from the scoring run" in doc for doc in stored["documents"])


def test_download_failure_falls_back_to_summary(store, monkeypatch, capsys, pipeline_config):
    """
    A 404 (or any download failure) must not fail the digest job: the paper
    falls back to a summary entry and a visible warning is printed.
    """
    def failing_download(arxiv_id: str, dest_dir: Path) -> Path:
        raise requests.HTTPError("404 Client Error: Not Found")

    monkeypatch.setattr(run_module, "download_arxiv_pdf", failing_download)
    paper = _arxiv_paper()

    outcome = ingest_full_text_paper(paper, _selected(), _NoCallProvider(), store)
    assert outcome == "added_summary"
    assert "falling back to a summary entry" in capsys.readouterr().out

    stored = store._collection.get(where={"source": {"$eq": paper["link"]}}, include=["metadatas"])
    assert stored["metadatas"][0]["storage_mode"] == "summary"


def test_already_indexed_paper_is_skipped(store, monkeypatch, pipeline_config):
    """Dedup runs before any download: a known source (or title) is skipped."""
    def download_must_not_run(arxiv_id: str, dest_dir: Path) -> Path:
        raise AssertionError("no download should happen for an already indexed paper")

    monkeypatch.setattr(run_module, "download_arxiv_pdf", download_must_not_run)
    paper = _arxiv_paper()
    add_paper(paper, dense_summary="Already here.", store=store)

    outcome = ingest_full_text_paper(paper, _selected(), _NoCallProvider(), store)
    assert outcome == "skipped"


# ── index_scored_papers tiers ──────────────────────────────────────────────────

def test_tiers_route_full_text_summary_and_not_indexed(store, monkeypatch, pipeline_config):
    """
    >=9 → full text; 8 <= score < 9 → summary via add_papers_batch;
    < 8 → not indexed per-paper at all.
    """
    monkeypatch.setattr(
        run_module, "download_arxiv_pdf",
        _fake_arxiv_download("Full text of the must-read paper."),
    )
    papers = [
        _arxiv_paper(),
        {"link": "https://arxiv.org/abs/2406.11111", "title": "Worth Reading Paper",
         "authors": "B. Author"},
        {"link": "https://arxiv.org/abs/2406.22222", "title": "Skim Paper",
         "authors": "C. Author"},
    ]
    selected = [
        _selected(index=0, score=9),
        _selected(index=1, score=8.5),
        _selected(index=2, score=7),
    ]

    index_scored_papers(selected, papers, _NoCallProvider(), store)

    must_read = store._collection.get(
        where={"source": {"$eq": papers[0]["link"]}}, include=["metadatas"]
    )
    assert must_read["metadatas"][0]["storage_mode"] == "full_text"

    worth_reading = store._collection.get(
        where={"source": {"$eq": papers[1]["link"]}}, include=["metadatas"]
    )
    assert worth_reading["metadatas"][0]["storage_mode"] == "summary"

    skim = store._collection.get(where={"source": {"$eq": papers[2]["link"]}}, include=[])
    assert skim["ids"] == []


def test_tier_boundary_scores_route_correctly(store, monkeypatch, pipeline_config):
    """
    Exact tier-boundary scores: 8.0 must land in the [8, 9) summary tier
    (index_scored_papers uses `8 <= s < 9`), and 9.0 must land in the >= 9
    full-text tier — proving the boundary comparisons are inclusive/exclusive
    exactly where the tier table says, not off by one at either edge.
    """
    monkeypatch.setattr(
        run_module, "download_arxiv_pdf",
        _fake_arxiv_download("Full text of the exactly-9.0 paper."),
    )
    papers = [
        _arxiv_paper(),
        {"link": "https://arxiv.org/abs/2406.33333", "title": "Exactly Eight Paper",
         "authors": "D. Author"},
    ]
    selected = [
        _selected(index=0, score=9.0),
        _selected(index=1, score=8.0),
    ]

    index_scored_papers(selected, papers, _NoCallProvider(), store)

    exactly_nine = store._collection.get(
        where={"source": {"$eq": papers[0]["link"]}}, include=["metadatas"]
    )
    assert exactly_nine["metadatas"][0]["storage_mode"] == "full_text"

    exactly_eight = store._collection.get(
        where={"source": {"$eq": papers[1]["link"]}}, include=["metadatas"]
    )
    assert exactly_eight["metadatas"][0]["storage_mode"] == "summary"


# ── index_digest_file ──────────────────────────────────────────────────────────

def test_index_digest_file_stores_digest_doc_type(store, tmp_path, pipeline_config):
    """
    The digest .md is indexed as doc_type="digest" (never "note" — the vault
    refresh would sweep it) with a file:// source pointing at the file on
    disk and a dated title.
    """
    digest_text = (
        "# Paper Digest\n\n"
        "## Must-Read\n\n### A Great Paper\nWhy it matters.\n\n"
        "## Skim\n\n### A Below-Threshold Paper\nOnly mentioned here.\n"
    )
    output_path = tmp_path / "digest-2026-07-06_05-00.md"
    output_path.write_text(digest_text)
    today = datetime(2026, 7, 6, 5, 0)

    ids = index_digest_file(digest_text, output_path, today, store)
    assert ids

    stored = store._collection.get(ids=ids, include=["metadatas"])
    for meta in stored["metadatas"]:
        assert meta["doc_type"] == "digest"
        assert meta["visibility"] == "public"
        assert meta["source"] == output_path.resolve().as_uri()
        assert meta["source"].startswith("file://")
        assert meta["title"] == "Paper Digest — 2026-07-06"
        assert meta["file_path"] == str(output_path.resolve())
        assert meta["storage_mode"] == "full_text"
