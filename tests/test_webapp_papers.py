"""
Tests for the webapp's papers manager routes (jarvis/webapp/app.py):

- GET /papers lists every indexed paper (de-duplicated by source) with the
  fields the frontend table renders
- q= filters case-insensitively across title/authors/doi/source
- POST /papers/meta updates only the fields the caller passes and 404s on
  an unknown source
- POST /papers/remove deletes the paper's ChromaDB chunks via execute_remove
  and 404s on an unknown source
- CRITICAL regression: /papers/remove never touches the filesystem — the
  same "database entry only" invariant enforced everywhere else in the
  codebase (see test_security.py), pinned here with spies on every
  plausible deletion API (Path.unlink/rmdir, os.remove, os.unlink,
  shutil.rmtree)

These use the real ChromaDB `store` fixture (see conftest.py), with
jarvis.kb.store.get_store monkeypatched to return it — the same pattern
test_webapp_chat.py's wired_session fixture uses to stand in for the
process-wide singleton — so the routes are exercised against real indexed
data rather than a stub.
"""

import os
from pathlib import Path

from starlette.testclient import TestClient

import jarvis.webapp.app as appmod
from jarvis.kb.store import add_paper, list_papers


def _paper(n: int, **overrides) -> dict:
    """Minimal paper dict accepted by add_paper(), same shape as test_store.py's helper."""
    paper = {
        "link": f"https://arxiv.org/abs/2401.{n:05d}",
        "title": f"Test Paper {n}",
        "authors": f"Author {n}",
    }
    paper.update(overrides)
    return paper


# ── GET /papers ──────────────────────────────────────────────────────────


def test_get_papers_lists_every_indexed_paper(store, monkeypatch):
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    add_paper(_paper(1), dense_summary="Attention is a mechanism.", score=8, track="nlp", store=store)
    add_paper(_paper(2), dense_summary="Diffusion models generate images.", store=store)

    client = TestClient(appmod.app, base_url="http://127.0.0.1")
    response = client.get("/papers")
    assert response.status_code == 200

    rows = response.json()
    titles = {row["title"] for row in rows}
    assert {"Test Paper 1", "Test Paper 2"} <= titles

    row1 = next(r for r in rows if r["title"] == "Test Paper 1")
    for field in (
        "title", "authors", "doi", "source", "storage_mode",
        "visibility", "score", "track", "date_added", "chunk_count",
    ):
        assert field in row1
    assert row1["authors"] == "Author 1"
    assert row1["score"] == 8
    assert row1["track"] == "nlp"
    assert row1["visibility"] == "public"
    assert row1["source"] == "https://arxiv.org/abs/2401.00001"


def test_get_papers_q_filters_title_case_insensitively(store, monkeypatch):
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    add_paper(_paper(1, title="Attention Is All You Need"), dense_summary="x", store=store)
    add_paper(_paper(2, title="Diffusion Models Beat GANs"), dense_summary="y", store=store)

    client = TestClient(appmod.app, base_url="http://127.0.0.1")
    response = client.get("/papers", params={"q": "ATTENTION"})
    assert response.status_code == 200
    assert [row["title"] for row in response.json()] == ["Attention Is All You Need"]


def test_get_papers_q_matches_authors_doi_or_source(store, monkeypatch):
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    add_paper(_paper(1, authors="Ada Lovelace"), dense_summary="x", store=store)
    add_paper(_paper(2, doi="10.1234/uniquedoi"), dense_summary="y", store=store)
    add_paper(_paper(3), dense_summary="z", store=store)

    client = TestClient(appmod.app, base_url="http://127.0.0.1")

    def titles_for(q):
        return {row["title"] for row in client.get("/papers", params={"q": q}).json()}

    assert titles_for("lovelace") == {"Test Paper 1"}
    assert titles_for("uniquedoi") == {"Test Paper 2"}
    assert titles_for("2401.00003") == {"Test Paper 3"}  # source URL substring


# ── POST /papers/meta ────────────────────────────────────────────────────


def test_post_papers_meta_updates_only_given_fields(store, monkeypatch):
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    paper = _paper(1)
    add_paper(paper, dense_summary="x", store=store)

    client = TestClient(appmod.app, base_url="http://127.0.0.1")
    response = client.post("/papers/meta", json={"source": paper["link"], "title": "New Title"})
    assert response.status_code == 200
    assert response.json()["chunks_updated"] >= 1

    updated = next(p for p in list_papers(store=store) if p["source"] == paper["link"])
    assert updated["title"] == "New Title"
    assert updated["authors"] == "Author 1"  # untouched — only title was sent


def test_post_papers_meta_unknown_source_404s(store, monkeypatch):
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    client = TestClient(appmod.app, base_url="http://127.0.0.1")
    response = client.post("/papers/meta", json={"source": "https://arxiv.org/abs/nope", "title": "x"})
    assert response.status_code == 404


# ── POST /papers/remove ──────────────────────────────────────────────────


def test_post_papers_remove_calls_execute_remove(store, monkeypatch):
    calls = []
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)

    def fake_execute_remove(action, s):
        calls.append(action)
        return f"Removed {action['title']}"

    monkeypatch.setattr(appmod, "execute_remove", fake_execute_remove)

    paper = _paper(1)
    ids = add_paper(paper, dense_summary="x", store=store)

    client = TestClient(appmod.app, base_url="http://127.0.0.1")
    response = client.post("/papers/remove", json={"source": paper["link"]})
    assert response.status_code == 200
    assert response.json()["result"] == "Removed Test Paper 1"

    assert len(calls) == 1
    assert sorted(calls[0]["ids"]) == sorted(ids)
    assert calls[0]["title"] == "Test Paper 1"


def test_post_papers_remove_unknown_source_404s(store, monkeypatch):
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    client = TestClient(appmod.app, base_url="http://127.0.0.1")
    response = client.post("/papers/remove", json={"source": "https://arxiv.org/abs/nope"})
    assert response.status_code == 404


def test_post_papers_remove_deletes_only_the_matching_paper(store, monkeypatch):
    """
    End-to-end with the real execute_remove (no stub): the removed paper's
    chunks are gone from the store afterwards, and an unrelated paper's
    chunks are untouched.
    """
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    paper_a = _paper(1)
    paper_b = _paper(2)
    add_paper(paper_a, dense_summary="x", store=store)
    add_paper(paper_b, dense_summary="y", store=store)

    client = TestClient(appmod.app, base_url="http://127.0.0.1")
    response = client.post("/papers/remove", json={"source": paper_a["link"]})
    assert response.status_code == 200

    remaining_sources = {p["source"] for p in list_papers(store=store)}
    assert paper_a["link"] not in remaining_sources
    assert paper_b["link"] in remaining_sources


# ── Critical: no filesystem deletion ─────────────────────────────────────


def test_papers_remove_never_touches_the_filesystem(store, monkeypatch):
    """
    Regression pin for the CLAUDE.md invariant that jarvis has no
    file-deletion code path anywhere: /papers/remove must only ever call
    ChromaDB's own delete. Spies on every plausible deletion API
    (pathlib.Path.unlink/rmdir, os.remove, os.unlink, shutil.rmtree —
    os.remove and os.unlink are distinct objects, so both need patching)
    must never fire during the request, however this route is wired.
    """
    import shutil

    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)

    deletion_calls = []
    monkeypatch.setattr(Path, "unlink", lambda self, *a, **k: deletion_calls.append(self))
    monkeypatch.setattr(Path, "rmdir", lambda self, *a, **k: deletion_calls.append(self))
    monkeypatch.setattr(os, "remove", lambda path, *a, **k: deletion_calls.append(path))
    monkeypatch.setattr(os, "unlink", lambda path, *a, **k: deletion_calls.append(path))
    monkeypatch.setattr(shutil, "rmtree", lambda path, *a, **k: deletion_calls.append(path))

    paper = _paper(1)
    add_paper(paper, dense_summary="x", store=store)

    client = TestClient(appmod.app, base_url="http://127.0.0.1")
    response = client.post("/papers/remove", json={"source": paper["link"]})

    assert response.status_code == 200
    assert deletion_calls == []


def test_papers_routes_are_scoped_to_papers(store, monkeypatch):
    """A note's source 404s on both /papers/meta and /papers/remove — the
    routes only ever act on doc_type="paper" entries."""
    from jarvis.kb.store import add_texts

    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    add_texts(
        content="A private thought that must not be reachable via the papers routes.",
        doc_type="note",
        visibility="public",
        source="file:///tmp/some-note.md",
        store=store,
    )

    client = TestClient(appmod.app, base_url="http://127.0.0.1")
    meta_response = client.post(
        "/papers/meta", json={"source": "file:///tmp/some-note.md", "title": "hijacked"}
    )
    remove_response = client.post("/papers/remove", json={"source": "file:///tmp/some-note.md"})

    assert meta_response.status_code == 404
    assert remove_response.status_code == 404
