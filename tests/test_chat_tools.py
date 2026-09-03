"""
Tests for the chat agent's tools and terminal loop (jarvis/chat/chat.py).

The bulk is chunk-first retrieval — the agent should be able to answer from
search hits and the get_document tool without falling back to reading raw
files.

Covers:
- _retrieve_papers / _search_notes now return the full chunk text (previously
  truncated to 300 chars), so a long passage stays fully visible to the model.
- _get_document: pagination (15 chunks/page), the header format, the
  summary-mode honesty note, and unknown-source handling.
- _dispatch_tool wraps get_document's output in the RETRIEVED DATA markers
  and flags the session private when the local provider returns private
  content, exactly like the other retrieval tools.
- _add_document stages its whole document in memory and commits it in one
  write, so a stopped ingest leaves the knowledge base untouched rather than
  half-indexed.
- run_session's Ctrl-C handling: an interrupt mid-turn drops the turn from the
  session and exits, instead of dumping a traceback out of main().

Privacy hard-stops for get_document are covered separately in
test_privacy_guard.py.
"""

from pathlib import Path

import pymupdf
import pytest

from jarvis.chat.chat import (
    _add_document, _dispatch_tool, _get_document, _retrieve_papers, _search_notes,
)
from jarvis.chat.sessions import new_session
from jarvis.core.cancel import CancelToken
from jarvis.core.errors import TurnCancelled
from jarvis.kb.store import add_paper, add_texts, count


# ── Full-text hits (no more 300-char truncation) ────────────────────────────────

def test_retrieve_papers_returns_text_beyond_300_chars(store, monkeypatch):
    """
    A paper chunk longer than 300 characters must appear in full in
    _retrieve_papers' output — the old behaviour truncated with "...".

    Input:  a paper summary >300 chars, indexed via add_paper
    Expected output: the full summary text is present, with no "..." elision
    """
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    long_summary = (
        "This paper introduces a graph neural network architecture for "
        "predicting protein-protein interaction networks from sequence data "
        "alone. The model combines message passing over a learned residue "
        "graph with an attention mechanism that highlights binding-site "
        "candidates. Benchmarked against three public interaction datasets, "
        "it improves F1 by twelve points over the prior state of the art "
        "while requiring an order of magnitude less training data."
    )
    assert len(long_summary) > 300
    paper = {"link": "https://arxiv.org/abs/9999.00001", "title": "GNN for PPI Prediction"}
    add_paper(paper, dense_summary=long_summary, store=store)

    result, _ = _retrieve_papers({"query": "protein interaction graph neural network"}, "ollama")
    assert long_summary in result
    assert "..." not in result


def test_search_notes_returns_text_beyond_300_chars(store, monkeypatch):
    """
    Same contract for _search_notes: full chunk text visible, no truncation.

    Input:  a note chunk >300 chars
    Expected output: full text present, no "..." elision
    """
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    long_note = (
        "Meeting notes on the wombat burrow census project: we walked the "
        "northern transect and counted fourteen active burrows, six of which "
        "showed fresh digging within the last week. Soil moisture readings "
        "were taken at each site and will be cross-referenced against the "
        "rainfall records from the regional station once they are digitised. "
        "Next visit should extend the transect another two kilometres east."
    )
    assert len(long_note) > 300
    add_texts(content=long_note, doc_type="note", visibility="public",
              source="local", extra_metadata={"file_path": "wombats.md", "title": "Wombat census"},
              store=store)

    result, _ = _search_notes({"query": "wombat burrow census transect"}, "ollama")
    assert long_note in result
    assert "..." not in result


def test_search_notes_includes_section_breadcrumb_when_present(store, monkeypatch):
    """
    A hit under a markdown heading carries a "Section:" line naming the
    heading breadcrumb, giving the model context beyond raw chunk text.

    Input:  a note with a "## Results" heading
    Expected output: "Section: Results" appears in the rendered hit
    """
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    add_texts(content="## Results\nThe population estimate came to roughly two hundred individuals.",
              doc_type="note", visibility="public", source="local",
              extra_metadata={"file_path": "survey.md", "title": "Survey"}, store=store)

    result, _ = _search_notes({"query": "population estimate two hundred individuals"}, "ollama")
    assert "Section: Results" in result


# ── _get_document pagination ────────────────────────────────────────────────────

def _index_many_chunks(store, source: str, n: int, title: str = "Long Paper") -> None:
    """Index n distinct, individually-searchable chunks under one source."""
    for i in range(n):
        add_texts(
            content=f"Chunk number {i} discusses topic area {i} of the long paper in detail.",
            doc_type="paper", visibility="public", source=source,
            extra_metadata={"title": title},
            store=store,
        )


def test_get_document_paginates_15_per_page(store, monkeypatch):
    """
    With more than 15 chunks stored, page 1 returns exactly the first 15 and
    names the total page count; page 2 returns the remainder.

    Input:  22 chunks under one source
    Expected output: page 1 header says "page 1 of 2" and contains chunk 0 but
        not chunk 15; page 2 header says "page 2 of 2" and contains chunk 15
    """
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    source = "file:///long-paper.pdf"
    _index_many_chunks(store, source, 22)

    page1, _ = _get_document({"source": source, "page": 1}, "ollama")
    assert "page 1 of 2" in page1
    assert "chunks 1–15 of 22" in page1
    assert "Chunk number 0 " in page1
    assert "Chunk number 15 " not in page1
    assert "Call get_document(source, page=2) for more." in page1

    page2, _ = _get_document({"source": source, "page": 2}, "ollama")
    assert "page 2 of 2" in page2
    assert "chunks 16–22 of 22" in page2
    assert "Chunk number 15 " in page2
    assert "Call get_document" not in page2  # last page: no "for more" hint


def test_get_document_unknown_source_returns_not_found(store, monkeypatch):
    """
    Input:  a source that was never indexed
    Expected output: a "[No document found ...]" string, not an exception
    """
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    result, saw_private = _get_document({"source": "file:///nope.pdf"}, "ollama")
    assert "No document found" in result
    assert saw_private is False


def test_get_document_summary_mode_appends_honesty_note(store, monkeypatch):
    """
    A document stored with storage_mode="summary" gets an appended note that
    the full text isn't in the KB — the model should not claim to have read
    the whole paper from a summary.

    Input:  a chunk with storage_mode="summary"
    Expected output: the honesty note is present in the result
    """
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    source = "https://arxiv.org/abs/1111.00001"
    add_texts(content="A dense one-paragraph summary of the paper's contribution.",
              doc_type="paper", visibility="public", source=source,
              extra_metadata={"title": "Summarised Paper", "storage_mode": "summary"},
              store=store)

    result, _ = _get_document({"source": source}, "ollama")
    assert "not in the knowledge base" in result
    assert "mode='full_text'" in result


# ── _dispatch_tool wiring ────────────────────────────────────────────────────────

def test_dispatch_get_document_wraps_output_and_flags_private_session(store, monkeypatch, tmp_path):
    """
    _dispatch_tool routes "get_document" the same way as the other retrieval
    tools: output wrapped in BEGIN/END RETRIEVED DATA markers, and a private
    hit under the local provider flips the session's private flag.

    Input:  a private document, ollama provider, a fresh (non-private) session
    Expected output: wrapped text; session.private becomes True
    """
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    source = "file:///private-doc.pdf"
    add_texts(content="Confidential lab notebook entry about the pilot experiment.",
              doc_type="note", visibility="private", source=source,
              extra_metadata={"title": "Lab notebook"}, store=store)

    session = new_session("ollama")
    assert session.private is False

    result = _dispatch_tool(
        "get_document", {"source": source}, tmp_path, "ollama", provider_obj=None, session=session,
    )
    assert result.startswith("=== BEGIN RETRIEVED DATA")
    assert result.rstrip().endswith("=== END RETRIEVED DATA ===")
    assert "Confidential lab notebook entry" in result
    assert session.private is True


# ── Staged ingest: a stopped add leaves nothing behind ─────────────────────────


class _StopOnSummarizeProvider:
    """
    Stands in for a provider whose summarize() the user stops part-way — the
    realistic case, since summarising is where an ingest spends its time.
    """

    def __init__(self, cancel):
        self._cancel = cancel

    def complete(self, messages, max_tokens=300, context_length=None, cancel=None):
        return "{}"  # metadata inference degrades to no fields

    def summarize(self, title, source, max_tokens=2048, cancel=None):
        self._cancel.stop()
        (cancel or self._cancel).check()
        raise AssertionError("summarize should have been cancelled")


class _CannedSummaryProvider:
    def complete(self, messages, max_tokens=300, context_length=None, cancel=None):
        return "{}"

    def summarize(self, title, source, max_tokens=2048, cancel=None):
        return "A canned summary of the paper."


def _one_page_pdf(path: Path, text: str) -> Path:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), text, fontsize=12)
    doc.save(path)
    doc.close()
    return path


def test_add_document_stopped_mid_ingest_writes_nothing(store, tmp_path, monkeypatch):
    """
    An ingest stopped before its commit must leave the knowledge base exactly
    as it was — this is the "nothing partial in the database" guarantee. The
    stop propagates as TurnCancelled rather than being reported back to the
    model as a tool error it would try to work around.
    """
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    pdf = _one_page_pdf(tmp_path / "stopped.pdf", "Body text of a paper being ingested.")
    before = count(store)
    cancel = CancelToken()

    with pytest.raises(TurnCancelled):
        _add_document(
            {"source": str(pdf), "mode": "summary"},
            _StopOnSummarizeProvider(cancel),
            cancel=cancel,
        )

    assert count(store) == before, "a stopped ingest must not write any chunks"
    assert store._collection.get(
        where={"source": {"$eq": pdf.resolve().as_uri()}}, include=[]
    )["ids"] == []


def test_add_document_commits_body_and_annotations_together(store, tmp_path, monkeypatch):
    """
    The happy path still indexes everything, and does it in the single commit
    the staging exists for — body chunks and annotation chunks share the
    source, so one query sees the whole document.
    """
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    pdf = _one_page_pdf(tmp_path / "good.pdf", "Body text of a paper being ingested.")

    result = _add_document(
        {"source": str(pdf), "mode": "summary", "title": "Ingested Paper"},
        _CannedSummaryProvider(),
        cancel=CancelToken(),
    )
    assert result.startswith("Added paper ")

    indexed = store._collection.get(
        where={"source": {"$eq": pdf.resolve().as_uri()}}, include=["documents"]
    )
    assert indexed["ids"], "the paper should be indexed"
    assert any("canned summary" in text.lower() for text in indexed["documents"])


# ── Terminal loop: Ctrl-C mid-turn ─────────────────────────────────────────────


def _drive_run_session(monkeypatch, session, agentic_turn):
    """
    Run one turn of the terminal loop with everything external stubbed: the
    provider's agentic_turn is supplied by the caller, input() answers once,
    and saves are recorded rather than written.
    """
    import jarvis.chat.chat as chat_module

    class _Provider:
        def __init__(self):
            self.agentic_turn = agentic_turn

        def complete(self, messages, max_tokens=2048, context_length=None, cancel=None):
            return "unused"

    saves = []
    answers = iter(["a question"])

    def one_question_then_eof(prompt=""):
        # The second prompt raises EOFError — the Ctrl-D path — so a completed
        # turn ends the loop instead of blocking on stdin.
        try:
            return next(answers)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr(chat_module, "make_provider", lambda spec: _Provider())
    monkeypatch.setattr("builtins.input", one_question_then_eof)
    monkeypatch.setattr("jarvis.chat.sessions.save_session", lambda s, store=None: saves.append(store))
    monkeypatch.setattr("jarvis.chat.sessions.maybe_compact", lambda *a, **k: False)
    monkeypatch.setattr("jarvis.chat.sessions.needs_compaction", lambda *a, **k: False)
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: "the-store")
    chat_module.run_session(Path("/tmp"), kb_only=True, session=session)
    return saves


def test_cli_ctrl_c_mid_turn_drops_the_turn_and_exits(monkeypatch, capsys):
    """
    Ctrl-C while the agent is working means quit — but the abandoned turn must
    not be left in the saved session, and the exit must be a clean 130 rather
    than a traceback escaping main(). Earlier completed turns survive.
    """
    session = new_session("ollama")
    session.turn_starts.append(0)
    session.messages.append({"role": "user", "content": "earlier question"})
    session.messages.append({"role": "assistant", "content": "earlier answer"})
    session.display.append({"role": "user", "content": "earlier question"})
    session.display.append({"role": "assistant", "content": "earlier answer"})

    def interrupted_turn(messages, tools, dispatch_fn, system, cancel=None):
        raise KeyboardInterrupt

    with pytest.raises(SystemExit) as exit_info:
        _drive_run_session(monkeypatch, session, interrupted_turn)
    assert exit_info.value.code == 130

    # The stopped turn left no trace; the earlier exchange is intact.
    assert [t["content"] for t in session.display] == ["earlier question", "earlier answer"]
    assert [m["content"] for m in session.messages] == ["earlier question", "earlier answer"]
    assert session.turn_starts == [0]
    assert "Stopped" in capsys.readouterr().out


def test_cli_completed_turn_commits_the_message_copy(monkeypatch, capsys):
    """
    The happy path still works with the working-copy commit: whatever the
    provider appended to the copy lands in the session once a reply arrives.
    """
    session = new_session("ollama")

    def working_turn(messages, tools, dispatch_fn, system, cancel=None):
        messages.append({"role": "assistant", "content": "the answer"})
        return "the answer"

    saves = _drive_run_session(monkeypatch, session, working_turn)

    assert [t["content"] for t in session.display] == ["a question", "the answer"]
    assert [m["content"] for m in session.messages] == ["a question", "the answer"]
    # One save per completed turn, and it indexes. (The extra store-free save
    # before the LLM call is a webapp-only thing — there it protects the
    # question from a session switch mid-turn, which the CLI cannot do.)
    assert saves == ["the-store"]


def test_add_document_arxiv_summary_mode_indexes_the_paper(store, monkeypatch):
    """
    The arXiv summary path — the one add_document branch no other test drove,
    which is how a missing import survived in it. Fetch metadata, summarise,
    stage, commit.
    """
    monkeypatch.setattr("jarvis.kb.store.get_store", lambda: store)
    paper = {
        "link": "https://arxiv.org/abs/2401.00042",
        "title": "A Paper About Soil",
        "abstract": "An abstract about soil formation.",
        "authors": "Ada Lovelace",
        "doi": "",
    }
    monkeypatch.setattr("jarvis.digest.arxiv.fetch.fetch_arxiv_paper", lambda arxiv_id: paper)

    result = _add_document(
        {"source": paper["link"], "mode": "summary", "score": 7, "track": "soil"},
        _CannedSummaryProvider(),
        cancel=CancelToken(),
    )
    assert result.startswith("Added ")

    indexed = store._collection.get(
        where={"source": {"$eq": paper["link"]}}, include=["documents", "metadatas"]
    )
    assert indexed["ids"]
    assert any("canned summary" in text.lower() for text in indexed["documents"])
    assert indexed["metadatas"][0]["title"] == paper["title"]
    assert indexed["metadatas"][0]["track"] == "soil"
