# Testing

## Running tests

```bash
# Install dev dependencies first (once)
uv sync --group dev

# Run all unit tests
uv run pytest

# Run integration tests (requires live services — see below)
uv run pytest -m integration

# Run unit tests only, explicitly skipping integration tests
uv run pytest -m "not integration"

# Run a single test file
uv run pytest tests/test_store.py

# Run a single test by name
uv run pytest tests/test_store.py::test_add_paper_is_idempotent
```

---

## Test infrastructure

### Dedicated ChromaDB store

KB tests use a real ChromaDB instance persisted at `tests/.chroma/` (gitignored,
never committed). Each test creates a collection named `test_<uuid8>` inside that
directory and deletes it at teardown. This means:

- Tests are fully isolated from each other (separate collections)
- The store directory itself persists between runs (no re-initialisation overhead)
- The embedding model is not reloaded for every test

### Real HuggingFace models

The actual embedding model named in the default config (`BAAI/bge-small-en-v1.5`)
is used in the KB tests rather than a mock or a deterministic stub, built through
the same `build_embeddings()` helper production uses so the query prefix and
normalisation match real retrieval. The cross-encoder reranker
(`cross-encoder/ms-marco-MiniLM-L6-v2`) is likewise loaded for real by any test
that calls `search()`. Both models download once to `~/.cache/huggingface/` on
the first run (embedding ~133 MB, reranker ~90 MB) and are reused from local cache
afterwards.

**Why not mock the models?** A fake embedding function that returns zeroes or
random vectors would hide real failure modes — ChromaDB filter behaviour, the
LangChain wrapper's batching logic, and search result ranking all depend on actual
vector values, and the reranker's effect can only be observed with a real model.
The one-off download cost is worth the fidelity. This preference (a modest one-off
setup cost over a mock that obscures real behaviour) is the general policy for this
project; apply the same reasoning to other test decisions.

### Real vs mocked, by boundary

The policy in practice:

- **Real dependencies:** embeddings, the reranker, ChromaDB, PyMuPDF, and
  pymupdf4llm all run for real. PDF fixtures (annotated pages, scanned-like
  pages) are generated in-test with PyMuPDF — no binary fixtures are committed,
  and the real extraction/conversion paths run end-to-end.
- **Mocked boundaries (sanctioned):** the arxiv client's network calls
  (live HTTP to export.arxiv.org — `_client.results` is stubbed, with real
  `arxiv.Result` objects), the bioRxiv API (`requests.get` stubbed with fake
  responses shaped like the real payload), and the LLM API clients (billed per
  token / need a running model server — mocked at the Ollama/Anthropic client
  boundary, including `describe_image` for figure captioning). These are
  genuine system boundaries where real calls are expensive or non-deterministic;
  the retry/tool-loop logic under test is ours, not the libraries'.

### Retrieval-quality benchmark

`test_retrieval_quality.py` measures *how good* retrieval is, not just whether it
runs. It seeds a small module-scoped corpus (paper summaries + markdown notes) once
and runs a fixed golden set of ~24 queries through the real `search()` pipeline
(hybrid BM25+RRF, the default), asserting hit-rate@5 and MRR@5 stay above a floor.
Its purpose is twofold: catch regressions (a broken query prefix, chunker, or
reranker drops the metrics well below the floor), and provide a place to observe
retrieval accuracy when tuning. The corpus includes acronym/proper-noun queries
(`LoRA`, `BERT`, `Dr. Tanaka`) and author-name queries (`"papers by Vaswani"`) that
only the embed_header prepended to every paper chunk can satisfy.

Thresholds are set as a regression floor with margin, not at the current ceiling —
on a small, topically distinct corpus the pipeline scores near-perfectly, so tighten
the thresholds only as far as stays stable across query rewording.

---

## Integration tests

Tests marked `@pytest.mark.integration` require live external services:

| Test | Requirement |
|---|---|
| `test_anthropic_client_initialises` | API key in `ANTHROPIC_API_KEY` env var or `~/.jarvis/config.toml [auth]` |
| `test_anthropic_models_list_confirms_auth` | API key + internet access |
| `test_ollama_is_reachable` | Running Ollama server (checked via `ollama.list()` at `http://localhost:11434`) |

Integration tests make no token-consuming LLM calls — they only validate
connectivity and credentials.

---

## What is tested

| File | Module | Behaviours covered |
|---|---|---|
| `test_config.py` | `jarvis/core/config.py` | Defaults when no TOML (incl. `figure_captions` false, `digest_hour` 5, `pdf_watch_minutes` 30); TOML overrides defaults; env vars override TOML; `~` in paths expanded; `[auth] api_key` loaded; `[chat] ollama_model` + `OLLAMA_MODEL` env; `[rag]` retrieval and figure-caption keys, including `hybrid` (defaults true, parses from TOML); `[digest]` bioRxiv keys; `[sync]` section defaults, overrides (incl. `pdf_watch_minutes`), and `PDF_WATCH_DIR` env override |
| `test_errors.py` | `jarvis/core/errors.py` | `@with_retries`: success on first try; retry on matching exception; raise after max attempts; no retry on non-matching exception; backoff grows exponentially with jitter (sleep monkeypatched) |
| `test_arxiv_convert.py` | `jarvis/digest/arxiv/convert.py` | `parse_arxiv_url()`: `/abs/` URL; `/pdf/` URL; version suffix preserved; non-arXiv URL returns None |
| `test_arxiv_fetch.py` | `jarvis/digest/arxiv/fetch.py` | `_to_paper` field mapping including `doi` (empty by default, surfaced when the arXiv result has one); fetch success; empty-feed-with-200 treated as transient (retried, then raises `FetchError`); recovery when the empty feed is transient; library errors wrapped as `FetchError`; single-paper fetch by ID; deduplication. The arxiv client's network calls are stubbed (real `arxiv.Result` objects, no HTTP); retries run with `time.sleep` monkeypatched |
| `test_biorxiv_fetch.py` | `jarvis/digest/biorxiv/fetch.py` | Record→paper mapping (doi→link, `bioRxiv:{category}` source); cursor pagination across pages and `max_results` cap; empty first page retried then raises `FetchError`; keyword matching (title/abstract, case-insensitive, non-match excluded, first-keyword source tag); DOI dedup across two matched keywords. `requests.get` stubbed with fake responses shaped like the real API; `time.sleep` monkeypatched |
| `test_images.py` | `jarvis/kb/images.py`, `jarvis/kb/store.py` | `extract_figures` keeps large images with their 1-indexed page and drops sub-`min_pixels` decoys; `max_figures` cap. `add_figures` (via the `enabled=True` per-document opt-in, since captions now default off) indexes `[FIGURE p.N]` caption chunks (`annotation_kind="figure"`) with a fake provider; private+anthropic skip writes nothing and never calls the model even with `enabled=True`; per-figure failure is tolerated; the config kill-switch disables it; delete-by-source sweeps figure chunks |
| `test_pdf_convert.py` | `jarvis/kb/convert.py` | Real pymupdf4llm conversion of PDFs generated in-test: text extracted; string returned with no intermediate files; scanned/image-only PDF raises `ConversionError` |
| `test_annotations.py` | `jarvis/kb/annotations.py` | Extraction from PDFs annotated in-test with PyMuPDF (the same annotation objects Preview/Foxit write — no binary fixtures): highlight text recovery; typed note on a highlight; underline treated as highlight; sticky note → comment; unannotated PDF → `[]`; Ink drawing ignored; multi-line highlight reading order |
| `test_store.py` | `jarvis/kb/store.py` | `add_texts` count; `add_paper` idempotency and authors line in content; title-based dedup (`_title_exists` normalisation, add skipped on same title from a different source, `allow_duplicate` forces the add); `add_papers_batch` returns `(added, skipped)`; visibility filter; privacy check (cloud and local); `delete_by_metadata`; `list_papers` deduplication and chunk count; `update_file_path` metadata and URI; `update_file_path` unknown source; `get_document_chunks` orders body chunks (by `chunk_index`) before annotation/figure chunks, and returns `[]` for an unknown source; `update_visibility` metadata-only; `add_annotations` indexing (and no-op on unannotated PDFs); delete-by-source sweeps body and annotation chunks together; `annotation_kind` search filter; `refresh_vault` add / update / delete / visibility re-check when config reclassifies a dir; embedding-model guard (mismatch / match / empty); re-ranking preserves visibility filter and `rerank=False` skips the reranker; chunk_index / section breadcrumb metadata; `KBCorruptionError` raised on an "Error finding id" failure (names `kb reindex`) and on the stale-collection-handle failure after a reindex swap ("Collection [...] does not exist" — names restarting the process, on both the search and add_texts paths), other failures stay a plain `RAGError`; `embed_header` prepended to every chunk; hybrid on/off routing (`_hybrid_search` called only when `cfg.hybrid`, byte-identical `similarity_search` path when disabled); `_reciprocal_rank_fusion` unit test; `update_paper_metadata` sets only the fields the caller passes; `doc_type` list filter (`["paper", "digest"]` returns both types, a plain string narrows to one); `add_figures` enabled matrix (`enabled=True` captions despite config off, `enabled=None` follows config); `find_pdf_notes` ignores vault `.md` notes and groups a legacy PDF note's chunks by source with title/visibility/chunk_count; `reclassify_notes_as_papers` flips only `doc_type` (content_hash/storage_mode/file_path/visibility untouched) and is a no-op on an empty source list |
| `test_metadata.py` | `jarvis/kb/metadata.py` | `infer_pdf_metadata`: DOI found by regex skips the LLM DOI guess, DOI guessed by the LLM when the regex misses, LLM failure degrades gracefully (title/authors absent, doi `""`); `resolve_pdf_metadata`: explicit overrides skip inference entirely, inference always runs for a local PDF even under Anthropic (papers are always public, so there is no private-note guard to apply here) |
| `test_kb_cli.py` | `jarvis/kb/cli.py` | `kb doctor`: healthy store reports success at each stage; a corrupted index (`KBCorruptionError`) exits non-zero with the diagnosis; a store that fails to open exits non-zero; an empty store skips the search probe entirely; `_check_legacy_pdf_notes` migration: a public legacy PDF note is reclassified to `doc_type="paper"` on a `y` answer, left as `"note"` on `n`, a private one is only listed (resolution options, `kb remove` mentioned) with the reclassify prompt never even shown, and an empty store prints nothing |
| `test_retrieval_quality.py` | `jarvis/kb/store.py` | Golden-set retrieval benchmark — hit-rate@5 and MRR@5 over a seeded corpus of paper summaries and markdown notes, run with hybrid retrieval (the default); includes author-name queries answered only by the embed_header |
| `test_privacy_guard.py` | `jarvis/chat/chat.py` | The chat-layer privacy guards: `read_file` vault containment, private-dir hard stop for cloud only, symlink-into-private-dir regression, path-escape rejection; `_search_notes` excluded-results caveat, private-only hard stop, no caveat for the local provider; `_get_document` — a public document is fine under Anthropic, a private source raises `PrivacyError` under Anthropic with no content leaked in the message, and the local provider reads it and reports `saw_private=True` |
| `test_daemon.py` | `jarvis/sync/daemon.py` | Pure decision functions plus real ingestion with the store fixture: `_build_scheduler` builds cleanly with digest/digest_catchup/vault_refresh jobs, adds pdf_scan only when a watch dir is set, and has a real (non-string) timezone — the regression for the `timezone="local"` crash-loop; `digest_is_overdue` (first start, within the week, missed slot, boundary); `run_digest_catchup_job` matrix (fresh success → not fired, 8-day-stale → fired, missing baseline → not fired, `run_digest_job` monkeypatched) plus a same-path two-call regression proving the job re-reads the status file each call rather than caching (stale fires, then a freshly written success on the same path does not); the digest double-fire lock (`run_digest_job` returns early without touching config when the lock is held) and a companion test proving the lock is released on the success path too (pipeline `main` stubbed, called once per `run_digest_job` call across two calls); `wait_for_stable` (settled / growing / vanished files); `ingest_pdf` add → skip (unchanged hash) → update (changed bytes), with a stub metadata-inference provider; `ingest_pdf` populates an inferred title/authors; `ingest_pdf` logs the stored title/authors/doi and source filename after a successful add; `_log_next_run_times` logs one "next run at" line per scheduled job; `_log_job_outcome` logs a job's next run time after both a success and an error event; `scan_watch_dir` listing and artifact skipping; `run_pdf_scan_job` (ingest + skip with status recorded to a tmp status file, per-file failure recorded without raising, unstable file left for the next cycle, no-op without a watch dir); `_validate_sync_config` (incl. `pdf_watch_minutes` ≥ 1); status file round-trip, job recording, and corrupt-file handling |
| `test_pipeline_run.py` | `jarvis/digest/pipeline/run.py` | The digest→KB indexing tiers, with the store fixture and `download_arxiv_pdf` mocked to write a real tiny PDF (conversion/chunking/embedding run for real): score ≥ 9 arXiv paper indexed full text with score/track/`storage_mode="full_text"` and the title/authors embed header; bioRxiv doi.org link falls back to a summary entry; a download failure (404) warns and falls back; an already-indexed paper is skipped before any download; `index_scored_papers` routes 9 → full text, 8.5 → summary via `add_papers_batch`, 7 → not indexed, plus an exact-boundary case (score 9.0 → full text, score 8.0 → summary); `index_digest_file` stores `doc_type="digest"` with a `file://` source, dated title, and `storage_mode="full_text"`. A provider stub raises on any LLM call, proving the whole tier path is LLM-free |
| `test_reingest_replace.py` | `jarvis/kb/cli.py`, `jarvis/chat/chat.py` | Regression coverage for the replace-on-duplicate reingest flow: a same-source duplicate re-add whose download/conversion then fails leaves the old entry's chunks (including a preseeded marker standing in for irreplaceable annotations) completely untouched, for both `jarvis.chat.chat._add_document` (arXiv full-text, `download_arxiv_pdf` stubbed to raise) and `jarvis.kb.cli.cmd_add` (local-PDF full-text, `pdf_to_markdown` stubbed to raise); a successful reingest on both paths deletes the old chunks exactly once and leaves only the new content, with a single entry for the source; a same-title-but-different-source duplicate with `allow_duplicate=true` never deletes the other entry |
| `test_sessions.py` | `jarvis/chat/sessions.py` | Save/load round-trip; pydantic message normalisation; empty sessions never written; malicious session-id rejection; pruning keeps newest unpinned and all pinned; sidebar ordering; `mark_private` flag + re-index; `check_resume` matrix and strict refusal of a retired `llamacpp`-provider session under `ollama`; chat-history search respects session privacy; delete removes file and chunks; rename round-trip, empty/whitespace rejected, 120-char cap, unknown id, and `update_chat_title` propagates to indexed chat chunks; compaction no-op below threshold, replaces old turns with a summary (fake provider), display untouched; token estimation |
| `test_skills.py` | `jarvis/chat/skills.py` | Folder-based skills (`<name>/SKILL.md` + supporting files): frontmatter `description:` parsing and fallback to the first non-empty body line; missing dir = feature off; stray flat `*.md` file warns (`capsys`) and is skipped; folder missing `SKILL.md` warns and is skipped; `read_skill` default output is SKILL.md content plus a sorted "Supporting files:" listing (omitted when there are none); `file=` reads one supporting file; traversal rejection on both `name` and `file`, including a symlink under the skill folder pointing outside it; unknown skill/file name errors listing what exists; a supporting file over the 64 KB cap is rejected with a clear message |
| `test_settings.py` | `jarvis/core/config.py` + `jarvis/chat/chat.py` | Response style lands in the system prompt (and absent when empty); skills advertised in the prompt; `set_config_value` tomlkit round-trip preserving comments and other keys; creates missing file/section; `reset_config` reloads the singleton |
| `test_security.py` | `jarvis/kb/store.py`, `jarvis/chat/chat.py`, `jarvis/webapp/app.py` | File deletion has been removed wholesale: `jarvis.kb.store.delete_local_file` no longer exists, and no `.unlink(` call survives anywhere in `jarvis/kb/store.py`, `jarvis/kb/cli.py`, or `jarvis/chat/chat.py`; `_remove_document` is a one-shot flow — no channel refuses, human decline blocks, human approval executes and never touches disk, a deferred (webapp) channel leaves a pending action that executes correctly later, and the "files on disk are never touched by jarvis" invariant line (with the full local path) appears in every preview and confirmation description; `truncate_middle` preserves head+tail so a `file:///` filename stays visible; webapp rejects foreign Host headers (TrustedHost); session-id traversal rejected; `pending_actions` dict (now `{token: {session_id, action}}`): an unknown/cleared token 409s without disturbing other entries, the matching token pops just its own entry and executes, two stacked tokens across different sessions are each independently confirmable with no session check, cancelling one pops only its own token, `/sessions/new` leaves every other session's dialogs pending (they still confirm normally), a resume clears only the resumed session's own tokens (a different session's survive), and starting a new `/chat` turn on one session clears only that session's tokens |
| `test_chat_logging.py` | `jarvis/chat/chat.py` | A tool wrapper that raises logs the exception (with traceback) to the `vault-chat` logger before returning its short error string, across more than one tool wrapper; `KBCorruptionError` is relayed verbatim (prefixed `[KNOWLEDGE BASE ERROR`) by `_retrieve_papers`, `_search_notes`, and `_search_chat_history`, with logging still firing first. An `isolated_log` fixture detaches the module's real `FileHandler` for the test so the run never appends to the user's actual `~/.jarvis/logs/chat.log` |
| `test_chat_tools.py` | `jarvis/chat/chat.py` | Chunk-first retrieval: `_retrieve_papers`/`_search_notes` return text past the old 300-char cutoff (a >300-char chunk asserted present verbatim, no `...` elision) and a `Section:` breadcrumb line when the chunk carries one; `_get_document` pagination over 22 indexed chunks (page 1 of 2 shows chunks 1-15 with the "Call get_document(...) for more" hint, page 2 of 2 shows chunks 16-22 with no further-page hint), an unknown source returns a `[No document found...]` string, and a `storage_mode="summary"` document gets the full-text-not-in-KB honesty note; `_dispatch_tool("get_document", ...)` wraps output in the `RETRIEVED DATA` markers and flags a fresh session private on a local-provider private hit, mirroring `read_file`/`retrieve_papers`/`search_notes` |
| `test_webapp_chat.py` | `jarvis/webapp/app.py` | The `/chat` turn lifecycle via `TestClient` with a fake provider (`agentic_turn` fully controlled, no live LLM): the early `save_session(session)` call lands before the LLM call with only the user's turn in `display` and no `store=`, the final save is indexed (`store=`) and includes the assistant turn; the busy guard (a session id in the `running` registry) makes a second `/chat` addressed at that same session 409, `DELETE` on it 409, and `GET /sessions` reports its id in the `busy` list; an uncaught exception in `agentic_turn` still logs an `ERROR` with a traceback, still yields a `reply` SSE event ("Internal error"), still saves the error turn, and the stream still terminates (`running` empties); an `LLMError` yields a `⚠️` reply, logs, and still saves; resuming a still-mid-turn session installs the exact live registry object the background thread is mutating (not a stale disk copy) and reports `busy: true`, and `GET /history` serves the finished reply once the turn is released — no reinstall step needed anywhere; two different sessions can be mid-turn at once (one blocked, one completing fully in between) with each session's own `display` holding only its own exchange and no cross-contamination; an unknown `session_id` 404s; a message addressed to a non-active session lands on that session and never touches whatever `_session["session"]` happens to be at that instant. A request FastAPI rejects at validation time (stale-tab `/chat` without `session_id`) returns the standard 422 detail list AND logs a `request validation failed` line to the vault-chat logger — 422s are no longer invisible in chat.log. Reuses the `isolated_log` fixture pattern from `test_chat_logging.py` so no test touches the real `~/.jarvis/logs/chat.log` |
| `test_llm.py` | `jarvis/core/llm.py` | Unit tests with the LLM clients mocked at the API boundary: `make_provider` spec dispatch; Ollama tool loop uses dict arguments directly (no JSON parsing) and normalises the pydantic message; both providers honour the `PrivacyError` contract (return the error text, restore message history, no further LLM call); Anthropic tool results bundled into one user message. Integration tests (marked): Anthropic client init, models-list auth check, Ollama reachability via `ollama.list()` |
| `test_webapp_papers.py` | `jarvis/webapp/app.py` | The papers manager routes via `TestClient` against a real ChromaDB store (`jarvis.kb.store.get_store` monkeypatched to return the `store` fixture, same pattern as `test_webapp_chat.py`'s `wired_session`): `GET /papers` lists every indexed paper (de-duplicated, all the fields the frontend table needs) and `q=` filters case-insensitively across title, authors, doi, and source; `POST /papers/meta` updates only the given fields (others untouched) and 404s on an unknown source; `POST /papers/remove` calls `execute_remove` with the expected `{ids, title}` action, actually removes only the matching paper's chunks end-to-end, and 404s on an unknown source; a dedicated regression test spies on every plausible deletion API (`Path.unlink`/`rmdir`, `os.remove`, `os.unlink`, `shutil.rmtree`) during a `/papers/remove` request and asserts none is ever called; a note's source 404s on both `/papers/meta` and `/papers/remove` (routes scoped to `doc_type="paper"`) |

## What is not tested

| Module | Reason |
|---|---|
| `jarvis/digest/pipeline/` (fetch + scoring halves) | Depend on live LLM responses; correctness is validated by running the pipeline. The post-scoring indexing tiers ARE tested — see `test_pipeline_run.py` |
| `jarvis/chat/chat.py` (full agentic loop) | The end-to-end loop needs a live LLM; the guards it relies on are covered by `test_privacy_guard.py` / `test_security.py`, the tool loop mechanics by `test_llm.py`'s boundary-mocked tests, and all KB behaviour by `test_store.py` |
| `jarvis/webapp/` (browser rendering) | Requires a live browser; `test_webapp_chat.py` and `test_security.py` cover the `/chat` turn lifecycle and the other routes against `TestClient` with a fake provider, so only the actual pixels/DOM are unverified here |
| `jarvis/webapp/static/app.js` copy button (`buildAssistantBubble`) | Pure frontend DOM/clipboard interaction; no JS test harness exists in this repo. Verified by manual click + paste. |
| `jarvis/webapp/static/app.js` selection-copy markdown conversion (`htmlFragmentToMarkdown`) | Pure frontend `copy`-event handling; no JS test harness exists. Verified by manual selection across bold/list/code + paste into a plain editor. |
| `jarvis/webapp/static/app.js` per-session input drafts (`drafts`, `switchDraft`) | Pure frontend state with no JS test harness. Verified manually: type a partial message, switch sessions (via sidebar, "New chat", and delete), confirm the draft doesn't leak into the new session and reappears when switching back; also confirm that a failed send restores its text into the textarea if still viewing that session, or into its draft if the user has since switched away. |
| `jarvis/webapp/static/app.js` busy-resume polling (`pollUntilTurnLands`) | Pure frontend polling/DOM update with no JS test harness; the `busy` list it polls for is covered server-side by `test_webapp_chat.py`. Verified manually: send a question, switch away before the reply lands, switch back and confirm a "Working..." placeholder appears and resolves into the finished reply. |
| `jarvis/webapp/static/app.js` per-session composer state (`inFlight`, `serverBusy`, `updateComposerState`) | Pure frontend state with no JS test harness; the server-side `busy` list it reads is covered by `test_webapp_chat.py`. Verified manually: start a slow question in session A, switch to session B, confirm B's send button stays enabled and a message sent there completes independently while A is still working; confirm A's sidebar row shows a pulsing busy dot throughout. |
| `jarvis/webapp/static/app.js` papers manager modal (open/close, search debounce, inline edit, two-step remove) | Pure frontend DOM/fetch wiring with no JS test harness; the routes it calls (`GET /papers`, `POST /papers/meta`, `POST /papers/remove`) are covered by `test_webapp_papers.py`. Verified manually: menu → "Papers…" lists everything with no truncation; typing in the search box narrows the table after a short pause and matches title/authors/doi/source; Edit → change title/authors/doi → Save persists (confirmed via `kb list` or reopening the modal) and Cancel discards; Remove shows the two-step "Database entry only…" confirmation with the paper's real path, and only Confirm removes it from the list and from `kb list`, while the PDF/file on disk is left untouched. |
| `jarvis/sync/daemon.py` (scheduler loop) | Process-runtime plumbing around APScheduler; the decision functions and job bodies it drives are covered by `test_daemon.py` |
