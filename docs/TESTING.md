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
| `test_store.py` | `jarvis/kb/store.py` | `add_texts` count; `add_paper` idempotency and authors line in content; title-based dedup (`_title_exists` normalisation, add skipped on same title from a different source, `allow_duplicate` forces the add); `add_papers_batch` returns `(added, skipped)`; visibility filter; privacy check (cloud and local); `delete_by_metadata`; `list_papers` deduplication and chunk count; `update_file_path` metadata and URI; `update_file_path` unknown source; `update_visibility` metadata-only; `add_annotations` indexing (and no-op on unannotated PDFs); delete-by-source sweeps body and annotation chunks together; `annotation_kind` search filter; `refresh_vault` add / update / delete / PDF notes preserved / visibility re-check when config reclassifies a dir; embedding-model guard (mismatch / match / empty); re-ranking preserves visibility filter and `rerank=False` skips the reranker; chunk_index / section breadcrumb metadata; `KBCorruptionError` raised on an "Error finding id" failure (names `kb reindex`), other failures stay a plain `RAGError`; `embed_header` prepended to every chunk; hybrid on/off routing (`_hybrid_search` called only when `cfg.hybrid`, byte-identical `similarity_search` path when disabled); `_reciprocal_rank_fusion` unit test; `update_paper_metadata` clears `meta_inferred`; `count_unverified_papers` de-duplication; `doc_type` list filter (`["paper", "digest"]` returns both types, a plain string narrows to one); `add_figures` enabled matrix (`enabled=True` captions despite config off, `enabled=None` follows config) |
| `test_metadata.py` | `jarvis/kb/metadata.py` | `infer_pdf_metadata`: DOI found by regex skips the LLM DOI guess, DOI guessed by the LLM when the regex misses, LLM failure degrades gracefully (title/authors absent, doi `""`); `resolve_pdf_metadata`: explicit overrides skip inference entirely, a private note under Anthropic skips inference (provider stubbed to raise if ever called), a public paper under Anthropic still runs inference (papers are always public) |
| `test_kb_cli.py` | `jarvis/kb/cli.py` | `kb doctor`: healthy store reports success at each stage; a corrupted index (`KBCorruptionError`) exits non-zero with the diagnosis; a store that fails to open exits non-zero; an empty store skips the search probe entirely |
| `test_retrieval_quality.py` | `jarvis/kb/store.py` | Golden-set retrieval benchmark — hit-rate@5 and MRR@5 over a seeded corpus of paper summaries and markdown notes, run with hybrid retrieval (the default); includes author-name queries answered only by the embed_header |
| `test_privacy_guard.py` | `jarvis/chat/chat.py` | The chat-layer privacy guards: `read_file` vault containment, private-dir hard stop for cloud only, symlink-into-private-dir regression, path-escape rejection; `_search_notes` excluded-results caveat, private-only hard stop, no caveat for the local provider; `_add_document` rejects private papers before touching the store (regression test for a segfault caused by validating after `get_store()`), allows private note PDFs |
| `test_daemon.py` | `jarvis/sync/daemon.py` | Pure decision functions plus real ingestion with the store fixture: `_build_scheduler` builds cleanly with digest/digest_catchup/vault_refresh jobs, adds pdf_scan only when a watch dir is set, and has a real (non-string) timezone — the regression for the `timezone="local"` crash-loop; `digest_is_overdue` (first start, within the week, missed slot, boundary); `run_digest_catchup_job` matrix (fresh success → not fired, 8-day-stale → fired, missing baseline → not fired, `run_digest_job` monkeypatched) plus a same-path two-call regression proving the job re-reads the status file each call rather than caching (stale fires, then a freshly written success on the same path does not); the digest double-fire lock (`run_digest_job` returns early without touching config when the lock is held) and a companion test proving the lock is released on the success path too (pipeline `main` stubbed, called once per `run_digest_job` call across two calls); `wait_for_stable` (settled / growing / vanished files); `ingest_pdf` add → skip (unchanged hash) → update (changed bytes), with a stub metadata-inference provider; `ingest_pdf` populates an inferred title/authors and sets `meta_inferred`; `scan_watch_dir` listing and artifact skipping; `run_pdf_scan_job` (ingest + skip with status recorded to a tmp status file, per-file failure recorded without raising, unstable file left for the next cycle, no-op without a watch dir); `_validate_sync_config` (incl. `pdf_watch_minutes` ≥ 1); status file round-trip, job recording, and corrupt-file handling |
| `test_pipeline_run.py` | `jarvis/digest/pipeline/run.py` | The digest→KB indexing tiers, with the store fixture and `download_arxiv_pdf` mocked to write a real tiny PDF (conversion/chunking/embedding run for real): score ≥ 9 arXiv paper indexed full text with score/track/`storage_mode="full_text"` and the title/authors embed header; bioRxiv doi.org link falls back to a summary entry; a download failure (404) warns and falls back; an already-indexed paper is skipped before any download; `index_scored_papers` routes 9 → full text, 8.5 → summary via `add_papers_batch`, 7 → not indexed, plus an exact-boundary case (score 9.0 → full text, score 8.0 → summary); `index_digest_file` stores `doc_type="digest"` with a `file://` source, dated title, and `storage_mode="full_text"`. A provider stub raises on any LLM call, proving the whole tier path is LLM-free |
| `test_reingest_replace.py` | `jarvis/kb/cli.py`, `jarvis/chat/chat.py` | Regression coverage for the replace-on-duplicate reingest flow: a same-source duplicate re-add whose download/conversion then fails leaves the old entry's chunks (including a preseeded marker standing in for irreplaceable annotations) completely untouched, for both `jarvis.chat.chat._add_document` (arXiv full-text, `download_arxiv_pdf` stubbed to raise) and `jarvis.kb.cli.cmd_add` (local-PDF full-text, `pdf_to_markdown` stubbed to raise); a successful reingest on both paths deletes the old chunks exactly once and leaves only the new content, with a single entry for the source; a same-title-but-different-source duplicate with `allow_duplicate=true` never deletes the other entry |
| `test_sessions.py` | `jarvis/chat/sessions.py` | Save/load round-trip; pydantic message normalisation; empty sessions never written; malicious session-id rejection; pruning keeps newest unpinned and all pinned; sidebar ordering; `mark_private` flag + re-index; `check_resume` matrix and strict refusal of a retired `llamacpp`-provider session under `ollama`; chat-history search respects session privacy; delete removes file and chunks; rename round-trip, empty/whitespace rejected, 120-char cap, unknown id, and `update_chat_title` propagates to indexed chat chunks; compaction no-op below threshold, replaces old turns with a summary (fake provider), display untouched; token estimation |
| `test_skills.py` | `jarvis/chat/skills.py` | Name/description listing; missing dir = feature off; full-content read; traversal-name rejection; unknown name lists available skills |
| `test_settings.py` | `jarvis/core/config.py` + `jarvis/chat/chat.py` | Response style lands in the system prompt (and absent when empty); skills advertised in the prompt; `set_config_value` tomlkit round-trip preserving comments and other keys; creates missing file/section; `reset_config` reloads the singleton |
| `test_security.py` | `jarvis/kb/store.py`, `jarvis/chat/chat.py`, `jarvis/webapp/app.py` | File deletion has been removed wholesale: `jarvis.kb.store.delete_local_file` no longer exists, and no `.unlink(` call survives anywhere in `jarvis/kb/store.py`, `jarvis/kb/cli.py`, or `jarvis/chat/chat.py`; `_remove_document` is a one-shot flow — no channel refuses, human decline blocks, human approval executes and never touches disk, a deferred (webapp) channel leaves a pending action that executes correctly later, and the "files on disk are never touched by jarvis" invariant line (with the full local path) appears in every preview and confirmation description; `truncate_middle` preserves head+tail so a `file:///` filename stays visible; webapp rejects foreign Host headers (TrustedHost); session-id traversal rejected; `/confirm-action` 409s on a token that doesn't match the pending action and executes when it matches |
| `test_chat_logging.py` | `jarvis/chat/chat.py` | A tool wrapper that raises logs the exception (with traceback) to the `vault-chat` logger before returning its short error string, across more than one tool wrapper; `KBCorruptionError` is relayed verbatim (prefixed `[KNOWLEDGE BASE ERROR`) by `_retrieve_papers`, `_search_notes`, and `_search_chat_history`, with logging still firing first. An `isolated_log` fixture detaches the module's real `FileHandler` for the test so the run never appends to the user's actual `~/.jarvis/logs/chat.log` |
| `test_llm.py` | `jarvis/core/llm.py` | Unit tests with the LLM clients mocked at the API boundary: `make_provider` spec dispatch; Ollama tool loop uses dict arguments directly (no JSON parsing) and normalises the pydantic message; both providers honour the `PrivacyError` contract (return the error text, restore message history, no further LLM call); Anthropic tool results bundled into one user message. Integration tests (marked): Anthropic client init, models-list auth check, Ollama reachability via `ollama.list()` |

## What is not tested

| Module | Reason |
|---|---|
| `jarvis/digest/pipeline/` (fetch + scoring halves) | Depend on live LLM responses; correctness is validated by running the pipeline. The post-scoring indexing tiers ARE tested — see `test_pipeline_run.py` |
| `jarvis/chat/chat.py` (full agentic loop) | The end-to-end loop needs a live LLM; the guards it relies on are covered by `test_privacy_guard.py` / `test_security.py`, the tool loop mechanics by `test_llm.py`'s boundary-mocked tests, and all KB behaviour by `test_store.py` |
| `jarvis/webapp/` (UI and SSE stream) | Requires a live browser, server, and LLM; the security-relevant endpoints (TrustedHost, session-id validation) are covered by `test_security.py`, and the routes are thin wrappers over already-covered `jarvis.chat` code |
| `jarvis/webapp/static/app.js` copy button (`buildAssistantBubble`) | Pure frontend DOM/clipboard interaction; no JS test harness exists in this repo. Verified by manual click + paste. |
| `jarvis/webapp/static/app.js` selection-copy markdown conversion (`htmlFragmentToMarkdown`) | Pure frontend `copy`-event handling; no JS test harness exists. Verified by manual selection across bold/list/code + paste into a plain editor. |
| `jarvis/sync/daemon.py` (scheduler loop) | Process-runtime plumbing around APScheduler; the decision functions and job bodies it drives are covered by `test_daemon.py` |
