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
and runs a fixed golden set of ~22 queries through the real `search()` pipeline,
asserting hit-rate@5 and MRR@5 stay above a floor. Its purpose is twofold: catch
regressions (a broken query prefix, chunker, or reranker drops the metrics well
below the floor), and provide a place to observe retrieval accuracy when tuning.
The corpus includes acronym/proper-noun queries (`LoRA`, `BERT`, `Dr. Tanaka`)
that are the sentinel for the deferred hybrid-BM25 work (see `docs/DESIGN.md`).

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
| `test_config.py` | `digest/config.py` | Defaults when no TOML; TOML overrides defaults; env vars override TOML; `~` in paths expanded; `[auth] api_key` loaded; `[chat] ollama_model` + `OLLAMA_MODEL` env; `[rag]` retrieval and figure-caption keys; `[digest]` bioRxiv keys; `[sync]` section defaults, overrides, and `PDF_WATCH_DIR` env override |
| `test_errors.py` | `digest/errors.py` | `@with_retries`: success on first try; retry on matching exception; raise after max attempts; no retry on non-matching exception; backoff grows exponentially with jitter (sleep monkeypatched) |
| `test_arxiv_convert.py` | `digest/arxiv/convert.py` | `parse_arxiv_url()`: `/abs/` URL; `/pdf/` URL; version suffix preserved; non-arXiv URL returns None |
| `test_arxiv_fetch.py` | `digest/arxiv/fetch.py` | `_to_paper` field mapping; fetch success; empty-feed-with-200 treated as transient (retried, then raises `FetchError`); recovery when the empty feed is transient; library errors wrapped as `FetchError`; single-paper fetch by ID; deduplication. The arxiv client's network calls are stubbed (real `arxiv.Result` objects, no HTTP); retries run with `time.sleep` monkeypatched |
| `test_biorxiv_fetch.py` | `digest/biorxiv/fetch.py` | Record→paper mapping (doi→link, `bioRxiv:{category}` source); cursor pagination across pages and `max_results` cap; empty first page retried then raises `FetchError`; keyword matching (title/abstract, case-insensitive, non-match excluded, first-keyword source tag); DOI dedup across two matched keywords. `requests.get` stubbed with fake responses shaped like the real API; `time.sleep` monkeypatched |
| `test_images.py` | `digest/kb/images.py`, `digest/kb/store.py` | `extract_figures` keeps large images with their 1-indexed page and drops sub-`min_pixels` decoys; `max_figures` cap. `add_figures` indexes `[FIGURE p.N]` caption chunks (`annotation_kind="figure"`) with a fake provider; private+anthropic skip writes nothing and never calls the model; per-figure failure is tolerated; the kill-switch disables it; delete-by-source sweeps figure chunks |
| `test_pdf_convert.py` | `digest/kb/convert.py` | Real pymupdf4llm conversion of PDFs generated in-test: text extracted; string returned with no intermediate files; scanned/image-only PDF raises `ConversionError` |
| `test_annotations.py` | `digest/kb/annotations.py` | Extraction from PDFs annotated in-test with PyMuPDF (the same annotation objects Preview/Foxit write — no binary fixtures): highlight text recovery; typed note on a highlight; underline treated as highlight; sticky note → comment; unannotated PDF → `[]`; Ink drawing ignored; multi-line highlight reading order |
| `test_store.py` | `digest/kb/store.py` | `add_texts` count; `add_paper` idempotency; title-based dedup (`_title_exists` normalisation, add skipped on same title from a different source, `allow_duplicate` forces the add); `add_papers_batch` returns `(added, skipped)`; visibility filter; privacy check (cloud and local); `delete_by_metadata`; `list_papers` deduplication and chunk count; `update_file_path` metadata and URI; `update_file_path` unknown source; `update_visibility` metadata-only; `add_annotations` indexing (and no-op on unannotated PDFs); delete-by-source sweeps body and annotation chunks together; `annotation_kind` search filter; `refresh_vault` add / update / delete / PDF notes preserved / visibility re-check when config reclassifies a dir; embedding-model guard (mismatch / match / empty); re-ranking preserves visibility filter and `rerank=False` skips the reranker; chunk_index / section breadcrumb metadata |
| `test_retrieval_quality.py` | `digest/kb/store.py` | Golden-set retrieval benchmark — hit-rate@5 and MRR@5 over a seeded corpus of paper summaries and markdown notes |
| `test_privacy_guard.py` | `vault_chat/chat.py` | The chat-layer privacy guards: `read_file` vault containment, private-dir hard stop for cloud only, symlink-into-private-dir regression, path-escape rejection; `_search_notes` excluded-results caveat, private-only hard stop, no caveat for the local provider; `_add_document` rejects private papers, allows private note PDFs |
| `test_daemon.py` | `digest/daemon.py` | Pure decision functions plus real ingestion with the store fixture: `_build_scheduler` builds cleanly with both jobs and a real (non-string) timezone — the regression for the `timezone="local"` crash-loop; `digest_is_overdue` (first start, within the week, missed slot, boundary); `wait_for_stable` (settled / growing / vanished files); `ingest_pdf` add → skip (unchanged hash) → update (changed bytes); `scan_watch_dir` queuing and artifact skipping; `_validate_sync_config`; status file round-trip, job recording, and corrupt-file handling |
| `test_sessions.py` | `vault_chat/sessions.py` | Save/load round-trip; pydantic message normalisation; empty sessions never written; malicious session-id rejection; pruning keeps newest unpinned and all pinned; sidebar ordering; `mark_private` flag + re-index; `check_resume` matrix and strict refusal of a retired `llamacpp`-provider session under `ollama`; chat-history search respects session privacy; delete removes file and chunks; rename round-trip, empty/whitespace rejected, 120-char cap, unknown id, and `update_chat_title` propagates to indexed chat chunks; compaction no-op below threshold, replaces old turns with a summary (fake provider), display untouched; token estimation |
| `test_skills.py` | `vault_chat/skills.py` | Name/description listing; missing dir = feature off; full-content read; traversal-name rejection; unknown name lists available skills |
| `test_settings.py` | `digest/config.py` + `vault_chat/chat.py` | Response style lands in the system prompt (and absent when empty); skills advertised in the prompt; `set_config_value` tomlkit round-trip preserving comments and other keys; creates missing file/section; `reset_config` reloads the singleton |
| `test_security.py` | `digest/kb/store.py`, `vault_chat/chat.py`, `webapp/app.py` | `delete_local_file` papers-only rule (deletes paper PDFs, never notes, refuses non-PDF papers); `_remove_document` human gate: unconfirmed previews only, confirmed without a channel refuses, human decline blocks, human approval executes, deferred (webapp) confirmation leaves everything intact; a keep-file (`delete_file=false`) removal leaves the PDF on disk and both the preview and dialog show the full path + KEPT wording; `truncate_middle` preserves head+tail so a `file:///` filename stays visible; webapp rejects foreign Host headers (TrustedHost); session-id traversal rejected |
| `test_llm.py` | `digest/llm.py` | Unit tests with the LLM clients mocked at the API boundary: `make_provider` spec dispatch; Ollama tool loop uses dict arguments directly (no JSON parsing) and normalises the pydantic message; both providers honour the `PrivacyError` contract (return the error text, restore message history, no further LLM call); Anthropic tool results bundled into one user message. Integration tests (marked): Anthropic client init, models-list auth check, Ollama reachability via `ollama.list()` |

## What is not tested

| Module | Reason |
|---|---|
| `digest/pipeline/` | Depends on LLM responses; correctness is validated by running the pipeline |
| `vault_chat/chat.py` (full agentic loop) | The end-to-end loop needs a live LLM; the guards it relies on are covered by `test_privacy_guard.py` / `test_security.py`, the tool loop mechanics by `test_llm.py`'s boundary-mocked tests, and all KB behaviour by `test_store.py` |
| `webapp/` (UI and SSE stream) | Requires a live browser, server, and LLM; the security-relevant endpoints (TrustedHost, session-id validation) are covered by `test_security.py`, and the routes are thin wrappers over already-covered `vault_chat` code |
| `digest/daemon.py` (scheduler loop, watchdog observer) | launchd/runtime plumbing around APScheduler and watchdog; the decision functions and job bodies they drive are covered by `test_daemon.py` |
