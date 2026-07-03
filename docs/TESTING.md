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
| `test_anthropic_client_initialises` | API key in `ANTHROPIC_API_KEY` env var or `~/.seshat/config.toml [auth]` |
| `test_anthropic_models_list_confirms_auth` | API key + internet access |
| `test_ollama_server_is_reachable` | Running Ollama server at `http://localhost:11434` |

Integration tests make no token-consuming LLM calls — they only validate
connectivity and credentials.

---

## What is tested

| File | Module | Behaviours covered |
|---|---|---|
| `test_config.py` | `digest/config.py` | Defaults when no TOML; TOML overrides defaults; env vars override TOML; `~` in paths expanded; `[auth] api_key` loaded |
| `test_errors.py` | `digest/errors.py` | `@with_retries`: success on first try; retry on matching exception; raise after max attempts; no retry on non-matching exception |
| `test_arxiv_convert.py` | `digest/arxiv/convert.py` | `parse_arxiv_url()`: `/abs/` URL; `/pdf/` URL; version suffix preserved; non-arXiv URL returns None |
| `test_store.py` | `digest/kb/store.py` | `add_texts` count; `add_paper` idempotency; visibility filter; privacy check (cloud and local); `delete_by_metadata`; `list_papers` deduplication and chunk count; `update_file_path` metadata and URI; `update_file_path` unknown source; `refresh_vault` add / update / delete / PDF notes preserved; embedding-model guard (mismatch / match / empty); re-ranking preserves visibility filter and `rerank=False` skips the reranker; chunk_index / section breadcrumb metadata |
| `test_retrieval_quality.py` | `digest/kb/store.py` | Golden-set retrieval benchmark — hit-rate@5 and MRR@5 over a seeded corpus of paper summaries and markdown notes |
| `test_llm.py` *(integration)* | `digest/llm.py` | Anthropic client initialisation; models list API call; Ollama server reachability |

## What is not tested

| Module | Reason |
|---|---|
| `digest/llm.py` (beyond connectivity) | Wraps external APIs; meaningful correctness tests would require mocking the full HTTP layer, which is not preferred |
| `digest/arxiv/fetch.py` | Same — live arXiv API |
| `digest/pipeline/` | Depends on LLM responses; correctness is validated by running the pipeline |
| `vault_chat/chat.py` | The agentic loop is integration-level; all KB behaviour it relies on is covered by `test_store.py` |
| `webapp/` | The web UI is integration-level (requires a live browser, server, and LLM); the FastAPI routes and SSE stream are thin wrappers over `vault_chat/chat.py` which is already covered |
