# Changelog

Prototype stage — no deployments. Changes documented for development reference only.

---

## [current] — knowledge source toggle (DB only / AI fallback)

### Added
- `USE_OWN_KNOWLEDGE_TOOL` pseudo-tool in `vault_chat/chat.py` — included in the tools list only when AI fallback is enabled. The LLM calls it before drawing on its training knowledge, giving the UI a structured signal to display to the user. Dispatch returns a simple acknowledgement string.
- `build_system_prompt(kb_only=True)` — replaces the old zero-argument function. Appends one of two addendums to the base prompt: a hard restriction ("answer only from KB tools") when `kb_only=True`, or a preference ("search KB first, call `use_own_knowledge` before using training knowledge") when `kb_only=False`.
- `run_session(vault, kb_only=True)` — `kb_only` parameter added; selects the correct system prompt and tools list.
- `vault-chat --no-db-only` flag — enables AI fallback mode from the terminal. Default behaviour (DB only) is unchanged with no flag.
- `POST /config` endpoint in `webapp/app.py` — accepts `{"kb_only": bool}`; updates session flag and rebuilds the system prompt for subsequent requests.
- `kb_only: True` added to the webapp session state dict.
- **DB only toggle** in the web UI input bar — pill toggle, on by default. Fires `POST /config` on change. Label reads "DB only".
- Amber status badge in the web UI — rendered when the `use_own_knowledge` tool event arrives via SSE, or when replaying history. Shown instead of a collapsible tool-call row.

### Changed
- `webapp/app.py` `chat()` route: snapshots `kb_only` at request time and passes either `TOOLS` or `TOOLS + [USE_OWN_KNOWLEDGE_TOOL]` to `agentic_turn()`.

---

## [previous] — privacy hard stop (PrivacyError); web UI rebuild (FastAPI + SSE); refresh_vault bug fix

### Added
- `PrivacyError(PaperDigestError)` in `digest/errors.py` — raised (not returned as a string) when a cloud provider attempts to access private content
- `PrivacyError` hard stop in both `OllamaProvider.agentic_turn()` and `AnthropicProvider.agentic_turn()`: catches `PrivacyError` from `dispatch_fn`, removes the orphaned assistant message so conversation history stays valid, and returns the error string immediately — no further LLM calls are made
- `webapp/app.py` — FastAPI web UI served at `http://127.0.0.1:8080`; launch with `uv run webapp`
- `webapp/index.html` — single self-contained HTML page; inline CSS and vanilla JS; no external dependencies
- Tool calls rendered live in an open `<details>` box while the agent is working; collapses when the reply arrives; history shown as collapsed `<details>` on re-render
- Conversation history survives browser refresh (restored from server-side in-memory display list via `/history`)
- `fastapi>=0.100.0` and `uvicorn>=0.20.0` added to project dependencies
- `webapp` entry point added to `pyproject.toml`
- `webapp --provider <ollama|anthropic>` CLI flag — overrides config and `CHAT_PROVIDER` env var for that server session

### Removed (prior to this rebuild)
- Streamlit web UI — Streamlit collects telemetry that cannot be reliably disabled, which conflicts with this project's privacy requirements
- `on_tool_call` callback parameter from `_dispatch_tool()` — was Streamlit-specific dead code; the FastAPI UI uses a `dispatch_fn` wrapper instead

### Fixed
- `refresh_vault` Phase 1 was including PDF notes (absolute `file_path` values ending in `.pdf`) in the `indexed` dict alongside vault `.md` notes (relative paths). The deletion sweep compared against `current` (relative `.md` paths only), so every PDF note's absolute path was "not found" and silently deleted on every `refresh_vault` call.
- `index-vault --force` was deleting all notes including PDF notes. Now it only clears vault `.md` chunks; PDF notes are preserved.

### Removed
- `kb refresh-vault` CLI subcommand — redundant with `kb index-vault` (which calls `refresh_vault()` internally). `kb index-vault` is incremental by default; use `--force` to clear and rebuild.
- `refresh_vault` tool from vault-chat agent — replaced by `index_vault` which covers both the incremental and force-rebuild cases.

### Changed
- `index_vault` tool description updated to reflect that it handles both incremental and forced rebuilds.

### Changed
- `_search_notes` and `_retrieve_papers` in `vault_chat/chat.py` now raise `PrivacyError` instead of returning a warning string when the query matches only private content. Mixed results (public + private) return the public results silently — the LLM is not told private content exists.
- `read_file` in `vault_chat/chat.py` now raises `PrivacyError` instead of returning a warning string when a cloud provider attempts to read a file inside a `private_vault_dirs` folder.
- `_privacy_warning` helper removed — no longer needed.

### Tests
- Added `test_refresh_vault_preserves_pdf_notes` to `tests/test_store.py` as a regression test for the Phase 1 deletion bug.

---

## [previous] — Streamlit web UI

### Added
- `webapp/app.py` — Streamlit chat UI served at `localhost:8501`; launch with `uv run streamlit run webapp/app.py`
- Tool calls rendered live in a collapsible `st.status()` box while the LLM is working; collapses to a summary when done; historical tool calls shown in `st.expander()` on re-render
- Sidebar shows active provider and vault path
- Vault auto-refreshed once per browser session on startup
- `streamlit>=1.40.0` added to project dependencies

### Changed
- `_dispatch_tool()` in `vault_chat/chat.py` accepts an optional `on_tool_call` callback — when provided, calls it instead of printing; terminal `vault-chat` behaviour is unchanged (no callback passed)

---

## [previous] — test suite

### Added
- `tests/` directory with pytest-based unit and integration test suite
- `tests/conftest.py` — shared fixtures: session-scoped HuggingFace embedding model; per-test isolated ChromaDB collection backed by a persistent local store at `tests/.chroma/`
- `tests/test_config.py` — `load_config()` resolution order (defaults → TOML → env vars), path expansion, API key loading
- `tests/test_errors.py` — `@with_retries` behaviour: success, retry on matching exception, raise after max attempts, no retry on unspecified exception
- `tests/test_arxiv_convert.py` — `parse_arxiv_url()` edge cases: abs URL, pdf URL, version suffix, non-arXiv URL
- `tests/test_store.py` — full KB operation coverage: `add_texts`, `add_paper` idempotency, visibility filter, `search_with_privacy_check` (cloud and local), `delete_by_metadata`, `list_papers` deduplication and chunk count, `update_file_path`, `refresh_vault` (add / update / delete)
- `tests/test_llm.py` — integration tests (marked `@pytest.mark.integration`): Anthropic client initialisation, `models.list()` API call ($0 tokens), Ollama server reachability
- `docs/TESTING.md` — test infrastructure overview, how to run, what is and isn't covered
- `[dependency-groups] dev = ["pytest>=8.0"]` in `pyproject.toml`; `[tool.pytest.ini_options]` with testpaths and markers
- `tests/.chroma/` added to `.gitignore`

### Changed
- `CLAUDE.md` updated: all code changes must pass `uv run pytest -m "not integration"` before being considered done; tests must not be skipped or deleted to force a pass

---

## [previous] — doc_type simplification, PDF notes, update_file_path, auth cleanup

### Added
- `storage_mode` metadata field (`"summary"` or `"full_text"`) stored on every indexed chunk
- `kb list` now shows chunk count and storage mode per entry — chunk count is ground truth for verifying full-text vs summary storage
- `kb clear` and `kb add` commands added to README
- `[build-system]` table added to `pyproject.toml` — `uv sync` now installs entry points without needing `uv pip install -e .`
- `kb add --doc-type paper|note` flag for local PDFs — user must specify whether a local PDF is a paper or a note
- Local PDF notes (`doc_type="note"`) always stored as `full_text`; `content_hash` (SHA-256) stored for change detection
- `refresh_vault` Phase 2: checks indexed local PDF notes — warns if file is missing, re-indexes if hash has changed
- `update_file_path(source, new_path)` in `store.py` — updates `file_path` metadata and `source` URI for all matching chunks without re-embedding
- `kb update-path <source> <new_path>` CLI subcommand
- `update_file_path` tool added to `vault-chat` so the agent can update paths conversationally
- `CLAUDE.md` created with commands, non-obvious implementation details, and code style guidance

### Fixed
- `--full-text` for local PDFs was always falling through to summary mode — now correctly converts and chunks the PDF
- `kb add --full-text` no longer instantiates the LLM provider when no summary is needed
- `paper_summary.md` prompt path in `llm.py` was wrong after subpackage restructure (`digest/prompts/` → `digest/kb/prompts/`)
- `kb auth` subcommand and all OAuth PKCE code removed — Anthropic banned third-party subscription OAuth in early 2026
- `oauth_client_id` removed from `~/.seshat/config.toml`

### Changed
- `doc_type` is now strictly `"paper"` or `"note"` — the `"pdf"` type has been removed; existing `"pdf"` chunks migrated to `"paper"`
- Anthropic API key can now be stored in `~/.seshat/config.toml` under `[auth] api_key` as an alternative to the `ANTHROPIC_API_KEY` env var
- `kb add-digest` default `--min-score` changed from `0` to `9`
- `add_paper()` in `store.py` accepts `storage_mode` parameter
- README: setup simplified to `uv sync` only; OAuth auth section replaced with config file option
- `docs/DESIGN.md` updated: removed OAuth config fields, updated `doc_type` schema, added `storage_mode`, `update_file_path`, and Phase 2 `refresh_vault`

---

## [previous] — project rename to seshat

### Renamed
- GitHub repository: `ghar1821/paper_digest` → `ghar1821/seshat`
- Project package name: `paper-digest` → `seshat` in `pyproject.toml`
- Config directory: `~/.paper_digest/` → `~/.seshat/` (config, auth, RAG store)
- launchd agent label: `com.putri.paper-digest` → `com.putri.seshat`
- launchd plist file: `com.putri.paper-digest.plist` → `com.putri.seshat.plist`

### Docs
- `LAUNCHD_SETUP.md` moved from project root to `docs/`
- `docs/RENAME.md` added — step-by-step rename procedure
- `docs/DESIGN.md` and `docs/CHANGELOG.md` added in prior phase, now tracked alongside

### Changed
- `config.toml` `rag_dir` default updated to `~/.seshat/rag`
- Vector DB cleared for fresh population following documented README steps

---

## [previous] — subpackage restructure and full-text mode

### Architecture
- Reorganised flat `digest/` package into three focused subpackages:
  - `digest/arxiv/` — arXiv fetching (`fetch.py`) and PDF conversion (`convert.py`)
  - `digest/pipeline/` — weekly digest automation (`run.py`, `score.py`, `format.py`, `prompts/`)
  - `digest/kb/` — knowledge base management (`store.py`, `cli.py`, `prompts/`)
- `digest/config.py`, `digest/errors.py`, `digest/llm.py` remain at package root as shared infrastructure

### Added
- `kb add --full-text` flag — stores full PDF text chunked via `RecursiveCharacterTextSplitter` instead of LLM-generated summary; uses marker-pdf for conversion
- `add_document` tool in vault-chat — adds papers by arXiv URL or local PDF path; supports both `summary` and `full_text` modes
- `index_vault` tool in vault-chat — triggers vault indexing or forced re-index conversationally
- Local PDF support in `kb add` — `--visibility` flag controls `public`/`private`
- `read_file` tool in vault-chat — reads a specific vault file by path

### Removed
- `download_must_reads()` from `format.py` — dead code; replaced by `add_papers_batch()` in the pipeline
- Stale `download_must_reads` import from `run.py`

### Changed
- `vault-chat` repositioned as a unified KB agent (query + management), not just a chat interface
- Every tool call in vault-chat now prints `→ tool_name(args)` to the terminal for transparency
- `add_paper` tool renamed to `add_document`; accepts local PDFs in addition to arXiv URLs

---

## [previous] — LangChain migration and privacy model

### Architecture
- Replaced direct ChromaDB usage with LangChain (`langchain-chroma`, `langchain-huggingface`, `langchain-text-splitters`)
- Unified two-collection schema (papers + vault_notes) into a single `knowledge_base` collection
- Flat document schema: `date_added`, `doc_type`, `visibility`, `source` + optional fields
- Privacy model: `visibility: "public" | "private"` — cloud providers search public only; warning when private docs match
- Vault privacy by folder (`private/` → private, all else → public)

### Added
- `search_with_privacy_check()` — provider-aware search
- `add_paper` tool in vault-chat — add papers by arXiv URL conversationally
- `list_papers`, `kb_stats`, `refresh_vault` tools in vault-chat
- `remove_document` — two-step (preview then confirm); shows what will be deleted; optionally deletes local file
- `kb remove --delete-file` flag
- `kb add --visibility` flag for local PDFs
- `docs/` folder with `DESIGN.md` and `CHANGELOG.md`

### Changed
- `kb remove` now shows a preview with title/type/source before asking for confirmation
- `kb clear` requires typing `yes` (not just `y`) and explicitly states no files will be deleted
- `search_vault` tool renamed to `search_notes`; `remove_paper` renamed to `remove_document`

---

## [previous] — Knowledge base and provider abstraction

### Architecture
- `digest/llm.py`: `ChatProvider` protocol, `OllamaProvider`, `AnthropicProvider`, `make_provider()`
- `digest/config.py`: central `Config` dataclass; `~/.seshat/config.toml` + env var overrides
- `digest/errors.py`: domain exceptions + `@with_retries` decorator
- All prompts moved to external `.md` files in `prompts/`

### Added
- `kb` CLI: `add`, `add-digest`, `list`, `stats`, `remove`, `clear`, `index-vault`, `refresh-vault`, `auth`
- `kb add-digest` — import papers from digest files without re-running LLM
- `vault-chat` Anthropic provider via `provider.agentic_turn()`
- `fetch_arxiv_paper()` — single-paper fetch; fixes `source` format bug
- Vault auto-refresh on `vault-chat` startup

### Changed
- `filter_and_score()` accepts `ChatProvider` instead of a model name string
- `vault-chat` single session loop replacing separate Ollama/Anthropic loops
- System prompt no longer injects vault file list (forces search-first behaviour)
- `retrieve_papers()` raises `RAGError` instead of silently returning `[]`

---

## [initial] — First working prototype

### Added
- arXiv fetch pipeline: `fetch_arxiv`, `deduplicate`
- LLM scoring: `filter_and_score` (local Ollama)
- Markdown digest formatter: `format_digest`
- PDF converter: `convert_pdf`, `download_arxiv_pdf`, `parse_arxiv_url`
- Digest pipeline entry point: `run.py`
- Local vector database: ChromaDB, two collections (papers + vault_notes)
- Obsidian vault chat: `vault_chat/chat.py` (Ollama, `read_file` tool)
- macOS launchd scheduling: `run_digest.sh`
