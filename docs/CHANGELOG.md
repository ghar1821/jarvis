# Changelog

Prototype stage — no deployments. Changes documented for development reference only.

---

## [current] — webapp bug fixes: modal visibility, session rename/pin, tool-call logging

Three bugs found while testing the webapp after the launchd-removal change,
plus the observability gap that made the second one hard to diagnose.

### Fixed
- **Response-style modal always visible, never closed**: `.hidden { display:
  none; }` and `.modal { display: flex; ... }` (`webapp/static/style.css`)
  have equal CSS specificity, so whichever rule appears later in the file
  wins regardless of which classes are actually applied — `.modal` (later in
  the file) always beat `.hidden`. Added `.modal.hidden { display: none; }`
  so the combined-class rule wins on its own merits, independent of source
  order.
- **Session rename and pin silently broken in the webapp**: `sessions_pin`
  and `sessions_rename` (`webapp/app.py`) declared their request bodies as
  quoted forward references (`req: "PinRequest"`, `req: "RenameRequest"`)
  while those Pydantic classes were defined later in the file. FastAPI
  couldn't resolve the annotation at route-registration time, so it treated
  `req` as a required **query parameter** instead of a JSON body — every
  call with a real body 422'd with `"loc": ["query", "req"]`. Moved all
  `*Request` model classes above the routes that use them and dropped the
  quotes; the underlying `rename_session()` / `set_pinned()` functions were
  never broken, only their HTTP wiring was.
- **Tool-call failures were unrecoverable after the fact**: every tool
  wrapper in `vault_chat/chat.py` caught `Exception` broadly and returned a
  short string (e.g. `"[kb_stats error: ...]"`) for the LLM to relay — but
  the LLM paraphrases rather than quotes, and nothing was ever logged, so a
  real failure (e.g. a transient ChromaDB lock during a concurrent
  `jarvis-sync` ingest) surfaced to the user as something as vague as
  "there is an internal error with the database" with no way to find out
  what actually happened. Added a `log.exception(...)` call at each of the
  ten `except Exception` sites, writing to a new `~/.jarvis/logs/chat.log`
  (file only, not echoed to the terminal, so an interactive session isn't
  interrupted by a raw traceback) with the full exception and stack trace.
  Shared by `vault-chat` and the webapp, since both dispatch through the
  same `_dispatch_tool`.
- **Code blocks ignored the 80-character cap**: `.assistant .bubble pre`
  used `overflow-x: auto` — correct for a "don't reflow my code" viewer, but
  the original request was for the 80-char cap to apply with no exceptions.
  A long unbroken line (e.g. reproduced pseudocode from a paper) scrolled
  sideways instead of wrapping, reading as "just continues on one line".
  Switched to `white-space: pre-wrap; overflow-wrap: break-word;` so code
  wraps like everything else, breaking mid-token if a single run of
  characters has no natural break point.
- **Message input couldn't wrap or hold a newline at all**: `#input`
  (`webapp/index.html`) was a native `<input type="text">`, which is a
  single-line control by construction — no CSS or JS can make it wrap or
  accept a literal `\n`, regardless of the 80-char work above (which only
  ever touched the rendered message *bubbles*, not the compose box). This
  is what the original TODO item was actually about. Switched to a
  `<textarea>` that grows with content up to a max height (then scrolls),
  reset back to one line after each send (`resizeInput()` in
  `webapp/static/app.js`). The existing Enter-sends / Shift+Enter-newline
  keydown handler needed no changes — it was already textarea-shaped, it
  just had no textarea to act on.

---

## [previous] — remove launchd support from jarvis-sync

`jarvis-sync` no longer assumes it is run under launchd. The daemon has no
launchd-specific code (KeepAlive-style restart-on-crash was always launchd's
job, not the daemon's), but the docs and log setup did assume it — that
assumption is now gone.

### Changed
- **Logging is self-contained**: `digest/daemon.py` `main()` now sets up
  `logging.FileHandler(~/.jarvis/logs/sync.log)` directly (creating the
  `logs/` directory if needed), alongside a `StreamHandler` to stderr. Log
  output no longer depends on launchd's `StandardOutPath`/`StandardErrorPath`
  redirecting stderr for you — running `uv run jarvis-sync` in a plain
  terminal now produces the same log file on its own.
- **Module docstring and `kb sync-status`'s "never run" message** no longer
  point at a LaunchAgent — they describe running `uv run jarvis-sync`
  directly and note that restart-on-crash is left to whatever keeps the
  process alive (a terminal multiplexer, a process manager, or nothing).
- **`docs/LAUNCHD_SETUP.md` removed.** README's daemon section replaces it
  with plain "run it in a terminal" instructions; DESIGN.md's file tree,
  command table, and daemon architecture section drop the launchd
  references accordingly.

### Removed
- The user's own `com.putri.jarvis.sync` LaunchAgent (unloaded and its plist
  deleted, outside the repo) — no longer documented or supported.

---

## [previous] — daemon tz fix, Ollama revert, bioRxiv, figure captioning, dark UI

Fixes the launchd crash-loop, reverts the local provider from llama.cpp back to
Ollama, adds bioRxiv as a digest source with knowledge-base title deduplication,
captions PDF figures at ingest with a vision model, and reworks the web UI
(dark theme, 80-char line cap, response-style modal, session rename, clearer
deletion dialog).

### Fixed
- **launchd crash-loop**: `digest/daemon.py` constructed the scheduler as
  `BlockingScheduler(timezone="local")`. The literal string `"local"` was
  passed to ZoneInfo, which has no such zone, so the daemon raised
  `ZoneInfoNotFoundError`/`ModuleNotFoundError` at startup and launchd restarted
  it forever (visible in `~/.jarvis/logs/sync.log`). The timezone argument is
  gone — APScheduler resolves the real local zone via tzlocal. Scheduler
  construction is extracted into `_build_scheduler(cfg)` with a regression test.
- **Deletion dialog hid the file path**: the confirmation description only named
  a file when one was being deleted; a database-only removal showed no path, and
  paths could read as a bare directory. `_remove_document` now always includes
  the full local path (or "no local file") and an unambiguous "KEPT" / "will be
  PERMANENTLY DELETED" line, in both the preview and the dialog.
- **Tool-arg display clipped filenames**: `repr(v)[:40]` cut off exactly the
  filename on a `file:///` URI. Replaced with a shared middle-ellipsis helper
  (`truncate_middle`, keeps head + tail) used by both the CLI and the webapp.

### Changed
- **Reverted local provider llama.cpp → Ollama** (`digest/llm.py`,
  `digest/config.py`). llama.cpp lasted exactly one iteration: running a
  separate `llama-server` with a fixed launch-time context window and a
  dedicated LaunchAgent proved fiddly ("llamacpp is hard to use"), whereas
  Ollama runs as a login-item, keeps the model resident, and honours a
  per-request context window. `OllamaProvider` replaces `LlamaCppProvider`;
  `make_provider` specs are now `"ollama" | "anthropic"` (default ollama);
  config keys `llamacpp_url`/`llamacpp_model` (+ their env vars) are replaced by
  `ollama_model` / `OLLAMA_MODEL`, default `qwen3-vl:30b` (a vision + thinking
  MoE that fits a 36GB M3 Max). Session provider matching in `check_resume` is
  now strict per provider name, so a session recorded under the retired
  `llamacpp` provider refuses to resume rather than replaying an incompatible
  history. Summary mode under Ollama converts the PDF to markdown first (via the
  existing `pdf_to_markdown`) rather than uploading the PDF.

### Added
- **bioRxiv digest source** (`digest/biorxiv/fetch.py`): `fetch_biorxiv` walks
  the details API for a server-side category (e.g. `bioinformatics`), and
  `fetch_biorxiv_keywords` matches free-text keywords (cytometry, spatial
  transcriptomics, scRNA-seq — topics bioRxiv has no category for) over the
  recent-preprint window, DOI-deduped. Wired into the pipeline after the arXiv
  loop; config keys `biorxiv_categories`, `biorxiv_keywords`, `biorxiv_days`.
- **Knowledge-base title dedup** (`_title_exists` in `digest/kb/store.py`): the
  same paper can now arrive via arXiv and bioRxiv under different URLs, so
  `add_paper` skips on a normalised-title match as well as a source-URL match.
  `add_papers_batch` returns `(added, skipped)` and the pipeline reports both.
  Manual adds prompt instead of skipping silently: `kb add` asks `[y/N]`, and
  the chat `add_document` tool returns an ask-the-user message plus a new
  `allow_duplicate` flag.
- **PDF figure captioning at ingest** (`digest/kb/images.py` +
  `add_figures` in `store.py`): embedded raster figures are captioned by the
  active provider's vision model (`describe_image` on both providers) and
  indexed as `[FIGURE p.N]` chunks (`annotation_kind="figure"`), so they share
  the parent PDF's delete/re-ingest sweeps. Config: `figure_captions`
  (kill-switch), `figure_max_per_doc`, `figure_min_pixels`. Private notes are
  never captioned under Anthropic — the images would reach the cloud.
- **Dark web UI**: a single dark palette via CSS custom properties (no toggle),
  chat bubbles capped at ~80 characters, the response-style setting moved from a
  sidebar box to a header ⋮ menu → modal (prefilled from `GET /settings`), and
  per-session rename (`rename_session`, `POST /sessions/{id}/rename`, a ✎
  button; indexed chat-chunk titles update via `update_chat_title`).

---

## [previous] — reliability, annotations, llama.cpp, sessions, security

A broad pass covering scheduling reliability, arXiv flakiness, the PDF
pipeline, PDF annotations, privacy/security hardening, the Ollama→llama.cpp
switch, and three new chat features (sessions, skills, response style).

### Fixed
- **Scheduled digest was broken**: `run_digest.sh` invoked the non-existent
  module `digest.run` (renamed to `digest.pipeline.run` long ago), so every
  launchd run failed instantly. The script is gone entirely (see daemon below).
- **arXiv flakiness**: the API's known empty-feed-with-HTTP-200 responses
  bypassed retries and silently produced 0 papers; malformed XML crashed the
  run. `digest/arxiv/fetch.py` now uses the `arxiv` package (built-in paging,
  per-page retries, 3s courtesy delay) with a `FetchError`-based retry layer
  on top; empty feeds are retried, `with_retries` backs off exponentially
  with jitter.
- **Privacy — `read_file` symlink bypass**: the private-dir check ran on the
  unresolved caller-supplied path; a symlink in a public folder pointing into
  `private/` leaked content to the cloud provider. Classification now uses
  `get_visibility()` on the resolved path — one policy for indexing and reads.
- **Privacy — stale visibility**: `refresh_vault` now re-checks each unchanged
  note's classification, so editing `private_vault_dirs` reclassifies indexed
  chunks (new `update_visibility()`, metadata-only, no re-embedding).
- **Privacy — cloud PDF upload**: `add_document` summary mode uploaded local
  PDFs to Anthropic without checking visibility. Resolved by the new invariant
  below rather than a per-path gate.
- **XSS**: the webapp markdown renderer didn't escape quotes in link hrefs —
  a crafted link in assistant output could break out of the attribute. `esc()`
  now escapes quotes and hrefs are validated with `new URL()`.

### Added
- **`jarvis-sync` daemon** (`digest/daemon.py`, entry point `jarvis-sync`) —
  one supervised process under launchd `KeepAlive` replacing the old weekly
  plist + `run_digest.sh` (both removed): weekly digest via APScheduler with
  catch-up across sleep *and* power-off (persistent stamp in
  `~/.jarvis/state/sync_status.json`); **PDF inbox watcher** (watchdog) that
  auto-indexes PDFs dropped into `[sync] pdf_watch_dir` as public full-text
  papers, with byte-hash dedup/update and wait-for-stable handling for cloud
  syncs; **periodic vault refresh** (default every 30 min). One failing job
  never kills the daemon; `kb sync-status` reports health. New `[sync]`
  config keys: `pdf_watch_dir`, `vault_refresh_minutes`, `digest_day`,
  `digest_hour`. The daemon no longer auto-starts the local LLM server.
- **PDF annotation extraction** (`digest/kb/annotations.py`): highlights
  (any colour; underline/squiggly/strikeout too) and typed notes (sticky
  notes, text boxes, comments on highlights) written by macOS Preview /
  Foxit Reader are indexed as their own chunks — `[HIGHLIGHT p.N]` /
  `[USER NOTE p.N]` prefixes, new metadata fields `annotation_kind`, `page`,
  `note_text`. Freehand/handwritten (Ink) annotations are not extractable
  (stroke geometry, no text). Wired into `kb add`, `add_document`,
  `refresh_vault`, and the daemon's inbox ingest; annotations share the
  parent PDF's source so deletes/re-ingests sweep them automatically.
  Re-saving a PDF with new highlights re-indexes it via the existing
  byte-hash change detection.
- **Persistent chat sessions** (`vault_chat/sessions.py`): every turn saved
  to `~/.jarvis/sessions/<id>.json`; webapp sidebar to resume/pin/delete;
  `vault-chat --list-sessions/--resume`. Retention: 50 most recent unpinned
  sessions (pinned exempt and uncounted). Sessions touching private content
  get a permanent private flag; exchanges are indexed as `doc_type="chat"`
  with the session's visibility, searchable via the new
  **`search_chat_history`** tool (cloud sees public sessions only); resuming
  a private session under the cloud provider is refused. **In-session
  compaction**: past `[chat] compact_after_tokens` (default 12000) old
  exchanges are summarised by the session's own provider, keeping the last
  `compact_keep_exchanges` turns verbatim — UI history stays complete.
- **User-defined skills** (`vault_chat/skills.py`): `*.md` files in
  `[chat] skills_dir` (default `~/.jarvis/skills`) are advertised in the
  system prompt as name + first-line description; the model loads full
  instructions on demand via the new `read_skill` tool.
- **Response style**: `[chat] response_style` free-text instruction appended
  to the system prompt; editable live in the webapp settings box, persisted
  back to `config.toml` via tomlkit (comment-preserving, atomic, chmod 600).
- **Cross-process write lock**: ChromaDB writes (daemon + webapp + CLI share
  one store) are serialised via `flock` on `<rag_dir>/.write.lock`.
- **`kb sync-status`** subcommand; `kb stats` warns about legacy private
  papers (see invariant).

### Changed
- **Invariant: papers are always public.** Only notes (vault files and
  note-type PDFs) can be private. Enforced at add time in `kb add` and
  `add_document`; this is what makes the cloud summary path safe by
  construction.
- **Local provider: Ollama → llama.cpp.** `OllamaProvider` and the `ollama`
  dependency are gone; `LlamaCppProvider` talks to an external `llama-server`
  (OpenAI-compatible API via the `openai` client, tool calling with
  `--jinja`). New `[chat]` keys `llamacpp_url` (default
  `http://127.0.0.1:8081/v1` — port 8081 because the webapp owns 8080) and
  `llamacpp_model`; `provider` default is now `"llamacpp"`; `ollama_model` /
  `OLLAMA_MODEL` removed. Local PDF summarisation now converts to markdown
  first (the old path sent the PDF bytes as an *image* to Ollama).
  **Migration**: update `~/.jarvis/config.toml` — replace `provider =
  "ollama"`/`ollama_model` with the new keys, and note the stale
  `rag_dir = "~/.seshat/rag"` in existing configs should be fixed to
  `~/.jarvis/rag` (then `kb reindex` or re-add).
- **PDF conversion: marker-pdf → pymupdf4llm** (`digest/kb/convert.py`,
  `pdf_to_markdown()` returns a string — no more temp-.md round-trips at any
  call site; orders of magnitude faster, no ML model downloads; lower
  fidelity on complex layouts/equations, accepted trade-off). Scanned PDFs
  without a text layer raise the new `ConversionError` (no OCR fallback).
  Standalone `convert-pdf` moved to `digest.kb.convert:main`; image
  extraction dropped (nothing consumed it; `write_images=True` is the
  one-line reinstatement if ever wanted). Existing marker-converted chunks
  are not retroactively reconverted.
- **Digest pipeline** runs whichever provider `[chat] provider` names; the
  llama.cpp path checks `llama-server`'s `/health` first and warns when the
  server's context size is smaller than the scoring call wants.
- Webapp restructured: `index.html` split into `static/style.css` +
  `static/app.js`; fetch errors render an error bubble instead of a stuck
  "Working..." placeholder.

### Security
- **Destructive actions now require a human.** `remove_document(confirmed=true)`
  no longer executes: the CLI prompts y/N in the terminal; the webapp shows a
  Confirm/Cancel dialog whose Confirm hits the new `/confirm-action` endpoint
  — outside the LLM loop, so prompt-injected deletions cannot fire.
- **Note files are never deleted from disk** — by anyone. The shared
  `delete_local_file()` choke point (used by `kb remove --delete-file` and
  the chat tool) only ever unlinks paper PDFs.
- The LLM-facing `index_vault` tool lost its destructive `force` option
  (`kb index-vault --force` remains for humans).
- Retrieved document content is wrapped in BEGIN/END RETRIEVED DATA markers
  with a system-prompt rule to treat it as data (defence in depth, not a
  guarantee).
- Webapp: `TrustedHostMiddleware` (DNS-rebinding), strict session-id
  validation before any path construction, still bound to 127.0.0.1.
- File permissions: config write-back and session files are 0600 (sessions
  dir 0700); `jarvis-sync` and `vault-chat` warn when `config.toml` is
  group/world-readable.

### Known issues / follow-ups
- `update_file_path` with `source="local"` would clobber all vault notes'
  metadata (vault note sources are not unique) — restrict to `file://`
  sources if ever touched.
- The `has_private` probe retrieves one private document into local process
  memory (never serialised or sent anywhere) — accepted for a single-user
  local tool.

---

## [previous] — project rename to jarvis

### Renamed
- GitHub repository: `ghar1821/seshat` → `ghar1821/jarvis`
- Project package name: `seshat` → `jarvis` in `pyproject.toml`
- Config directory: `~/.seshat/` → `~/.jarvis/` (config, auth, RAG store)
- Local project directory: `~/projects/seshat/` → `~/projects/jarvis/`
- launchd agent label: `com.putri.seshat` → `com.putri.jarvis`
- launchd plist file: `com.putri.seshat.plist` → `com.putri.jarvis.plist`

### Changed
- README blurb: no longer named after the Egyptian goddess of writing and knowledge; now named after Iron Man's J.A.R.V.I.S. ("Just A Rather Very Intelligent System")
- All `~/.seshat/...` path references across code, tests, and docs updated to `~/.jarvis/...`

---

## [earlier] — retrieval accuracy: BGE embeddings, cross-encoder re-ranking, section-aware chunking

Improves document-retrieval accuracy using techniques from the LlamaIndex playbook, implemented with the existing LangChain + sentence-transformers stack (no new dependencies, no framework migration). See `docs/DESIGN.md` → "Retrieval pipeline" and "Deferred retrieval improvements".

### Added
- **Cross-encoder re-ranking** in `search()` (`digest/kb/store.py`) — fetches the top `rerank_top_n` (default 25) dense candidates, then re-orders them with a local cross-encoder (`rerank_model`, default `cross-encoder/ms-marco-MiniLM-L6-v2`) and returns the top `n_results`. New `rerank: bool = True` parameter; the private-existence probe in `search_with_privacy_check()` passes `rerank=False`. Re-ranking runs after the visibility filter, so the privacy invariant is unchanged. `_get_reranker()` lazily loads the model (a singleton, like `_get_embeddings()`); `rerank_model = ""` disables it.
- **Section-aware chunking** — `add_texts()` now splits on markdown headers (`MarkdownHeaderTextSplitter`) before the recursive size split via the new `_split_markdown()` helper. Each chunk records `chunk_index` and a `section` breadcrumb (e.g. `"CRISPR screens › Results"`), and the breadcrumb is prepended to the embedded text. Headerless content (paper summaries) is unaffected.
- **Embedding-model guard** — `get_store()` tags the collection with `embed_model` and calls `_check_embedding_model_matches()`, which raises `RAGError` when a non-empty collection's model tag differs from config (or is absent, as in pre-upgrade collections). Fails loudly instead of silently mixing embedding spaces.
- **`kb reindex` command** (`digest/kb/cli.py` `cmd_reindex`) — re-embeds every stored chunk with the configured `embed_model` into a temporary collection, then swaps it in atomically. No LLM calls or re-summarising: chunk texts are already stored. This is the fix the guard points users to.
- **`build_embeddings(model_name, query_prefix)`** helper in `store.py` — L2-normalised embeddings with an optional query-side instruction prefix; shared by production and the test fixtures.
- **Config fields** (`digest/config.py`, `[rag]`): `query_prefix`, `rerank_model`, `rerank_top_n`.
- **`tests/test_retrieval_quality.py`** — a golden-set benchmark (paper summaries + markdown notes, ~22 queries) reporting hit-rate@5 and MRR@5, so retrieval changes are measurable and regressions are caught. Runs as a normal unit test (local cached models only).

### Changed
- **Default embedding model**: `all-MiniLM-L6-v2` → `BAAI/bge-small-en-v1.5` (same 384 dims, stronger retrieval; needs a query-side prefix, now set via `query_prefix`).
- **Default chunk size / overlap**: `2048/256` → `1024/128` (smaller chunks fit the cross-encoder's window better and give finer-grained matches).
- `tests/conftest.py` — the `embeddings` fixture now builds via `build_embeddings()` from the default config, so tests exercise the real query prefix and normalisation.

### Migration
- **Existing installs must run `uv run kb reindex` once after upgrading.** The embedding-model guard fires on any pre-upgrade collection (it has no `embed_model` tag), and the default model changed — `reindex` re-embeds with the configured model and writes the tag. To adopt `bge-small`, remove any `embed_model` pin from `~/.jarvis/config.toml` (or set it to `BAAI/bge-small-en-v1.5`) before reindexing.
- **Re-chunking** (to benefit from section-aware chunking / smaller chunks) is separate: run `uv run kb index-vault --force` for vault notes. Summary-mode papers (1–2 chunks) are unaffected; full-text papers keep their existing chunk boundaries until re-added (`kb remove` + `kb add --full-text`). `kb reindex` deliberately does not re-chunk — it only re-embeds existing chunk texts.

### Dependencies
- No new packages. The cross-encoder uses `sentence-transformers` (already required) and markdown splitting uses `langchain-text-splitters` (already required).

---

## [previous] — knowledge source toggle (DB only / AI fallback)

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
