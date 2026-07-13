# Changelog

Prototype stage — no deployments. Changes documented for development reference only.

---

## [current] — webapp: true parallel sessions, papers manager; unverified-metadata flag removed; daemon job logging

The webapp's `/chat` route used to apply every message to a single mutable
"active session" global, guarded by one global busy flag (`running_turn`).
That meant: a brand-new session that hit the busy guard could never be
revisited (it existed only in memory, with nothing on disk to resume); only
one session could ever be mid-turn at a time, so sending in session B while
session A was still generating 409'd even though the user wanted true
parallelism; and worst, a session swap racing a send could apply the
message to the wrong session entirely, because `/chat` never said which
session it meant — it just mutated whatever the shared dict currently
pointed at.

### Changed

- **`/chat` is now session-addressed.** `ChatRequest` gains a required
  `session_id`; the route resolves the session named in the request — the
  active in-memory object if the id matches (so a brand-new, not-yet-saved
  session can accept its first message), otherwise loads it from disk and
  runs the same `check_resume` safety checks `/sessions/{id}/resume` already
  applied. A message can no longer land on the wrong session.
- **The single `running_turn` slot became a `running` registry**
  (`{session_id: live Session object}`), so any number of sessions can be
  mid-turn at once, each in its own background thread and SSE stream. The
  busy guard, `sessions_delete`'s 409, and `GET /sessions`' `busy` field
  (now a list, not a single id-or-null) all key off this registry instead of
  a single global.
- **`pending_actions` entries are now session-scoped**
  (`{token: {session_id, action}}`). A new turn or a resume clears only its
  own session's stale confirmation dialogs, never another session's —
  including one that's mid-turn concurrently. `/confirm-action` itself still
  does no session check (token possession is the capability).
- **System prompt and tools are built fresh per turn** from the *resolved*
  session's own `kb_only`, rather than a cached global rebuilt only by
  `/config` and `/settings`. This also fixes a latent staleness bug: a
  `/config` change while viewing a different session, or a resumed
  session's own `kb_only`, used to be silently ignored by `/chat`.
  `POST /config` now updates both the default for new sessions and the
  active session's own flag.
- **Resuming a still-mid-turn session installs the live registry object
  directly** instead of a stale disk copy, which is what let `run_agent`'s
  `finally` block drop its old mid-turn-resume reinstall step entirely —
  there is no longer a stale copy to reconcile.
- **Frontend:** `sendMessage()` captures the active session id up front and
  posts it as `session_id`, so a send always targets the session it was
  typed into even across a session switch. The composer's disabled state
  is now per-session (`inFlight` set + the server's `busy` list via
  `updateComposerState()`), not a single global `sendBtn.disabled` — sending
  in one session no longer locks the composer for another. A failed send
  fully rolls back the optimistic user bubble and placeholder and restores
  the typed text (to the textarea if still viewing that session, to its
  draft otherwise) instead of leaving an error over an orphaned message.
  `loadSessions()` now runs on the first SSE event, so a brand-new session
  appears in the sidebar as soon as its file exists (the early save), and
  busy sessions get a pulsing-dot indicator in the sidebar.

#### Tests

- `tests/test_webapp_chat.py`: fixture inits the `running` registry; every
  `/chat` post now carries `session_id`; the mid-turn reinstall regression
  test is replaced with one driving the real resume route against a blocked
  turn (asserts `busy: true` and the live object installed); new tests for
  two genuinely parallel turns with no cross-contamination, an unknown
  `session_id` 404ing, and a message landing on the addressed (non-active)
  session rather than the active one.
- `tests/test_security.py`: seeded `pending_actions` entries adopt the
  `{session_id, action}` shape; `/sessions/new` is now proven to leave other
  sessions' dialogs pending (the opposite of its old behaviour); resume and
  new-turn tests both prove only the targeted session's tokens are cleared.

## Removed `meta_inferred` / the "unverified metadata" flag

Every locally-added PDF got flagged `meta_inferred: true` once auto-inference
ran, with reminders in `kb stats`, a `kb list --unverified` filter, a webapp
header banner (`GET /info` → `unverified_count`), and a `vault-chat` startup
line all nagging about it. In practice almost every paper ended up flagged
(inference nearly always runs) and getting the LLM to act on the reminder by
actually reviewing and fixing metadata rarely went anywhere — so the flag was
noise without a workable follow-through. Removed outright.

### Removed

- `count_unverified_papers()` (`jarvis/kb/store.py`) and its schema docstring
  line; the `kb stats` unverified-count reminder; `kb list --unverified`
  (filter, tag, arg, and help text); the `vault-chat` startup "N papers have
  unverified metadata" line; the webapp's `unverified_count` field in
  `GET /info` and its header banner (`showUnverifiedBadge`, `#unverified-banner`
  markup/CSS).
- The three add-path `meta_inferred` stamping blocks (`kb add`, chat
  `add_document`, daemon `ingest_pdf`) and the field itself from
  `resolve_pdf_metadata`'s return value.

### Changed

- `update_paper_metadata` (`store.py`), `kb set-meta`, and the
  `update_document_metadata` chat tool are unchanged in behaviour — still the
  way to correct a paper's title/authors/doi metadata-only, no re-embedding —
  just without the now-gone flag to clear.

## Daemon job logging states next-run time

`jarvis-sync`'s log used to be flooded with APScheduler's own "Added job ...
to job store default" noise at startup, and gave no indication of when a job
would actually run next — you had to read the schedule config and do the
arithmetic yourself.

### Added

- `main()` sets `logging.getLogger("apscheduler")` to `WARNING`, silencing
  APScheduler's own startup noise.
- After building the scheduler, one `job <id>: next run at <time>` line is
  logged per job (`_log_next_run_times`), computed via
  `job.trigger.get_next_fire_time(None, now)` since `job.next_run_time` stays
  `None` until `BlockingScheduler.start()` is actually running.
- A `EVENT_JOB_EXECUTED` / `EVENT_JOB_ERROR` listener (`_log_job_outcome`)
  logs `job <id> finished — next run at <time>` after every run, so the
  running log always answers "when will it run next" without cross-checking
  the schedule by hand.
- `ingest_pdf` now logs one line per successfully added/updated inbox PDF
  with its stored title, authors, doi, and source filename — the sync log
  shows exactly what metadata ended up in the KB for each ingested paper.

## Webapp papers manager

There was previously no way to review or fix a paper's metadata from the
webapp — only the LLM's own `update_document_metadata`/`remove_document`
tools, both requiring a chat round trip. A new ⋮ menu → "Papers…" modal lists
every indexed paper, searchable, with inline metadata editing and removal.

### Added

- `GET /papers?q=<search>`, `POST /papers/meta`, and `POST /papers/remove`
  (`jarvis/webapp/app.py`). The list route wraps `list_papers` (now sorted
  most-recent-first, with its default `limit` raised from 50 so a
  single-user KB is never silently truncated) and applies an optional
  case-insensitive substring filter over title/authors/doi/source. The meta
  route wraps `update_paper_metadata` (metadata-only, no re-embedding); the
  remove route builds the same `{ids, title, doc_type, source}` action
  `remove_document` does and hands it to the existing `execute_remove()` —
  **chunks only, never a file on disk**. `/papers/remove` skips the
  token-confirmed `/confirm-action` flow entirely: it is human-only by
  construction, since no chat tool references it and the model has no way to
  reach it.
- Frontend: a "Papers…" entry in the header menu opens a modal (same
  open/prefill/close pattern as the response-style modal) with a debounced
  search box and a scrollable table. Each row supports inline Edit (title/
  authors/doi become inputs, Save posts `/papers/meta` and re-renders the
  row) and Remove (a two-step in-modal confirmation stating the "Database
  entry only — files on disk are never touched by jarvis: `<path>`"
  invariant verbatim before the explicit Confirm posts `/papers/remove`).
- `tests/test_webapp_papers.py`: route coverage against a real ChromaDB
  store — listing, `q=` filtering (title/authors/doi/source), metadata
  update (only the given fields change) and its 404, removal and its 404,
  plus a regression pinning that `/papers/remove` never invokes
  `pathlib.Path.unlink` or `os.remove` (spied via monkeypatch) — the same
  "database entry only" invariant enforced everywhere else in the codebase.

## Diagnosable validation errors

A stale browser tab still running pre-upgrade JavaScript posted the old
`/chat` shape (no `session_id`), got a 422, and surfaced as
`Request failed: [object Object]` with nothing in any log — the rejection
happened before any route body ran, so chat.log never saw it.

### Added

- A `RequestValidationError` handler logs every 422 (method, path, field
  errors) to the vault-chat logger before returning FastAPI's standard
  detail shape, so schema-mismatch requests are diagnosable from chat.log.
- The frontend renders error `detail` of any shape readably
  (`errorDetail()` flattens FastAPI's validation-error list into
  `body.session_id: Field required` instead of `[object Object]`).
- README: hard-reload an already-open webapp tab after upgrading jarvis.

## Stale-handle diagnosis after `kb reindex`

`kb reindex` swaps in a rebuilt collection with a new UUID, so any jarvis
process that was already running (webapp, jarvis-sync, vault-chat) is left
holding a handle to the deleted collection — and every KB operation then
failed with the cryptic `Collection [<uuid>] does not exist`.

### Added

- Store errors are now routed through a shared diagnosis (`_diagnose_kb_error`):
  the "Collection … does not exist" signature raises `KBCorruptionError` with
  a message naming the actual fix (restart the process — nothing is lost),
  alongside the existing "Error finding id" → `kb reindex` corruption
  diagnosis. Applied on both the search and add-documents paths.
- `kb reindex` now ends by warning that already-running jarvis processes hold
  a stale handle and need a restart.

---

## [previous] — chunk-first retrieval; PDF notes removed; model config relocation + sync logging; skills as folders

The chat agent used to truncate every search hit to 300 characters and had no
way to read a stored document except `read_file`, which only opens vault
Markdown — so a question needing more than a snippet forced either a
speculative `read_file` call (which fails outright on PDFs) or a guess from
the truncated text.

### Changed

- **`_retrieve_papers` and `_search_notes` no longer truncate hits.** Each
  result now shows the full chunk text (chunks are ≤1024 chars by
  construction, and the existing `n_results` clamp of 20 still bounds the
  reply size) plus a `Section:` line when the chunk carries a markdown
  header breadcrumb. `_search_chat_history` keeps its 300-char truncation —
  those results are recall cues, not answer material, and are left alone.
- **`read_file`'s description** now says it reads one vault Markdown file and
  cannot open PDFs, pointing to the new `get_document` tool instead of
  claiming to return "the whole document."
- **`retrieve_papers`'s description** now notes that hits include the full
  matching passage — usually enough to answer from directly.
- **The system prompt's querying workflow** now reads: search first and
  answer from the full-text hits when they suffice; if not, refine the query
  and search again, or call `get_document(source)` to read the whole stored
  document page by page; `read_file` is only for vault text files found by
  `search_notes` and must never be called speculatively (neither must
  `get_document`). Anyone with a `~/.jarvis/system_prompt.md` override keeps
  their existing prompt text — this rewrite only touches the built-in default.

### Added

- **`get_document_chunks(source, store=None)`** (`jarvis/kb/store.py`):
  fetches every chunk sharing a source and returns them in reading order —
  body chunks first (by `chunk_index`), then annotation/figure chunks
  (identified by the `annotation_kind` metadata key). Returns `[]` for an
  unknown source, mirroring `update_file_path`.
- **`get_document` chat tool** (`source`, `page` — 1-based, default 1): pages
  through a document's stored chunks 15 at a time (~4K tokens), works for
  everything indexed including PDFs. Header format: `"<title>" — chunks
  1–15 of 87 (page 1 of 6). Call get_document(source, page=2) for more.`
  Privacy mirrors `read_file` exactly — under Anthropic, any private chunk
  raises `PrivacyError` before any content (even a title or length hint) is
  returned; under Ollama the call succeeds and flags the session private.
  A `storage_mode="summary"` document gets an appended note that the full
  text isn't in the KB (re-add with `mode='full_text'` for that). Wired into
  `_dispatch_tool`'s existing privacy/`RETRIEVED DATA`-wrapping tuple
  alongside `read_file`/`retrieve_papers`/`search_notes` — no other changes
  needed in the webapp or either provider.

### Rejected

- Automatic neighbor-chunk expansion on a search hit — over-engineering for
  a single-user app. The agentic search → `get_document` ladder already gets
  the model surrounding context transparently, one tool call at a time.

### Tests

- `tests/test_store.py`: `get_document_chunks` orders body chunks before
  annotation/figure chunks; unknown source returns `[]`.
- `tests/test_privacy_guard.py`: `get_document` — public doc fine under
  Anthropic; a private source hard-stops with `PrivacyError` and no content
  leak; the local provider reads it and reports `saw_private=True`.
- New `tests/test_chat_tools.py`: `_retrieve_papers`/`_search_notes` return
  text beyond the old 300-char cutoff and include the `Section:` breadcrumb;
  `get_document` pagination (22 chunks → page 1 of 2 / page 2 of 2, correct
  chunk ranges); unknown-source message; the `storage_mode="summary"`
  honesty note; and `_dispatch_tool("get_document", ...)` wraps output in
  the `RETRIEVED DATA` markers and flags a session private on a local-provider
  private hit, exactly like the other retrieval tools.

---

### PDF notes removed — local PDFs are always public papers

Local PDFs could previously be added as either a `"paper"` or a `"note"`
(`--doc-type`/`--visibility`), and a note-type PDF was the only supported way
to index a *private* local document. This split the "local PDF" concept
across two `doc_type` values with different storage rules (notes were always
full-text, hash-tracked for `refresh_vault`'s Phase 2) for no real benefit —
private local documents are better served by moving them into the Obsidian
vault, which already has first-class privacy handling. Decision: **local PDFs
are now always public papers**; notes come exclusively from the vault's
`.md` files.

#### Changed

- **`kb add <pdf>`, the chat `add_document` tool, and the daemon's inbox
  ingest** no longer accept a `doc_type`/`visibility` choice for local PDFs —
  every local PDF is unconditionally indexed as `doc_type="paper"`,
  `visibility="public"`. The `--doc-type`/`--visibility` CLI flags are gone;
  the chat tool's schema dropped the matching properties.
- **`resolve_pdf_metadata()`** (`jarvis/kb/metadata.py`) dropped its
  `provider_str`/`doc_type`/`visibility` parameters along with the
  private-note-skips-inference guard they gated — inference always runs now,
  since a local PDF can never be private.
- **`index-vault --force`** now clears every indexed note chunk. The filter
  that used to spare PDF-note chunks (identified by an absolute `.pdf`
  `file_path`) from the clear is gone along with the PDF-note concept it
  protected.
- **`kb stats`'s legacy-private-paper warning** now suggests moving the
  paper's content into the vault as a note (rather than re-adding it with
  `--doc-type note`, which no longer exists).

#### Removed

- `refresh_vault`'s Phase 2 (`jarvis/kb/store.py`) — the pass that tracked
  local PDF notes by absolute `file_path`, warning if the file went missing
  and re-converting it on a byte-hash change. PDF notes no longer exist, so
  there is nothing left for it to track.
- `_caption_figures_for_note` (`jarvis/kb/store.py`) — only called from the
  now-removed Phase 2.
- Phase 1's `.pdf` skip filter — it existed solely to hand PDF notes off to
  Phase 2 without them being misread as a deleted vault file; with Phase 2
  gone, Phase 1 now treats every `doc_type="note"` chunk's `file_path`
  uniformly.

#### Added — `kb doctor` legacy PDF-note migration

Existing knowledge bases may still hold `doc_type="note"` chunks with an
absolute `.pdf` `file_path`, added before this change. `kb doctor` now finds
them (`find_pdf_notes()`) after its health checks pass:

- **Public** legacy PDF notes are listed, then a single `y/N` prompt
  reclassifies all of them to `doc_type="paper"` in one pass
  (`reclassify_notes_as_papers()`) — only `doc_type` changes;
  `content_hash`/`storage_mode`/`file_path` are left exactly as they were, so
  the result has the same shape a daemon-ingested paper carries.
- **Private** legacy PDF notes are never silently made public. `kb doctor`
  only lists them, with two resolutions: `kb remove <source>` then re-add
  the PDF as a public paper, or move its content into the vault as a private
  `.md` note. It keeps reporting them, unprompted, until one of those is done.

**Upgrade note:** if `kb doctor` reports legacy PDF notes, resolve them
before relying on `index-vault --force` — an un-migrated private PDF note is
just a `doc_type="note"` chunk like any other, so `--force` (which clears
every note chunk) will remove it same as a deleted vault file.

#### Tests

- `tests/test_metadata.py`: `resolve_pdf_metadata` signature change; the
  private-note-skips-inference test removed (nothing left to skip).
- `tests/test_store.py`: `test_refresh_vault_preserves_pdf_notes` removed
  (the bug it regression-tested no longer has a code path); new coverage for
  `find_pdf_notes` (ignores vault `.md` notes, groups a legacy PDF note's
  chunks by source) and `reclassify_notes_as_papers` (flips only `doc_type`,
  no-op on an empty source list).
- `tests/test_kb_cli.py`: new `_check_legacy_pdf_notes` coverage — a public
  legacy PDF note reclassifies on `y`, stays a note on `n`; a private one is
  only listed (the reclassify prompt is never shown for it); an empty store
  prints nothing.
- `tests/test_privacy_guard.py`: removed the two `_add_document` invariant
  tests (`doc_type='paper'`+`visibility='private'` rejection, and the
  private-note-PDF allow path) — both exercised code paths that no longer
  exist.
- `tests/test_reingest_replace.py`: dropped the now-nonexistent
  `doc_type`/`visibility` fields from the CLI args helper and the chat
  `_add_document` call in the same-title-different-source test.

### Model config relocation + sync daemon LLM logging

`anthropic_model` lived under `[digest]`, which was misleading — it is the
model used for chat too whenever `provider = "anthropic"`, and for the
digest pipeline regardless of `[chat] provider`. It now lives under `[chat]`
next to `provider`/`ollama_model`, where the other model-selection keys
already are.

#### Changed

- **`[chat] anthropic_model`** is now the canonical config key
  (`jarvis/core/config.py`). A `[digest] anthropic_model` still works as a
  fallback for existing configs, but `load_config()` prints a one-line
  warning telling the user to move it — no silent auto-rewrite of the
  config file. Precedence: env `ANTHROPIC_MODEL` > `[chat]` > `[digest]`.
  Default value is unchanged (`claude-sonnet-4-6`).
- **`active_model(cfg)`** (`jarvis/core/llm.py`, next to `make_provider`)
  returns `cfg.anthropic_model` or `cfg.ollama_model` depending on
  `cfg.provider` — consolidates the three copies of that conditional that
  had drifted into `jarvis/chat/chat.py`, `jarvis/webapp/app.py`, and
  `jarvis/digest/pipeline/run.py` for display purposes.
- **`jarvis-sync` now logs which LLM it's using.** A startup line names the
  active provider/model and the embedding model; `run_digest_job` logs the
  provider/model right before handing off to the pipeline; `ingest_pdf`
  logs the model performing metadata inference (runs on every
  added/updated inbox PDF); `_caption_figures` logs the model only when
  captioning actually fires (figures found and `figure_captions` on). One
  line per job invocation, nothing logged on a no-op — `sync.log` now
  answers "which model produced this?" without cross-referencing config.

#### Upgrade note

Move `[digest] anthropic_model` to `[chat] anthropic_model` in
`~/.jarvis/config.toml`. jarvis keeps reading the old location as a
fallback and will warn at startup until it's moved.

#### Tests

- `tests/test_config.py`: reading from `[chat]`; `[digest]` fallback still
  works and prints the warning (`capsys`); `[chat]` wins over `[digest]`
  with no warning; env wins over both; default when unset.
- `tests/test_llm.py`: `active_model()` for both providers.
- `tests/test_daemon.py`: caplog coverage of the digest-job log line and
  the `ingest_pdf` metadata-inference log line (present on add, absent on
  the skip path).

### Skills as folders

A skill used to be a single flat `.md` file, which had no room for the
templates, checklists, or reference material a real procedure often needs.

#### Changed

- **Skill format** (`jarvis/chat/skills.py`): a skill is now a folder,
  `skills_dir/<name>/SKILL.md`, plus any supporting files the instructions
  reference (any depth under the folder). The folder name is the skill
  name, replacing the old filename-stem convention.
- **Description parsing** now checks SKILL.md's leading `---` frontmatter
  block for a `description:` key first (a small hand-rolled single-line-value
  parser — no new dependency), falling back to the first non-empty body
  line (`#` stripped) exactly as before when there's no frontmatter or no
  description key.
- **`read_skill(name, skills_dir, file=None)`** gained the `file` parameter.
  With no `file`, it returns SKILL.md's content followed by a
  "Supporting files:" listing (sorted relative paths, `rglob` of the folder
  minus SKILL.md itself; omitted when there are none). With `file`, it
  returns that one supporting file's content instead, capped at 64 KB
  (a clear error string above that size). `file` is untrusted LLM input,
  so it gets the same rigor as `name`: absolute paths and `..` are rejected
  outright, then `resolve()` + `relative_to()` confirm the path stays inside
  the skill folder — this also defeats a supporting file that is a symlink
  pointing outside it.
- **`READ_SKILL_TOOL`** (`jarvis/chat/chat.py`) gained the optional `file`
  parameter, describing it as reading one of the skill's supporting files
  by the path shown in the "Supporting files:" listing; `_read_skill`
  passes it through unchanged otherwise.

#### Removed

- **The flat `skills_dir/*.md` format.** No dual-format support — a stray
  flat file no longer loads. `list_skills` prints a one-line warning per
  stray file naming the fix, and a folder missing SKILL.md warns and is
  skipped, rather than either failing silently.

**Upgrade note:** move each existing `skills_dir/x.md` into its own folder
as `skills_dir/x/SKILL.md` (`mkdir x && mv x.md x/SKILL.md`). Flat files
left behind print a warning but otherwise stop loading silently.

#### Tests

- `tests/test_skills.py` rewritten for the folder format: frontmatter
  description parsing, fallback to the first body line, stray-flat-file
  warning (`capsys`) and skip, folder-without-SKILL.md warning and skip,
  `read_skill` default output (content + sorted supporting-files listing,
  omitted when none), `read_skill(file=...)` content, traversal rejection
  on both `name` and `file` (including a symlink escaping the skill
  folder), unknown skill/file errors, oversize supporting-file rejection.

## [previous] — webapp chat fixes: crash logging, mid-turn message persistence, stacked confirmations, per-session drafts

Four bugs reported against the webapp's chat flow, all traced to the same
`run_agent`/`_session` area of `jarvis/webapp/app.py`:

### Fixed

- **Errors weren't logged.** `run_agent` (the background thread each `/chat`
  call runs) now imports the same `vault-chat` logger `chat.py` already
  writes `~/.jarvis/logs/chat.log` with. Its `except LLMError` branch calls
  `log.exception(...)` before building the `⚠️` reply, and a new broad
  `except Exception` catches anything else (a genuine bug, not an expected
  provider failure), logs the full traceback, and replies with
  `⚠️ Internal error: ...` instead of leaving the SSE stream to hang forever
  with the "Working..." placeholder stuck on screen. Both error branches
  persist the error turn with `save_session`, so it survives a refresh. Both
  the reply event and the sentinel that ends the stream now live in a
  `finally` block, so the browser always gets a terminating response no
  matter which path the turn took. The CLI's own `except LLMError` in
  `run_session` (`chat.py`) gained the same `log.exception(...)` call.
- **A sent question could disappear.** The user's message was only appended
  to the in-memory `Session` and saved to disk *after* the LLM call
  returned — switching sessions (or a crash) mid-turn lost it from the
  sidebar's history until the reply eventually landed back on the original
  session. `run_agent` now calls `save_session(session)` (no `store=`, so no
  indexing/prune side effects) immediately after recording the user turn,
  before the agentic call even starts.
- **Bulk removal confirmations superseded each other.** `_session`'s single
  `pending_action` slot has become `pending_actions: {token: action}`. Each
  confirmation dialog owns its own token; `/confirm-action` pops only that
  token, so N stacked dialogs (e.g. the model proposing removal of several
  documents in one turn) are all independently confirmable instead of the
  next dialog's token clobbering the previous one. The whole dict is cleared
  on a new `/chat` turn, `POST /sessions/new`, resuming a session, and
  deleting the active session — an abandoned dialog's token then 409s
  instead of silently executing later.
- **The input draft leaked between sessions.** Pure frontend fix:
  `static/app.js` now keeps a `drafts` map keyed by session id.
  `switchDraft(newId)` saves the outgoing session's textarea text and loads
  the incoming session's (or blank), and is called from `resumeSession`,
  "New chat", and session delete.

### Added

- **Busy state**, to make the message-persistence fix visible in the UI: a
  new `running_turn` field on `_session` (the id of the session whose turn is
  currently in flight, or `None`). `/chat` 409s ("a reply is still being
  generated — wait for it to finish") if a turn is already running;
  `DELETE /sessions/{id}` 409s on the busy session; `GET /sessions` returns
  `busy` (the running session's id or `null`); `POST /sessions/{id}/resume`
  returns `busy: bool` for that session. The frontend's `resumeSession` shows
  a "Working..." placeholder and polls `/sessions` every ~2 s (guarded by a
  generation counter, bumped only on a successful resume, so a stale poll
  from an earlier resume can't overwrite a conversation the user has since
  switched away from again — and a failed resume doesn't kill the previous
  session's still-legitimate poll) until the busy flag clears, then
  re-renders from `/history`. Because a mid-turn resume installs a
  fresh-from-disk session object that the background thread never writes to,
  `run_agent` re-installs its completed object at the end of the turn
  (before `running_turn` clears) — otherwise the poll's `/history` fetch
  would render the stale copy and the reply, though saved, would never
  appear on screen.

### Tests

- `tests/test_security.py`: the single-slot `pending_action` tests rewritten
  for the dict — two stacked tokens independently confirmable, cancel pops
  only its own token, an unknown/cleared token 409s without disturbing the
  rest of the dict, and `/sessions/new`/resume both clear the dict.
- New `tests/test_webapp_chat.py`: a real `Session` plus a fake provider
  wired into `/chat` via `TestClient` (no live LLM) covers the
  save→turn→save ordering (the user message is present at the *first* save),
  the busy guard (second `/chat` 409, delete-busy 409, `/sessions` reports
  `busy`, `running_turn` clears once the turn drains), the crash path (an
  uncaught exception still yields a reply event, the stream still
  terminates, `caplog` sees an `ERROR` record with a traceback, and the
  error turn is saved), the `LLMError` path (`⚠️` reply, logged, session
  saved), and the mid-turn resume regression (a blocked turn's completed
  session object is re-installed over a fresh copy swapped in while it ran,
  so `/history` serves the finished reply).

Bug 8 (draft leakage) is frontend-only with no JS test harness in this repo —
verified manually; see `docs/TESTING.md`.

## [previous] — restructure into a single `jarvis` package

**Run `uv sync` after pulling this change** — the installed package name is
unchanged (`jarvis`), but every module's dotted path has moved; a stale
editable install can leave `import digest` / `import vault_chat` resolvable
from cached `.pyc` files until the venv is resynced.

Mechanical restructure — no behaviour change. Everything that used to live
under three top-level packages (`digest/`, `vault_chat/`, `webapp/`) now lives
under one: `jarvis/`.

- `digest/config.py`, `digest/errors.py`, `digest/llm.py` → `jarvis/core/`
  (shared infrastructure with no product-specific logic of its own)
- `digest/arxiv/`, `digest/biorxiv/`, `digest/pipeline/` → `jarvis/digest/`
  (the automated weekly digest, unchanged internally)
- `digest/kb/` → `jarvis/kb/` (knowledge base management, unchanged internally)
- `digest/daemon.py` → `jarvis/sync/daemon.py` (the `jarvis-sync` background
  daemon)
- `vault_chat/` → `jarvis/chat/` (the conversational KB agent)
- `webapp/` → `jarvis/webapp/` (the FastAPI browser UI)

`jarvis/kb/cli.py` also had two concerns split out of it that never belonged
next to the `kb` CLI surface itself:
- `cmd_sync_status` (+ its `..daemon` import) moved to a new
  `jarvis/sync/status.py`, since it reports on the sync daemon, not the
  knowledge base
- `cmd_add_digest` and its two Markdown-parsing helpers
  (`_parse_digest_file`, `_parse_paper_block`) moved to a new
  `jarvis/digest/import_digest.py`, since they are about importing digest
  files, not managing the KB

`kb`'s subcommands, help text, and behaviour are unchanged — `cli.py` now
just imports `cmd_sync_status` and `cmd_add_digest` from their new homes and
registers them exactly as before. All `uv run` entry points
(`run-digest`, `jarvis-sync`, `convert-pdf`, `vault-chat`, `kb`, `webapp`)
keep their names; only the `pyproject.toml` module targets they point at
changed. Full test suite (211 passed, 3 deselected) is unchanged before and
after.

## [previous] — digest tiers with full-text must-reads, opt-in figure captioning, periodic PDF inbox scan, digest catch-up

**Run `uv sync` after pulling this change** — the `watchdog` dependency was
removed.

### Changed behaviour
- **Figure captioning is now OFF by default** (`[rag] figure_captions` default
  `true` → `false`) — each figure costs a vision-model call. Opt in per
  document with `kb add --figures` or the chat tool's `with_figures=true`
  param on `add_document`; `add_figures()` gained a keyword-only
  `enabled: bool | None` (None = follow config, True = force for this
  document; the private-note privacy guard is never overridden). The daemon
  inbox and vault refresh stay config-gated, so they now no-op by default.
- **Duplicates replace instead of duplicate**: answering yes to a
  same-source duplicate (`kb add` `[y/N]` prompt, or `add_document` with
  `allow_duplicate=true`) deletes the old entry's chunks — annotations and
  figures share `source`, so the whole entry is swept and re-added. This is
  the "reingest paper X with figures" path end-to-end. A
  same-title-but-different-source duplicate still adds a separate entry. The
  delete only runs after the new content has been produced (PDF downloaded
  and converted, or summary generated), immediately before the add call — a
  failed download/conversion/summary leaves the old entry untouched instead
  of destroying it before the replacement exists.
- **Digest indexing is tiered** (previously: score ≥ 9 → summary, everything
  else → nothing):
  - **≥ 9 → full text**: the arXiv PDF is downloaded inside the digest job,
    converted, and chunked with score/track metadata and the title/authors
    embed header — no `summarize()` call. bioRxiv links (doi.org, no
    derivable PDF URL) and failed downloads fall back to a summary entry
    built from the digest's own text; one 404 never fails the digest job.
  - **8–8.9 → summary entries** (new — these papers were previously not
    indexed at all), reusing the scoring run's summary+why, zero extra LLM
    calls.
  - **< 8 → not indexed per-paper**, but now reachable via the digest
    document (below).
- **The weekly digest `.md` is itself indexed** as a new `doc_type="digest"`
  with a `file://` source and dated title, so every mentioned paper is
  searchable. `retrieve_papers` now queries `doc_type=["paper", "digest"]`
  (`search()` accepts a doc_type list, `$in` filter); `kb stats` reports a
  digest count. `"note"` was deliberately not reused — `refresh_vault` would
  have deleted the digest entries as missing vault files.
- **PDF inbox is scanned periodically instead of watched**: the watchdog
  Observer, event handler, and ingest worker thread/queue are gone. A new
  `pdf_scan` job sweeps `pdf_watch_dir` every `[sync] pdf_watch_minutes`
  (default 30, validated ≥ 1) and at daemon start; byte-hash dedup makes
  the sweep idempotent, so saving highlights costs at most one re-ingest
  per interval instead of one per save. Inbox latency is now at most one
  interval. Inbox-not-mirror semantics unchanged.
- **Digest slot moved 02:00 → 05:00** (`digest_hour` default), and a missed
  digest now catches up without a restart: `run_digest_catchup_job`
  re-checks the persisted last-success stamp at daemon start **and every 6
  hours** (job id `digest_catchup`). A module-level non-blocking lock in
  `run_digest_job` guards against the cron and catch-up jobs double-firing
  (separate APScheduler ids, so `max_instances=1` alone can't).
- The digest header now shows the actual generation time instead of a
  hardcoded "Generated 03:00".

### Dependency
- `watchdog` removed (no filesystem-event watcher any more) — run `uv sync`.

---

## [previous] — actionable KB-corruption errors, metadata inference + hybrid retrieval, one-shot removal confirm

**⚠️ `uv run kb reindex` is REQUIRED after this change.** It clears any
existing index corruption (see below) and prepends the title/authors header
to old paper chunks so author-name queries work against papers indexed
before this change too. Not run automatically — this is destructive-adjacent
enough (a full re-embed) that it stays a deliberate, user-run step.

### Fixed
- **Segfault in `_add_document`** (`vault_chat/chat.py`): the papers-are-
  always-public validation ran *after* `store = get_store()`, so a rejected
  private-paper call still opened the live store first. Against a corrupted
  index this could hard-crash the process (a Rust-side ChromaDB segfault,
  uncatchable in Python). Reordered: the cheap argument check now runs before
  any store or provider interaction.
- **Opaque "internal error" on a corrupted index**: `similarity_search`
  failing with ChromaDB's `"Error finding id"` (a stale HNSW reference to a
  deleted chunk) was flattened into a generic `RAGError` that the LLM
  paraphrased as an unhelpful internal error. New `KBCorruptionError`
  (`digest/errors.py`) is raised instead, naming `uv run kb reindex` as the
  fix; `retrieve_papers`, `search_notes`, and `search_chat_history` relay it
  to the user verbatim instead of paraphrasing or retrying. New `uv run kb
  doctor` command diagnoses this proactively (open store → count → search
  probe) — note that on a badly corrupted store even `count()` can segfault
  the process; that abrupt death is itself diagnostic.
- **Author-name retrieval failures**: for full-text papers the author names
  appeared in *no* chunk, so no retriever could ever match a query like
  "papers by Vaswani" — this was a data-absence problem, not a ranking one.
  `add_texts` gained `embed_header`, prepended to the embedded text of every
  chunk (title, or `"{title} — {authors}"`); `add_paper`'s content now
  includes an authors line too.
- **Wrong / missing paper authors**: local PDFs previously fell back to the
  filename as the title with no authors at all. New `digest/kb/metadata.py`
  infers title/authors/DOI from a PDF's first pages (DOI via regex first,
  then one small LLM call for whatever's left), wired into every local-PDF
  add path (`kb add`, chat `add_document`, daemon `ingest_pdf`). A private
  note's text still never reaches a cloud provider — inference is skipped
  under Anthropic with a visible warning in that case.

### Added
- **Hybrid BM25 + reciprocal-rank fusion retrieval**, gated by `[rag] hybrid`
  (default `true` — **this changes ranking for every existing search**, not
  just new installs). `_hybrid_search` fuses dense (embedding) and sparse
  (BM25, `rank-bm25`) rankings over the same privacy-filtered candidate pool,
  so privacy holds by construction. `hybrid = false` reproduces the old
  dense-only pipeline exactly.
- **`doi` metadata field** for papers: inferred for local PDFs, passed
  through from the arXiv API result when present, surfaced in every
  formatter (`retrieve_papers`, `list_papers`, `kb list`).
- **Verified-metadata loop**: inferred fields are flagged `meta_inferred:
  true`. `kb set-meta <source> [--title] [--authors] [--doi]` and the chat
  tool `update_document_metadata` apply a human correction (metadata only,
  no re-embedding) and clear the flag. Reminders: `kb stats` and `kb list
  --unverified`, a dismissible banner in the webapp header (`GET /info` →
  `unverified_count`), and one `vault-chat` startup line.
- **One-shot removal confirmation**: `remove_document` dropped the
  `confirmed` round-trip — a single call immediately shows the human
  confirmation prompt (terminal y/N or webapp dialog); only that out-of-band
  answer executes the removal.
- **File deletion removed from the codebase wholesale**: `delete_local_file`,
  `kb remove --delete-file`, and the tool's `delete_file` param are gone —
  not just disabled. `execute_remove()` only ever deletes ChromaDB chunks;
  every preview and dialog states the same invariant line verbatim:
  "Database entry only — files on disk are never touched by jarvis: `<path>`".
- **Stale confirm-dialog token guard** (webapp): one-shot confirms make it
  possible for an older, unclicked dialog to still be on screen when a newer
  removal is requested. `request_confirmation` now tags each pending action
  with a UUID token; `POST /confirm-action` 409s unless the posted token
  matches the currently pending one.

### Dependency
- `rank-bm25` added for hybrid retrieval — run `uv sync`.

---

## [previous] — copy-as-markdown for chat responses

### Added
- **Copy button on every assistant response** (`webapp/static/app.js`,
  `webapp/static/style.css`): a hover-revealed button in the top-right corner of each response
  bubble copies the whole reply to the clipboard as raw markdown, with a ~1.5s "✓" confirmation.
  Extracted into a shared `buildAssistantBubble()` helper used by both the history-restore path
  and the live SSE reply path, so newly-streamed and page-refreshed responses behave identically.
- **Native selection-copy now yields markdown**: selecting text inside an assistant response and
  copying it with the OS-native Cmd+C/Ctrl+C now places markdown notation on the clipboard (bold,
  code, headings, lists, links) instead of plain rendered text, via a new `htmlFragmentToMarkdown()`
  walker that mirrors `renderMarkdown`'s element vocabulary in reverse. This replaces the originally
  proposed "create Obsidian notes from conversations" feature (TODO item 4) — copy/paste into the
  user's own vault was judged simpler than a write path into Obsidian, and this change makes that
  manual copy markdown-faithful. Selections outside an assistant bubble, or spanning more than one,
  are untouched (default browser copy behaviour).

---

## [previous] — webapp bug fixes: modal visibility, session rename/pin, tool-call logging

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
