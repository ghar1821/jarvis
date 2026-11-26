# Jarvis

### NEW FUNCTIONALITY

- Jarvis is now a read-only knowledge server for MCP clients (Claude
  Desktop, Google Antigravity, Claude Code, …). The client is the assistant;
  jarvis makes no LLM calls and holds no API keys (PR #5):
  - Added `jarvis-mcp`, started by the client, with seven read-only tools:
    `search_kb`, `get_document`, `read_note`, `list_documents`, `kb_stats`,
    `fetch_recent_papers` and `reveal_in_finder`. Results show each file's
    absolute path and, for papers, a DOI link.
  - The index now mirrors two folders: the vault (minus private folders) and
    a new `papers_dir`. Save a PDF there to add it, delete it to remove it.
    Moving or renaming a PDF is followed without re-indexing.
  - Paper title and authors come from Crossref using the DOI printed in the
    PDF (only the DOI is sent), then the PDF's own metadata, then the filename.
  - `kb serve` replaces `kb server` + `jarvis-sync`: one process runs the
    index and syncs both folders every `sync_minutes`.
  - `kb rebuild` recreates the whole index from the two folders. It is also
    the fix for an index too damaged to open, since it never reads the old one.
  - Upgrading is one command: `kb rebuild` drops every entry not backed by a
    file (arXiv-URL papers, PDFs outside `papers_dir`, digests, chat history,
    private notes). `kb doctor` previews that list first, with a DOI link for
    each paper in case you want to save its PDF into `papers_dir`.
  - `jarvis-mcp` needs `kb serve` running, because it only ever connects to
    the index server. When the server is down the tools say how to start it,
    and they recover by themselves once it's up, without restarting the client.
  - The minimum config is two lines under `[kb]`: `vault_path` and
    `papers_dir`. There are no API keys.
  - The weekly digest is now a scheduled task in your MCP client, using
    `fetch_recent_papers`. A ready-to-paste prompt is in
    `docs/examples/weekly-digest-task.md`.
  - Obsidian's `.trash/` and other dot-folders are no longer indexed.

- Added `uv run kb server`, which now owns and manages the knowledge base.
  It must be run alongside `jarvis-sync` and the webapp.
  The `kb` commands do not need it as they read the index directly from the database.
  This fixes the bug where chat searches failing with "the index is likely damaged: run
  `uv run kb reindex`" when nothing was damaged. Every process opened
  its own copy of the knowledge base without any synchronisation process
  which causes stale index in the copy kept by the webapp when `jarvis-sync` rewrote
  them (PR #4).

- Turned off third-party telemetry. HuggingFace usage reporting was on
  by default and was never disabled (PR #4).

- The knowledge-base directory is now created with `0700` permission (PR #4).

- Added the ability to stop a reply while it is being generated (PR #3).

- `Ctrl-C` on `uv run webapp` now cancels every live turn on the way out
  (PR #3).

- Added an indicator for compaction, which used to happen invisibly (PR #3).

- Update Jarvis into a general assistant and introduced OpenRouter (PR #1):
  - Added support to use OpenRouter instead of just Anthropic or Ollama.
    Model can be switched mid conversation and a real time cost is shown.
    The model picker reads the list of models from the config file.
    Thus, new models can be added without restarting the webapp (PR #1).
  - Added support to store schema for vault notes.
  - Added support for agent to write documents into a folder outside of
    the vault, and ui to reveal the location for user to manually copy to their
    vault or wherever they want. 
    Agents can suggest changes to the document which must be approved or rejected
    by the user ala git diff style.
  - Added a document (latex, md, txt, csv) editor to the webapp. Editor is
    provided by CodeMirror API. Documents can be previewed in the webapp and 
    exported into PDF file when needed.
  - Added version history for documents. Every earlier version is kept and can
    be restored from the editor. Restoring is itself undoable.
  - Added automatic clean up of the draft folder. Documents untouched for 30
    days are swept, unless marked to keep.

- Added new CLI commands to inspect what is indexed and what is available
  (PR #1): `kb schema`, `kb list --notes`, `kb models`, `kb drafts`.

- Webapp prints its configuration on startup, and ⋮ → Show config… shows the
  same thing in the UI. API keys show as set or not set, not their value (PR #1). 

- Prompts used to instruct agent on how to behave and how to summarise and select
  papers for paper digest are now editable from the UI. 
  Default prompts now exist in the repo and are copied to
  `~/.jarvis/prompts/` on first run. 
  In the UI, ⋮ → Edit prompts… edits the copy, and a
  revert button puts the default back (PR #1).

### BREAKING CHANGES

- The previous version of jarvis, the full assistant with its own web UI, is
  archived on the `archive/pre-mcp-simplification` branch. It was cut from
  `main` at `c231389` (PR #4), just before this change, and is kept for
  reference only (PR #5).

- Removed everything the MCP client now does (PR #5). Your old
  `~/.jarvis/sessions/`, `~/.jarvis/drafts/` and `~/.jarvis/prompts/` folders
  are left untouched:
  - The webapp: chat, sessions, model picker, cost display, prompt editor,
    drafts, the document editor, PDF export. Removed with it: the
    Ollama/Anthropic/OpenRouter providers, and `[chat]`, `[auth]`,
    `[models]`, `[openrouter]` and `[drafts]` config.
  - All LLM work inside jarvis: summary-mode papers, LLM title/author
    inference, figure captioning, and the scheduled digest pipeline
    (`run-digest`, `[digest]` config).
  - Write commands: `kb add`, `kb add-digest`, `kb remove`, `kb clear`,
    `kb set-meta`, `kb update-path`. Files are the source of truth instead.
  - Other commands: `kb index-vault` (now `kb sync`), `kb reindex` (now
    `kb rebuild`), `kb server` (now `kb serve`), `kb sync-status` (now
    `kb status`), `kb models`, `kb drafts`, `jarvis-sync`, `webapp` and
    `convert-pdf`.

- Entries not backed by a file are dropped by `kb rebuild`: papers added by
  arXiv URL or from outside `papers_dir`, weekly digest documents, and indexed
  chat history. They stay searchable until you rebuild (PR #5).

- Private vault folders are no longer indexed at all, instead of being
  filtered per provider. Every MCP client sends what it reads to a cloud
  model, so the only safe place for a private note is outside the index.
  Private notes are deleted from the index on the first sync (PR #5).

- Config moved to `[kb]`: `vault_path`, `private_vault_dirs`, `papers_dir`
  (was `[sync] pdf_watch_dir`) and `sync_minutes` (was
  `vault_refresh_minutes`). The old locations still work and print a note.
  Settings that no longer do anything are named once at start-up (PR #5).

- Remove vault chat access via terminal to simplify codebase and deprecate 
    features that are hardly used (PR #1).

- Chat tools renamed to make naming reflect more of a general assistant (PR #1):
  - `retrieve_papers` + `search_notes` became one `search_kb`
  - `list_papers` became `list_documents`. 
  - Update `~/.jarvis/system_prompt.md` if you have one naming the old tools.
    The built in prompt was rewritten but an override is left alone.

- Webapp routes renamed (PR #1):
  - `/papers`, `/papers/meta`, `/papers/remove` became 
    `/documents*`, with `?kind=notes`. This will break any bookmarks if any.

- Removed skills as nothing was using it (PR #1). 

### MAJOR CHANGES

- All three providers now stream every request This is what makes a
  turn interruptible, since closing the connection is the only stop generating
  signal Ollama, Anthropic and OpenRouter offer (PR #3). 

- Knowledge base writes are now all or nothing (PR #3). 

- Paper digest are turned off by default in a switch to make Jarvis more general
  assistant. Feature can be turned back on through config file (PR #1).

- Sessions store conversation transcript in a model agnostic format (PR #1).
  This enable model switching within each session without losing context.
  When switching model within a session, the existing one is converted to
  the format accepted by the model before loaded. 

- Functions that do privacy checks now no longer checking whether provider is
  Anthropic. A new generic `is_cloud_provider` function replaces it. 
  An unknown provider is by default treated as a cloud provider (PR #1). 

- Introduced new config to support changes introduced for making Jarvis more of
  a general assistant (PR #1): `[drafts]`, `[openrouter]`, 
  `[models]`, `[chat]`, `[auth]`.

- Simplify changelog (PR #2).

### MINOR CHANGES

- PDF export from the editor no longer gives up after a minute. The ceiling on
  one LaTeX or pandoc run (`[drafts] compile_timeout_seconds`) is now 180
  seconds, since the first Markdown export on a machine spends minutes building
  a font cache before it even starts on the document (PR #1).

- The PDF button now shows how long an export has been running, and is disabled
  while it runs so a second click cannot stack another LaTeX run on the first.
  After 20 seconds the counter also says why a first export can take minutes
  (PR #1).

- Fix a test in `test_security.py` that had quietly stopped testing anything.
  Its fake provider did not accept the new `cancel` argument, so the webapp
  caught the `TypeError` in its crash handler and the test carried on passing
  (PR #3).

- Dropped the stale `docs/TODO.md` and `docs/ROADMAP.md` references from the
  repository tree in `DESIGN.md` (PR #3).

- Change the header bar on the UI so it is more readable (PR #1).
  The header now shows which model is currently used by the session.
  If `openrouter/auto` model is used, then the model name that is handling
  the prompt is showed (PR #1).

- Further UI polish (PR #1):
  - Dropped the vault path from the header.
  - The header's ⋮ button is replaced with a plain wrench icon.
  - Dropped the trailing "…" from every item in the ⋮ menu.
  - The editor toggle button now reads "Show editor" / "Hide editor"
    instead of "Editor" / "Hide editor".
  - Remove some captions from the prompt editor page.
  - Add icons to the UI header's next to cost and model name.
  - Add explicit "USD" next to cost.

### BUG FIXES

- Fix re-adding a document leaving you with nothing when the add process failed 
  part way. Re-adding deletes the old entry and indexes the new one, and
  those used to be two separate writes, so a failure or a stop in between wiped
  the old entry without putting the new one in. They now happen in 
  one write (PR #3).

- Fix errors being silently swallowed in thirteen places. Exceptions were
  caught but not logged, making troubleshooting almost impossible.
  Introduced a logger in `~/.jarvis/logs/jarvis.log` and printing errors out
  now to help troubleshooting (PR #1).

- Fix a stale database index reported as index corruption. This used to tell 
  the user to rebuild the whole database when in actual fact an `index-vault`
  to update it is all that is needed (PR #1). 

- Fix `kb reindex` could not run on a database store that is too corrupt to 
  read. New `kb reindex --from-storage` was introduced to fix corrupt HNSW index
  by using the chunked text (either full doc or LLM summary) stored in sqlite.
  That way, the rechunking or the expensive LLM summary doesn't need to be
  repeated when reindexing (PR #1).

- Fix `kb doctor` died without output instead of diagnosing when the index was badly
  corrupt (PR #1). 
