# Jarvis

## PR #3 — Forceful stop

A reply can now be stopped while it is being generated, and a stop leaves
nothing behind — not in the session, and not in the knowledge base.

**Stopping a reply**

- Web UI: the Send button becomes a red **Stop** while a reply is in flight.
  Clicking it returns control immediately.
- The upstream connection is closed, so Ollama or Anthropic actually stops
  generating rather than being left to finish an answer nobody will read.
  Measured against both live services, the worker dies within 0.1 s of the stop.
- All three providers stream, OpenRouter included: its `_create()` folds the
  OpenAI-style chunk stream back into one result, concatenating content deltas
  and rejoining each tool call's arguments JSON from the fragments it arrives
  in (keyed by the delta `index`, so interleaved calls don't bleed into each
  other). Cost still lands — it rides the stream's final chunk, which is where
  OpenRouter puts it when usage accounting is on.
- A stopped turn leaves **no trace**: the question returns to the input box,
  nothing is written to the session history, and nothing is indexed as a chat
  exchange. The next message can be sent straight away.
- New `POST /chat/stop` endpoint. It cancels the turn and abandons it in the
  same breath — ending the SSE stream and rolling the turn back itself rather
  than waiting for the worker thread, because how long the worker takes to
  notice is not under our control. (Testing against a real cold Ollama model
  found this: the cancel check runs *between* streamed events, so a request
  still waiting on its **first** event can sit there for 20+ seconds, which
  under the earlier wait-for-the-worker design left the session busy and made
  the next message 409 — exactly what the stop exists to prevent.)
- `Ctrl-C` on `uv run webapp` now cancels every live turn on the way out.

**Both providers now stream**

- `OllamaProvider` and `AnthropicProvider` each make every request through a
  single `_request()` helper using the streaming APIs. This is what makes a
  turn interruptible: a blocking call offers no moment to bail out of, whereas
  a stream can be checked between events and closed part-way — and closing the
  connection is the only "stop generating" signal either service has.
- Replies are still delivered whole; nothing about the output changed.
- Consolidation, not just addition: `_request()` replaced four near-identical
  `try/except → LLMError` blocks per provider, and assembling Ollama's message
  from the stream removed the pydantic `model_dump` normalisation
  (`_message_to_dict`) that kept session history JSON-serialisable.

**All-or-nothing knowledge-base writes**

- Every `add_*` in `store.py` split into a `build_*` half that touches nothing
  and one shared `commit_documents()` that performs the only write. The public
  `add_texts` / `add_annotations` / `add_figures` / `add_paper` signatures and
  all their callers are unchanged.
- `add_document` now stages a whole paper — body, annotations, figure captions
  — and commits it in a single atomic write. An add that is stopped or that
  fails part-way leaves the knowledge base exactly as it was.
- This also fixes a latent bug: a re-ingest used to delete the old entry
  before indexing the new one's annotations, so a failure in between lost the
  old entry's irreplaceable annotation chunks. The delete and the add now share
  one commit.
- Cancellation is never checked inside the commit — an interrupted ChromaDB
  write is the corruption the staging exists to prevent.

**Compaction is visible**

- Compaction is a second LLM call made before the turn's own, and it used to
  show nothing at all. The webapp now shows a pulsing "Compacting conversation
  history..." indicator and the CLI prints the equivalent line. New
  `needs_compaction()` predicate so the indicator only appears when compaction
  is actually about to run.

**Under the hood**

- New `jarvis/core/cancel.py` (`CancelToken`) and `TurnCancelled`, which is
  deliberately not an `LLMError` — the `LLMError` handlers save a "⚠️ …" reply
  to the session, which is precisely the trace a stop must not leave.
- A cancelled turn needs no unwinding: each adapter already builds its turn in
  a provider-wire copy and publishes it to the neutral transcript with one
  `commit()` at the return points, so raising in between leaves `messages`
  exactly as it was found. Shared `rollback_turn()` drops the question from
  `messages`, `display`, and `turn_starts` together.
- Cancel checks are placed so that two things hold the moment a stop is
  requested: nothing further is sent, and the message list is never appended to
  again — an assistant `tool_use` block is never left without its matching
  `tool_result` bundle, which would 400 the following turn.
- `_session["running"]` in the webapp now holds a `RunningTurn` (live session,
  cancel token, event queue, thread, commit lock) rather than a bare `Session`.
  The lock is what keeps a stop and a just-landed reply from both writing: the
  reply stands if it committed first, the worker stands down if the stop did.
- 23 new tests: stream cancellation and connection close for both providers,
  the `/chat/stop` lifecycle end to end including both sides of that race,
  staged-write atomicity, a stopped ingest leaving the store untouched, and the
  neutral-transcript rollback. Verified live against Ollama and Anthropic as
  well: a cancelled turn's thread dies within 0.1 s, and `/chat/stop` returns
  in 0.02 s with the session already clean.
- Fixed a stub in `test_security.py` that had silently started exercising the
  crash handler instead of the path it was testing (and writing tracebacks to
  the real `~/.jarvis/logs/chat.log`) once `agentic_turn` gained its `cancel`
  parameter.
- Dropped the stale `docs/TODO.md` / `docs/ROADMAP.md` references from
  `DESIGN.md`'s repository tree; neither file exists.



## PR #1 — general assistant: OpenRouter, records, a draft sandbox, an editor

Turns jarvis from a research-paper tool into a general assistant that can also
produce documents, not just retrieve them. The paper digest stays, but as a
feature you switch on rather than the thing the daemon is built around.

**Run `uv sync` after merging** — new dependencies (`openai`, `pyyaml`,
`markdown-it-py`, `mdit-py-plugins`, `latex2mathml`).

### NEW FUNCTIONALITY

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
