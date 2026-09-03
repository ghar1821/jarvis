# Jarvis



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
