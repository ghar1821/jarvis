# Changelog

## PR #3 — Forceful stop

A reply can now be stopped while it is being generated, and a stop leaves
nothing behind — not in the session, and not in the knowledge base.

**Stopping a reply**

- Web UI: the Send button becomes a red **Stop** while a reply is in flight.
  Clicking it returns control immediately.
- Terminal: `Ctrl-C` during a turn cancels the request and exits, instead of
  dumping a traceback out of `main()`.
- Either way the upstream connection is closed, so Ollama or Anthropic actually
  stops generating rather than being left to finish an answer nobody will read.
  Measured against both live services, the worker dies within 0.1 s of the stop.
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
- `agentic_turn` is now given a *copy* of the session's messages, committed
  back only once a reply arrives, so a cancelled turn has nothing to unwind.
  Shared `rollback_turn()` drops the question from `messages`, `display`, and
  `turn_starts` together.
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
  CLI's Ctrl-C rollback. Verified live against both providers as well: a
  cancelled turn's thread dies within 0.1 s, `/chat/stop` returns in 0.02 s
  with the session already clean, and a real SIGINT exits `vault-chat` in
  0.5 s writing no session file.
- Fixed a stub in `test_security.py` that had silently started exercising the
  crash handler instead of the path it was testing (and writing tracebacks to
  the real `~/.jarvis/logs/chat.log`) once `agentic_turn` gained its `cancel`
  parameter.
- Dropped the stale `docs/TODO.md` / `docs/ROADMAP.md` references from
  `DESIGN.md`'s repository tree; neither file exists.

## 2026-09-02 — where the project stands

This file was reset on 2026-09-02. The entry below is a snapshot of what jarvis
does today rather than a retelling of how it got here — the commit-by-commit
detail is in `git log`.


### Paper discovery

- Weekly digest pulls from arXiv (per-category limits) and bioRxiv (real
  categories server-side, free-text keywords for topics with no category), with
  the same paper arriving twice deduplicated by title.
- Papers are scored against a custom relevance prompt by the configured LLM and
  written as a tiered Markdown digest to `[digest] output_dir`.
- Indexing follows the tier: score ≥ 9 is stored full text (PDF downloaded and
  chunked), 8–8.9 reuses the digest's own summary text, and the digest file
  itself is indexed so papers below the threshold are still findable.
- `jarvis-sync` schedules the run (default Monday 05:00). A run missed while the
  Mac slept fires on wake; one missed while it was off is caught by an overdue
  re-check at start and every 6 hours. `kb sync-status` reports daemon health
  and recent job outcomes.
- A PDF dropped into `[sync] pdf_watch_dir` is indexed within `pdf_watch_minutes`
  (default 30). Re-saving a PDF with new annotations re-indexes it via byte-hash
  detection; the folder is an inbox, not a mirror, so removing a file never
  deletes its knowledge base entry.

### Knowledge base

- One local ChromaDB collection holds papers, Obsidian vault notes, indexed past
  conversations, and digest files. Embedding runs on the machine — no external
  calls.
- Papers are stored either as an LLM summary (~1000 words) or as fully chunked
  text for paragraph-level querying.
- Retrieval fuses BGE dense embeddings with BM25 keyword ranking by reciprocal
  rank fusion, over section-aware chunks, then re-ranks with a cross-encoder.
  `[rag] hybrid = false` takes the dense-only path.
- Highlights and typed notes in an annotated PDF become their own searchable
  chunks (`[HIGHLIGHT p.N]`, `[USER NOTE p.N]`). Freehand ink is stroke
  geometry, not text, and is not extracted.
- Figure captioning by a vision model produces `[FIGURE p.N]` chunks. Off by
  default since every figure costs a call — opt in per document with
  `kb add --figures` or by asking the agent to add it "with figures".
- Title, authors, and DOI are inferred for local PDFs; `kb set-meta` or the
  agent's `update_document_metadata` corrects whatever the inference got wrong.
- `kb doctor` diagnoses index health, `kb reindex` re-embeds every chunk without
  any LLM calls.

### Chat agent

- Terminal (`vault-chat`) and browser (`webapp`, localhost only) run the same
  agent and the same tools.
- Provider is Ollama locally or Anthropic Claude, switchable per session.
- Tools cover retrieval across papers, notes, and past chats; reading a stored
  document or vault file in full; adding papers by arXiv URL or local PDF;
  metadata corrections; vault indexing; stats; and requesting removals.
- DB-only is on by default. Switching it off lets the model fall back to its
  training knowledge, and it calls `use_own_knowledge` first so the fallback is
  visible rather than silent.
- Sessions persist to `~/.jarvis/sessions/` and can be resumed, renamed, pinned,
  deleted, and searched. The 50 most recent unpinned are kept; pinned ones are
  exempt. Long sessions compact themselves, keeping recent turns verbatim.
- The webapp runs sessions genuinely in parallel — each turn has its own thread
  and event stream, and a message is addressed to the session it was typed into
  even if you switch away mid-send.
- A papers manager in the header menu lists every indexed paper, narrows as you
  type, and allows in-place metadata edits and removals.
- Replies copy out as raw Markdown, so pasting into Obsidian keeps its
  formatting. Skills under `~/.jarvis/skills/<name>/SKILL.md` are advertised by
  name and description and loaded on demand.

### Privacy and safety

- Vault folders listed in `[chat] private_vault_dirs` are visible to the local
  model only. The check resolves symlinks, so a link in a public folder cannot
  reach into a private one.
- A cloud provider that touches private content stops the turn with a
  `PrivacyError` instead of working around it — private notes never reach a
  cloud model, even indirectly.
- Papers are always public. Only vault notes can be private, which is what makes
  the cloud summary path safe.
- A session that ever touches private content is flagged private permanently and
  cannot be resumed under Anthropic.
- Deletion always needs a human. The agent can only request one; confirmation
  happens out of band (terminal y/N or a webapp dialog), and only database
  entries are ever removed. There is no code path in jarvis that deletes a file
  on disk.

### Project infrastructure

- The Markdown in `docs/` is published to <https://ghar1821.github.io/jarvis/>
  on every push to `main`, with the README itself as the home page so the site
  and the repository cannot disagree. The build is `--strict`, so a broken link
  fails CI.
- CI runs the unit suite (`pytest -m "not integration"`) on pushes to `main` and
  on pull requests targeting it.
