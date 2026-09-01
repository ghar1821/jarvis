# Changelog

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
