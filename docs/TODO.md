Resolved 2026-07-13 (branch todo-batch-2) — see docs/CHANGELOG.md for details:

1) Fixed: a brand-new session's first message is always accepted now (no global busy
   409), and the early save persists it — the session can always be revisited.
2) Fixed: a new chat appears in the sidebar as soon as the first message is sent
   (the session file is written before the LLM call).
3) Implemented parallel sessions: turns in different sessions run concurrently
   (per-session serialization — only a second message to the SAME session waits).
   No queueing needed.
4) Fixed: /chat is session-addressed now (the request carries the session id), so a
   message can never land in a different session than the one it was typed in.
5) Fixed: jarvis sync logs each job's next run time at startup and after every run,
   and APScheduler's own registration noise is silenced.
6) Removed: the unverified-metadata (meta_inferred) flag and all its surfacing
   (kb stats reminder, kb list --unverified, webapp banner). kb set-meta and the
   chat metadata tool remain as plain editors.
7) Done: jarvis sync logs the stored title/authors/doi per ingested paper, and the
   webapp has a Papers… manager (top-right menu): searchable list, inline
   title/authors/doi editing, and DB-only removal behind a two-step confirmation
   ("Database entry only — files on disk are never touched by jarvis"). No chmod
   hardening needed: jarvis has no file-deletion code at all (regression-tested),
   and chmod would also block your own edits since the webapp runs as your user.
