# jarvis

A personal assistant that knows your own notes, documents and papers, runs on
your machine, and can write documents with you.

Named after Iron Man's J.A.R.V.I.S. — Just A Rather Very Intelligent System.

> This README is about **using** jarvis. For how it works inside — architecture,
> data flows, the privacy and safety guarantees and why they hold — see
> [`docs/DESIGN.md`](docs/DESIGN.md).

---

## Get started

### What you need first

- **[uv](https://github.com/astral-sh/uv)** and **Python ≥ 3.12**. On a machine
  with neither: `curl -LsSf https://astral.sh/uv/install.sh | sh`, then
  `uv python install 3.12`.
- **An OpenRouter key** — get one at
  [openrouter.ai/keys](https://openrouter.ai/keys). An Anthropic key works too.
- **Or, to run locally instead:** [Ollama](https://ollama.com) with a model
  that does tool calling *and* vision — `ollama pull qwen3-vl:30b`. Tool
  calling is required (jarvis works by calling tools); vision is only needed
  for figure captioning.
- Optional, only for the editor's PDF output: a LaTeX distribution (MacTeX,
  TeX Live) to compile `.tex`, and `pandoc` to export Markdown as PDF. Buttons
  for a missing tool are hidden rather than broken, so skip these at first.

### 1. Install

```bash
git clone <your-repo-url> jarvis && cd jarvis
uv sync
```

### 2. Write the config

**Jarvis does not create this file for you.** Without it you get defaults —
local Ollama, and a vault at `~/vault` that probably doesn't exist.

```bash
mkdir -p ~/.jarvis
$EDITOR ~/.jarvis/config.toml
```

A complete working OpenRouter config, copy-pasteable:

```toml
[chat]
provider = "openrouter"
openrouter_model = "anthropic/claude-sonnet-4.6"
vault_path = "~/Documents/obsidian"          # your notes; must exist

[auth]
openrouter_api_key = "sk-or-..."             # or the OPENROUTER_API_KEY env var

# Models offered in the picker. Jarvis ships no vendor list of its own —
# `uv run kb models --refresh` fills this in from OpenRouter's own index.
[models]
openrouter = ["anthropic/claude-sonnet-4.6", "openai/gpt-5"]
```

```bash
chmod 600 ~/.jarvis/config.toml    # it holds an API key; jarvis warns if it's readable
```

Both `provider` and `openrouter_model` are needed. Setting `provider` alone
fails with *"No model configured for provider 'openrouter'"*, and a missing key
fails at the first request with *"No OpenRouter credentials found"* — the
client is built lazily, so nothing complains until then.

**On OpenRouter specifically:** it is a broker — it routes your request to
somebody else's hardware. Jarvis sends strict settings by default
(`data_collection = "deny"`, no silent fallbacks). See [Choose a
model](#choose-a-model) to loosen them.

### 3. Index your notes

```bash
uv run kb index-vault
```

First run downloads the embedding model (`BAAI/bge-small-en-v1.5`, ~130 MB from
HuggingFace) and caches it. That is the only model jarvis downloads, it runs
locally, and it is why indexing works with no API calls. If your vault path is
wrong you get `Error: vault path does not exist: ...` immediately, before
anything is downloaded.

No notes yet? Skip this — the editor and chat work without a vault.

### 4. Check it worked

```bash
uv run kb models     # which providers are configured, and which lack a key
uv run kb stats      # what got indexed
uv run kb doctor     # embedding model and index health
```

`kb models` makes no network call — it reads your config, so it is the quickest
way to confirm the file is being picked up:

```
ollama:qwen3-vl:30b  [local]
anthropic:claude-sonnet-4-6  [cloud]  (no API key)
openrouter:anthropic/claude-sonnet-4.6  [cloud]
```

An entry marked `(no API key)` is configured but unusable. If `openrouter:` is
missing entirely, `openrouter_model` isn't set.

### 5. Run it

```bash
uv run webapp        # browser at http://127.0.0.1:8080 (localhost only)
```

`webapp --reload` restarts the server when you edit Python. Note that changes
to `static/app.js` or `index.html` need a **browser** hard-reload (⇧⌘R) rather
than a server restart — and Python changes need the opposite.

### Working on jarvis

```bash
uv sync --group dev                  # once, to get pytest
uv run pytest -m "not integration"   # the suite that must pass before any change
uv run pytest -m integration         # needs live services (API key, running Ollama)
uv run pytest tests/test_drafts.py   # one file
```

Everything jarvis owns lives in `~/.jarvis/`: `config.toml`, the index (`rag/`),
sessions, drafts and logs. Deleting that directory resets you to a fresh
install without touching your vault. Architecture, data flows and the privacy
guarantees are in [`docs/DESIGN.md`](docs/DESIGN.md); what is covered by tests
and why is in [`docs/TESTING.md`](docs/TESTING.md).

---

## Ask it things

Just talk to it. It searches your notes, papers and past conversations before
answering, and shows you every step it takes.

```
what did I conclude about batch effects in the cytometry project?
which papers do I have on sparse autoencoders?
what did we discuss about this last week?
add https://arxiv.org/abs/2406.04093
```

By default it answers **only** from what it found in your knowledge base. Flip
the **DB only** toggle off to let it fall back on the model's own knowledge —
it says so on screen when it does.

Conversations are saved automatically. Resume, rename, pin or delete them from
the **Chats** section of the sidebar.

---

## Write documents with it

Ask for a document and you get a real file you can open, not a wall of chat
text:

```
tailor my CV to this job ad
draft the methods section from my notes on the pipeline
```

Those land in **drafts** — a scratch folder the assistant can write to freely.
Your vault is read-only to it, permanently.

```
~/.jarvis/drafts/          your vault
assistant writes here      it can never write here, with or without your say-so
you edit here              you copy files across yourself, in Finder
```

**A draft is a folder, not a file.** That is what makes LaTeX work: `main.tex`,
its chapters and its `.bib` live in one draft and compile together, because
compiling copies the whole folder. Markdown is usually one file, so a
single-file draft just shows as one row; a multi-file one lists its parts
underneath with a count beside the name. Right-click a document to add a file
to it.

The sidebar has two sections, **Chats** and **Documents**, each with a `+` to
start a new one. Click a document, or press **Editor** in the header, and the
editor opens above the chat — so you can talk about the document while looking
at it. Inside the editor: source on the left, preview on the right, with
**Recompile** to re-render and a layout control for split / source only /
output only. Markdown previews as you type; LaTeX compiles to a PDF (with the
log underneath when it fails); both export to PDF.

- **Open several at once.** Each file gets a tab. The control on a tab is a
  filled dot while it has unsaved changes and an × once it does not, so you can
  see what still needs saving without switching to it. Clicking it saves first
  if it needs saving, then closes — closing a tab is never how work is lost.
- **⌘S saves** the tab you are in. Previewing, compiling or exporting saves
  first, so what you act on is always what you see.
- Every previous version is kept — **History** restores one, and restoring is
  itself undoable.
- When the assistant proposes a change you get **a diff with a checkbox per
  hunk**. Accept some, reject others; only what you tick is written. A ✎ on a
  tab means a suggestion is still waiting there; reopening the file brings it
  back. ⋮ → **Discard pending suggestions…** clears the lot, and they go on
  their own when you restart the app.
- Drafts **expire** after 30 days untouched. `uv run kb drafts` shows how long
  each has left, and **Keep** exempts one. `[drafts] retention_days = 0` turns the
  sweep off.

### Getting a document out

Right-click it → **Show in Finder**, then copy it wherever you want.

That is deliberately all there is. Jarvis has no route into your vault at all —
there is no password to set, and nothing to get past, because moving a file is
something you do in your own file manager.

**Preview and export need:** nothing for Markdown preview; a LaTeX
distribution (MacTeX, TeX Live) to compile `.tex`; `pandoc` as well to export
Markdown as PDF. Buttons for a missing tool are hidden rather than broken.

### Skills — teaching it how you like things done

A **skill** is a set of instructions you write once, in plain English, that the
assistant follows whenever a matching task comes up. It saves you re-explaining
your own process every time.

Say your methods sections always need to cite the pipeline version and name
which dataset each figure came from. Write that down once as a skill, and you
stop having to say it.

A skill is just a folder with a `SKILL.md` in it:

```
~/.jarvis/skills/
└── methods-section/
    ├── SKILL.md          # the instructions
    └── template.tex      # any file the instructions refer to
```

```markdown
---
description: Draft a methods section the way I write them.
---

# Methods sections

1. Read the project note for the work being described.
2. Cite the pipeline version from its frontmatter — never say "the latest".
3. Name the dataset behind every figure.
4. Past tense, no first person, no hedging about future work.
```

The folder name is the skill's name, and the `description` is all the assistant
sees up front — it loads the full instructions only when a task actually
matches, so having twenty skills costs nothing until one is used. Delete the
folder to switch it off.

Skills are your own files. They're never indexed and never sent anywhere the
rest of a conversation wouldn't go.

Two worked examples are in [`examples/skills/`](examples/skills/) — copy a
folder into `~/.jarvis/skills/` to try them:

- **`draft-from-notes`** — write a paper or report section from your notes and
  indexed papers, showing you the gaps rather than papering over them.
- **`tailor-document`** — reshape an existing document for a specific target,
  using evidence from your own records.

---

## Keep records, not just notes

Any note can carry YAML frontmatter, and jarvis turns it into something you can
filter on. Useful for anything you have a lot of and need to track the state of
— manuscripts, grants, experiments, meetings.

```markdown
---
type: manuscript
entity: Nature Methods        # where it's going
status: under_review          # or: drafting, submitted, revising, accepted
date: 2026-04-18              # submitted on
tags: [cytometry, benchmarking]
coauthors: Ada Lovelace       # jarvis has never seen this key; still filterable
---

# Benchmarking batch correction for high-dimensional cytometry

Submitted 18 April. Reviewer 2 wants the ablation on the spillover step.
```

Then ask questions by record rather than by wording:

```
which manuscripts are under review, and what are reviewers asking for?
what have I got in drafting for Nature Methods?
show me everything I submitted this year
```

The vocabulary is yours: `type`, `status`, `entity`, `date` and `tags` get
first-class filters, and every other key is kept too — jarvis has no idea what
a "manuscript" is, it just indexes what you wrote. Run `uv run kb schema` to
see which keys and values actually exist; that is how you catch a typo like
`stauts:` that would otherwise silently never match.

The same shape works for anything else you want to keep track of — job
applications with outcomes, experiments with conditions, reading with verdicts.

You edit records in Obsidian as usual. Jarvis indexes them; it doesn't own them.

---

## Add papers and PDFs

```bash
uv run kb add https://arxiv.org/abs/2406.04093       # a summary (fast, default)
uv run kb add https://arxiv.org/abs/2406.04093 --full-text
uv run kb add paper.pdf                               # title/authors/DOI inferred
uv run kb add paper.pdf --authors "Ada Lovelace"      # or set them yourself
```

Or just ask in chat: *"add ~/Downloads/paper.pdf, full text"*.

**Drop PDFs in a folder** and they index themselves. Set
`[sync] pdf_watch_dir = "~/Documents/papers/inbox"` and the background daemon
sweeps it every half hour.

**Your highlights come along.** Highlights and typed notes made in macOS
Preview or Foxit Reader become searchable, so *"what did I highlight in that
paper?"* works. Re-save a PDF with new annotations and it re-indexes itself.
(Freehand pen scribbles aren't text, so they can't be extracted.)

**Figures** can be captioned by a vision model and made searchable — off by
default since each figure costs a call. Add `--figures`, or ask for a paper
"with figures".

---

## Choose a model

Switch mid-conversation without losing the thread: **⋮ → Switch model…**. It
applies from your next message, per conversation — two sessions can run
different models at once. The header shows the active model and what the
session has cost.

**You do not have to configure a catalogue.** Whatever you set as
`openrouter_model` (or `ollama_model`) already appears in the picker. `[models]`
just adds more to choose from:

```toml
[models]
openrouter = ["anthropic/claude-sonnet-4.6", "openai/gpt-5", "openrouter/auto"]
```

```bash
uv run kb models              # what's on offer right now (no network)
uv run kb models --refresh    # pull OpenRouter's full catalogue into [models]
```

`--refresh` is optional and only for browsing — it writes OpenRouter's model
ids into your config so you can pick from a list instead of typing one. It is
the only place jarvis fetches a model list; the picker itself never touches the
network. Consider writing two or three by hand instead: a picker with three
hundred rows is worse than one with the models you actually use.

**Editing the config needs a webapp restart** before the picker sees the change
— switching between models already listed does not.

The list is a convenience, not a restriction — the picker also has a box to
type any model id OpenRouter accepts, listed or not, which is applied to the
current conversation without touching your config.

**Automatic routing.** OpenRouter's auto router is just a model id, so set
`openrouter_model = "openrouter/auto"` (or add it to `[models]`) and it picks a
model per request. Jarvis sends `allow_fallbacks = false` by default, which is
untested against the auto router — loosen it under `[openrouter]` if requests
start failing.

**Cost** is shown only for OpenRouter, which reports what each request actually
cost, shown in the header. A local model costs
nothing, and jarvis won't invent a figure for anything else.

**If you use OpenRouter**, know that it is a broker — it routes your request to
somebody else's hardware. Jarvis sends the strict settings by default
(`data_collection = "deny"`, no silent fallbacks); you can loosen them under
`[openrouter]` if you want to.

---

## Keep things private

Notes in the folders listed under `private_vault_dirs` (default: `private/`)
are only ever visible to a local model.

```
vault/
├── private/    ← local model only, never sent anywhere
└── research/   ← any model
```

A conversation that touches private content is marked private for good, and
can't be switched to a cloud model afterwards. Papers are always public, so put
anything sensitive in a note.

Two things jarvis will never do: **delete a file** (removing a document removes
its database entry only — the file stays), and **write to your vault** (there
is no code path that can — you copy drafts across yourself, in Finder).

---

## Run it in the background

```bash
uv run jarvis-sync
```

Keeps everything current: re-indexes your vault, sweeps the PDF inbox, clears
expired drafts. It stays in the foreground, so run it under `tmux`/`screen` if
you want it to survive closing the terminal. Check on it any time:

```bash
uv run kb sync-status
```

---

## Weekly paper digest (optional, off)

If you want automated arXiv/bioRxiv discovery, switch it on:

```toml
[digest]
enabled = true
arxiv_categories = [["cs.LG", 150], ["cs.AI", 80]]
```

It fetches weekly, scores everything against your relevance prompt, writes a
tiered Markdown digest, and indexes the best papers. Catch-up is automatic if
your machine was asleep. Run one by hand any time with
`uv run run-digest --force`.

---

## Command reference

```bash
# Everyday
uv run webapp                  # the UI
uv run jarvis-sync             # background sync

# Knowledge base
uv run kb index-vault          # re-index your notes (--force for a clean rebuild)
uv run kb add <url|file.pdf>   # add a paper or PDF
uv run kb list                 # indexed papers (--notes for records)
uv run kb schema               # which metadata keys/values you actually have
uv run kb stats                # counts
uv run kb remove <source>      # remove a database entry (never a file)
uv run kb set-meta <source> --authors "..."
uv run kb doctor               # diagnose a sick index
uv run kb reindex              # re-embed everything (no LLM calls)

# Drafts
uv run kb drafts               # list, with expiry
uv run kb drafts --prune --dry-run

# Models
uv run kb models [--refresh]
```

Everything lives in `~/.jarvis/`: `config.toml`, your index, sessions, drafts
and logs. Keep the config private (`chmod 600 ~/.jarvis/config.toml`) — it can
hold an API key, and jarvis warns you at startup if it's readable by others.

The full configuration reference is in
[`docs/DESIGN.md`](docs/DESIGN.md#configuration--jarviscoreconfigpy).

---

## If something goes wrong

**"No model configured for provider 'openrouter'".** You set `provider` but
not `openrouter_model`. Both are needed — see [Get started](#get-started).

**"No OpenRouter credentials found".** No key in `[auth] openrouter_api_key`
and none in `OPENROUTER_API_KEY`. The client is built lazily, so this appears
at the first message rather than at startup. `uv run kb models` shows which
providers have a key without sending a request.

**Nothing you configured seems to apply.** Jarvis reads
`~/.jarvis/config.toml` and nothing else — not a file in the repo, and not the
working directory. `uv run kb models` reflects what was actually loaded.

**Searches fail with a database error.** Run `uv run kb doctor`. It will tell
you whether to `uv run kb reindex` or just restart the process.

**"Embedding model mismatch".** You changed `embed_model`. Run
`uv run kb reindex` once — it re-embeds stored text, makes no LLM calls, and
downloads nothing.

**The webapp behaves oddly after an upgrade.** Hard-reload the tab
(Cmd+Shift+R); an old tab can send request shapes the new server rejects.

**Upgrading from an older jarvis.** Run `uv run kb reindex` once. If your config
still has `provider = "llamacpp"`, switch it to `"ollama"`; if it has
`rag_dir = "~/.seshat/rag"`, change it to `~/.jarvis/rag`; and move
`anthropic_model` from `[digest]` to `[chat]`. Jarvis warns rather than
rewriting your file.

---

## Documentation

- [`docs/DESIGN.md`](docs/DESIGN.md) — architecture, data flows, and the
  security and privacy guarantees
- [`docs/CHANGELOG.md`](docs/CHANGELOG.md) — what changed and why
- [`docs/TESTING.md`](docs/TESTING.md) — what's covered and how to run the tests
