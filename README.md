# jarvis

[![Tests](https://github.com/ghar1821/jarvis/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/ghar1821/jarvis/actions/workflows/tests.yml)

A personal assistant that knows your own notes, documents and papers, runs on
your machine, and can write documents with you.

Named after Iron Man's J.A.R.V.I.S. — Just A Rather Very Intelligent System.

Full documentation: **<https://ghar1821.github.io/jarvis/>**

---

## Get started

### What you need first

- **[uv](https://github.com/astral-sh/uv)** and **Python ≥ 3.12**. Neither
  installed yet? `curl -LsSf https://astral.sh/uv/install.sh | sh`, then
  `uv python install 3.12`.
- **An OpenRouter key.** Get one at
  [openrouter.ai/keys](https://openrouter.ai/keys). An Anthropic key works too.
- **Or run locally instead:** install [Ollama](https://ollama.com) and pull a
  model that does tool calling *and* vision — `ollama pull qwen3-vl:30b`. Tool
  calling is required, since jarvis works by calling tools; vision is only
  needed for figure captioning.
- Optional, and only for the editor's PDF output: a LaTeX distribution (MacTeX,
  TeX Live) to compile `.tex`, and `pandoc` to export Markdown as PDF. Skip
  these for now if you want — a button for a missing tool just stays hidden.

### 1. Install

```bash
git clone <repo-url> jarvis && cd jarvis
uv sync
```

### 2. Setup the config

Jarvis won't create this file for you. Skip it and you get the defaults —
local Ollama, and a vault at `~/vault` that probably doesn't exist.

```bash
mkdir -p ~/.jarvis
$EDITOR ~/.jarvis/config.toml
```

An example of working OpenRouter config:

```toml
[chat]
provider = "openrouter"
openrouter_model = "anthropic/claude-sonnet-4.6"
vault_path = "~/Documents/obsidian"          # your notes; must exist

[auth]
openrouter_api_key = "sk-or-..."             # or the OPENROUTER_API_KEY env var

# Models offered in the picker. Optional — whatever you set as
# openrouter_model already appears there, and the picker has a box for
# typing any model id that is not listed.
[models]
openrouter = ["anthropic/claude-sonnet-4.6", "openai/gpt-5"]
```

After writing the config file, remember to update its permission so it is only readable by you the user.
Why? It may hold an API key if it is specified there instead of your env var. 

```bash
chmod 600 ~/.jarvis/config.toml    # it holds an API key; jarvis warns if it's readable
```

You need both `provider` and `openrouter_model`. 
If you set `provider` only, Jarvis will complain with *"No model configured for provider 'openrouter'"*.
If you leave out the API key, it won't complain until you send the first message.
Then it will say *"No OpenRouter credentials found"*.

OpenRouter is a broker: it routes your request to a server somewhere in the world hosting the LLM.
Jarvis sends strict settings by default (`data_collection = "deny"`, no
silent fallbacks) — see [Choose a model](#choose-a-model) if you want to change them.

### 3. Index your notes

Jarvis need to setup the RAG database.

```bash
uv run kb index-vault
```

It will first download `BAAI/bge-small-en-v1.5` embedding model and cache it. 
Make sure the vault path you specified in the config exists or else you will get `Error: vault path does not
exist: ...` error before anything downloads at all.

If you don't have any notes in the vault yet, you don't need to run the indexing.
Feel free to skip it for now and move on to step 4.

### 4. Check it worked

```bash
uv run kb models     # which providers are configured, and which lack a key
uv run kb stats      # what got indexed
uv run kb doctor     # embedding model and index health
```

`kb models` reads your config rather than the network, so it's the fastest way
to confirm the file is actually being picked up:

```
ollama:qwen3-vl:30b  [local]
anthropic:claude-sonnet-4-6  [cloud]  (no API key)
openrouter:anthropic/claude-sonnet-4.6  [cloud]
```

`(no API key)` means configured but unusable. No `openrouter:` line at all
means `openrouter_model` was never set.

The last two commands check whether your notes have been indexed.
If you did not run step 3, then don't bother running them.

### 5. Run it

Run the webapp interface and start chatting.

```bash
uv run webapp        # browser at http://127.0.0.1:8080 (localhost only)
```

---

## Ask it things

Just talk to it. It searches your notes, papers and past conversations before
answering, and shows you every step along the way.

```
what did I conclude about batch effects in the cytometry project?
which papers do I have on sparse autoencoders?
what did we discuss about this last week?
add https://arxiv.org/abs/2406.04093
```

By default it only answers from what it finds in your knowledge base. Flip
the **DB only** toggle off and it can fall back on the model's own training
knowledge instead — and it says so on screen when it does.

Conversations save themselves. Resume, rename, pin or delete one from the
**Chats** section of the sidebar.

---

## Write documents with it

Ask for a document and you get a real file you can open — not a wall of chat
text.

```
tailor my CV to this job ad
draft the methods section from my notes on the pipeline
```

These land in **drafts**, a scratch folder the assistant can write to freely.
Your vault stays read-only to it, permanently.

```
~/.jarvis/drafts/          your vault
assistant writes here      it can never write here, with or without your say-so
you edit here              you copy files across yourself, in Finder
```

A draft is a folder, not a file, and that's what makes LaTeX work: `main.tex`,
its chapters and its `.bib` live in one draft and compile together, because
compiling copies the whole folder at once. Markdown is usually one file, so a
single-file draft just shows as one row; a multi-file one lists its parts
underneath, with a count beside the name. Right-click a document to add
another file to it.

The sidebar has two sections — **Chats** and **Documents** — each with a `+`
to start a new one. Click a document, or press **Show editor** in the header,
and the editor opens above the chat, so you can talk about the document while
looking at it. Source sits on the left, preview on the right, **Recompile**
re-renders, and a layout control switches between split / source only /
output only. Markdown renders to HTML; LaTeX compiles to a PDF (with the
log underneath when it fails); both export to PDF.

- **Open several at once.** Each file gets its own tab. A filled dot on the
  tab means unsaved changes; an × means none. Click it and it saves first if
  it needs to, then closes — a tab can never cost you work just by closing it.
- **⌘S saves** the tab you're in, and re-renders the preview with it, so what
  you are looking at is never text you have since changed. Previewing,
  compiling or exporting also saves first.
- **Scrolling the source scrolls the preview** to the matching place, keeping
  the two lined up as you move through a long document. Markdown only — a
  compiled LaTeX PDF is displayed by the browser itself, which doesn't let the
  page scroll it.
- Every earlier version is kept. **History** restores one, and restoring is
  itself undoable.
- When the assistant proposes a change, you get a diff with a checkbox per
  hunk — accept some, reject others, and only what you tick gets written. A ✎
  on a tab means a suggestion is still waiting there; open the file and it
  comes back. ⋮ → **Discard pending suggestions** clears the lot at once, and
  they clear themselves whenever you restart the app.
- Drafts expire after 30 days untouched. `uv run kb drafts` shows how long
  each has left; **Keep** exempts one for good. Set
  `[drafts] retention_days = 0` to turn the sweep off entirely.

### Getting a document out

Right-click it, choose **Show in Finder**, then copy it wherever you like.

That's the whole mechanism, on purpose. Jarvis has no route into your vault at
all — no password to set, nothing to get past — because moving a file is
something you do yourself, in your own file manager.

The first PDF you export on a new machine can be slow: Markdown export runs
through xelatex and fontspec, and fontspec builds a system font cache the
first time it runs — sometimes past the compile timeout. It's a one-time cost.
Get it over with before you need it:

```bash
printf '\\documentclass{article}\\usepackage{fontspec}\\begin{document}x\\end{document}' > /tmp/warm.tex
xelatex -output-directory=/tmp /tmp/warm.tex
```

(`.tex` documents compile through latexmk/pdflatex instead and skip all of
this, which is why compiling can work fine while exporting hangs.)

Preview needs nothing extra. Compiling `.tex` needs a LaTeX distribution
(MacTeX, TeX Live). Exporting Markdown as PDF needs `pandoc` on top of that.
Buttons for a missing tool stay hidden rather than sitting there broken.

### Editing the prompts

Everything jarvis asks a model to do comes from three prompts, and all three
are yours to change: ⋮ → **Edit prompts**

| Prompt | What it controls |
|---|---|
| Assistant instructions | How the chat agent behaves — when it searches, how it cites |
| Paper summary | How a paper is summarised when you add one |
| Digest scoring | Which papers the weekly digest thinks matter |

The repo ships a generic default for each. Your editable copy lives in
`~/.jarvis/prompts/` and is created the first time jarvis runs, so the shipped
default is never overwritten and **Revert to default** always has something
clean to go back to. Edits apply to your next message — no restart.

The scoring one is worth your time if you turn the digest on. It ships with a
placeholder research-context section, and what you write there is what decides
which papers score well. Be specific.

### Seeing what jarvis loaded

The webapp prints its configuration on startup, and ⋮ → **Show config**
shows the same thing in the browser. Both report the values actually in
effect — after the file *and* any environment variables — which is usually
what you want to know when a setting seems to be ignored. API keys show as
`set` or `not set`, never their value.

---

## Keep records, not just notes

Any note can carry YAML frontmatter, and jarvis turns that into something you
can filter on — useful for anything you accumulate a lot of and need to track
the state of: manuscripts, grants, experiments, meetings.

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

Then ask questions by record instead of by wording:

```
which manuscripts are under review, and what are reviewers asking for?
what have I got in drafting for Nature Methods?
show me everything I submitted this year
```

The vocabulary is yours. `type`, `status`, `entity`, `date` and `tags` get
first-class filters, and every other key still gets kept — jarvis has no idea
what a "manuscript" actually is, it just indexes what you wrote. Run
`uv run kb schema` to see which keys and values actually exist. That's how
you catch a typo like `stauts:` that would otherwise silently never match
anything.

The same shape works for whatever else you want to track — job applications
with outcomes, experiments with conditions, reading with verdicts.

You still edit records in Obsidian, same as always. Jarvis indexes them; it
doesn't own them.

---

## Add papers and PDFs

```bash
uv run kb add https://arxiv.org/abs/2406.04093       # a summary (fast, default)
uv run kb add https://arxiv.org/abs/2406.04093 --full-text
uv run kb add paper.pdf                               # title/authors/DOI inferred
uv run kb add paper.pdf --authors "Ada Lovelace"      # or set them yourself
```

Or just ask in chat: *"add ~/Downloads/paper.pdf, full text"*.

Drop PDFs in a folder and they index themselves. Set
`[sync] pdf_watch_dir = "~/Documents/papers/inbox"` and the background daemon
sweeps it every half hour.

Your highlights come along too. Highlights and typed notes made in macOS
Preview or Foxit Reader become searchable, so *"what did I highlight in that
paper?"* actually works. Re-save a PDF with new annotations and it re-indexes
itself. (Freehand pen scribbles aren't text, so those can't be pulled out.)

Figures can be captioned by a vision model and made searchable — off by
default, since every figure costs a call. Add `--figures`, or just ask for a
paper "with figures."

---

## Choose a model

Switch mid-conversation without losing the thread: **⋮ → Switch model**. It
applies from your next message, per conversation, so two sessions can run
different models at once. The header shows the active model and what the
session has cost so far.

You don't have to configure a catalogue. Whatever you set as
`openrouter_model` (or `ollama_model`) already shows up in the picker.
`[models]` just adds more to choose from:

```toml
[models]
openrouter = ["anthropic/claude-sonnet-4.6", "openai/gpt-5", "openrouter/auto"]
```

```bash
uv run kb models              # what's on offer right now (no network)
```

That makes no network call — it reads your config, so it doubles as the
quickest check that the file is being picked up and which providers have a
key.

The list is a convenience, not a restriction: the picker also has a box for
typing any model id OpenRouter accepts, listed or not, applied to the current
conversation without touching your config. Keep `[models]` to the handful you
actually switch between.

OpenRouter's auto router is just a model id, so automatic routing needs
nothing built for it: set `openrouter_model = "openrouter/auto"` (or add it
to `[models]`) and it picks a model per request. The header then shows both
halves — `openrouter/auto → claude-sonnet-4.6` — so you can see what actually
answered, and hovering the cost shows what each model it picked has cost you.
Jarvis sends `allow_fallbacks = false` by default, which hasn't been tested
against the auto router — loosen it under `[openrouter]` if requests start
failing.

Cost is shown only for OpenRouter, which reports what each request actually
cost — you'll see it in the header, and hovering it breaks the total down by
model. A local model costs nothing, and jarvis won't invent a figure for
anything else.

If you use OpenRouter, know that it's a broker: your request routes to
somebody else's hardware. Jarvis sends strict settings by default
(`data_collection = "deny"`, no silent fallbacks); loosen them under
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

Once a conversation touches private content it's marked private for good, and
can't switch to a cloud model afterwards. Papers are always public, so keep
anything sensitive in a note instead.

Two things jarvis will never do: delete a file (removing a document only
removes its database entry — the file stays put), and write to your vault
(there's no code path that could — you copy drafts across yourself, in
Finder).

---

## Run it in the background

```bash
uv run jarvis-sync
```

Entirely optional. It's a scheduler and nothing more, running the same vault
sync, PDF sweep and draft cleanup you can run by hand — just on a timer. Skip
it and nothing breaks; you just index manually after editing your notes. It
stays in the foreground, so run it under `tmux` or `screen` if you want it to
survive closing the terminal. Check on it any time:

```bash
uv run kb sync-status
```

### Updating the index by hand

```bash
uv run kb index-vault     # pick up notes you added, edited or deleted
```

That's the one you want — it's exactly what the daemon runs on its own timer,
incremental, so unchanged notes cost nothing.

`kb reindex` is a different tool and usually isn't what you want. It re-embeds
text already in the database and never looks at your vault, so it won't
notice a single note you changed. It exists for two situations only: you
changed `embed_model`, or `kb doctor` told you the index is damaged.

| You did this | Run this |
|---|---|
| Edited, added or deleted notes | `kb index-vault` |
| Want a clean rebuild of the note index | `kb index-vault --force` |
| Changed `embed_model` in the config | `kb reindex` |
| `kb doctor` reported a corrupt index | `kb reindex` (or `--from-storage`) |

---

## Weekly paper digest (optional, off)

Want automated arXiv/bioRxiv discovery? Switch it on:

```toml
[digest]
enabled = true
arxiv_categories = [["cs.LG", 150], ["cs.AI", 80]]
```

It fetches weekly, scores everything against your relevance prompt, writes a
tiered Markdown digest, and indexes the best papers. If your machine was
asleep, catch-up runs automatically. Run one by hand any time with
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
uv run kb models
```

Everything lives under `~/.jarvis/`: `config.toml`, your index, sessions,
drafts, logs. Keep the config private (`chmod 600 ~/.jarvis/config.toml`) — it
can hold an API key, and jarvis warns you at startup if others can read it.

The full configuration reference is in
[`docs/DESIGN.md`](docs/DESIGN.md#configuration--jarviscoreconfigpy).

---

## If something goes wrong

**"No model configured for provider 'openrouter'".** You set `provider` but
not `openrouter_model`. You need both — see [Get started](#get-started).

**"No OpenRouter credentials found".** No key in `[auth] openrouter_api_key`
and none in `OPENROUTER_API_KEY` either. The client is built lazily, so this
only shows up at your first message, not at startup. `uv run kb models` shows
which providers have a key without sending anything.

**Nothing you configured seems to apply.** Jarvis only reads
`~/.jarvis/config.toml` — never a file in the repo, never the working
directory. `uv run kb models` shows what was actually loaded.

**Searches fail with a database error.** Run `uv run kb doctor`. It tells you
whether to run `uv run kb reindex` or just restart the process.

**"Embedding model mismatch".** You changed `embed_model`. Run
`uv run kb reindex` once — it re-embeds stored text, makes no LLM calls, and
downloads nothing.

**The webapp behaves oddly after an upgrade.** Hard-reload the tab
(Cmd+Shift+R). An old tab can send request shapes the new server doesn't
recognize.

**Upgrading from an older jarvis.** Run `uv run kb reindex` once. If your
config still has `provider = "llamacpp"`, switch it to `"ollama"`; if it has
`rag_dir = "~/.seshat/rag"`, change it to `~/.jarvis/rag`; and move
`anthropic_model` from `[digest]` to `[chat]`. Jarvis warns about these
instead of rewriting your file for you.

---

### Working on jarvis

```bash
uv run jarvis-sync
```

It logs to `~/.jarvis/logs/sync.log` (and to the terminal) by default, and stays in the foreground — `Ctrl-C` to stop it. There is no built-in service/daemon management (no launchd, no auto-restart-on-crash): if you want it to survive closing the terminal or to restart automatically, run it under a terminal multiplexer (`tmux`/`screen`) or a process manager of your choice. The daemon does not start Ollama for you either; run Ollama as a login-item app or `ollama serve`.

---

## Documentation site

The Markdown in `docs/` is published as a site at
<https://ghar1821.github.io/jarvis/>, rebuilt by `.github/workflows/docs.yml`
on every push to `main`. This README is its home page — the build reads the
real file rather than a copy, so the two cannot drift apart.

```bash
uv run --group docs mkdocs serve   # preview at http://127.0.0.1:8000
uv run --group docs mkdocs build --strict   # what CI runs
```

Blog posts go in `docs/blog/posts/` as Markdown files with a `date:` in their
front matter; `draft: true` keeps one visible in `mkdocs serve` but out of the
published site. `docs/blog/posts/hello.md` is a placeholder that spells out the
format.

---

## Requirements

- [uv](https://github.com/astral-sh/uv)
- Python ≥ 3.12
- [Ollama](https://ollama.com) with a tool-calling + vision model pulled, e.g. `qwen3-vl:30b` (for local inference)
- Anthropic API key (for cloud inference only; set via env var or `~/.jarvis/config.toml`)
- `fastapi` and `uvicorn` (included in `uv sync`; required for the web UI only)
