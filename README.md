# Jarvis

[![Tests](https://github.com/ghar1821/jarvis/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/ghar1821/jarvis/actions/workflows/tests.yml)

A personal assistant that knows your notes, documents and papers, runs on
your machine, and can write documents with you.

Named after Iron Man's J.A.R.V.I.S. (Just A Rather Very Intelligent System).

Full documentation: **<https://ghar1821.github.io/jarvis/>**

---

## Setup

### Requirements

- [uv](https://github.com/astral-sh/uv) and Python 3.12 or newer. If you have
  neither: `curl -LsSf https://astral.sh/uv/install.sh | sh`, then
  `uv python install 3.12`.
- An [OpenRouter](https://openrouter.ai/keys) key, or an Anthropic key.
- To run locally instead, install [Ollama](https://ollama.com) and pull a model
  that supports tool calling and vision, e.g. `ollama pull qwen3-vl:30b`. Tool
  calling is required; vision is only used for figure captioning.
- Optional, for the editor's PDF output: a LaTeX distribution (MacTeX, TeX Live)
  to compile `.tex`, and `pandoc` to export Markdown as PDF. You can add these
  later; a button for a missing tool stays hidden rather than sitting there
  broken.

### 1. Install

```bash
git clone <repo-url> jarvis && cd jarvis
uv sync
```

### 2. Write the config

Jarvis reads `~/.jarvis/config.toml` and nothing else. It won't create the file
for you, and without it you get the defaults: local Ollama, and a vault at
`~/vault`.

```bash
mkdir -p ~/.jarvis
$EDITOR ~/.jarvis/config.toml
```

A working OpenRouter config:

```toml
[chat]
provider = "openrouter"
openrouter_model = "anthropic/claude-sonnet-4.6"
vault_path = "~/Documents/obsidian"          # your notes; must exist

[auth]
openrouter_api_key = "sk-or-..."             # or the OPENROUTER_API_KEY env var

# Extra models for the picker. Optional.
[models]
openrouter = ["anthropic/claude-sonnet-4.6", "openai/gpt-5"]
```

Then restrict the permission of the file, since it can hold your API key:

```bash
chmod 600 ~/.jarvis/config.toml
```

In the config file, the `provider` is mandatory, and can be the following:
- `openrouter`: uses Openrouter service to route your prompts to their hosted LLM. 
Make sure you then supply the following as well:
  - `openrouter_api_key` under `[auth]`.
  - `openrouter_model` under `[chat]` which denotes what default model it will route your prompt to
  - `openrouter` setting under `[models]` which list all available models you can switch to.
  - If you want to use Openrouter's automatic routing, add `openrouter/auto` to either `openrouter_model` or `openrouter`. 
- `anthropic`: uses Anthropic service. Make sure you then supply `api_key` field under `[auth]` and fill it in with your Anthropic API key.
- `ollama`. Make sure you then supply `ollama_model` under `[chat]` which denotes what model it will route your prompt to.

### 3. Index your notes

```bash
uv run kb index-vault
```

The first run downloads and caches the `BAAI/bge-small-en-v1.5` embedding
model. If `vault_path` doesn't exist you get `Error: vault path does not exist`
before anything downloads. Skip this step if your vault is empty.

### 4. Check it worked

```bash
uv run kb models     # which providers are configured, and which lack a key
uv run kb stats      # what got indexed
uv run kb doctor     # embedding model and index health
```

`kb models` reads your config instead of the network, so it's the quickest way
to confirm the file is being picked up:

```
ollama:qwen3-vl:30b  [local]
anthropic:claude-sonnet-4-6  [cloud]  (no API key)
openrouter:anthropic/claude-sonnet-4.6  [cloud]
```

`(no API key)` means configured but unusable. A missing `openrouter:` line
means `openrouter_model` was never set.

### 5. Run it

```bash
uv run webapp        # browser at http://127.0.0.1:8080 (localhost only)
```

---

## Ask it things

Talk to it. It searches your notes, papers and past conversations before
answering, and shows you every step along the way.

```
what did I conclude about batch effects in the cytometry project?
which papers do I have on sparse autoencoders?
what did we discuss about this last week?
add https://arxiv.org/abs/2406.04093
```

By default it answers only from your knowledge base. Turn the **DB only**
toggle off and it can fall back on the model's own training knowledge, saying
on screen when it does.

Conversations save themselves. Resume, rename, pin or delete one from the
**Chats** section of the sidebar.

While a reply is generating, the Send button becomes a red **Stop**. A stopped
turn leaves no trace: your question returns to the input box, nothing is
written to the conversation, and nothing is indexed. Tokens a paid model
already generated are still billed.

---

## Write documents with it

Ask for a document and you get a real file you can open.

```
draft the methods section from my notes on the pipeline
```

Documents land in `~/.jarvis/drafts/`, which the assistant can write to freely.
Your vault stays read-only to it. To get a document out, right-click it, choose
**Show in Finder**, and copy it wherever you like.

A draft is a folder rather than a single file, so `main.tex`, its chapters and
its `.bib` live together and compile as a unit. A single-file draft shows as one
row; a multi-file one lists its parts underneath. Right-click a document to add
another file to it.

Click a document, or press **Show editor** in the header, and the editor opens
above the chat. Source on the left, preview on the right, **Recompile** to
re-render, and a layout control for split / source only / output only. Markdown
renders to HTML; LaTeX compiles to a PDF, with the log underneath when it
fails. Both export to PDF.

- Each file opens in its own tab. A filled dot means unsaved changes; an ×
  means none. Closing a tab saves it first if it needs saving.
- ⌘S saves the current tab and re-renders the preview. Previewing, compiling
  and exporting all save first.
- Scrolling the source scrolls the preview to the matching place. Markdown
  only, since a compiled PDF is displayed by the browser itself.
- Every earlier version is kept. **History** restores one, and restoring is
  itself undoable.
- When the assistant proposes a change you get a diff with a checkbox per hunk,
  and only what you tick gets written. A ✎ on a tab means a suggestion is still
  waiting. ⋮ → **Discard pending suggestions** clears them all, and they clear
  themselves on restart.
- Drafts expire after 30 days untouched. `uv run kb drafts` shows how long each
  has left, and **Keep** exempts one permanently. Set
  `[drafts] retention_days = 0` to turn the sweep off.

Exporting your first Markdown PDF on a new machine can be slow, sometimes past
the compile timeout, because fontspec builds a system font cache once. Get it
over with ahead of time:

```bash
printf '\\documentclass{article}\\usepackage{fontspec}\\begin{document}x\\end{document}' > /tmp/warm.tex
xelatex -output-directory=/tmp /tmp/warm.tex
```

### Editing the prompts

Three prompts drive everything jarvis asks a model to do, and all three are
configurable: ⋮ → **Edit prompts**.

| Prompt | What it controls |
|---|---|
| Assistant instructions | How the chat agent behaves: when it searches, how it cites |
| Paper summary | How a paper is summarised when you add one |
| Digest scoring | Which papers the weekly digest thinks matter |

Your editable copies live in `~/.jarvis/prompts/`, created on first run.
**Revert to default** will revert to a default prompt stored in this repo.
Any edits will apply to your next message with no restart.

If you turn the digest on, the scoring prompt important. It ships with
What you write there decides which papers are searched and retained.

---

## Keep records, not just notes

Any note can carry YAML config, and jarvis turns that into something you
can filter on. Handy for anything you accumulate a lot of and need to track the
state of: manuscripts, grants, experiments, meetings.

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

Then ask by record instead of by wording:

```
which manuscripts are under review, and what are reviewers asking for?
what have I got in drafting for Nature Methods?
show me everything I submitted this year
```

What fields you specify in the YAML config is yours to decide. 
The default config parsed by Jarvis are: `type`, `status`, `entity`, `date` and `tags`. 
Run `uv run kb schema` to see which keys and values exist. 


---

## Add papers and PDFs

```bash
uv run kb add https://arxiv.org/abs/2406.04093       # a summary (fast, default)
uv run kb add https://arxiv.org/abs/2406.04093 --full-text
uv run kb add paper.pdf                               # title/authors/DOI inferred
uv run kb add paper.pdf --authors "Ada Lovelace"      # or set them yourself
```

Or ask in chat: *"add ~/Downloads/paper.pdf, full text"*.

Set 
```
[sync] pdf_watch_dir = "~/Documents/papers/inbox"
``` 
and the background
daemon sweeps that folder every half hour.

Highlights and typed notes made in macOS Preview or Foxit Reader are indexed
Freehand pen scribbles aren't text, so they can't be extracted.

Figures can be captioned by a vision model and made searchable. This is off by
default, since every figure costs a call. Add `--figures`, or ask for a paper
"with figures".

---

## Choose a model

⋮ → **Switch model** changes model mid-conversation without losing the thread.
It applies from your next message, per conversation, which means two sessions
can run different models at once. The header shows the active model and what the
session has cost so far.

Whatever you set as `openrouter_model` (or `ollama_model`) already appears in
the picker. `[models]` adds more:

```toml
[models]
openrouter = ["anthropic/claude-sonnet-4.6", "openai/gpt-5", "openrouter/auto"]
```

The picker also has a box for typing any model id OpenRouter accepts, listed or
not, applied to the current conversation without touching your config. Keep
`[models]` to the handful you actually switch between.

OpenRouter's auto router is just a model id: set
`openrouter_model = "openrouter/auto"` and it picks a model per request. The
header then shows both halves, `openrouter/auto → claude-sonnet-4.6`, and
hovering the cost breaks the total down by model. Jarvis sends
`allow_fallbacks = false` by default, which hasn't been tested against the auto
router, so loosen it under `[openrouter]` if requests start failing.

Cost is shown only for OpenRouter, which reports what each request actually
cost. A local model costs nothing, and jarvis won't invent a figure for
anything else.

OpenRouter is a broker: your request routes to somebody else's hardware. Jarvis
sends `data_collection = "deny"` and no silent fallbacks by default.

---

## Keep things private

Notes in the folders listed under `private_vault_dirs` (default: `private/`)
are only ever visible to a local model.

```
vault/
├── private/    ← local model only, never sent anywhere
└── research/   ← any model
```

Once a conversation touches private content it stays private and can't switch
to a cloud model. Papers are always public, so keep anything sensitive in a
note.

Jarvis never deletes a file (removing a document removes its database entry
only) and never writes to your vault.

---

## Jarvis sync and index refresh

```bash
uv run jarvis-sync
```

Optional. It's a scheduler that runs the same vault sync, PDF sweep and draft
cleanup you can run by hand, just on a timer. Skip it and you index manually
after editing your notes. It logs to `~/.jarvis/logs/sync.log` and to the
terminal, and stays in the foreground, so run it under `tmux` or `screen` if
you want it to survive closing the terminal. It won't start Ollama for you.


### Updating the index by hand

`kb index-vault` is almost always the one you want. It's incremental and it's
exactly what the daemon runs on its timer. `kb reindex` re-embeds text already
in the database and never looks at your vault. It won't notice a note you
changed.

| You did this | Run this |
|---|---|
| Edited, added or deleted notes | `kb index-vault` |
| Want a clean rebuild of the note index | `kb index-vault --force` |
| Changed `embed_model` in the config | `kb reindex` |
| `kb doctor` reported a corrupt index | `kb reindex` (or `--from-storage`) |

---

## Weekly paper digest

Off by default. To switch on automated arXiv/bioRxiv discovery:

```toml
[digest]
enabled = true
arxiv_categories = [["cs.LG", 150], ["cs.AI", 80]]
```

It fetches weekly, scores everything against your relevance prompt, writes a
tiered Markdown digest, and indexes the best papers. If your machine was
asleep, it catches up automatically. Run one by hand with
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
uv run kb schema               # which metadata keys/values exist
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
drafts, logs. The full configuration reference is in
[`docs/DESIGN.md`](docs/DESIGN.md#configuration--jarviscoreconfigpy).
