# jarvis

A personal assistant that knows your own notes, documents and papers, runs on
your machine, and can write documents with you.

Named after Iron Man's J.A.R.V.I.S. — Just A Rather Very Intelligent System.

> This README is about **using** jarvis. For how it works inside — architecture,
> data flows, the privacy and safety guarantees and why they hold — see
> [`docs/DESIGN.md`](docs/DESIGN.md).

---

## Get started

```bash
uv sync
```

Then pick a model to run against. Either works, and you can switch later:

**Local (nothing leaves your machine).** Install [Ollama](https://ollama.com)
and pull a model with tool-calling and vision support:

```bash
ollama pull qwen3-vl:30b
```

**Cloud.** Put a key in `~/.jarvis/config.toml`:

```toml
[chat]
provider = "openrouter"                     # or "anthropic"
openrouter_model = "anthropic/claude-sonnet-4.6"

[auth]
openrouter_api_key = "sk-or-..."            # or ANTHROPIC_API_KEY / [auth] api_key
```

Point jarvis at your notes and index them:

```toml
[chat]
vault_path = "~/Documents/obsidian"
```

```bash
uv run kb index-vault
```

Now start it:

```bash
uv run webapp        # browser at http://127.0.0.1:8080 (localhost only)
uv run vault-chat    # or the terminal
```

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
the **DB only** toggle off (or run `vault-chat --no-db-only`) to let it fall
back on the model's own knowledge — it says so on screen when it does.

Conversations are saved automatically. Resume, rename, pin or delete them from
the sidebar, or with `vault-chat --list-sessions` and `--resume <id>`.

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

Open the **Editor** view in the webapp: drafts on the left, source in the
middle, live preview on the right, chat docked beside it. Markdown previews as
you type; LaTeX compiles to a PDF (with the log underneath when it fails); both
export to PDF.

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

Switch mid-conversation without losing the thread: `/model` in the terminal, or
the ⌘ button in the webapp header.

```bash
uv run kb models              # what's on offer
uv run kb models --refresh    # pull OpenRouter's catalogue into your config
```

List the models you want offered under `[models]` in your config — jarvis ships
no vendor list of its own.

**Cost** is shown only for OpenRouter, which reports what each request actually
cost (`/cost` in the terminal, the header in the webapp). A local model costs
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
uv run webapp                  # browser UI
uv run vault-chat              # terminal chat
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

## Requirements

- [uv](https://github.com/astral-sh/uv) and Python ≥ 3.12
- For local inference: [Ollama](https://ollama.com) with a tool-calling +
  vision model (e.g. `qwen3-vl:30b`)
- For cloud inference: an OpenRouter or Anthropic API key
- Optional, for the editor's PDF output: a LaTeX distribution, and `pandoc` for
  Markdown export

---

## Documentation

- [`docs/DESIGN.md`](docs/DESIGN.md) — architecture, data flows, and the
  security and privacy guarantees
- [`docs/CHANGELOG.md`](docs/CHANGELOG.md) — what changed and why
- [`docs/TESTING.md`](docs/TESTING.md) — what's covered and how to run the tests
