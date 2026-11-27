# Jarvis

[![Tests](https://github.com/ghar1821/jarvis/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/ghar1821/jarvis/actions/workflows/tests.yml)

Your Obsidian notes and your research papers, searchable from any AI assistant: Claude Desktop, Google
Antigravity, Claude Code and others.

Jarvis indexes two folders you already have, your vault and a folder of PDFs.
It keeps the index in step with them, and gives your assistant read-only
tools to search and read them via [MCP](https://modelcontextprotocol.io). 
The assistant does the thinking. 
Jarvis makes no LLM calls, holds no API keys, never writes to your folders.

Named after Iron Man's J.A.R.V.I.S. (Just A Rather Very Intelligent System).

Full documentation: **<https://ghar1821.github.io/jarvis/>**

---

## Setup

### Requirements

- [uv](https://github.com/astral-sh/uv) and Python 3.12 or newer. If you have
  neither: `curl -LsSf https://astral.sh/uv/install.sh | sh`, then
  `uv python install 3.12`.
- An MCP client, such as Claude Desktop or Google Antigravity.

### 1. Install

```bash
git clone <repo-url> jarvis && cd jarvis
uv sync
```

### 2. Write the config

Jarvis reads `~/.jarvis/config.toml`. The minimum is where your two folders
are:

```toml
[kb]
vault_path = "~/Documents/obsidian"     # your Obsidian vault (default: ~/vault)
papers_dir = "~/Documents/papers"       # PDFs here, and in its subfolders, are your papers
```

- **`vault_path`** can be left out if your vault is at `~/vault`.
- **`papers_dir`** can be left out, but then no papers are indexed, only notes.

Every note in `vault_path` is indexed and readable by your assistant, so keep
private notes in a separate folder outside it (see
[Keep things private](#keep-things-private)).

There are no API keys to add. Everything else has a sensible default (see
[`docs/DESIGN.md`](docs/DESIGN.md#configuration)).

### 3. Start the index

```bash
uv run kb serve
```

Leave it running in its own terminal or a tmux pane. It runs the search index
and re-syncs both folders every 30 minutes. The first start downloads and
caches the embedding model (`BAAI/bge-small-en-v1.5`), then indexes everything.
`uv run kb doctor` checks it's healthy, and `uv run kb status` shows the last
sync.

Your assistant needs `kb serve` running. The MCP server only connects to the
index that `kb serve` runs; it never opens the index files itself, so it can't
end up searching a stale copy while the sync rewrites them. If `kb serve` isn't
running:

- **The client still connects.** Jarvis and its tools show up as normal.
- **Every tool call fails.** The error says "The knowledge-base server is not
  running. Start it with: `uv run kb serve`".
- **It recovers by itself.** Once you start `kb serve`, the next tool call
  works without restarting the client.

### 4. Connect your assistant

**Claude Desktop**

1. Open Settings → Developer → Edit Config. This opens
   `~/Library/Application Support/Claude/claude_desktop_config.json`.
2. Add a `jarvis` entry under `mcpServers`. If other servers are already
   listed, add it beside them rather than replacing the file:

   ```json
   {
     "mcpServers": {
       "jarvis": {
         "command": "/absolute/path/to/uv",
         "args": ["--directory", "/absolute/path/to/jarvis", "run", "jarvis-mcp"]
       }
     }
   }
   ```

   Use the full path to `uv` (`which uv` prints it). Claude Desktop starts
   servers with a minimal `PATH` and won't find a bare `uv`.
3. Quit Claude Desktop completely (⌘Q, not just closing the window) and reopen
   it.
4. "jarvis" now appears in the chat box's tools menu with seven tools. Try
   *"what's in my knowledge base?"*. The first call takes a few seconds while
   the models load, and Claude asks permission the first time each tool runs.

If it doesn't connect:

- **Read the log.** It's at `~/Library/Logs/Claude/mcp-server-jarvis.log`.
- **Run the command yourself.** Run the same `command` plus `args` in a
  terminal. It should sit silently waiting for input (Ctrl-C to stop). Any
  error it prints is the one Claude Desktop hits.

**Claude Code**, including scheduled tasks created in Claude Desktop's
**Code** tab. Claude Code keeps its own list of MCP servers and doesn't read
`claude_desktop_config.json`, so add jarvis to it separately:

```bash
claude mcp add --scope user jarvis -- /absolute/path/to/uv --directory /absolute/path/to/jarvis run jarvis-mcp
```

`--scope user` makes jarvis available in every Claude Code session.
`claude mcp list` should then show `jarvis: … ✔ Connected`.

**Antigravity, or any other MCP client.** Add the same `command` and `args` to
its MCP server configuration. Any client that can launch a local (stdio) MCP
server works the same way.

The client starts and stops `jarvis-mcp` itself, so you never run it by hand.
It runs whatever code is checked out in the jarvis folder, so after you switch
branches or pull, restart the client.

---

## Ask it things

Just ask. Your assistant decides when to search your notes and papers:

```
what did I conclude about batch effects in the cytometry project?
which papers do I have on sparse autoencoders?
read me the methods section of the DIA-NN paper
where's that paper on spillover correction? show it in Finder
do I already have 10.1101/2024.05.01.592000?
```

The tools jarvis gives it:

| Tool | What it does |
|---|---|
| `search_kb` | Searches notes and papers, combining semantic and keyword search with re-ranking. Each hit has the full passage, the file path and, for papers, a DOI link |
| `get_document` | Reads a whole paper, page by page |
| `read_note` | Reads one vault note in full |
| `list_documents` | Lists your papers or notes, optionally filtered by record fields |
| `kb_stats` | Shows counts, and which record types and statuses exist |
| `has_paper` | Checks whether you already have a paper, by DOI, arXiv ID or title |
| `reveal_in_finder` | Opens Finder with a paper or note selected |

Every tool only reads. Your assistant can't add, edit, move or delete
anything through jarvis.

---

## Add and remove papers and notes

The index mirrors your files:

| You do this | The index |
|---|---|
| Save a PDF into `papers_dir` (any subfolder) | Adds it as a paper, full text |
| Write or edit a note in the vault | Adds it or updates it |
| Move or rename a PDF | Follows it, without re-indexing |
| Delete a file | Drops it |
| Move a file out of the folder | Drops it |

Changes show up at the next sync (within `sync_minutes`, 30 by default).
`uv run kb sync` runs one now.

**Paper details.** Jarvis reads the DOI printed on the first page and looks
up the title and authors on [Crossref](https://www.crossref.org). arXiv
preprints usually print no DOI, so for those it reads the arXiv ID that arXiv
stamps in the page margin and asks arXiv instead. Each lookup sends only the
DOI or arXiv ID. With neither, it uses the PDF's own metadata, and failing
that, the filename.

**Highlights and typed notes.** Highlights and typed notes made in macOS
Preview or Foxit Reader are indexed too, and saving new ones re-indexes the
paper. Freehand pen scribbles aren't text, so they can't be read.

**Scanned PDFs.** A PDF with no text layer can't be searched, but its
highlights and notes still are.

---

## Keep records, not just notes

Any note can carry YAML frontmatter, and jarvis turns it into fields you can
filter on. That's handy for anything you accumulate and need to track:
manuscripts, grants, experiments, job applications.

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
```

`type`, `status`, `entity`, `date` and `tags` are the named fields; any other
key works too. `uv run kb schema` shows which keys and values exist, which is
also how you catch a typo like `stauts`.

---

## Keep things private

Every MCP client sends what it reads to a cloud model. jarvis only ever reads
inside `vault_path` and `papers_dir`, so the way to keep a note private is to
keep it outside both, for example in a second Obsidian vault:

```
Documents/
├── obsidian/   ← vault_path: searchable by your assistant
└── private/    ← a separate vault: jarvis never reads it
```

A link (symlink) inside the vault that points somewhere else does not pull
that file in. jarvis skips it and logs a warning naming where it points.

The index that `kb serve` runs only answers requests that carry a secret
token, which `kb serve` creates in `~/.jarvis/state/server_token`, readable
only by you. Other programs on your Mac can't read the index or add to it.

---

## Checking whether you already have a paper

jarvis doesn't fetch new papers. Your assistant's own connectors do that
better, for example PubMed in Claude Desktop. When it suggests papers, ask it
to check each one with `has_paper` first, so it skips the ones in your papers
folder. A prompt line like this works:

```text
Before listing a paper, call jarvis's has_paper with its DOI, arXiv ID and
title, and leave it out if I already have it.
```

`has_paper` checks your papers folder, not the assistant. A paper matches if
it has the same DOI (ignoring case), the same arXiv ID (any version) or the
same title (ignoring case and punctuation). It can miss a paper whose PDF
prints neither identifier and whose title is worded differently, for example
a journal PDF of a paper you only know as a preprint. A PDF you saved in the
last few minutes is found once `kb serve` has synced it.

To keep a paper, open its DOI link and save the PDF into `papers_dir`.

For a full example, see
[`docs/examples/paper-digest-task.md`](docs/examples/paper-digest-task.md).
It is a weekly scheduled task that finds new arXiv and bioRxiv papers, uses
`has_paper` to drop the ones you already have, and scores the rest against
your research interests. Edit its research context to match your work.

---

## Upgrading from an older jarvis

Older versions had their own chat app (a web UI with model switching, drafts
and a document editor), added papers by arXiv URL, and kept digests and chat
history in the index. That version is preserved on the
`archive/pre-mcp-simplification` branch if you ever need it.

An index built by an older version has to be rebuilt once. Until it is,
jarvis refuses to use it and says so. To upgrade:

1. If your config sets `private_vault_dirs`, move those notes out of the
   vault first, then delete the setting. jarvis no longer keeps folders
   inside the vault private, and it won't start while the setting is there.
2. Put your settings under `[kb]` and `[rag]` as shown above. Old sections
   such as `[chat]` and `[sync]` are no longer read; jarvis names every
   setting it ignores, and never edits the file itself.
3. Stop `kb serve`, then run `uv run kb rebuild` and answer `y`. It rebuilds
   the index from your vault and `papers_dir`. Everything else is dropped:
   arXiv-URL papers, digests, chat history, PDFs outside `papers_dir`. Only
   index entries go; no file is touched.
4. Start `kb serve` again, then restart your MCP client.

Old `~/.jarvis/sessions/`, `~/.jarvis/drafts/` and `~/.jarvis/prompts/`
folders are left untouched. Delete them yourself if you don't need them.

---

## Command reference

```bash
uv run kb serve       # run the index and keep it in sync (leave running)
uv run kb sync        # sync both folders now
uv run kb rebuild     # recreate the index from your folders
uv run kb list        # indexed papers (--notes for notes and records)
uv run kb stats       # counts
uv run kb schema      # which metadata keys and values exist
uv run kb status      # is kb serve running, and how did the last sync go
uv run kb doctor      # diagnose the index
# jarvis-mcp is started by your MCP client, not by hand
```

`kb serve` is the only long-running process. The other commands work whether
or not it's running.

| Something's wrong | Run this |
|---|---|
| A note or paper is missing or stale | `kb sync`, then check `kb status` |
| Your assistant says the index can't resolve an id | `kb doctor` |
| `kb doctor` dies or reports damage | `kb rebuild` (nothing is lost; every entry comes from a file) |
| You changed `embed_model` in the config | `kb rebuild` |
| jarvis says the index was built by an older version | `kb rebuild` |
| jarvis says the server refused its token | Restart `kb serve`, then your MCP client |

Everything jarvis stores lives under `~/.jarvis/`: `config.toml`, the index,
logs, the `kb serve` status file and its server token. All of it is readable
only by you.
