# jarvis

A personal research knowledge base for a computational biologist who monitors AI/ML literature. Jarvis combines automated arXiv paper discovery with a persistent, locally-run vector database and a conversational agent — so you can query your reading history, add papers on demand, and manage your Obsidian notes, all through natural language.

Named after Iron Man's J.A.R.V.I.S. — Just A Rather Very Intelligent System.

See [`docs/DESIGN.md`](docs/DESIGN.md) for architecture documentation.

---

## What it does

**Automated paper discovery and background sync**
- The `jarvis-sync` daemon fetches papers weekly from configured arXiv categories — with catch-up if the machine was asleep or powered off at the scheduled time
- Scores and ranks them with the configured LLM against a custom relevance prompt
- Writes a tiered Markdown digest; papers scoring ≥ 9 are automatically indexed into the knowledge base
- Watches a PDF inbox folder: any PDF dropped in is auto-indexed; the vault index is refreshed periodically

**Personal knowledge base**
- Stores papers as LLM-generated summaries (~1000 words) or as full chunked text for deep querying
- Indexes your Obsidian vault notes alongside papers in a single local vector store (runs entirely on your machine)
- Extracts highlights and typed notes from annotated PDFs and makes them searchable (see [PDF annotations](#pdf-annotations))
- Retrieval combines BGE embeddings, section-aware chunking, and a cross-encoder reranker for accurate matches — all local, no external calls
- Privacy model: vault folders marked private are accessible to the local model only — never sent to cloud APIs. Papers are always public; only notes can be private

**Conversational agent (`vault-chat` and web UI)**
- Query your knowledge base in natural language
- Add papers by arXiv URL or local PDF mid-conversation
- Remove entries, trigger vault re-indexing, and check stats through the same chat interface — deletions always require a human confirmation, and note files are never deleted from disk
- Persistent chat sessions: resume, pin, delete, and search past conversations
- User-defined skills and a configurable response style
- Runs against a local model via Ollama or Anthropic Claude (switchable per session)
- Terminal interface (`vault-chat`) or browser interface (`webapp`, localhost only)

---

## Repository structure

```
├── digest/
│   ├── config.py, errors.py, llm.py   # Shared infrastructure
│   ├── daemon.py                       # jarvis-sync background daemon
│   ├── arxiv/                          # arXiv fetching and PDF download
│   │   ├── fetch.py
│   │   └── convert.py
│   ├── pipeline/                       # Automated weekly digest
│   │   ├── run.py, score.py, format.py
│   │   └── prompts/prompt_filter_score.md
│   └── kb/                             # Knowledge base management
│       ├── store.py, cli.py
│       ├── convert.py                  # PDF → Markdown (pymupdf4llm)
│       ├── annotations.py              # PDF highlight/note extraction
│       └── prompts/paper_summary.md
├── vault_chat/
│   ├── chat.py                         # Conversational KB agent
│   ├── sessions.py                     # Persistent chat sessions
│   └── skills.py                       # User-defined skills
├── webapp/
│   ├── app.py, run.py                  # FastAPI web UI (localhost:8080)
│   ├── index.html                      # Chat UI page
│   └── static/                         # style.css, app.js
├── docs/
│   ├── DESIGN.md
│   ├── CHANGELOG.md
│   ├── LAUNCHD_SETUP.md
│   └── RENAME.md
└── pyproject.toml
```

---

## Setup

```bash
uv sync
```

**Local model (Ollama):** the local provider talks to [Ollama](https://ollama.com) on `http://localhost:11434`. Install Ollama and pull a model that supports **tool calling and vision** — vision is needed for figure captioning and vision-based summaries:

```bash
ollama pull qwen3-vl:30b
```

`qwen3-vl:30b` is the default (a vision + thinking MoE that fits a 36GB M3 Max); confirm the exact tag with `ollama list`, since Ollama's registry names can shift. Ollama runs as a macOS login-item app or via `ollama serve` — no LaunchAgent needed. To keep the sync daemon running permanently via launchd, see [docs/LAUNCHD_SETUP.md](docs/LAUNCHD_SETUP.md). Summary mode converts the PDF to markdown locally (via `pymupdf4llm`) before summarising, so it does not need the PDF-document API.

**Upgrading an existing config:** old configs with `provider = "llamacpp"` and `llamacpp_url` / `llamacpp_model` should switch back to `provider = "ollama"` and `ollama_model` (see [Configuration](#configuration)). Old configs may also carry a stale `rag_dir = "~/.seshat/rag"` — fix it to `~/.jarvis/rag`.

**Upgrading an existing knowledge base:** the default embedding model is now `BAAI/bge-small-en-v1.5`. Run `uv run kb reindex` once to re-embed your existing chunks — the app refuses to search a knowledge base built with a different embedding model until you do. This re-embeds stored chunk text only (no LLM calls, nothing re-downloaded). To also benefit from the newer section-aware chunking, run `uv run kb index-vault --force` for vault notes.

---

## Configuration

All settings live in `~/.jarvis/config.toml`. Optional — defaults apply if absent.

```toml
[digest]
anthropic_model = "claude-sonnet-4-6"
output_dir = "~/Documents/papers/digest"
max_results = 10
# arxiv_categories is a list of [category, limit] pairs:
# arxiv_categories = [["cs.LG", 150], ["cs.AI", 80]]
# bioRxiv: categories filter server-side (only real bioRxiv categories work);
# keywords match title+abstract client-side over the recent window.
# biorxiv_categories = [["bioinformatics", 100]]
# biorxiv_keywords = [["cytometry", 50], ["spatial transcriptomics", 50], ["scRNA-seq", 50]]
# biorxiv_days = 7

[rag]
rag_dir = "~/.jarvis/rag"
embed_model = "BAAI/bge-small-en-v1.5"                                      # changing it requires `kb reindex`
query_prefix = "Represent this sentence for searching relevant passages: " # query-side prefix for BGE models; "" to disable
chunk_size = 1024
chunk_overlap = 128
rerank_model = "cross-encoder/ms-marco-MiniLM-L6-v2"                        # cross-encoder reranker; "" to disable
rerank_top_n = 25                                                          # candidates fetched before re-ranking
figure_captions = true                                                     # caption PDF figures at ingest (needs a vision model); false to disable
figure_max_per_doc = 20                                                    # cap on figures captioned per document
figure_min_pixels = 40000                                                  # skip images smaller than this (logos, rules)

[chat]
provider = "ollama"              # "ollama" | "anthropic"
ollama_model = "qwen3-vl:30b"    # Ollama tag; needs tool calling + vision for full functionality
vault_path = "~/Documents/obsidian"
private_vault_dirs = ["private"] # top-level vault dirs only accessible to local model
skills_dir = "~/.jarvis/skills"  # user-written skill files (*.md); missing folder = feature off
response_style = ""              # free-text instruction for how the assistant should write replies
compact_after_tokens = 12000     # compact long sessions past this estimated context size
compact_keep_exchanges = 6       # recent turns kept verbatim when compacting

[sync]
pdf_watch_dir = "~/Documents/papers/inbox"  # PDF inbox for jarvis-sync; omit to disable the watcher
vault_refresh_minutes = 30
digest_day = "mon"               # day of week for the weekly digest
digest_hour = 2

[auth]
api_key = ""                     # Anthropic API key (alternative to ANTHROPIC_API_KEY env var)
```

Env var overrides: `OLLAMA_MODEL`, `ANTHROPIC_MODEL`, `CHAT_PROVIDER`, `VAULT_PATH`, `PDF_WATCH_DIR`.

Because the file can hold your API key, keep it private (`chmod 600 ~/.jarvis/config.toml`) — `jarvis-sync` and `vault-chat` warn at startup when it is group/world-readable.

To customise the agent's behaviour, create `~/.jarvis/system_prompt.md`.

---

## Privacy model

Vault notes under top-level directories listed in `private_vault_dirs` are private — visible to the local Ollama model only. Cloud providers (Anthropic) skip those chunks entirely and cannot read those files via `read_file`. The check runs on the resolved path, so a symlink in a public folder cannot reach into a private one.

```
vault/
├── private/    ← local model only
│   └── journal/
└── research/   ← cloud + local
```

**Papers are always public.** Only notes — vault files and note-type PDFs — can be private; adding a private paper is rejected. Chat sessions that touch private content are flagged private and stay local-only (see [Chat sessions](#chat-sessions)).

---

## Usage

All commands require the `uv run` prefix (entry points live in `.venv/bin/`). Alternatively, activate the venv once with `source .venv/bin/activate`.

### Vault chat — conversational KB agent

```bash
uv run vault-chat                           # uses provider from config
uv run vault-chat ~/path/to/vault           # override vault path
CHAT_PROVIDER=anthropic uv run vault-chat   # use Anthropic for this session
uv run vault-chat --no-db-only              # allow AI knowledge fallback when DB has no results
uv run vault-chat --list-sessions           # list stored chat sessions
uv run vault-chat --resume <SESSION_ID>     # resume a stored session
```

By default (`--db-only` behaviour), the agent answers only from documents in the knowledge base. Pass `--no-db-only` to allow the LLM to fall back to its training knowledge when the database returns no relevant results — it will call `use_own_knowledge` first to make the fallback visible.

### Chat sessions

Every conversation is saved automatically to `~/.jarvis/sessions/` after each turn. Resume, pin, or delete sessions from the webapp sidebar, or from the terminal with `vault-chat --list-sessions` / `--resume <id>`.

- **Retention:** the 50 most recent unpinned sessions are kept; pinned sessions are exempt and never counted.
- **Search:** past exchanges are indexed into the knowledge base, so the agent can recall earlier conversations via its `search_chat_history` tool ("that paper we discussed last week").
- **Privacy:** a session that ever touches private content is flagged private permanently (shown with a lock badge in the webapp). Its history and indexed exchanges stay local-only, and resuming it under the Anthropic provider is refused.
- **Compaction:** long sessions are compacted automatically — older exchanges are summarised by the session's own provider once the context passes `compact_after_tokens`, keeping the last `compact_keep_exchanges` turns verbatim. The visible history in the UI is never trimmed.

### Skills

Drop `.md` files into `~/.jarvis/skills/` (configurable via `[chat] skills_dir`) to teach the agent reusable procedures. The filename is the skill name and the first non-empty line is its one-line description. Only name + description go into the system prompt; the model loads the full instructions on demand with its `read_skill` tool when a task matches. Delete the folder (or leave it empty) to switch the feature off.

### Response style

Set `[chat] response_style` to a free-text instruction ("short and concise, no filler") to control how the assistant writes replies. It can also be edited live in the webapp via the header ⋮ menu → "Set response style…", which opens a modal that persists it back to `config.toml` (comments preserved).

### Web UI

```bash
uv run webapp                          # uses provider from config
uv run webapp --provider anthropic     # override provider for this session
uv run webapp --provider ollama
```

Same agent as `vault-chat`, in a dark theme. Tool calls appear live in a collapsible box while the agent is working. Localhost only — not accessible from other machines.

The sidebar lists stored chat sessions: click to resume, rename (✎), pin to protect from pruning, or delete. Private sessions show a lock badge and cannot be resumed under the Anthropic provider. The response style is set from the header ⋮ menu (see above).

A **DB only** toggle (on by default) restricts the agent to the knowledge base. Switch it off to allow the model to fall back to its training knowledge — an amber status badge appears whenever this happens.

When the agent requests a deletion, the webapp shows a Confirm/Cancel dialog — nothing is removed until you click Confirm. The model can only ask; the decision is always yours.

Available tools the agent can call:

| Tool | What it does |
|---|---|
| `retrieve_papers` | Semantic search over indexed papers |
| `search_notes` | Semantic search over vault notes |
| `search_chat_history` | Semantic search over past conversations |
| `read_file` | Read a specific vault file in full |
| `read_skill` | Load a user-defined skill's full instructions (only available when skills exist) |
| `add_document` | Add a paper by arXiv URL or local PDF |
| `remove_document` | Preview a removal, then request it — a human must confirm (terminal prompt or webapp dialog) before anything is deleted; only paper PDFs can be deleted from disk, note files never are |
| `list_papers` | List all indexed papers |
| `kb_stats` | Paper, note, and chunk counts |
| `update_file_path` | Update stored path for a moved or renamed local file |
| `index_vault` | Incremental vault sync (the destructive clean rebuild is CLI-only: `kb index-vault --force`) |
| `use_own_knowledge` | Called by the LLM before answering from training knowledge (only available when DB only is off) |

Example interactions:

```
You: index my vault
You: add https://arxiv.org/abs/2406.04093, score 9, Track 1
You: add ~/Downloads/paper.pdf as a private document, full text mode
You: what papers do we have on sparse autoencoders?
You: remove the paper about SAE probing
You: what are my notes on transformers?
You: how many papers are indexed?
```

#### Paper storage modes

When adding a paper the agent uses `summary` mode by default. Specify `full text` in your message to override.

| Mode | What is stored | Use when |
|---|---|---|
| `summary` (default) | LLM generates a dense ~1000-word summary; 1–2 chunks | Most papers — fast and compact |
| `full_text` | PDF converted to Markdown and fully chunked | Papers you want to query at paragraph level |

### PDF annotations

Highlights and typed notes made in macOS Preview or Foxit Reader are extracted automatically whenever a PDF is ingested — via `kb add`, the chat `add_document` tool, the sync daemon's inbox, or the vault refresh of PDF notes. Each annotation becomes its own searchable chunk, prefixed `[HIGHLIGHT p.N]` or `[USER NOTE p.N]`, so the agent can answer "what did I highlight in that paper?".

What is extracted:

- **Highlights in any colour** — extraction keys on the annotation type, never the colour. Underline, squiggly, and strikeout markup count as highlights too.
- **Typed notes** — sticky notes, text boxes, and comments typed onto a highlight.

What is not:

- **Freehand/handwritten drawing (Ink)** — Preview's Sketch/Draw tools and stylus scribbles store stroke geometry, not text, and are not extracted. Anything you want searchable must be typed or highlighted.

Re-saving a PDF after adding annotations re-indexes it automatically: the inbox watcher and vault refresh detect the changed byte hash and replace the old chunks.

### PDF figure captioning

Text embeddings can't see images, so figures in a PDF would otherwise be lost. On every PDF ingest (the same sites as annotations) each embedded figure is captioned by the active provider's **vision model** and indexed as a `[FIGURE p.N]` chunk, so the agent can answer "what does the figure on page 4 show?". Requires a vision-capable model — the default `qwen3-vl:30b` qualifies; a text-only model will error per figure and skip it. Turn it off with `[rag] figure_captions = false`; tune volume with `figure_max_per_doc` and `figure_min_pixels` (tiny images like logos are skipped).

Privacy: figures of a **private** note are never captioned under the Anthropic provider — the images would reach the cloud. Switch to the local model to caption them. Papers are always public, so paper figures caption under either provider.

### Knowledge base CLI (`kb`)

For scripted use, batch imports, and initial setup.

```bash
# Vault indexing (incremental by default; --force clears vault .md index first)
uv run kb index-vault
uv run kb index-vault --vault-path ~/path/to/vault --force

# Re-embed all chunks with the configured embed_model (no LLM calls).
# Run once after upgrading or after changing embed_model / query_prefix.
uv run kb reindex

# Add a paper by arXiv URL
uv run kb add https://arxiv.org/abs/2406.04093
uv run kb add https://arxiv.org/abs/2406.04093 --score 9 --track "Track 1"
uv run kb add https://arxiv.org/abs/2406.04093 --full-text   # store full PDF text

# Add a local PDF (--doc-type paper or note; papers are always public,
# so --visibility private requires --doc-type note)
uv run kb add paper.pdf --doc-type paper --full-text
uv run kb add notes.pdf --doc-type note --visibility private

# Override the provider used for summary generation
uv run kb add https://arxiv.org/abs/2406.04093 --provider anthropic

# Bulk-import previous digest files (no LLM call — reuses existing summaries)
uv run kb add-digest ~/Documents/papers/digest/
uv run kb add-digest ~/Documents/papers/digest/ --min-score 7

# Inspect
uv run kb list
uv run kb list --limit 100
uv run kb stats          # also warns about legacy private papers (papers must be public)
uv run kb sync-status    # jarvis-sync daemon health and last job outcomes

# Clear everything (prompts for confirmation, no files deleted)
uv run kb clear

# Remove (shows preview and asks for confirmation; --delete-file removes the
# source file too, but only for paper PDFs — note files are never deleted)
uv run kb remove https://arxiv.org/abs/2406.04093
uv run kb remove file:///path/to/paper.pdf --delete-file
```

### Weekly digest

```bash
uv run run-digest
```

Fetches papers from arXiv and bioRxiv, scores them against the relevance prompt, writes a tiered Markdown digest to `output_dir`, and automatically indexes papers with score ≥ 9 into the knowledge base. bioRxiv is pulled by category (only real bioRxiv categories like `bioinformatics` filter server-side) plus free-text keywords for topics with no category (cytometry, spatial transcriptomics, scRNA-seq); the same paper arriving from two sources is deduplicated by title before indexing. Scoring uses whichever provider `[chat] provider` names; with the local provider, Ollama must be running. Normally you never run this by hand — the sync daemon schedules it (see [Background sync daemon](#background-sync-daemon-jarvis-sync)).

### Anthropic authentication

**Option 1 — environment variable:**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

**Option 2 — config file** (persists across sessions, never leaves your machine):
```toml
# ~/.jarvis/config.toml
[auth]
api_key = "sk-ant-..."
```

### PDF conversion (standalone)

```bash
uv run convert-pdf --input https://arxiv.org/abs/2301.07041
uv run convert-pdf --input paper.pdf --output-dir ./output
```

Converts arXiv PDFs (by URL or local file) to Markdown using pymupdf4llm. Text only — images are not extracted, and a scanned PDF without a text layer produces an error (there is no OCR fallback).

---

## Background sync daemon (`jarvis-sync`)

One supervised process handles all background work:

- **Weekly arXiv digest** (default Monday 02:00, configurable via `[sync]`) — with catch-up: a run missed while the Mac was asleep fires on wake, and a run missed while powered off runs at the next daemon start.
- **PDF inbox watcher** — any PDF dropped into `pdf_watch_dir` is auto-indexed as a public full-text paper, annotations included. Re-saving a PDF with new annotations re-indexes it (byte-hash change detection). The folder is an inbox, not a mirror: removing a file never deletes its knowledge base entry.
- **Periodic vault refresh** — the incremental Obsidian sync, every `vault_refresh_minutes` (default 30).

```toml
[sync]
pdf_watch_dir = "~/Documents/papers/inbox"
vault_refresh_minutes = 30
digest_day = "mon"
digest_hour = 2
```

One failing job never takes the daemon down; check health any time with:

```bash
uv run kb sync-status
```

Run the daemon permanently under launchd (`KeepAlive`) — setup, plists, and troubleshooting are in [docs/LAUNCHD_SETUP.md](docs/LAUNCHD_SETUP.md). The daemon does not start Ollama for you; run Ollama as a login-item app or `ollama serve`.

---

## Requirements

- [uv](https://github.com/astral-sh/uv)
- Python ≥ 3.12
- [Ollama](https://ollama.com) with a tool-calling + vision model pulled, e.g. `qwen3-vl:30b` (for local inference)
- Anthropic API key (for cloud inference only; set via env var or `~/.jarvis/config.toml`)
- `fastapi` and `uvicorn` (included in `uv sync`; required for the web UI only)
