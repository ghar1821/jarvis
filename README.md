# seshat

A personal research knowledge base for a computational biologist who monitors AI/ML literature. Seshat combines automated arXiv paper discovery with a persistent, locally-run vector database and a conversational agent — so you can query your reading history, add papers on demand, and manage your Obsidian notes, all through natural language.

Named for the ancient Egyptian goddess of knowledge and writing.

See [`docs/DESIGN.md`](docs/DESIGN.md) for architecture documentation.

---

## What it does

**Automated paper discovery**
- Fetches papers weekly from configured arXiv categories
- Scores and ranks them with a local LLM against a custom relevance prompt
- Writes a tiered Markdown digest; papers scoring ≥ 9 are automatically indexed into the knowledge base

**Personal knowledge base**
- Stores papers as LLM-generated summaries (~1000 words) or as full chunked text for deep querying
- Indexes your Obsidian vault notes alongside papers in a single local vector store (ChromaDB)
- Privacy model: vault folders marked private are accessible to the local model only — never sent to cloud APIs

**Conversational agent (`vault-chat`)**
- Query your knowledge base in natural language
- Add papers by arXiv URL or local PDF mid-conversation
- Remove entries, trigger vault re-indexing, and check stats through the same chat interface
- Runs against a local Ollama model or Anthropic Claude (switchable per session)

---

## Repository structure

```
├── digest/
│   ├── config.py, errors.py, llm.py   # Shared infrastructure
│   ├── arxiv/                          # arXiv fetching and PDF conversion
│   │   ├── fetch.py
│   │   └── convert.py
│   ├── pipeline/                       # Automated weekly digest
│   │   ├── run.py, score.py, format.py
│   │   └── prompts/prompt_filter_score.md
│   └── kb/                             # Knowledge base management
│       ├── store.py, cli.py
│       └── prompts/paper_summary.md
├── vault_chat/
│   └── chat.py                         # Conversational KB agent
├── docs/
│   ├── DESIGN.md
│   ├── CHANGELOG.md
│   ├── LAUNCHD_SETUP.md
│   └── RENAME.md
├── run_digest.sh                       # Shell wrapper for launchd scheduling
└── pyproject.toml
```

---

## Setup

```bash
uv sync
uv pip install -e .
```

Requires [Ollama](https://ollama.com) running locally with the configured model pulled (default: `gemma4:26b`).

---

## Configuration

All settings live in `~/.seshat/config.toml`. Optional — defaults apply if absent.

```toml
[digest]
ollama_model = "gemma4:26b"
output_dir = "~/Documents/papers/digest"
max_results = 10

[rag]
rag_dir = "~/.seshat/rag"

[chat]
provider = "ollama"              # "ollama" | "anthropic"
vault_path = "~/Documents/obsidian"
private_vault_dirs = ["private"] # vault subdirs only accessible to local model

[auth]
oauth_client_id = ""             # required for kb auth login
```

Env var overrides: `OLLAMA_MODEL`, `ANTHROPIC_MODEL`, `CHAT_PROVIDER`, `VAULT_PATH`.

To customise the agent's behaviour, create `~/.seshat/system_prompt.md`.

---

## Privacy model

Vault notes under directories listed in `private_vault_dirs` are private — visible to the local Ollama model only. Cloud providers (Anthropic) skip those chunks entirely and cannot read those files via `read_file`.

```
vault/
├── private/    ← local model only
│   └── journal/
└── research/   ← cloud + local
```

---

## Usage

All commands require the `uv run` prefix (entry points live in `.venv/bin/`). Alternatively, activate the venv once with `source .venv/bin/activate`.

### Vault chat — conversational KB agent

The primary way to interact with the knowledge base.

```bash
uv run vault-chat                           # uses provider from config
uv run vault-chat ~/path/to/vault           # override vault path
CHAT_PROVIDER=anthropic uv run vault-chat   # use Anthropic for this session
```

Available tools the agent can call:

| Tool | What it does |
|---|---|
| `retrieve_papers` | Semantic search over indexed papers |
| `search_notes` | Semantic search over vault notes |
| `read_file` | Read a specific vault file in full |
| `add_document` | Add a paper by arXiv URL or local PDF |
| `remove_document` | Two-step remove: preview then confirm; optionally deletes the local file |
| `list_papers` | List all indexed papers |
| `kb_stats` | Paper, note, and chunk counts |
| `index_vault` | Build or rebuild the vault index |
| `refresh_vault` | Incremental vault sync (new/changed/deleted files) |

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

### Knowledge base CLI (`kb`)

For scripted use, batch imports, and initial setup.

```bash
# Vault indexing
uv run kb index-vault
uv run kb index-vault --vault-path ~/path/to/vault --force   # clear and rebuild
uv run kb refresh-vault

# Add a paper by arXiv URL
uv run kb add https://arxiv.org/abs/2406.04093
uv run kb add https://arxiv.org/abs/2406.04093 --score 9 --track "Track 1"
uv run kb add https://arxiv.org/abs/2406.04093 --full-text   # store full PDF text

# Add a local PDF
uv run kb add paper.pdf --visibility private
uv run kb add paper.pdf --visibility private --full-text

# Override the provider used for summary generation
uv run kb add https://arxiv.org/abs/2406.04093 --provider anthropic

# Bulk-import previous digest files (no LLM call — reuses existing summaries)
uv run kb add-digest ~/Documents/papers/digest/
uv run kb add-digest ~/Documents/papers/digest/ --min-score 7

# Inspect
uv run kb list
uv run kb stats

# Remove (shows preview and asks for confirmation; optionally deletes the source file)
uv run kb remove https://arxiv.org/abs/2406.04093
uv run kb remove file:///path/to/paper.pdf --delete-file
```

### Weekly digest

```bash
uv run run-digest
```

Fetches papers from arXiv, scores them against the relevance prompt, writes a tiered Markdown digest to `output_dir`, and automatically indexes papers with score ≥ 9 into the knowledge base. No interaction needed — designed to run on a schedule (see [Scheduling](#scheduling)).

### Anthropic authentication

**Option 1 — API key (recommended):**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

**Option 2 — claude.ai OAuth** (one-time browser login, token persists in `~/.seshat/auth.json`):
```bash
uv run kb auth login    # opens browser, saves token
uv run kb auth status   # show active auth method
```

OAuth requires `oauth_client_id` set in `config.toml`.

### PDF conversion (standalone)

```bash
uv run convert-pdf --input https://arxiv.org/abs/2301.07041
uv run convert-pdf --input paper.pdf --output-dir ./output
```

Converts arXiv PDFs (by URL or local file) to Markdown using marker-pdf.

---

## Scheduling (macOS launchd)

See [docs/LAUNCHD_SETUP.md](docs/LAUNCHD_SETUP.md). The digest runs weekly (Monday 02:00) via `run_digest.sh` as the launchd target.

---

## Requirements

- [uv](https://github.com/astral-sh/uv)
- Python ≥ 3.12
- [Ollama](https://ollama.com) with the configured model pulled (for local inference)
- Anthropic API key or OAuth client ID (for cloud inference only)
