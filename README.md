# paper_digest

arXiv paper digest for a computational biologist interested in cytometry, single-cell genomics, and AI/ML research. Fetches papers weekly, scores them with an LLM, and writes a ranked Markdown digest. High-scoring papers are automatically indexed into a local RAG database for retrieval and chat.

## Repository structure

```
paper_digest/
├── digest/
│   ├── config.py               # Central configuration (loads ~/.paper_digest/config.toml)
│   ├── errors.py               # Domain exceptions and retry decorator
│   ├── llm.py                  # LLM provider abstraction (Ollama and Anthropic adapters)
│   ├── fetch.py                # Fetch papers from arXiv
│   ├── score.py                # LLM-based filtering and scoring
│   ├── format.py               # Markdown digest formatter
│   ├── convert.py              # PDF-to-Markdown converter (standalone CLI)
│   ├── run.py                  # Main pipeline entry point
│   ├── rag.py                  # Local RAG database (ChromaDB + sentence-transformers)
│   ├── kb.py                   # kb CLI
│   └── prompts/
│       ├── prompt_filter_score.md  # Prompt template for paper scoring
│       └── paper_summary.md        # Prompt template for paper summarisation
├── vault_chat/
│   └── chat.py                 # Multi-turn vault + RAG chat
├── run_digest.sh               # Shell wrapper for launchd scheduling
└── pyproject.toml
```

## Configuration

All settings live in `~/.paper_digest/config.toml`. The file is optional — built-in defaults are used if it doesn't exist. Environment variables override TOML values.

```toml
[digest]
ollama_model = "gemma4:26b"
anthropic_model = "claude-sonnet-4-6"
output_dir = "~/Documents/papers/digest"
max_results = 10
# arxiv_categories = [["cs.LG", 150], ["cs.AI", 80], ...]  # override fetch targets

[rag]
rag_dir = "~/.paper_digest/rag"
embed_model = "all-MiniLM-L6-v2"

[chat]
provider = "ollama"          # "ollama" | "anthropic"
vault_path = "~/vault"

[auth]
oauth_client_id = ""         # required for kb auth login
```

Environment variables (`OLLAMA_MODEL`, `ANTHROPIC_MODEL`, `CHAT_PROVIDER`, `VAULT_PATH`) override the corresponding TOML values when set.

## Features

### Digest pipeline

Fetches recent papers from arXiv (cs.LG, cs.AI, cs.NE, cs.CV, cs.CL, cs.MA), scores them with the configured LLM, and writes a tiered Markdown digest to `output_dir`. Papers are scored 1–10 and grouped into Must-Read (≥ 9), Worth Reading (7–8), and Skim/Bookmark (5–6).

After each run, papers with score ≥ 9 are automatically added to the local RAG database. The scoring pipeline works with either Ollama or Anthropic — the same `provider` setting used for chat applies.

### LLM providers

Both the digest pipeline and vault chat use the same provider abstraction. Switching providers changes the model used for scoring, summarisation, and chat all at once.

| Provider | Config | Env var override |
|---|---|---|
| Ollama (local, default) | `[chat] provider = "ollama"` | `CHAT_PROVIDER=ollama` |
| Anthropic Claude | `[chat] provider = "anthropic"` | `CHAT_PROVIDER=anthropic` |

### RAG database

A persistent local vector database (ChromaDB at `~/.paper_digest/rag/`) with two collections:

- **Papers** — one document per paper, stored as an LLM-generated dense summary (up to 1000 words). Papers from digest runs reuse the existing scoring summary; manually added papers get a fresh summary generated at add time.
- **Vault notes** — Obsidian vault `.md` files, chunked and indexed for semantic search.

Embeddings use `all-MiniLM-L6-v2` (runs locally, no API call needed).

### Vault chat

A multi-turn agentic CLI with four tools:

| Tool | What it does |
|---|---|
| `read_file` | Read a specific vault file by relative path |
| `search_vault` | Semantic search over vault notes to discover relevant files |
| `retrieve_papers` | Semantic search over indexed papers |
| `remove_paper` | Remove a paper — LLM identifies it from a description, confirms before acting |

## Setup

```bash
uv sync
uv pip install -e .
```

Requires [Ollama](https://ollama.com) running locally with the configured model pulled (for Ollama provider).

## Usage

All commands below must be prefixed with `uv run` — they are entry points installed into the project's virtual environment, not system-wide executables:

```bash
uv run run-digest
uv run vault-chat --help
uv run kb --help
uv run convert-pdf --help
```

Alternatively, activate the venv once (`source .venv/bin/activate`) and then use the command names directly for the rest of the session.

### Weekly digest

```bash
uv run run-digest
```

### Vault chat

```bash
uv run vault-chat
uv run vault-chat ~/path/to/vault          # override vault path for this session
CHAT_PROVIDER=anthropic uv run vault-chat  # use Anthropic Claude for this session
```

The default provider comes from `[chat] provider` in `~/.paper_digest/config.toml`.

On startup, vault-chat refreshes the vault index (new/changed/deleted files). Index the vault first if you haven't already:

```bash
uv run kb index-vault
```

Example interactions:
```
You: what papers do we have on mechanistic interpretability?
You: remove the paper about SAE probing in biology
You: what are my notes on transformers?
```

### Paper RAG database

```bash
# Index / refresh vault
uv run kb index-vault
uv run kb index-vault --vault-path ~/path/to/vault --force   # full re-index
uv run kb refresh-vault

# Add papers
uv run kb add https://arxiv.org/abs/2406.04093 --score 9 --track "Track 1"
uv run kb add paper.pdf --provider anthropic

# Inspect
uv run kb list
uv run kb stats

# Remove by ID (use vault-chat for fuzzy removal by description)
uv run kb remove 2406.04093
```

`--provider` accepts `anthropic` or an Ollama model name. Defaults to the `[chat] provider` value from config — if that is already set to `anthropic`, you don't need to pass `--provider` explicitly. PDFs are sent directly to the model without a conversion step; the model must support document/vision input.

### Anthropic authentication

**Option 1 — API key:**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

**Option 2 — claude.ai OAuth** (browser flow, persists across sessions):
```bash
uv run kb auth login     # opens browser, saves token to ~/.paper_digest/auth.json
uv run kb auth status    # show current auth method
```

OAuth login requires `oauth_client_id` in `~/.paper_digest/config.toml` (or `ANTHROPIC_OAUTH_CLIENT_ID` env var). Confirm the client ID and endpoint URLs from [Anthropic's developer documentation](https://docs.anthropic.com).

### PDF conversion (standalone)

```bash
uv run convert-pdf --input https://arxiv.org/abs/2301.07041
uv run convert-pdf --input paper.pdf --output-dir ./output
```

## Scheduling (macOS launchd)

See [LAUNCHD_SETUP.md](LAUNCHD_SETUP.md). The shell wrapper [run_digest.sh](run_digest.sh) is the launchd target.

## Requirements

- [uv](https://github.com/astral-sh/uv)
- [Ollama](https://ollama.com) (for Ollama provider)
- Python ≥ 3.12
