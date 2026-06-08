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
│   ├── rag_cli.py              # paper-rag CLI
│   ├── prompt_filter_score.md  # Prompt template for paper scoring
│   └── prompts/
│       └── paper_summary.md    # Prompt template for paper summarisation
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
oauth_client_id = ""         # required for paper-rag auth login
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

### Weekly digest

```bash
run-digest
```

### Vault chat

```bash
vault-chat
vault-chat ~/path/to/vault      # override vault path for this session
```

On startup, vault-chat refreshes the vault index (new/changed/deleted files). Index the vault first if you haven't already:

```bash
paper-rag index-vault
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
paper-rag index-vault
paper-rag index-vault --vault-path ~/path/to/vault --force   # full re-index
paper-rag refresh-vault

# Add papers
paper-rag add https://arxiv.org/abs/2406.04093 --score 9 --track "Track 1"
paper-rag add paper.pdf --provider anthropic

# Search and inspect
paper-rag query "sparse autoencoders" --score-min 8
paper-rag list
paper-rag stats

# Remove by ID (use vault-chat for fuzzy removal by description)
paper-rag remove 2406.04093
```

`--provider` accepts `anthropic` or an Ollama model name. PDFs are sent directly to the model — no conversion step, provided the model supports document input.

### Anthropic authentication

**Option 1 — API key:**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

**Option 2 — claude.ai OAuth** (browser flow, persists across sessions):
```bash
paper-rag auth login     # opens browser, saves token to ~/.paper_digest/auth.json
paper-rag auth status    # show current auth method
```

OAuth login requires `oauth_client_id` in `~/.paper_digest/config.toml` (or `ANTHROPIC_OAUTH_CLIENT_ID` env var). Confirm the client ID and endpoint URLs from [Anthropic's developer documentation](https://docs.anthropic.com).

### PDF conversion (standalone)

```bash
convert-pdf --input https://arxiv.org/abs/2301.07041
convert-pdf --input paper.pdf --output-dir ./output
```

## Scheduling (macOS launchd)

See [LAUNCHD_SETUP.md](LAUNCHD_SETUP.md). The shell wrapper [run_digest.sh](run_digest.sh) is the launchd target.

## Requirements

- [uv](https://github.com/astral-sh/uv)
- [Ollama](https://ollama.com) (for Ollama provider)
- Python ≥ 3.12
