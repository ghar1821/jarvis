# Design Document

## Purpose

A personal research tool that:

1. Fetches papers from arXiv weekly and scores them with an LLM
2. Writes a tiered Markdown digest (Must-Read / Worth Reading / Skim)
3. Indexes papers, vault notes, PDF annotations, and past chat exchanges into a local knowledge base
4. Provides a conversational agent for querying and managing the knowledge base, with persistent sessions and user-defined skills
5. Runs the recurring work (digest, PDF inbox, vault refresh) in one supervised background daemon (`jarvis-sync`)

---

## Repository layout

```
├── jarvis/                          # Python package
│   ├── core/                        # Shared infrastructure
│   │   ├── config.py                # Central configuration (incl. tomlkit write-back)
│   │   ├── errors.py                # Domain exceptions + retry decorator
│   │   └── llm.py                   # LLM provider abstraction
│   │
│   ├── digest/                      # Automated weekly digest
│   │   ├── arxiv/                   # arXiv paper fetching
│   │   │   ├── fetch.py             # Fetch papers via the `arxiv` package
│   │   │   └── convert.py           # Parse arXiv URLs + download PDFs
│   │   ├── biorxiv/                 # bioRxiv paper fetching
│   │   │   └── fetch.py             # Category + keyword search over the details API
│   │   ├── pipeline/
│   │   │   ├── run.py               # Entry point: orchestrates full digest run
│   │   │   ├── score.py             # LLM-based paper scoring
│   │   │   ├── format.py            # Markdown digest renderer
│   │   │   └── prompts/
│   │   │       └── prompt_filter_score.md
│   │   └── import_digest.py         # `kb add-digest` implementation
│   │
│   ├── kb/                          # Knowledge base management
│   │   ├── store.py                 # Vector store operations (LangChain + ChromaDB)
│   │   ├── cli.py                   # `kb` CLI entry point
│   │   ├── convert.py               # PDF → Markdown (pymupdf4llm) + `convert-pdf` CLI
│   │   ├── annotations.py           # PDF highlight/typed-note extraction (PyMuPDF)
│   │   ├── images.py                # PDF figure extraction (PyMuPDF)
│   │   ├── metadata.py              # Title/authors/DOI inference for local PDFs
│   │   └── prompts/
│   │       └── paper_summary.md
│   │
│   ├── sync/                        # Background sync daemon
│   │   ├── daemon.py                # `jarvis-sync` entry point
│   │   └── status.py                # `kb sync-status` implementation
│   │
│   ├── chat/
│   │   ├── chat.py                  # `vault-chat` entry point (KB agent)
│   │   ├── sessions.py              # Persistent sessions: save/resume/pin/prune/compact
│   │   └── skills.py                # User-defined skills (list + read)
│   │
│   └── webapp/
│       ├── app.py                   # FastAPI application (routes, SSE stream, session state)
│       ├── index.html               # Chat UI page
│       ├── static/                  # style.css + app.js (vanilla JS, no build step)
│       └── run.py                   # `webapp` entry point (uvicorn launcher)
│
├── tests/                           # See docs/TESTING.md
│
├── docs/
│   ├── DESIGN.md                    # This file
│   ├── TESTING.md
│   ├── TODO.md
│   ├── ROADMAP.md
│   └── CHANGELOG.md
└── pyproject.toml
```

### Module responsibilities at a glance

| Module | Concern |
|---|---|
| `jarvis/digest/arxiv/` | Fetching papers from the arXiv API; downloading PDFs |
| `jarvis/digest/biorxiv/` | Fetching recent preprints from the bioRxiv API (category + keyword) |
| `jarvis/digest/pipeline/` | Weekly automated digest: scoring, formatting, orchestration |
| `jarvis/digest/import_digest.py` | `kb add-digest`: bulk-import papers from digest Markdown files |
| `jarvis/kb/` | Knowledge base: vector store, PDF conversion, annotation + figure extraction, the `kb` CLI |
| `jarvis/sync/daemon.py` | `jarvis-sync`: scheduled digest (+ 6-hourly catch-up), periodic PDF inbox scan, periodic vault refresh |
| `jarvis/sync/status.py` | `kb sync-status`: reports daemon liveness and per-job outcomes |
| `jarvis/chat/chat.py` | Conversational agent: query and manage via natural language |
| `jarvis/chat/sessions.py` | Persistent chat sessions: persistence, privacy flag, retention, compaction, rename |
| `jarvis/chat/skills.py` | User-defined skills: discovery and on-demand loading |
| `jarvis/webapp/` | Browser-based chat UI: FastAPI routes, SSE stream, session state, frontend |
| `jarvis/core/llm.py` | Shared: LLM provider abstraction (Ollama + Anthropic) |
| `jarvis/core/config.py` | Shared: central configuration |
| `jarvis/core/errors.py` | Shared: domain exceptions and retry decorator |

---

## Dependencies

| Package | Purpose |
|---|---|
| `langchain-chroma` | LangChain wrapper over ChromaDB vector store |
| `langchain-huggingface` | HuggingFace embeddings via LangChain |
| `langchain-text-splitters` | `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter` for section-aware chunking |
| `chromadb` | Underlying persistent vector store (SQLite + HNSW) |
| `sentence-transformers` | Local embedding model (`BAAI/bge-small-en-v1.5`) and cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L6-v2`) |
| `rank-bm25` | Sparse (BM25) ranking for hybrid retrieval, fused with dense results by reciprocal rank fusion (`[rag] hybrid`) |
| `anthropic` | Anthropic Claude API client |
| `ollama` | Client for the local Ollama server (chat, tools, vision) |
| `arxiv` | arXiv API client with built-in paging, per-page retries, and courtesy delay |
| `pymupdf4llm` | Fast rule-based PDF-to-Markdown conversion (no ML models) |
| `pymupdf` | PDF annotation extraction (`page.annots()`, quad geometry) and figure extraction (`page.get_images`) |
| `apscheduler` | Cron/interval scheduling inside the `jarvis-sync` daemon (all four jobs, including the periodic PDF inbox scan) |
| `tomlkit` | Comment-preserving `config.toml` write-back (settings persistence) |
| `requests` | HTTP client (arXiv PDF download, bioRxiv API, Ollama health check) |
| `fastapi` | Web framework for the browser UI (`jarvis/webapp/`) |
| `uvicorn` | ASGI server that runs the FastAPI app |

---

## CLI entry points

All require `uv run` prefix unless the venv is activated (`source .venv/bin/activate`).

| Command | Module | Purpose |
|---|---|---|
| `uv run run-digest` | `jarvis.digest.pipeline.run:main` | Run the weekly digest pipeline once |
| `uv run jarvis-sync` | `jarvis.sync.daemon:main` | Start the background sync daemon (foreground; run directly, no service manager) |
| `uv run vault-chat` | `jarvis.chat.chat:main` | Start the KB agent chat session |
| `uv run kb` | `jarvis.kb.cli:main` | Manage the knowledge base (CLI) |
| `uv run convert-pdf` | `jarvis.kb.convert:main` | Convert a PDF to Markdown (standalone) |
| `uv run webapp` | `jarvis.webapp.run:main` | Start the web UI at `http://127.0.0.1:8080` |

---

## Runtime file locations

| Path | Contents |
|---|---|
| `~/.jarvis/config.toml` | User configuration (mode 0600 after any settings write-back) |
| `~/.jarvis/rag/` | ChromaDB persistent store (+ `.write.lock` for cross-process writes) |
| `~/.jarvis/state/sync_status.json` | `jarvis-sync` daemon/job status (read by `kb sync-status`) |
| `~/.jarvis/sessions/` | Persistent chat sessions, one JSON file each (dir 0700, files 0600) |
| `~/.jarvis/skills/` | User-defined skill `.md` files (configurable via `skills_dir`) |
| `~/.jarvis/logs/sync.log` | `jarvis-sync` daemon log (written directly by the daemon; also echoed to stderr) |
| `~/.jarvis/logs/chat.log` | Chat-tool failures — full exception + traceback for every caught tool error, shared by `vault-chat` and the webapp (file only, not echoed to the terminal) |
| `~/Documents/papers/digest/` | Weekly digest `.md` output files (configurable) |

---

## Configuration — `jarvis/core/config.py`

Resolution order (later wins): defaults → `~/.jarvis/config.toml` → env vars.

| Field | Default | Env var | Description |
|---|---|---|---|
| `anthropic_model` | `claude-sonnet-4-6` | `ANTHROPIC_MODEL` | Anthropic model |
| `output_dir` | `~/Documents/papers/digest` | — | Digest output directory |
| `max_results` | `10` | — | Max papers per digest |
| `arxiv_cats` | 6 categories | — | `[(category, limit), ...]` (TOML key `arxiv_categories`) |
| `rag_dir` | `~/.jarvis/rag` | — | ChromaDB storage path |
| `embed_model` | `BAAI/bge-small-en-v1.5` | — | Embedding model (changing it requires `kb reindex`) |
| `query_prefix` | BGE search instruction | — | Prepended to queries only (BGE-style asymmetric prefix); `""` disables |
| `chunk_size` | `1024` | — | Characters per chunk |
| `chunk_overlap` | `128` | — | Overlap between chunks |
| `rerank_model` | `cross-encoder/ms-marco-MiniLM-L6-v2` | — | Cross-encoder reranker; `""` disables re-ranking |
| `rerank_top_n` | `25` | — | Candidates fetched before re-ranking down to `n_results` |
| `hybrid` | `True` | — | Hybrid dense+BM25 retrieval fused by reciprocal-rank fusion; `False` reproduces the pre-hybrid dense-only pipeline exactly |
| `figure_captions` | `False` | — | Caption PDF figures at ingest (needs a vision model). Off by default — each figure costs a vision call; opt in per document via `kb add --figures` or the chat tool's `with_figures` |
| `figure_max_per_doc` | `20` | — | Cap on figures captioned per document |
| `figure_min_pixels` | `40000` | — | Skip embedded images smaller than this (logos, rules) |
| `biorxiv_cats` | `[("bioinformatics", 100)]` | — | bioRxiv server-side categories (TOML key `biorxiv_categories`) |
| `biorxiv_keywords` | `[("cytometry", 50), ...]` | — | bioRxiv client-side keyword filters (TOML key `biorxiv_keywords`) |
| `biorxiv_days` | `7` | — | Recent-preprint window for bioRxiv fetches |
| `provider` | `ollama` | `CHAT_PROVIDER` | Active LLM provider (`"ollama"` \| `"anthropic"`) |
| `ollama_model` | `qwen3-vl:30b` | `OLLAMA_MODEL` | Ollama model tag (needs tool calling + vision for full functionality) |
| `vault_path` | `~/vault` | `VAULT_PATH` | Obsidian vault root |
| `private_vault_dirs` | `["private"]` | — | Top-level vault folders treated as private |
| `skills_dir` | `~/.jarvis/skills` | — | User-defined skill files; missing folder = feature off |
| `response_style` | `""` | — | Free-text style instruction appended to the system prompt |
| `compact_after_tokens` | `12000` | — | Session compaction threshold (estimated context tokens) |
| `compact_keep_exchanges` | `6` | — | Recent turns kept verbatim when compacting |
| `pdf_watch_dir` | `None` | `PDF_WATCH_DIR` | PDF inbox scanned periodically by `jarvis-sync`; `None` disables the scan |
| `pdf_watch_minutes` | `30` | — | Minutes between PDF inbox scans (≥ 1); inbox latency is at most one interval |
| `vault_refresh_minutes` | `30` | — | Daemon vault refresh interval |
| `digest_day` | `mon` | — | Digest day of week (APScheduler token) |
| `digest_hour` | `5` | — | Digest hour (0–23) |
| `anthropic_api_key` | `""` | `ANTHROPIC_API_KEY` | Anthropic API key (alternative to env var) |

Two config helpers matter beyond `load_config()`:

- **`set_config_value(section, key, value)`** — persists one key back into `config.toml` via tomlkit, preserving every other key, comment, and the formatting. The write is atomic (temp file + `os.replace`) and leaves the file mode 0600 (it can hold the API key). Used by the webapp settings endpoint.
- **`warn_if_config_readable()`** — prints a loud warning at `jarvis-sync` and `vault-chat` startup when `config.toml` is group/world-readable. Fail visibly; never silently chmod the user's file.

---

## Knowledge base — `jarvis/kb/store.py`

Single LangChain + ChromaDB collection (`knowledge_base`).

### Document schema

```
page_content : str   — chunked text (embedded)
metadata:
  date_added  : str  — ISO timestamp
  doc_type    : str  — "paper" | "note" | "chat" (past chat exchanges) |
                       "digest" (indexed weekly digest .md files)
  visibility  : str  — "public" | "private" (papers are always public)
  source      : str  — arXiv/DOI URL for papers; "local" for vault .md notes;
                       file:/// URI for local PDFs and digest files;
                       "session:<id>" for chat exchanges
  title       : str  — display title (optional)
  authors     : str  — papers only (optional)
  doi         : str  — papers only (optional); regex/LLM-inferred for local PDFs,
                       passed through from the arXiv API result when present
  meta_inferred: bool — title/authors/doi came from auto-inference and has not
                       been human-verified yet (papers only, optional)
  score       : int  — relevance 0–10, papers only (optional)
  track       : str  — research track, papers only (optional)
  storage_mode: str  — "summary" | "full_text" (optional)
  file_path   : str  — vault-relative path for .md notes; absolute path for local PDFs (optional)
  content_hash: str  — SHA-256 of the full file, used for change detection
  chunk_index : int  — 0-based position of this chunk within its source document
  section     : str  — markdown header breadcrumb ("H1 › H2"); "" when the chunk has no heading
  modified_at : str  — ISO mtime of the source file, vault notes only (optional)

PDF annotation and figure chunks (see Annotations below) additionally carry:
  annotation_kind : str — "highlight" | "comment" | "figure" (absent on body chunks)
  page            : int — 1-indexed PDF page the annotation/figure came from
  note_text       : str — the user's typed comment, "" if none (always "" for figures)
```

Annotation and figure chunks share `source`/`file_path`/`doc_type`/`visibility` with the parent PDF's body chunks, so every existing delete and re-ingest path sweeps them along automatically — no separate cleanup logic. Figure chunks store a vision-model caption as `[FIGURE p.N] <caption>`.

**`doc_type` rules:**
- arXiv URL → always `"paper"`
- Local PDF → user must specify `"paper"` or `"note"` via `--doc-type`
- Vault `.md` files → always `"note"`
- Chat exchanges (indexed per turn by `jarvis/chat/sessions.py`) → `"chat"`
- Weekly digest `.md` files (indexed by the digest pipeline) → `"digest"`. Deliberately not `"note"`: `refresh_vault` deletes note entries whose vault-relative path no longer exists, and a digest's absolute path would look exactly like that and get wiped on the next sync. Searched by `retrieve_papers` alongside papers (`doc_type=["paper", "digest"]`).

**`storage_mode` rules:**
- `"note"` documents are always `full_text`
- `"paper"` documents default to `"summary"` (LLM-generated ~1000-word summary, 1–2 chunks); `--full-text` stores all PDF chunks

### Privacy model

| | Ollama (local) | Anthropic (cloud) |
|---|---|---|
| `"public"` | ✓ | ✓ |
| `"private"` | ✓ | Raises `PrivacyError`; tool loop terminates immediately |

When a cloud provider query matches only private content, or tries to read a file in a private vault directory, `PrivacyError` is raised from the tool implementation. `agentic_turn()` catches it, removes the orphaned assistant message from `messages` to keep conversation history valid, and returns the error string directly to the user — no further LLM calls are made. This is a prompt-injection defence: private notes may contain adversarial content that must never reach a cloud model.

**Papers are always public (invariant).** Only notes — vault `.md` files and note-type PDFs — can be private. Enforced at add time in `kb add` and the `add_document` tool; this is what makes the cloud summary path (which uploads the PDF to Anthropic) safe by construction rather than by a per-path gate. `kb stats` warns about legacy private papers added before the invariant existed.

**One classification policy.** `get_visibility(file_path, vault_root)` is the single rule that maps a path to a visibility, and it is used by *both* indexing and `read_file`. `read_file` classifies the **resolved** path — checking the caller-supplied relative path instead would let a symlink placed in a public folder reach into `private/`.

**Private dirs are top-level-only by contract.** `get_visibility` checks only the first path component under the vault root against `private_vault_dirs`. A folder named `private/` nested deeper (e.g. `research/private/`) is **not** recognised as private.

**Visibility is re-checked on refresh.** `refresh_vault` re-derives each unchanged note's classification, so editing `private_vault_dirs` in config reclassifies already-indexed chunks (`update_visibility()`, metadata-only, no re-embedding). Without this, a note moved behind the private rule would stay visible to the cloud provider until its content next changed.

**Mixed results caveat (`_search_notes`).** When a cloud query matches both public and private notes, the public results are returned along with a static caveat line telling the model (and user) that some matches were excluded. The caveat text is fixed app text — it carries no private content. Only when a query matches *exclusively* private content does the hard `PrivacyError` stop fire.

**Session privacy** is described under Sessions below: the first private retrieval flags the session private permanently, chat exchanges are indexed as `doc_type="chat"` with the session's visibility, and private sessions cannot be resumed under a cloud provider.

Files under top-level `private_vault_dirs` folders → `"private"`. All papers → `"public"`.

### Key functions

| Function | Description |
|---|---|
| `get_store()` | Process-wide Chroma singleton; tags the collection with `embed_model` and enforces the mismatch guard |
| `build_embeddings(model_name, query_prefix)` | Construct a normalised HuggingFace embedding model with an optional query-side prefix |
| `add_paper(paper, summary, score, track)` | Add paper (always public); idempotent by source URL; content includes an authors line so author-name queries can match |
| `add_papers_batch(entries)` | Batch add from digest; no extra LLM call |
| `add_texts(content, doc_type, visibility, source, ..., embed_header="")` | Low-level: section-aware chunk and add; `embed_header` is prepended to the embedded text of every chunk (metadata untouched) |
| `add_annotations(pdf_path, doc_type, visibility, source, ...)` | Extract highlights/typed notes from a PDF and index each as its own chunk (see Annotations) |
| `search(query, n_results, visibility, doc_type, annotation_kind, rerank=True)` | Hybrid (dense+BM25, gated by `[rag] hybrid`) or dense-only search with filters, then optional cross-encoder re-ranking; `doc_type` accepts one type or a list (`$in` filter, e.g. `["paper", "digest"]`); raises `KBCorruptionError` on a stale-id failure |
| `search_with_privacy_check(query, provider, ...)` | Provider-aware; returns `(results, has_private_hits)` |
| `delete_by_metadata(key, value)` | Delete all chunks matching key=value |
| `update_paper_metadata(source, title, authors, doi)` | Metadata-only correction of a paper's title/authors/doi; clears `meta_inferred` |
| `count_unverified_papers()` | Count papers whose metadata is still auto-inferred and unverified |
| `count()` · `count_unique_documents()` · `list_papers()` | Inspection |
| `update_file_path(source, new_path)` | Update `file_path` (and `source` URI) for all chunks matching a source; no re-embedding |
| `update_visibility(file_path, new_visibility)` | Metadata-only reclassification of a note's chunks; no re-embedding |
| `get_visibility(file_path, vault_root)` | The one visibility policy: derive public/private from the top-level folder |
| `index_vault_file(file_path, vault_root)` | Chunk and index one vault file |
| `refresh_vault(vault_root)` | Incremental sync (Phase 1: vault `.md` files incl. visibility re-check; Phase 2: local PDF notes incl. annotations); returns `(added, updated, deleted)` |

**Cross-process write lock (`_kb_write_lock`).** The daemon, webapp, and CLI all open the same ChromaDB `PersistentClient` directory, and Chroma's SQLite backend is not safe for concurrent multi-process writers. Every write path takes an advisory `flock` on `<rag_dir>/.write.lock` (re-entrant per thread, so composite operations like `refresh_vault` → `add_texts` don't self-deadlock). Reads stay unlocked — SQLite WAL handles concurrent readers.

### Annotations — `jarvis/kb/annotations.py`

macOS Preview and Foxit Reader both write standard ISO 32000 annotation objects into the page `/Annots` array on save, so one generic reader (PyMuPDF's `page.annots()`) covers both apps.

**Extraction mechanics:**
- Text markup (Highlight/Underline/Squiggly/StrikeOut) stores `/QuadPoints` marking the affected glyphs. The covered text is recovered by intersecting the quads with the page's words: every word whose bounding-box centre falls inside one of the annotation's line rects is kept, then joined in reading order (handles multi-line highlights).
- Typed notes live in the annotation's `/Contents`: standalone sticky notes (Text) and text boxes (FreeText) become `kind="comment"` chunks; a comment typed onto a highlight's popup is attached to that highlight's chunk as `note_text`.
- Indexed chunks are prefixed `[HIGHLIGHT p.N]` / `[USER NOTE p.N]` so retrieval (and the agent reading results) can tell user-marked passages from body prose.

**Supported vs not:**

| Annotation | Extracted? |
|---|---|
| Highlight — any colour | ✓ (extraction keys on annotation type, never colour) |
| Underline / squiggly / strikeout | ✓ (treated as highlights — all four mean "this passage matters") |
| Sticky note / text box (typed) | ✓ |
| Comment typed on a highlight | ✓ (as the highlight's `note_text`) |
| Freehand/handwritten drawing (Ink) | ✗ — stores stroke geometry, not text; would need handwriting OCR |

**Where it is wired in:** `kb add` (local PDFs and arXiv full-text), the chat `add_document` tool, `refresh_vault` Phase 2 (PDF notes), and the daemon's inbox ingest. Annotations are indexed *before* body conversion, so a scanned PDF whose body fails to convert still keeps its highlights. Re-saving a PDF with new annotations changes its byte hash, which triggers a full re-index through the existing change-detection paths.

### Figure captioning — `jarvis/kb/images.py` + `add_figures`

Text embeddings can't see images, so figures would be lost when a PDF is chunked as text. `extract_figures(pdf_path, max_figures, min_pixels)` pulls embedded raster images back out (PyMuPDF `page.get_images` + `doc.extract_image`), normalises each to PNG, deduplicates by xref, and drops images below `min_pixels` (logos, rules). It is a pure extraction function with no store/provider knowledge — the same shape as `annotations.py`.

`add_figures(...)` (in `store.py`) captions each figure via the active provider's `describe_image()` and indexes one chunk per figure — `page_content = "[FIGURE p.N] <caption>"`, `annotation_kind="figure"`, sharing `source`/`file_path`/`doc_type`/`visibility` with the parent PDF so deletes and re-ingests sweep figures along. Behaviour:

- **Off by default, opt-in per document:** `[rag] figure_captions` defaults to `false` (each figure costs a vision-model call). `add_figures` takes a keyword-only `enabled: bool | None = None` — `None` follows the config, `True` forces captioning for this one document. The opt-ins are `kb add --figures` and the chat tool's `with_figures=true`; the daemon inbox and `refresh_vault` stay config-gated (they pass nothing, so they no-op by default). `figure_max_per_doc` and `figure_min_pixels` bound cost/noise when captioning runs.
- **Reingest an existing paper with figures:** re-adding the *same source* with the duplicate override replaces the old entry — chat: `add_document(source, mode="full_text", with_figures=true)` → duplicate notice → re-call with `allow_duplicate=true`; CLI: `kb add <source> --figures --full-text` and answer `y`. The old chunks are deleted by source first (body, annotations, and figures share `source`, so the whole entry is swept); a same-title-but-different-source duplicate deletes nothing and adds a separate entry.
- **Privacy guard:** when `visibility == "private"` and the provider is `anthropic`, captioning is skipped entirely with a visible `⚠️` warning and no chunk is written — the images must never reach the cloud. `enabled=True` never overrides this guard, only the config kill-switch. Papers are always public, so paper figures caption under either provider.
- **Failure tolerance:** a per-figure `LLMError` warns and skips that one figure; the ingest never aborts.
- **Where it is wired in:** the same sites as annotations. The daemon and `refresh_vault` build the provider **lazily** — they peek with `extract_figures(..., max_figures=1)` first and only construct a provider when a PDF actually has a qualifying figure.

### Retrieval pipeline

A query flows through four stages, all local — no data leaves the machine:

1. **Chunking (index time).** `add_texts` splits content on markdown headers (`MarkdownHeaderTextSplitter`) and then by size (`RecursiveCharacterTextSplitter`). Each chunk stores its `chunk_index` and a `section` breadcrumb, and the breadcrumb is prepended to the embedded text so a query naming both the document topic and a section can match. Headerless content (paper summaries) passes through unchanged as a single unlabelled chunk. When the caller passes `embed_header` (papers only — the title, or `"{title} — {authors}"`), it is prepended to the embedded text of **every** chunk, not just the first, so an author-name or title-word query can match any chunk of a long paper.
2. **Hybrid retrieval.** Gated by `[rag] hybrid` (default `true`). When enabled, `_hybrid_search` fetches the ChromaDB candidate pool filtered by `visibility`/`doc_type` first, then ranks it two ways over that same filtered pool: dense (the query embedded with a BGE-style model, `embed_model`, prefixed by `query_prefix` on the query side only) and sparse (a BM25 index rebuilt fresh per query, via `rank-bm25`). The two rankings are fused by reciprocal rank fusion (`_reciprocal_rank_fusion`, `c=60`, identity by chunk id) — an id's score is the sum of `1/(c+rank)` across whichever ranking(s) it appears in. Because the sparse index and the dense query both operate on the already-filtered pool, privacy holds by construction — no id outside the filtered pool can ever be scored or returned. Setting `hybrid = false` skips straight to plain `similarity_search`, reproducing the pre-hybrid pipeline byte-for-byte.
3. **Re-ranking.** A cross-encoder (`rerank_model`) scores each `(query, chunk)` pair jointly and reorders the (dense or fused) candidates, returning the top `n_results`. Re-ranking is far more accurate than the bi-encoder's independent embeddings at deciding which chunk is actually most relevant. It runs **after** the visibility filter, so it never widens what a cloud provider can see; set `rerank_model = ""` to disable it.
4. **Corruption detection.** If ChromaDB raises with `"Error finding id"` in the message — a stale HNSW reference to a chunk id that no longer exists — `search()` raises `KBCorruptionError` instead of a generic `RAGError`, naming `uv run kb reindex` as the fix (chunk texts are already stored, so nothing is lost). This is not retried automatically: retrying persistent corruption just hides it. `uv run kb doctor` diagnoses this proactively (open store → count → search-probe) without waiting for a real query to hit it; on a badly corrupted store even `count()` can hard-segfault the process (a Rust-side ChromaDB crash, uncatchable in Python) — `kb doctor` dying abruptly is itself the diagnosis, not a bug in the doctor command.

**Embedding-model guard.** ChromaDB records `embed_model` in the collection metadata when the collection is first created. `get_store()` compares that tag against the configured model and raises `RAGError` on any mismatch — including legacy collections created before the tag existed. This prevents silently comparing vectors from two incompatible embedding spaces. The fix is always `uv run kb reindex`, which re-embeds every stored chunk (no LLM calls, chunk texts are already stored) into a fresh collection and swaps it in atomically. `kb reindex` also migrates old paper chunks that predate the `embed_header` convention: it prepends `"{title} — {authors}"` to any `doc_type="paper"` body chunk that doesn't already start with its title, so author-name queries work against papers indexed before this migration too (idempotent — a chunk already carrying the header is left alone).

### Metadata inference — `jarvis/kb/metadata.py`

Local PDFs arrive with nothing but a filename, so `infer_pdf_metadata(pdf_path, provider)` reads the first ~2 pages and asks the active provider (one small `complete()` call) to extract a title and author list. A DOI is looked for with a regex (`10.\d{4,9}/\S+`) first — cheap and exact when printed on the page — and the LLM is only asked to guess one when the regex misses. Degrades to `{}` on any LLM failure: inference is best-effort, never fatal to the add.

`resolve_pdf_metadata(...)` is the policy every add path (`kb add`, chat `add_document`, daemon `ingest_pdf`) shares, applied in order: (1) explicit `--title`/`--authors`/`--doi` overrides always win, skipping inference entirely once all three are given; (2) the privacy guard — a private note's text must never reach a cloud provider, so inference is skipped under Anthropic with a visible warning, leaving the filename stem as the title; (3) automatic inference for whatever is still unset. Papers are always public, so inference always runs for papers regardless of provider. Fields inferred this way are flagged `meta_inferred: true`.

**Verified-metadata loop.** `kb set-meta <source> [--title] [--authors] [--doi]` and the matching `update_document_metadata` chat tool apply a human correction metadata-only (no re-embedding) and clear `meta_inferred`. Reminders surface the unverified count in three places: `kb stats`, the webapp header (`GET /info` → `unverified_count` → a dismissible banner), and one `vault-chat` startup line.

### Deferred retrieval improvements

These were designed but intentionally not built, to keep the retrieval stack simple. Each has a concrete trigger for revisiting so the decision has a paper trail. The `tests/test_retrieval_quality.py` golden set is the instrument that makes the triggers observable.

- **Better embedding/rerank models.** *Trigger:* hybrid BM25+RRF (above) isn't enough — the golden set's harder queries still regress. *Sketch:* both are drop-in config changes — `embed_model = "BAAI/bge-base-en-v1.5"` or `bge-large` (requires another `uv run kb reindex`) and `rerank_model = "BAAI/bge-reranker-v2-m3"`.
- **Multi-query expansion.** *Trigger:* evidence that pre-rerank recall@`rerank_top_n` is the bottleneck. *Why deferred:* needs an LLM call per search inside the currently LLM-free `store.py`, and the agentic chat loop already reformulates queries across tool calls.
- **MMR (diversity re-ranking).** *Trigger:* top results dominated by near-duplicate chunks of one document. *Why deferred:* conflicts with cross-encoder ordering; the cheaper first fix would be a per-source cap applied after re-ranking.
- **Score thresholds.** *Why deferred:* cosine scores are poorly calibrated and corpus-dependent, and the reranker already sinks irrelevant results. Revisit only if junk results demonstrably pollute answers.

---

## arXiv module — `jarvis/digest/arxiv/`

`fetch.py` uses the `arxiv` package (lukasschwab/arxiv.py) rather than hand-rolled Atom-feed parsing. The library's `Client` exists to work around the arXiv API's known flakiness: it pages requests, retries responses that come back empty despite HTTP 200, and enforces the 3-second courtesy delay arXiv's terms ask for. A single shared client is used so the courtesy delay applies across successive category fetches. This replaced a raw `requests` implementation whose failure mode was silent: the empty-feed-with-200 bug bypassed retries and produced a digest with 0 papers.

Retry layering (two levels, deliberately):
1. **Inside the library** — per-page retries for paging hiccups.
2. **`@with_retries(exceptions=(FetchError,))` on top** — whole-search failures. Library errors and connection problems are wrapped in `FetchError`; a fully empty result set is *also* raised as `FetchError`, because a recent arXiv category is never genuinely empty, so an empty feed is treated as transient and retried. `with_retries` backs off exponentially (`backoff * 2**(attempt-1)`) with up to 25 % random jitter so repeated failures don't hammer a struggling service in lockstep.

- `fetch_arxiv(cat, max_results)` — batch fetch by category
- `fetch_arxiv_paper(arxiv_id)` — single paper by ID; `source` from the result's primary category
- `deduplicate(papers)` — remove duplicate titles

`convert.py`:
- `parse_arxiv_url(url)` — extract arXiv ID from any URL format
- `download_arxiv_pdf(arxiv_id, dest_dir)` — download PDF

PDF-to-Markdown conversion lives in `jarvis/kb/convert.py` (see below).

---

## bioRxiv module — `jarvis/digest/biorxiv/`

`fetch.py` pulls recent preprints from the bioRxiv details API
(`https://api.biorxiv.org/details/biorxiv/{start}/{end}/{cursor}/json`), which
returns 30 records per page walked by a numeric cursor. Records map to the same
paper dict shape as arXiv (`title`, `abstract`, `authors`, `link` =
`https://doi.org/{doi}`, `published` = date, `source`).

- `fetch_biorxiv(category, max_results, days=7)` — one server-side category over the last `days`. Only real bioRxiv categories (e.g. `bioinformatics`) filter server-side.
- `fetch_biorxiv_keywords(keywords, max_results, days=7)` — one uncategorised window, client-side case-insensitive match of any keyword against title+abstract, tagged `source = "bioRxiv:{keyword}"` and DOI-deduped (a paper matching two keywords appears once). Covers topics with no bioRxiv category (cytometry, spatial transcriptomics, scRNA-seq).

Both are wrapped in `@with_retries(exceptions=(FetchError,))`; an empty first page is treated as a transient failure and retried, mirroring the arXiv layering. The pipeline fetches bioRxiv after arXiv into the same `all_papers` list, so title-based `deduplicate()` and scoring run once over the combined set.

---

## PDF conversion — `jarvis/kb/convert.py`

`pdf_to_markdown(pdf_path) -> str` converts via **pymupdf4llm** — fast, rule-based extraction with no ML model downloads (replacing marker-pdf; orders of magnitude faster, at the accepted cost of lower fidelity on complex layouts and equations). Returning a string means no call site needs an intermediate `.md` file or temp-dir round-trip.

A PDF that yields no extractable text — typically a scanned/image-only PDF without an OCR text layer — raises `ConversionError` rather than silently indexing an empty document. There is no OCR fallback. Image extraction is not performed (nothing consumed it; `write_images=True` is the one-line reinstatement if ever wanted).

The standalone `convert-pdf` CLI (entry point `jarvis.kb.convert:main`) accepts a local path or arXiv URL and writes the Markdown to a file for manual use.

---

## Sync daemon — `jarvis/sync/daemon.py` (`jarvis-sync`)

One supervised long-running process, run directly with `uv run jarvis-sync` — it stays in the foreground; all scheduling lives inside the daemon, where catch-up can be handled properly. Restart-on-crash is not the daemon's concern: it's whatever keeps the process running (a terminal multiplexer, a process manager, or nothing at all).

**Process architecture:** one thread — an APScheduler `BlockingScheduler` running four jobs. There is no filesystem-event watcher and no worker thread/queue any more; everything is a scheduled job body.

| Job id | Trigger | What it does |
|---|---|---|
| `digest` | `CronTrigger(day_of_week=digest_day, hour=digest_hour)`, `coalesce=True`, `misfire_grace_time=3600` | Weekly digest; a run missed during sleep fires on wake |
| `digest_catchup` | `IntervalTrigger(hours=6)` + once at startup | Re-reads the persisted `last_success` stamp and runs the digest if a slot was missed while powered off |
| `vault_refresh` | `IntervalTrigger(minutes=vault_refresh_minutes)` + once at startup | Incremental Obsidian vault sync |
| `pdf_scan` | `IntervalTrigger(minutes=pdf_watch_minutes)` + once at startup; only registered when `pdf_watch_dir` is set | Sweep the PDF inbox and ingest new/changed PDFs serially |

**Status file** — `~/.jarvis/state/sync_status.json` records the daemon pid/start time and each job's `last_run` / `last_success` / `last_error` (written atomically). `kb sync-status` reads it, checks pid liveness, and tails the log. Every job body catches its own exceptions and records the outcome — one failing job never takes the daemon down. Fatal setup problems (invalid `[sync]` config, embedding-model mismatch) exit non-zero at startup with the reason logged to `~/.jarvis/logs/sync.log` and stderr.

**Digest catch-up** — `run_digest_catchup_job(trigger)` re-reads `jobs.digest.last_success` from the status file and calls `digest_is_overdue(trigger, last_success, now)`: if a scheduled fire time has passed since the stamp (machine was powered off across the slot), the digest runs immediately. It runs once at daemon start and then every 6 hours (job id `digest_catchup`) — a missed Monday fires within hours of the machine coming back, not at the next restart or the next Monday. On the very first start there is no baseline, so it waits for the next slot rather than surprise-running. The misfire grace handles sleep; the stamp + interval re-check handle power-off. **Double-fire guard:** the cron job and the catch-up job are separate APScheduler ids, so `max_instances=1` cannot stop them overlapping — `run_digest_job` acquires a module-level `threading.Lock` non-blocking at the top and returns early (with a log line) if another digest run holds it.

**Inbox semantics** — the watch dir is an *inbox, not a mirror*: removing a file never deletes its KB entry. Every `pdf_watch_minutes`, `run_pdf_scan_job` lists the inbox (`scan_watch_dir`, skipping dotfiles, `~$` lock files, and `.icloud` placeholders), checks each file is done being written (`wait_for_stable`, short parameters — a file still changing is left for the next cycle rather than waited on), and calls `ingest_pdf()` inline with a per-file try/except. `ingest_pdf()` indexes each PDF as a public full-text paper (annotations first, so a scanned PDF whose body can't convert still keeps its highlights), deduplicated by byte hash: unchanged file → skipped at zero LLM cost, which is what makes the periodic sweep idempotent; changed bytes (e.g. new annotations saved into the file) → old chunks replaced. Saving highlights repeatedly therefore costs at most one re-ingest per interval instead of one per save — that is the point of the periodic design. Title/authors/DOI are auto-inferred (`resolve_pdf_metadata`, see Knowledge base) — inbox PDFs are always public papers, so a provider is built unconditionally for this and reused for figure captioning rather than constructed twice (captioning itself is config-gated and off by default). New PDFs appear in the KB within one scan interval. The daemon refuses to start if `pdf_watch_dir` is set but missing — silently `mkdir`-ing a typo'd path would watch the wrong place.

**Why the cross-process write lock exists** — the daemon runs alongside the webapp and CLI, all writing to the same Chroma store; Chroma's SQLite backend is not multi-process-writer safe, hence the `flock`-based `_kb_write_lock` in `store.py`.

The daemon does not manage other daemons: if the provider is local and Ollama is down, the digest job fails fast (a `GET /api/tags` probe) with a pointer to the docs rather than auto-starting the server.

---

## Digest pipeline — `jarvis/digest/pipeline/`

`run.py` orchestrates:
```
make_provider(cfg.provider)     # whichever provider [chat] provider names
  ↓
fetch_arxiv() × 6 categories  →  ~490 paper dicts
deduplicate()                  →  ~400 unique papers
  ↓
filter_and_score(papers, provider, max_results, PROMPT_PATH)
  →  selected: [{index, track, score, slop, vetted, summary, why}]
  ↓
format_digest()  →  ~/Documents/papers/digest/digest-{date}.md
  ↓
index_digest_file()             →  the digest .md itself, doc_type="digest"
index_scored_papers()           →  score-tiered knowledge-base indexing
```

**Indexing tiers** (`index_scored_papers`):

| Score | What is indexed |
|---|---|
| `>= 9` | Full text via `ingest_full_text_paper`: dedup by source/title first → arXiv PDF downloaded to a temp dir → `pdf_to_markdown` → annotations + figures (config-gated) + chunked body with `{title, authors, doi, score, track, storage_mode: "full_text"}` and the title/authors embed header. **No `summarize()` call.** bioRxiv links (doi.org — no derivable PDF URL) and any download/conversion failure fall back to a summary entry built from the scoring run's own summary+why text, with a visible warning; one 404 never fails the digest job. Outcome counts (full-text / summary-fallback / skipped) are printed |
| `8 <= s < 9` | Summary entry via `add_papers_batch` — reuses the scoring run's summary+why, zero extra LLM calls |
| `< 8` | Not indexed per-paper; discoverable only through the indexed digest document |

**Digest document** (`index_digest_file`): the digest `.md` is indexed as `doc_type="digest"` with a `file://` source pointing at the file on disk, title `"Paper Digest — YYYY-MM-DD"`, and `storage_mode="full_text"` — so every paper it mentions (including the `< 8` tier) is searchable via `retrieve_papers`, which queries `doc_type=["paper", "digest"]`. See the `doc_type` rules above for why this is not `"note"`. There is no dedup against previously indexed digests: a manual same-day re-run of `run-digest` writes a second `digest-{date}.md` file and indexes it as a second digest document (each file gets its own `file://` source) — accepted because normal operation writes exactly one digest file per scheduled slot, and the catch-up job that could otherwise double-fire is lock-guarded (see `jarvis/sync/daemon.py`).

`score.py` — `filter_and_score()` sends all abstracts in one large prompt, parses JSON response. Under the local provider this requests a large `context_length`, which `OllamaProvider` passes through as `num_ctx`. The daemon's digest job additionally checks that Ollama is reachable (`GET /api/tags`) before starting.
`format.py` — `format_digest()` renders tiered Markdown digest (the "Generated HH:MM" line uses the actual run time).
`prompts/prompt_filter_score.md` — scoring rubric loaded at run time.

---

## LLM providers — `jarvis/core/llm.py`

`ChatProvider` protocol — four methods used across the system:

```python
complete(messages, max_tokens, context_length) -> str
# Single-shot completion. context_length sets Ollama's num_ctx; ignored by Anthropic.

summarize(title, source, max_tokens) -> str
# Dense paper summary. source: str (abstract) or Path (PDF).

agentic_turn(messages, tools, dispatch_fn, system) -> str
# Full tool-calling loop. Modifies messages in place.

describe_image(image_bytes, context) -> str
# Caption one PDF figure for indexing. context is the document title.
```

**`OllamaProvider`** talks to a local Ollama server (`http://localhost:11434`) via the `ollama` python client. One Ollama process keeps the model resident across the CLI, webapp, and sync daemon. Notes:

- Requires a model with tool-calling and (for figure captioning / vision summaries) vision support; the default is `qwen3-vl:30b`.
- Ollama returns tool arguments as a **mapping already** (not a JSON string like the OpenAI wire format), so they are used directly; a defensive `json.loads` covers the unlikely string case.
- The assistant message with tool calls is a pydantic object; it is normalised to a plain dict via `model_dump(exclude_none=True)` (`_message_to_dict`) so session history stays JSON-serialisable.
- Ollama honours a per-request context window, so `complete()` passes `context_length` through as `options={"num_ctx": ...}`.
- `summarize()` with a PDF path converts to Markdown locally first (`pdf_to_markdown`) — Ollama has no document input in this flow.
- `describe_image()` sends the image via `images=[bytes]`.
- `PrivacyError` from a tool pops the just-appended assistant message and returns immediately, same contract as the Anthropic adapter.

**`AnthropicProvider`** — API-key auth (`ANTHROPIC_API_KEY` env var, then `config.anthropic_api_key`); `summarize()` uploads the PDF as a base64 `document` block (safe because only public papers ever reach that path — see the invariant); `describe_image()` sends a base64 `image` block; tool results are bundled into a single `user` message of `tool_result` blocks.

A single `_FIGURE_CAPTION_PROMPT` is shared by both providers' `describe_image()`, so captions read the same regardless of model.

`make_provider(spec, model=None)` factory:
- `"anthropic"` → `AnthropicProvider` with config `anthropic_model` (or the override)
- `"ollama"` → `OllamaProvider` with config `ollama_model` (or the override)
- anything else → `ValueError`

---

## KB agent — `jarvis/chat/chat.py`

Single `run_session(vault, kb_only=True, session=None)` loop using `provider.agentic_turn()`. Every tool call is printed to the terminal (`→ tool_name(args)`) so the user sees each step. Each turn runs through the persistent `Session` (see Sessions below): compaction check, turn recorded, saved after the reply. CLI flags `--list-sessions` and `--resume <id>` list and resume stored sessions.

`build_system_prompt(kb_only=True, response_style="", skills=None)` loads the base prompt from `~/.jarvis/system_prompt.md` if present, otherwise uses the built-in default, then appends:
1. a knowledge-source instruction based on `kb_only`,
2. the list of available skills as `name: description` lines (when `skills` is non-empty),
3. the user's `response_style` preference (when set).

**Retrieved-data wrapping:** results from the retrieval tools (`retrieve_papers`, `search_notes`, `read_file`, `search_chat_history`) are wrapped in `BEGIN/END RETRIEVED DATA` markers, and the system prompt instructs the model to treat that text strictly as data, never as instructions. This is defence in depth against prompt injection from malicious documents — a mitigation, not a guarantee; the hard protections are the human-confirmation gate on deletions and the `PrivacyError` stops (see Security).

### Knowledge source modes

| Mode | `kb_only` | System prompt addendum | Tools list | How to enable |
|---|---|---|---|---|
| DB only (default) | `True` | LLM forbidden from drawing on training knowledge | `TOOLS` | `vault-chat` (no flag) |
| AI fallback | `False` | LLM searches KB first; may fall back to training knowledge after calling `use_own_knowledge` | `TOOLS + [USE_OWN_KNOWLEDGE_TOOL]` | `vault-chat --no-db-only` |

### Tools

| Tool | Concern | Cloud provider behaviour |
|---|---|---|
| `retrieve_papers` | Search indexed papers and digest documents (`doc_type=["paper", "digest"]`) | Public only; `PrivacyError` if query only matches private content |
| `search_notes` | Search vault notes | Public only; `PrivacyError` if query only matches private content; static caveat line appended when private matches were excluded from mixed results |
| `search_chat_history` | Search past conversations (`doc_type="chat"`), excluding the running session | Public sessions only; `PrivacyError` if query only matches private sessions |
| `read_file` | Read one vault file in full (after search identifies it) | `PrivacyError` for files whose resolved path is in `private_vault_dirs` |
| `read_skill` | Load a user-defined skill's full instructions; only in the tools list when skills exist | Any (skills are the user's own trusted files) |
| `add_document` | Add a paper or PDF; requires `doc_type` for local PDFs; two storage modes (see below); rejects private papers; title/authors/DOI auto-inferred for local PDFs unless overridden; `with_figures=true` opts this document into figure captioning; on a source/title duplicate returns an ask-the-user message unless `allow_duplicate=true` — a same-source re-add then **replaces** the old entry (old chunks deleted first), which is the reingest-with-figures path | Any |
| `update_file_path` | Update stored path for a local document without re-embedding | Any |
| `update_document_metadata` | Set verified title/authors/doi for a paper, metadata-only; clears `meta_inferred` | Any |
| `remove_document` | One call: immediately shows a **human** confirmation prompt; only that human answer executes the removal — database entry only, files on disk are never touched (see Security) | Any |
| `list_papers` | List indexed papers | Any |
| `kb_stats` | Document and chunk counts | Any |
| `index_vault` | Incremental vault sync (new/changed/deleted files). No `force` option — the destructive clean rebuild is CLI-only (`kb index-vault --force`) | Any |
| `use_own_knowledge` | Pseudo-tool called by the LLM before answering from training knowledge; dispatch returns an acknowledgement string; only included in the tools list when `kb_only=False` | Any |

The three retrieval tools additionally report whether they returned private content; under the local provider, the first private sighting flags the whole session as private (see Sessions).

### `add_document` storage modes

The tool exposes two modes; the LLM asks the user which to use if not specified:

| Mode | Flow | Chunks stored | Best for |
|---|---|---|---|
| `summary` (default for papers) | abstract/PDF → LLM generates ~1000-word summary → chunk | 1–2 | Most papers — fast, compact |
| `full_text` | download PDF → `pdf_to_markdown()` → chunk raw Markdown | Many | Papers the user wants to query at paragraph level |

Notes (`doc_type="note"`) are **always** stored as `full_text` regardless of what the caller requests. Both modes also run `add_annotations()` and `add_figures()` on local PDFs, so highlights/typed notes and captioned figures are indexed even when the body is stored as a summary.

For local PDFs, `doc_type` (`"paper"` or `"note"`), `visibility` (`"public"` / `"private"`, note-type only — private papers are rejected), and an optional `title` override are also accepted.

**Duplicate handling** — a paper can now arrive via arXiv and bioRxiv under different URLs, so `add_paper` and the manual-add paths skip on a normalised-title match as well as a source-URL match (`_title_exists` in `store.py`). The digest batch skips silently and reports `(added, skipped)`; `kb add` prompts `[y/N]`; the chat `add_document` tool returns an ask-the-user message and only proceeds when re-invoked with `allow_duplicate=true`. **Re-adding replaces:** once the user opts in, a SAME-SOURCE duplicate has its old chunks deleted by source before the re-add (annotations and figures share `source`, so the whole old entry is swept) — the store never holds two copies of one source. A same-title-but-different-source duplicate deletes nothing and becomes a separate entry. This replace path is how an already-indexed paper gets reingested with figure captions on.

### `remove_document` flow — one-shot human confirmation

1. The model calls `remove_document(source)` **once**. The tool immediately builds a preview — title, type, source, chunk count, and a line that always names the full local path (or "no local file") and states the fixed invariant: `"Database entry only — files on disk are never touched by jarvis: <path>"` — and hands it to a human via a `request_confirmation` channel: a `y/N` prompt in the terminal CLI, or a Confirm/Cancel dialog in the webapp (whose Confirm hits `/confirm-action`, entirely outside the LLM tool loop).
2. If the channel defers (webapp — returns `None`), the tool returns the preview plus an instruction not to call `remove_document` again for this request and not to claim the removal happened until the human confirms.
3. Only the human's out-of-band answer executes `execute_remove()`, which deletes the DB chunks and returns "No files were touched."

There is no model-controllable `confirmed` flag left to inject — the tool schema doesn't accept one. **File deletion has been removed from the codebase wholesale** (see Security): `execute_remove()` has no code path that can touch a file, so the scary case ("did it just delete my PDF?") is made impossible rather than better-worded.

---

## Sessions — `jarvis/chat/sessions.py`

One JSON file per session in `~/.jarvis/sessions/<id>.json` (dir 0700, files 0600, atomic writes). Each file holds **both** the provider wire-format `messages` (what the LLM sees) and the `display` list (what the human sees) — the two cannot be rebuilt from each other, and compaction deliberately shrinks only `messages`. Also stored: `pinned`, `private`, `provider`, `kb_only`, `turn_starts` (the `messages` index where each user turn began), and `indexed_exchanges` (how many exchange pairs are already in Chroma). Sessions are saved after every completed turn (crash-safe); empty sessions are never written.

**Retention / pinning** — `prune_sessions()` (run on every save) keeps the 50 most recently updated unpinned sessions; pinned sessions are exempt and uncounted, deleted only explicitly. Deleting a session removes both its file and its indexed `doc_type="chat"` chunks.

**Rename** — `rename_session(session_id, title)` trims the title, caps it at 120 characters, rejects an empty title, and rewrites the file atomically (same pattern as `set_pinned`). The webapp route also propagates the new title to the in-memory active session and, via `update_chat_title()` (metadata-only Chroma update), to the session's indexed chat chunks, so `search_chat_history` shows the new name.

**Chat-history indexing** — after each turn, new `(user, assistant)` exchange pairs are indexed as `doc_type="chat"` with `source="session:<id>"` and the session's visibility. Exchanges are built from the `display` list, so raw tool results are never indexed (they would duplicate document content already in the store). The `search_chat_history` tool searches these chunks via the same `search_with_privacy_check` machinery that protects notes, filtering out the running session.

**Privacy rules:**
- The first tool result containing private content flags the session private (`mark_private`) — the flag never clears, and any already-indexed public chunks for the session are deleted and re-indexed as private on the next save (fail-closed, even for pre-flip exchanges).
- `check_resume()` refuses to resume a private session under the cloud provider (it would replay private history to Anthropic) and refuses cross-provider resumes. The provider match is strict per name (only `anthropic` shares a family with itself), so a session recorded under the retired `llamacpp` provider refuses to resume under `ollama` rather than replaying an incompatible history.

**Compaction** — `maybe_compact()` runs before each turn. When `estimate_tokens(messages)` (serialised JSON length / 4 — crude but adequate) exceeds `compact_after_tokens`, everything before the last `compact_keep_exchanges` turns is summarised by the session's **own provider** (a private session is by definition local, so private history never goes to a cloud model for summarisation) and replaced with a two-message summary pair. The cut always lands on a `turn_starts` boundary, keeping `tool_use`/`tool_result` message structure intact. The `display` list is untouched — the UI always shows full history — and chat-history indexing is display-driven, so search is unaffected.

---

## Skills — `jarvis/chat/skills.py`

A skill is a plain `.md` file in `skills_dir` (default `~/.jarvis/skills`); the filename stem is the skill name and the first non-empty line (leading `#` stripped) is its one-line description. A missing or empty folder means the feature is off — the `read_skill` tool is not even advertised.

The design is **progressive disclosure**: the system prompt carries only `name: description` lines; the model calls `read_skill(name)` to pull in the full instructions when a task matches, so full skill text never occupies context until actually needed. Skill names coming from the LLM are treated as untrusted: separators/traversal sequences are rejected and the resolved path must stay inside `skills_dir`. Skills are the user's own local files — trusted content, never indexed into the vector store, outside the visibility model.

**Response style** — the related `[chat] response_style` free-text instruction is appended to the system prompt by `build_system_prompt()`. The webapp edits it live via the header ⋮ menu → modal (prefilled from `GET /settings`) and persists it via `set_config_value()` (tomlkit write-back, comments preserved, atomic, mode 0600).

---

## Web UI — `jarvis/webapp/`

Browser-based alternative to `vault-chat`. Runs on `http://127.0.0.1:8080` (localhost only).

**Stack:** FastAPI + Server-Sent Events + vanilla JS. No npm, no build step, no external JS dependencies. The frontend is `index.html` plus `static/style.css` and `static/app.js` (served via a `/static` mount).

**Hardening:** `TrustedHostMiddleware` allows only `127.0.0.1` / `localhost` Host headers — a DNS-rebinding page pointing an attacker domain at 127.0.0.1 gets refused. Session ids arriving over the network are validated against the generated alphabet before any file path is built (see Security).

**Session state:** a single in-memory dict shared across browser tabs, holding the active persistent `Session` object. Appropriate for a local single-user tool.

| State field | Default | Description |
|---|---|---|
| `session` | new `Session` at startup | The active persistent session (messages + display + privacy flag) |
| `provider` | set at startup | Active `ChatProvider` instance |
| `system` | set at startup | Active system prompt string |
| `kb_only` | `True` | Knowledge source mode; updated by `POST /config` |
| `response_style` | from config | Current style instruction; updated by `POST /settings` |
| `pending_action` | `None` | Deletion awaiting the user's Confirm/Cancel click |

**Routes:**

| Route | Purpose |
|---|---|
| `GET /` | Serves `index.html` |
| `GET /info` | `{provider, provider_kind, vault}` for the header |
| `GET /history` | The active session's display list for page-refresh restore |
| `GET /sessions` | `{active, sessions}` — stored session metadata for the sidebar (pinned first, newest first) |
| `POST /sessions/new` | Swap in a fresh session (the outgoing one is already persisted per turn) |
| `POST /sessions/{id}/resume` | Load and activate a stored session; 409 if `check_resume` refuses (private-under-cloud or provider-family mismatch) |
| `POST /sessions/{id}/pin` | `{pinned: bool}` — flip the pinned flag |
| `POST /sessions/{id}/rename` | `{title: str}` — rename; also updates the active session and indexed chat-chunk titles; 404 on unknown id or empty title |
| `DELETE /sessions/{id}` | Delete the session file and its indexed chat chunks; swaps in a fresh session if it was active |
| `POST /config` | `{kb_only: bool}`; updates the flag and rebuilds the system prompt |
| `GET /settings` | `{response_style}` |
| `POST /settings` | `{response_style}` — applies immediately and persists to `config.toml` via tomlkit |
| `POST /confirm-action` | `{confirmed: bool}` — the human decision point for a pending deletion; executes `execute_remove()` or cancels |
| `POST /chat` | Accepts `{message}`, streams SSE events |

**Request flow:**

```
Browser POST /chat
  → FastAPI builds the tools list (TOOLS [+ READ_SKILL_TOOL] [+ USE_OWN_KNOWLEDGE_TOOL])
  → runs maybe_compact(), records the turn in the session
  → spawns a background thread running provider.agentic_turn()
  → thread pushes {type: "tool"} events to a queue as each tool fires
  → a deletion request pushes {type: "confirm", description} and stores pending_action
  → async SSE generator drains the queue (50 ms poll) and yields data: lines
  → thread pushes {type: "reply", content, tool_calls, private} + sentinel when done;
    the session is saved (and its new exchanges indexed)
Browser reads the stream via fetch() + ReadableStream
  → tool events (regular): appended live to an open <details> box
  → tool event (use_own_knowledge): amber status badge inserted
  → confirm event: Confirm/Cancel dialog; Confirm POSTs /confirm-action
  → reply event: <details> collapses; reply bubble appears; private=true shows the
    session's private badge and greys it out for cloud resume
  → fetch errors render an error bubble instead of a stuck "Working..." placeholder
```

**SSE event types:** `tool` (name + arg summary), `confirm` (deletion description), `reply` (final text, tool-call log, session `private` flag). The tool-call arg summary elides overly long values with a shared middle-ellipsis helper (`truncate_middle` in `chat.py`, used by both the CLI and the webapp) so a `file:///` URI's filename stays visible.

**Theme + layout:** dark theme only (a single palette via CSS custom properties, no toggle). Chat bubbles cap line length at `min(80ch, 100%)`. The header carries a ⋮ menu → "Set response style…" modal (prefilled from `GET /settings`, Save posts, Cancel/Esc/backdrop closes). Each sidebar session row has a ✎ rename button (a `prompt()` → `POST /sessions/{id}/rename`) alongside pin and delete.

**DB only toggle:** A pill toggle in the input bar (on by default). Fires `POST /config` on change. When on, `kb_only=True` and the LLM is restricted to KB tools. When off, `kb_only=False` and `USE_OWN_KNOWLEDGE_TOOL` is added to the tools list.

**Why fetch + ReadableStream instead of EventSource:** `EventSource` only supports `GET`; sending the message body requires `POST`.

---

## Security

**Threat model.** A single-user application bound to loopback that nonetheless ingests untrusted content: arXiv PDFs, downloaded papers, and anything dropped into the inbox can contain adversarial text aimed at the LLM (prompt injection). The protections are layered — some are hard guarantees, some are mitigations, and the docs below say which is which.

**Human-in-the-loop for destructive actions (hard).** The model can *request* a deletion; only the human can *execute* it. `remove_document(source)` is a single call that never deletes anything itself — it immediately routes the preview through `request_confirmation`: a terminal `y/N` prompt in the CLI, a Confirm/Cancel dialog in the webapp whose Confirm hits `POST /confirm-action` outside the LLM tool loop. There is no model-controllable `confirmed` boolean left to inject — one round-trip was removed, zero security layers were.

**File deletion has been removed from the codebase wholesale (hard).** There is no code path anywhere in `jarvis/kb/store.py`, `jarvis/kb/cli.py`, or `jarvis/chat/chat.py` that unlinks a file — `delete_local_file()` and the `--delete-file` / `delete_file` params were deleted, not just disabled. `execute_remove()` only ever deletes ChromaDB chunks; the preview, the webapp dialog, and the system prompt all state the same invariant line verbatim: `"Database entry only — files on disk are never touched by jarvis: <path>"`, rendered visually distinct in the webapp dialog. This resolves what was previously an unclear-wording complaint by making the scary case impossible rather than better-worded.

**Stale confirm-dialog token guard.** The one-shot flow makes it possible for an older, unclicked confirmation dialog to still be on screen when a newer removal is requested. `request_confirmation` tags each pending action with a fresh UUID token; `POST /confirm-action` 409s unless the posted token matches the currently pending one, so an old dialog's Confirm button can never fire against a different (newer) action.

**Reduced LLM-facing surface.** The `index_vault` tool lost its destructive `force` option; the clean rebuild lives only in the human-driven CLI (`kb index-vault --force`).

**Retrieved-data delimiters (mitigation, not a guarantee).** Retrieval results are wrapped in `BEGIN/END RETRIEVED DATA` markers with a system-prompt rule to treat the content as data. This raises the bar against prompt injection from malicious documents, but a sufficiently persuasive payload can still influence the model — which is exactly why the deletion gate and `PrivacyError` stops do not rely on the model behaving.

**Network hardening.** `TrustedHostMiddleware` rejects non-localhost Host headers (DNS-rebinding defence); the server binds to 127.0.0.1 only. Session ids from the network are validated (`[0-9a-z-]{1,64}`) before any file path construction, blocking traversal. Skill names from the LLM get the same treatment (separator/traversal rejection + resolved-path containment).

**File permissions.** Config write-back and session files are 0600; the sessions directory is 0700. `jarvis-sync` and `vault-chat` warn at startup when `config.toml` (which can hold the API key) is group/world-readable — fail visibly rather than silently chmod.

The privacy model (papers-always-public invariant, `PrivacyError` hard stops, resolved-path classification) is part of the same defence and is documented under "Privacy model" above.

---

## Error handling — `jarvis/core/errors.py`

```
PaperDigestError
├── FetchError          arXiv API failures (incl. transient empty feeds)
├── LLMError            LLM failures
├── RAGError            Vector store failures
├── ConversionError     PDF→Markdown produced no usable text (scanned/image-only PDF)
├── AuthenticationError Missing credentials
└── PrivacyError        Cloud provider attempted to access private content
                        (caught by agentic_turn() for an immediate hard stop)
```

`@with_retries(max_attempts, backoff, exceptions)` — exponential backoff (`backoff * 2**(attempt-1)`) with up to 25 % random jitter; used in `arxiv/fetch.py` and `pipeline/score.py`.

---

## Data flows

### Background sync (`jarvis-sync`)

```
weekly cron slot → run_digest_job (non-blocking lock guards double-fire)
every 6 h (and at start) → run_digest_catchup_job: last_success stale? → run_digest_job
every pdf_watch_minutes (and at start) → run_pdf_scan_job:
  scan_watch_dir() → per file: wait_for_stable() (else leave for next cycle)
  → ingest_pdf(): hash dedup → add_annotations() → caption figures (config-gated, off by default)
  → pdf_to_markdown() → add_texts()
every vault_refresh_minutes (and at start) → refresh_vault()
```

### Weekly digest

```
arXiv (arxiv package) + bioRxiv (details API: categories + keywords)
  → fetch → deduplicate (title) → score → format digest (written to output_dir)
  → index_digest_file(): the digest .md itself → doc_type="digest", file:// source
  → index_scored_papers():
      score >= 9   → ingest_full_text_paper(): dedup → arXiv PDF → full text
                     (bioRxiv link / download failure → summary fallback, no LLM call)
      8 <= s < 9   → add_papers_batch() summary entries (no LLM call)
      score < 8    → per-paper: nothing (searchable via the digest document)
  dedup skips papers already present by source URL or title
```

### Vault chat turn

```
User message → maybe_compact() → provider.agentic_turn() → tool loop → reply
  → save_session(): write JSON, index new exchanges as doc_type="chat", prune old sessions

  retrieve_papers / search_notes  → search_with_privacy_check() → wrap in RETRIEVED DATA markers
  search_chat_history             → search_with_privacy_check(doc_type="chat") → wrap
  read_file                       → resolved-path privacy check → filesystem read → wrap
  read_skill                      → validated name → skill file content
  add_document (summary mode)     → resolve_pdf_metadata() (local PDFs) → provider.summarize() → add_texts() (+ annotations)
  add_document (full_text mode)   → download PDF → pdf_to_markdown() → chunk → add_texts() + add_annotations()
  add_document (note, local PDF)  → resolve_pdf_metadata() → pdf_to_markdown() → chunk → add_texts() + add_annotations()
  update_file_path                → update file_path + source URI in all matching chunks; no re-embedding
  update_document_metadata        → update_paper_metadata(); clears meta_inferred; no re-embedding
  remove_document                 → lookup metadata → build preview → request_confirmation → human decides → execute_remove()
  index_vault                     → refresh_vault() (incremental only)
  refresh_vault Phase 1           → compare hashes → index new/changed vault .md, delete removed,
                                    re-check visibility of unchanged notes (skips PDF notes)
  refresh_vault Phase 2           → check local PDF notes: warn if missing, re-index (with annotations) if hash changed
```
