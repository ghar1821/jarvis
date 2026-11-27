# Design Document

## Purpose

jarvis makes an Obsidian vault and a folder of research papers searchable by
any MCP client — Claude Desktop, Google Antigravity, Claude Code and others.
It is a **read-only knowledge server**:

1. It mirrors two folders the user manages — the vault and `papers_dir` —
   into a local search index.
2. It exposes that index over MCP as read-only tools.
3. The client is the assistant: it holds the conversation, runs the model,
   schedules tasks. jarvis makes no LLM calls and holds no API keys.

Three invariants shape everything below:

- **Files are the source of truth.** Every index entry comes from a file. Add a
  file to add a document, delete it to remove one; the index can always be
  rebuilt from disk.
- **jarvis only reads your folders.** It writes nothing outside `~/.jarvis/`
  and deletes no file anywhere (enforced by `tests/test_security.py`).
- **jarvis reads nothing outside those two folders.** Every MCP client sends
  what it reads to a cloud model, so everything inside the folders is treated
  as shareable and everything outside them is never touched. Private notes
  belong in a separate folder outside the vault.

---

## Architecture

```
  vault/ (public folders) ─┐                              ┌─ Claude Desktop
                           ├─► kb serve ──► Chroma index ◄─┤  jarvis-mcp (stdio)
  papers_dir/ (PDFs) ──────┘   (127.0.0.1)                 └─ Antigravity / any client
                               mirror sync
                               every sync_minutes
```

- **`kb serve`** — the one long-running process. Runs the Chroma server as a
  child and syncs both folders into it on a timer.
- **`jarvis-mcp`** — started and stopped by the MCP client over stdio. Connects
  to the Chroma server; never opens the index files itself.
- **`kb <command>`** — one-shot commands (`sync`, `rebuild`, `doctor`, …) that
  go through the server when it is up and open the files directly when it is
  down.

---

## Repository layout

```
├── jarvis/
│   ├── __init__.py          # switches off third-party telemetry before any import
│   ├── core/
│   │   ├── config.py        # ~/.jarvis/config.toml → Config; unknown-key notes
│   │   ├── errors.py        # domain exceptions
│   │   ├── files.py         # owner-only folders, and the "is it really inside?" check
│   │   └── logs.py          # ~/.jarvis/logs/jarvis.log for every library module
│   ├── kb/
│   │   ├── mirror.py        # the two folders → the index (sync, move detection, rebuild)
│   │   ├── store.py         # chunk storage and hybrid retrieval (LangChain + ChromaDB)
│   │   ├── metadata.py      # paper title/authors/DOI/arXiv ID without an LLM (regex, Crossref, arXiv)
│   │   ├── convert.py       # PDF → Markdown (pymupdf4llm)
│   │   ├── annotations.py   # PDF highlights and typed notes (PyMuPDF)
│   │   ├── frontmatter.py   # note YAML frontmatter → flat record metadata
│   │   ├── serve.py         # `kb serve` and `kb status`
│   │   └── cli.py           # the `kb` command
│   └── mcp/
│       └── server.py        # `jarvis-mcp`: the read-only MCP tools
├── tests/                   # see docs/TESTING.md
└── docs/
    ├── DESIGN.md            # this file
    ├── TESTING.md
    ├── CHANGELOG.md
    └── examples/
        └── paper-digest-task.md    # a weekly digest prompt that uses has_paper
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `mcp` | Official MCP Python SDK (2.x, `MCPServer`), stdio transport |
| `chromadb` | Vector store (SQLite + HNSW), run as a local server by `kb serve` |
| `langchain-chroma` | LangChain wrapper over ChromaDB |
| `langchain-huggingface` | HuggingFace embeddings via LangChain |
| `langchain-text-splitters` | Section-aware chunking (`MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter`) |
| `sentence-transformers` | Local embedding model (`BAAI/bge-small-en-v1.5`) and cross-encoder reranker |
| `rank-bm25` | Keyword (BM25) half of hybrid retrieval |
| `pymupdf4llm` | Fast rule-based PDF → Markdown |
| `pymupdf` | PDF annotations, embedded metadata, DOI extraction |
| `pyyaml` | Note frontmatter (`safe_load` only) |
| `arxiv` | arXiv API client, for the title and authors of a preprint whose PDF has no Crossref record |
| `requests` | Crossref lookups |

---

## Entry points

| Command | Module | Purpose |
|---|---|---|
| `uv run kb <command>` | `jarvis.kb.cli:main` | Everything a person runs: `serve`, `sync`, `rebuild`, `list`, `stats`, `schema`, `status`, `doctor` |
| `jarvis-mcp` | `jarvis.mcp.server:main` | The MCP server. Launched by the client, never by hand |

---

## Runtime file locations

Everything jarvis writes is under `~/.jarvis/`:

| Path | Contents |
|---|---|
| `~/.jarvis/config.toml` | User configuration (read only; never rewritten by jarvis) |
| `~/.jarvis/rag/` | The Chroma index (`rag_dir`), plus `.write.lock` for direct access. Dir 0700 |
| `~/.jarvis/state/serve_status.json` | `kb serve`'s pid, start time and last sync outcome (read by `kb status`) |
| `~/.jarvis/state/server_token` | The secret every request to the index server must carry. Created once by `kb serve`, file 0600 |
| `~/.jarvis/logs/serve.log` | `kb serve` log (also echoed to stderr) |
| `~/.jarvis/logs/chroma.log` | Output of the Chroma server child |
| `~/.jarvis/logs/jarvis.log` | Library warnings: the store, the mirror, metadata lookups, the MCP server |

`logs/` and `state/` are made 0700 like the index, because the logs name
papers and notes. The permissions are re-applied every time jarvis starts, so
folders made by older versions are fixed too.

Folders left by older versions (`~/.jarvis/sessions/`, `drafts/`, `prompts/`)
are not read and not deleted.

---

## Configuration

Resolution order (later wins): defaults → `~/.jarvis/config.toml` → env vars.

**The minimum config** is the two folders:

```toml
[kb]
vault_path = "~/Documents/obsidian"
papers_dir = "~/Documents/papers"
```

Jarvis runs with no file at all, but then the vault is `~/vault` and no papers
are indexed. `kb serve` and the client's launch of `jarvis-mcp` must see the
same values, which is why a file beats environment variables in practice.

| Key | Default | Env var | Description |
|---|---|---|---|
| `[kb] vault_path` | `~/vault` | `VAULT_PATH` | Obsidian vault root |
| `[kb] papers_dir` | unset | `PAPERS_DIR` | Folder of PDFs (recursive). Unset means no papers |
| `[kb] sync_minutes` | `30` | — | Minutes between `kb serve` sync passes (≥ 1) |
| `[rag] rag_dir` | `~/.jarvis/rag` | — | Index location |
| `[rag] embed_model` | `BAAI/bge-small-en-v1.5` | — | Embedding model (changing it needs `kb rebuild`) |
| `[rag] query_prefix` | BGE search instruction | — | Prepended to queries only; `""` disables |
| `[rag] chunk_size` / `chunk_overlap` | `1024` / `128` | — | Characters per chunk / overlap |
| `[rag] rerank_model` | `cross-encoder/ms-marco-MiniLM-L6-v2` | — | Cross-encoder reranker; `""` disables re-ranking |
| `[rag] rerank_top_n` | `25` | — | Candidates fetched before re-ranking |
| `[rag] hybrid` | `true` | — | Dense + BM25 fused by reciprocal rank fusion; `false` is dense-only. Must be a TOML boolean: a quoted `"false"` is an error, not true |
| `[rag] server_port` | `8321` | — | Chroma server port, always on 127.0.0.1 |

**Unknown settings are named, not guessed at.** Any section or key jarvis does
not read (a typo, or a setting from an older version such as `[chat]` or
`[sync]`) gets one line on stderr saying it is ignored. stderr, because stdout
is the MCP channel inside `jarvis-mcp`. The file is never rewritten.

**`private_vault_dirs` stops jarvis.** Older versions kept named vault folders
out of the index. This one indexes the whole vault, so a config that still
lists private folders (under `[kb]` or the old `[chat]`) raises `ConfigError`
at start-up with the fix: move those notes outside the vault, then delete
the setting. A warning would scroll past while the notes went to the cloud.

---

## The mirror — `jarvis/kb/mirror.py`

The index mirrors two folders, and nothing else can add to it.

| Source | `doc_type` | `source` | `file_path` |
|---|---|---|---|
| `.md` files in the vault, outside dot-folders | `note` | `"local"` | vault-relative path |
| `.pdf` files under `papers_dir`, recursively | `paper` | `file://` URI | absolute path |

### Vault notes — `refresh_vault`

Builds `file_path → (content_hash, meta_schema)` for indexed notes, scans the
vault, and reconciles: new file → index; changed hash → delete and re-index;
older `meta_schema` → re-index only if the note has frontmatter, otherwise
stamp the marker (see Records); file gone → delete. Dot-folders
(`.obsidian/`, `.trash/`) are skipped; `.trash/` would otherwise keep deleted
notes searchable. A `.md` that is a symlink to a file outside the vault is
skipped with a warning (see Safety rules).

### Papers — `refresh_papers`

Every indexed paper takes part. One whose file is no longer under
`papers_dir` (because the setting changed) is treated like a deleted file,
unless the same bytes turn up inside the folder, which makes it a move. For
each PDF on disk:

1. **Indexed at this path, same mtime** → skip without hashing. This is what
   makes a pass over hundreds of PDFs cheap.
2. **Indexed, mtime changed** → wait until the file is stable, hash it. Same
   hash: refresh `modified_at` only. Different hash: re-index (new highlights
   saved into a PDF land here).
3. **Not indexed at this path** → hash it. If the hash matches an entry whose
   file has vanished, it is a **move**: `update_file_path` re-points `source`
   and `file_path` on every chunk, with no re-embedding. Otherwise it is new:
   index it.

Entries whose file vanished and matched no new path are deleted.

**`index_paper_file`** builds the annotation chunks and the full-text body
chunks, stamps every chunk with title/authors/doi/`arxiv_id`/`file_path`/
`content_hash`/`modified_at`, embeds `"{title} — {authors}"` on every body chunk, and
commits once with `replace_source` — so a re-index never leaves the paper
half-present. There is no summary mode: every paper is full text.

### Safety rules

- **A missing folder deletes nothing.** If the vault or `papers_dir` is not a
  directory (unmounted drive, moved folder), that sync is skipped with a
  warning — deletions included.
- **iCloud eviction is not deletion.** An evicted file appears only as a
  `.name.pdf.icloud` placeholder; its entry is kept.
- **Half-written files wait.** `wait_for_stable` polls size and mtime; a PDF
  still being written by a sync client is left for the next pass.
- **One bad PDF doesn't stop a pass.** Indexing failures are logged per file.
- **Nothing outside the folders is read.** `resolves_inside()`
  (`jarvis/core/files.py`) follows symlinks and `..` and checks the result is
  inside the folder. A note or PDF that is really a link to somewhere else
  is skipped, with a warning naming where it points. Symlinked *folders* are
  not followed at all (`rglob` does not follow them on Python 3.12).
- **Paths are not resolved for keys.** A paper is keyed on the path where it
  sits under `papers_dir`, so a PDF reached through a symlink inside the
  folder does not look like it lives elsewhere and get re-indexed every pass.
  `update_file_path` stores a moved paper's new path as given, for the same
  reason.

### Rebuild — `rebuild_index`

Opens the raw client (`open_index_client`: the server when up, the files only
for a one-shot command), builds a fresh `knowledge_base_rebuild` collection by
running `sync_all` against it, then deletes the live collection and renames
the new one into place. An interrupted rebuild leaves the old index as it
was. It never reads the old collection, which is why it also fixes an index
too damaged to open — and why everything not backed by a file is gone
afterwards. The swap gives the collection a new id, so already-connected
processes need a restart (`kb serve`'s sync reconnects by itself after one
failed pass).

This is also the whole upgrade path from older versions, and a required one:
`get_store()` refuses an index whose collection lacks the current
`index_schema` tag (see "Knowledge base storage"). The rebuild drops every
entry no file backs (papers added by arXiv URL or from outside `papers_dir`,
digest documents, chat history and private notes) and tags the new
collection.

---

## Paper metadata — `jarvis/kb/metadata.py`

`resolve_pdf_metadata(pdf_path)` takes the first answer for each field from:

1. **Identifiers.** The DOI is found by the regex `10\.\d{4,9}/\S+` over
   the first two pages, with trailing punctuation stripped. The arXiv ID is
   taken from an arXiv DOI (`10.48550/arXiv.<id>`) if there is one, and
   otherwise from the first page. There, arXiv's margin stamp
   (`arXiv:2411.01234v2 [cs.LG] …`) is preferred over any other `arXiv:`
   mention, which may be a citation. Both ID formats are recognised (current
   `2411.01234` and pre-2007 `hep-th/9901001`), and the version is dropped
   so every version counts as the same paper.
2. **Crossref** — `GET api.crossref.org/works/<doi>` for the canonical title
   and `"Given Family, …"` authors. Only the DOI is sent, with a generic
   User-Agent: Crossref's "polite pool" asks for a contact email and jarvis
   never sends one. 10-second timeout.
3. **arXiv** — only when there is an arXiv ID and Crossref gave no title,
   so a published paper keeps its journal title. The `arxiv` package's
   client asks for that one ID; nothing else is sent.
4. **Embedded PDF metadata** — `title`/`author`, skipping placeholders such
   as "untitled".
5. **Filename** — the title of last resort.

arXiv preprints are why step 3 exists. Their PDFs usually print no DOI, and
their embedded title is often empty, so before arXiv IDs were read they were
indexed under filenames like `2411.01234v2`.

Every step is best-effort and logged on failure; a paper is always indexed.
There is no correction command: Crossref is usually right, and the files are
the source of truth.

---

## Privacy model

There is one rule: **jarvis reads nothing outside `vault_path` and
`papers_dir`.** Everything inside them is treated as shareable with the MCP
client's cloud model. There are no private folders inside the vault; keep
private notes in a separate folder (for example a second Obsidian vault).

- `resolves_inside(path, root)` (`jarvis/core/files.py`) is the single check.
  It follows symlinks and `..`, then tests whether the real path is inside
  the folder. It is applied wherever a file is read: `refresh_vault` and
  `refresh_papers` for every file, `read_note` for the path it is given, and
  `reveal_in_finder` again just before calling `open`.
- `read_note` also refuses dot-folders, which hold app settings and
  Obsidian's bin of deleted notes. Its refusal carries no content.
- `reveal_in_finder` only accepts the `file_path` of an indexed document, so
  it can't be pointed at a system file.

**Why private folders were removed.** They were checked by comparing folder
names, which is fragile: APFS is case-insensitive and `Path.resolve()` keeps
whatever case it is given, so `read_note("PRIVATE/x.md")` reached a note in
`private/`. Keeping private notes outside the folders jarvis reads is a
boundary that a spelling trick cannot get around.

**Prompt injection.** Text from documents is untrusted input to the model.
Every tool result that contains document text, titles or frontmatter values
is wrapped between `BEGIN/END RETRIEVED DATA` markers, and any marker already
inside the text is defused first, so a document cannot close the fence
early. That is a mitigation; the real protection is that no tool can change
anything.

**The index server needs a token.** The Chroma server listens on 127.0.0.1,
which any program on the machine can reach, and a web page can too via DNS
rebinding. So it refuses any request without the token in
`~/.jarvis/state/server_token` (see "Who owns the index"). Without that, a
local program could read every note or plant text in the index for the
assistant to read.

Every MCP client is treated the same, as a cloud model. There is no
local-model path: a local MCP client would get exactly the same tools.

---

## Knowledge base storage — `jarvis/kb/store.py`

A single LangChain + ChromaDB collection (`knowledge_base`) with a flat
metadata schema; the schema is documented at the top of `store.py`.

| Function | Description |
|---|---|
| `get_store()` | Process-wide store; through the server, or the files when `allow_direct_index_access()` was called; refuses an index from an older version or another embedding model |
| `open_index_client()` · `open_collection()` · `forget_store()` | Raw client access for `kb rebuild`, and dropping the cached store after a swap |
| `build_text_documents()` · `build_annotation_documents()` | Turn input into `Document`s without touching the database |
| `commit_documents(documents, replace_source=)` | The only write: optional delete of a replaced source plus the add, in one lock |
| `search(query, n_results, doc_type, category, status, entity, fields, tags)` | Hybrid or dense search with metadata filters, then re-ranking |
| `list_documents()` · `count_unique_documents()` · `count()` | De-duplicated listing and counts |
| `metadata_key_counts()` · `metadata_value_counts()` | `kb schema` and `kb_stats` |
| `get_document_chunks(source)` | Every chunk of one document in reading order |
| `delete_by_metadata()` · `update_file_path()` · `update_metadata_fields()` | The mirror's metadata-level edits |

### Who owns the index

One process does: the Chroma server that `uv run kb serve` runs as a child.
`jarvis-mcp` and `kb serve`'s own sync connect to it over loopback and fail
with the command to run when it is down. One-shot `kb` commands call
`allow_direct_index_access()` in `main()`, which lets them open the index
files themselves when the server is down — safe because they exit in
seconds, and it keeps `kb doctor` usable at exactly the moment the server is
the broken thing.

**The server and its token.** The child is Chroma's Python server
(`python -m uvicorn chromadb.app:app --host 127.0.0.1`), configured through
environment variables to persist to `rag_dir` and to require a token
(`TokenAuthenticationServerProvider`). It is not `chroma run`, Chroma's Rust
server, because that one has no way to require authentication. Chroma 1.x
treats the Python server's dependencies (`fastapi`,
`opentelemetry-instrumentation-fastapi`) as dev extras, so jarvis lists them
itself, and `tests/test_serve.py` starts the real server so an upgrade that
breaks it fails the suite.

- `kb serve` creates `~/.jarvis/state/server_token` the first time
  (`secrets.token_urlsafe(32)`, created 0600 in one step with `O_EXCL`) and
  reuses it afterwards, so an MCP client that is already running keeps
  working across restarts. The server reads the token from that file; it
  never appears in a command line or environment value.
- Clients read the same file. Constructing the `HttpClient` makes an
  authenticated round trip, so a wrong token fails at connect time with
  `ServerAuthError` and a message that says to restart `kb serve`.
- A refused token is never a reason to fall back to the files: the server is
  still running and writing to them. Only "not running" (including "no token
  file yet") allows the direct path.

This replaced every process opening its own `PersistentClient`. That model had
a fault that looked like corruption and wasn't: the chunk text and metadata
live in SQLite, where WAL gives readers a consistent view, but the vectors
live in a separate segment WAL knows nothing about, cached in the process and
never revalidated. When the sync rewrote a changed note (delete + re-add),
a long-lived reader's cached segment still listed the deleted ids, and the
next filtered query failed with `"Error finding id"`. Worse was the quiet
half: a reader that didn't happen to touch a deleted id simply searched an
out-of-date index, so freshly edited notes were unfindable until the process
restarted, with no error at all.

**Cross-process write lock (`_kb_write_lock`).** The server serialises
everything that goes through it, so this advisory `flock` on
`<rag_dir>/.write.lock` only matters on the direct path — two one-shot `kb`
commands writing to the same SQLite file while the server is down. Nothing
takes it while already holding it, so it is a plain lock.

**Permissions.** `rag_dir` is created (and chmod'ed on every open) `0700`,
since it holds the full text of your notes and papers.

**The compatibility check.** Every collection is tagged when it is created
with `embed_model` and `index_schema` (currently 2: no `visibility` field and
no private, chat or digest chunks). `get_store()` refuses a non-empty
collection whose tags differ and names `kb rebuild` as the fix. An empty one
is re-tagged instead.

### Staged writes

Indexing is split into `build_*` functions that only create `Document`s and
one shared `commit_documents()` that performs the write. `index_paper_file`
builds a paper's body and annotation chunks, then commits them — with the
delete of the previous version, via `replace_source` — under one lock. A
failure part-way leaves the index exactly as it was: never a body without its
highlights, never an old entry deleted for a replacement that never arrived.

## Retrieval pipeline — `jarvis/kb/store.py`

A query flows through four stages, all local — no data leaves the machine.

```
INDEXING (§1-2)
  a vault note, or a PDF from papers_dir
        │
        ├─ chunk ──────────────────► chroma.sqlite3   text + metadata
        │                                             (the data)
        └─ embed_model (BAAI) ─────► HNSW *.bin       the vectors
                                                      (derived — rebuildable)

RETRIEVAL (§2-3)
  query
    │
    ├─ embed_model embeds it ─► HNSW nearest neighbours ─► ranking A  semantic
    │
    └─ BM25, built in memory over the same
       filtered pool ──────────────────────────────────► ranking B  keyword
                                       │
                    reciprocal rank fusion of A and B
                                       │
                            rerank_top_n candidates (25)
                                       │
              rerank_model (cross-encoder) reads (query, chunk TEXT)
                                       │
                                 top n_results
```

Two things that are easy to get backwards:

- **BM25 is retrieval, not re-ranking.** It is the keyword half of stage 2,
  fused with the dense half. Re-ranking is a separate, later stage.
- **The cross-encoder never touches the index or the vectors.** It reads the
  raw text of the candidates: `reranker.predict([(query, doc.page_content)
  ...])`. That is what makes it accurate and why it cannot replace stage 2 —
  see stage 3.

**1. Chunking, at index time.** `build_text_documents` splits content on markdown
headers (`MarkdownHeaderTextSplitter`) and then by size
(`RecursiveCharacterTextSplitter`). Each chunk stores its `chunk_index` and a
`section` breadcrumb, and the breadcrumb is prepended to the embedded text so
a query naming both the document topic and a section can match. Headerless
content passes through as a single unlabelled section. When the caller passes
`embed_header` (papers: the title, or
`"{title} — {authors}"`), it's prepended to the embedded text of **every**
chunk, not just the first, so an author-name or title-word query can match
any chunk of a long paper.

**2. Hybrid retrieval.** Gated by `[rag] hybrid` (default `true`). When
enabled, `_hybrid_search` fetches the ChromaDB candidate pool filtered by
`doc_type` and any record filters first, then ranks it two ways over that same filtered
pool: dense (the query embedded with a BGE-style model, `embed_model`,
prefixed by `query_prefix` on the query side only) and sparse (a BM25 index
rebuilt fresh per query, via `rank-bm25` — held in memory only, so unlike the
HNSW index there is nothing here to persist or corrupt). The sparse half
earns its place on exact tokens: a query like `LoRA` or `Dr. Tanaka` is
where a vector model drifts to topically similar text that never contains
the string, and BM25 catches the literal match. The two rankings are fused by
reciprocal rank fusion (`_reciprocal_rank_fusion`, `c=60`, identity by chunk
id): an id's score is the sum of `1/(c+rank)` across whichever ranking(s) it
appears in. Because the sparse index and the dense query both operate on the
already-filtered pool, privacy holds by construction — no id outside the
filtered pool can ever be scored or returned. Setting `hybrid = false` skips
straight to plain `similarity_search`, reproducing the pre-hybrid pipeline
byte-for-byte.

**3. Re-ranking.** A cross-encoder (`rerank_model`) scores each `(query,
chunk)` pair jointly and reorders the dense or fused candidates, returning
the top `n_results`. It runs **after** the metadata filters, so it never
widens what a query can return; set `rerank_model = ""` to disable it.

The two-stage shape is forced by the difference between the models.
`embed_model` is a **bi-encoder**: query and document are embedded
*separately* into fixed vectors and compared by distance, so every document
can be embedded once, ahead of time, and the comparison is arithmetic — but
the model never sees a query and a document together. The cross-encoder reads
them **jointly**, as one text input, and returns a relevance score. That is
far more accurate, because it can weigh how the query relates to that
specific passage, and it is also why nothing can be precomputed: the score
exists only for the pair. Running it over the whole store per query would
mean thousands of forward passes. So the cheap index narrows the corpus to
`rerank_top_n` candidates and the expensive model reads only those.

This is also why the cross-encoder is unaffected by a damaged vector index —
it consumes `page_content`, the chunk text, and never reads a vector.

**4. When retrieval fails.** `_diagnose_kb_error()` maps three ChromaDB
signatures to something actionable rather than a generic `RAGError` the model
would paraphrase into noise:

- **The server isn't answering** — everything else is unreachable too, so the
  message names `uv run kb serve` rather than describing a search failure.
- **`"Error finding id"`** — the index can't resolve an id it references.
  This used to be diagnosed as on-disk corruption; that was wrong. It was the
  stale-cache fault described under "Who owns the index", which no rebuild
  could fix and a restart always did. With the server owning the index a
  reader can't hold a stale copy, so the remaining cases are worth reporting:
  `kb doctor` checks, `kb rebuild` recreates the index from the files.
- **`"Collection [...] does not exist"`** — survives the move to a server. A
  client resolves a collection to an id once, and `kb rebuild` swaps in a
  rebuilt collection with a new id, so anything already connected still names
  the replaced one. Restarting that process is the fix, and `kb rebuild` says
  so when it finishes. `kb serve`'s sync drops its cached store after a failed
  pass, so it reconnects on its own at the next interval.

There is deliberately **no reopen-and-retry** here. The previous one was
unreachable in production — every caller passed `store=`, which disarmed it —
while its message told the user a retry had already happened and the index was
therefore damaged. That sent people to rebuild a healthy index.

`uv run kb doctor` diagnoses proactively (open store → count → search-probe)
and reports whether it reached the index through the server or directly, which
is the first thing worth knowing when MCP searches fail but CLI ones don't.
On a badly corrupted store even `count()` can hard-segfault the process — a
Rust-side ChromaDB crash, uncatchable in Python — so `kb doctor` dying
abruptly is itself the diagnosis, not a bug in the doctor command.

**Embedding-model guard.** ChromaDB records `embed_model` in the collection
metadata when the collection is first created. `get_store()` compares that
tag against the configured model and raises `RAGError` on any mismatch. This prevents
silently comparing vectors from two incompatible embedding spaces. The fix is
`uv run kb rebuild`, which re-indexes everything from the files under the
new model into a fresh collection and swaps it in.

### Deferred retrieval improvements

These were designed but intentionally not built, to keep the retrieval stack
simple. Each has a concrete trigger for revisiting, so the decision has a
paper trail. The `tests/test_retrieval_quality.py` golden set is the
instrument that makes the triggers observable.

- **Better embedding/rerank models.** *Trigger:* hybrid BM25+RRF (above) isn't enough — the golden set's harder queries still regress. *Sketch:* both are drop-in config changes — `embed_model = "BAAI/bge-base-en-v1.5"` or `bge-large` (requires a `uv run kb rebuild`) and `rerank_model = "BAAI/bge-reranker-v2-m3"`.
- **Multi-query expansion.** *Trigger:* evidence that pre-rerank recall@`rerank_top_n` is the bottleneck. *Why deferred:* needs an LLM call per search, and jarvis makes none; the MCP client already reformulates queries across tool calls.
- **MMR (diversity re-ranking).** *Trigger:* top results dominated by near-duplicate chunks of one document. *Why deferred:* conflicts with cross-encoder ordering; the cheaper first fix would be a per-source cap applied after re-ranking.
- **Score thresholds.** *Why deferred:* cosine scores are poorly calibrated and corpus-dependent, and the reranker already sinks irrelevant results. Revisit only if junk results demonstrably pollute answers.

---

## Records — `jarvis/kb/frontmatter.py`

A vault note can be a job application with an outcome, a manuscript with a
venue and a deadline, a meeting record — whatever the user decides. Jarvis
knows nothing about what any of those *are*. It just reads whatever
frontmatter a note carries and makes it filterable.

**Well-known keys** — `type`/`category`, `status`, `entity`, `date`, `tags` —
map onto named metadata fields, because those are the axes worth a UI filter.
Every other scalar key passes through as `x_<key>`, so adding a new record
type needs no code change.

The `x_` prefix is a security boundary, not tidiness. A `.md` file can arrive
in the vault from anywhere, and without namespacing, a note carrying
`doc_type: paper` or a `source` in its frontmatter would overwrite jarvis's
own schema when the two dicts merged — a note could pass itself off as a
paper and slip past a filter. Prefixing makes that structurally
impossible rather than something to remember to check. The merge order backs
it up too: both `index_vault_file` and `build_text_documents` spread the
caller's metadata *first* and jarvis's own fields (`doc_type`, `source`,
`file_path`, `content_hash`, `meta_schema`) *after*, so a collision is won by
jarvis even if the `x_` table were ever wrong.

Parsing is best-effort. `yaml.safe_load` only, since frontmatter is untrusted
input — malformed YAML warns and yields no metadata, but the note is still
indexed from its text. Losing a note because its header had a stray colon
would be far worse than losing its filters. Nested values are skipped with a
warning too: flat over nested, and a silent drop would be worse than a noisy
one.

Tags are stored as `"|a|b|c|"`. ChromaDB has no list-contains or substring
operator, so membership is a substring test, and the leading and trailing
separators are what stop `|remote|` from matching `|remote-first|`.

The record header rides along in every chunk. `record_header()` builds a
string like `"job_application · Acme Bio · rejected"` and passes it to
`build_text_documents` as the `embed_header`, so every chunk carries it in its embedded
text — that's what makes "jobs I was rejected from" match a record whose body
never uses those words. It's the same mechanism papers already use for
title/authors.

`doc_type` itself is untouched. A record is a `note` with a `category`, so
`refresh_vault`'s delete logic and every search filter keyed on `doc_type`
need no changes at all.

Migration happens by version marker, and it's cheap. Every note is stamped
with `meta_schema` (frontmatter or not — a plain note without one would look
perpetually out of date and re-index on every sweep otherwise).
`refresh_vault` handles a note whose marker is behind by asking one question:
does the file have a frontmatter block? If it does, both the indexed body
(the block is now stripped from it) and the embedded record header change, so
it's deleted and re-indexed. If it doesn't — the common case for a plain note
— nothing about its vectors would differ, so `update_metadata_fields` stamps
the marker in place instead: no re-embedding, no delete, no window where the
note goes missing from the index. A schema migration shouldn't have to
re-embed an entire vault to record that most of it had nothing to record.

`kb schema` lists every metadata key with chunk counts; `kb schema <key>`
lists its distinct values. Jarvis enforces no vocabulary, which means a typo
(`stauts: rejected`) becomes its own key that silently never matches a
filter — listing what's actually there is how you catch it. `kb_stats` shows
the same record types and statuses to the model, so it filters with real
values instead of guessing.

---

## Annotations — `jarvis/kb/annotations.py`

macOS Preview and Foxit Reader both write standard ISO 32000 annotation
objects into the page `/Annots` array on save, so one generic reader
(PyMuPDF's `page.annots()`) covers both apps.

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

**Where it is wired in:** `index_paper_file` in the mirror. Annotation
chunks are built alongside the body and committed in the same write, and are
kept even when the body fails to convert, so a scanned PDF still keeps its
highlights. Re-saving a PDF with new annotations changes its byte hash, which
the next sync sees as a changed paper and re-indexes.

---

## PDF conversion — `jarvis/kb/convert.py`

`pdf_to_markdown(pdf_path) -> str` converts via **pymupdf4llm** — fast,
rule-based extraction with no ML model downloads. This replaced marker-pdf:
orders of magnitude faster, at the accepted cost of lower fidelity on complex
layouts and equations. Returning a string means no call site needs an
intermediate `.md` file or a temp-dir round-trip.

A PDF that yields no extractable text — typically a scanned or image-only PDF
with no OCR text layer — raises `ConversionError` rather than silently
indexing an empty document. There's no OCR fallback. Image extraction isn't
performed either, since nothing consumed it; `write_images=True` is the
one-line reinstatement if it's ever wanted.

---

---

## MCP server — `jarvis/mcp/server.py`

Built on the official SDK's `MCPServer`, stdio transport. Seven tools, all
annotated `read_only_hint`:

| Tool | Returns |
|---|---|
| `search_kb(query, kinds, n_results ≤ 20, category, status, entity, tags, fields)` | Full matching passages, each with title, absolute file path, record fields or authors and DOI link, and (for papers) the source to pass to `get_document` |
| `get_document(source, page)` | A paper's chunks in reading order, 15 per page. `source="local"` is refused (every note shares it) with a pointer to `read_note` |
| `read_note(path)` | One vault file from disk, by relative or absolute path. Refused if its real path is outside the vault or in a dot-folder |
| `list_documents(kind, limit ≤ 50, category, status, entity)` | Papers or notes, newest first |
| `kb_stats()` | Counts, and the record types and statuses in use (fenced as retrieved data, since they come from frontmatter) |
| `has_paper(doi, arxiv_id, title)` | Whether an indexed paper matches any of the given identifiers, with its file path, DOI link, `get_document` source and which identifier matched |
| `reveal_in_finder(path)` | `open -R` on an indexed document whose real path is still inside the vault or papers folder (macOS); list arguments, no shell |

**No write tools.** There is no add, remove, edit or move. Text from papers
and notes is untrusted input to the model reading it; with
nothing writable, there is nothing an injected instruction can change. As a
further mitigation, document text comes back between `BEGIN/END RETRIEVED
DATA` markers, and the server `instructions` tell the model to treat it as
data. Any such marker already inside a document is defused before wrapping,
so a document cannot end the fence early.

**Errors.** Expected failures raise the SDK's `ToolError`, which the client
sees as `is_error=True` with the message. A `KBCorruptionError` is prefixed
`KNOWLEDGE BASE ERROR — relay this to the user verbatim`, and the
instructions tell the model not to paraphrase it or keep searching.

**Process shape.** Long-lived (the client keeps it running), so it never
calls `allow_direct_index_access()`. Importing the module never imports
`jarvis.kb.store`, so the handshake is fast and the embedding models load on
the first tool call. stdout is the JSON-RPC channel: config notes go to
stderr, and while serving the SDK points fd 1 at stderr so a stray `print`
from a library can't corrupt the stream.

**`kb serve` must be running.** `jarvis-mcp` only ever reaches the index
through the Chroma server that `kb serve` runs (see "Who owns the index").
When it is down:

- **The client still connects.** The handshake needs no index, so the client
  connects and lists the tools.
- **Each tool call fails.** It returns the "start `uv run kb serve`" error.
- **The next call recovers.** `get_store()` caches only a successful
  connection, so once the server is up the next call succeeds without
  restarting the client.

**How `has_paper` decides.** jarvis does the matching, not the model. The
tool reads the metadata of every indexed paper (`doc_type="paper"`) and keeps
one entry per `source`. A paper matches
when any identifier given matches:

- **DOI.** Both sides go through `normalise_doi()` (`jarvis/kb/metadata.py`),
  which pulls the DOI out of a `doi:` prefix or a doi.org link and
  lower-cases it, since DOIs are case-insensitive.
- **arXiv ID.** `normalise_arxiv_id()` accepts an ID, an `arXiv:` mention or
  an arxiv.org link and drops the version, matching the version-less
  `arxiv_id` stored at indexing. An arXiv DOI (`10.48550/arXiv.<id>`) is also
  checked as an arXiv ID.
- **Title.** Both sides are reduced to their lower-cased words, so case,
  punctuation and spacing don't matter. The title must otherwise be the same.

An identifier that was given but cannot be parsed is a `ToolError`, not a
silent "no match". A paper is still missed if its PDF prints neither
identifier and its indexed title is worded differently, for example a
journal PDF of a paper the model knows as a preprint; the "no match" reply
says to try `search_kb` on the title.

**Finding new papers is the client's job.** jarvis used to fetch recent
arXiv and bioRxiv papers itself (`fetch_recent_papers`, removed in PR #6).
MCP clients have their own literature connectors, which do this more
reliably, and can call `has_paper` to skip what the user already has.

---

## `kb serve` — `jarvis/kb/serve.py`

1. Prints the resolved config.
2. Creates the server token if it does not exist yet, then starts Chroma's
   server (`uvicorn chromadb.app:app --host 127.0.0.1 --port <server_port>`)
   as a child, run from the owner-only `rag_dir`, its output going to
   `chroma.log`. It waits until the server answers an authenticated request.
   A child that exits at once is reported as "is another `kb serve` already
   using the port?".
3. Opens the store, so an index from an older version or an embedding-model
   mismatch stops start-up loudly.
4. Loops: `run_sync()`, then `Event.wait(sync_minutes * 60)`. SIGINT/SIGTERM
   set the event; the loop also stops if the Chroma child dies.
5. On exit, terminates the child (kill after 10 s).

`run_sync()` never raises: it records `{at, ok, counts | error}` in
`serve_status.json` and, on failure, drops the cached store so the next pass
reconnects. A plain loop rather than a scheduler library — there is one
periodic job, and `Event.wait` is both the sleep and the shutdown signal.

`kb status` reads the status file, checks the pid is alive, and tails
`serve.log`.

---

## The `kb` CLI — `jarvis/kb/cli.py`

| Command | What it does |
|---|---|
| `serve` | The long-running process above. Never uses direct index access |
| `sync` | One `sync_all` pass now |
| `rebuild` | `rebuild_index` after a y/N prompt that names both folders and says no file is touched |
| `list` · `stats` · `schema` | Inspection |
| `status` | `kb serve` liveness and last sync |
| `doctor` | Health probe in a child process |

Every command except `serve` calls `allow_direct_index_access()`, so it works
whether or not the server is running.

---

## Error handling — `jarvis/core/errors.py`

```
JarvisError
├── RAGError          Vector store failures
│   ├── KBCorruptionError   unresolvable id, or a collection replaced by kb rebuild
│   └── ServerAuthError     the index server refused this process's token
├── ConfigError       config.toml asks for something unsafe (private_vault_dirs, a quoted boolean)
└── ConversionError   PDF → Markdown produced no text (scanned PDF)
```

**Nothing is caught silently.** Every handler either surfaces the error, logs
it before degrading, or is narrow expected control flow. The ones that matter:

- `refresh_vault` / `refresh_papers` **raise** if they can't read the
  indexed map, rather than treating it as empty — that would re-index
  everything and mishandle deletions.
- A missing source folder is a logged skip, never "everything was deleted".
- Metadata lookups, single-PDF indexing failures and a failed sync pass are
  logged with the real exception, then the next source / file / interval
  carries on.
- Every store read and write goes through `_diagnose_kb_error`, so "server
  down" and "collection replaced by a rebuild" get their own message
  wherever they happen. Counts raise rather than report 0.

---

## What was removed, and why

jarvis used to be a full assistant: a FastAPI webapp with an agentic loop
over Ollama, Anthropic and OpenRouter, a provider-neutral transcript,
sessions with compaction, model switching with cost tracking, cancellation, a
drafts sandbox with an editor, per-hunk review and PDF export, a scheduled
LLM-scored digest, summary-mode papers, figure captioning and LLM metadata
inference. MCP clients now do all of the assistant's job, so all of it was
removed in PR #5.

**The old version is archived.** The `archive/pre-mcp-simplification` branch
was cut from `main` at `c231389` (PR #4), immediately before the removal. It
holds the full assistant as it last worked, including its own DESIGN.md
describing every removed component. Check it out to consult or run the old
code; nothing on it is maintained.

What made that safe rather than a loss of capability:

- **The privacy model got simpler and stronger.** It used to depend on the
  provider (local Ollama could see private notes, cloud models couldn't),
  with session private-flags, hard stops and a result caveat. With every
  client treated as cloud, private notes are kept out of the folders jarvis
  reads, so there is nothing to filter. (Private folders *inside* the vault
  were dropped later for the same reason; see "Privacy model".)
- **Write paths became file operations.** `kb add/remove/set-meta/update-path`
  and the confirmation dialogs existed because the index was a separate thing
  to manage. With the index mirroring files, Finder and Obsidian are the
  interface.
- **Server-side LLM work moved to the client.** The digest is a scheduled
  client task; metadata comes from Crossref instead of an LLM.
