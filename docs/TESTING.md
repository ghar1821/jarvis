# Testing

## Running tests

```bash
# Install dev dependencies first (once)
uv sync --group dev

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_mirror.py

# Run a single test by name
uv run pytest tests/test_mirror.py::test_a_moved_paper_is_repointed_without_re_embedding
```

---

## Continuous integration

[`.github/workflows/tests.yml`](../.github/workflows/tests.yml) runs the unit
suite on GitHub Actions — `uv run pytest` on
`ubuntu-latest` with Python 3.12, exactly the command you'd run locally.

It fires on pushes to `main` and on pull requests targeting `main`, and nothing
else. Feature branches stay unwatched until they open a PR at `main`, which
matches the git workflow here: work happens on a local branch, and `main` is
the thing worth guarding.

Two caches keep a run to a couple of minutes:

- **uv's dependency cache**, keyed off `uv.lock`, so the torch download that
  comes with `sentence-transformers` happens once per lockfile change.
- **`~/.cache/huggingface`**, holding the embedding model and the reranker
  (~220 MB together). The cache key names both models, so swapping either one
  in the config means changing the key to match, otherwise CI keeps restoring
  the old pair and re-downloading the new one every run.

Nothing in the suite reaches the network or needs a secret, so the workflow
declares none. jarvis has no LLM calls left to test against a live service,
which is why there are no integration tests any more.

---

## Test infrastructure

### Dedicated ChromaDB store

KB tests run against a real ChromaDB instance at `tests/.chroma/` (gitignored,
never committed). They open it directly rather than through `kb serve`, so the
whole suite runs with nothing started — the store fixture is passed explicitly
to every KB function, which is also what keeps tests isolated from each other. Each test creates its own `test_<uuid8>` collection there and
deletes it at teardown, which gets us three things for free: tests are fully
isolated from each other, the store directory itself persists between runs
without needing to be rebuilt, and the embedding model doesn't reload for
every single test.

### Real HuggingFace models

The KB tests use the real embedding model from the default config
(`BAAI/bge-small-en-v1.5`) rather than a mock or a deterministic stub, built
through the same `build_embeddings()` helper production uses, so the query
prefix and normalization actually match real retrieval. Any test that calls
`search()` loads the real cross-encoder reranker too
(`cross-encoder/ms-marco-MiniLM-L6-v2`). Both models download once, to
`~/.cache/huggingface/` (embedding ~133 MB, reranker ~90 MB), then come from
local cache after that.

Why not mock them? A fake embedding function that returns zeros or random
vectors would hide the failure modes that actually matter — ChromaDB filter
behavior, the LangChain wrapper's batching logic, and search ranking all
depend on real vector values, and you can't observe what the reranker
actually does without running it. The one-off download is a cheap price for
that fidelity, and it's the general policy here: pay a modest one-time setup
cost rather than test against a mock that hides how the thing really behaves.

### Real vs mocked, by boundary

- **Real:** embeddings, the reranker, ChromaDB, PyMuPDF, and pymupdf4llm all
  run for real. PDF fixtures — annotated pages, scanned-like pages — are
  generated in-test with PyMuPDF, so no binary fixtures are committed and the
  real extraction/conversion paths run end to end.
- **Real, end to end:** the MCP tests talk to the server through the SDK's
  in-process `Client`, and the mirror tests sync real files in `tmp_path`.
- **Mocked, and deliberately so:** the network and the desktop. The arxiv
  clients' calls (`_client.results` in the feed and `_arxiv_client.results`
  in the metadata lookup, stubbed with real `arxiv.Result` objects), the bioRxiv and Crossref APIs (`requests.get` stubbed with
  responses shaped like the real payload), and `open -R` in
  `reveal_in_finder` (so no Finder windows open). These are genuine system
  boundaries: what's under test at each is our logic, not the service's.
- **One shortcut:** `test_mirror.py` replaces `wait_for_stable` (which polls
  a file for ~2s to make sure a sync client has finished writing it) with an
  instant stub, because files a test wrote are complete. One test marked
  `real_settle` runs the real poll.

### Retrieval-quality benchmark

`test_retrieval_quality.py` measures how *good* retrieval is, not just
whether it runs. It seeds a small module-scoped corpus once — paper
texts and Markdown notes — and runs a fixed golden set of about 24
queries through the real `search()` pipeline (hybrid BM25+RRF, the default),
checking that hit-rate@5 and MRR@5 stay above a floor. It does two jobs at
once: catching regressions (a broken query prefix, chunker, or reranker drops
the metrics well below floor) and giving you a place to actually watch
retrieval accuracy while tuning. The corpus includes acronym and proper-noun
queries (`LoRA`, `BERT`, `Dr. Tanaka`) plus author-name queries (`"papers by
Vaswani"`) that only match because of the `embed_header` prepended to every
paper chunk.

The thresholds are set as a regression floor with margin, not at the current
ceiling. On a corpus this small and topically distinct, the pipeline scores
near-perfectly — tighten the thresholds only as far as stays stable across
query rewording.

---

## What is tested

| File | Module | Behaviours covered |
|---|---|---|
| `test_mirror.py` | `jarvis/kb/mirror.py` | The index as a mirror of two folders, on real files. Vault: new, changed and deleted notes; private folders never indexed; a note moved into `private/` leaves the index; `.trash/` and other dot-folders skipped; a missing vault deletes nothing. Records: frontmatter becomes filterable metadata, record filters and tags narrow search, the record header is embedded, and schema migration re-embeds only notes that have frontmatter. Papers: added then skipped when unchanged (same chunk ids); a changed PDF re-indexed with its new highlights; a moved PDF re-pointed by content hash **without re-embedding**; a deleted PDF removed; a missing papers folder, an iCloud-evicted placeholder and entries from outside `papers_dir` all delete nothing; a file still being written waits for the next pass; a scanned PDF keeps its typed notes; Crossref metadata lands on every chunk; an arXiv preprint is stored with its arXiv ID and its arXiv title; a paper indexed before arXiv IDs were recorded is upgraded on the next pass, and not read again on the pass after. `sync_all` covers both folders, and `rebuild_index` (direct path, server forced down so it can never touch a real index) keeps only file-backed entries and leaves one collection |
| `test_store.py` | `jarvis/kb/store.py` | Staged writes (build touches nothing; commit replaces and adds together); `add_texts`; search visibility filter; delete by metadata; `list_documents` de-duplication; `update_file_path`; `get_document_chunks` ordering; the embedding-model guard; re-ranking; chunk index and section breadcrumbs; annotation indexing and sweeping; `KBCorruptionError` diagnosis (unresolvable id names `kb rebuild`, stale collection names a restart, connection reset is not "server down"); embed header on every chunk; hybrid vs dense-only search and RRF; `doc_type` list filters; `get_store` refusing direct access for long-lived processes and allowing it for one-shot commands; owner-only index dir; telemetry off |
| `test_mcp_server.py` | `jarvis/mcp/server.py` | Through the SDK's in-process `Client`: exactly seven tools, all read-only; importing the server loads neither the store nor the embedding stack. Nothing private comes back — the private note was never indexed, and a leftover private chunk from an older index is filtered from search, record-filtered search, listings and stats. Hits show absolute paths and DOI links, wrapped as retrieved data. `read_note` accepts relative or absolute paths and refuses private, hidden, symlinked-into-private and outside-the-vault paths. `get_document` reads a paper and refuses `source="local"`. A `KBCorruptionError` is relayed verbatim. `fetch_recent_papers` marks papers already indexed (by DOI case-insensitively, and by arXiv ID when the DOI and title don't match), drops ones older than the window and reports a failed category. `reveal_in_finder` reveals an indexed note and refuses system files, private notes and legacy private entries |
| `test_metadata.py` | `jarvis/kb/metadata.py` | DOI regex with trailing punctuation stripped; Crossref mapping (title whitespace collapsed, "Given Family" authors, nameless authors skipped); the Crossref request carries no email or mailto; Crossref failures fall through; embedded metadata read with placeholder titles skipped; precedence Crossref → arXiv → embedded → filename; no Crossref call without a DOI; an unreadable PDF still resolves. arXiv IDs: the margin stamp is found with its version dropped, preferred over a cited preprint, old-style IDs recognised, and read from an arXiv DOI; the arXiv lookup sends only the ID and falls through on failure; a published paper keeps its journal title and arXiv isn't asked |
| `test_kb_cli.py` | `jarvis/kb/cli.py` | `kb doctor`: healthy, probe killed by signal (names `kb rebuild`, says nothing is lost), corruption and open failures exit non-zero, empty store points at `kb sync`, says whether `kb serve` is running. The migration report groups unbacked papers (with DOI links), digests, private documents (counted, never named) and chat chunks, and is silent on a clean index. `kb rebuild` does nothing without a yes |
| `test_serve.py` | `jarvis/kb/serve.py` | Status file round trip and tolerance of garbage; a successful sync records its counts; a failed sync is recorded, not raised, and drops the cached store so the next pass reconnects; `kb status` before first run, running with its last sync, and stopped with a failed sync; start-up explains a Chroma child that exits at once |
| `test_security.py` | the whole package | An AST scan of every module: the set of file-writing calls equals a reviewed allowlist (the serve status file, the Chroma log, the index write lock), nothing anywhere deletes a file, and the allowed writes point inside `~/.jarvis` |
| `test_config.py` | `jarvis/core/config.py` | Defaults; `[kb]` and `[rag]` parsing with `~` expansion; moved keys still read from `[chat]`/`[sync]` with a note on stderr (never stdout, the MCP channel); the new location wins; obsolete keys named once without echoing an API key; env overrides; `format_describe`; the README's example config loads with no notes, and every `uv run …` and `uv run kb …` it names exists |
| `test_frontmatter.py` | `jarvis/kb/frontmatter.py` | Well-known keys mapping to named fields and the frontmatter block being stripped from the body. Unknown keys passing through as `x_<key>`. **The shadowing defence** — `visibility`/`doc_type`/`source`/`file_path` in a note's own frontmatter cannot overwrite jarvis's schema. Tags joined to `"|a|b|c|"` with exact membership. Malformed YAML warning but keeping the note; nested values skipped with a warning; `record_header` for full, partial and empty records |
| `test_annotations.py` | `jarvis/kb/annotations.py` | Extraction from PDFs annotated in-test with PyMuPDF (the same annotation objects Preview and Foxit write): highlights recovered by quad geometry, typed comments, comments attached to highlights |
| `test_pdf_convert.py` | `jarvis/kb/convert.py` | Real pymupdf4llm conversion of PDFs generated in-test; `ConversionError` for a PDF with no text layer |
| `test_arxiv_fetch.py` | `jarvis/feeds/arxiv.py` | `_to_paper` field mapping including `doi` and the version-less `arxiv_id`; retries on a transient empty feed; library errors wrapped as `FetchError`; title de-duplication |
| `test_biorxiv_fetch.py` | `jarvis/feeds/biorxiv.py` | Record → paper mapping (doi, doi.org link, `bioRxiv:{category}` source); cursor pagination; empty first page retried; keyword matching over title and abstract with DOI de-duplication |
| `test_errors.py` | `jarvis/core/errors.py` | `@with_retries`: success on first try; retry on matching exception; raise after max attempts; no retry on other exceptions; exponential backoff with jitter |
| `test_retrieval_quality.py` | `jarvis/kb/store.py` | Golden-set retrieval benchmark — hit-rate@5 and MRR@5 over a seeded corpus of papers and notes, indexed with the same title/authors embed header the mirror uses |

## What is not tested

| Module | Reason |
|---|---|
| `jarvis/kb/serve.py` `serve()` loop | Process plumbing: launches `chroma run`, waits for a heartbeat, loops on `Event.wait`. The pieces it calls (`run_sync`, `_wait_until_answering`, the status file) are tested; the loop itself was verified by running it end to end against a throwaway vault, papers folder and port — start-up, first sync, a move picked up by `kb sync`, `kb rebuild` while serving, and a Ctrl-C that also stopped the Chroma child |
| `jarvis/mcp/server.py` `main()` over real stdio | Covered in-process by `test_mcp_server.py`; the stdio launch was verified by connecting `mcp.Client` to `jarvis-mcp` as a subprocess, which is what a desktop client does |
| A real MCP client (Claude Desktop, Antigravity) | Needs the desktop app. Checked by hand: search a paper, read a note, reveal a file, and ask about a private note, which must be refused |
