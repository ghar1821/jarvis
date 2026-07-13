"""
Central configuration for jarvis.

Resolution order (later wins):
  1. Built-in defaults
  2. ~/.jarvis/config.toml
  3. Environment variables

Example ~/.jarvis/config.toml:

    [digest]
    output_dir = "~/Documents/papers/digest"
    max_results = 10
    # arxiv_categories is a list of [category, limit] pairs:
    # arxiv_categories = [["cs.LG", 150], ["cs.AI", 80]]
    # bioRxiv sources — categories (server-side filter) and free-text keywords
    # (client-side match over the recent-preprint window), each [name, limit]:
    # biorxiv_categories = [["bioinformatics", 100]]
    # biorxiv_keywords = [["cytometry", 50], ["spatial transcriptomics", 50]]
    # biorxiv_days = 7

    [rag]
    rag_dir = "~/.jarvis/rag"
    embed_model = "BAAI/bge-small-en-v1.5"
    # Query-side instruction prefix for BGE-style models; "" to disable
    query_prefix = "Represent this sentence for searching relevant passages: "
    chunk_size = 1024
    chunk_overlap = 128
    rerank_model = "cross-encoder/ms-marco-MiniLM-L6-v2"   # "" to disable re-ranking
    rerank_top_n = 25
    # Vision captioning of PDF figures at ingest (needs a vision-capable model).
    # Off by default — each figure costs a vision-model call. Opt in per
    # document with `kb add --figures` or the chat tool's with_figures flag.
    figure_captions = false
    figure_max_per_doc = 20
    figure_min_pixels = 40000
    hybrid = true

    [chat]
    provider = "ollama"          # "ollama" | "anthropic"
    anthropic_model = "claude-sonnet-4-6"
    # Ollama model tag (must support tool calling; vision for figure captioning)
    ollama_model = "qwen3-vl:30b"
    vault_path = "~/vault"
    # Folder of user-written skill files (*.md); missing folder = feature off
    skills_dir = "~/.jarvis/skills"
    # Natural-language instruction for how the assistant should write replies
    response_style = ""
    # Long sessions get their LLM context compacted (old exchanges summarised)
    compact_after_tokens = 12000
    compact_keep_exchanges = 6

    [sync]
    # Folder scanned periodically by the jarvis-sync daemon; new PDFs dropped
    # here are auto-indexed as public papers (full text). Omit to disable.
    pdf_watch_dir = "~/Documents/papers/inbox"
    pdf_watch_minutes = 30       # minutes between PDF inbox scans
    vault_refresh_minutes = 30
    digest_day = "mon"           # APScheduler day_of_week token
    digest_hour = 5

    [auth]
    api_key = "sk-ant-..."    # Anthropic API key (alternative to ANTHROPIC_API_KEY env var)
"""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_FILE = Path.home() / ".jarvis" / "config.toml"

_DEFAULT_ARXIV_CATS: list[tuple[str, int]] = [
    ("cs.LG", 150),
    ("cs.AI", 80),
    ("cs.NE", 50),
    ("cs.CV", 80),
    ("cs.CL", 80),
    ("cs.MA", 50),
]


@dataclass
class Config:
    # ── Digest pipeline ───────────────────────────────────────────────────────
    anthropic_model: str = "claude-sonnet-4-6"
    output_dir: Path = field(default_factory=lambda: Path("~/Documents/papers/digest").expanduser())
    max_results: int = 10
    arxiv_cats: list[tuple[str, int]] = field(default_factory=lambda: list(_DEFAULT_ARXIV_CATS))
    # bioRxiv sources. Categories use the API's server-side filter; only
    # "bioinformatics" is a real bioRxiv category — topics with no category
    # (cytometry, spatial, scRNA-seq) go through keyword matching instead.
    biorxiv_cats: list[tuple[str, int]] = field(
        default_factory=lambda: [("bioinformatics", 100)]
    )
    biorxiv_keywords: list[tuple[str, int]] = field(
        default_factory=lambda: [
            ("cytometry", 50),
            ("spatial transcriptomics", 50),
            ("scRNA-seq", 50),
        ]
    )
    biorxiv_days: int = 7

    # ── RAG ───────────────────────────────────────────────────────────────────
    rag_dir: Path = field(default_factory=lambda: Path("~/.jarvis/rag").expanduser())
    embed_model: str = "BAAI/bge-small-en-v1.5"
    # Instruction prepended to queries (not documents) before embedding. BGE-style
    # models are trained with this asymmetric prefix; empty string disables it.
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    chunk_size: int = 1024
    chunk_overlap: int = 128
    # Cross-encoder that re-ranks the top rerank_top_n candidates down to the
    # requested number of results. Empty string disables re-ranking.
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    rerank_top_n: int = 25
    # Caption PDF figures at ingest via the active provider's vision model.
    # Off by default — each figure costs a vision-model call. Users opt in per
    # document (kb add --figures, the chat tool's with_figures) or flip this
    # on globally. The other two knobs bound cost/noise when captioning runs.
    figure_captions: bool = False
    figure_max_per_doc: int = 20
    figure_min_pixels: int = 40000
    # Hybrid dense+BM25 retrieval fused by reciprocal-rank fusion; false
    # reproduces the pre-hybrid dense-only pipeline exactly.
    hybrid: bool = True

    # ── Chat / LLM provider ──────────────────────────────────────────────────
    provider: str = "ollama"  # "ollama" | "anthropic"
    # Ollama model tag. qwen3-vl:30b is a vision + thinking MoE (3.3B active
    # params) that fits comfortably in 36GB on an M3 Max. Confirm the exact
    # registry tag with `ollama list` — Ollama's naming can shift over time.
    ollama_model: str = "qwen3-vl:30b"
    vault_path: Path = field(default_factory=lambda: Path("~/vault").expanduser())
    # Vault folders whose contents are treated as private (local model only)
    private_vault_dirs: list[str] = field(default_factory=lambda: ["private"])
    # User-written skill files the agent can load on demand
    skills_dir: Path = field(default_factory=lambda: Path("~/.jarvis/skills").expanduser())
    # Free-text style instruction appended to the system prompt ("" = none)
    response_style: str = ""
    # Session compaction: summarise old exchanges once the estimated context
    # passes compact_after_tokens, keeping the last compact_keep_exchanges verbatim
    compact_after_tokens: int = 12000
    compact_keep_exchanges: int = 6

    # ── Sync daemon ──────────────────────────────────────────────────────────
    # PDF inbox scanned periodically by jarvis-sync; None disables the scan.
    pdf_watch_dir: Path | None = None
    # Minutes between inbox scans. A periodic sweep (rather than filesystem
    # events) means saving a highlight into a PDF triggers at most one
    # re-ingest per interval instead of one per save.
    pdf_watch_minutes: int = 30
    vault_refresh_minutes: int = 30
    digest_day: str = "mon"
    # 05:00 rather than the small hours: a Mac asleep at 02:00 relies on
    # misfire handling, so a slot closer to working hours misses less often.
    digest_hour: int = 5

    # ── Auth ──────────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""


def load_config(config_file: Path = CONFIG_FILE) -> Config:
    """Load a Config, applying TOML file values then env var overrides."""
    cfg = Config()

    if config_file.exists():
        with open(config_file, "rb") as f:
            data = tomllib.load(f)

        d = data.get("digest", {})
        if "output_dir" in d:
            cfg.output_dir = Path(str(d["output_dir"])).expanduser()
        if "max_results" in d:
            cfg.max_results = int(d["max_results"])
        if "arxiv_categories" in d:
            cfg.arxiv_cats = [(str(c[0]), int(c[1])) for c in d["arxiv_categories"]]
        if "biorxiv_categories" in d:
            cfg.biorxiv_cats = [(str(c[0]), int(c[1])) for c in d["biorxiv_categories"]]
        if "biorxiv_keywords" in d:
            cfg.biorxiv_keywords = [(str(c[0]), int(c[1])) for c in d["biorxiv_keywords"]]
        if "biorxiv_days" in d:
            cfg.biorxiv_days = int(d["biorxiv_days"])

        r = data.get("rag", {})
        if "rag_dir" in r:
            cfg.rag_dir = Path(str(r["rag_dir"])).expanduser()
        if "embed_model" in r:
            cfg.embed_model = str(r["embed_model"])
        if "query_prefix" in r:
            cfg.query_prefix = str(r["query_prefix"])
        if "chunk_size" in r:
            cfg.chunk_size = int(r["chunk_size"])
        if "chunk_overlap" in r:
            cfg.chunk_overlap = int(r["chunk_overlap"])
        if "rerank_model" in r:
            cfg.rerank_model = str(r["rerank_model"])
        if "rerank_top_n" in r:
            cfg.rerank_top_n = int(r["rerank_top_n"])
        if "figure_captions" in r:
            cfg.figure_captions = bool(r["figure_captions"])
        if "figure_max_per_doc" in r:
            cfg.figure_max_per_doc = int(r["figure_max_per_doc"])
        if "figure_min_pixels" in r:
            cfg.figure_min_pixels = int(r["figure_min_pixels"])
        if "hybrid" in r:
            cfg.hybrid = bool(r["hybrid"])

        c = data.get("chat", {})
        if "provider" in c:
            cfg.provider = str(c["provider"])
        # [chat] anthropic_model is the canonical home; [digest] anthropic_model
        # is a deprecated fallback kept for existing configs (fail visibly
        # instead of silently rewriting the user's file).
        if "anthropic_model" in c:
            cfg.anthropic_model = str(c["anthropic_model"])
        elif "anthropic_model" in d:
            cfg.anthropic_model = str(d["anthropic_model"])
            print(
                f"⚠️  [digest] anthropic_model is deprecated — move it to "
                f"[chat] anthropic_model in {config_file}",
                flush=True,
            )
        if "ollama_model" in c:
            cfg.ollama_model = str(c["ollama_model"])
        if "vault_path" in c:
            cfg.vault_path = Path(str(c["vault_path"])).expanduser()
        if "private_vault_dirs" in c:
            cfg.private_vault_dirs = [str(d) for d in c["private_vault_dirs"]]
        if "skills_dir" in c:
            cfg.skills_dir = Path(str(c["skills_dir"])).expanduser()
        if "response_style" in c:
            cfg.response_style = str(c["response_style"])
        if "compact_after_tokens" in c:
            cfg.compact_after_tokens = int(c["compact_after_tokens"])
        if "compact_keep_exchanges" in c:
            cfg.compact_keep_exchanges = int(c["compact_keep_exchanges"])

        s = data.get("sync", {})
        if "pdf_watch_dir" in s:
            cfg.pdf_watch_dir = Path(str(s["pdf_watch_dir"])).expanduser()
        if "pdf_watch_minutes" in s:
            cfg.pdf_watch_minutes = int(s["pdf_watch_minutes"])
        if "vault_refresh_minutes" in s:
            cfg.vault_refresh_minutes = int(s["vault_refresh_minutes"])
        if "digest_day" in s:
            cfg.digest_day = str(s["digest_day"])
        if "digest_hour" in s:
            cfg.digest_hour = int(s["digest_hour"])

        a = data.get("auth", {})
        if "api_key" in a:
            cfg.anthropic_api_key = str(a["api_key"])

    # Env var overrides (always win over TOML)
    if v := os.environ.get("OLLAMA_MODEL"):
        cfg.ollama_model = v
    if v := os.environ.get("ANTHROPIC_MODEL"):
        cfg.anthropic_model = v
    if v := os.environ.get("CHAT_PROVIDER"):
        cfg.provider = v
    if v := os.environ.get("VAULT_PATH"):
        cfg.vault_path = Path(v).expanduser()
    if v := os.environ.get("PDF_WATCH_DIR"):
        cfg.pdf_watch_dir = Path(v).expanduser()

    return cfg


_config: Config | None = None


def get_config() -> Config:
    """Return the process-wide Config singleton."""
    global _config
    if _config is None:
        # Resolve CONFIG_FILE through the module namespace at call time so it
        # can be repointed (tests do this via monkeypatch).
        _config = load_config(CONFIG_FILE)
    return _config


def reset_config() -> None:
    """Clear the singleton so the next get_config() reloads from disk."""
    global _config
    _config = None


def warn_if_config_readable(config_file: Path = CONFIG_FILE) -> None:
    """
    Print a loud warning when the config file (which can hold the Anthropic
    API key) is readable by group/others. Fail visibly, don't silently chmod
    the user's file.
    """
    try:
        mode = config_file.stat().st_mode & 0o777
    except OSError:
        return
    if mode & 0o077:
        print(
            f"⚠️  {config_file} is readable by other users (mode {mode:o}) and may "
            f"contain your API key. Fix with: chmod 600 {config_file}",
            flush=True,
        )


def set_config_value(section: str, key: str, value, config_file: Path = CONFIG_FILE) -> None:
    """
    Persist one key into the user's config.toml, preserving every other key,
    comment, and the file's formatting (tomlkit round-trips the document).
    The write is atomic (temp file + os.replace) and the file ends up mode
    0600 — it can hold the API key.
    """
    import tomlkit

    if config_file.exists():
        document = tomlkit.parse(config_file.read_text(encoding="utf-8"))
    else:
        document = tomlkit.document()

    if section not in document:
        document[section] = tomlkit.table()
    document[section][key] = value

    config_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = config_file.with_suffix(".toml.tmp")
    tmp_file.write_text(tomlkit.dumps(document), encoding="utf-8")
    os.chmod(tmp_file, 0o600)
    os.replace(tmp_file, config_file)
