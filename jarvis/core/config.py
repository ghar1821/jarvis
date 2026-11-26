"""
Central configuration for jarvis.

Resolution order (later wins):
  1. Built-in defaults
  2. ~/.jarvis/config.toml
  3. Environment variables (VAULT_PATH, PAPERS_DIR)

Example ~/.jarvis/config.toml:

    [kb]
    vault_path = "~/vault"
    private_vault_dirs = ["private"]   # top-level vault folders never indexed
    papers_dir = "~/Documents/papers"  # PDFs here (and in subfolders) are indexed
    sync_minutes = 30                  # how often `kb serve` re-mirrors both folders

    [rag]
    rag_dir = "~/.jarvis/rag"
    embed_model = "BAAI/bge-small-en-v1.5"
    # Query-side instruction prefix for BGE-style models; "" to disable
    query_prefix = "Represent this sentence for searching relevant passages: "
    chunk_size = 1024
    chunk_overlap = 128
    rerank_model = "cross-encoder/ms-marco-MiniLM-L6-v2"   # "" to disable re-ranking
    rerank_top_n = 25
    hybrid = true
    server_port = 8321           # `kb serve` listens here, on 127.0.0.1 only

jarvis has no API keys and makes no LLM calls: the MCP client does all of
that. Keys left over from older versions are reported once and otherwise
ignored, and keys that moved to [kb] are still read from where they used to
be. The file is never rewritten for you.
"""

import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_FILE = Path.home() / ".jarvis" / "config.toml"


@dataclass
class Config:
    # ── The two folders the index mirrors ─────────────────────────────────────
    vault_path: Path = field(default_factory=lambda: Path("~/vault").expanduser())
    # Top-level vault folders that are never indexed and never readable over
    # MCP. Every MCP client sends what it reads to a cloud model, so the only
    # safe place for private notes is outside the index altogether.
    private_vault_dirs: list[str] = field(default_factory=lambda: ["private"])
    # PDFs under this folder (recursively) are the papers. None means no
    # papers are indexed at all.
    papers_dir: Path | None = None
    # Minutes between mirror passes in `kb serve`. A periodic sweep rather than
    # filesystem events, so saving a highlight into a PDF re-indexes it at
    # most once per interval instead of once per save.
    sync_minutes: int = 30

    # ── Retrieval ─────────────────────────────────────────────────────────────
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
    # Hybrid dense+BM25 retrieval fused by reciprocal-rank fusion; false
    # reproduces the dense-only pipeline exactly.
    hybrid: bool = True
    # Port the knowledge-base server listens on. It always binds 127.0.0.1 —
    # Chroma's server has no authentication, so being unreachable from outside
    # this machine is the protection, and the host is deliberately not
    # configurable.
    server_port: int = 8321


# Keys that moved to [kb]: (old section, old key) -> new key. Still read from
# the old place, with a note, so an existing config keeps working.
_MOVED_KEYS = {
    ("chat", "vault_path"): "vault_path",
    ("chat", "private_vault_dirs"): "private_vault_dirs",
    ("sync", "pdf_watch_dir"): "papers_dir",
    ("sync", "vault_refresh_minutes"): "sync_minutes",
}

# Whole sections and single keys that no longer do anything.
_OBSOLETE_SECTIONS = {"digest", "chat", "drafts", "openrouter", "models", "auth", "sync"}
_OBSOLETE_RAG_KEYS = {"figure_captions", "figure_max_per_doc", "figure_min_pixels"}

_KB_PATH_KEYS = {"vault_path", "papers_dir"}


def _apply_kb_value(cfg: Config, key: str, value) -> None:
    """Set one [kb] value on cfg, converting it to the field's type."""
    if key in _KB_PATH_KEYS:
        setattr(cfg, key, Path(str(value)).expanduser())
    elif key == "private_vault_dirs":
        cfg.private_vault_dirs = [str(name) for name in value]
    elif key == "sync_minutes":
        cfg.sync_minutes = int(value)


def _note(message: str) -> None:
    """Config notes go to stderr: stdout is the MCP channel in jarvis-mcp."""
    print(f"⚠️  {message}", file=sys.stderr, flush=True)


def load_config(config_file: Path = CONFIG_FILE) -> Config:
    """Load a Config, applying TOML file values then env var overrides."""
    cfg = Config()

    if config_file.exists():
        with open(config_file, "rb") as f:
            data = tomllib.load(f)

        # Old locations first, so a value in [kb] wins if both are present.
        for (section, old_key), new_key in _MOVED_KEYS.items():
            if old_key in data.get(section, {}):
                _apply_kb_value(cfg, new_key, data[section][old_key])
                _note(
                    f"[{section}] {old_key} has moved to [kb] {new_key} in {config_file} "
                    "(still read from the old place for now)"
                )

        kb = data.get("kb", {})
        for key in ("vault_path", "private_vault_dirs", "papers_dir", "sync_minutes"):
            if key in kb:
                _apply_kb_value(cfg, key, kb[key])

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
        if "hybrid" in r:
            cfg.hybrid = bool(r["hybrid"])
        if "server_port" in r:
            cfg.server_port = int(r["server_port"])

        # Report everything that is now ignored, once, so a setting that
        # silently stopped doing anything is never a mystery.
        moved_old_keys = {pair for pair in _MOVED_KEYS}
        for section in sorted(_OBSOLETE_SECTIONS & data.keys()):
            leftover = [
                key for key in data[section] if (section, key) not in moved_old_keys
            ]
            if leftover:
                _note(
                    f"[{section}] {', '.join(sorted(leftover))} in {config_file} "
                    "no longer does anything — safe to delete"
                )
        stale_rag = sorted(_OBSOLETE_RAG_KEYS & r.keys())
        if stale_rag:
            _note(
                f"[rag] {', '.join(stale_rag)} in {config_file} "
                "no longer does anything — safe to delete"
            )

    # Env var overrides (always win over TOML)
    if v := os.environ.get("VAULT_PATH"):
        cfg.vault_path = Path(v).expanduser()
    if v := os.environ.get("PAPERS_DIR"):
        cfg.papers_dir = Path(v).expanduser()

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


def format_describe(cfg: "Config | None" = None) -> str:
    """
    The resolved configuration as printable text, shown when `kb serve`
    starts. Almost every setup problem is "jarvis is not reading the config I
    think it is", and the resolved values answer that at a glance.
    """
    cfg = cfg or get_config()
    rows = [
        ("config file", f"{CONFIG_FILE}{'' if CONFIG_FILE.exists() else '  (not found — defaults)'}"),
        ("vault_path", cfg.vault_path),
        ("private_vault_dirs", ", ".join(cfg.private_vault_dirs) or "—"),
        ("papers_dir", cfg.papers_dir or "— (no papers indexed)"),
        ("sync_minutes", cfg.sync_minutes),
        ("rag_dir", cfg.rag_dir),
        ("server_port", f"{cfg.server_port} (127.0.0.1 only)"),
        ("embed_model", cfg.embed_model),
        ("rerank_model", cfg.rerank_model or "— (re-ranking off)"),
        ("hybrid", "true" if cfg.hybrid else "false"),
    ]
    width = max(len(key) for key, _ in rows)
    return "\n".join(f"  {key:<{width}}  {value}" for key, value in rows)
