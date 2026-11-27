"""
Central configuration for jarvis.

Resolution order (later wins):
  1. Built-in defaults
  2. ~/.jarvis/config.toml
  3. Environment variables (VAULT_PATH, PAPERS_DIR)

Example ~/.jarvis/config.toml:

    [kb]
    vault_path = "~/vault"             # every .md file in here is indexed
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
that. A section or key jarvis does not know is reported on stderr and
otherwise ignored, so a typo or a setting left over from an older version is
never silently mistaken for a working one. The file is never rewritten for
you.
"""

import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from jarvis.core.errors import ConfigError

CONFIG_FILE = Path.home() / ".jarvis" / "config.toml"


@dataclass
class Config:
    # ── The two folders the index mirrors ─────────────────────────────────────
    # Every .md file under this folder is indexed and readable by the MCP
    # client, which sends what it reads to a cloud model. Private notes belong
    # in a separate folder outside it.
    vault_path: Path = field(default_factory=lambda: Path("~/vault").expanduser())
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


# Every key jarvis reads, by section. Anything else is reported as unknown.
_KB_KEYS = {"vault_path", "papers_dir", "sync_minutes"}
_RAG_KEYS = {
    "rag_dir", "embed_model", "query_prefix", "chunk_size", "chunk_overlap",
    "rerank_model", "rerank_top_n", "hybrid", "server_port",
}
_KNOWN_KEYS = {"kb": _KB_KEYS, "rag": _RAG_KEYS}


def _note(message: str) -> None:
    """Config notes go to stderr: stdout is the MCP channel in jarvis-mcp."""
    print(f"⚠️  {message}", file=sys.stderr, flush=True)


def _refuse_private_vault_dirs(data: dict, config_file: Path) -> None:
    """
    Stop if the config still lists private vault folders.

    Older versions kept those folders out of the index. This one indexes the
    whole vault, so carrying on would quietly send private notes to the MCP
    client's cloud model. That is too serious for a warning that scrolls past.
    """
    for section in ("kb", "chat"):
        folders = data.get(section, {}).get("private_vault_dirs")
        if folders:
            raise ConfigError(
                f"{config_file} sets [{section}] private_vault_dirs = {folders!r}, "
                "but jarvis no longer keeps private folders out of the index. "
                "Every note under vault_path is indexed and can be read by your "
                "MCP client, which sends it to a cloud model.\n"
                "Move your private notes to a folder outside vault_path, then "
                f"delete the private_vault_dirs line from {config_file}."
            )


def _boolean(section: str, key: str, value, config_file: Path) -> bool:
    """A TOML true/false. A quoted "false" is a string, and bool("false") is True."""
    if not isinstance(value, bool):
        raise ConfigError(
            f"[{section}] {key} in {config_file} is {value!r}, but it has to be "
            "true or false, written without quotes."
        )
    return value


def _report_unknown_keys(data: dict, config_file: Path) -> None:
    """Name every section and key jarvis does not read, so none is mistaken for working."""
    for section in sorted(data):
        known = _KNOWN_KEYS.get(section)
        if known is None:
            _note(
                f"[{section}] in {config_file} is not a section jarvis reads, so it "
                "is ignored. It may be left over from an older version; you can delete it."
            )
            continue
        for key in sorted(set(data[section]) - known):
            _note(
                f"[{section}] {key} in {config_file} is not a setting jarvis knows, "
                "so it is ignored. Check the spelling, or delete it."
            )


def load_config(config_file: Path = CONFIG_FILE) -> Config:
    """Load a Config, applying TOML file values then env var overrides."""
    cfg = Config()

    if config_file.exists():
        with open(config_file, "rb") as f:
            data = tomllib.load(f)
        _refuse_private_vault_dirs(data, config_file)
        _report_unknown_keys(data, config_file)

        kb = data.get("kb", {})
        if "vault_path" in kb:
            cfg.vault_path = Path(str(kb["vault_path"])).expanduser()
        if "papers_dir" in kb:
            cfg.papers_dir = Path(str(kb["papers_dir"])).expanduser()
        if "sync_minutes" in kb:
            cfg.sync_minutes = int(kb["sync_minutes"])

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
            cfg.hybrid = _boolean("rag", "hybrid", r["hybrid"], config_file)
        if "server_port" in r:
            cfg.server_port = int(r["server_port"])

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
