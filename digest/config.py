"""
Central configuration for paper_digest.

Resolution order (later wins):
  1. Built-in defaults
  2. ~/.seshat/config.toml
  3. Environment variables

Example ~/.seshat/config.toml:

    [digest]
    ollama_model = "gemma4:26b"
    anthropic_model = "claude-sonnet-4-6"
    output_dir = "~/Documents/papers/digest"
    max_results = 10
    # arxiv_categories is a list of [category, limit] pairs:
    # arxiv_categories = [["cs.LG", 150], ["cs.AI", 80]]

    [rag]
    rag_dir = "~/.seshat/rag"
    embed_model = "BAAI/bge-small-en-v1.5"
    # Query-side instruction prefix for BGE-style models; "" to disable
    query_prefix = "Represent this sentence for searching relevant passages: "
    chunk_size = 1024
    chunk_overlap = 128
    rerank_model = "cross-encoder/ms-marco-MiniLM-L6-v2"   # "" to disable re-ranking
    rerank_top_n = 25

    [chat]
    provider = "ollama"          # "ollama" | "anthropic"
    vault_path = "~/vault"

    [auth]
    api_key = "sk-ant-..."    # Anthropic API key (alternative to ANTHROPIC_API_KEY env var)
"""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_FILE = Path.home() / ".seshat" / "config.toml"

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
    ollama_model: str = "gemma4:26b"
    anthropic_model: str = "claude-sonnet-4-6"
    output_dir: Path = field(default_factory=lambda: Path("~/Documents/papers/digest").expanduser())
    max_results: int = 10
    arxiv_cats: list[tuple[str, int]] = field(default_factory=lambda: list(_DEFAULT_ARXIV_CATS))

    # ── RAG ───────────────────────────────────────────────────────────────────
    rag_dir: Path = field(default_factory=lambda: Path("~/.seshat/rag").expanduser())
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

    # ── Chat ──────────────────────────────────────────────────────────────────
    provider: str = "ollama"  # "ollama" | "anthropic"
    vault_path: Path = field(default_factory=lambda: Path("~/vault").expanduser())
    # Vault folders whose contents are treated as private (local Ollama only)
    private_vault_dirs: list[str] = field(default_factory=lambda: ["private"])

    # ── Auth ──────────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""


def load_config(config_file: Path = CONFIG_FILE) -> Config:
    """Load a Config, applying TOML file values then env var overrides."""
    cfg = Config()

    if config_file.exists():
        with open(config_file, "rb") as f:
            data = tomllib.load(f)

        d = data.get("digest", {})
        if "ollama_model" in d:
            cfg.ollama_model = str(d["ollama_model"])
        if "anthropic_model" in d:
            cfg.anthropic_model = str(d["anthropic_model"])
        if "output_dir" in d:
            cfg.output_dir = Path(str(d["output_dir"])).expanduser()
        if "max_results" in d:
            cfg.max_results = int(d["max_results"])
        if "arxiv_categories" in d:
            cfg.arxiv_cats = [(str(c[0]), int(c[1])) for c in d["arxiv_categories"]]

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

        c = data.get("chat", {})
        if "provider" in c:
            cfg.provider = str(c["provider"])
        if "vault_path" in c:
            cfg.vault_path = Path(str(c["vault_path"])).expanduser()
        if "private_vault_dirs" in c:
            cfg.private_vault_dirs = [str(d) for d in c["private_vault_dirs"]]

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

    return cfg


_config: Config | None = None


def get_config() -> Config:
    """Return the process-wide Config singleton."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
