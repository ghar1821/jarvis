"""
Central configuration for paper_digest.

Resolution order (later wins):
  1. Built-in defaults
  2. ~/.paper_digest/config.toml
  3. Environment variables

Example ~/.paper_digest/config.toml:

    [digest]
    ollama_model = "gemma4:26b"
    anthropic_model = "claude-sonnet-4-6"
    output_dir = "~/Documents/papers/digest"
    max_results = 10
    # arxiv_categories is a list of [category, limit] pairs:
    # arxiv_categories = [["cs.LG", 150], ["cs.AI", 80]]

    [rag]
    rag_dir = "~/.paper_digest/rag"
    embed_model = "all-MiniLM-L6-v2"
    chunk_size = 2048
    chunk_overlap = 256

    [chat]
    provider = "ollama"          # "ollama" | "anthropic"
    vault_path = "~/vault"

    [auth]
    oauth_client_id = ""
    oauth_auth_url = "https://claude.ai/oauth/authorize"
    oauth_token_url = "https://api.anthropic.com/oauth/token"
"""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_FILE = Path.home() / ".paper_digest" / "config.toml"

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
    rag_dir: Path = field(default_factory=lambda: Path("~/.paper_digest/rag").expanduser())
    embed_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 2048
    chunk_overlap: int = 256

    # ── Chat ──────────────────────────────────────────────────────────────────
    provider: str = "ollama"  # "ollama" | "anthropic"
    vault_path: Path = field(default_factory=lambda: Path("~/vault").expanduser())

    # ── Auth ──────────────────────────────────────────────────────────────────
    auth_file: Path = field(default_factory=lambda: Path("~/.paper_digest/auth.json").expanduser())
    oauth_client_id: str = ""
    oauth_auth_url: str = "https://claude.ai/oauth/authorize"
    oauth_token_url: str = "https://api.anthropic.com/oauth/token"


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
        if "chunk_size" in r:
            cfg.chunk_size = int(r["chunk_size"])
        if "chunk_overlap" in r:
            cfg.chunk_overlap = int(r["chunk_overlap"])

        c = data.get("chat", {})
        if "provider" in c:
            cfg.provider = str(c["provider"])
        if "vault_path" in c:
            cfg.vault_path = Path(str(c["vault_path"])).expanduser()

        a = data.get("auth", {})
        if "oauth_client_id" in a:
            cfg.oauth_client_id = str(a["oauth_client_id"])
        if "oauth_auth_url" in a:
            cfg.oauth_auth_url = str(a["oauth_auth_url"])
        if "oauth_token_url" in a:
            cfg.oauth_token_url = str(a["oauth_token_url"])

    # Env var overrides (always win over TOML)
    if v := os.environ.get("OLLAMA_MODEL"):
        cfg.ollama_model = v
    if v := os.environ.get("ANTHROPIC_MODEL"):
        cfg.anthropic_model = v
    if v := os.environ.get("CHAT_PROVIDER"):
        cfg.provider = v
    if v := os.environ.get("VAULT_PATH"):
        cfg.vault_path = Path(v).expanduser()
    if v := os.environ.get("ANTHROPIC_OAUTH_CLIENT_ID"):
        cfg.oauth_client_id = v
    if v := os.environ.get("ANTHROPIC_OAUTH_AUTH_URL"):
        cfg.oauth_auth_url = v
    if v := os.environ.get("ANTHROPIC_OAUTH_TOKEN_URL"):
        cfg.oauth_token_url = v

    return cfg


_config: Config | None = None


def get_config() -> Config:
    """Return the process-wide Config singleton."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
