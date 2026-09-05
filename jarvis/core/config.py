"""
Central configuration for jarvis.

Resolution order (later wins):
  1. Built-in defaults
  2. ~/.jarvis/config.toml
  3. Environment variables

Example ~/.jarvis/config.toml:

    [digest]
    # The weekly paper digest is off by default — jarvis is a general
    # assistant, and fetching arXiv on a schedule is opt-in.
    enabled = false
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
    provider = "ollama"          # "ollama" | "anthropic" | "openrouter"
    anthropic_model = "claude-sonnet-4-6"
    # Default model when provider = "openrouter"
    openrouter_model = "anthropic/claude-sonnet-4.6"
    # Ollama model tag (must support tool calling; vision for figure captioning)
    ollama_model = "qwen3-vl:30b"
    vault_path = "~/vault"
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

    [drafts]
    # The agent-writable sandbox. It sits OUTSIDE the vault on purpose: the
    # boundary between "the agent may write here" and "only you may put things
    # here" is a filesystem fact, not a config rule that could be mistyped.
    dir = "~/.jarvis/drafts"
    extensions = [".md", ".tex", ".bib", ".txt", ".csv"]
    max_file_bytes = 2000000
    retention_days = 30          # 0 disables the sweep entirely
    gc_hour = 4                  # daily sweep slot
    latex_engine = "latexmk"     # "" disables .tex compilation
    pdf_engine = "xelatex"       # engine pandoc drives for Markdown -> PDF
    compile_timeout_seconds = 180   # ceiling on one LaTeX/pandoc run
    pdf_margin = "2cm"           # margin for Markdown -> PDF export

    [openrouter]
    # OpenRouter is a broker: without these it may route a request to any
    # upstream inference provider it has a route for. The defaults are strict.
    data_collection = "deny"     # exclude providers that train on prompts
    allow_fallbacks = false      # never silently reroute to an unvetted provider
    only = []                    # optional allowlist of upstream provider slugs

    [models]
    # The switchable catalogue shown by /model and the webapp picker. Entirely
    # user-maintained — jarvis ships no vendor model list. Populate the
    openrouter = ["anthropic/claude-sonnet-4.6", "openai/gpt-5"]
    ollama = ["qwen3-vl:30b"]

    [auth]
    api_key = "sk-ant-..."         # Anthropic API key (or ANTHROPIC_API_KEY env var)
    openrouter_api_key = "sk-or-..."  # OpenRouter key (or OPENROUTER_API_KEY env var)
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
    # Off unless the user opts in. Everything below only matters when it's on:
    # the daemon skips both digest jobs and `run-digest` refuses to fetch.
    digest_enabled: bool = False
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
    # "ollama" | "anthropic" | "openrouter", optionally "<provider>:<model>"
    provider: str = "ollama"
    # Default model when the provider is openrouter and no model is named.
    # Empty by default: there is no sensible default model for a broker that
    # fronts hundreds of them, so jarvis asks rather than guessing.
    openrouter_model: str = ""
    # Ollama model tag. qwen3-vl:30b is a vision + thinking MoE (3.3B active
    # params) that fits comfortably in 36GB on an M3 Max. Confirm the exact
    # registry tag with `ollama list` — Ollama's naming can shift over time.
    ollama_model: str = "qwen3-vl:30b"
    vault_path: Path = field(default_factory=lambda: Path("~/vault").expanduser())
    # Vault folders whose contents are treated as private (local model only)
    private_vault_dirs: list[str] = field(default_factory=lambda: ["private"])
    # Free-text style instruction appended to the system prompt ("" = none)
    response_style: str = ""
    # Session compaction: summarise old exchanges once the estimated context
    # passes compact_after_tokens, keeping the last compact_keep_exchanges verbatim
    compact_after_tokens: int = 12000
    compact_keep_exchanges: int = 6

    # ── Drafts sandbox ───────────────────────────────────────────────────────
    # The only place on disk the model can write. Outside the vault so the
    # boundary is physical rather than a config rule.
    drafts_dir: Path = field(default_factory=lambda: Path("~/.jarvis/drafts").expanduser())
    drafts_extensions: list[str] = field(
        default_factory=lambda: [".md", ".tex", ".bib", ".txt", ".csv"]
    )
    drafts_max_file_bytes: int = 2_000_000
    # Drafts untouched for this long are removed by the daemon's sweep. This is
    # the only file deletion anywhere in jarvis; 0 disables it.
    drafts_retention_days: int = 30
    drafts_gc_hour: int = 4
    # Compilation and PDF export run locally with no model in the loop, which
    # is why they are allowed on a private draft. "" hides the compile button.
    latex_engine: str = "latexmk"
    pdf_engine: str = "xelatex"
    # Ceiling on one LaTeX or pandoc run. Three minutes rather than one because
    # the first Markdown export on a machine pays for fontspec building its font
    # cache, which on a laptop can take a couple of minutes on its own — a limit
    # tight enough to kill that turns a slow first run into a broken feature.
    compile_timeout_seconds: int = 180
    # Margin for Markdown -> PDF export. pandoc's default leaves an inch and a
    # half of white space on every side, which wastes most of the page.
    pdf_margin: str = "2cm"

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

    # ── OpenRouter routing ───────────────────────────────────────────────────
    # Sent with every OpenRouter request. Strict by default: a broker that can
    # route to any upstream provider should not do so silently.
    openrouter_data_collection: str = "deny"
    openrouter_allow_fallbacks: bool = False
    openrouter_only: list[str] = field(default_factory=list)

    # ── Model catalogue ──────────────────────────────────────────────────────
    # {provider: [model, ...]} offered by /model and the webapp picker.
    # User-maintained — jarvis never hardcodes a vendor model list.
    models: dict[str, list[str]] = field(default_factory=dict)

    # ── Auth ──────────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    openrouter_api_key: str = ""


def load_config(config_file: Path = CONFIG_FILE) -> Config:
    """Load a Config, applying TOML file values then env var overrides."""
    cfg = Config()

    if config_file.exists():
        with open(config_file, "rb") as f:
            data = tomllib.load(f)

        d = data.get("digest", {})
        if "enabled" in d:
            cfg.digest_enabled = bool(d["enabled"])
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
        if "openrouter_model" in c:
            cfg.openrouter_model = str(c["openrouter_model"])
        if "vault_path" in c:
            cfg.vault_path = Path(str(c["vault_path"])).expanduser()
        if "private_vault_dirs" in c:
            cfg.private_vault_dirs = [str(d) for d in c["private_vault_dirs"]]
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

        dr = data.get("drafts", {})
        if "dir" in dr:
            cfg.drafts_dir = Path(str(dr["dir"])).expanduser()
        if "extensions" in dr:
            cfg.drafts_extensions = [str(ext).lower() for ext in dr["extensions"]]
        if "max_file_bytes" in dr:
            cfg.drafts_max_file_bytes = int(dr["max_file_bytes"])
        if "retention_days" in dr:
            cfg.drafts_retention_days = int(dr["retention_days"])
        if "gc_hour" in dr:
            cfg.drafts_gc_hour = int(dr["gc_hour"])
        if "latex_engine" in dr:
            cfg.latex_engine = str(dr["latex_engine"])
        if "pdf_engine" in dr:
            cfg.pdf_engine = str(dr["pdf_engine"])
        if "compile_timeout_seconds" in dr:
            cfg.compile_timeout_seconds = int(dr["compile_timeout_seconds"])
        if "pdf_margin" in dr:
            cfg.pdf_margin = str(dr["pdf_margin"])

        o = data.get("openrouter", {})
        if "data_collection" in o:
            cfg.openrouter_data_collection = str(o["data_collection"])
        if "allow_fallbacks" in o:
            cfg.openrouter_allow_fallbacks = bool(o["allow_fallbacks"])
        if "only" in o:
            cfg.openrouter_only = [str(slug) for slug in o["only"]]

        # [models] is the user's own switchable catalogue: every key is a
        # provider name, every value a list of model names. Nothing is
        # validated against a vendor list — jarvis does not keep one.
        m = data.get("models", {})
        if m:
            cfg.models = {
                str(provider): [str(name) for name in names]
                for provider, names in m.items()
            }

        a = data.get("auth", {})
        if "api_key" in a:
            cfg.anthropic_api_key = str(a["api_key"])
        if "openrouter_api_key" in a:
            cfg.openrouter_api_key = str(a["openrouter_api_key"])

    # Env var overrides (always win over TOML)
    if v := os.environ.get("OPENROUTER_API_KEY"):
        cfg.openrouter_api_key = v
    if v := os.environ.get("OPENROUTER_MODEL"):
        cfg.openrouter_model = v
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


# ── Describing the loaded config ──────────────────────────────────────────────
#
# One description, rendered two ways: printed when the webapp starts, and shown
# in the UI. Sharing it means the terminal and the browser can never disagree
# about what jarvis actually loaded, which is the whole point of showing it.

def _secret_values(cfg) -> list[str]:
    """The actual secrets, so nothing can render one by accident."""
    import os

    values = [cfg.anthropic_api_key, cfg.openrouter_api_key]
    # Env vars too: a key supplied that way never reaches Config, but it is
    # just as capable of turning up in an SDK error message.
    values += [os.environ.get("ANTHROPIC_API_KEY", ""), os.environ.get("OPENROUTER_API_KEY", "")]
    # A short value would match far too much text to blank out safely.
    return [v for v in values if v and len(v) >= 12]


def redact_secrets(text: str, cfg: "Config | None" = None) -> str:
    """
    Blank any configured API key out of arbitrary text.

    Error messages are the leak that matters. An SDK exception can quote the
    credential it just failed with, and jarvis puts that text in three durable
    places at once: the reply shown to the user, ~/.jarvis/logs/chat.log, and
    the saved session — which is then indexed into the vector store as a chat
    chunk. A key reaching any of those outlives the request that produced it.
    """
    if not text:
        return text
    try:
        secrets = _secret_values(cfg or get_config())
    except Exception:
        return text          # never let redaction itself break error reporting
    for secret in secrets:
        text = text.replace(secret, "[REDACTED]")
    return text


def _render(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None or value == "":
        return "—"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value) if value else "—"
    if isinstance(value, dict):
        return ", ".join(f"{k}: {', '.join(v)}" for k, v in value.items()) if value else "—"
    return str(value)


def describe(cfg: "Config | None" = None) -> list[dict]:
    """
    The loaded configuration, grouped for display, with secrets redacted.

    Values are the *resolved* ones — after the TOML file and any environment
    variables have been applied — so this answers "what is jarvis actually
    using", not "what does my file say". Those differ often enough to be the
    usual source of confusion.
    """
    cfg = cfg or get_config()

    def secret(value) -> str:
        return "set" if value else "not set"

    groups = [
        ("Source", [
            ("config file", str(CONFIG_FILE)),
            ("exists", CONFIG_FILE.exists()),
        ]),
        ("Chat", [
            ("provider", cfg.provider),
            ("ollama_model", cfg.ollama_model),
            ("anthropic_model", cfg.anthropic_model),
            ("openrouter_model", cfg.openrouter_model),
            ("response_style", cfg.response_style),
        ]),
        ("Models offered in the picker", [
            (provider, models) for provider, models in (cfg.models or {}).items()
        ] or [("[models]", "not set — the picker offers each provider's default")]),
        ("Credentials", [
            ("anthropic_api_key", secret(cfg.anthropic_api_key)),
            ("openrouter_api_key", secret(cfg.openrouter_api_key)),
        ]),
        ("OpenRouter routing", [
            ("data_collection", cfg.openrouter_data_collection),
            ("allow_fallbacks", cfg.openrouter_allow_fallbacks),
            ("only", cfg.openrouter_only),
        ]),
        ("Knowledge base", [
            ("vault_path", cfg.vault_path),
            ("private_vault_dirs", cfg.private_vault_dirs),
            ("rag_dir", cfg.rag_dir),
            ("embed_model", cfg.embed_model),
            ("rerank_model", cfg.rerank_model),
            ("hybrid", cfg.hybrid),
            ("chunk_size", cfg.chunk_size),
            ("figure_captions", cfg.figure_captions),
        ]),
        ("Drafts", [
            ("drafts_dir", cfg.drafts_dir),
            ("retention_days", cfg.drafts_retention_days),
            ("latex_engine", cfg.latex_engine),
            ("pdf_engine", cfg.pdf_engine),
        ]),
        ("Sync daemon", [
            ("vault_refresh_minutes", cfg.vault_refresh_minutes),
            ("pdf_watch_dir", cfg.pdf_watch_dir),
            ("pdf_watch_minutes", cfg.pdf_watch_minutes),
        ]),
        ("Paper digest", [
            ("enabled", cfg.digest_enabled),
            ("day / hour", f"{cfg.digest_day} {cfg.digest_hour:02d}:00"),
            ("output_dir", cfg.output_dir),
            ("max_results", cfg.max_results),
        ]),
    ]

    # Credentials are already reduced to "set"/"not set" above. This is the
    # backstop: if a field is ever added to a group raw, it still cannot reach
    # a terminal or a browser tab as its actual value.
    secrets = _secret_values(cfg)

    def safe(value: str) -> str:
        return "set" if any(s and s in value for s in secrets) else value

    return [
        {
            "section": name,
            "values": [{"key": key, "value": safe(_render(value))} for key, value in rows],
        }
        for name, rows in groups
    ]


def format_describe(cfg: "Config | None" = None) -> str:
    """The same description as plain text, for printing at startup."""
    lines = []
    for group in describe(cfg):
        lines.append(f"\n  {group['section']}")
        width = max((len(v["key"]) for v in group["values"]), default=0)
        for value in group["values"]:
            lines.append(f"    {value['key']:<{width}}  {value['value']}")
    return "\n".join(lines)
