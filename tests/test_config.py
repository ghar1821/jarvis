"""
Tests for jarvis/core/config.py — load_config() resolution order.

load_config() builds a Config in three layers, each overriding the previous:
  1. Built-in defaults (hardcoded in the Config dataclass)
  2. ~/.jarvis/config.toml values
  3. Environment variable values

All tests call load_config() with an explicit config_file path so the real
~/.jarvis/config.toml on disk is never read. The monkeypatch fixture restores
any environment variables changed during a test.
"""

import pytest

from jarvis.core.config import load_config


def test_defaults_when_no_config_file(tmp_path):
    """
    When the config file does not exist every field comes from the built-in defaults.

    Input:  path to a non-existent file
    Expected output:
        ollama_model    == "qwen3-vl:30b"
        provider        == "ollama"
        chunk_size      == 1024
        chunk_overlap   == 128
        embed_model     == "BAAI/bge-small-en-v1.5"
        query_prefix    == the BGE search instruction
        rerank_model    == "cross-encoder/ms-marco-MiniLM-L6-v2"
        rerank_top_n    == 25
        figure_captions == False   (off by default — vision calls cost per figure)
    """
    cfg = load_config(tmp_path / "nonexistent.toml")
    assert cfg.ollama_model == "qwen3-vl:30b"
    assert cfg.provider == "ollama"
    assert cfg.chunk_size == 1024
    assert cfg.chunk_overlap == 128
    assert cfg.embed_model == "BAAI/bge-small-en-v1.5"
    assert cfg.query_prefix == "Represent this sentence for searching relevant passages: "
    assert cfg.rerank_model == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert cfg.rerank_top_n == 25
    assert cfg.figure_captions is False
    assert cfg.figure_max_per_doc == 20
    assert cfg.figure_min_pixels == 40000
    assert cfg.hybrid is True


def test_toml_values_override_defaults(tmp_path):
    """
    Values present in the TOML file replace the built-in defaults.
    Fields not mentioned in the TOML stay at their defaults.

    Input:  config.toml with [chat] ollama_model override and [digest] max_results = 5
    Expected output:
        ollama_model == "llava:13b"   (overridden)
        max_results  == 5             (overridden)
        provider     == "ollama"      (default, untouched)
    """
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        '[digest]\nmax_results = 5\n\n'
        '[chat]\nollama_model = "llava:13b"\n'
    )
    cfg = load_config(config_file)
    assert cfg.ollama_model == "llava:13b"
    assert cfg.max_results == 5
    assert cfg.provider == "ollama"


def test_rag_settings_override_defaults(tmp_path):
    """
    The [rag] section fields — including the retrieval-tuning and figure-caption
    ones — are read from the TOML. An empty rerank_model is honoured as the
    disable switch, and figure_captions can be turned on globally (it now
    defaults off).

    Input:  config.toml [rag] overriding embed_model, query_prefix, rerank_model,
            rerank_top_n, and the figure-caption knobs
    Expected output:
        each field takes the TOML value; rerank_model == "" (re-ranking off)
    """
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        '[rag]\n'
        'embed_model = "all-MiniLM-L6-v2"\n'
        'query_prefix = ""\n'
        'rerank_model = ""\n'
        'rerank_top_n = 40\n'
        'figure_captions = true\n'
        'figure_max_per_doc = 5\n'
        'figure_min_pixels = 10000\n'
    )
    cfg = load_config(config_file)
    assert cfg.embed_model == "all-MiniLM-L6-v2"
    assert cfg.query_prefix == ""
    assert cfg.rerank_model == ""
    assert cfg.rerank_top_n == 40
    assert cfg.figure_captions is True
    assert cfg.figure_max_per_doc == 5
    assert cfg.figure_min_pixels == 10000


def test_hybrid_defaults_true_and_parses_from_toml(tmp_path):
    """
    hybrid defaults to True (dense-only retrieval was the pre-hybrid
    behaviour) and can be turned off via [rag] hybrid = false.
    """
    cfg = load_config(tmp_path / "nonexistent.toml")
    assert cfg.hybrid is True

    config_file = tmp_path / "config.toml"
    config_file.write_text('[rag]\nhybrid = false\n')
    cfg = load_config(config_file)
    assert cfg.hybrid is False


def test_biorxiv_settings_override_defaults(tmp_path):
    """
    The [digest] bioRxiv keys parse into lists of (name, limit) tuples, the
    same shape as arxiv_categories.

    Input:  config.toml [digest] with biorxiv_categories/keywords/days
    Expected output:
        biorxiv_cats/keywords are lists of tuples; biorxiv_days overridden
    """
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        '[digest]\n'
        'biorxiv_categories = [["genomics", 40]]\n'
        'biorxiv_keywords = [["flow cytometry", 25]]\n'
        'biorxiv_days = 14\n'
    )
    cfg = load_config(config_file)
    assert cfg.biorxiv_cats == [("genomics", 40)]
    assert cfg.biorxiv_keywords == [("flow cytometry", 25)]
    assert cfg.biorxiv_days == 14


def test_env_var_overrides_toml(tmp_path, monkeypatch):
    """
    Environment variables win over TOML values when both are present.

    Input:  TOML sets ollama_model; env var OLLAMA_MODEL differs
    Expected output:
        ollama_model == env var value   (env var wins)
    """
    config_file = tmp_path / "config.toml"
    config_file.write_text('[chat]\nollama_model = "llava:13b"\n')
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3-vl:8b")
    cfg = load_config(config_file)
    assert cfg.ollama_model == "qwen3-vl:8b"


def test_tilde_in_paths_is_expanded(tmp_path):
    """
    Paths written with ~ in the TOML are expanded to absolute paths at load time.

    Input:  config.toml with output_dir = "~/my/papers"
    Expected output:
        output_dir is an absolute Path (does not start with "~")
    """
    config_file = tmp_path / "config.toml"
    config_file.write_text('[digest]\noutput_dir = "~/my/papers"\n')
    cfg = load_config(config_file)
    assert not str(cfg.output_dir).startswith("~")
    assert cfg.output_dir.is_absolute()


def test_sync_section_defaults_and_overrides(tmp_path):
    """
    The [sync] section drives the jarvis-sync daemon. Defaults leave the
    watcher off; TOML values override; PDF_WATCH_DIR env wins over TOML.

    Input:  defaults, then a TOML [sync] block, then the env var
    Expected output: each layer takes effect in order
    """
    cfg = load_config(tmp_path / "nonexistent.toml")
    assert cfg.pdf_watch_dir is None
    assert cfg.pdf_watch_minutes == 30
    assert cfg.vault_refresh_minutes == 30
    assert cfg.digest_day == "mon"
    assert cfg.digest_hour == 5

    config_file = tmp_path / "config.toml"
    config_file.write_text(
        '[sync]\n'
        'pdf_watch_dir = "~/papers/inbox"\n'
        'pdf_watch_minutes = 15\n'
        'vault_refresh_minutes = 10\n'
        'digest_day = "fri"\n'
        'digest_hour = 6\n'
    )
    cfg = load_config(config_file)
    assert cfg.pdf_watch_dir is not None
    assert not str(cfg.pdf_watch_dir).startswith("~")
    assert cfg.pdf_watch_minutes == 15
    assert cfg.vault_refresh_minutes == 10
    assert cfg.digest_day == "fri"
    assert cfg.digest_hour == 6


def test_pdf_watch_dir_env_override(tmp_path, monkeypatch):
    """
    PDF_WATCH_DIR env var overrides the TOML value.

    Input:  TOML sets one path; env var sets another
    Expected output: pdf_watch_dir == env var path (expanded)
    """
    config_file = tmp_path / "config.toml"
    config_file.write_text('[sync]\npdf_watch_dir = "~/from-toml"\n')
    monkeypatch.setenv("PDF_WATCH_DIR", str(tmp_path / "from-env"))
    cfg = load_config(config_file)
    assert cfg.pdf_watch_dir == tmp_path / "from-env"


def test_api_key_loaded_from_auth_section(tmp_path):
    """
    The [auth] api_key field is read into cfg.anthropic_api_key, allowing the
    key to be stored in the config file instead of an environment variable.

    Input:  config.toml with [auth] api_key = "sk-ant-test"
    Expected output:
        anthropic_api_key == "sk-ant-test"
    """
    config_file = tmp_path / "config.toml"
    config_file.write_text('[auth]\napi_key = "sk-ant-test"\n')
    cfg = load_config(config_file)
    assert cfg.anthropic_api_key == "sk-ant-test"
