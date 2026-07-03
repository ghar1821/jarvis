"""
Tests for digest/config.py — load_config() resolution order.

load_config() builds a Config in three layers, each overriding the previous:
  1. Built-in defaults (hardcoded in the Config dataclass)
  2. ~/.seshat/config.toml values
  3. Environment variable values

All tests call load_config() with an explicit config_file path so the real
~/.seshat/config.toml on disk is never read. The monkeypatch fixture restores
any environment variables changed during a test.
"""

import pytest

from digest.config import load_config


def test_defaults_when_no_config_file(tmp_path):
    """
    When the config file does not exist every field comes from the built-in defaults.

    Input:  path to a non-existent file
    Expected output:
        ollama_model    == "gemma4:26b"
        provider        == "ollama"
        chunk_size      == 1024
        chunk_overlap   == 128
        embed_model     == "BAAI/bge-small-en-v1.5"
        query_prefix    == the BGE search instruction
        rerank_model    == "cross-encoder/ms-marco-MiniLM-L6-v2"
        rerank_top_n    == 25
    """
    cfg = load_config(tmp_path / "nonexistent.toml")
    assert cfg.ollama_model == "gemma4:26b"
    assert cfg.provider == "ollama"
    assert cfg.chunk_size == 1024
    assert cfg.chunk_overlap == 128
    assert cfg.embed_model == "BAAI/bge-small-en-v1.5"
    assert cfg.query_prefix == "Represent this sentence for searching relevant passages: "
    assert cfg.rerank_model == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert cfg.rerank_top_n == 25


def test_toml_values_override_defaults(tmp_path):
    """
    Values present in the TOML file replace the built-in defaults.
    Fields not mentioned in the TOML stay at their defaults.

    Input:  config.toml with [digest] ollama_model = "llama3.2" and max_results = 5
    Expected output:
        ollama_model == "llama3.2"    (overridden)
        max_results  == 5             (overridden)
        provider     == "ollama"      (default, untouched)
    """
    config_file = tmp_path / "config.toml"
    config_file.write_text('[digest]\nollama_model = "llama3.2"\nmax_results = 5\n')
    cfg = load_config(config_file)
    assert cfg.ollama_model == "llama3.2"
    assert cfg.max_results == 5
    assert cfg.provider == "ollama"


def test_rag_settings_override_defaults(tmp_path):
    """
    The [rag] section fields — including the retrieval-tuning ones — are read
    from the TOML. An empty rerank_model is honoured as the disable switch.

    Input:  config.toml [rag] overriding embed_model, query_prefix, rerank_model,
            rerank_top_n
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
    )
    cfg = load_config(config_file)
    assert cfg.embed_model == "all-MiniLM-L6-v2"
    assert cfg.query_prefix == ""
    assert cfg.rerank_model == ""
    assert cfg.rerank_top_n == 40


def test_env_var_overrides_toml(tmp_path, monkeypatch):
    """
    Environment variables win over TOML values when both are present.

    Input:  TOML sets ollama_model = "llama3.2"; env var OLLAMA_MODEL = "mistral"
    Expected output:
        ollama_model == "mistral"   (env var wins)
    """
    config_file = tmp_path / "config.toml"
    config_file.write_text('[digest]\nollama_model = "llama3.2"\n')
    monkeypatch.setenv("OLLAMA_MODEL", "mistral")
    cfg = load_config(config_file)
    assert cfg.ollama_model == "mistral"


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
