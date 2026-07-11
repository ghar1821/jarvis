"""
Tests for the response-style setting: system-prompt wiring
(jarvis.chat.chat.build_system_prompt) and the tomlkit-based config
write-back (jarvis.core.config.set_config_value / reset_config).
"""

import jarvis.core.config as config_mod
from jarvis.core.config import load_config, reset_config, set_config_value
from jarvis.chat.chat import build_system_prompt


# ── System prompt ──────────────────────────────────────────────────────────────

def test_prompt_contains_style_instruction():
    """A non-empty response_style lands as a delimited instruction block."""
    prompt = build_system_prompt(response_style="short and concise, no filler")
    assert "Response style (user preference): short and concise, no filler" in prompt


def test_prompt_without_style_adds_nothing():
    """Empty/whitespace style leaves the prompt free of the style block."""
    assert "Response style" not in build_system_prompt(response_style="")
    assert "Response style" not in build_system_prompt(response_style="   ")


def test_prompt_lists_skills():
    """Skills appear as name + description lines with the read_skill hint."""
    prompt = build_system_prompt(
        skills=[("paper-review", "Checklist for reviewing a methods paper")]
    )
    assert "read_skill" in prompt
    assert "- paper-review: Checklist for reviewing a methods paper" in prompt
    assert "Available skills" not in build_system_prompt(skills=[])


# ── Config write-back ──────────────────────────────────────────────────────────

def test_set_config_value_preserves_comments_and_other_keys(tmp_path):
    """
    Writing one key must round-trip everything else byte-for-byte —
    comments, unrelated sections, formatting.
    """
    config_file = tmp_path / "config.toml"
    original = (
        "# my precious config\n"
        "[digest]\n"
        "max_results = 7  # tuned by hand\n"
        "\n"
        "[chat]\n"
        'provider = "ollama"\n'
    )
    config_file.write_text(original)

    set_config_value("chat", "response_style", "be terse", config_file=config_file)

    text = config_file.read_text()
    assert "# my precious config" in text
    assert "max_results = 7  # tuned by hand" in text
    assert 'response_style = "be terse"' in text

    cfg = load_config(config_file)
    assert cfg.response_style == "be terse"
    assert cfg.max_results == 7
    # File holds credentials in real life — must end up private.
    assert (config_file.stat().st_mode & 0o777) == 0o600
    # No stray temp file from the atomic write.
    assert list(tmp_path.glob("*.tmp")) == []


def test_set_config_value_creates_missing_file_and_section(tmp_path):
    """A fresh config file and section are created on demand."""
    config_file = tmp_path / "config.toml"
    set_config_value("chat", "response_style", "flowery", config_file=config_file)
    assert load_config(config_file).response_style == "flowery"


def test_reset_config_reloads_singleton(tmp_path, monkeypatch):
    """
    After reset_config(), the next get_config() re-reads the file — the
    webapp relies on this after persisting a settings change.
    """
    config_file = tmp_path / "config.toml"
    config_file.write_text('[chat]\nresponse_style = "first"\n')
    monkeypatch.setattr(config_mod, "CONFIG_FILE", config_file)
    monkeypatch.setattr(config_mod, "_config", None)

    assert config_mod.get_config().response_style == "first"

    set_config_value("chat", "response_style", "second", config_file=config_file)
    # Singleton still holds the old value until reset.
    assert config_mod.get_config().response_style == "first"
    reset_config()
    assert config_mod.get_config().response_style == "second"