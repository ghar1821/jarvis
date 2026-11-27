"""
Tests for jarvis/core/config.py — load_config() and the README's promises.

Each test writes its own config.toml into tmp_path and passes it to
load_config() directly, so the user's real ~/.jarvis/config.toml is never
read. Config notes (unknown keys) go to stderr, because stdout is the MCP
channel inside jarvis-mcp.
"""

import re
import tomllib
from pathlib import Path

import pytest

from jarvis.core.config import Config, format_describe, load_config
from jarvis.core.errors import ConfigError

ROOT = Path(__file__).parent.parent


def _write(tmp_path: Path, text: str) -> Path:
    config_file = tmp_path / "config.toml"
    config_file.write_text(text)
    return config_file


def test_defaults_when_no_config_file(tmp_path):
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg == Config()
    assert cfg.papers_dir is None
    assert cfg.sync_minutes == 30 and cfg.hybrid is True


def test_kb_and_rag_sections_are_read(tmp_path):
    cfg = load_config(_write(tmp_path, """
[kb]
vault_path = "~/notes"
papers_dir = "~/papers"
sync_minutes = 10

[rag]
rag_dir = "~/index"
embed_model = "BAAI/bge-base-en-v1.5"
rerank_model = ""
hybrid = false
server_port = 9000
"""))
    # "~" is expanded — a literal one would be a path that never resolves.
    assert cfg.vault_path == Path.home() / "notes"
    assert cfg.papers_dir == Path.home() / "papers"
    assert cfg.sync_minutes == 10
    assert cfg.rag_dir == Path.home() / "index"
    assert cfg.embed_model == "BAAI/bge-base-en-v1.5"
    assert cfg.rerank_model == "" and cfg.hybrid is False and cfg.server_port == 9000


@pytest.mark.parametrize("section", ["kb", "chat"])
def test_private_vault_dirs_stops_jarvis_with_the_fix(tmp_path, section):
    """
    jarvis no longer keeps private folders out of the index. A config that
    still lists them would quietly send those notes to the cloud model, so it
    is refused outright, with a message that says what to do instead.
    """
    config_file = _write(tmp_path, f'[{section}]\nprivate_vault_dirs = ["private"]\n')
    with pytest.raises(ConfigError) as caught:
        load_config(config_file)
    message = str(caught.value)
    assert "private_vault_dirs" in message
    assert "outside vault_path" in message
    assert str(config_file) in message


def test_an_empty_private_vault_dirs_is_harmless(tmp_path):
    """An empty list kept nothing out of the index, so it does not stop jarvis."""
    load_config(_write(tmp_path, "[kb]\nprivate_vault_dirs = []\n"))


def test_unknown_sections_and_keys_are_named_and_ignored(tmp_path, capsys):
    """
    A typo or a setting left over from an older version must not be mistaken
    for one that works. Each is named on stderr, and an API key left in the
    file is not echoed.
    """
    cfg = load_config(_write(tmp_path, """
[kb]
vault_pth = "~/typo"

[auth]
api_key = "sk-ant-should-never-be-printed"

[rag]
figure_captions = true
"""))
    assert cfg == Config()
    captured = capsys.readouterr()
    assert captured.out == ""  # stdout is the MCP channel
    assert "[kb] vault_pth" in captured.err
    assert "[auth] in" in captured.err
    assert "[rag] figure_captions" in captured.err
    assert "sk-ant-should-never-be-printed" not in captured.err


def test_a_quoted_boolean_is_an_error_not_true(tmp_path):
    """bool("false") is True, so a quoted "false" would silently turn hybrid on."""
    with pytest.raises(ConfigError, match="without quotes"):
        load_config(_write(tmp_path, '[rag]\nhybrid = "false"\n'))


def test_env_vars_override_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("VAULT_PATH", "~/env-vault")
    monkeypatch.setenv("PAPERS_DIR", "~/env-papers")
    cfg = load_config(_write(tmp_path, '[kb]\nvault_path = "~/file-vault"\n'))
    assert cfg.vault_path == Path.home() / "env-vault"
    assert cfg.papers_dir == Path.home() / "env-papers"


def test_format_describe_shows_the_resolved_folders(tmp_path):
    text = format_describe(Config(vault_path=tmp_path / "v", papers_dir=None))
    assert str(tmp_path / "v") in text
    assert "no papers indexed" in text
    assert "127.0.0.1 only" in text


def test_the_readme_example_config_loads_cleanly(tmp_path, capsys):
    """
    The config block a new user copies out of the README must load with no
    notes about unknown keys and produce absolute paths.
    """
    readme = (ROOT / "README.md").read_text()
    blocks = re.findall(r"```toml\n(.*?)```", readme, re.DOTALL)
    assert blocks, "the README should show an example config.toml"

    cfg = load_config(_write(tmp_path, blocks[0]))

    assert capsys.readouterr().err == ""
    assert cfg.vault_path.is_absolute()
    assert cfg.papers_dir is not None and cfg.papers_dir.is_absolute()


def test_the_readme_names_commands_that_exist():
    """
    Every `uv run <entry-point>` the README tells a new user to type has to be
    a real entry point, and every `kb <subcommand>` a real subcommand.
    """
    readme = (ROOT / "README.md").read_text()
    with open(ROOT / "pyproject.toml", "rb") as handle:
        scripts = set(tomllib.load(handle)["project"]["scripts"])

    named = set(re.findall(r"uv run ([a-z][a-z-]+)", readme)) - {"pytest", "python"}
    assert named <= scripts, f"README names commands that do not exist: {sorted(named - scripts)}"

    from jarvis.kb import cli

    subcommands = set(re.findall(r"^  (\w+)\s", cli.__doc__, re.MULTILINE))
    named_kb = set(re.findall(r"uv run kb ([a-z-]+)", readme))
    assert named_kb <= subcommands, f"README names kb commands that do not exist: {sorted(named_kb - subcommands)}"
