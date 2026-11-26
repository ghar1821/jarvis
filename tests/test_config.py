"""
Tests for jarvis/core/config.py — load_config() and the README's promises.

Each test writes its own config.toml into tmp_path and passes it to
load_config() directly, so the user's real ~/.jarvis/config.toml is never
read. Config notes (moved and obsolete keys) go to stderr, because stdout is
the MCP channel inside jarvis-mcp.
"""

import re
import tomllib
from pathlib import Path

from jarvis.core.config import Config, format_describe, load_config

ROOT = Path(__file__).parent.parent


def _write(tmp_path: Path, text: str) -> Path:
    config_file = tmp_path / "config.toml"
    config_file.write_text(text)
    return config_file


def test_defaults_when_no_config_file(tmp_path):
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg == Config()
    assert cfg.papers_dir is None
    assert cfg.private_vault_dirs == ["private"]
    assert cfg.sync_minutes == 30 and cfg.hybrid is True


def test_kb_and_rag_sections_are_read(tmp_path):
    cfg = load_config(_write(tmp_path, """
[kb]
vault_path = "~/notes"
private_vault_dirs = ["private", "journal"]
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
    assert cfg.private_vault_dirs == ["private", "journal"]
    assert cfg.sync_minutes == 10
    assert cfg.rag_dir == Path.home() / "index"
    assert cfg.embed_model == "BAAI/bge-base-en-v1.5"
    assert cfg.rerank_model == "" and cfg.hybrid is False and cfg.server_port == 9000


def test_moved_keys_still_work_from_their_old_place_and_say_so(tmp_path, capsys):
    """
    An existing config keeps working: [chat] vault_path and [sync]
    pdf_watch_dir are read as [kb] vault_path and papers_dir, with a note.
    """
    cfg = load_config(_write(tmp_path, """
[chat]
vault_path = "~/old-vault"
private_vault_dirs = ["secret"]

[sync]
pdf_watch_dir = "~/old-inbox"
vault_refresh_minutes = 15
"""))
    assert cfg.vault_path == Path.home() / "old-vault"
    assert cfg.private_vault_dirs == ["secret"]
    assert cfg.papers_dir == Path.home() / "old-inbox"
    assert cfg.sync_minutes == 15

    captured = capsys.readouterr()
    assert captured.out == ""  # stdout is the MCP channel
    assert "[sync] pdf_watch_dir has moved to [kb] papers_dir" in captured.err
    assert "[chat] vault_path has moved to [kb] vault_path" in captured.err


def test_the_new_location_wins_over_the_old_one(tmp_path):
    cfg = load_config(_write(tmp_path, """
[chat]
vault_path = "~/old-vault"

[kb]
vault_path = "~/new-vault"
"""))
    assert cfg.vault_path == Path.home() / "new-vault"


def test_obsolete_keys_are_reported_once_and_ignored(tmp_path, capsys):
    """
    Settings from the old assistant (providers, API keys, drafts, digest
    schedule, figure captions) do nothing now. Each is named so a setting
    that silently stopped working is never a mystery — and an API key left
    in the file is not echoed.
    """
    load_config(_write(tmp_path, """
[chat]
provider = "anthropic"
vault_path = "~/vault"

[auth]
api_key = "sk-ant-should-never-be-printed"

[digest]
enabled = true

[rag]
figure_captions = true
"""))
    err = capsys.readouterr().err
    assert "[chat] provider in" in err and "no longer does anything" in err
    assert "[auth] api_key" in err and "sk-ant-should-never-be-printed" not in err
    assert "[digest] enabled" in err
    assert "[rag] figure_captions" in err
    # A moved key is reported as moved, not as obsolete.
    obsolete_chat_line = next(line for line in err.splitlines() if "[chat] provider" in line)
    assert "vault_path" not in obsolete_chat_line
    assert "[chat] vault_path has moved" in err


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
    "moved" or "obsolete" notes and produce absolute paths.
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
