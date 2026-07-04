"""
Tests for vault_chat/skills.py — user-defined skill files.
"""

from vault_chat.skills import list_skills, read_skill


def _write_skill(skills_dir, name, content):
    skills_dir.mkdir(exist_ok=True)
    (skills_dir / f"{name}.md").write_text(content)


def test_list_skills_names_and_descriptions(tmp_path):
    """
    Every *.md becomes (stem, first non-empty line with '#' stripped),
    sorted by name.
    """
    _write_skill(tmp_path, "paper-review", "# Checklist for reviewing a methods paper\n\nSteps...")
    _write_skill(tmp_path, "weekly-summary", "\nFormat for the weekly digest note\nMore text")

    skills = list_skills(tmp_path)
    assert skills == [
        ("paper-review", "Checklist for reviewing a methods paper"),
        ("weekly-summary", "Format for the weekly digest note"),
    ]


def test_list_skills_missing_dir_is_feature_off(tmp_path):
    """A nonexistent skills dir simply disables the feature."""
    assert list_skills(tmp_path / "no-such-dir") == []


def test_read_skill_returns_full_content(tmp_path):
    """read_skill returns the whole file, not just the description."""
    _write_skill(tmp_path, "labmeeting", "# Lab meeting summary\n\n1. Context\n2. Findings\n")
    content = read_skill("labmeeting", tmp_path)
    assert "1. Context" in content and "2. Findings" in content


def test_read_skill_rejects_traversal_names(tmp_path):
    """
    Skill names come from the LLM — separators and traversal sequences are
    refused before touching the filesystem.
    """
    _write_skill(tmp_path, "real", "A real skill")
    for bad in ("../secrets", "a/b", "a\\b", "..", ""):
        result = read_skill(bad, tmp_path)
        assert result.startswith("[Error: invalid skill name")


def test_read_skill_unknown_name_lists_available(tmp_path):
    """An unknown skill errors helpfully, naming what exists."""
    _write_skill(tmp_path, "paper-review", "Reviewing")
    result = read_skill("nonexistent", tmp_path)
    assert "unknown skill" in result
    assert "paper-review" in result