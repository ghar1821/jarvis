"""
Tests for jarvis/chat/skills.py — user-defined skill folders.
"""

from jarvis.chat.skills import list_skills, read_skill


def _write_skill(skills_dir, name, content, extra_files=None):
    """Create skills_dir/<name>/SKILL.md with content, plus any extra files
    (mapping of relative-path -> content) under the same folder."""
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(content)
    for rel_path, file_content in (extra_files or {}).items():
        path = skill_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(file_content)


def test_list_skills_frontmatter_description(tmp_path):
    """description: in --- frontmatter wins over the body."""
    _write_skill(
        tmp_path,
        "paper-review",
        "---\ndescription: Checklist for reviewing a methods paper\n---\n\n# Steps\n\nDo things.",
    )
    assert list_skills(tmp_path) == [
        ("paper-review", "Checklist for reviewing a methods paper"),
    ]


def test_list_skills_fallback_to_first_body_line(tmp_path):
    """No frontmatter (or no description key) falls back to the first non-empty line."""
    _write_skill(tmp_path, "weekly-summary", "\n# Format for the weekly digest note\nMore text")
    _write_skill(
        tmp_path,
        "no-desc-key",
        "---\nauthor: someone\n---\n\nFormat for something else\n",
    )
    skills = list_skills(tmp_path)
    assert skills == [
        ("no-desc-key", "Format for something else"),
        ("weekly-summary", "Format for the weekly digest note"),
    ]


def test_list_skills_unclosed_frontmatter_gives_empty_description(tmp_path):
    """An opening --- fence that never closes yields an empty description,
    not the literal '---' or a stray key: value line."""
    _write_skill(tmp_path, "broken", "---\nauthor: someone\n")
    assert list_skills(tmp_path) == [("broken", "")]


def test_list_skills_missing_dir_is_feature_off(tmp_path):
    """A nonexistent skills dir simply disables the feature."""
    assert list_skills(tmp_path / "no-such-dir") == []


def test_list_skills_warns_and_skips_stray_flat_file(tmp_path, capsys):
    """A leftover flat *.md file no longer loads and prints a one-line warning."""
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "old-style.md").write_text("# An old skill\n\nSteps...")
    _write_skill(tmp_path, "new-style", "# A folder skill\n\nSteps...")

    skills = list_skills(tmp_path)

    assert skills == [("new-style", "A folder skill")]
    warning = capsys.readouterr().out
    assert "old-style.md" in warning
    assert "SKILL.md" in warning


def test_list_skills_warns_and_skips_folder_without_skill_md(tmp_path, capsys):
    """A skill folder missing SKILL.md warns and is skipped."""
    (tmp_path / "incomplete").mkdir(parents=True)
    (tmp_path / "incomplete" / "notes.md").write_text("just some notes")
    _write_skill(tmp_path, "complete", "# A complete skill\n\nSteps...")

    skills = list_skills(tmp_path)

    assert skills == [("complete", "A complete skill")]
    warning = capsys.readouterr().out
    assert "incomplete" in warning
    assert "SKILL.md" in warning


def test_read_skill_returns_content_with_no_supporting_files(tmp_path):
    """When the skill folder has only SKILL.md, no 'Supporting files:' section appears."""
    _write_skill(tmp_path, "labmeeting", "# Lab meeting summary\n\n1. Context\n2. Findings\n")
    content = read_skill("labmeeting", tmp_path)
    assert "1. Context" in content and "2. Findings" in content
    assert "Supporting files:" not in content


def test_read_skill_lists_supporting_files_sorted(tmp_path):
    """Supporting files (any depth) are listed, sorted, relative to the skill folder."""
    _write_skill(
        tmp_path,
        "paper-review",
        "# Reviewing papers\n",
        extra_files={
            "template.md": "template body",
            "reference/checklist.md": "checklist body",
        },
    )
    content = read_skill("paper-review", tmp_path)
    assert "Supporting files:" in content
    listing = content.split("Supporting files:")[1]
    assert listing.index("reference/checklist.md") < listing.index("template.md")


def test_read_skill_with_file_returns_supporting_file_content(tmp_path):
    """file= reads one supporting file instead of SKILL.md."""
    _write_skill(
        tmp_path,
        "paper-review",
        "# Reviewing papers\n",
        extra_files={"reference/checklist.md": "1. Check the abstract\n2. Check the methods\n"},
    )
    content = read_skill("paper-review", tmp_path, file="reference/checklist.md")
    assert "Check the abstract" in content
    assert "Reviewing papers" not in content


def test_read_skill_rejects_traversal_names(tmp_path):
    """
    Skill names come from the LLM — separators and traversal sequences are
    refused before touching the filesystem.
    """
    _write_skill(tmp_path, "real", "A real skill")
    for bad in ("../secrets", "a/b", "a\\b", "..", ""):
        result = read_skill(bad, tmp_path)
        assert result.startswith("[Error: invalid skill name")


def test_read_skill_rejects_traversal_in_file_arg(tmp_path):
    """The file= argument is untrusted too — absolute paths and .. are refused."""
    _write_skill(tmp_path, "real", "A real skill", extra_files={"notes.md": "notes"})
    for bad in ("../../etc/passwd", "/etc/passwd", "..", "sub/../../escape.md"):
        result = read_skill("real", tmp_path, file=bad)
        assert result.startswith("[Error: invalid supporting file path")


def test_read_skill_rejects_symlink_escape_in_file_arg(tmp_path):
    """A supporting-file symlink pointing outside the skill folder is rejected."""
    outside_secret = tmp_path / "outside-secret.txt"
    outside_secret.write_text("top secret contents")

    _write_skill(tmp_path, "real", "A real skill")
    (tmp_path / "real" / "escape-link").symlink_to(outside_secret)

    result = read_skill("real", tmp_path, file="escape-link")
    assert result.startswith("[Error:")
    assert "top secret" not in result


def test_read_skill_rejects_directory_symlink_escape_in_file_arg(tmp_path):
    """A symlinked DIRECTORY inside the skill folder can't be used to escape either."""
    outside_dir = tmp_path / "outside-dir"
    outside_dir.mkdir()
    (outside_dir / "SECRET.txt").write_text("top secret contents")

    _write_skill(tmp_path, "real", "A real skill")
    (tmp_path / "real" / "dlink").symlink_to(outside_dir, target_is_directory=True)

    result = read_skill("real", tmp_path, file="dlink/SECRET.txt")
    assert result.startswith("[Error:")
    assert "top secret" not in result


def test_read_skill_file_arg_cannot_reach_another_skill(tmp_path):
    """file= containment is against the named skill's folder, not just skills_dir."""
    _write_skill(tmp_path, "real", "A real skill")
    _write_skill(tmp_path, "other", "Another skill's private instructions")

    result = read_skill("real", tmp_path, file="../other/SKILL.md")
    assert result.startswith("[Error:")
    assert "Another skill" not in result


def test_read_skill_unknown_name_lists_available(tmp_path):
    """An unknown skill errors helpfully, naming what exists."""
    _write_skill(tmp_path, "paper-review", "Reviewing")
    result = read_skill("nonexistent", tmp_path)
    assert "unknown skill" in result
    assert "paper-review" in result


def test_read_skill_unknown_file_errors(tmp_path):
    """An unknown supporting file name errors rather than falling back silently."""
    _write_skill(tmp_path, "real", "A real skill")
    result = read_skill("real", tmp_path, file="does-not-exist.md")
    assert result.startswith("[Error:")
    assert "does-not-exist.md" in result


def test_read_skill_oversize_supporting_file_rejected(tmp_path):
    """Supporting-file reads are capped (~64 KB) and fail visibly when exceeded."""
    _write_skill(tmp_path, "real", "A real skill")
    big_file = tmp_path / "real" / "big.md"
    big_file.write_text("x" * (64 * 1024 + 1))

    result = read_skill("real", tmp_path, file="big.md")
    assert result.startswith("[Error:")
    assert "64" in result
