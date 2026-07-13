"""
User-defined skills — reusable instruction files the agent loads on demand.

A skill is a folder `skills_dir/<name>/` containing `SKILL.md` plus any
supporting files the instructions reference (scripts, templates, reference
docs — at any depth under the folder). The folder name is the skill name.
The description comes from a `description:` key in SKILL.md's `---`
frontmatter, falling back to the first non-empty body line (leading '#'
stripped) when there is no frontmatter or no description key. The system
prompt advertises only name + description; the model calls the read_skill
tool to pull in the full instructions when a task matches — full skill text
never occupies context until it is actually needed.

Skills are the user's own local files: trusted, never indexed into the
vector store, and outside the public/private visibility model.
"""

from pathlib import Path

MAX_SUPPORTING_FILE_BYTES = 64 * 1024


def _parse_frontmatter_description(text: str) -> str | None:
    """
    Pull a `description:` value out of a leading `---` ... `---` frontmatter
    block. Single-line values only — this is a deliberately small hand-rolled
    parser, not a YAML implementation. Returns None when there is no
    frontmatter block or no description key inside it.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for line in lines[1:]:
        if line.strip() == "---":
            return None
        key, _, value = line.partition(":")
        if key.strip() == "description":
            return value.strip().strip("\"'")
    return None


def _skill_description(skill_md: Path) -> str:
    """Description for one SKILL.md: frontmatter first, then first body line."""
    text = skill_md.read_text(encoding="utf-8", errors="replace")
    description = _parse_frontmatter_description(text)
    if description:
        return description

    lines = text.splitlines()
    body_start = 0
    if lines and lines[0].strip() == "---":
        # Skip past the frontmatter block (if any) before scanning for a body line.
        body_start = None
        for i, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                body_start = i + 1
                break
        if body_start is None:
            # Opening fence never closed — everything reads as frontmatter, so
            # there is no body line to fall back to. Better an empty description
            # than advertising a stray "---" or a random key: value line.
            return ""

    for line in lines[body_start:]:
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped
    return ""


def list_skills(skills_dir: Path) -> list[tuple[str, str]]:
    """
    Return [(name, description)] for every `<skills_dir>/<name>/SKILL.md`,
    sorted by name. A missing or non-directory skills_dir means the feature
    is off — return [] without complaint.

    Only the folder format is supported. A stray flat `*.md` file, or a
    folder missing SKILL.md, prints a one-line warning naming the fix and is
    skipped rather than silently ignored.
    """
    if not skills_dir.is_dir():
        return []

    for stray in sorted(skills_dir.glob("*.md")):
        print(
            f"[skills] {stray.name} is a flat file — skills must be a folder "
            f"containing SKILL.md now. Fix: mkdir {stray.stem} && "
            f"mv {stray.name} {stray.stem}/SKILL.md"
        )

    skills = []
    for entry in sorted(skills_dir.iterdir()):
        if not entry.is_dir():
            continue
        skill_md = entry / "SKILL.md"
        if not skill_md.is_file():
            print(f"[skills] {entry.name}/ has no SKILL.md — skipping")
            continue
        try:
            description = _skill_description(skill_md)
        except OSError:
            continue
        skills.append((entry.name, description))
    return skills


def _resolve_within(skills_dir: Path, relative: Path) -> Path | None:
    """Resolve `relative` under `skills_dir`, or None if it escapes (symlinks included)."""
    resolved = (skills_dir / relative).resolve()
    try:
        resolved.relative_to(skills_dir.resolve())
    except ValueError:
        return None
    return resolved


def read_skill(name: str, skills_dir: Path, file: str | None = None) -> str:
    """
    Return a skill's content, or an error string listing what is available.

    With no `file`, returns SKILL.md's content followed by a "Supporting
    files:" listing of every other file under the skill folder (sorted
    relative paths), omitted when there are none. With `file`, returns that
    one supporting file's content instead.

    Both `name` and `file` come from the LLM and are treated as untrusted:
    separators/traversal sequences are rejected outright, and the resolved
    path must stay inside the skill folder (defeats symlink escapes too).
    """
    if not name or "/" in name or "\\" in name or ".." in name:
        return f"[Error: invalid skill name: {name!r}]"

    skill_dir = _resolve_within(skills_dir, Path(name))
    if skill_dir is None:
        return f"[Error: invalid skill name: {name!r}]"

    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        available = ", ".join(n for n, _ in list_skills(skills_dir)) or "none"
        return f"[Error: unknown skill {name!r}. Available: {available}]"

    if file is not None:
        if not file or file.startswith("/") or ".." in Path(file).parts:
            return f"[Error: invalid supporting file path: {file!r}]"
        target = _resolve_within(skill_dir, Path(file))
        if target is None:
            return f"[Error: invalid supporting file path: {file!r}]"
        if not target.is_file() or target == skill_md:
            return f"[Error: unknown supporting file {file!r} for skill {name!r}]"
        size = target.stat().st_size
        if size > MAX_SUPPORTING_FILE_BYTES:
            return (
                f"[Error: {file} is {size} bytes, over the "
                f"{MAX_SUPPORTING_FILE_BYTES // 1024} KB supporting-file read cap]"
            )
        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"[Error: {file} is not a text file]"

    content = skill_md.read_text(encoding="utf-8", errors="replace")
    supporting = sorted(
        p.relative_to(skill_dir).as_posix()
        for p in skill_dir.rglob("*")
        if p.is_file() and p != skill_md
    )
    if supporting:
        content += "\n\nSupporting files:\n" + "\n".join(f"- {p}" for p in supporting)
    return content
