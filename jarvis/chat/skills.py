"""
User-defined skills — reusable instruction files the agent loads on demand.

A skill is a plain .md file in cfg.skills_dir (default ~/.jarvis/skills).
The filename stem is the skill name; the first non-empty line (leading '#'
stripped) is its one-line description. The system prompt advertises only
name + description; the model calls the read_skill tool to pull in the full
instructions when a task matches — full skill text never occupies context
until it is actually needed.

Skills are the user's own local files: trusted, never indexed into the
vector store, and outside the public/private visibility model.
"""

from pathlib import Path


def list_skills(skills_dir: Path) -> list[tuple[str, str]]:
    """
    Return [(name, description)] for every *.md directly in skills_dir,
    sorted by name. A missing or non-directory skills_dir means the feature
    is off — return [] without complaint.
    """
    if not skills_dir.is_dir():
        return []
    skills = []
    for skill_file in sorted(skills_dir.glob("*.md")):
        description = ""
        try:
            for line in skill_file.read_text(encoding="utf-8", errors="replace").splitlines():
                stripped = line.strip().lstrip("#").strip()
                if stripped:
                    description = stripped
                    break
        except OSError:
            continue
        skills.append((skill_file.stem, description))
    return skills


def read_skill(name: str, skills_dir: Path) -> str:
    """
    Return a skill file's full content, or an error string listing what is
    available. The name comes from the LLM, so it is treated as untrusted:
    separators and traversal sequences are rejected outright, and the
    resolved path must stay inside skills_dir.
    """
    if not name or "/" in name or "\\" in name or ".." in name:
        return f"[Error: invalid skill name: {name!r}]"

    skill_file = (skills_dir / f"{name}.md").resolve()
    try:
        skill_file.relative_to(skills_dir.resolve())
    except ValueError:
        return f"[Error: invalid skill name: {name!r}]"

    if not skill_file.is_file():
        available = ", ".join(n for n, _ in list_skills(skills_dir)) or "none"
        return f"[Error: unknown skill {name!r}. Available: {available}]"
    return skill_file.read_text(encoding="utf-8", errors="replace")
