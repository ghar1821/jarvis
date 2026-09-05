"""
The prompts jarvis sends to an LLM, and the user's editable copies of them.

Three prompts drive everything jarvis asks a model to do, and all three are
things a user will eventually want to reword — the digest scoring prompt most
of all, since it encodes what *they* find worth reading and nobody else's
version of that is right.

So the repo ships a generic default for each, and the copy that actually runs
lives beside `config.toml` in `~/.jarvis/prompts/`. A missing copy is created
from the default on first use, editing writes to the copy, and reverting
overwrites it from the default again. The repo file is never written to.

The scoring prompt in particular used to be committed with one researcher's
active topics in it. That is configuration, not code — it belongs in the
user's own directory where they can change it without editing the repo.
"""

import shutil
from pathlib import Path

_DEFAULTS_DIR = Path(__file__).parent.parent / "prompts"


# name -> (filename, title, what editing it changes)
PROMPTS: dict[str, tuple[str, str, str]] = {
    "system_prompt": (
        "system_prompt.md",
        "Assistant instructions",
        "How the chat agent behaves: when it searches, how it cites, what it "
        "refuses to do. Sent with every message.",
    ),
    "paper_summary": (
        "paper_summary.md",
        "Paper summary",
        "How a paper is summarised when you add one in summary mode. "
        "`{title}` is replaced with the paper's title.",
    ),
    "digest_scoring": (
        "digest_scoring.md",
        "Digest scoring",
        "How the weekly digest decides which papers matter. Describe your own "
        "work in the research context section — that is what drives the "
        "scores. `{num_papers}`, `{max_results}` and `{abstracts_text}` are "
        "filled in at run time and must stay.",
    ),
}


class PromptError(Exception):
    """An unknown prompt name, or a copy that cannot be written."""


def _entry(name: str) -> tuple[str, str, str]:
    if name not in PROMPTS:
        raise PromptError(
            f"Unknown prompt {name!r} — expected one of {', '.join(sorted(PROMPTS))}"
        )
    return PROMPTS[name]


def prompts_dir() -> Path:
    """Where the editable copies live, beside config.toml."""
    from .config import CONFIG_FILE

    return CONFIG_FILE.parent / "prompts"


def default_path(name: str) -> Path:
    """The read-only default shipped in the repo."""
    return _DEFAULTS_DIR / _entry(name)[0]


def user_path(name: str) -> Path:
    """The editable copy. May not exist yet — `load` creates it."""
    return prompts_dir() / _entry(name)[0]


def default_text(name: str) -> str:
    return default_path(name).read_text(encoding="utf-8")


def load(name: str) -> str:
    """
    The prompt as it will actually be sent, creating the user's copy first if
    it isn't there.

    Every consumer goes through here, so the copy exists no matter which entry
    point ran first — the webapp, the daemon, or a one-off `kb add`.
    """
    ensure(name)
    return user_path(name).read_text(encoding="utf-8")


def ensure(name: str) -> bool:
    """
    Create the user's copy from the default if it is missing. Returns whether
    it was created, so a caller can log a first-run seed.
    """
    path = user_path(name)
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)

    # `~/.jarvis/system_prompt.md` was the old override location, and predates
    # this whole mechanism. Carry a customised one across rather than seeding
    # over the top of it with the default and losing someone's wording.
    legacy = prompts_dir().parent / "system_prompt.md"
    if name == "system_prompt" and legacy.exists():
        shutil.copy2(legacy, path)
        return True

    shutil.copy2(default_path(name), path)
    return True


def ensure_all() -> list[str]:
    """Seed every missing copy. Returns the names that were created."""
    return [name for name in PROMPTS if ensure(name)]


def save(name: str, text: str) -> None:
    """
    Replace the user's copy. Written atomically, because a half-written prompt
    would be sent to a model on the very next turn.
    """
    path = user_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def reset(name: str) -> str:
    """Overwrite the user's copy with the shipped default and return it."""
    text = default_text(name)
    save(name, text)
    return text


def is_customised(name: str) -> bool:
    """Whether the copy differs from the default it was seeded from."""
    path = user_path(name)
    if not path.exists():
        return False
    return path.read_text(encoding="utf-8") != default_text(name)


def listing() -> list[dict]:
    """Every prompt, annotated for the editor UI."""
    return [
        {
            "name": name,
            "title": title,
            "description": description,
            "customised": is_customised(name),
        }
        for name, (_file, title, description) in PROMPTS.items()
    ]
