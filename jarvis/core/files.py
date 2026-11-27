"""
Two small helpers about folders: keeping jarvis's own folders private, and
keeping every read inside the folders the user configured.
"""

import logging
from pathlib import Path


def make_private_dir(path: Path) -> None:
    """
    Create a folder (and its parents) if needed and make it owner-only.

    Used for everything jarvis keeps under ~/.jarvis: the index, the logs and
    the server state. They hold note text, paper titles and file paths, so
    no other user on the machine should be able to read them. The chmod runs
    every time, not only when the folder is first created, so a folder made
    by an older version is fixed the next time jarvis starts.
    """
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError as exc:
        # Not fatal: jarvis still works. But other users may be able to read
        # the folder, so log a warning.
        logging.getLogger("jarvis").warning(
            "could not make %s readable by you alone: %s", path, exc
        )


def resolves_inside(path: Path, root: Path) -> bool:
    """
    Whether path is really inside root once symlinks and ".." are followed.

    This is the check that stops jarvis reading outside the vault and papers
    folder. A symlink in the vault that points at ~/Documents/secret.md is
    seen for what it is, so it is treated as outside.
    """
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, RuntimeError):
        # A broken symlink loop, or a path we are not allowed to look at.
        return False
