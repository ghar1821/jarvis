"""
Structural guarantees about what jarvis can do to your files.

jarvis only ever reads your vault and papers folder. It writes nothing
anywhere except under ~/.jarvis/ (the index, its lock, logs and the serve
status file), and it deletes no file at all — removing something from the
index is done by you deleting the file. These tests check that by reading
the source rather than by exercising each code path, so a new write or
delete anywhere in the package fails here until it has been reviewed and
added to the allowlist on purpose.
"""

import ast
from pathlib import Path

import jarvis

PACKAGE_ROOT = Path(jarvis.__file__).parent

# Calls that write, move or delete files, by method or function name.
_WRITING_CALLS = {
    "write_text", "write_bytes", "unlink", "rename", "replace", "rmtree",
    "remove", "rmdir", "move", "copy", "copyfile", "copytree", "save", "touch",
}

# Every reviewed write in the package: (file relative to jarvis/, call name).
# Each one targets ~/.jarvis/ (or the configured rag_dir) and nothing else.
_ALLOWED_WRITES = {
    # serve_status.json, written atomically: temp file then os.replace.
    ("kb/serve.py", "write_text"),
    ("kb/serve.py", "replace"),
    # ~/.jarvis/logs/chroma.log, the Chroma child's output.
    ("kb/serve.py", "open:a"),
    # <rag_dir>/.write.lock, the advisory lock for direct index access.
    ("kb/store.py", "open:w"),
    # ~/.jarvis/state/server_token, created owner-only in one step.
    ("kb/serve.py", "os.open"),
}


def _write_sites() -> set[tuple[str, str]]:
    """Every call in the package that can write, move or delete a file."""
    sites = set()
    for path in PACKAGE_ROOT.rglob("*.py"):
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in _WRITING_CALLS:
                # str.replace is everywhere and harmless; only os.replace moves files.
                if name == "replace" and not (
                    isinstance(func, ast.Attribute) and getattr(func.value, "id", "") == "os"
                ):
                    continue
                sites.add((relative, name))
            elif name == "open" and isinstance(func, ast.Attribute) and getattr(func.value, "id", "") == "os":
                # os.open takes flags, not a mode string, so any call counts.
                sites.add((relative, "os.open"))
            elif name == "open" and isinstance(func, ast.Name):
                mode = node.args[1] if len(node.args) > 1 else next(
                    (kw.value for kw in node.keywords if kw.arg == "mode"), None
                )
                mode_text = mode.value if isinstance(mode, ast.Constant) else "r"
                if any(flag in mode_text for flag in "wax+"):
                    sites.add((relative, f"open:{mode_text}"))
    return sites


def test_every_file_write_in_the_package_is_reviewed():
    """
    The full set of writing call sites equals the allowlist above. A new one
    fails this test until someone has checked it stays inside ~/.jarvis.
    """
    assert _write_sites() == _ALLOWED_WRITES


def test_no_file_is_ever_deleted():
    """No unlink/remove/rmtree anywhere: jarvis cannot delete a file of yours."""
    deleting = {"unlink", "remove", "rmtree", "rmdir"}
    assert not {site for site in _write_sites() if site[1] in deleting}


def test_the_allowed_writes_point_inside_dot_jarvis():
    """The serve status file, server token and logs live under ~/.jarvis, never in a mirrored folder."""
    from jarvis.kb import serve, store

    dot_jarvis = Path.home() / ".jarvis"
    assert serve.STATUS_FILE.is_relative_to(dot_jarvis)
    assert serve.LOG_DIR.is_relative_to(dot_jarvis)
    assert store.TOKEN_FILE.is_relative_to(dot_jarvis)


def test_folders_under_dot_jarvis_are_made_owner_only(tmp_path):
    """Logs, state and the index name your papers and notes, so only you can read them."""
    from jarvis.core.files import make_private_dir

    folder = tmp_path / "state"
    make_private_dir(folder)
    assert folder.stat().st_mode & 0o777 == 0o700

    # An existing folder with looser permissions is tightened too.
    folder.chmod(0o755)
    make_private_dir(folder)
    assert folder.stat().st_mode & 0o777 == 0o700


def test_the_server_token_is_created_owner_only_and_kept(tmp_path):
    from jarvis.kb.serve import ensure_server_token

    token_file = ensure_server_token(tmp_path / "state" / "server_token")
    token = token_file.read_text()
    assert len(token) >= 40
    assert token_file.stat().st_mode & 0o777 == 0o600
    assert token_file.parent.stat().st_mode & 0o777 == 0o700

    # Restarting kb serve reuses the token, so running clients keep working.
    assert ensure_server_token(token_file).read_text() == token
