"""
The draft sandbox: the only place on disk the model can write.

A draft is a folder — `~/.jarvis/drafts/<id>/` — so a `.tex` can keep its
`.bib` and figures beside it. Each holds `draft.json` with the draft's
metadata, and a `.versions/` folder of prior copies of every file that has been
overwritten.

Three rules shape everything here:

- **One containment policy.** `resolve_in_draft()` is the single function every
  read and write goes through. The draft id is validated against the same
  pattern session ids use, the filename is rejected on separators and
  traversal, the extension must be allowlisted, and the *resolved* path must
  land under the drafts root — so a symlink planted inside a draft cannot reach
  out of it.
- **Creation is free, agent mutation is reviewed.** Writing a new draft or a new
  file inside one happens directly. Changing existing content goes through
  `propose_edit()` → per-hunk human approval → `apply_hunks()`. Your own edits
  (a manual save from the editor) write straight through — that is you editing
  your own file, not an agent action.
- **Nothing here is ever indexed.** Work in progress would pollute retrieval
  with half-written text. A draft reaches the knowledge base only once the user
  has copied it into their vault themselves, through the ordinary vault sync.
"""

import difflib
import hashlib
import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from jarvis.core.config import get_config

# Same shape as a session id, and validated the same way — these arrive from
# the LLM and from HTTP requests, so they are checked before any path is built.
_VALID_ID = re.compile(r"^[0-9a-z-]{1,64}$")

# Internal folder inside a draft; excluded from listings and from the extension
# rules, since it holds timestamped copies rather than editable files.
VERSIONS_DIR = ".versions"

METADATA_FILE = "draft.json"

# Pending edit proposals, {token: {...}}. Process-local and deliberately not
# persisted: a proposal is only meaningful while the human who was shown it is
# still there to answer, and an abandoned one going stale is the safe outcome.
_proposals: dict[str, dict] = {}


class DraftError(Exception):
    """A draft operation that cannot proceed — always with a reason to show."""


# ── Containment ────────────────────────────────────────────────────────────────


def drafts_root() -> Path:
    """The sandbox root, resolved. Created on demand with private permissions."""
    root = get_config().drafts_dir.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)  # drafts can hold anything the user is working on
    return root.resolve()


def _require_valid_id(draft_id: str) -> str:
    if not _VALID_ID.match(draft_id or ""):
        raise DraftError(f"Invalid draft id {draft_id!r}")
    return draft_id


def draft_dir(draft_id: str) -> Path:
    """The folder for one draft, without requiring that it exists yet."""
    return drafts_root() / _require_valid_id(draft_id)


def resolve_in_draft(draft_id: str, filename: str) -> Path:
    """
    The single containment policy for every read and write in the sandbox.

    Raises DraftError with a reason rather than returning a path that only
    looks safe. The resolved path is checked against the drafts *root* (not the
    draft folder), and the draft id is then re-checked as the first component
    under it — resolving the draft folder first would follow a symlink planted
    there and happily validate a path outside the sandbox.
    """
    _require_valid_id(draft_id)

    name = (filename or "").strip()
    if not name:
        raise DraftError("No filename given")
    if "/" in name or "\\" in name or name.startswith("."):
        raise DraftError(
            f"Invalid filename {filename!r} — a plain name inside the draft, no paths"
        )
    if ".." in name:
        raise DraftError(f"Invalid filename {filename!r} — path traversal is not allowed")
    if name.startswith("-"):
        # Not a path concern: this name is handed to latexmk and pandoc as a
        # positional argument, and a leading dash makes it parse as an OPTION
        # instead. `-latex=pdflatex -shell-escape ...` would re-enable the shell
        # escape that render.py disables. Filenames come from the model, so a
        # prompt injection could otherwise reach the compiler's argument vector.
        raise DraftError(
            f"Invalid filename {filename!r} — a name cannot start with '-'"
        )

    cfg = get_config()
    if Path(name).suffix.lower() not in cfg.drafts_extensions:
        raise DraftError(
            f"Extension {Path(name).suffix!r} is not allowed here — "
            f"permitted: {', '.join(cfg.drafts_extensions)}"
        )

    root = drafts_root()
    candidate = (root / draft_id / name).resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        raise DraftError(f"{filename!r} resolves outside the drafts folder") from None
    if relative.parts[0] != draft_id:
        raise DraftError(f"{filename!r} resolves outside draft {draft_id!r}")
    return candidate


# ── Metadata ───────────────────────────────────────────────────────────────────


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _metadata_path(draft_id: str) -> Path:
    return draft_dir(draft_id) / METADATA_FILE


def read_metadata(draft_id: str) -> dict:
    path = _metadata_path(draft_id)
    if not path.exists():
        raise DraftError(f"No draft {draft_id!r}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_metadata(draft_id: str, metadata: dict) -> None:
    path = _metadata_path(draft_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


# ── Writing ────────────────────────────────────────────────────────────────────


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _snapshot(path: Path) -> "Path | None":
    """
    Copy the current contents aside before overwriting them.

    This is what makes undo work across a page reload, and what makes an
    accepted-by-mistake agent hunk recoverable. Snapshots are never pruned.
    """
    if not path.exists():
        return None
    versions = path.parent / VERSIONS_DIR
    versions.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    target = versions / f"{path.name}.{stamp}"
    shutil.copy2(path, target)
    return target


def _check_size(content: str) -> None:
    limit = get_config().drafts_max_file_bytes
    if len(content.encode("utf-8")) > limit:
        raise DraftError(f"Content is larger than the {limit} byte limit for a draft file")


def create_draft(
    title: str,
    filename: str,
    content: str,
    visibility: str = "public",
    session_id: str = "",
) -> dict:
    """
    Start a new draft. A free write — nothing existed to overwrite.

    Visibility is inherited from the session that asked for it: a draft built
    from private notes is private, and can then only be opened under a local
    model.
    """
    _check_size(content)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    draft_id = f"{stamp}-{uuid.uuid4().hex[:6]}"
    path = resolve_in_draft(draft_id, filename)

    _write_atomic(path, content)
    metadata = {
        "id": draft_id,
        "title": title.strip() or filename,
        "main_file": filename,
        "created": _now(),
        "updated": _now(),
        "visibility": visibility,
        "session_id": session_id,
        "keep": False,
    }
    write_metadata(draft_id, metadata)
    return metadata


def add_draft_file(draft_id: str, filename: str, content: str) -> Path:
    """
    Add a NEW file to an existing draft (a .bib alongside a .tex, say).

    Free only when the file does not exist — overwriting existing content is a
    mutation and goes through propose_edit like any other.
    """
    _check_size(content)
    path = resolve_in_draft(draft_id, filename)
    if path.exists():
        raise DraftError(
            f"{filename!r} already exists in this draft — propose an edit to change it"
        )
    read_metadata(draft_id)  # confirms the draft exists before writing into it
    _write_atomic(path, content)
    touch(draft_id)
    return path


def save_draft_file(draft_id: str, filename: str, content: str, expect_hash: str = "") -> str:
    """
    Write a file directly — the human's own save from the editor.

    `expect_hash`, when given, must match what is on disk; a mismatch means the
    file changed underneath (a second tab, an external edit) and the write is
    refused rather than clobbering it. Returns the new content hash.
    """
    _check_size(content)
    path = resolve_in_draft(draft_id, filename)
    read_metadata(draft_id)

    if expect_hash and path.exists():
        current = content_hash(path.read_text(encoding="utf-8"))
        if current != expect_hash:
            raise DraftError(
                "This file changed since it was opened — reload it before saving, "
                "or your edits would overwrite the newer version."
            )

    _snapshot(path)
    _write_atomic(path, content)
    touch(draft_id)
    return content_hash(content)


def touch(draft_id: str) -> None:
    """Record activity, so retention sees the draft as live."""
    try:
        metadata = read_metadata(draft_id)
    except DraftError:
        return
    metadata["updated"] = _now()
    write_metadata(draft_id, metadata)


def set_keep(draft_id: str, keep: bool) -> dict:
    """Exempt a draft from retention (or stop exempting it)."""
    metadata = read_metadata(draft_id)
    metadata["keep"] = bool(keep)
    write_metadata(draft_id, metadata)
    return metadata


# ── Reading ────────────────────────────────────────────────────────────────────


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def draft_files(draft_id: str) -> list[str]:
    """Editable files in a draft, sorted. Excludes metadata and .versions/."""
    folder = draft_dir(draft_id)
    if not folder.is_dir():
        return []
    extensions = get_config().drafts_extensions
    return sorted(
        entry.name
        for entry in folder.iterdir()
        if entry.is_file()
        and entry.name != METADATA_FILE
        and not entry.name.startswith(".")
        and entry.suffix.lower() in extensions
    )


def read_draft(draft_id: str, filename: str = "") -> dict:
    """One file's text plus the hash an edit proposal will be checked against."""
    metadata = read_metadata(draft_id)
    name = filename or metadata.get("main_file", "")
    path = resolve_in_draft(draft_id, name)
    if not path.exists():
        raise DraftError(f"No file {name!r} in draft {draft_id!r}")
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "draft_id": draft_id,
        "file": name,
        "text": text,
        "hash": content_hash(text),
        "visibility": metadata.get("visibility", "public"),
    }


def list_drafts() -> list[dict]:
    """Every draft's metadata, most recently updated first, with its files."""
    root = drafts_root()
    drafts = []
    for folder in root.iterdir():
        if not folder.is_dir() or not _VALID_ID.match(folder.name):
            continue
        try:
            metadata = read_metadata(folder.name)
        except (DraftError, json.JSONDecodeError):
            continue
        metadata["files"] = draft_files(folder.name)
        metadata["age_days"] = draft_age_days(folder.name)
        drafts.append(metadata)
    drafts.sort(key=lambda d: d.get("updated", ""), reverse=True)
    return drafts


# ── Edit proposals ─────────────────────────────────────────────────────────────


def _hunks(old_lines: list[str], new_lines: list[str], context: int = 3) -> list[dict]:
    """
    Split a rewrite into reviewable hunks.

    Built from SequenceMatcher's grouped opcodes rather than by parsing a text
    diff: each group already carries the exact old and new line ranges, so
    applying a subset is arithmetic rather than diff re-parsing, and the
    rendered diff and the applied change can never disagree.
    """
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    hunks = []
    for index, group in enumerate(matcher.get_grouped_opcodes(context)):
        old_start, old_end = group[0][1], group[-1][2]
        new_start, new_end = group[0][3], group[-1][4]

        # What this change actually does, judged from the opcodes rather than
        # from the line counts: a group carries context lines on both sides, so
        # neither old_lines nor new_lines is ever empty and counting them would
        # call every change a replacement.
        tags = {tag for tag, *_ in group if tag != "equal"}
        if tags == {"insert"}:
            kind = "add"
        elif tags == {"delete"}:
            kind = "remove"
        else:
            kind = "replace"

        diff_lines = []
        # Which lines *within* the hunk actually changed. A group spans context
        # lines on both sides of the change, so a caller that highlights the
        # whole span paints unchanged text as though it were being removed —
        # three deleted blocks looked like four, the fourth being context.
        # Offsets are relative to old_lines/new_lines, so the caller can use
        # them directly against the lists it was handed.
        old_spans: list[list[int]] = []
        new_spans: list[list[int]] = []
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                diff_lines.extend(f"  {line}" for line in old_lines[i1:i2])
                continue
            diff_lines.extend(f"- {line}" for line in old_lines[i1:i2])
            diff_lines.extend(f"+ {line}" for line in new_lines[j1:j2])
            if i2 > i1:
                old_spans.append([i1 - old_start, i2 - old_start])
            if j2 > j1:
                new_spans.append([j1 - new_start, j2 - new_start])
        hunks.append({
            "index": index,
            "kind": kind,
            "old_start": old_start,
            "old_end": old_end,
            "new_start": new_start,
            "new_end": new_end,
            "header": f"@@ -{old_start + 1},{old_end - old_start} "
                      f"+{new_start + 1},{new_end - new_start} @@",
            "diff": diff_lines,
            # The literal lines on each side, so a caller can lay the change
            # out inside an editor rather than only render a patch. Kept
            # newline-free: the editor works in lines, not raw text.
            "old_lines": [line.rstrip("\n") for line in old_lines[old_start:old_end]],
            "new_lines": [line.rstrip("\n") for line in new_lines[new_start:new_end]],
            "old_spans": old_spans,
            "new_spans": new_spans,
        })
    return hunks


def propose_edit(draft_id: str, filename: str, new_text: str, rationale: str = "") -> dict:
    """
    Record a proposed rewrite. WRITES NOTHING.

    Returns the proposal — token, hunks, and the base hash the file must still
    have when the human accepts. Only apply_hunks() touches the file, and only
    with a token a human answered.
    """
    _check_size(new_text)
    current = read_draft(draft_id, filename)
    old_lines = current["text"].splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    hunks = _hunks(old_lines, new_lines)

    token = uuid.uuid4().hex
    proposal = {
        "token": token,
        "draft_id": draft_id,
        "file": current["file"],
        "base_hash": current["hash"],
        "new_text": new_text,
        "rationale": rationale,
        "hunks": hunks,
        "created": _now(),
    }
    _proposals[token] = proposal
    return proposal


def read_proposal(token: str) -> dict:
    proposal = _proposals.get(token)
    if proposal is None:
        raise DraftError("That edit proposal is no longer pending — ask for it again.")
    return proposal


def discard_proposal(token: str) -> None:
    _proposals.pop(token, None)


def clear_proposals_for(draft_id: str) -> None:
    """Drop a draft's pending proposals — used when it is deleted."""
    for token, proposal in list(_proposals.items()):
        if proposal["draft_id"] == draft_id:
            del _proposals[token]


def apply_hunks(token: str, indices: "list[int] | None" = None) -> dict:
    """
    Apply the accepted hunks of a proposal. The only path that writes an agent's
    change to disk, and it needs a token a human answered.

    Refuses if the file changed since the proposal was made — a stale proposal
    is a conflict to report, never something to resolve by overwriting. The
    previous contents are snapshotted first.
    """
    proposal = read_proposal(token)
    path = resolve_in_draft(proposal["draft_id"], proposal["file"])
    if not path.exists():
        raise DraftError(f"{proposal['file']!r} no longer exists in this draft")

    current_text = path.read_text(encoding="utf-8", errors="replace")
    if content_hash(current_text) != proposal["base_hash"]:
        discard_proposal(token)
        raise DraftError(
            f"{proposal['file']!r} changed since this edit was proposed, so applying it "
            "would overwrite the newer version. Ask for the edit again against the "
            "current text."
        )

    selected = set(range(len(proposal["hunks"])) if indices is None else indices)
    old_lines = current_text.splitlines(keepends=True)
    new_lines = proposal["new_text"].splitlines(keepends=True)

    # Stitch the file back together: untouched regions from the original, and
    # each hunk taken from whichever side the human chose.
    result: list[str] = []
    position = 0
    for hunk in proposal["hunks"]:
        result.extend(old_lines[position:hunk["old_start"]])
        if hunk["index"] in selected:
            result.extend(new_lines[hunk["new_start"]:hunk["new_end"]])
        else:
            result.extend(old_lines[hunk["old_start"]:hunk["old_end"]])
        position = hunk["old_end"]
    result.extend(old_lines[position:])

    merged = "".join(result)
    _snapshot(path)
    _write_atomic(path, merged)
    touch(proposal["draft_id"])
    discard_proposal(token)
    return {
        "draft_id": proposal["draft_id"],
        "file": proposal["file"],
        "applied": sorted(selected),
        "rejected": [h["index"] for h in proposal["hunks"] if h["index"] not in selected],
        "hash": content_hash(merged),
    }


# ── Versions ───────────────────────────────────────────────────────────────────


def list_versions(draft_id: str, filename: str) -> list[dict]:
    """Snapshots of one file, newest first."""
    resolve_in_draft(draft_id, filename)  # validates before touching the folder
    versions = draft_dir(draft_id) / VERSIONS_DIR
    if not versions.is_dir():
        return []
    entries = [
        {"name": entry.name, "saved_at": entry.name.rsplit(".", 1)[-1]}
        for entry in versions.iterdir()
        if entry.is_file() and entry.name.startswith(f"{filename}.")
    ]
    entries.sort(key=lambda e: e["saved_at"], reverse=True)
    return entries


def read_version(draft_id: str, filename: str, version_name: str) -> str:
    """One snapshot's text. The version name is validated like any other input."""
    resolve_in_draft(draft_id, filename)
    if "/" in version_name or "\\" in version_name or ".." in version_name:
        raise DraftError(f"Invalid version name {version_name!r}")
    if not version_name.startswith(f"{filename}."):
        raise DraftError(f"{version_name!r} is not a version of {filename!r}")
    path = (draft_dir(draft_id) / VERSIONS_DIR / version_name).resolve()
    if not path.is_file() or path.parent != (draft_dir(draft_id) / VERSIONS_DIR).resolve():
        raise DraftError(f"No version {version_name!r}")
    return path.read_text(encoding="utf-8", errors="replace")


def restore_version(draft_id: str, filename: str, version_name: str) -> str:
    """
    Put a snapshot back. The current contents are snapshotted first, so a
    restore is itself undoable.
    """
    text = read_version(draft_id, filename, version_name)
    path = resolve_in_draft(draft_id, filename)
    _snapshot(path)
    _write_atomic(path, text)
    touch(draft_id)
    return content_hash(text)


# ── Retention ──────────────────────────────────────────────────────────────────


def draft_age_days(draft_id: str, now: "datetime | None" = None) -> float:
    """
    Days since anything in this draft was last touched.

    Taken from the newest mtime across the whole folder, `.versions/` included
    (a snapshot means you were working on it). So editing one file keeps the
    whole draft alive — the behaviour you want for a .tex and its .bib.
    """
    folder = draft_dir(draft_id)
    if not folder.is_dir():
        raise DraftError(f"No draft {draft_id!r}")
    newest = max(
        (entry.stat().st_mtime for entry in folder.rglob("*") if entry.is_file()),
        default=folder.stat().st_mtime,
    )
    reference = now or datetime.now(timezone.utc)
    return (reference.timestamp() - newest) / 86400


def stale_drafts(now: "datetime | None" = None) -> list[dict]:
    """
    Drafts eligible for removal: older than retention_days and not kept.

    `retention_days = 0` disables the sweep entirely and returns nothing.
    """
    cfg = get_config()
    if cfg.drafts_retention_days <= 0:
        return []
    stale = []
    for metadata in list_drafts():
        if metadata.get("keep"):
            continue
        age = draft_age_days(metadata["id"], now)
        if age >= cfg.drafts_retention_days:
            stale.append({**metadata, "age_days": age})
    return stale


def delete_draft(draft_id: str) -> dict:
    """
    Delete one draft outright, at a human's explicit request.

    The second — and last — code path in jarvis that removes a file, and like
    retention it is confined to the drafts root and re-checks containment
    immediately before the delete rather than trusting its caller. It takes a
    draft id, not a path, so there is no argument to aim somewhere else, and
    **no chat tool reaches it**: this is reachable only from the UI's delete
    button and it is human-only by construction, exactly as
    `/documents/remove` is.

    Anything the user already copied into their vault is untouched — that copy
    is an ordinary file of theirs, and deleting a draft never reaches it.
    """
    metadata = read_metadata(draft_id)          # 404s on an unknown id
    folder = draft_dir(draft_id).resolve()
    if folder.parent != drafts_root() or not folder.is_dir():
        raise DraftError(f"Refusing to delete {draft_id!r} — it is not a draft folder")

    clear_proposals_for(draft_id)
    shutil.rmtree(folder)                       # not followed for symlinked dirs
    return metadata


def prune_drafts(dry_run: bool = False, now: "datetime | None" = None) -> list[dict]:
    """
    Remove stale drafts. The ONLY code path in jarvis that deletes a file.

    Its scope is fixed by construction: it takes no path argument, only ever
    walks the drafts root, and is reachable from exactly two places — the
    daemon's scheduled sweep and `kb drafts --prune`. No chat tool and no HTTP
    route can trigger it. Everything outside `~/.jarvis/drafts/` keeps the
    absolute no-file-deletion guarantee.
    """
    removed = []
    root = drafts_root()
    for metadata in stale_drafts(now):
        folder = draft_dir(metadata["id"]).resolve()
        # Re-check containment immediately before deleting rather than trusting
        # the listing: this is the one destructive operation in the codebase.
        if folder.parent != root or not folder.is_dir():
            continue
        if not dry_run:
            clear_proposals_for(metadata["id"])
            # Not followed for symlinked directories, which rmtree refuses.
            shutil.rmtree(folder)
        removed.append(metadata)
    return removed
