"""
Tests for jarvis/drafts/workspace.py — the agent-writable sandbox.

Writing to disk is the genuinely new capability in jarvis, so the containment
policy and the hunk arithmetic get the heaviest coverage in the suite: a
containment hole would let a prompt-injected model write outside the sandbox,
and a hunk bug would corrupt the user's own work.
"""

from datetime import datetime, timedelta, timezone

import pytest

from jarvis.core.config import Config
from jarvis.drafts import workspace
from jarvis.drafts.workspace import (
    DraftError,
    add_draft_file,
    apply_hunks,
    create_draft,
    draft_age_days,
    list_drafts,
    propose_edit,
    prune_drafts,
    read_draft,
    resolve_in_draft,
    save_draft_file,
    set_keep,
    stale_drafts,
)


@pytest.fixture
def drafts(tmp_path, monkeypatch):
    """An isolated drafts sandbox — never the developer's real ~/.jarvis."""
    # Every path this fixture leaves at its default points into the real
    # ~/.jarvis. One was missed once and a test wrote into the developer's own
    # home directory; the autouse guard in conftest.py now fails the run if
    # that happens again, but overriding the paths is still the first defence.
    cfg = Config(
        drafts_dir=tmp_path / "drafts",
        vault_path=tmp_path / "vault",
    )
    monkeypatch.setattr("jarvis.drafts.workspace.get_config", lambda: cfg)
    workspace._proposals.clear()
    return cfg


# ── Containment (W2) ───────────────────────────────────────────────────────────

def test_resolve_accepts_a_plain_filename(drafts):
    path = resolve_in_draft("20260101-000000-abcdef", "cv.tex")
    assert path.name == "cv.tex"
    assert path.parent.name == "20260101-000000-abcdef"


@pytest.mark.parametrize("filename", [
    "../escape.md",
    "../../etc/passwd.md",
    "sub/dir.md",
    "sub\\dir.md",
    ".hidden.md",
    "",
])
def test_resolve_rejects_paths_and_traversal(drafts, filename):
    """A filename is a plain name inside one draft — never a path."""
    with pytest.raises(DraftError):
        resolve_in_draft("20260101-000000-abcdef", filename)


@pytest.mark.parametrize("filename", [
    "-latex=pdflatex -shell-escape %O %S.tex",
    "-shell-escape.tex",
    "--lua-filter=evil.md",
    "-o.md",
])
def test_resolve_rejects_a_filename_that_would_parse_as_an_option(drafts, filename):
    """
    Not a path concern: these names are handed to latexmk and pandoc as
    positional arguments, and a leading dash makes them parse as OPTIONS —
    `-latex=pdflatex -shell-escape ...` would re-enable the very shell escape
    render.py disables. Filenames come from the model, so without this a prompt
    injection in a retrieved document could reach the compiler's argv.
    """
    with pytest.raises(DraftError, match="cannot start with"):
        resolve_in_draft("20260101-000000-abcdef", filename)


def test_resolve_rejects_a_disallowed_extension(drafts):
    """The allowlist is what stops a draft becoming an executable drop."""
    for name in ("payload.sh", "run.py", "thing.exe"):
        with pytest.raises(DraftError, match="not allowed"):
            resolve_in_draft("20260101-000000-abcdef", name)


@pytest.mark.parametrize("draft_id", ["../other", "a/b", "UPPER", "x" * 65, ""])
def test_resolve_rejects_an_invalid_draft_id(drafts, draft_id):
    with pytest.raises(DraftError):
        resolve_in_draft(draft_id, "notes.md")


def test_resolve_refuses_a_symlinked_draft_folder_pointing_outside(drafts, tmp_path):
    """
    The reason containment is checked against the drafts ROOT and not against
    the draft folder: resolving the draft folder first would follow this
    symlink and then happily validate a path outside the sandbox.
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "drafts"
    root.mkdir(parents=True, exist_ok=True)
    (root / "20260101-000000-abcdef").symlink_to(outside, target_is_directory=True)

    with pytest.raises(DraftError, match="outside"):
        resolve_in_draft("20260101-000000-abcdef", "stolen.md")


def test_resolve_refuses_a_symlinked_file_escaping_the_draft(drafts, tmp_path):
    """A symlink planted inside a draft cannot reach out of it either."""
    draft = create_draft("A draft", "notes.md", "hello\n")
    outside = tmp_path / "secret.md"
    outside.write_text("secret", encoding="utf-8")
    (tmp_path / "drafts" / draft["id"] / "link.md").symlink_to(outside)

    with pytest.raises(DraftError, match="outside"):
        resolve_in_draft(draft["id"], "link.md")


# ── Creating and reading ───────────────────────────────────────────────────────

def test_create_draft_writes_the_file_and_metadata(drafts):
    draft = create_draft("My CV", "cv.tex", "\\documentclass{article}\n", visibility="public")

    stored = read_draft(draft["id"])
    assert stored["file"] == "cv.tex"
    assert stored["text"] == "\\documentclass{article}\n"
    assert draft["title"] == "My CV"
    assert draft["keep"] is False


def test_create_draft_inherits_session_visibility(drafts):
    """
    A draft built from private notes is private, and can then only be opened
    under a local model — the transcript rule extended to the artefact.
    """
    draft = create_draft("Private", "notes.md", "sensitive\n", visibility="private")
    assert read_draft(draft["id"])["visibility"] == "private"


def test_add_draft_file_is_free_only_for_a_new_file(drafts):
    """Creation is free; overwriting existing content is a mutation."""
    draft = create_draft("Paper", "paper.tex", "\\documentclass{article}\n")
    add_draft_file(draft["id"], "refs.bib", "@article{a}\n")

    assert set(list_drafts()[0]["files"]) == {"paper.tex", "refs.bib"}
    with pytest.raises(DraftError, match="already exists"):
        add_draft_file(draft["id"], "refs.bib", "@article{b}\n")


def test_a_file_larger_than_the_limit_is_refused(drafts):
    drafts.drafts_max_file_bytes = 32
    with pytest.raises(DraftError, match="larger than"):
        create_draft("Big", "big.md", "x" * 100)


# ── Proposals write nothing (W1/W3) ────────────────────────────────────────────

def test_propose_edit_writes_nothing(drafts):
    """The whole point: an agent's change is a proposal until a human accepts."""
    draft = create_draft("Doc", "doc.md", "one\ntwo\nthree\n")
    before = read_draft(draft["id"])["text"]

    proposal = propose_edit(draft["id"], "doc.md", "one\nTWO\nthree\n", "capitalise")

    assert read_draft(draft["id"])["text"] == before
    assert proposal["hunks"]
    assert proposal["rationale"] == "capitalise"


def test_apply_all_hunks_gives_the_proposed_text(drafts):
    draft = create_draft("Doc", "doc.md", "one\ntwo\nthree\n")
    proposal = propose_edit(draft["id"], "doc.md", "one\nTWO\nthree\n")

    apply_hunks(proposal["token"])

    assert read_draft(draft["id"])["text"] == "one\nTWO\nthree\n"


def test_applying_no_hunks_leaves_the_file_unchanged(drafts):
    draft = create_draft("Doc", "doc.md", "one\ntwo\nthree\n")
    proposal = propose_edit(draft["id"], "doc.md", "one\nTWO\nthree\n")

    apply_hunks(proposal["token"], indices=[])

    assert read_draft(draft["id"])["text"] == "one\ntwo\nthree\n"


def test_applying_a_subset_takes_exactly_those_hunks(drafts):
    """
    The arithmetic that matters: two well-separated changes become two hunks,
    and accepting one must apply that one and leave the other alone.
    """
    original = "".join(f"line {i}\n" for i in range(30))
    edited = original.replace("line 2\n", "LINE TWO\n").replace("line 27\n", "LINE TWENTY-SEVEN\n")

    draft = create_draft("Doc", "doc.md", original)
    proposal = propose_edit(draft["id"], "doc.md", edited)
    assert len(proposal["hunks"]) == 2

    apply_hunks(proposal["token"], indices=[0])
    result = read_draft(draft["id"])["text"]

    assert "LINE TWO" in result
    assert "LINE TWENTY-SEVEN" not in result
    assert "line 27" in result
    # Everything outside the hunks is byte-identical.
    assert result.count("\n") == original.count("\n")


def test_applying_the_second_hunk_only(drafts):
    original = "".join(f"line {i}\n" for i in range(30))
    edited = original.replace("line 2\n", "LINE TWO\n").replace("line 27\n", "LINE TWENTY-SEVEN\n")

    draft = create_draft("Doc", "doc.md", original)
    proposal = propose_edit(draft["id"], "doc.md", edited)

    apply_hunks(proposal["token"], indices=[1])
    result = read_draft(draft["id"])["text"]

    assert "LINE TWO" not in result
    assert "LINE TWENTY-SEVEN" in result


def test_an_insertion_at_the_end_applies_cleanly(drafts):
    draft = create_draft("Doc", "doc.md", "a\nb\n")
    proposal = propose_edit(draft["id"], "doc.md", "a\nb\nc\n")
    apply_hunks(proposal["token"])
    assert read_draft(draft["id"])["text"] == "a\nb\nc\n"


def test_a_deletion_applies_cleanly(drafts):
    draft = create_draft("Doc", "doc.md", "a\nb\nc\n")
    proposal = propose_edit(draft["id"], "doc.md", "a\nc\n")
    apply_hunks(proposal["token"])
    assert read_draft(draft["id"])["text"] == "a\nc\n"


# ── Stale proposals (W4) ───────────────────────────────────────────────────────

def test_applying_a_stale_proposal_is_refused(drafts):
    """
    A hand-edit between proposal and accept means the diff no longer describes
    the file. Refuse and say so — never resolve a conflict by overwriting.
    """
    draft = create_draft("Doc", "doc.md", "one\ntwo\n")
    proposal = propose_edit(draft["id"], "doc.md", "one\nTWO\n")

    save_draft_file(draft["id"], "doc.md", "one\ntwo\nthree (typed by hand)\n")

    with pytest.raises(DraftError, match="changed since"):
        apply_hunks(proposal["token"])

    assert "typed by hand" in read_draft(draft["id"])["text"]


def test_a_token_can_only_be_used_once(drafts):
    draft = create_draft("Doc", "doc.md", "one\n")
    proposal = propose_edit(draft["id"], "doc.md", "ONE\n")
    apply_hunks(proposal["token"])

    with pytest.raises(DraftError, match="no longer pending"):
        apply_hunks(proposal["token"])


def test_an_unknown_token_is_refused(drafts):
    with pytest.raises(DraftError, match="no longer pending"):
        apply_hunks("not-a-real-token")


# ── Versions (W3) ──────────────────────────────────────────────────────────────

def test_the_previous_version_is_snapshotted_before_a_write(drafts):
    """
    What makes an accepted-by-mistake hunk recoverable, and what makes undo
    survive a page reload.
    """
    draft = create_draft("Doc", "doc.md", "original\n")
    proposal = propose_edit(draft["id"], "doc.md", "replaced\n")
    apply_hunks(proposal["token"])

    versions = workspace.list_versions(draft["id"], "doc.md")
    assert len(versions) == 1
    assert workspace.read_version(draft["id"], "doc.md", versions[0]["name"]) == "original\n"


def test_restore_snapshots_the_current_text_first(drafts):
    """Restoring is itself undoable."""
    draft = create_draft("Doc", "doc.md", "v1\n")
    save_draft_file(draft["id"], "doc.md", "v2\n")
    versions = workspace.list_versions(draft["id"], "doc.md")

    workspace.restore_version(draft["id"], "doc.md", versions[0]["name"])

    assert read_draft(draft["id"])["text"] == "v1\n"
    # v2 was snapshotted on the way past, so it is still recoverable.
    saved = [
        workspace.read_version(draft["id"], "doc.md", v["name"])
        for v in workspace.list_versions(draft["id"], "doc.md")
    ]
    assert "v2\n" in saved


def test_a_version_from_another_file_is_refused(drafts):
    draft = create_draft("Doc", "a.md", "one\n")
    add_draft_file(draft["id"], "b.md", "two\n")
    save_draft_file(draft["id"], "b.md", "two changed\n")
    b_version = workspace.list_versions(draft["id"], "b.md")[0]["name"]

    with pytest.raises(DraftError, match="not a version"):
        workspace.read_version(draft["id"], "a.md", b_version)


def test_save_refuses_when_the_file_moved_underneath(drafts):
    """A second tab (or an external edit) must not be silently clobbered."""
    draft = create_draft("Doc", "doc.md", "one\n")
    stale_hash = read_draft(draft["id"])["hash"]
    save_draft_file(draft["id"], "doc.md", "edited elsewhere\n")

    with pytest.raises(DraftError, match="changed since"):
        save_draft_file(draft["id"], "doc.md", "my version\n", expect_hash=stale_hash)


# ── Retention (W10) ────────────────────────────────────────────────────────────

def _age_draft(cfg, draft_id, days):
    """Back-date every file in a draft so it looks untouched for `days`."""
    import os as _os

    when = (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()
    for entry in (cfg.drafts_dir / draft_id).rglob("*"):
        if entry.is_file():
            _os.utime(entry, (when, when))


def test_a_stale_draft_is_removed(drafts):
    draft = create_draft("Old", "old.md", "stale\n")
    _age_draft(drafts, draft["id"], 40)

    removed = prune_drafts()

    assert [d["id"] for d in removed] == [draft["id"]]
    assert not (drafts.drafts_dir / draft["id"]).exists()


def test_activity_in_any_file_keeps_the_whole_draft(drafts):
    """
    Age is the newest mtime across the folder, so editing the .bib keeps the
    .tex alive — the behaviour you want for a paper and its bibliography.
    """
    draft = create_draft("Paper", "paper.tex", "old\n")
    add_draft_file(draft["id"], "refs.bib", "@article{a}\n")
    _age_draft(drafts, draft["id"], 40)
    save_draft_file(draft["id"], "refs.bib", "@article{b}\n")  # fresh again

    assert prune_drafts() == []
    assert (drafts.drafts_dir / draft["id"]).exists()


def test_a_kept_draft_is_exempt(drafts):
    draft = create_draft("Keeper", "keep.md", "important\n")
    set_keep(draft["id"], True)
    _age_draft(drafts, draft["id"], 400)

    assert prune_drafts() == []
    assert (drafts.drafts_dir / draft["id"]).exists()


def test_retention_zero_disables_the_sweep(drafts):
    draft = create_draft("Old", "old.md", "stale\n")
    _age_draft(drafts, draft["id"], 999)
    drafts.drafts_retention_days = 0

    assert stale_drafts() == []
    assert prune_drafts() == []
    assert (drafts.drafts_dir / draft["id"]).exists()


def test_dry_run_reports_without_deleting(drafts):
    """`kb drafts --prune --dry-run` must never be the thing that deletes."""
    draft = create_draft("Old", "old.md", "stale\n")
    _age_draft(drafts, draft["id"], 40)

    listed = prune_drafts(dry_run=True)

    assert [d["id"] for d in listed] == [draft["id"]]
    assert (drafts.drafts_dir / draft["id"]).exists()


def test_prune_never_follows_a_symlinked_draft_folder(drafts, tmp_path):
    """
    This is the only code path in jarvis that deletes a file, so its
    containment is re-checked immediately before the delete rather than
    trusted from the listing.
    """
    outside = tmp_path / "precious"
    outside.mkdir()
    (outside / "keepme.md").write_text("do not delete", encoding="utf-8")
    root = tmp_path / "drafts"
    root.mkdir(parents=True, exist_ok=True)
    (root / "20200101-000000-aaaaaa").symlink_to(outside, target_is_directory=True)

    prune_drafts()

    assert (outside / "keepme.md").exists()


def test_draft_age_uses_the_newest_file(drafts):
    draft = create_draft("Doc", "a.md", "one\n")
    add_draft_file(draft["id"], "b.md", "two\n")
    _age_draft(drafts, draft["id"], 10)
    save_draft_file(draft["id"], "b.md", "two again\n")

    assert draft_age_days(draft["id"]) < 1


# ── Explicit deletion (W10's second, human-only path) ──────────────────────────

def test_delete_draft_removes_it(drafts):
    from jarvis.drafts.workspace import delete_draft

    draft = create_draft("Scratch", "s.md", "text\n")
    metadata = delete_draft(draft["id"])

    assert metadata["title"] == "Scratch"
    assert not (drafts.drafts_dir / draft["id"]).exists()
    assert list_drafts() == []


def test_delete_draft_refuses_an_unknown_or_invalid_id(drafts):
    from jarvis.drafts.workspace import delete_draft

    with pytest.raises(DraftError):
        delete_draft("20260101-000000-abcdef")     # well-formed but absent
    with pytest.raises(DraftError):
        delete_draft("../escape")                  # not even an id


def test_delete_draft_never_follows_a_symlinked_folder(drafts, tmp_path):
    """
    The same re-check retention does, for the same reason: this is one of only
    two code paths in jarvis that remove a file, so containment is confirmed
    immediately before the delete rather than trusted from the caller.
    """
    from jarvis.drafts.workspace import delete_draft

    outside = tmp_path / "precious"
    outside.mkdir()
    (outside / "keepme.md").write_text("do not delete", encoding="utf-8")
    root = tmp_path / "drafts"
    root.mkdir(parents=True, exist_ok=True)
    (root / "20200101-000000-aaaaaa").symlink_to(outside, target_is_directory=True)

    with pytest.raises(DraftError):
        delete_draft("20200101-000000-aaaaaa")
    assert (outside / "keepme.md").exists()


def test_deleting_a_draft_leaves_a_copy_of_it_alone(drafts, tmp_path):
    """
    Once a file has been copied out of the sandbox that copy is an ordinary
    file of the user's, and removing the working draft must never reach it.
    """
    import shutil

    from jarvis.drafts.workspace import delete_draft, resolve_in_draft

    draft = create_draft("Report", "r.md", "the final text\n")
    destination = tmp_path / "somewhere-else"
    destination.mkdir()
    shutil.copy2(resolve_in_draft(draft["id"], "r.md"), destination / "r.md")

    delete_draft(draft["id"])

    assert (destination / "r.md").read_text() == "the final text\n"


def test_no_chat_tool_can_delete_a_draft():
    """Deletion stays a human action; the model has no tool for it."""
    from jarvis.chat.chat import TOOLS

    names = {tool["function"]["name"] for tool in TOOLS}
    assert not any("delete" in name or "remove_draft" in name for name in names)


# ── How a change should be laid out for review ────────────────────────────────

@pytest.mark.parametrize("before,after,expected", [
    ("a\nb\n",           "a\nb\nc\nd\n",  "add"),
    ("a\nb\nc\n",        "a\nc\n",        "remove"),
    ("a\nold line\nc\n", "a\nnew line\nc\n", "replace"),
    ("a\nold\nb\n",      "a\nnew\nb\nextra\n", "replace"),
])
def test_hunk_kind_comes_from_the_opcodes_not_the_line_counts(drafts, before, after, expected):
    """
    The editor shows additions and removals inline and replacements as two
    columns, so it needs to know which a change is. Counting lines cannot tell
    it: a hunk carries context on both sides, so neither side is ever empty and
    every change would look like a replacement.
    """
    draft = create_draft("Doc", "doc.md", before)
    hunks = propose_edit(draft["id"], "doc.md", after)["hunks"]

    assert [h["kind"] for h in hunks] == [expected]


def test_a_latex_document_compiles_against_the_files_beside_it(drafts, monkeypatch):
    """
    The reason a document is a folder rather than a file. compile_latex seeds
    its temp directory from the whole draft, so a .tex reaches its .bib and its
    chapters — and a file in a *different* draft is not reachable, which is the
    same containment that keeps the sandbox a sandbox.
    """
    from jarvis.drafts import add_draft_file
    from jarvis.drafts.render import compile_latex

    paper = create_draft("Paper", "main.tex", "\\documentclass{article}\n")
    add_draft_file(paper["id"], "refs.bib", "@article{x, title={X}}\n")
    add_draft_file(paper["id"], "chapter1.tex", "chapter one\n")
    elsewhere = create_draft("Other", "other.tex", "unrelated\n")

    seeded = []
    monkeypatch.setattr("jarvis.drafts.render._tool_available", lambda tool: True)

    def fake_run(command, cwd, timeout):
        seeded.extend(sorted(p.name for p in cwd.iterdir()))
        return type("R", (), {"returncode": 1, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("jarvis.drafts.render._run", fake_run)
    compile_latex(paper["id"], "main.tex")

    assert seeded == ["chapter1.tex", "main.tex", "refs.bib"]
    assert "other.tex" not in seeded
    assert elsewhere["id"] != paper["id"]


def test_hunk_spans_cover_only_the_lines_that_change():
    """
    The bug behind "it says it removes 3 blocks but highlights 4".

    A grouped opcode carries `context` lines on each side of the change, so
    old_lines/new_lines span more than the change does. A caller highlighting
    the whole span paints untouched text as removed — here the fourth block,
    which is only ever context, appeared to be going too.
    """
    from jarvis.drafts.workspace import _hunks

    old = [f"{line}\n" for line in [
        "# Report", "", "## Block A", "content A", "",
        "## Status", "status 1", "",
        "## Status", "status 2", "",
        "## Status", "status 3", "",
        "## Status", "status 4 keep", "",
        "## Block Z", "content Z",
    ]]
    new = old[:5] + old[14:]          # drop exactly three status blocks

    hunk = _hunks(old, new)[0]
    highlighted = [
        hunk["old_lines"][i]
        for start, end in hunk["old_spans"]
        for i in range(start, end)
    ]

    assert "status 4 keep" not in highlighted, (
        "a line that survives the edit must never be highlighted as removed"
    )
    assert "content A" not in highlighted and "content Z" not in highlighted, (
        "leading and trailing context must not be highlighted"
    )
    # Three blocks' worth of lines go: three headers, three bodies, three blanks.
    assert len(highlighted) == 9
    assert highlighted.count("## Status") == 3


def test_hunk_spans_are_offsets_into_the_lines_the_caller_was_handed():
    """
    The spans index old_lines/new_lines, not the original file. Returning file
    offsets would silently highlight the wrong region, since a hunk's lines
    start at old_start rather than at zero.
    """
    from jarvis.drafts.workspace import _hunks

    old = [f"line {i}\n" for i in range(30)]
    new = old[:20] + old[21:]         # remove one line, well past the start

    hunk = _hunks(old, new)[0]
    start, end = hunk["old_spans"][0]

    assert 0 <= start < end <= len(hunk["old_lines"])
    assert hunk["old_lines"][start:end] == ["line 20"]


def test_an_addition_marks_only_the_added_lines():
    """The same guarantee on the other side of the diff."""
    from jarvis.drafts.workspace import _hunks

    old = [f"line {i}\n" for i in range(10)]
    new = old[:5] + ["brand new\n"] + old[5:]

    hunk = _hunks(old, new)[0]
    added = [
        hunk["new_lines"][i]
        for start, end in hunk["new_spans"]
        for i in range(start, end)
    ]

    assert added == ["brand new"]
    assert hunk["old_spans"] == [], "nothing is removed by a pure addition"
