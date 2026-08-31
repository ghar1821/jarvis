"""
Tests for the editor routes in jarvis/webapp/app.py.

These routes are the human side of the draft sandbox: reading, saving,
previewing, compiling, reviewing an agent's diff, and revealing a file in the
OS file manager. The model reaches drafts only through its chat tools and
cannot reach any of these routes, which is what lets `/reveal` exist at all.

The LaTeX cases assert on how the compiler is *invoked* rather than running it,
so the security properties are verified on a machine with no TeX installed.
"""

import json

import pytest
from starlette.testclient import TestClient

import jarvis.webapp.app as appmod
from jarvis.core.config import Config
from jarvis.drafts import workspace


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Isolated drafts folder and vault for the webapp under test."""
    cfg = Config(
        drafts_dir=tmp_path / "drafts",
        vault_path=tmp_path / "vault",
    )
    (tmp_path / "vault").mkdir()
    monkeypatch.setattr("jarvis.drafts.workspace.get_config", lambda: cfg)
    monkeypatch.setattr("jarvis.drafts.render.get_config", lambda: cfg)
    monkeypatch.setattr(appmod, "cfg", cfg)
    workspace._proposals.clear()
    return cfg


@pytest.fixture
def client():
    return TestClient(appmod.app, base_url="http://127.0.0.1")


# ── Listing, reading, saving ───────────────────────────────────────────────────

def test_drafts_route_lists_with_capabilities(sandbox, client):
    """
    The UI hides the compile and export buttons when the toolchain is absent
    rather than offering something that can only fail.
    """
    from jarvis.drafts import create_draft

    create_draft("My CV", "cv.md", "# CV\n")
    body = client.get("/drafts").json()

    assert [d["title"] for d in body["drafts"]] == ["My CV"]
    assert body["retention_days"] == sandbox.drafts_retention_days
    assert "latex" in body and "pandoc" in body


def test_save_writes_through_and_returns_the_new_hash(sandbox, client):
    from jarvis.drafts import create_draft, read_draft

    draft = create_draft("Doc", "doc.md", "one\n")
    response = client.post("/drafts/save", json={
        "draft_id": draft["id"], "file": "doc.md", "content": "two\n",
    })

    assert response.status_code == 200
    assert read_draft(draft["id"])["text"] == "two\n"
    assert response.json()["hash"] == read_draft(draft["id"])["hash"]


def test_save_refuses_a_stale_hash(sandbox, client):
    """A second tab must not silently clobber the first."""
    from jarvis.drafts import create_draft, read_draft

    draft = create_draft("Doc", "doc.md", "one\n")
    stale = read_draft(draft["id"])["hash"]
    client.post("/drafts/save", json={
        "draft_id": draft["id"], "file": "doc.md", "content": "edited elsewhere\n",
    })

    response = client.post("/drafts/save", json={
        "draft_id": draft["id"], "file": "doc.md", "content": "mine\n", "expect_hash": stale,
    })

    assert response.status_code == 400
    assert "changed since" in response.json()["detail"]
    assert read_draft(draft["id"])["text"] == "edited elsewhere\n"


def test_routes_refuse_paths_outside_the_sandbox(sandbox, client):
    """The containment policy applies to the HTTP surface too, as a 400."""
    from jarvis.drafts import create_draft

    draft = create_draft("Doc", "doc.md", "one\n")

    assert client.get(f"/drafts/{draft['id']}/file", params={"file": "../escape.md"}).status_code == 400
    assert client.post("/drafts/save", json={
        "draft_id": draft["id"], "file": "../escape.md", "content": "x",
    }).status_code == 400
    assert client.get("/drafts/..%2F..%2Fetc/file", params={"file": "passwd.md"}).status_code in (400, 404)


# ── Preview ────────────────────────────────────────────────────────────────────

def test_preview_escapes_embedded_html(sandbox, client):
    """
    A draft can hold text the model produced from an untrusted document. The
    preview must never be able to run script in the app's origin — the HTML is
    escaped at render time, and the browser puts it in a sandbox="" iframe.
    """
    from jarvis.drafts import create_draft

    draft = create_draft("Doc", "doc.md", "# Title\n\n<script>alert(1)</script>\n")
    html = client.post("/preview", json={"draft_id": draft["id"], "file": "doc.md"}).json()["html"]

    assert "<h1>Title</h1>" in html
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_preview_iframe_is_sandboxed_in_the_markup(sandbox):
    """
    The other half of the pair: the frame the HTML lands in must carry an empty
    sandbox attribute, which denies scripts and same-origin access.
    """
    from pathlib import Path

    markup = (Path(appmod.__file__).parent / "index.html").read_text()
    assert 'id="preview-frame" sandbox=""' in markup


# ── Compilation sandboxing ─────────────────────────────────────────────────────

def _capture_compile(monkeypatch):
    """Capture the argv and environment a compile would run with."""
    captured = {}

    class _Result:
        stdout = ""
        stderr = ""

    def fake_run(command, cwd, env, capture_output, text, timeout, check):
        captured["command"] = command
        captured["env"] = env
        captured["cwd"] = cwd
        captured["timeout"] = timeout
        return _Result()

    monkeypatch.setattr("jarvis.drafts.render.subprocess.run", fake_run)
    monkeypatch.setattr("jarvis.drafts.render.shutil.which", lambda name: f"/usr/bin/{name}")
    return captured


def test_compile_disables_shell_escape_and_restricts_file_access(sandbox, client, monkeypatch):
    """
    Compiling a .tex the model wrote from untrusted input is the sharpest edge
    in the codebase. Shell escape off blocks \\write18; openin/openout paranoid
    blocks reading /etc/passwd into the PDF and writing outside the work dir.
    """
    from jarvis.drafts import create_draft

    captured = _capture_compile(monkeypatch)
    draft = create_draft("Paper", "paper.tex", "\\documentclass{article}\n")

    client.post("/compile", json={"draft_id": draft["id"], "file": "paper.tex"})

    assert "-no-shell-escape" in captured["command"]
    assert captured["env"]["openin_any"] == "p"
    assert captured["env"]["openout_any"] == "p"
    assert captured["env"]["TEXMFHOME"] == ""
    assert captured["timeout"] == sandbox.compile_timeout_seconds


def test_compile_never_runs_in_the_draft_folder(sandbox, client, monkeypatch):
    """
    A temp copy, never in place — so nothing the document writes can reach the
    user's draft.
    """
    from jarvis.drafts import create_draft
    from jarvis.drafts.workspace import draft_dir

    captured = _capture_compile(monkeypatch)
    draft = create_draft("Paper", "paper.tex", "\\documentclass{article}\n")

    client.post("/compile", json={"draft_id": draft["id"], "file": "paper.tex"})

    assert captured["cwd"] != draft_dir(draft["id"])


def test_compile_returns_the_log_when_it_fails(sandbox, client, monkeypatch):
    """A LaTeX error is part of writing LaTeX — show the log, don't hide it."""
    from jarvis.drafts import create_draft

    _capture_compile(monkeypatch)   # produces no PDF
    draft = create_draft("Paper", "paper.tex", "\\documentclass{article}\n")

    response = client.post("/compile", json={"draft_id": draft["id"], "file": "paper.tex"})

    assert response.status_code == 422
    assert "log" in response.json()


def test_compile_degrades_cleanly_without_a_toolchain(sandbox, client, monkeypatch):
    """A machine with no LaTeX should still be able to edit .tex."""
    from jarvis.drafts import create_draft

    monkeypatch.setattr("jarvis.drafts.render.shutil.which", lambda name: None)
    draft = create_draft("Paper", "paper.tex", "\\documentclass{article}\n")

    response = client.post("/compile", json={"draft_id": draft["id"], "file": "paper.tex"})

    assert response.status_code == 400
    assert "not installed" in response.json()["detail"]
def test_restore_puts_a_version_back_and_keeps_the_current_one(sandbox, client):
    from jarvis.drafts import create_draft, read_draft

    draft = create_draft("Doc", "doc.md", "v1\n")
    client.post("/drafts/save", json={"draft_id": draft["id"], "file": "doc.md", "content": "v2\n"})
    versions = client.get(f"/drafts/{draft['id']}/file", params={"file": "doc.md"}).json()["versions"]

    client.post("/drafts/restore", json={
        "draft_id": draft["id"], "file": "doc.md", "version": versions[0]["name"],
    })

    assert read_draft(draft["id"])["text"] == "v1\n"
    # v2 was snapshotted on the way past, so a restore is itself undoable.
    after = client.get(f"/drafts/{draft['id']}/file", params={"file": "doc.md"}).json()
    assert len(after["versions"]) == 2
def test_preview_renders_maths_as_mathml(sandbox, client):
    """
    MathML rather than a JavaScript typesetter: the preview iframe is
    `sandbox=""` and runs no scripts, so KaTeX or MathJax could never execute
    there. Browsers render MathML natively, so the maths arrives already laid
    out and the sandbox stays shut.
    """
    from jarvis.drafts import create_draft

    draft = create_draft(
        "Math", "m.md",
        "Inline $E = mc^2$ here.\n\n$$\\int_0^\\infty e^{-x}\\,dx = 1$$\n",
    )
    html = client.post("/preview", json={"draft_id": draft["id"], "file": "m.md"}).json()["html"]

    assert "<math" in html
    assert 'display="inline"' in html      # the inline one
    assert 'display="block"' in html       # the displayed one, centred and full size
    assert "<script" not in html


def test_malformed_maths_shows_its_source_rather_than_breaking_the_preview(sandbox, client):
    from jarvis.drafts import create_draft
    import jarvis.drafts.render as render

    def explode(*args, **kwargs):
        raise ValueError("no")

    draft = create_draft("Math", "m.md", "Broken: $\\frac{$\n")
    # Force the conversion to fail, whatever the converter happens to tolerate.
    original = render.latex2mathml if hasattr(render, "latex2mathml") else None
    import latex2mathml.converter
    saved = latex2mathml.converter.convert
    latex2mathml.converter.convert = explode
    try:
        html = client.post("/preview", json={"draft_id": draft["id"], "file": "m.md"}).json()["html"]
    finally:
        latex2mathml.converter.convert = saved

    assert "math-error" in html          # shown as source, not swallowed
    assert "&lt;" not in html or "<script" not in html


def test_pdf_export_uses_the_configured_margin(sandbox, monkeypatch):
    """pandoc's default leaves about an inch and a half on every side."""
    from jarvis.drafts import create_draft
    import jarvis.drafts.render as render

    captured = {}

    class _Result:
        stdout = stderr = ""

    def fake_run(command, cwd, env, capture_output, text, timeout, check):
        captured["command"] = command
        (cwd / (command[1].lstrip("./").replace(".md", ".pdf"))).write_bytes(b"%PDF-1.4\n")
        return _Result()

    monkeypatch.setattr(render.subprocess, "run", fake_run)
    monkeypatch.setattr(render.shutil, "which", lambda name: f"/usr/bin/{name}")
    sandbox.pdf_margin = "1.5cm"

    draft = create_draft("Doc", "d.md", "# Hi\n")
    render.markdown_to_pdf(draft["id"], "d.md")

    assert "geometry:margin=1.5cm" in captured["command"]
    assert "markdown+tex_math_dollars" in captured["command"]


# ── /reveal: show a draft file in the OS file manager ─────────────────────────
#
# This replaced the password-gated archive route. It runs a command, so the
# path it is given matters more than most: the containment check and the
# absence of a shell are the whole security story.


def test_reveal_runs_a_file_manager_on_the_draft_file(sandbox, client, monkeypatch):
    """The happy path: a fixed argv, no shell, pointing at the real file."""
    draft = workspace.create_draft("Paper", "paper.tex", "\\documentclass{article}\n")
    calls = []
    monkeypatch.setattr(appmod.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)))

    response = client.post("/reveal", json={"draft_id": draft["id"], "file": "paper.tex"})

    assert response.status_code == 200
    command, kwargs = calls[0]
    assert isinstance(command, list), "a string command would go through a shell"
    assert kwargs.get("shell") is not True
    assert str(workspace.resolve_in_draft(draft["id"], "paper.tex")) in command
    assert response.json()["path"].endswith("paper.tex")


@pytest.mark.parametrize(
    "draft_id, file",
    [
        ("../../etc", "passwd"),
        ("20260101-000000-abcdef", "../../../etc/passwd"),
        ("20260101-000000-abcdef", "/etc/passwd"),
        ("not a draft id!", "x.md"),
    ],
)
def test_reveal_refuses_anything_outside_the_sandbox(sandbox, client, monkeypatch, draft_id, file):
    """
    The route hands a path to a subprocess, so containment is what stops it
    revealing — or on a file manager that opens what it is given, exposing —
    something outside the drafts folder.
    """
    def _never(*args, **kwargs):
        raise AssertionError("a rejected path must never reach a subprocess")

    monkeypatch.setattr(appmod.subprocess, "run", _never)

    response = client.post("/reveal", json={"draft_id": draft_id, "file": file})
    assert response.status_code in (400, 404)


def test_reveal_404s_on_a_file_that_is_not_there(sandbox, client, monkeypatch):
    """A valid-looking name for a file that does not exist is a 404, not a run."""
    draft = workspace.create_draft("Paper", "paper.tex", "x\n")
    monkeypatch.setattr(
        appmod.subprocess, "run",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")),
    )

    response = client.post("/reveal", json={"draft_id": draft["id"], "file": "missing.tex"})
    assert response.status_code == 404


def test_no_chat_tool_can_reveal_a_file():
    """
    Opening a file manager is a human action. If the model could trigger it,
    a prompt injection could make windows appear on the user's desktop.
    """
    from jarvis.chat.chat import TOOLS

    names = {tool["function"]["name"] for tool in TOOLS}
    assert not [n for n in names if "reveal" in n or "finder" in n]


# ── Pending proposals ─────────────────────────────────────────────────────────
#
# Proposals live in this process's memory. Within one run a suggestion the user
# navigated away from used to be unreachable, with nothing saying it was still
# there — these cover the routes that make it findable and clearable.


def test_proposals_lists_what_is_still_waiting(sandbox, client):
    draft = workspace.create_draft("Report", "r.md", "one\ntwo\n")
    proposal = workspace.propose_edit(draft["id"], "r.md", "one\nTWO\n", "tighten it")

    listed = client.get("/proposals").json()["proposals"]

    assert [p["token"] for p in listed] == [proposal["token"]]
    assert listed[0]["file"] == "r.md"
    assert listed[0]["rationale"] == "tighten it"
    assert listed[0]["hunks"], "a proposal with no hunks cannot be reviewed"


def test_a_listed_proposal_carries_what_the_live_event_carries(sandbox, client):
    """
    The browser renders a re-opened suggestion with the same code that renders
    a live one, so the two shapes have to match — a missing `kind` would send
    every re-opened replacement down the inline path instead of side by side.
    """
    draft = workspace.create_draft("Report", "r.md", "a\nold\nb\n")
    workspace.propose_edit(draft["id"], "r.md", "a\nnew\nb\n")

    hunk = client.get("/proposals").json()["proposals"][0]["hunks"][0]

    assert set(hunk) >= {
        "index", "kind", "old_start", "old_end", "old_lines", "new_lines",
        # Without the spans the browser tints the whole hunk, context and all,
        # so a removal looks like it is taking its neighbours with it.
        "old_spans", "new_spans",
    }


def test_a_listed_proposal_does_not_carry_the_whole_proposed_file(sandbox, client):
    """The browser needs the hunks; the full rewrite is not its business."""
    draft = workspace.create_draft("Report", "r.md", "one\n")
    workspace.propose_edit(draft["id"], "r.md", "one\nsecret new draft body\n")

    assert "new_text" not in client.get("/proposals").json()["proposals"][0]


def test_discard_all_clears_every_pending_proposal(sandbox, client):
    first = workspace.create_draft("One", "a.md", "a\n")
    second = workspace.create_draft("Two", "b.md", "b\n")
    workspace.propose_edit(first["id"], "a.md", "a changed\n")
    workspace.propose_edit(second["id"], "b.md", "b changed\n")

    assert client.post("/proposals/discard-all").json()["discarded"] == 2
    assert client.get("/proposals").json()["proposals"] == []


def test_discarding_proposals_changes_no_file(sandbox, client):
    """Clearing suggestions is not applying or reverting them."""
    draft = workspace.create_draft("Report", "r.md", "the original\n")
    workspace.propose_edit(draft["id"], "r.md", "something else\n")

    client.post("/proposals/discard-all")

    assert workspace.read_draft(draft["id"], "r.md")["text"] == "the original\n"


# ── A draft is a folder ───────────────────────────────────────────────────────


def test_new_file_can_join_an_existing_document(sandbox, client):
    """
    A LaTeX document is several files that compile together, which only works
    if they share a folder — so the route has to be able to add to one.
    """
    draft = client.post("/drafts/new", json={"filename": "main.tex", "title": "Paper"}).json()

    response = client.post(
        "/drafts/new", json={"draft_id": draft["id"], "filename": "refs.bib"}
    )

    assert response.status_code == 200
    assert response.json()["id"] == draft["id"], "it must not start a second document"
    listed = {d["id"]: d for d in client.get("/drafts").json()["drafts"]}
    assert sorted(listed[draft["id"]]["files"]) == ["main.tex", "refs.bib"]


def test_joining_a_document_still_refuses_a_path(sandbox, client):
    """The containment check is not skipped on the add-to-existing branch."""
    draft = client.post("/drafts/new", json={"filename": "main.tex"}).json()

    response = client.post(
        "/drafts/new", json={"draft_id": draft["id"], "filename": "../escape.tex"}
    )

    assert response.status_code == 400


def test_joining_a_document_refuses_an_existing_filename(sandbox, client):
    """Adding a file must never be a way to blank one."""
    draft = client.post("/drafts/new", json={"filename": "main.tex"}).json()
    workspace.save_draft_file(draft["id"], "main.tex", "\\documentclass{article}\n")

    response = client.post(
        "/drafts/new", json={"draft_id": draft["id"], "filename": "main.tex"}
    )

    assert response.status_code == 400
    assert workspace.read_draft(draft["id"], "main.tex")["text"] != ""


def test_the_compile_directory_excludes_the_sandbox_bookkeeping(sandbox, client, monkeypatch):
    """
    draft.json is the sandbox's own metadata, not part of the document. Copying
    it into the working directory would put it within reach of an \\input{} in
    a .tex the model wrote.
    """
    from jarvis.drafts import render

    draft = workspace.create_draft("Paper", "main.tex", "\\documentclass{article}\n")
    seeded = []
    monkeypatch.setattr(render, "_tool_available", lambda tool: True)
    monkeypatch.setattr(
        render, "_run",
        lambda command, cwd, timeout: (
            seeded.extend(p.name for p in cwd.iterdir()),
            type("R", (), {"returncode": 1, "stdout": "", "stderr": ""})(),
        )[1],
    )

    client.post("/compile", json={"draft_id": draft["id"], "file": "main.tex"})

    assert "draft.json" not in seeded
