"""
Static checks on the webapp's frontend.

There is no JS test harness here, so these are structural checks over the
markup, script, and stylesheet — cheap, precise, and each one guards a bug that
actually happened.

Deliberately NOT here: a check that every function called in app.js is still
declared. That is the gap that let an edit remove `maybeOfferDraft` while
`renderTurn` still called it, blanking the chat history while the session list
looked perfectly healthy — so it is the check one most wants. Two attempts at
it were abandoned. JavaScript cannot be lexed with regular expressions: the
version that stripped string literals silently ate a third of the file
(including the declarations it was looking for), and the version that left them
in flagged English prose like "Pin (never auto-deleted)" as an undefined call.
A test that cries wolf gets muted, which is worse than no test. Doing this
properly needs a real JS parser, which would mean a Node toolchain this project
does not have and does not otherwise want. Until then it is caught by running
the app.
"""

import re
from pathlib import Path

import pytest

WEBAPP = Path(__file__).parent.parent / "jarvis" / "webapp"
APP_JS = WEBAPP / "static" / "app.js"
INDEX_HTML = WEBAPP / "index.html"
STYLE_CSS = WEBAPP / "static" / "style.css"

def test_every_element_id_the_js_reaches_for_exists_in_the_markup():
    """A getElementById for markup that is not there returns null, and the
    next property access throws — taking the rest of the script with it."""
    html = INDEX_HTML.read_text()
    js = APP_JS.read_text()

    present = set(re.findall(r'id="([^"]+)"', html))
    wanted = set(re.findall(r"getElementById\('([^']+)'\)", js))

    assert not (wanted - present), sorted(wanted - present)


@pytest.mark.parametrize("element", ["editor-view"])
def test_hideable_panes_can_actually_be_hidden(element):
    """
    An id selector outranks `.hidden`, so a pane given `display` by its id rule
    cannot be hidden by the class unless something out-specifies it. That bug
    made the editor toggle do nothing at all.
    """
    css = STYLE_CSS.read_text()
    has_id_display = re.search(rf"#{element}\s*\{{[^}}]*display:", css, re.S)
    if not has_id_display:
        return
    assert re.search(rf"#{element}\.hidden\s*\{{[^}}]*display:\s*none", css), (
        f"#{element} sets display via its id, so it needs a "
        f"`#{element}.hidden {{ display: none }}` rule to be hideable"
    )


def test_the_editor_never_saves_on_a_timer():
    """
    An idle autosave used to write the editor buffer to disk. During a review
    that buffer deliberately holds the current text and the suggested text at
    once, so it wrote both — which reads afterwards as "the change was accepted
    but the old text is still there". Saving now only ever follows something
    the user did.
    """
    js = APP_JS.read_text()

    assert "scheduleAutosave" not in js, "the autosave scheduler is gone"
    # Checked against what a timer would actually look like rather than the
    # word "autosave", which legitimately appears in a comment explaining why
    # there is no longer one.
    timed = [
        js[match.start():match.start() + 160]
        for match in re.finditer(r"set(?:Timeout|Interval)\s*\(", js)
    ]
    assert not [block for block in timed if "saveDraft" in block], (
        "saving must follow something the user did, never a timer"
    )
    # The original failure was a guard reading a variable that nothing
    # assigned any more, so it never fired once. The state the editor guards on
    # must actually be populated somewhere.
    assert re.search(r"pendingProposals = new Map\(proposals", js), (
        "pendingProposals must be populated from the server, or every check "
        "against it silently reports nothing pending"
    )
    assert "scheduleAutosave" not in js and "autosaveTimer" not in js, (
        "the autosave state is gone, not merely unused"
    )


def test_a_save_reads_the_tab_own_document_not_the_screen():
    """
    The structural half of the same bug. A review is rendered into a Doc of its
    own, so the only way a save can write two versions of a file is by reading
    what the editor is displaying instead of what the tab holds. Reading
    `tab.doc` makes that impossible rather than merely guarded against.
    """
    js = APP_JS.read_text()
    saver = re.search(r"async function saveDraft\(.*?\n\}", js, re.S)
    assert saver, "saveDraft not found"

    assert "tab.doc.getValue()" in saver.group(0), (
        "saveDraft must take its content from the tab's own Doc"
    )
    assert "cm.getValue()" not in saver.group(0), (
        "saveDraft must never read the editor, which during a review is "
        "showing a different Doc holding both versions of the file"
    )


def test_the_tab_close_control_distinguishes_saved_from_unsaved():
    """
    The dot/× distinction is the only thing telling you an unopened tab has
    unsaved work in it, and it lives entirely in CSS `content`. A rule lost in
    an edit would leave every tab looking saved.
    """
    css = STYLE_CSS.read_text()

    saved = re.search(r"\.editor-tab-close::before\s*\{([^}]*)\}", css)
    unsaved = re.search(r"\.editor-tab-close\.unsaved::before\s*\{([^}]*)\}", css)
    assert saved and "content:" in saved.group(1), "no glyph for a saved tab"
    assert unsaved and "content:" in unsaved.group(1), "no glyph for an unsaved tab"
    assert saved.group(1) != unsaved.group(1), (
        "saved and unsaved tabs must not render the same glyph"
    )
