"""
The agent-writable draft sandbox.

One zone, and no door out of it:

    ~/.jarvis/drafts/          |  ~/vault/
    agent writes here freely   |  read-only to the model, indexed by sync
    you edit here              |  nothing here is reachable from the sandbox
    edits arrive as diffs      |  you copy files across yourself, in Finder

The agent CAN write — that is the point: "tailor my CV to this job ad" should
produce a real file you can open, not a wall of chat text. It simply cannot
write anywhere that matters. Prompt injection therefore buys an attacker a file
in a scratch folder that is never indexed, never executed, and never leaves the
sandbox at all.

There was once a password-gated archive route that copied a draft into the
vault. It is gone: the editor now just reveals a draft in the file manager, and
moving it anywhere is an ordinary file operation the user performs themselves.
A gate that no code can open is a stronger guarantee than a gate with a
password on it.
"""

from .render import (
    RenderError,
    compile_latex,
    export_engine,
    latex_available,
    markdown_to_html,
    markdown_to_pdf,
    pandoc_available,
    pdf_export_available,
)
from .workspace import (
    DraftError,
    add_draft_file,
    apply_hunks,
    create_draft,
    delete_draft,
    draft_age_days,
    list_drafts,
    prune_drafts,
    propose_edit,
    read_draft,
    read_proposal,
    resolve_in_draft,
    save_draft_file,
    set_keep,
    stale_drafts,
)

__all__ = [
    "DraftError",
    "RenderError",
    "compile_latex",
    "export_engine",
    "latex_available",
    "markdown_to_html",
    "markdown_to_pdf",
    "pandoc_available",
    "pdf_export_available",
    "add_draft_file",
    "apply_hunks",
    "create_draft",
    "delete_draft",
    "draft_age_days",
    "list_drafts",
    "propose_edit",
    "prune_drafts",
    "read_draft",
    "read_proposal",
    "resolve_in_draft",
    "save_draft_file",
    "set_keep",
    "stale_drafts",
]
