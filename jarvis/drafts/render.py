"""
Preview and export for draft documents — Markdown to HTML, LaTeX to PDF.

Both run entirely on the user's machine with no model in the loop, which is why
they are allowed on a private draft: the privacy model is about what reaches a
cloud provider, and nothing here leaves the laptop.

Compiling a `.tex` the model wrote from untrusted input is the sharpest edge in
this codebase, so the compiler is boxed in on four sides:

- `-no-shell-escape` — blocks `\\write18`, which would otherwise run arbitrary
  commands from inside a document.
- `openin_any=p` / `openout_any=p` — blocks `\\input{/etc/passwd}`-style
  exfiltration into the PDF, and writes outside the working directory.
- A temporary directory seeded with a copy of the draft, never compiled in
  place, so a stray `\\write` cannot touch the user's files.
- A hard timeout, so a `\\loop` bomb dies rather than pinning a core.

Everything degrades with a clear message when the toolchain is missing, rather
than crashing: a machine without LaTeX should still be able to edit `.tex`.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

from jarvis.core.config import get_config

from .workspace import METADATA_FILE, DraftError, draft_dir, resolve_in_draft

# TeX's own environment switches for restricting file access. "p" means
# "paranoid": no absolute paths, no dotfiles, nothing above the working dir.
_RESTRICTED_TEX_ENV = {
    "openin_any": "p",
    "openout_any": "p",
    # Never let a document pull inputs from the user's home TeX tree.
    "TEXMFHOME": "",
}


class RenderError(Exception):
    """A preview or compile that could not run — always with a reason."""


def _mathml(latex: str, block: bool) -> str:
    """
    Convert one LaTeX expression to MathML, or fall back to showing the source.

    MathML rather than a JavaScript typesetter because the preview iframe is
    `sandbox=""` and runs no scripts — KaTeX or MathJax simply cannot execute
    there. Browsers render MathML natively, so the maths arrives already laid
    out and the sandbox stays shut.
    """
    import latex2mathml.converter

    try:
        # display="block" is what makes a displayed equation centre and use
        # full-size operators. The plugin already supplies the surrounding
        # <div class="math block">, so no extra wrapper is added here.
        return latex2mathml.converter.convert(
            latex, display="block" if block else "inline"
        )
    except Exception:
        # Malformed maths should show as the source the user typed, not
        # disappear or take the whole preview down with it.
        from html import escape

        return '<code class="math-error">' + escape(latex) + '</code>'


def markdown_to_html(text: str) -> str:
    """
    Render Markdown for the preview pane, maths included.

    HTML embedded in the source is NOT passed through: a draft can contain text
    the model produced from an untrusted document, and the preview is displayed
    inside the app's own origin. The caller sandboxes the iframe as well; this
    is the belt to that pair of braces.
    """
    from markdown_it import MarkdownIt
    from mdit_py_plugins.dollarmath import dollarmath_plugin

    renderer = MarkdownIt("commonmark", {"html": False, "linkify": True, "typographer": True})
    renderer.enable("table")
    renderer.enable("strikethrough")
    # $inline$ and $$display$$, converted to MathML at render time. The plugin
    # hands the renderer an options dict whose "display_mode" says which of the
    # two it is.
    renderer.use(
        dollarmath_plugin,
        renderer=lambda content, options: _mathml(content, block=options.get("display_mode", False)),
    )
    return renderer.render(text)


def _tool_available(name: str) -> bool:
    return bool(name) and shutil.which(name) is not None


def latex_available() -> bool:
    return _tool_available(get_config().latex_engine)


def pandoc_available() -> bool:
    return _tool_available("pandoc")


def _run(command: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    import os

    environment = {**os.environ, **_RESTRICTED_TEX_ENV}
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RenderError(
            f"Timed out after {timeout}s. A document that never finishes compiling is "
            "usually an unterminated environment or an accidental infinite loop."
        ) from exc


def compile_latex(draft_id: str, filename: str) -> dict:
    """
    Compile one draft `.tex` to PDF in a sandboxed temp directory.

    Returns {"pdf": bytes | None, "log": str, "ok": bool}. A failed compile is
    a normal outcome with a log to show, not an exception — LaTeX errors are
    part of writing LaTeX.
    """
    cfg = get_config()
    engine = cfg.latex_engine
    if not engine:
        raise RenderError("LaTeX compilation is disabled ([drafts] latex_engine is empty).")
    if not _tool_available(engine):
        raise RenderError(
            f"{engine!r} is not installed, so .tex files cannot be compiled here. "
            "Install a TeX distribution (MacTeX, TeX Live) or set "
            "[drafts] latex_engine = \"\" to hide the compile button."
        )

    source = resolve_in_draft(draft_id, filename)
    if not source.exists():
        raise DraftError(f"No file {filename!r} in draft {draft_id!r}")
    if source.suffix.lower() != ".tex":
        raise RenderError(f"{filename!r} is not a .tex file")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        # Seed with the draft's own files (the .bib and any inputs) — but never
        # compile in place, so nothing the document writes can reach the draft.
        # draft.json is the sandbox's own bookkeeping and no part of the
        # document; leaving it here would put it within reach of an \input{}
        # in a .tex the model wrote.
        for entry in draft_dir(draft_id).iterdir():
            if entry.is_file() and not entry.name.startswith(".") and entry.name != METADATA_FILE:
                shutil.copy2(entry, work / entry.name)

        result = _run(
            [
                engine,
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-no-shell-escape",
                # "./" so the name is unambiguously the input file even if a
                # future change lets a leading dash through resolve_in_draft.
                f"./{source.name}",
            ],
            cwd=work,
            timeout=cfg.compile_timeout_seconds,
        )

        pdf_path = work / (Path(filename).stem + ".pdf")
        log_path = work / (Path(filename).stem + ".log")
        log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        if not log:
            log = (result.stdout or "") + (result.stderr or "")

        return {
            "ok": pdf_path.exists(),
            "pdf": pdf_path.read_bytes() if pdf_path.exists() else None,
            "log": log,
        }


def markdown_to_pdf(draft_id: str, filename: str) -> bytes:
    """
    Export a Markdown draft as PDF via pandoc and the local LaTeX engine.

    Same sandboxing as a .tex compile: a temp working directory, restricted TeX
    file access, and a hard timeout. Raises RenderError with what to do when
    either tool is missing.
    """
    cfg = get_config()
    if not pandoc_available():
        raise RenderError(
            "pandoc is not installed, so Markdown cannot be exported as PDF. "
            "Install pandoc, or export the preview as HTML instead."
        )
    if not latex_available():
        raise RenderError(
            "PDF export needs a LaTeX engine as well as pandoc, and none is installed. "
            "Install a TeX distribution, or export the preview as HTML instead."
        )

    source = resolve_in_draft(draft_id, filename)
    if not source.exists():
        raise DraftError(f"No file {filename!r} in draft {draft_id!r}")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        shutil.copy2(source, work / source.name)
        output = work / (source.stem + ".pdf")
        result = _run(
            [
                "pandoc",
                f"./{source.name}",
                "-o",
                f"./{output.name}",
                f"--pdf-engine={cfg.latex_engine.replace('latexmk', 'xelatex')}",
                # pandoc's default page geometry leaves about an inch and a
                # half on every side, which wastes most of the page.
                "-V", f"geometry:margin={cfg.pdf_margin}",
                # $maths$ in the source should reach the PDF, matching what
                # the preview shows.
                "-f", "markdown+tex_math_dollars",
                # A document is data, not a script: never let it run filters or
                # pull in files of its own choosing.
                "--sandbox",
            ],
            cwd=work,
            timeout=cfg.compile_timeout_seconds,
        )
        if not output.exists():
            raise RenderError(
                "pandoc could not produce a PDF:\n"
                + ((result.stderr or result.stdout or "").strip()[:2000])
            )
        return output.read_bytes()
