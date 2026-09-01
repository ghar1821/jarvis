"""
Build hooks for the documentation site.

Two problems, both solved here so that no Markdown source file has to change
and nothing gets duplicated into docs/:

1. The site's home page should be the README, and there should still be
   exactly one README. So instead of copying it into docs/ (where it would
   quietly drift from the real one), the build generates the home page from
   the README on the fly.

2. Some docs link to files outside docs/ — a workflow under .github/, say.
   Those have no page on the site to resolve to, so they are rewritten to
   point at the file on GitHub instead of turning into dead links.
"""

import posixpath
import re
from pathlib import Path, PurePosixPath

from mkdocs.structure.files import File

REPO_ROOT = Path(__file__).parent
GITHUB_BLOB_URL = "https://github.com/ghar1821/jarvis/blob/main/"

# A relative Markdown link that walks up out of the directory it sits in, e.g.
# [tests.yml](../.github/workflows/tests.yml) in docs/TESTING.md. Some of these
# stay inside docs/ and some don't, which is what on_page_markdown works out.
UPWARD_LINK = re.compile(r"\]\((\.\./[^)]+)\)")


def on_files(files, config):
    """Add the README to the build as the site's index page."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    # From the repository root the README points at docs/DESIGN.md, but on the
    # site the index and the docs pages are siblings, so the prefix has to go.
    readme = readme.replace("](docs/", "](")

    files.append(File.generated(config, "index.md", content=readme))
    return files


def on_page_markdown(markdown, page, config, files):
    """Send links that leave the docs tree to the file on GitHub."""
    docs_root = Path(config.docs_dir).relative_to(REPO_ROOT).as_posix()
    page_dir = PurePosixPath(page.file.src_uri).parent

    def rewrite(match):
        # Follow the link the way a browser would — from the page's own
        # directory — and express where it lands from the repository root.
        target = posixpath.normpath(
            posixpath.join(docs_root, str(page_dir), match.group(1))
        )

        # A link that stays inside docs/ is an ordinary page link; MkDocs
        # resolves those itself and they must be left alone.
        if target == docs_root or target.startswith(f"{docs_root}/"):
            return match.group(0)

        return f"]({GITHUB_BLOB_URL}{target})"

    return UPWARD_LINK.sub(rewrite, markdown)
