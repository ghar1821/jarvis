# Vendored frontend assets

The webapp has no build step and makes **no outbound requests on page load** —
opening it works offline, and nothing about what you are editing reaches a CDN.
Anything the frontend needs is therefore committed here, pinned to an exact
version.

## codemirror-5.65.19

- **Source:** https://registry.npmjs.org/codemirror/-/codemirror-5.65.19.tgz
- **License:** MIT (see `codemirror-5.65.19/LICENSE`)
- **Why version 5:** CodeMirror 6 ships as ES modules that must be bundled.
  This project deliberately has no bundler, and adding one to get syntax
  highlighting would be a poor trade. CodeMirror 5 is a drop-in `<script>`,
  feature-frozen but maintained for security fixes.

Only the files actually used are vendored, not the whole package:

| Path | Why |
|---|---|
| `lib/codemirror.js`, `lib/codemirror.css` | The editor itself |
| `mode/markdown/markdown.js` | `.md` highlighting |
| `mode/xml/xml.js` | Required by the markdown mode for embedded HTML |
| `mode/stex/stex.js` | `.tex` and `.bib` highlighting |
| `addon/edit/closebrackets.js` | Auto-closing brackets and quotes |
| `addon/display/placeholder.js` | Placeholder text in an empty editor |
| `theme/ayu-dark.css` | The editor's colour scheme (see below) |

### Why ayu-dark

CodeMirror's default token colours are a **light** palette, so on a dark
background several of them are unreadable — markdown list markers are `#05a`,
which is 2.6:1 against this app's background, well under the 4.5:1 needed for
body text.

Overriding tokens by hand was the first attempt and is a treadmill: the
defaults cover 24 classes, and any one left unstyled silently falls back to a
light-theme colour. A vendored theme covers them as a designed set.

ayu-dark was chosen over the (otherwise excellent) material family for one
concrete reason: **it defines `.cm-header`**. Themes that do not leave headings
on CodeMirror's default `blue`, which reproduces exactly the bug this was
meant to fix. Its background (`#0a0e14`) also sits close to the app's own
`#111`.

To try another, copy its file into `theme/`, add a `<link>` in `index.html`,
and change the `theme:` option in `ensureEditor()`. Check first that it styles
`.cm-header` and `.cm-variable-2` — headings and list markers are what
markdown editing leans on.

## Updating

1. Download the tarball for the new version from the URL above.
2. Copy the same file list into `codemirror-<new version>/`.
3. Update the `<script>`/`<link>` paths in `jarvis/webapp/index.html`.
4. Delete the old directory and update this file.

If the mode list changes, keep `xml.js` — the markdown mode depends on it, and
dropping it fails silently with unhighlighted Markdown rather than an error.
