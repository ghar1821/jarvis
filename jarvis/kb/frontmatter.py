"""
YAML frontmatter → flat knowledge-base metadata.

A vault note can be a job application with an outcome, a manuscript with a
venue and a deadline, a meeting record — anything the user decides. Jarvis does
not know what any of those *are*; it reads whatever frontmatter a note carries
and makes it filterable.

Two kinds of key:

- **Well-known keys** (`type`/`category`, `status`, `entity`, `date`, `tags`)
  map onto named metadata fields the CLI and UI expose as filters, because
  they are the axes worth having a UI for.
- **Every other scalar key passes through** under an `x_` prefix, so
  `venue: NeurIPS` becomes `x_venue` and needs no code change.

**The `x_` prefix is a security boundary, not tidiness.** A `.md` file can
arrive in the vault from anywhere — downloaded, synced, shared. Without
namespacing, a note carrying `visibility: public` or `doc_type: paper` in its
frontmatter would overwrite jarvis's own schema when the two dicts merged, and
a private note could reclassify itself into cloud-visible. Prefixing makes that
structurally impossible rather than something to remember to check.

ChromaDB metadata values must be scalars, so lists are joined and nested
objects are skipped with a warning (flat over nested, and a silent drop would
be worse than a noisy one).
"""

import re

# Frontmatter is a `---` fenced block at the very top of the file. The body and
# its trailing newline are one optional unit, so a genuinely empty block
# (`---` directly followed by `---`, which is what Obsidian leaves behind when
# you clear a note's properties) is still recognised as frontmatter rather than
# being indexed as two lines of content. Keeping the newline inside that group
# — rather than making it optional on its own — is what stops a `---` embedded
# mid-line from being mistaken for the closing fence.
_FRONTMATTER = re.compile(r"\A---\r?\n(?:(.*?)\r?\n)?---\r?\n?", re.DOTALL)

# Frontmatter key → the metadata field it populates. These are the axes the
# CLI and webapp offer as filters; everything else becomes x_<key>.
_WELL_KNOWN = {
    "type": "category",
    "category": "category",
    "status": "status",
    "entity": "entity",
    "org": "entity",
    "company": "entity",
    "date": "event_date",
    "applied": "event_date",
}

# Bumped when this mapping changes meaning, so refresh_vault knows to re-index
# notes that were indexed under an older interpretation. Stamped on EVERY note
# by index_vault_file — including notes with no frontmatter at all, which would
# otherwise look perpetually out of date and re-index on every single sweep.
META_SCHEMA = 1

# Tags are stored as "|a|b|c|" so a substring match can test membership —
# ChromaDB has no list-contains or substring operator to filter on.
TAG_SEPARATOR = "|"


def split_frontmatter(content: str) -> tuple[str, str]:
    """Split a note into (frontmatter_text, body). No frontmatter → ("", content)."""
    match = _FRONTMATTER.match(content)
    if not match:
        return "", content
    # group(1) is None for an empty block, where the group never participated.
    return match.group(1) or "", content[match.end():]


def format_tags(tags) -> str:
    """
    Render a tag list as the delimited string metadata form: "|a|b|c|".

    The leading and trailing separators are what make a substring test exact —
    searching for "|remote|" cannot accidentally match "|remote-first|".
    """
    values = [str(tag).strip() for tag in tags if str(tag).strip()]
    if not values:
        return ""
    return TAG_SEPARATOR + TAG_SEPARATOR.join(values) + TAG_SEPARATOR


def has_tag(stored: str, tag: str) -> bool:
    """Whether a stored tag string contains one tag."""
    if not stored or not tag:
        return False
    return f"{TAG_SEPARATOR}{tag.strip()}{TAG_SEPARATOR}" in stored


# What _scalar returns when a value cannot be stored, as distinct from a value
# that is simply absent. An Obsidian template full of `title:` with nothing
# after it is the normal case, not something to warn about on every sync.
UNSTORABLE = object()


def _scalar(value):
    """
    Coerce one frontmatter value to something ChromaDB will store.

    Returns None for an empty value (the key is absent — nothing to store and
    nothing to say about it), UNSTORABLE for a nested structure the caller
    should warn about, and the value otherwise. Lists of scalars are joined.
    """
    if isinstance(value, (str, int, float, bool)):
        return value
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if any(isinstance(item, (dict, list, tuple)) for item in value):
            return UNSTORABLE
        return format_tags(value) or None
    if hasattr(value, "isoformat"):  # a date/datetime YAML parsed for us
        return value.isoformat()
    return UNSTORABLE


def parse_frontmatter(content: str, file_label: str = "") -> tuple[dict, str]:
    """
    Read a note's frontmatter into flat metadata. Returns (metadata, body).

    Best-effort by design: malformed YAML warns and yields no metadata, but the
    note is still indexed from its full text. Losing a note because its header
    had a stray colon would be far worse than losing its filters.
    """
    raw, body = split_frontmatter(content)
    if not raw.strip():
        return {}, body

    import yaml

    try:
        # safe_load only: frontmatter is untrusted input, and full load can
        # construct arbitrary Python objects.
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        print(
            f"  ⚠️  {file_label or 'note'}: could not parse frontmatter ({exc.__class__.__name__})"
            " — indexing its text without metadata",
            flush=True,
        )
        return {}, body

    if not isinstance(parsed, dict):
        return {}, body

    metadata: dict = {}
    for key, value in parsed.items():
        name = str(key).strip()
        if not name:
            continue

        if name == "tags":
            tags = value if isinstance(value, (list, tuple)) else [value]
            if tag_string := format_tags(tags):
                metadata["tags"] = tag_string
            continue

        scalar = _scalar(value)
        if scalar is UNSTORABLE:
            print(
                f"  ⚠️  {file_label or 'note'}: skipping frontmatter key {name!r} "
                "— only scalars and flat lists can be stored",
                flush=True,
            )
            continue
        if scalar is None:
            continue  # an empty value: the key is simply absent

        # Well-known keys take their dedicated field; everything else is
        # namespaced so user frontmatter can never shadow jarvis's own schema.
        field = _WELL_KNOWN.get(name.lower(), f"x_{name.lower()}")
        metadata.setdefault(field, scalar)

    return metadata, body


def record_header(metadata: dict) -> str:
    """
    A one-line summary of a record's identity, e.g.
    "job_application · Acme Bio · rejected".

    Passed to add_texts as the embed_header so EVERY chunk of the record
    carries it in its embedded text — that is what makes "jobs I was rejected
    from" match a record whose body never uses those words. Reuses the
    mechanism papers already use for title/authors rather than inventing one.
    """
    parts = [
        str(metadata.get(field, "")).strip()
        for field in ("category", "entity", "status")
    ]
    return " · ".join(part for part in parts if part)
