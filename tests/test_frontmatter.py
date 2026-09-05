"""
Tests for jarvis/kb/frontmatter.py — YAML frontmatter → flat metadata.

The design goal is that jarvis knows nothing about what a "job application" or
a "manuscript" is: well-known keys get named fields worth having a UI filter
for, and every other key passes through namespaced. So the tests cover both the
mapped keys and arbitrary ones, plus the two failure modes that matter —
malformed YAML must not lose the note, and user frontmatter must never be able
to overwrite jarvis's own schema.
"""

from jarvis.kb.frontmatter import (
    format_tags,
    has_tag,
    parse_frontmatter,
    record_header,
    split_frontmatter,
)

RECORD = """\
---
type: job_application
entity: Acme Bio
status: rejected
date: 2026-05-02
tags: [bioinformatics, remote, senior]
venue: NeurIPS
salary: 145000
remote: true
---

# Senior Bioinformatician — Acme Bio

Rejected after the technical screen.
"""


def test_well_known_keys_map_to_named_fields():
    metadata, body = parse_frontmatter(RECORD)

    assert metadata["category"] == "job_application"
    assert metadata["entity"] == "Acme Bio"
    assert metadata["status"] == "rejected"
    assert metadata["event_date"] == "2026-05-02"
    # The body no longer carries the frontmatter block — it is structure, not
    # content anyone would want back from a search.
    assert body.lstrip().startswith("# Senior Bioinformatician")
    assert "type: job_application" not in body


def test_unknown_keys_pass_through_namespaced():
    """
    A record type jarvis has never heard of needs no code change — any scalar
    key becomes x_<key> and is filterable.
    """
    metadata, _ = parse_frontmatter(RECORD)

    assert metadata["x_venue"] == "NeurIPS"
    assert metadata["x_salary"] == 145000
    assert metadata["x_remote"] is True


def test_user_frontmatter_cannot_shadow_jarvis_schema():
    """
    The `x_` prefix is a security boundary, not tidiness. A .md file can arrive
    in the vault from anywhere; without namespacing, `visibility: public` in a
    private note's frontmatter would overwrite the folder-derived
    classification when the metadata dicts merged.
    """
    hostile = """\
---
visibility: public
doc_type: paper
source: https://evil.example/paper
file_path: ../../etc/passwd
---

body text
"""
    metadata, _ = parse_frontmatter(hostile)

    for reserved in ("visibility", "doc_type", "source", "file_path"):
        assert reserved not in metadata
    assert metadata["x_visibility"] == "public"
    assert metadata["x_doc_type"] == "paper"


def test_tags_become_a_delimited_string():
    """
    ChromaDB stores scalars only, so a tag list is joined. The leading and
    trailing separators are what keep a substring test exact.
    """
    metadata, _ = parse_frontmatter(RECORD)
    assert metadata["tags"] == "|bioinformatics|remote|senior|"

    assert has_tag(metadata["tags"], "remote") is True
    assert has_tag(metadata["tags"], "bioinformatics") is True
    # The delimiters are what stop a prefix match from succeeding by accident.
    assert has_tag(metadata["tags"], "remo") is False
    assert has_tag("|remote-first|", "remote") is False


def test_a_single_tag_value_is_accepted():
    metadata, _ = parse_frontmatter("---\ntags: solo\n---\nbody\n")
    assert metadata["tags"] == "|solo|"


def test_malformed_yaml_warns_but_keeps_the_note(capsys):
    """
    Losing a note because its header had a stray character would be far worse
    than losing its filters, so parsing degrades to no metadata and the body
    still comes back for indexing.
    """
    broken = "---\ntype: [unclosed\nstatus: rejected\n---\n\nthe body survives\n"
    metadata, body = parse_frontmatter(broken, file_label="notes/broken.md")

    assert metadata == {}
    assert "the body survives" in body
    warning = capsys.readouterr().out
    assert "notes/broken.md" in warning
    assert "frontmatter" in warning


def test_nested_values_are_skipped_with_a_warning(capsys):
    """Flat over nested: a dict value can't be stored, and a silent drop
    would be worse than a noisy one."""
    nested = "---\ntype: project\nowner:\n  name: Ada\n  role: lead\n---\nbody\n"
    metadata, _ = parse_frontmatter(nested, file_label="notes/n.md")

    assert metadata["category"] == "project"
    assert "x_owner" not in metadata
    assert "owner" in capsys.readouterr().out


def test_a_note_without_frontmatter_is_untouched():
    content = "# Just a note\n\nSome prose.\n"
    metadata, body = parse_frontmatter(content)
    assert metadata == {}
    assert body == content


def test_non_mapping_frontmatter_yields_no_metadata():
    """A YAML list or scalar at the top level isn't a record header."""
    metadata, body = parse_frontmatter("---\n- one\n- two\n---\nbody\n")
    assert metadata == {}
    assert "body" in body


def test_an_empty_block_is_still_frontmatter():
    """
    `---` directly followed by `---` is what Obsidian leaves behind when you
    clear a note's properties. It has to read as an empty header, not as two
    lines of body text that end up indexed as content.
    """
    metadata, body = parse_frontmatter("---\n---\n\n# Note\n\nBody.\n")
    assert metadata == {}
    assert body.strip().startswith("# Note")
    assert "---" not in body


def test_a_dashed_run_inside_a_value_is_not_the_closing_fence():
    """
    The reason the body and its newline are one optional group rather than the
    newline being optional on its own: `foo---bar` must not end the block.
    """
    metadata, body = parse_frontmatter("---\nnote: foo---bar\n---\n\n# Note\n\nBody.\n")
    assert metadata == {"x_note": "foo---bar"}
    assert body.strip().startswith("# Note")


def test_split_frontmatter_requires_the_block_at_the_very_top():
    """A `---` rule partway down a note is a horizontal rule, not a header."""
    content = "# Title\n\n---\n\ntype: not-frontmatter\n"
    raw, body = split_frontmatter(content)
    assert raw == ""
    assert body == content


def test_record_header_summarises_identity_for_embedding():
    """
    The header is embedded into every chunk, which is what makes "jobs I was
    rejected from" match a record whose body never uses those words.
    """
    metadata, _ = parse_frontmatter(RECORD)
    assert record_header(metadata) == "job_application · Acme Bio · rejected"

    # Partial records still produce something useful, and a note with no
    # record fields produces nothing rather than a string of separators.
    assert record_header({"category": "manuscript"}) == "manuscript"
    assert record_header({}) == ""


def test_format_tags_drops_blanks():
    assert format_tags(["a", "  ", "b"]) == "|a|b|"
    assert format_tags([]) == ""


def test_empty_values_are_absent_not_errors(capsys):
    """
    An Obsidian template is full of keys with nothing after them. Those are
    absent values, not unstorable ones — warning about each on every vault
    sync would be noise, and noise trains you to ignore real warnings.
    """
    template = "---\ntitle:\nstatus:\nversion:\ntype: project\n---\n\nbody\n"
    metadata, _ = parse_frontmatter(template, file_label="templates/project.md")

    assert metadata == {"category": "project"}
    assert capsys.readouterr().out == ""


def test_an_empty_tag_list_is_absent():
    metadata, _ = parse_frontmatter("---\ntags: []\ntype: note\n---\nbody\n")
    assert "tags" not in metadata
