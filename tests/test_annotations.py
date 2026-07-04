"""
Tests for digest/kb/annotations.py — PDF highlight/comment extraction.

Fixture PDFs are generated in-test with PyMuPDF itself (the same annotation
objects Preview/Foxit write), so no binary fixtures are committed and the
real extraction path runs end-to-end.
"""

from pathlib import Path

import pymupdf

from digest.kb.annotations import extract_annotations

SENTENCE = "The quick brown fox jumps over the lazy dog."


def _new_doc_with_text(text: str = SENTENCE) -> pymupdf.Document:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    return doc


def _save(doc: pymupdf.Document, path: Path) -> Path:
    doc.save(path)
    doc.close()
    return path


def test_highlight_text_recovered(tmp_path):
    """
    A highlight over a phrase yields kind='highlight' with the covered words.

    Input:  PDF with SENTENCE; highlight over 'quick brown fox'
    Expected output: one annotation, text == 'quick brown fox', page 1, no note
    """
    doc = _new_doc_with_text()
    page = doc[0]
    quads = page.search_for("quick brown fox", quads=True)
    page.add_highlight_annot(quads)
    pdf = _save(doc, tmp_path / "highlighted.pdf")

    annotations = extract_annotations(pdf)
    assert len(annotations) == 1
    ann = annotations[0]
    assert ann["kind"] == "highlight"
    assert ann["text"] == "quick brown fox"
    assert ann["note_text"] == ""
    assert ann["page"] == 1


def test_highlight_with_typed_note(tmp_path):
    """
    A comment typed on a highlight's popup lands in note_text alongside the
    highlighted passage.

    Input:  highlight over 'lazy dog' with content 'this part matters'
    Expected output: text contains 'lazy dog' (word-level recovery keeps
            attached punctuation), note_text == 'this part matters'
    """
    doc = _new_doc_with_text()
    page = doc[0]
    quads = page.search_for("lazy dog", quads=True)
    annot = page.add_highlight_annot(quads)
    annot.set_info(content="this part matters")
    annot.update()
    pdf = _save(doc, tmp_path / "noted_highlight.pdf")

    annotations = extract_annotations(pdf)
    assert len(annotations) == 1
    assert "lazy dog" in annotations[0]["text"]
    assert annotations[0]["note_text"] == "this part matters"


def test_underline_treated_as_highlight(tmp_path):
    """
    Underline (and other text markup) counts as a highlight — users pick
    different markup tools for the same intent.

    Input:  underline annotation over 'brown fox jumps'
    Expected output: kind == 'highlight' with the underlined words
    """
    doc = _new_doc_with_text()
    page = doc[0]
    quads = page.search_for("brown fox jumps", quads=True)
    page.add_underline_annot(quads)
    pdf = _save(doc, tmp_path / "underlined.pdf")

    annotations = extract_annotations(pdf)
    assert len(annotations) == 1
    assert annotations[0]["kind"] == "highlight"
    assert annotations[0]["text"] == "brown fox jumps"


def test_sticky_note_becomes_comment(tmp_path):
    """
    A standalone sticky note yields kind='comment' with its typed content.

    Input:  text annotation with 'Remember to follow up'
    Expected output: kind == 'comment', note_text set, text empty
    """
    doc = _new_doc_with_text()
    page = doc[0]
    page.add_text_annot((72, 150), "Remember to follow up")
    pdf = _save(doc, tmp_path / "sticky.pdf")

    annotations = extract_annotations(pdf)
    assert len(annotations) == 1
    ann = annotations[0]
    assert ann["kind"] == "comment"
    assert ann["text"] == ""
    assert ann["note_text"] == "Remember to follow up"


def test_unannotated_pdf_returns_empty(tmp_path):
    """
    A PDF without annotations produces no entries.

    Input:  plain text PDF
    Expected output: []
    """
    pdf = _save(_new_doc_with_text(), tmp_path / "plain.pdf")
    assert extract_annotations(pdf) == []


def test_ink_drawing_is_ignored(tmp_path):
    """
    Freehand drawing (Ink) carries no text and must not produce entries —
    handwritten notes are documented as unsupported.

    Input:  PDF with only an ink annotation
    Expected output: []
    """
    doc = _new_doc_with_text()
    page = doc[0]
    page.add_ink_annot([[(72, 200), (100, 210), (130, 205)]])
    pdf = _save(doc, tmp_path / "ink.pdf")

    assert extract_annotations(pdf) == []


def test_multiline_highlight_preserves_reading_order(tmp_path):
    """
    A highlight spanning two lines (one quad per line) joins the words in
    reading order.

    Input:  two-line text, highlight covering the end of line 1 and start of
            line 2
    Expected output: words in original order across the line break
    """
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "alpha beta gamma", fontsize=12)
    page.insert_text((72, 90), "delta epsilon zeta", fontsize=12)
    quads = page.search_for("gamma", quads=True) + page.search_for("delta epsilon", quads=True)
    page.add_highlight_annot(quads)
    pdf = _save(doc, tmp_path / "multiline.pdf")

    annotations = extract_annotations(pdf)
    assert len(annotations) == 1
    assert annotations[0]["text"] == "gamma delta epsilon"
