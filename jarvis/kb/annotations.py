"""
Extract highlights and typed notes from a PDF's native annotations.

macOS Preview and Foxit Reader (desktop and Android) both write standard
ISO 32000 annotation objects into the page /Annots array when the file is
saved: text markup (Highlight/Underline/Squiggly/StrikeOut) with /QuadPoints
marking the affected glyphs, and Text ("sticky note") / FreeText annotations
whose typed body lives in /Contents. Because this is the PDF spec's own
interchange format rather than a per-app extension, one generic reader
(PyMuPDF's page.annots()) covers both apps — and any colour of highlight,
since extraction keys on annotation type, never colour.

What is NOT extracted: freehand drawing (Ink annotations — Preview's Sketch/
Draw tools, stylus scribbles in Foxit). Those store stroke geometry, not
text; recovering them would need handwriting OCR. Notes must be typed to be
searchable.
"""

from pathlib import Path

import pymupdf

# Text-markup types all treated as "highlight" — users reach for underline or
# squiggly instead of highlight depending on the tool at hand, and all four
# mean "this passage matters".
_MARKUP_TYPES = [
    pymupdf.PDF_ANNOT_HIGHLIGHT,
    pymupdf.PDF_ANNOT_UNDERLINE,
    pymupdf.PDF_ANNOT_SQUIGGLY,
    pymupdf.PDF_ANNOT_STRIKE_OUT,
]

# Standalone typed notes: sticky notes and free-text boxes.
_COMMENT_TYPES = [
    pymupdf.PDF_ANNOT_TEXT,
    pymupdf.PDF_ANNOT_FREE_TEXT,
]


def _quads_from_vertices(vertices) -> list[pymupdf.Rect]:
    """Group a markup annotation's vertices (4 points per line) into rects."""
    rects = []
    for i in range(0, len(vertices) - 3, 4):
        quad = pymupdf.Quad(vertices[i : i + 4])
        rects.append(quad.rect)
    return rects


def _text_under_quads(page: pymupdf.Page, rects: list[pymupdf.Rect]) -> str:
    """
    Recover the text a markup annotation covers.

    Keeps every word whose bounding-box centre falls inside one of the
    annotation's line rects, then joins them in reading order — this handles
    multi-line highlights, where each line is a separate quad.
    """
    words = page.get_text("words")  # (x0, y0, x1, y1, word, block, line, word_no)
    covered = []
    for x0, y0, x1, y1, word, block, line, word_no in words:
        centre = pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)
        if any(rect.contains(centre) for rect in rects):
            covered.append((block, line, word_no, word))
    covered.sort()
    return " ".join(w for _, _, _, w in covered)


def extract_annotations(pdf_path: Path) -> list[dict]:
    """
    Extract highlights and typed comments from a PDF.

    Returns one dict per annotation:
      {"kind": "highlight" | "comment", "text": str, "note_text": str, "page": int}

    kind="highlight": text is the highlighted/underlined passage; note_text is
      any comment typed onto the annotation's popup ("" if none — Preview
      highlights often have no /Contents at all).
    kind="comment": a standalone sticky note or text box; text is "" and
      note_text holds the typed content.

    page is 1-indexed. Returns [] for PDFs without relevant annotations.
    A highlight over pure imagery with no typed note yields no text and is
    dropped; one with a typed note is kept for the note alone.
    """
    annotations = []
    with pymupdf.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            for annot in page.annots(types=_MARKUP_TYPES + _COMMENT_TYPES):
                note_text = (annot.info.get("content") or "").strip()
                if annot.type[0] in _MARKUP_TYPES:
                    rects = _quads_from_vertices(annot.vertices or [])
                    text = _text_under_quads(page, rects) if rects else ""
                    if not text and not note_text:
                        continue
                    annotations.append(
                        {
                            "kind": "highlight",
                            "text": text,
                            "note_text": note_text,
                            "page": page_index + 1,
                        }
                    )
                else:
                    if not note_text:
                        continue
                    annotations.append(
                        {
                            "kind": "comment",
                            "text": "",
                            "note_text": note_text,
                            "page": page_index + 1,
                        }
                    )
    return annotations
