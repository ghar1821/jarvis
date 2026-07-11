"""
Tests for jarvis/kb/convert.py — PDF-to-Markdown conversion via pymupdf4llm.

pymupdf4llm is a pure-Python extraction library (no ML models, no network),
so these tests run the real converter against tiny PDFs generated in-test
with PyMuPDF — no binary fixtures committed, no mocking.
"""

from pathlib import Path

import pymupdf
import pytest

from jarvis.core.errors import ConversionError
from jarvis.kb.convert import pdf_to_markdown


def _make_text_pdf(path: Path, text: str) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    doc.save(path)
    doc.close()
    return path


def test_pdf_to_markdown_extracts_text(tmp_path):
    """
    A PDF with a known sentence converts to markdown containing that sentence.

    Input:  one-page PDF with 'The quick brown fox jumps over the lazy dog.'
    Expected output: markdown string containing the sentence
    """
    pdf_path = _make_text_pdf(tmp_path / "sample.pdf", "The quick brown fox jumps over the lazy dog.")
    markdown = pdf_to_markdown(pdf_path)
    assert "quick brown fox" in markdown


def test_pdf_to_markdown_returns_string_no_files(tmp_path):
    """
    Conversion is purely in-memory: no .md or image artifacts appear next to
    the source PDF.

    Input:  a text PDF in an otherwise empty directory
    Expected output: str result; directory still contains only the PDF
    """
    pdf_path = _make_text_pdf(tmp_path / "sample.pdf", "Hello conversion.")
    result = pdf_to_markdown(pdf_path)
    assert isinstance(result, str)
    assert [p.name for p in tmp_path.iterdir()] == ["sample.pdf"]


def test_scanned_pdf_raises_conversion_error(tmp_path):
    """
    A PDF with no text layer (blank page, as in a scan without OCR) must fail
    visibly rather than index an empty document.

    Input:  one-page PDF with no text objects
    Expected output: ConversionError naming the file
    """
    doc = pymupdf.open()
    doc.new_page()
    pdf_path = tmp_path / "scanned.pdf"
    doc.save(pdf_path)
    doc.close()

    with pytest.raises(ConversionError, match="scanned.pdf"):
        pdf_to_markdown(pdf_path)
