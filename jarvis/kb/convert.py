"""
PDF to Markdown via pymupdf4llm — fast, rule-based extraction with no ML
model downloads. Returns the Markdown as a string, so indexing never needs
an intermediate file.
"""

from pathlib import Path

from jarvis.core.errors import ConversionError


def pdf_to_markdown(pdf_path: Path) -> str:
    """
    Convert a PDF to Markdown text and return it as a string.

    Raises ConversionError when the PDF yields no extractable text — typically
    a scanned/image-only PDF with no OCR text layer, which pymupdf4llm cannot
    read. Failing loudly beats silently indexing an empty document.
    """
    import pymupdf4llm

    markdown_text = pymupdf4llm.to_markdown(str(pdf_path))
    if not markdown_text.strip():
        raise ConversionError(
            f"No extractable text in {pdf_path.name} — likely a scanned/image-only "
            "PDF. pymupdf4llm has no OCR fallback."
        )
    return markdown_text
