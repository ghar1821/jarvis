#!/usr/bin/env python3
"""
Convert a PDF (local file or arXiv URL) to Markdown.
Uses pymupdf4llm — fast, rule-based extraction with no ML model downloads.

The library API is pdf_to_markdown(), which returns the Markdown as a string
so ingestion never needs intermediate files. The standalone CLI below writes
the result to a .md file for manual use.

Usage:
    uv run convert-pdf --input <pdf_path_or_arxiv_url> [--output-dir <dir>]

Examples:
    uv run convert-pdf --input paper.pdf
    uv run convert-pdf --input https://arxiv.org/abs/2301.07041
    uv run convert-pdf --input paper.pdf --output-dir ./output
"""

import argparse
import sys
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


def main() -> None:
    from jarvis.digest.arxiv.convert import download_arxiv_pdf, parse_arxiv_url

    parser = argparse.ArgumentParser(
        description="Convert a PDF (local or arXiv) to Markdown.",
    )
    parser.add_argument("--input", required=True, help="Local PDF path or arXiv URL/ID")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: same folder as the PDF, or ./output for downloads)",
    )
    args = parser.parse_args()

    input_str = args.input

    if input_str.startswith("http://") or input_str.startswith("https://"):
        arxiv_id = parse_arxiv_url(input_str)
        if arxiv_id is None:
            print(
                f"Error: Could not parse arXiv ID from URL: {input_str}",
                file=sys.stderr,
            )
            sys.exit(1)
        download_dir = Path(args.output_dir) if args.output_dir else Path("output")
        download_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = download_arxiv_pdf(arxiv_id, download_dir)
        output_dir = download_dir
    else:
        pdf_path = Path(input_str)
        if not pdf_path.exists():
            print(f"Error: File not found: {pdf_path}", file=sys.stderr)
            sys.exit(1)
        if pdf_path.suffix.lower() != ".pdf":
            print(f"Error: Not a PDF file: {pdf_path}", file=sys.stderr)
            sys.exit(1)
        output_dir = Path(args.output_dir) if args.output_dir else pdf_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting: {pdf_path.name}")
    try:
        markdown_text = pdf_to_markdown(pdf_path)
    except ConversionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    md_path = output_dir / f"{pdf_path.stem}.md"
    md_path.write_text(markdown_text, encoding="utf-8")
    print(f"Markdown saved to: {md_path}")
    print("Done.")


if __name__ == "__main__":
    main()
