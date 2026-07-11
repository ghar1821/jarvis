"""arXiv URL parsing and PDF download helpers.

PDF-to-Markdown conversion lives in digest/kb/convert.py (pdf_to_markdown).
"""

import re
from pathlib import Path

import requests


def parse_arxiv_url(url: str) -> str | None:
    """Extract arXiv ID from various arXiv URL formats."""
    patterns = [
        r"arxiv\.org/abs/([0-9]+\.[0-9]+(?:v[0-9]+)?)",
        r"arxiv\.org/pdf/([0-9]+\.[0-9]+(?:v[0-9]+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def download_arxiv_pdf(arxiv_id: str, dest_dir: Path) -> Path:
    """Download a PDF from arXiv by its ID."""
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    dest_path = dest_dir / f"{arxiv_id.replace('/', '_')}.pdf"

    print(f"Downloading arXiv:{arxiv_id} ...")
    response = requests.get(
        pdf_url, stream=True, timeout=60, headers={"User-Agent": "pdf-to-md/1.0"}
    )
    response.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Saved PDF to: {dest_path}")
    return dest_path
