"""
Extract embedded raster figures from a PDF for vision captioning.

Text embeddings can't see images, so a figure-heavy paper loses whatever the
figures convey once it is chunked as text. This module pulls the raster images
back out (PyMuPDF's page.get_images / doc.extract_image) so the store layer can
caption each one with a vision model and index the caption as searchable text.

What is NOT extracted: vector diagrams drawn directly on the page (line art,
shapes) — those are not embedded raster objects and carry no image xref. Tiny
raster objects below min_pixels (logos, rules, icons, math glyphs saved as
images) are skipped, since they are decoration rather than figures. Each image
is normalised to PNG bytes so both providers' describe_image() receive a
consistent media type.
"""

from pathlib import Path

import pymupdf


def extract_figures(
    pdf_path: Path,
    max_figures: int = 20,
    min_pixels: int = 40000,
) -> list[dict]:
    """
    Extract embedded raster images from a PDF as figure candidates.

    Returns up to max_figures dicts, in page order:
      {"page": int (1-indexed), "image_bytes": bytes (PNG)}

    Images whose width*height is below min_pixels are skipped (decorative
    content). The same xref is only emitted once even if a figure repeats
    across pages (e.g. a running header logo).
    """
    figures: list[dict] = []
    seen_xrefs: set[int] = set()

    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            for image in page.get_images(full=True):
                if len(figures) >= max_figures:
                    return figures
                xref = image[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                extracted = doc.extract_image(xref)
                if not extracted:
                    continue
                if extracted.get("width", 0) * extracted.get("height", 0) < min_pixels:
                    continue

                # Normalise to PNG so both providers get a consistent media type,
                # regardless of how the image was stored in the PDF.
                pixmap = pymupdf.Pixmap(doc, xref)
                if pixmap.n - pixmap.alpha >= 4:  # CMYK/other → convert to RGB first
                    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
                png_bytes = pixmap.tobytes("png")

                figures.append({"page": page.number + 1, "image_bytes": png_bytes})

    return figures
