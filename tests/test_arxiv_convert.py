"""
Tests for jarvis/digest/arxiv/convert.py — parse_arxiv_url().

parse_arxiv_url() accepts any arXiv URL and returns the bare paper ID
(e.g. "2301.07041"). It handles the two common URL forms (/abs/ and /pdf/)
and preserves version suffixes (v1, v2, …). Returns None for non-arXiv URLs.

These are pure-function tests: no network calls, no filesystem access.
"""

from jarvis.digest.arxiv.convert import parse_arxiv_url


def test_parses_abs_url():
    """
    Standard arXiv abstract page URL.

    Input:  "https://arxiv.org/abs/2301.07041"
    Expected output: "2301.07041"
    """
    assert parse_arxiv_url("https://arxiv.org/abs/2301.07041") == "2301.07041"


def test_parses_pdf_url():
    """
    Direct PDF download URL (same ID, different path).

    Input:  "https://arxiv.org/pdf/2301.07041"
    Expected output: "2301.07041"
    """
    assert parse_arxiv_url("https://arxiv.org/pdf/2301.07041") == "2301.07041"


def test_preserves_version_suffix():
    """
    Version suffixes (v1, v2, …) are part of the canonical ID and must be kept.

    Input:  "https://arxiv.org/abs/2301.07041v2"
    Expected output: "2301.07041v2"
    """
    assert parse_arxiv_url("https://arxiv.org/abs/2301.07041v2") == "2301.07041v2"


def test_returns_none_for_non_arxiv_url():
    """
    URLs that do not contain an arXiv ID return None rather than raising or
    returning an empty string.

    Input:  "https://example.com/some-paper"
    Expected output: None
    """
    assert parse_arxiv_url("https://example.com/some-paper") is None
