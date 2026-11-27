"""
Shortcuts for putting test content straight into a store.

Production never indexes loose text: jarvis/kb/mirror.py builds documents
from files and commits them in one write. Tests often just want "this text is
in the index", so these wrap the same two steps (build, then commit) without
needing a file on disk for every case.
"""

from pathlib import Path

from jarvis.kb.store import build_annotation_documents, build_text_documents, commit_documents


def index_text(
    content: str,
    doc_type: str,
    source: str,
    extra_metadata: dict | None = None,
    store=None,
    embed_header: str = "",
) -> list[str]:
    """Chunk content and add it to the store. Returns the new chunk ids."""
    documents = build_text_documents(
        content, doc_type, source, extra_metadata=extra_metadata, embed_header=embed_header,
    )
    return commit_documents(documents, store) if documents else []


def index_annotations(
    pdf_path: Path,
    source: str,
    title: str = "",
    file_path: str = "",
    store=None,
) -> list[str]:
    """Add a PDF's highlights and typed notes to the store. Returns the new chunk ids."""
    documents = build_annotation_documents(pdf_path, source, title=title, file_path=file_path)
    return commit_documents(documents, store) if documents else []
