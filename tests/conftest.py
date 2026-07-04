"""
Shared fixtures for the jarvis test suite.

Test ChromaDB store
-------------------
KB tests use a real ChromaDB instance persisted at tests/.chroma/ (gitignored).
A fresh collection with a unique uuid-based name is created per test and deleted
at teardown, so tests are fully isolated from each other without rebuilding the
store from scratch on every run.

HuggingFace embedding model
----------------------------
The real embedding model named in the default config is used — no mock
embeddings. It is built through the same build_embeddings() helper production
uses, so the query prefix and normalisation match real retrieval exactly. The
model downloads once to ~/.cache/huggingface/ on first run and is reused from
cache afterwards. The fixture is session-scoped so the model loads once per
pytest session regardless of how many tests use it.

This follows the project preference for real dependencies over mocks when the
one-off setup cost is modest and the gain is genuine fidelity to production
behaviour.
"""

import uuid
from pathlib import Path

import pytest
from langchain_chroma import Chroma

from digest.config import Config
from digest.kb.store import build_embeddings

# Persistent directory for the test ChromaDB store. Gitignored — never committed.
TEST_CHROMA_DIR = Path(__file__).parent / ".chroma"


@pytest.fixture(scope="session")
def embeddings():
    """
    Real HuggingFace embedding model, loaded once for the entire test session.

    Uses the default config's embed_model and query_prefix via build_embeddings,
    so tests exercise the same embedding behaviour as production. First run
    downloads the model to ~/.cache/huggingface/; later runs load from cache.
    """
    defaults = Config()
    return build_embeddings(defaults.embed_model, defaults.query_prefix)


@pytest.fixture
def store(embeddings):
    """
    Isolated ChromaDB collection for one test.

    Each test gets a collection named test_<uuid8> inside the shared store
    directory. The collection is deleted at teardown so tests cannot affect
    each other, and the store directory itself persists between runs.
    """
    TEST_CHROMA_DIR.mkdir(exist_ok=True)
    collection_name = f"test_{uuid.uuid4().hex[:8]}"
    s = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(TEST_CHROMA_DIR),
    )
    yield s
    s.delete_collection()
