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

from jarvis.core.config import Config
from jarvis.kb.store import build_embeddings

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


@pytest.fixture(autouse=True)
def _never_read_the_real_config(tmp_path_factory, monkeypatch):
    """
    Point every test at a config file that does not exist, so the developer's
    own ~/.jarvis/config.toml (with their real vault and papers folder) can
    never leak into a test. A test that wants a config writes its own.
    """
    home = tmp_path_factory.mktemp("jarvis-home")
    monkeypatch.setattr("jarvis.core.config.CONFIG_FILE", home / "config.toml")


@pytest.fixture(autouse=True)
def _never_touch_the_real_jarvis_home():
    """
    Fail any test that changes the developer's own server token or config.

    A test that forgets to point TOKEN_FILE or CONFIG_FILE at tmp_path would
    otherwise overwrite the token a running `kb serve` checks, or the real
    config. Watching them is cheap (a stat call per test) and turns that
    mistake into a failing test instead of a surprise on someone's machine.
    The index and status file are not watched: a `kb serve` running on the
    same machine changes them during the test run on its own.
    """
    watched = [
        Path.home() / ".jarvis" / "state" / "server_token",
        Path.home() / ".jarvis" / "config.toml",
    ]

    def snapshot():
        state = {}
        for path in watched:
            try:
                state[path] = path.stat().st_mtime_ns
            except FileNotFoundError:
                state[path] = None
        return state

    before = snapshot()
    yield
    for path, was in snapshot().items():
        assert was == before[path], (
            f"a test modified {path}, which is real user data. Give the test "
            f"a tmp_path for it (see running_server in test_serve.py)."
        )
