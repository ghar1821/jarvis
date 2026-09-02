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
def _prompts_never_seed_into_the_real_home(tmp_path_factory, monkeypatch):
    """
    Building a system prompt seeds ~/.jarvis/prompts/ on first use, so a test
    that does it without isolating CONFIG_FILE writes into the developer's own
    home. Point every test at a throwaway directory; a test that wants to
    assert on the copies overrides CONFIG_FILE itself and still lands in tmp.
    """
    home = tmp_path_factory.mktemp("jarvis-home")
    monkeypatch.setattr("jarvis.core.config.CONFIG_FILE", home / "config.toml")


@pytest.fixture(autouse=True)
def _never_touch_the_real_jarvis_home():
    """
    Fail any test that writes into the developer's own ~/.jarvis.

    A `Config()` built in a test inherits home-rooted defaults for every path
    it does not override, so forgetting one silently points a test at real
    data. That happened: a fixture overrode drafts_dir and vault_path but left
    a third path at its default, and a test wrote into the developer's actual
    home directory.

    Watching the paths that get *written* by default-constructed Configs is
    cheap (a stat call per test) and turns that whole class of mistake into a
    failing test instead of a surprise on someone's machine.
    """
    watched = [
        Path.home() / ".jarvis" / "drafts",
        # Prompt copies are seeded on first use, so any test that builds a
        # system prompt without isolating CONFIG_FILE writes here.
        Path.home() / ".jarvis" / "prompts",
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
            f"a test modified {path}, which is real user data — give the test "
            f"a tmp_path for it (see the drafts fixture)"
        )
