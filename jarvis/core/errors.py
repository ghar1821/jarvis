"""Domain exceptions and retry utility."""

import functools
import random
import time
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)


class PaperDigestError(Exception):
    """Base exception for all paper_digest errors."""


class FetchError(PaperDigestError):
    """Failed to fetch from an external service (arXiv, etc.)."""


class LLMError(PaperDigestError):
    """LLM call failed or returned an unparseable response."""


class RAGError(PaperDigestError):
    """Vector database operation failed."""


class KBCorruptionError(RAGError):
    """
    Raised when ChromaDB's HNSW index references a chunk id that no longer
    exists — a corrupted index, not a transient failure. Surfaces as
    "Error finding id" from similarity_search(). The fix is always the same
    (uv run kb reindex, which re-embeds from the chunk texts already stored,
    so nothing is lost) — no retry, because retrying a persistent corruption
    just hides it (fail visibly, per CLAUDE.md).
    """


class ConversionError(PaperDigestError):
    """PDF-to-Markdown conversion produced no usable text (e.g. scanned/image-only PDF)."""


class AuthenticationError(PaperDigestError):
    """Missing or invalid credentials for an LLM provider."""


class PrivacyError(PaperDigestError):
    """
    Raised when a cloud provider attempts to access private content.

    Caught by agentic_turn() to terminate the tool loop immediately —
    no further LLM calls are made after this is raised. This is a hard
    prompt-injection defence: private notes may contain adversarial content
    that must never reach a cloud model, even as a tool result.
    """


def with_retries(
    max_attempts: int = 5,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator: retry a function up to max_attempts times with exponential backoff.

    Each failed attempt waits backoff * 2**(attempt-1) seconds, plus up to 25%
    random jitter so repeated failures don't hammer a struggling service in
    lockstep. Prints a warning on each failure. Raises the last exception if
    all attempts fail.

    Usage:
        @with_retries(max_attempts=5, backoff=2.0, exceptions=(FetchError,))
        def fetch(...): ...
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        wait = backoff * 2 ** (attempt - 1) * random.uniform(1.0, 1.25)
                        print(
                            f"  Warning: {fn.__name__} failed (attempt {attempt}/{max_attempts}): {exc}. "
                            f"Retrying in {wait:.0f}s...",
                            flush=True,
                        )
                        time.sleep(wait)
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator
