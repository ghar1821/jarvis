"""Domain exceptions and retry utility."""

import functools
import random
import time
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)


class JarvisError(Exception):
    """Base exception for all jarvis errors."""


class FetchError(JarvisError):
    """Failed to fetch from an external service (arXiv, bioRxiv, Crossref)."""


class RAGError(JarvisError):
    """Vector database operation failed."""


class KBCorruptionError(RAGError):
    """
    Raised when the knowledge-base index cannot resolve a chunk id it holds a
    reference to ("Error finding id"), or still names a collection that was
    replaced by `kb rebuild`.

    Historically this was diagnosed as on-disk corruption. That was wrong: the
    usual cause was a process holding a cached copy of the vector index while
    another process rewrote it, which a restart cleared and no rebuild could
    fix. The knowledge-base server owns the index now, so a reader cannot hold
    a stale copy — which is what makes the remaining cases worth reporting.

    Not retried: both describe persistent state that a retry would only hide
    (fail visibly, per CLAUDE.md). `kb doctor` says which case it is.
    """


class ConversionError(JarvisError):
    """PDF-to-Markdown conversion produced no usable text (e.g. scanned/image-only PDF)."""


def with_retries(
    max_attempts: int = 5,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator: retry a function up to max_attempts times with exponential backoff.

    Each failed attempt waits backoff * 2**(attempt-1) seconds, plus up to 25%
    random jitter so repeated failures don't hammer a struggling service in
    lockstep. Logs a warning on each failure. Raises the last exception if
    all attempts fail.

    Usage:
        @with_retries(max_attempts=5, backoff=2.0, exceptions=(FetchError,))
        def fetch(...): ...
    """
    from jarvis.core.logs import get_logger

    log = get_logger("jarvis.retry")

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
                        # Logged, not printed: fetchers now run inside the MCP
                        # server, where stdout is the protocol channel.
                        log.warning(
                            "%s failed (attempt %d/%d): %s. Retrying in %.0fs...",
                            fn.__name__, attempt, max_attempts, exc, wait,
                        )
                        time.sleep(wait)
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator
