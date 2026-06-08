"""Domain exceptions and retry utility."""

import functools
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


class AuthenticationError(PaperDigestError):
    """Missing or invalid credentials for an LLM provider."""


def with_retries(
    max_attempts: int = 5,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator: retry a function up to max_attempts times with linear backoff.

    Each failed attempt waits backoff * attempt seconds before the next try.
    Prints a warning on each failure. Raises the last exception if all attempts fail.

    Usage:
        @with_retries(max_attempts=5, backoff=2.0, exceptions=(requests.RequestException,))
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
                        wait = backoff * attempt
                        print(
                            f"  Warning: {fn.__name__} failed (attempt {attempt}/{max_attempts}): {exc}. "
                            f"Retrying in {wait:.0f}s...",
                            flush=True,
                        )
                        time.sleep(wait)
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator
