"""Domain exceptions."""


class JarvisError(Exception):
    """Base exception for all jarvis errors."""


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


class ServerAuthError(RAGError):
    """
    The knowledge-base server is running but refused this process's token.

    Kept apart from "server not running" on purpose: a one-shot `kb` command
    may open the index files itself when the server is down, but must never do
    so while the server is up. Two processes writing the same index is the
    fault the server exists to prevent.
    """


class ConfigError(JarvisError):
    """~/.jarvis/config.toml asks for something jarvis will not do safely."""


class ConversionError(JarvisError):
    """PDF-to-Markdown conversion produced no usable text (e.g. scanned/image-only PDF)."""

