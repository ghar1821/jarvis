"""
One place that gives every module somewhere to report a failure to.

Library modules — the vector store, the mirror sync, metadata lookup — used
to swallow their exceptions because there was nowhere to send them. A caught
error with no handler is indistinguishable from no error at all, and the whole
reason `~/.jarvis/logs/` exists is to tell those two apart after the fact.

Handlers attach to the `jarvis` logger, so every `jarvis.*` child inherits
one.
"""

import logging
from pathlib import Path

LOG_FILE = Path.home() / ".jarvis" / "logs" / "jarvis.log"

_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"


def get_logger(name: str) -> logging.Logger:
    """
    A logger that writes to `~/.jarvis/logs/jarvis.log`.

    `name` should be the module's dotted path (`jarvis.kb.store`), so a line
    in the log says which part of jarvis produced it.

    Attaching the handler is idempotent and lazy: nothing is created until
    something actually logs, so importing jarvis never makes a directory.
    """
    logger = logging.getLogger(name)
    root = logging.getLogger("jarvis")
    if not root.handlers:
        try:
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(LOG_FILE)
            handler.setFormatter(logging.Formatter(_FORMAT))
            root.addHandler(handler)
            root.setLevel(logging.INFO)
        except OSError:
            # An unwritable log directory must not stop jarvis working. Fall
            # back to a null handler so logging calls stay harmless.
            root.addHandler(logging.NullHandler())
    return logger
