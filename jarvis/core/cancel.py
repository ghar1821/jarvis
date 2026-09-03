"""
Cooperative cancellation for a single chat turn.

A Python thread cannot be killed from outside, and the LLM HTTP call cannot be
interrupted by a signal, so stopping a turn is cooperative: whoever wants the
turn to stop calls stop(), and the thread running the turn notices at its next
check() and unwinds.

The checks are placed so that two things are guaranteed once stop() returns:
nothing further is ever SENT to the provider, and the session's message list is
never mutated again. What makes the stop feel instant rather than merely
eventual is that both providers stream their responses — check() runs between
streamed events, and raising out of the streaming context manager closes the
HTTP response, which is what tells the cloud or Ollama server to stop
generating (neither has a cancel-this-request endpoint; disconnecting is the
signal).

Only the webapp needs this, because its turns run on a background thread. The
terminal CLI runs the turn on the main thread, where Ctrl-C already delivers
KeyboardInterrupt into the blocked call and unwinds the same way.
"""

import threading

from .errors import TurnCancelled


class CancelToken:
    """
    One turn's stop switch.

    stop() is called from another thread (the webapp's event loop); check() is
    called on the thread doing the work and raises TurnCancelled at that point.
    """

    def __init__(self) -> None:
        self._stopped = threading.Event()

    def stop(self) -> None:
        """Ask the turn to stop. Safe to call from any thread, and more than once."""
        self._stopped.set()

    @property
    def stopped(self) -> bool:
        """True once stop() has been called — for callers that want to check without raising."""
        return self._stopped.is_set()

    def check(self) -> None:
        """Raise TurnCancelled if the turn has been stopped, otherwise do nothing."""
        if self._stopped.is_set():
            raise TurnCancelled("Stopped.")
