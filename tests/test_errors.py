"""
Tests for jarvis/core/errors.py — the @with_retries decorator.

@with_retries wraps a callable and retries it up to max_attempts times when one
of the specified exception types is raised. An exponential backoff with jitter
(backoff * 2**(attempt-1) * uniform(1.0, 1.25) seconds) is applied between
retries.

All tests use backoff=0 to skip sleep delays and keep the suite fast, except
the backoff-schedule test which stubs time.sleep to capture the waits.
"""

import pytest

import jarvis.core.errors
from jarvis.core.errors import with_retries


def test_succeeds_on_first_try():
    """
    When the function raises no exception it is called exactly once and its
    return value is passed through unchanged.

    Input:  a function that always returns 42
    Expected output:
        return value == 42
        call count   == 1
    """
    calls = 0

    @with_retries(max_attempts=3, backoff=0)
    def always_works():
        nonlocal calls
        calls += 1
        return 42

    assert always_works() == 42
    assert calls == 1


def test_retries_on_matching_exception_and_eventually_succeeds():
    """
    When the function raises a matching exception on the first call but succeeds
    on the second, it is retried and the successful return value is returned.

    Input:  a function that raises ValueError on call 1, returns "ok" on call 2
    Expected output:
        return value == "ok"
        call count   == 2
    """
    calls = 0

    @with_retries(max_attempts=3, backoff=0, exceptions=(ValueError,))
    def fails_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("first attempt fails")
        return "ok"

    assert fails_once() == "ok"
    assert calls == 2


def test_raises_after_all_attempts_exhausted():
    """
    When every attempt raises a matching exception the last exception propagates
    to the caller after max_attempts tries.

    Input:  a function that always raises ValueError
    Expected output:
        raises ValueError
        call count == max_attempts (3)
    """
    calls = 0

    @with_retries(max_attempts=3, backoff=0, exceptions=(ValueError,))
    def always_fails():
        nonlocal calls
        calls += 1
        raise ValueError("always fails")

    with pytest.raises(ValueError, match="always fails"):
        always_fails()

    assert calls == 3


def test_does_not_retry_on_unspecified_exception():
    """
    When the function raises an exception type not listed in exceptions=, it
    propagates immediately without any retries.

    Input:  a function that raises TypeError; decorator only handles ValueError
    Expected output:
        raises TypeError immediately
        call count == 1  (no retries)
    """
    calls = 0

    @with_retries(max_attempts=3, backoff=0, exceptions=(ValueError,))
    def wrong_error():
        nonlocal calls
        calls += 1
        raise TypeError("not retried")

    with pytest.raises(TypeError, match="not retried"):
        wrong_error()

    assert calls == 1


def test_backoff_grows_exponentially_with_jitter(monkeypatch):
    """
    The wait between retries follows backoff * 2**(attempt-1), scaled by a
    jitter factor in [1.0, 1.25).

    Input:  a function that always fails, backoff=2.0, max_attempts=4
    Expected output:
        three sleeps, each within [base, base * 1.25) for bases 2, 4, 8
    """
    waits: list[float] = []
    monkeypatch.setattr(jarvis.core.errors.time, "sleep", waits.append)

    @with_retries(max_attempts=4, backoff=2.0, exceptions=(ValueError,))
    def always_fails():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        always_fails()

    assert len(waits) == 3
    for wait, base in zip(waits, [2.0, 4.0, 8.0]):
        assert base <= wait < base * 1.25
