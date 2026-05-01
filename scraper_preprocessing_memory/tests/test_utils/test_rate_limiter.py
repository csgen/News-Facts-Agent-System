"""Unit tests for src.utils.rate_limiter.RateLimiter.

The limiter is constructed with `rpm` (requests per minute) and computes its
internal `min_interval = 60.0 / rpm`. We use small `rpm` values that yield a
short min_interval (e.g. rpm=300 → 0.2s) so tests stay fast.
"""
import threading
import time

from src.utils.rate_limiter import RateLimiter


def test_first_call_does_not_throttle():
    """No prior call → wait() returns immediately."""
    rl = RateLimiter(rpm=60)         # min_interval = 1.0s
    start = time.monotonic()
    rl.wait()
    elapsed = time.monotonic() - start
    assert elapsed < 0.1, f"first call should not block, slept {elapsed:.3f}s"


def test_min_interval_derived_from_rpm():
    """`min_interval = 60 / rpm` — sanity-check the arithmetic."""
    assert RateLimiter(rpm=60).min_interval == 1.0
    assert RateLimiter(rpm=300).min_interval == 0.2
    assert RateLimiter(rpm=120).min_interval == 0.5


def test_second_call_within_interval_is_throttled():
    """Two back-to-back calls → second blocks for ~min_interval."""
    rl = RateLimiter(rpm=300)        # min_interval = 0.2s
    rl.wait()                         # warm up _last
    start = time.monotonic()
    rl.wait()
    elapsed = time.monotonic() - start
    # Allow a generous lower bound; upper bound caps runaway sleeps.
    assert 0.15 <= elapsed <= 0.40, f"expected ~0.2s wait, got {elapsed:.3f}s"


def test_concurrent_threads_serialized_by_lock():
    """Two threads racing to wait() at the same instant — one waits for the other."""
    rl = RateLimiter(rpm=300)        # min_interval = 0.2s
    rl.wait()                         # warm up _last so both threads see throttling
    finish_times: list[float] = []
    barrier = threading.Barrier(2)

    def call():
        barrier.wait()                # release both threads simultaneously
        rl.wait()
        finish_times.append(time.monotonic())

    t1 = threading.Thread(target=call)
    t2 = threading.Thread(target=call)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # The two finish times should be at least min_interval apart because the
    # lock + sleep inside wait() forces serialization. Slightly relaxed lower
    # bound to absorb scheduler jitter.
    diff = abs(finish_times[0] - finish_times[1])
    assert diff >= 0.15, f"second thread should wait, got diff={diff:.3f}s"
