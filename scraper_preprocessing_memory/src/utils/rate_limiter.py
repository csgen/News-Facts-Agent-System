"""Thread-safe rate limiter for Google AI Studio free-tier calls.

EMBED_LIMITER throttles calls to `gemini-embedding-001` so bursty call patterns
don't trip 429/RESOURCE_EXHAUSTED.
"""

import threading
import time


class RateLimiter:
    def __init__(self, rpm: int):
        self.min_interval = 60.0 / rpm
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delta = now - self._last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self._last = time.monotonic()


# Safety margin: 95 instead of 100 RPM to absorb clock skew and retry jitter.
EMBED_LIMITER = RateLimiter(rpm=95)
