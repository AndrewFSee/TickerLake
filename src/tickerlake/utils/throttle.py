"""Rate limiting and adaptive backoff for unofficial APIs.

Yahoo publishes no rate limit. It infers abuse from request cadence and IP, and
responds by silently returning empty frames or raising ``YFRateLimitError``. Two
behaviours matter for a 8,000-request options run:

1. **Jitter.** A perfectly metronomic request every 600ms is a much stronger bot
   signal than the same average rate with noise on it.
2. **Adaptive slowdown.** Once you have been flagged, continuing at the same rate
   extends the block. ``AdaptiveThrottle`` multiplies its delay on a rate-limit
   signal and decays back toward baseline only after sustained success, so a run
   that hits a wall mid-way finishes slower rather than failing.
"""

from __future__ import annotations

import logging
import random
import threading
import time

log = logging.getLogger(__name__)

# Substrings that identify a rate-limit rejection across the libraries we use.
_RATE_LIMIT_MARKERS = (
    "too many requests",
    "429",
    "rate limit",
    "ratelimit",
    "yfratelimit",
    "temporarily blocked",
    "unusual traffic",
)


def is_rate_limit_error(exc: BaseException) -> bool:
    """Detect a rate-limit rejection without depending on a specific exception class.

    yfinance has moved this exception around between versions, so matching on the
    message is more durable than importing ``YFRateLimitError``.
    """
    if type(exc).__name__ in {"YFRateLimitError", "TooManyRequests"}:
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    if any(marker in text for marker in _RATE_LIMIT_MARKERS):
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in (429, 503)


class RateLimiter:
    """Fixed minimum interval between calls, with proportional jitter.

    Thread-safe, so it can be shared if collection is ever parallelised.
    """

    def __init__(self, min_interval: float, jitter_pct: float = 0.0) -> None:
        self.min_interval = max(0.0, float(min_interval))
        self.jitter_pct = max(0.0, float(jitter_pct))
        self._last: float = 0.0
        self._lock = threading.Lock()

    def _next_interval(self) -> float:
        if self.jitter_pct <= 0:
            return self.min_interval
        factor = 1.0 + random.uniform(-self.jitter_pct, self.jitter_pct)
        return max(0.0, self.min_interval * factor)

    def wait(self) -> float:
        """Block until the next call is allowed. Returns seconds actually slept."""
        with self._lock:
            interval = self._next_interval()
            elapsed = time.monotonic() - self._last
            sleep_for = interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                sleep_for = 0.0
            self._last = time.monotonic()
            return sleep_for


class AdaptiveThrottle:
    """A RateLimiter that reacts to rate-limit signals.

    On a rate-limit hit: sleep a penalty period, then multiply the ongoing delay.
    On sustained success: decay the multiplier back toward 1.0.
    """

    def __init__(
        self,
        base_interval: float,
        jitter_pct: float = 0.3,
        slowdown_factor: float = 2.0,
        backoff_seconds: float = 90.0,
        max_backoff_seconds: float = 900.0,
        decay_after_successes: int = 25,
        max_multiplier: float = 16.0,
    ) -> None:
        self.base_interval = max(0.0, float(base_interval))
        self.jitter_pct = jitter_pct
        self.slowdown_factor = max(1.0, float(slowdown_factor))
        self.backoff_seconds = float(backoff_seconds)
        self.max_backoff_seconds = float(max_backoff_seconds)
        self.decay_after_successes = max(1, int(decay_after_successes))
        self.max_multiplier = float(max_multiplier)

        self.multiplier = 1.0
        self.rate_limit_hits = 0
        self.total_backoff_seconds = 0.0
        self._consecutive_successes = 0
        self._limiter = RateLimiter(self.base_interval, jitter_pct)

    @property
    def current_interval(self) -> float:
        return self.base_interval * self.multiplier

    def wait(self) -> float:
        self._limiter.min_interval = self.current_interval
        return self._limiter.wait()

    def record_success(self) -> None:
        self._consecutive_successes += 1
        if self.multiplier > 1.0 and self._consecutive_successes >= self.decay_after_successes:
            previous = self.multiplier
            # Decay halfway back to baseline, never below it.
            self.multiplier = max(1.0, self.multiplier / 1.5)
            self._consecutive_successes = 0
            log.info(
                "throttle recovering: interval %.2fs -> %.2fs after %d clean calls",
                self.base_interval * previous,
                self.current_interval,
                self.decay_after_successes,
            )

    def record_rate_limit(self) -> float:
        """Apply a penalty sleep and slow down. Returns seconds slept."""
        self.rate_limit_hits += 1
        self._consecutive_successes = 0

        penalty = min(
            self.backoff_seconds * (2 ** (self.rate_limit_hits - 1)),
            self.max_backoff_seconds,
        )
        self.multiplier = min(self.multiplier * self.slowdown_factor, self.max_multiplier)

        log.warning(
            "rate limited (hit #%d): sleeping %.0fs, request interval now %.2fs",
            self.rate_limit_hits,
            penalty,
            self.current_interval,
        )
        time.sleep(penalty)
        self.total_backoff_seconds += penalty
        return penalty

    def stats(self) -> dict[str, float | int]:
        return {
            "rate_limit_hits": self.rate_limit_hits,
            "total_backoff_seconds": round(self.total_backoff_seconds, 1),
            "final_multiplier": round(self.multiplier, 2),
            "final_interval_seconds": round(self.current_interval, 3),
        }
