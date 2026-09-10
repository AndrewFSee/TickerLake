"""Shared HTTP client with polite rate limiting and retry.

Used by every source that speaks plain REST (SEC, FRED, GDELT, Finnhub).
yfinance brings its own session, so it does not go through here.
"""

from __future__ import annotations

import logging
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from tickerlake.utils.throttle import RateLimiter, is_rate_limit_error

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30


class HttpError(RuntimeError):
    """Non-retryable HTTP failure."""


class RetryableHttpError(RuntimeError):
    """Transient HTTP failure worth retrying."""


class HttpClient:
    """requests.Session + rate limiter + exponential backoff."""

    def __init__(
        self,
        requests_per_second: float = 5.0,
        user_agent: str = "TickerLake/0.1",
        jitter_pct: float = 0.15,
        max_attempts: int = 4,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        self.limiter = RateLimiter(interval, jitter_pct)
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        """Rate-limited GET with retry on transient failures."""

        @retry(
            retry=retry_if_exception_type(RetryableHttpError),
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(multiplier=2, min=2, max=60),
            reraise=True,
        )
        def _do() -> requests.Response:
            self.limiter.wait()
            try:
                resp = self.session.get(url, timeout=self.timeout, **kwargs)
            except requests.RequestException as exc:
                raise RetryableHttpError(f"{type(exc).__name__}: {exc}") from exc

            if resp.status_code == 429 or resp.status_code >= 500:
                raise RetryableHttpError(f"HTTP {resp.status_code} from {url}")
            if resp.status_code == 404:
                raise HttpError(f"HTTP 404 from {url}")
            if resp.status_code >= 400:
                raise HttpError(f"HTTP {resp.status_code} from {url}: {resp.text[:200]}")
            return resp

        return _do()

    def get_json(self, url: str, **kwargs: Any) -> Any:
        resp = self.get(url, **kwargs)
        try:
            return resp.json()
        except ValueError as exc:
            raise HttpError(f"non-JSON response from {url}: {resp.text[:200]}") from exc

    def get_text(self, url: str, **kwargs: Any) -> str:
        resp = self.get(url, **kwargs)
        # requests falls back to ISO-8859-1 whenever a text/* response omits a
        # charset, which mangles UTF-8 (Wikipedia's "Brown-Forman" en-dash, for
        # one). Trust the document's own declared encoding instead.
        if not resp.encoding or resp.encoding.lower() in {"iso-8859-1", "ascii"}:
            resp.encoding = resp.apparent_encoding or "utf-8"
        return resp.text


__all__ = ["HttpClient", "HttpError", "RetryableHttpError", "is_rate_limit_error"]
