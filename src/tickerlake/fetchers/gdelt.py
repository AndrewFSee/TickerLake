"""GDELT news and event data.

GDELT is free, unauthenticated, and run as a public good, so the polite ceiling
matters more than the technical one. Querying 500 company names daily would be
abusive; instead each run covers the configured macro themes plus a **rotating
slice** of the universe, sized by ``symbols_per_run``. The slice advances by day
so every symbol comes round on a predictable cycle.

Tone is GDELT's own sentiment score (roughly -100..+100, typically -10..+10),
which is the main reason to collect this at all: it is a free, dated,
symbol-linkable sentiment feature.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd

from tickerlake.fetchers.base import BaseFetcher, FetchResult
from tickerlake.storage import paths as P
from tickerlake.utils.http import HttpClient, HttpError

SOURCE = "gdelt"
DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"


class GdeltFetcher(BaseFetcher):
    """News articles and tone scores, by macro theme and by company."""

    name = "gdelt"
    dataset = P.NEWS_EVENTS

    def collect(self, run_date: date, result: FetchResult) -> None:
        client = HttpClient(
            requests_per_second=float(self.cfg("requests_per_second", 1.0)),
            user_agent="TickerLake/0.1 (research)",
            # Deliberately low: GDELT throttles hard and unpredictably, and every
            # retry is time an unattended run is not spending on anything useful.
            max_attempts=int(self.cfg("max_attempts", 2)),
        )
        lookback = int(self.cfg("lookback_days", 1))
        max_records = int(self.cfg("max_records_per_query", 250))
        timespan = f"{max(lookback, 1) * 24}h"

        # Circuit breaker. GDELT is a free public service that returns HTTP 429
        # for minutes at a time regardless of pacing. Without this, 44 queries
        # each burning their retry budget can stall the pipeline for the better
        # part of an hour for zero rows. Partial data beats a stalled run.
        self._consecutive_failures = 0
        self._breaker_limit = int(self.cfg("circuit_breaker_failures", 5))
        self._breaker_open = False

        rows: list[dict[str, Any]] = []
        try:
            for theme in self.cfg("themes", []) or []:
                if self._breaker_open:
                    break
                rows.extend(
                    self._query(
                        client,
                        query=f"theme:{theme}",
                        timespan=timespan,
                        max_records=max_records,
                        run_date=run_date,
                        theme=theme,
                        symbol=None,
                        result=result,
                    )
                )

            for symbol, name in self._company_slice(run_date):
                if self._breaker_open:
                    break
                rows.extend(
                    self._query(
                        client,
                        query=f'"{name}"',
                        timespan=timespan,
                        max_records=min(max_records, 75),
                        run_date=run_date,
                        theme=None,
                        symbol=symbol,
                        result=result,
                    )
                )
        finally:
            client.close()

        if self._breaker_open:
            result.add_warning(
                f"GDELT circuit breaker tripped after {self._breaker_limit} consecutive "
                "failures; remaining queries skipped for this run"
            )

        if not rows:
            result.add_warning("GDELT returned no articles")
            return

        write = self.writer.write(
            pd.DataFrame(rows),
            P.NEWS_EVENTS,
            self.paths.news_file(run_date, SOURCE),
            mode="merge",
        )
        result.record_write(write)
        result.details["articles"] = len(rows)

    # ------------------------------------------------------------- querying

    def _query(
        self,
        client: HttpClient,
        query: str,
        timespan: str,
        max_records: int,
        run_date: date,
        theme: str | None,
        symbol: str | None,
        result: FetchResult,
    ) -> list[dict[str, Any]]:
        try:
            payload = client.get_json(
                DOC_API,
                params={
                    "query": query,
                    "mode": "artlist",
                    "format": "json",
                    "maxrecords": max_records,
                    "timespan": timespan,
                    "sort": "datedesc",
                },
            )
        except Exception as exc:
            result.items_failed += 1
            self._record_failure(result)
            detail = exc if isinstance(exc, HttpError) else f"{type(exc).__name__}: {exc}"
            result.add_warning(f"gdelt query {query!r}: {detail}")
            return []

        articles = payload.get("articles") or []
        result.items_succeeded += 1
        self._consecutive_failures = 0
        now = datetime.now(UTC)

        rows = []
        for art in articles:
            url = art.get("url") or ""
            if not url:
                continue
            published = _parse_gdelt_date(art.get("seendate"))
            rows.append(
                {
                    # GDELT has no stable article id, so hash the identity we do
                    # have; this is what makes re-runs idempotent.
                    "event_id": hashlib.sha1(f"{url}|{symbol or theme or ''}".encode()).hexdigest(),
                    "published_at": published,
                    "date": (published.date() if published else run_date),
                    "symbol": symbol,
                    "title": art.get("title"),
                    "summary": None,
                    "url": url,
                    "domain": art.get("domain"),
                    "language": art.get("language"),
                    "country": art.get("sourcecountry"),
                    "theme": theme,
                    "tone": _safe_float(art.get("tone")),
                    "category": "theme" if theme else "company",
                    "embedding_status": "pending",
                    "source": SOURCE,
                    "ingested_at": now,
                }
            )
        return rows

    def _record_failure(self, result: FetchResult) -> None:
        """Trip the breaker once failures stop looking like bad luck."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._breaker_limit and not self._breaker_open:
            self._breaker_open = True
            self.log.warning(
                "GDELT circuit breaker OPEN after %d consecutive failures; "
                "skipping remaining queries this run",
                self._consecutive_failures,
            )

    # ------------------------------------------------------- rotating slice

    def _company_slice(self, run_date: date) -> list[tuple[str, str]]:
        """The day's slice of (symbol, company_name) to query.

        Rotates deterministically by day so coverage is even over time and a
        re-run on the same date asks for exactly the same companies.
        """
        per_run = int(self.cfg("symbols_per_run", 40))
        if per_run <= 0 or self.tracker is None:
            return []

        df = self.tracker.load()
        if df.empty:
            return []
        current = df[df["end_date"].isna() & df["company_name"].notna()]
        pairs = sorted(
            {(r.symbol, str(r.company_name)) for r in current.itertuples()},
            key=lambda p: p[0],
        )
        if not pairs:
            return []

        offset = (run_date.toordinal() * per_run) % len(pairs)
        rotated = pairs[offset:] + pairs[:offset]
        return rotated[:per_run]


def _parse_gdelt_date(value: Any) -> datetime | None:
    """GDELT stamps are 'YYYYMMDDTHHMMSSZ'."""
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except ValueError:
        try:
            return pd.to_datetime(value, utc=True).to_pydatetime()
        except Exception:
            return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
