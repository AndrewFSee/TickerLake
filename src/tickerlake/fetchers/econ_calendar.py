"""Upcoming US data releases: what comes out when.

FRED publishes the release calendar of every series it carries, a curated subset
of which is kept here (``econ_calendar.releases``). Curated on purpose: FRED's
list runs to hundreds of releases and dates "FOMC Press Release" every single
day, so a calendar built from all of it says the Fed meets daily.

Two releases come from Kalshi instead, from the closing time of its markets on
them: FOMC meetings (FRED has no meeting calendar) and the ISM PMI (not a FRED
release). Kalshi closes a release's markets a minute before it is published.

Each run stores the calendar as known that day, so a moved release date is
visible as a change between snapshots rather than overwritten.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd

from tickerlake.fetchers.base import BaseFetcher, FetchResult
from tickerlake.fetchers.prediction_markets import KALSHI_URL
from tickerlake.storage import paths as P
from tickerlake.utils.http import HttpClient

FRED_DATES_URL = "https://api.stlouisfed.org/fred/release/dates"
NEW_YORK = "America/New_York"


def kalshi_release_dates(events: list[dict]) -> list[date]:
    """New York dates on which an event's markets close: the release date."""
    out = set()
    for e in events:
        closes = [pd.Timestamp(m["close_time"]) for m in e.get("markets", []) if m.get("close_time")]
        if closes:
            out.add(min(closes).tz_convert(NEW_YORK).date())
    return sorted(out)


class EconCalendarFetcher(BaseFetcher):
    """Snapshot of the release calendar for the next ``horizon_days``."""

    name = "econ_calendar"
    dataset = P.ECON_CALENDAR

    def collect(self, run_date: date, result: FetchResult) -> None:
        now = datetime.now(UTC)
        end = run_date + timedelta(days=int(self.cfg("horizon_days", 60)))
        releases: dict = self.cfg("releases", {}) or {}
        client = HttpClient(requests_per_second=float(self.cfg("requests_per_second", 2.0)),
                            user_agent="Mozilla/5.0 (TickerLake)")
        key = self.config.secrets.fred_api_key
        rows = []
        for rkey, spec in releases.items():
            try:
                if spec.get("dates_from") == "kalshi":
                    dates = self._kalshi_dates(client, spec.get("kalshi") or [])
                    source = "kalshi"
                elif spec.get("fred") and key:
                    dates = self._fred_dates(client, int(spec["fred"]), run_date, end, key)
                    source = "fred"
                else:
                    continue
            except Exception as exc:
                result.add_warning(f"{rkey}: {type(exc).__name__}: {exc}")
                continue
            for d in dates:
                if run_date <= d <= end:
                    rows.append({
                        "snapshot_date": run_date, "release_key": rkey, "release_date": d,
                        "name": spec.get("name", rkey), "time_et": spec.get("time"),
                        "importance": spec.get("importance"),
                        "fred_release_id": int(spec["fred"]) if spec.get("fred") else None,
                        "source": source, "ingested_at": now,
                    })
        if not rows:
            result.add_error("no upcoming releases found")
            return
        df = pd.DataFrame(rows).drop_duplicates(["release_key", "release_date"])
        write = self.writer.write(df, P.ECON_CALENDAR, self.paths.econ_calendar_file(run_date.year), mode="merge")
        result.record_write(write)
        result.items_succeeded = len(df)
        result.details["releases"] = int(df["release_key"].nunique())
        self.log.info("econ calendar: %d release dates through %s", len(df), end)

    def _fred_dates(self, client: HttpClient, release_id: int, start: date, end: date, key: str) -> list[date]:
        data = client.get_json(FRED_DATES_URL, params={
            "api_key": key, "file_type": "json", "release_id": release_id,
            "realtime_start": start.isoformat(), "realtime_end": end.isoformat(),
            "include_release_dates_with_no_data": "true", "sort_order": "asc", "limit": 1000})
        return sorted({date.fromisoformat(x["date"]) for x in data.get("release_dates", [])})

    def _kalshi_dates(self, client: HttpClient, series: list[str]) -> list[date]:
        dates: set[date] = set()
        for s in series:
            data = client.get_json(f"{KALSHI_URL}/events", params={
                "series_ticker": s, "status": "open", "with_nested_markets": "true", "limit": 200})
            dates.update(kalshi_release_dates(data.get("events", [])))
        return sorted(dates)
