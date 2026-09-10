"""Earnings surprises, forward calendar, and analyst recommendations (Finnhub).

All three sit on Finnhub's free tier, which is what makes an Alpha Vantage key
unnecessary: AV's free tier is 25 requests/day, which cannot cover a 500-symbol
universe under any rotation scheme.

Two of these fill genuine gaps in the lake:

* **Earnings surprise** (actual vs. consensus) is among the strongest
  short-horizon cross-sectional features available for free, and nothing else
  here provides it.
* **The forward calendar** is one of the few genuinely *forward-looking* fields
  in the whole dataset. Knowing a symbol reports in three days -- and whether it
  reports before the open or after the close -- is actionable in a way that
  everything else here, which is backward-looking by construction, is not.

The calendar costs one request for the entire market. Surprises and
recommendations are per-symbol, so they rotate a deterministic daily slice.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd

from tickerlake.fetchers.base import BaseFetcher, FetchResult
from tickerlake.storage import paths as P
from tickerlake.universe.sources import to_yahoo_symbol
from tickerlake.utils.http import HttpClient

SOURCE = "finnhub"
BASE = "https://finnhub.io/api/v1"


class EarningsFetcher(BaseFetcher):
    """Estimates, actuals, surprises, and analyst coverage."""

    name = "earnings"
    dataset = P.EARNINGS
    requires_secret = "finnhub_api_key"

    def collect(self, run_date: date, result: FetchResult) -> None:
        token = self.config.secrets.finnhub_api_key
        client = HttpClient(
            requests_per_second=float(self.cfg("requests_per_second", 1.0)),
            user_agent="TickerLake/0.1",
            max_attempts=3,
        )
        try:
            if self.cfg("fetch_calendar", True):
                self._calendar(client, token, run_date, result)
            if self.cfg("fetch_surprises", True):
                self._surprises(client, token, run_date, result)
            if self.cfg("fetch_recommendations", True):
                self._recommendations(client, token, run_date, result)
        finally:
            client.close()

    # -------------------------------------------------------------- calendar

    def _calendar(
        self, client: HttpClient, token: str, run_date: date, result: FetchResult
    ) -> None:
        """Forward earnings calendar. One request covers the whole market."""
        ahead = int(self.cfg("calendar_days_ahead", 45))
        tracked = set(self._tracked())
        try:
            payload = client.get_json(
                f"{BASE}/calendar/earnings",
                params={
                    "from": run_date.isoformat(),
                    "to": (run_date + timedelta(days=ahead)).isoformat(),
                    "token": token,
                },
            )
        except Exception as exc:
            result.add_warning(f"earnings calendar: {type(exc).__name__}: {exc}")
            return

        now = datetime.now(UTC)
        rows = []
        for entry in (payload or {}).get("earningsCalendar", []) or []:
            symbol = str(entry.get("symbol", "")).strip().upper().replace("-", ".")
            if tracked and symbol not in tracked:
                continue
            period = _to_date(entry.get("date"))
            if period is None:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "record_type": "calendar",
                    "period": period,
                    "fiscal_year": _int(entry.get("year")),
                    "fiscal_quarter": _int(entry.get("quarter")),
                    "eps_estimate": _float(entry.get("epsEstimate")),
                    "eps_actual": _float(entry.get("epsActual")),
                    "revenue_estimate": _float(entry.get("revenueEstimate")),
                    "revenue_actual": _float(entry.get("revenueActual")),
                    # bmo = before market open, amc = after market close. The
                    # distinction decides which session absorbs the surprise.
                    "report_hour": entry.get("hour") or None,
                    "source": SOURCE,
                    "ingested_at": now,
                }
            )

        if rows:
            write = self.writer.write(
                pd.DataFrame(rows), P.EARNINGS, self.paths.earnings_file("calendar"), mode="merge"
            )
            result.record_write(write)
            result.items_succeeded += 1
            result.details["calendar_entries"] = len(rows)
            self.log.info(
                "earnings calendar: %d upcoming reports in the tracked universe", len(rows)
            )

    # ------------------------------------------------------------- surprises

    def _surprises(
        self, client: HttpClient, token: str, run_date: date, result: FetchResult
    ) -> None:
        """Historical estimate-vs-actual for a rotating slice of the universe."""
        symbols = self._slice(run_date, int(self.cfg("surprise_symbols_per_run", 60)))
        if not symbols:
            return

        now = datetime.now(UTC)
        rows = []
        for symbol in symbols:
            try:
                payload = client.get_json(
                    f"{BASE}/stock/earnings",
                    params={"symbol": to_yahoo_symbol(symbol), "token": token},
                )
                result.items_succeeded += 1
            except Exception as exc:
                result.items_failed += 1
                self.log.debug("earnings %s: %s", symbol, exc)
                continue

            for entry in payload or []:
                period = _to_date(entry.get("period"))
                if period is None:
                    continue
                rows.append(
                    {
                        "symbol": symbol,
                        "record_type": "surprise",
                        "period": period,
                        "fiscal_year": _int(entry.get("year")),
                        "fiscal_quarter": _int(entry.get("quarter")),
                        "eps_estimate": _float(entry.get("estimate")),
                        "eps_actual": _float(entry.get("actual")),
                        "eps_surprise": _float(entry.get("surprise")),
                        "eps_surprise_pct": _float(entry.get("surprisePercent")),
                        "source": SOURCE,
                        "ingested_at": now,
                    }
                )

        if rows:
            write = self.writer.write(
                pd.DataFrame(rows), P.EARNINGS, self.paths.earnings_file("surprises"), mode="merge"
            )
            result.record_write(write)
            result.details["surprise_rows"] = len(rows)
            self.log.info("earnings surprises: %d rows across %d symbols", len(rows), len(symbols))

    # ------------------------------------------------------- recommendations

    def _recommendations(
        self, client: HttpClient, token: str, run_date: date, result: FetchResult
    ) -> None:
        """Analyst recommendation distribution for a rotating slice."""
        symbols = self._slice(run_date, int(self.cfg("recommendation_symbols_per_run", 60)))
        if not symbols:
            return

        now = datetime.now(UTC)
        rows = []
        for symbol in symbols:
            try:
                payload = client.get_json(
                    f"{BASE}/stock/recommendation",
                    params={"symbol": to_yahoo_symbol(symbol), "token": token},
                )
            except Exception as exc:
                self.log.debug("recommendations %s: %s", symbol, exc)
                continue

            for entry in payload or []:
                period = _to_date(entry.get("period"))
                if period is None:
                    continue
                rows.append(
                    {
                        "symbol": symbol,
                        "record_type": "recommendation",
                        "period": period,
                        "strong_buy": _int(entry.get("strongBuy")),
                        "buy": _int(entry.get("buy")),
                        "hold": _int(entry.get("hold")),
                        "sell": _int(entry.get("sell")),
                        "strong_sell": _int(entry.get("strongSell")),
                        "source": SOURCE,
                        "ingested_at": now,
                    }
                )

        if rows:
            write = self.writer.write(
                pd.DataFrame(rows),
                P.EARNINGS,
                self.paths.earnings_file("recommendations"),
                mode="merge",
            )
            result.record_write(write)
            result.details["recommendation_rows"] = len(rows)

    # ---------------------------------------------------------------- helpers

    def _tracked(self) -> list[str]:
        if self.symbols_override is not None:
            return self.symbols_override
        if self.tracker is None:
            return []
        return self.tracker.tracked_symbols()

    def _slice(self, run_date: date, per_run: int) -> list[str]:
        if self.symbols_override is not None:
            return self.symbols_override
        members = self.tracker.current_members() if self.tracker else []
        if not members or per_run <= 0:
            return []
        offset = (run_date.toordinal() * per_run) % len(members)
        return (members[offset:] + members[:offset])[:per_run]


def _to_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
