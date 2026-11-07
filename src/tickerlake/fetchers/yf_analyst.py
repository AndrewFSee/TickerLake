"""Analyst EPS estimates and broker rating changes from Yahoo (via yfinance).

Two datasets with opposite histories:

* **Estimates** (``analyst_estimates``). Yahoo serves only the *current*
  consensus per fiscal period, plus what it was 7/30/60/90 days ago and how many
  analysts revised up or down. Nothing older can ever be fetched, so the value
  of this dataset is the daily snapshot itself: every member, every run. A
  skipped day is a hole that cannot be filled later.

* **Rating changes** (``analyst_ratings``). Yahoo's upgrade/downgrade history
  goes back to about 2012 and is returned whole on every request, so a rotating
  slice keeps it current and ``tickerlake backfill analyst`` pulls it once for
  every symbol that was ever an index member -- delisted names included, which
  is what lets a backtest of rating changes avoid survivorship bias.

One request per symbol and dataset: estimates, trend and revision counts all
come from the same quoteSummary module (``earningsTrend``).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import yfinance as yf

from tickerlake.fetchers.base import BaseFetcher, FetchResult
from tickerlake.storage import paths as P
from tickerlake.universe.sources import to_yahoo_symbol
from tickerlake.utils.throttle import AdaptiveThrottle, is_rate_limit_error

SOURCE = "yfinance"
PERIODS = ("0q", "+1q", "0y", "+1y")


class YFinanceAnalystFetcher(BaseFetcher):
    """Daily consensus snapshots and the broker rating history."""

    name = "analyst"
    dataset = P.ANALYST_ESTIMATES

    def __init__(self, *args: Any, backfill: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        #: Backfill fetches rating history for every symbol ever in the index
        #: and skips the estimate snapshot (which only describes today).
        self.backfill = backfill

    def collect(self, run_date: date, result: FetchResult) -> None:
        throttle = AdaptiveThrottle(base_interval=float(self.cfg("throttle_seconds", 0.5)))
        if not self.backfill and self.cfg("fetch_estimates", True):
            self._run(
                self.company_members(), "estimates", run_date, throttle, result,
                P.ANALYST_ESTIMATES, self._estimates,
            )
        if self.cfg("fetch_ratings", True):
            symbols = (
                self._history_symbols()
                if self.backfill
                else self._slice(run_date, int(self.cfg("rating_symbols_per_run", 100)))
            )
            self._run(symbols, "ratings", run_date, throttle, result, P.ANALYST_RATINGS, self._ratings)
        result.details["throttle"] = throttle.stats()

    # ------------------------------------------------------------------ loop

    def _run(self, symbols, label, run_date, throttle, result, dataset, fetch) -> None:
        if not symbols:
            return
        every = max(1, int(self.cfg("write_every", 100)))
        now = datetime.now(UTC)
        rows: list[dict] = []
        empty = 0
        for i, symbol in enumerate(symbols, 1):
            got = self._with_retry(symbol, fetch, run_date, now, throttle)
            if got is None:
                result.items_failed += 1
            else:
                result.items_succeeded += 1
                empty += not got
                rows.extend(got)
            if rows and (i % every == 0 or i == len(symbols)):
                # Written in batches so a crash midway keeps what was fetched.
                write = self.writer.write(
                    pd.DataFrame(rows), dataset, self.paths.analyst_file(dataset), mode="merge"
                )
                result.record_write(write)
                rows = []
            if i % 100 == 0:
                self.log.info("%s: %d/%d symbols", label, i, len(symbols))
        result.details[f"{label}_symbols"] = len(symbols)
        result.details[f"{label}_empty"] = empty
        self.log.info("%s: %d symbols, %d with no data", label, len(symbols), empty)

    def _with_retry(self, symbol, fetch, run_date, now, throttle) -> list[dict] | None:
        for _ in range(3):
            throttle.wait()
            try:
                rows = fetch(yf.Ticker(to_yahoo_symbol(symbol)), symbol, run_date, now)
            except Exception as exc:
                if is_rate_limit_error(exc):
                    throttle.record_rate_limit()
                    continue
                self.log.debug("analyst %s: %s: %s", symbol, type(exc).__name__, exc)
                return None
            throttle.record_success()
            return rows
        return None

    # --------------------------------------------------------------- fetches

    @staticmethod
    def _estimates(ticker, symbol: str, run_date: date, now: datetime) -> list[dict]:
        est, trend, rev = ticker.earnings_estimate, ticker.eps_trend, ticker.eps_revisions
        # Period end dates are only in the raw module; yfinance drops them from its frames.
        raw = getattr(getattr(ticker, "_analysis", None), "_earnings_trend", None) or []
        ends = {item.get("period"): item.get("endDate") for item in raw if isinstance(item, dict)}
        return estimate_rows(symbol, est, trend, rev, ends, run_date, now)

    @staticmethod
    def _ratings(ticker, symbol: str, run_date: date, now: datetime) -> list[dict]:
        try:
            df = ticker.upgrades_downgrades
        except Exception as exc:
            if "no upgrade/downgrade history" in str(exc).lower():
                return []  # a real answer: nobody covers it
            raise
        return rating_rows(symbol, df, now)

    # --------------------------------------------------------------- symbols

    def _slice(self, run_date: date, per_run: int) -> list[str]:
        if self.symbols_override is not None:
            return self.symbols_override
        members = self.company_members()
        if not members or per_run <= 0:
            return []
        offset = (run_date.toordinal() * per_run) % len(members)
        return (members[offset:] + members[:offset])[:per_run]

    def _history_symbols(self) -> list[str]:
        """Every symbol that was ever an index member, current or not."""
        if self.symbols_override is not None:
            return self.symbols_override
        if self.tracker is None:
            return []
        return sorted(self.tracker.constituent_symbols())


# ---------------------------------------------------------------- transforms


def estimate_rows(symbol, est, trend, rev, ends, snapshot: date, now: datetime) -> list[dict]:
    """One row per fiscal period from yfinance's three earningsTrend frames."""

    def cell(df, period, col):
        if df is None or getattr(df, "empty", True) or period not in df.index or col not in df.columns:
            return None
        v = df.at[period, col]
        return None if pd.isna(v) else v

    rows = []
    for period in PERIODS:
        row = {
            "symbol": symbol,
            "snapshot_date": snapshot,
            "period": period,
            "period_end": _to_date(ends.get(period)),
            "eps_avg": _float(cell(est, period, "avg")),
            "eps_low": _float(cell(est, period, "low")),
            "eps_high": _float(cell(est, period, "high")),
            "eps_year_ago": _float(cell(est, period, "yearAgoEps")),
            "growth": _float(cell(est, period, "growth")),
            "analysts": _int(cell(est, period, "numberOfAnalysts")),
            "eps_trend_current": _float(cell(trend, period, "current")),
            "eps_trend_7d": _float(cell(trend, period, "7daysAgo")),
            "eps_trend_30d": _float(cell(trend, period, "30daysAgo")),
            "eps_trend_60d": _float(cell(trend, period, "60daysAgo")),
            "eps_trend_90d": _float(cell(trend, period, "90daysAgo")),
            "up_7d": _int(cell(rev, period, "upLast7days")),
            "up_30d": _int(cell(rev, period, "upLast30days")),
            # Yahoo's own casing differs between these two keys.
            "down_7d": _int(cell(rev, period, "downLast7Days")),
            "down_30d": _int(cell(rev, period, "downLast30days")),
            "source": SOURCE,
            "ingested_at": now,
        }
        values = [v for k, v in row.items() if k.startswith(("eps_", "analysts", "up_", "down_"))]
        if any(v is not None for v in values):
            rows.append(row)
    return rows


def rating_rows(symbol: str, df: pd.DataFrame | None, now: datetime) -> list[dict]:
    """Rows from yfinance's upgrades_downgrades frame (index = naive UTC timestamps)."""
    if df is None or df.empty:
        return []
    out = df.reset_index()
    stamp = out.columns[0]
    when = pd.to_datetime(out[stamp], errors="coerce", utc=True)
    rows = []
    for i, r in out.iterrows():
        firm = str(r.get("Firm") or "").strip()
        if pd.isna(when[i]) or not firm:
            continue
        rows.append({
            "symbol": symbol,
            "graded_at": when[i].to_pydatetime(),
            "firm": firm,
            "to_grade": _str(r.get("ToGrade")),
            "from_grade": _str(r.get("FromGrade")),
            "action": _str(r.get("Action")),
            "price_target_action": _str(r.get("priceTargetAction")),
            # Yahoo sends 0 for "no target"; a real $0 target doesn't exist.
            "price_target": _target(r.get("currentPriceTarget")),
            "prior_price_target": _target(r.get("priorPriceTarget")),
            "source": SOURCE,
            "ingested_at": now,
        })
    return rows


def _to_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _float(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(v) else v


def _target(value: Any) -> float | None:
    v = _float(value)
    return v if v and v > 0 else None


def _int(value: Any) -> int | None:
    v = _float(value)
    return None if v is None else int(v)


def _str(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    return s or None
