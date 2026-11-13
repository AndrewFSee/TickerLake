"""Daily bars for every listed common stock on an exchange, for breadth.

The index stage (``ohlcv``) collects the S&P 500 and the tracked ETFs. Exchange
breadth -- the share of NYSE stocks advancing, or above their 200-day average --
needs every common stock on the exchange, about 2,100 for the NYSE. Those come
from the ``listings`` snapshots, never from the membership table: every stage that
iterates the membership table (options, intraday, filings, news) would otherwise
start fetching them too.

Same code path as ``ohlcv`` (lookback rewrite, rescale detection, dividend check,
newcomer backfill), written to its own dataset.

Yahoo posts the day's daily bar for most of these names hours after the close:
at 21:30 ET on 2026-11-12, 11 of 40 sampled NYSE stocks had one, while every one
had that day's hourly bars. So after the regular download, any symbol still
missing the run date's bar gets one built from its hourly bars, stored with
``source = "yfinance_1h"``. Its close is the last trade before 16:00, a few cents
from the official closing auction; the next run's lookback replaces it with
Yahoo's daily bar.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from tickerlake.fetchers.base import FetchResult
from tickerlake.fetchers.listings import listed_symbols
from tickerlake.fetchers.yf_ohlcv import YFinanceOHLCVFetcher
from tickerlake.storage import paths as P
from tickerlake.universe.sources import to_yahoo_symbol
from tickerlake.utils.market_calendar import is_trading_day
from tickerlake.utils.throttle import AdaptiveThrottle

PROVISIONAL_SOURCE = "yfinance_1h"
LAST_HOURLY_BAR = (15, 30)  # the 15:30-16:00 bar must exist, or the session isn't over


class ListedOHLCVFetcher(YFinanceOHLCVFetcher):
    """Daily OHLCV for the listed universe chosen in config (default: NYSE common stock)."""

    name = "listed_ohlcv"
    dataset = P.OHLCV_LISTED

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._listed: list[str] = []

    def collect(self, run_date: date, result: FetchResult) -> None:
        super().collect(run_date, result)
        if not self.backfill and self.cfg("provisional_from_hourly", True):
            self._provisional_session(run_date, result)

    # ------------------------------------------------------------ universe

    def _symbols(self, run_date: date) -> list[str]:
        if self.symbols_override is not None:
            self._listed = list(self.symbols_override)
            return self._listed
        try:
            self._listed = listed_symbols(
                self.paths.root, run_date,
                list(self.cfg("exchanges", ["N"])), list(self.cfg("categories", ["common"])),
            )
        except Exception as exc:
            self.log.warning("no listings to read (%s); run the listings stage first", exc)
            self._listed = []
        return self._listed

    def _current_symbols(self) -> set[str] | None:
        return set(self._listed)

    def _month_file(self, year: int, month: int):
        return self.paths.listed_ohlcv_file(year, month)

    def _record_observations(self, observations: dict[str, date | None], result: FetchResult) -> None:
        """Delistings show up in the listings snapshots; the membership table is not ours."""

    # ------------------------------------------------- same-day provisional bars

    def _provisional_session(self, run_date: date, result: FetchResult) -> None:
        if not is_trading_day(run_date) or not self._listed:
            return
        missing = self._missing_on(run_date)
        if not missing:
            return
        limiter = AdaptiveThrottle(base_interval=float(self.cfg("throttle_seconds", 1.0)), jitter_pct=0.25)
        size = int(self.cfg("batch_size", 50))
        bars: list[pd.DataFrame] = []
        for i in range(0, len(missing), size):
            limiter.wait()
            try:
                raw = self._download_hourly(missing[i : i + size], run_date)
                limiter.record_success()
            except Exception as exc:
                result.add_warning(f"hourly batch {i // size + 1}: {type(exc).__name__}: {exc}")
                continue
            if raw is not None:
                bars.append(raw)
        built = pd.concat(bars, ignore_index=True) if bars else pd.DataFrame()
        if built.empty:
            result.add_warning(f"{len(missing)} symbols lack a {run_date} bar and hourly bars gave none")
            return
        self._write_by_month(built, run_date, "merge", result)
        result.details["provisional_bars"] = len(built)
        result.details["provisional_missing"] = len(missing) - len(built)
        self.log.info("%d of %d missing %s bars built from hourly bars", len(built), len(missing), run_date)

    def _missing_on(self, run_date: date) -> list[str]:
        from tickerlake.storage.query import LakeQuery

        try:
            with LakeQuery(self.paths.root) as q:
                have = q.sql(f"SELECT DISTINCT symbol FROM {self.dataset} WHERE date = ?", [run_date])
        except Exception as exc:
            self.log.debug("could not read stored bars: %s", exc)
            return []
        return sorted(set(self._listed) - set(have["symbol"].astype(str)))

    def _download_hourly(self, symbols: list[str], day: date) -> pd.DataFrame | None:
        yahoo = [to_yahoo_symbol(s) for s in symbols]
        raw = yf.download(
            tickers=yahoo, start=day, end=day + timedelta(days=1), interval="1h",
            auto_adjust=False, actions=False, group_by="ticker", threads=False, progress=False,
        )
        if raw is None or raw.empty:
            return None
        return hourly_to_daily(raw, yahoo, {to_yahoo_symbol(s): s for s in symbols}, day)


def hourly_to_daily(raw: pd.DataFrame, yahoo: list[str], back: dict[str, str], day: date) -> pd.DataFrame:
    """One daily bar per symbol from a finished session's hourly bars (yfinance layout)."""
    now = datetime.now(UTC)
    rows = []
    for ysym in yahoo:
        if isinstance(raw.columns, pd.MultiIndex):
            if ysym not in set(raw.columns.get_level_values(0)):
                continue
            sub = raw[ysym]
        else:
            sub = raw
        sub = sub.dropna(subset=["Close"])
        if sub.empty:
            continue
        idx = pd.DatetimeIndex(sub.index)
        idx = idx.tz_convert("America/New_York") if idx.tz is not None else idx
        sub = sub[idx.date == day]
        idx = idx[idx.date == day]
        if sub.empty or (idx[-1].hour, idx[-1].minute) < LAST_HOURLY_BAR:
            continue
        rows.append({
            "symbol": back.get(ysym, ysym), "date": day,
            "open": float(sub["Open"].iloc[0]), "high": float(sub["High"].max()),
            "low": float(sub["Low"].min()), "close": float(sub["Close"].iloc[-1]),
            "adj_close": float(sub["Close"].iloc[-1]),
            "volume": int(sub["Volume"].fillna(0).sum()),
            "dividends": 0.0, "stock_splits": 0.0, "repaired": False,
            "source": PROVISIONAL_SOURCE, "ingested_at": now,
        })
    return pd.DataFrame(rows)
