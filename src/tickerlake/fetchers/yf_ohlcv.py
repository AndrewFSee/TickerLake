"""yfinance OHLCV fetcher.

Two modes share one code path:

* **backfill** -- full history from ``ohlcv.start_date``, written straight into
  each month's consolidated ``data.parquet`` with merge semantics.
* **incremental** (the daily run) -- a short lookback window rather than just
  yesterday, because Yahoo revises recent bars (splits, late corrections) and a
  merge-on-write makes re-fetching them free of duplicates.

Rows are always routed to the partition of the **bar's own date**, not the run
date. A Monday run covering the previous Thursday must not file Thursday's bar
under the current month if the month rolled over in between, or partition
pruning starts silently skipping real data.

Symbols are stored in canonical dot form (``BRK.B``) and converted to Yahoo's
dash dialect only for the request itself.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from tickerlake.fetchers.base import BaseFetcher, FetchResult
from tickerlake.storage import paths as P
from tickerlake.universe.sources import to_yahoo_symbol
from tickerlake.utils.throttle import AdaptiveThrottle, is_rate_limit_error

SOURCE = "yfinance"

# Canonical output columns in schema order (minus the constant ones).
_FIELD_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
    "Dividends": "dividends",
    "Stock Splits": "stock_splits",
    "Repaired?": "repaired",
}


class YFinanceOHLCVFetcher(BaseFetcher):
    """Daily OHLCV for the full tracked universe."""

    name = "ohlcv"
    dataset = P.OHLCV

    def __init__(self, *args: Any, backfill: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.backfill = backfill

    # ------------------------------------------------------------------ main

    def collect(self, run_date: date, result: FetchResult) -> None:
        symbols = self._symbols(run_date)
        if not symbols:
            result.add_warning("no symbols to fetch; is the membership table seeded?")
            return

        if self.backfill:
            start = _as_date(self.cfg("start_date", "2010-01-01"))
            end = run_date + timedelta(days=1)
            batch_size = int(self.cfg("backfill_batch_size", 25))
            throttle = float(self.cfg("backfill_throttle_seconds", 2.0))
            mode = "merge"
        else:
            # Five calendar days comfortably covers a long weekend plus a holiday,
            # so a run that is skipped for a couple of days still self-heals.
            start = run_date - timedelta(days=int(self.cfg("lookback_days", 5)))
            end = run_date + timedelta(days=1)
            batch_size = int(self.cfg("batch_size", 50))
            throttle = float(self.cfg("throttle_seconds", 1.0))
            mode = "merge"

        self.log.info(
            "%s %d symbols from %s to %s (batch=%d, throttle=%.1fs)",
            "backfilling" if self.backfill else "fetching",
            len(symbols),
            start,
            end - timedelta(days=1),
            batch_size,
            throttle,
        )

        limiter = AdaptiveThrottle(
            base_interval=throttle,
            jitter_pct=0.25,
            slowdown_factor=2.0,
            backoff_seconds=60.0,
            max_backoff_seconds=600.0,
        )

        frames: list[pd.DataFrame] = []
        observations: dict[str, date | None] = {}
        batches = [symbols[i : i + batch_size] for i in range(0, len(symbols), batch_size)]

        for n, batch in enumerate(batches, 1):
            limiter.wait()
            try:
                frame = self._download(batch, start, end)
                limiter.record_success()
            except Exception as exc:
                if is_rate_limit_error(exc):
                    limiter.record_rate_limit()
                    try:
                        frame = self._download(batch, start, end)  # one retry after backoff
                    except Exception as retry_exc:
                        result.items_failed += len(batch)
                        result.add_error(f"batch {n} failed after backoff: {retry_exc}")
                        continue
                else:
                    result.items_failed += len(batch)
                    result.add_error(f"batch {n} ({batch[0]}...): {type(exc).__name__}: {exc}")
                    continue

            if frame is not None and not frame.empty:
                frames.append(frame)
                returned = set(frame["symbol"].unique())
            else:
                returned = set()

            for symbol in batch:
                if symbol in returned:
                    last = frame.loc[frame["symbol"] == symbol, "date"].max()
                    observations[symbol] = last
                    result.items_succeeded += 1
                else:
                    observations[symbol] = None
                    result.items_skipped += 1

            self.log.info(
                "batch %d/%d: %d/%d symbols returned data",
                n,
                len(batches),
                len(returned),
                len(batch),
            )

        if not frames:
            result.add_error("no OHLCV data returned for any symbol")
            self._record_observations(observations, result)
            return

        combined = pd.concat(frames, ignore_index=True)
        self._write_by_month(combined, run_date, mode, result)
        self._record_observations(observations, result)

        result.details.update(
            {
                "symbols_requested": len(symbols),
                "symbols_with_data": result.items_succeeded,
                "date_range": f"{combined['date'].min()} .. {combined['date'].max()}",
                "throttle": limiter.stats(),
            }
        )

    # ------------------------------------------------------------- downloads

    def _symbols(self, run_date: date) -> list[str]:
        """Tracked universe: current members plus retained historical names."""
        if self.symbols_override is not None:
            return self.symbols_override
        if self.tracker is None:
            return []
        return self.tracker.tracked_symbols(as_of=run_date)

    def _download(self, symbols: list[str], start: date, end: date) -> pd.DataFrame | None:
        """Download one batch and return it in long form."""
        yahoo = [to_yahoo_symbol(s) for s in symbols]
        back = {to_yahoo_symbol(s): s for s in symbols}

        raw = yf.download(
            tickers=yahoo,
            start=start,
            end=end,
            interval=str(self.cfg("interval", "1d")),
            auto_adjust=bool(self.cfg("auto_adjust", False)),
            actions=bool(self.cfg("include_actions", True)),
            group_by="ticker",
            # Threading is what turns a polite batch into a burst; our own
            # throttle can only pace requests it actually controls.
            threads=False,
            progress=False,
            # Yahoo emits occasional 100x price errors and missed split
            # adjustments; yfinance's repair pass catches most of them.
            repair=bool(self.cfg("repair", True)),
        )
        if raw is None or raw.empty:
            return None
        return self._to_long(raw, yahoo, back)

    def _to_long(self, raw: pd.DataFrame, yahoo: list[str], back: dict[str, str]) -> pd.DataFrame:
        """Flatten yfinance's wide/MultiIndex output into our long schema."""
        parts: list[pd.DataFrame] = []
        now = datetime.now(UTC)

        if isinstance(raw.columns, pd.MultiIndex):
            available = set(raw.columns.get_level_values(0))
            for ysym in yahoo:
                if ysym not in available:
                    continue
                sub = raw[ysym].dropna(how="all")
                if sub.empty:
                    continue
                parts.append(self._one_symbol(sub, back.get(ysym, ysym), now))
        else:
            # Single-symbol batches come back with flat columns.
            sub = raw.dropna(how="all")
            if not sub.empty:
                ysym = yahoo[0]
                parts.append(self._one_symbol(sub, back.get(ysym, ysym), now))

        if not parts:
            return pd.DataFrame(columns=["symbol", "date"])
        return pd.concat(parts, ignore_index=True)

    def _one_symbol(self, sub: pd.DataFrame, symbol: str, now: datetime) -> pd.DataFrame:
        out = pd.DataFrame(index=sub.index)
        for src, dst in _FIELD_MAP.items():
            out[dst] = sub[src] if src in sub.columns else pd.NA

        # auto_adjust=True collapses Adj Close into Close; keep the column populated
        # either way so downstream code never has to branch on the setting.
        if out["adj_close"].isna().all():
            out["adj_close"] = out["close"]
        out["repaired"] = out["repaired"].fillna(False).astype(bool)

        out = out.reset_index()
        index_col = out.columns[0]
        dates = pd.to_datetime(out[index_col], errors="coerce")
        if isinstance(dates.dtype, pd.DatetimeTZDtype):
            dates = dates.dt.tz_convert(None)
        out["date"] = dates.dt.date
        out = out.drop(columns=[index_col])

        out["symbol"] = symbol
        out["source"] = SOURCE
        out["ingested_at"] = now
        # A bar with no close is Yahoo padding a non-trading day, not real data.
        out = out.dropna(subset=["date", "close"])
        return out

    # --------------------------------------------------------------- writing

    def _write_by_month(
        self, df: pd.DataFrame, run_date: date, mode: str, result: FetchResult
    ) -> None:
        """Route rows to the partition of each bar's own date.

        Both backfill and the daily incremental merge into the month's single
        consolidated file rather than the incremental dropping a separate delta
        beside it. The delta pattern is right for options, where per-symbol files
        are the unit of resumability -- but here it is actively harmful: the
        incremental re-fetches a multi-day lookback that *overlaps* whatever the
        backfill already wrote, so a delta sitting next to ``data.parquet``
        duplicates every overlapping bar until the weekly compaction runs.
        Queries in that window silently double-count.

        A month file is a few thousand rows, so merging into it daily is cheap,
        and the writer's atomic replace makes it safe.
        """
        df = df.copy()
        dates = pd.to_datetime(df["date"])
        df["_year"] = dates.dt.year
        df["_month"] = dates.dt.month

        for (year, month), group in df.groupby(["_year", "_month"], sort=True):
            group = group.drop(columns=["_year", "_month"])
            path = self.paths.ohlcv_backfill_file(int(year), int(month))

            write = self.writer.write(group, P.OHLCV, path, mode=mode)
            result.record_write(write)
            self.log.debug("%04d-%02d: %s", year, month, write)

    def _record_observations(
        self, observations: dict[str, date | None], result: FetchResult
    ) -> None:
        """Feed data availability back into silent-delisting detection."""
        if self.tracker is None or not observations:
            return
        flagged = self.tracker.record_data_observations(observations)
        self.tracker.save()
        if flagged:
            result.details["newly_flagged_delisted"] = flagged
            result.add_warning(
                f"{len(flagged)} symbol(s) newly flagged as suspected delisted: {', '.join(flagged[:10])}"
            )


def _as_date(value: str | date) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value)[:10])
