"""yfinance intraday bars -- the finest free granularity, and it expires.

Why this exists
---------------
True tick data is not free anywhere (see the README). What *is* free is Yahoo's
intraday bar series, and the important property is that it is a **rolling
window**: Yahoo serves roughly 30 days of 1-minute bars and then drops them
permanently. Measured limits, per request:

    1m   8 days max per request, ~30 days retained
    2m   60 days
    5m   60 days
    15m  60 days
    1h   730 days

That makes 1-minute data exactly like the option chains: if you do not capture it
today, that day is gone for good. Everything below 1h is worth collecting daily
for that reason alone, regardless of whether a model needs it yet.

What this is not
----------------
These are bars, not ticks. There is no trade-level detail, no bid/ask, no trade
conditions, and no size beyond per-bar volume. Anything requiring order-flow
microstructure needs a real tape (see ``alpaca`` in the README).
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

# Yahoo's hard per-request span for each interval, measured empirically.
# Exceeding these returns an error rather than a truncated result.
MAX_REQUEST_DAYS = {
    "1m": 7,  # documented as 8; 7 keeps a safety margin
    "2m": 59,
    "5m": 59,
    "15m": 59,
    "30m": 59,
    "60m": 729,
    "1h": 729,
}

_FIELD_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}


class YFinanceIntradayFetcher(BaseFetcher):
    """Intraday bars for the current index members."""

    name = "intraday"
    dataset = P.INTRADAY_BARS

    def __init__(self, *args: Any, limit: int | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.limit = limit

    # ------------------------------------------------------------------ main

    def collect(self, run_date: date, result: FetchResult) -> None:
        symbols = self._symbols(run_date)
        if not symbols:
            result.add_warning("no symbols to fetch; is the membership table seeded?")
            return

        interval = str(self.cfg("interval", "1m"))
        lookback = int(self.cfg("lookback_days", 5))
        cap = MAX_REQUEST_DAYS.get(interval, 7)
        if lookback > cap:
            result.add_warning(
                f"lookback_days={lookback} exceeds Yahoo's {cap}-day limit for {interval}; "
                f"clamping. Increase the run frequency instead of the window."
            )
            lookback = cap

        start = run_date - timedelta(days=lookback)
        end = run_date + timedelta(days=1)
        batch_size = int(self.cfg("batch_size", 25))
        throttle = AdaptiveThrottle(
            base_interval=float(self.cfg("throttle_seconds", 1.0)),
            jitter_pct=0.25,
            backoff_seconds=60.0,
            max_backoff_seconds=600.0,
        )

        self.log.info(
            "fetching %s bars for %d symbols, %s to %s (batch=%d)",
            interval,
            len(symbols),
            start,
            run_date,
            batch_size,
        )

        frames: list[pd.DataFrame] = []
        batches = [symbols[i : i + batch_size] for i in range(0, len(symbols), batch_size)]

        for n, batch in enumerate(batches, 1):
            throttle.wait()
            try:
                frame = self._download(batch, start, end, interval)
                throttle.record_success()
            except Exception as exc:
                if is_rate_limit_error(exc):
                    throttle.record_rate_limit()
                    try:
                        frame = self._download(batch, start, end, interval)
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
                returned = frame["symbol"].nunique()
            else:
                returned = 0
            result.items_succeeded += returned
            result.items_skipped += len(batch) - returned
            self.log.info(
                "batch %d/%d: %d/%d symbols returned bars", n, len(batches), returned, len(batch)
            )

        if not frames:
            result.add_error(f"no {interval} bars returned for any symbol")
            return

        combined = pd.concat(frames, ignore_index=True)
        self._write_by_day(combined, interval, result)

        result.details.update(
            {
                "interval": interval,
                "symbols_requested": len(symbols),
                "symbols_with_data": result.items_succeeded,
                "bars": len(combined),
                "sessions": int(combined["date"].nunique()),
                "date_range": f"{combined['date'].min()} .. {combined['date'].max()}",
                "throttle": throttle.stats(),
            }
        )

    # ------------------------------------------------------------- downloads

    def _symbols(self, run_date: date) -> list[str]:
        """Current members only.

        Intraday history for delisted names is not retained by Yahoo anyway, so
        requesting it spends rate-limit budget on guaranteed-empty responses.
        """
        if self.symbols_override is not None:
            symbols = list(self.symbols_override)
        elif self.tracker is not None:
            symbols = self.tracker.current_members()
        else:
            return []
        if self.limit:
            symbols = symbols[: self.limit]
        return symbols

    def _download(
        self, symbols: list[str], start: date, end: date, interval: str
    ) -> pd.DataFrame | None:
        yahoo = [to_yahoo_symbol(s) for s in symbols]
        back = {to_yahoo_symbol(s): s for s in symbols}

        raw = yf.download(
            tickers=yahoo,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            actions=False,
            group_by="ticker",
            threads=False,
            progress=False,
            # Pre/post-market bars are thin and often erroneous, but they are
            # also where gap-driven signals live. Configurable, off by default.
            prepost=bool(self.cfg("include_prepost", False)),
        )
        if raw is None or raw.empty:
            return None

        now = datetime.now(UTC)
        parts = []
        if isinstance(raw.columns, pd.MultiIndex):
            available = set(raw.columns.get_level_values(0))
            for ysym in yahoo:
                if ysym not in available:
                    continue
                sub = raw[ysym].dropna(how="all")
                if not sub.empty:
                    parts.append(self._one_symbol(sub, back.get(ysym, ysym), interval, now))
        else:
            sub = raw.dropna(how="all")
            if not sub.empty:
                parts.append(self._one_symbol(sub, back.get(yahoo[0], yahoo[0]), interval, now))

        return pd.concat(parts, ignore_index=True) if parts else None

    def _one_symbol(
        self, sub: pd.DataFrame, symbol: str, interval: str, now: datetime
    ) -> pd.DataFrame:
        out = pd.DataFrame(index=sub.index)
        for src, dst in _FIELD_MAP.items():
            out[dst] = sub[src] if src in sub.columns else pd.NA

        out = out.reset_index()
        index_col = out.columns[0]
        stamps = pd.to_datetime(out[index_col], errors="coerce", utc=True)
        out["datetime"] = stamps
        # Session date in US Eastern, not UTC: a 16:00 ET bar is 20:00 UTC, and
        # partitioning on the UTC date would file the whole afternoon session
        # under the correct day only by luck of the timezone offset.
        out["date"] = stamps.dt.tz_convert("America/New_York").dt.date
        out = out.drop(columns=[index_col])

        out["symbol"] = symbol
        out["interval"] = interval
        out["source"] = SOURCE
        out["ingested_at"] = now
        return out.dropna(subset=["datetime", "close"])

    # --------------------------------------------------------------- writing

    def _write_by_day(self, df: pd.DataFrame, interval: str, result: FetchResult) -> None:
        """One file per session, so a re-run overwrites exactly that day."""
        for session, group in df.groupby("date", sort=True):
            path = self.paths.intraday_file(interval, session)
            write = self.writer.write(group, P.INTRADAY_BARS, path, mode="merge")
            result.record_write(write)
            self.log.debug("%s %s: %s", interval, session, write)
