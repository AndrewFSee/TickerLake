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
from tickerlake.utils.dates import as_date
from tickerlake.utils.throttle import AdaptiveThrottle, is_rate_limit_error

SOURCE = "yfinance"

# Plan label for the newcomer pass, matched on in a couple of places.
NEWCOMER_LABEL = "backfilling newcomers"

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

        backfill_start = _as_date(self.cfg("start_date", "2010-01-01"))
        end = run_date + timedelta(days=1)

        if self.backfill:
            plans = [("backfilling", symbols, backfill_start)]
            batch_size = int(self.cfg("backfill_batch_size", 25))
            throttle = float(self.cfg("backfill_throttle_seconds", 2.0))
        else:
            # Five calendar days comfortably covers a long weekend plus a holiday,
            # so a run that is skipped for a couple of days still self-heals.
            lookback = run_date - timedelta(days=int(self.cfg("lookback_days", 5)))
            batch_size = int(self.cfg("batch_size", 50))
            throttle = float(self.cfg("throttle_seconds", 1.0))

            newcomers = self._newcomers(symbols, run_date, result)
            regular = [s for s in symbols if s not in newcomers]
            plans = [("fetching", regular, lookback)]
            if newcomers:
                plans.append((NEWCOMER_LABEL, newcomers, backfill_start))
                self.log.info(
                    "%d symbol(s) have no history to speak of and will be backfilled: %s",
                    len(newcomers),
                    ", ".join(newcomers[:10]) + ("..." if len(newcomers) > 10 else ""),
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
        backfilled: list[str] = []

        for label, group, group_start in plans:
            if not group:
                continue
            # Smaller batches for newcomers: a full-history request per symbol is
            # far heavier than a five-day window.
            size = min(batch_size, 25) if label == NEWCOMER_LABEL else batch_size
            self.log.info(
                "%s %d symbols from %s to %s (batch=%d, throttle=%.1fs)",
                label,
                len(group),
                group_start,
                end - timedelta(days=1),
                size,
                throttle,
            )
            self._fetch_range(group, group_start, end, size, limiter, result, frames, observations)
            if label == NEWCOMER_LABEL:
                backfilled = list(group)

        if not frames:
            result.add_error("no OHLCV data returned for any symbol")
            self._record_observations(observations, result)
            return

        combined = pd.concat(frames, ignore_index=True)
        self._write_by_month(combined, run_date, "merge", result)
        self._record_observations(observations, result)

        result.details.update(
            {
                "symbols_requested": len(symbols),
                "symbols_with_data": result.items_succeeded,
                "date_range": f"{combined['date'].min()} .. {combined['date'].max()}",
                "throttle": limiter.stats(),
            }
        )
        if backfilled:
            result.details["backfilled_newcomers"] = backfilled

    # ------------------------------------------------------------- newcomers

    def _newcomers(self, symbols: list[str], run_date: date, result: FetchResult) -> list[str]:
        """Symbols to backfill today, with a ceiling on how many.

        A reconstitution moves a handful of names. If nearly the whole universe
        looks new, the lake is empty rather than the index rewritten -- a fresh
        install, or a data directory pointed somewhere unexpected -- and turning
        the nightly run into a full-universe backfill is the wrong response to
        that. ``tickerlake backfill ohlcv`` is the deliberate way to do it.
        """
        found = self._needs_history(symbols, run_date)

        # Only names currently in the index. The tracked universe also carries
        # every symbol ever removed, and the delisted ones have no history for
        # the same reason they have no future: the vendor will not serve it.
        # AGL Resources (GAS) left the index in 2016 with zero bars stored, and
        # without this it would be re-requested every night forever.
        if self.tracker is not None and found:
            current = set(self.tracker.current_members())
            skipped = [s for s in found if s not in current]
            if skipped:
                self.log.debug(
                    "%d symbol(s) lack history but are no longer index members: %s",
                    len(skipped),
                    ", ".join(skipped[:10]),
                )
            found = [s for s in found if s in current]

        ceiling = int(self.cfg("max_newcomers", 25))
        if len(found) > ceiling:
            result.add_warning(
                f"{len(found)} symbols have no stored history, above the {ceiling} "
                "expected from a reconstitution; skipping the newcomer backfill. "
                "Run `tickerlake backfill ohlcv` if this lake is genuinely empty."
            )
            self.log.warning(
                "%d symbols look new (ceiling %d); not backfilling them in a daily run",
                len(found),
                ceiling,
            )
            return []
        return found

    def _needs_history(self, symbols: list[str], run_date: date) -> list[str]:
        """Symbols whose stored history is too short to be their real history.

        A name joining the index mid-quarter arrives with only the bars collected
        since it joined. Bloom Energy and P entered on 2026-09-21 with five bars
        each, reaching back only to the day the universe first saw them. Nothing
        failed -- that is the problem. The symbol is simply short, so any feature
        needing a lookback has nothing to compute from, and the gap shows up only
        if someone thinks to look.

        The test is on *recency* rather than row count so that it stops firing by
        itself: once the history is fetched the earliest bar moves back years and
        the symbol drops out. A genuinely recent listing keeps qualifying for
        ``recent_history_days`` and then stops, costing one small extra request a
        day meanwhile and fetching exactly the short history it really has.
        """
        if not symbols:
            return []
        from tickerlake.storage.query import LakeQuery

        cutoff = run_date - timedelta(days=int(self.cfg("recent_history_days", 30)))
        placeholders = ",".join("?" * len(symbols))
        try:
            with LakeQuery(self.paths.root) as q:
                stored = q.sql(
                    "SELECT symbol, MIN(date) AS first_bar FROM ohlcv "
                    f"WHERE symbol IN ({placeholders}) GROUP BY symbol",
                    list(symbols),
                )
        except Exception as exc:
            # A lake too young to query is not a reason to derail the run; the
            # first backfill populates everything anyway.
            self.log.debug("could not check stored history: %s", exc)
            return []

        earliest = {str(row.symbol): as_date(row.first_bar) for row in stored.itertuples()}
        return [s for s in symbols if (earliest.get(s) or date.max) >= cutoff]

    # ------------------------------------------------------------- downloads

    def _fetch_range(
        self,
        symbols: list[str],
        start: date,
        end: date,
        batch_size: int,
        limiter: AdaptiveThrottle,
        result: FetchResult,
        frames: list[pd.DataFrame],
        observations: dict[str, date | None],
    ) -> None:
        """Download one symbol group over one date range, batch by batch."""
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
