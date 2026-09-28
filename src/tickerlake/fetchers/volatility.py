"""Volatility: CBOE index levels, daily and by the minute, and the VIX futures curve.

Three things, each chosen because the lake had no other source for it.

**Index levels, daily.** FRED already supplies ``VIXCLS``, a single close per day
for one index. CBOE publishes full OHLC history for its whole family --
maturities from one day to one year, the volatility of VIX itself, SKEW, and
the sector and asset-class indices. CBOE posts them in two evening batches,
around 18:00 ET and 20:30 ET, which is after the nightly run; a day's bars are
therefore written by the *next* run's lookback, one trading day late. They are
CBOE's own numbers, so they are used in preference to
Yahoo's, which carries no history at all for four of them (VIX1Y, VXEEM, VXSLV,
VXAPL). The MOVE index is ICE's, not CBOE's, and comes from Yahoo.

**Index levels, by the minute.** Yahoo serves 1-minute bars for the indices that
are computed intraday. VIX is computed through CBOE's global trading hours, so
with pre/post bars enabled its feed starts at 03:15 ET -- overnight volatility
that nothing else in the lake records. SKEW and MOVE are published once a day;
a minute feed of them is a single bar and is not requested. Yahoo keeps only
about thirty days of minute history, so the first run takes what exists and
every run after keeps the series going.

**The VIX futures curve.** Spot VIX is not tradeable and says nothing about the
term structure; the futures do. CBOE publishes each monthly contract's full daily
history -- settlement, volume, open interest -- back to 2013. Monthly contracts
only: the weeklies are thinly traded, and CBOE's settlement file shows them at
the monthly's exact price rather than a settlement of their own.

The indices are written into ``ohlcv`` and ``intraday_bars`` under a leading
caret (``^VIX``), Yahoo's convention for an index, so they can never collide
with a ticker and join to equities on the same timestamp. They are deliberately
*not* registered in the membership table. That table drives every fetcher, and
the ETFs registered there were found being asked for earnings, insider trades
and 8-Ks. An index is worse: Tiingo and Finnhub do not quote it, and it files
nothing, so it would reproduce that failure in every stage at once.
"""

from __future__ import annotations

import io
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from tickerlake.fetchers.base import FetchResult
from tickerlake.fetchers.yf_intraday import MAX_REQUEST_DAYS, YFinanceIntradayFetcher
from tickerlake.storage import paths as P
from tickerlake.utils.dates import as_date
from tickerlake.utils.http import HttpClient, HttpError
from tickerlake.utils.market_calendar import is_trading_day, previous_trading_day

CBOE_INDEX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"
VX_URL = "https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/VX_{expiry}.csv"
SOURCE_CBOE = "cboe"
SOURCE_YAHOO = "yfinance"
INDEX_PREFIX = "^"
# CBOE's CDN answers a bare client with an HTML error page.
USER_AGENT = "Mozilla/5.0 (compatible; TickerLake/0.1; research data collection)"


def lake_symbol(name: str) -> str:
    """``VIX`` -> ``^VIX``: the stored form of an index."""
    name = str(name).strip().upper()
    return name if name.startswith(INDEX_PREFIX) else INDEX_PREFIX + name


def _third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(4 - first.weekday()) % 7 + 14)


def vix_expiration(year: int, month: int) -> date:
    """Final settlement date of the monthly VX contract for ``year``/``month``.

    CBOE's rule: the Wednesday thirty days before the third Friday of the
    *following* month -- the SPX options expiration VIX is computed against.
    If that Friday is an exchange holiday the count starts from the business
    day before it, and if the resulting day is a holiday settlement moves back
    a business day. March 2025 is the case that needs both halves: 18 April
    2025 was Good Friday, so the March contract settled on Tuesday the 18th.
    """
    ny, nm = (year + 1, 1) if month == 12 else (year, month + 1)
    friday = _third_friday(ny, nm)
    if not is_trading_day(friday):
        friday = previous_trading_day(friday)
    settle = friday - timedelta(days=30)
    if not is_trading_day(settle):
        settle = previous_trading_day(settle)
    return settle


def monthly_expirations(start: date, end: date) -> list[date]:
    """Every monthly VX expiration falling within ``[start, end]``."""
    out: list[date] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        exp = vix_expiration(year, month)
        if start <= exp <= end:
            out.append(exp)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


class VolatilityFetcher(YFinanceIntradayFetcher):
    """Volatility index levels and the VIX futures curve.

    Inherits the intraday fetcher for its Yahoo download and per-session write;
    everything else is its own.
    """

    name = "volatility"
    dataset = P.OHLCV
    requires_trading_day = True

    def collect(self, run_date: date, result: FetchResult) -> None:
        cboe = {str(k).upper(): v for k, v in (self.cfg("indices", {}) or {}).items()}
        yahoo = {str(k).upper(): v for k, v in (self.cfg("yahoo_indices", {}) or {}).items()}
        daily_only = {str(n).upper() for n in (self.cfg("daily_only", []) or [])}
        if not cboe and not yahoo:
            result.add_warning("no volatility indices configured")
            return

        client = HttpClient(
            requests_per_second=float(self.cfg("requests_per_second", 2.0)),
            user_agent=USER_AGENT,
            max_attempts=3,
            timeout=60,
        )
        try:
            self._daily(run_date, cboe, yahoo, client, result)
            minute = [lake_symbol(n) for n in [*cboe, *yahoo] if n not in daily_only]
            self._minutes(run_date, minute, result)
            if self.cfg("futures.enabled", True):
                self._futures(run_date, client, result)
        finally:
            client.close()

    # ------------------------------------------------------------ daily levels

    def _daily(
        self,
        run_date: date,
        cboe: dict[str, Any],
        yahoo: dict[str, Any],
        client: HttpClient,
        result: FetchResult,
    ) -> None:
        symbols = [lake_symbol(n) for n in [*cboe, *yahoo]]
        first = self._first_stored(P.OHLCV, "date", symbols)
        lookback = run_date - timedelta(days=int(self.cfg("daily_lookback_days", 10)))
        now = datetime.now(UTC)

        frames: list[pd.DataFrame] = []
        backfilled: list[str] = []
        for name in cboe:
            symbol = lake_symbol(name)
            try:
                text = client.get_text(CBOE_INDEX_URL.format(name=name))
                frame = self.parse_cboe_index(text, symbol, now)
                result.items_succeeded += 1
            except Exception as exc:
                result.items_failed += 1
                result.add_warning(f"{symbol}: CBOE history unavailable ({type(exc).__name__})")
                continue
            frames.append(self._window(frame, symbol, first, lookback, backfilled))

        for name in yahoo:
            symbol = lake_symbol(name)
            full = first.get(symbol) is None or first[symbol] >= lookback
            try:
                frame = self._yahoo_daily(symbol, None if full else lookback, run_date, now)
                result.items_succeeded += 1
            except Exception as exc:
                result.items_failed += 1
                result.add_warning(f"{symbol}: Yahoo daily unavailable ({type(exc).__name__})")
                continue
            if frame is not None and not frame.empty:
                if full:
                    backfilled.append(symbol)
                frames.append(frame)

        frames = [f for f in frames if f is not None and not f.empty]
        if not frames:
            result.add_error("no volatility index history returned")
            return
        df = pd.concat(frames, ignore_index=True)
        for (year, month), group in df.groupby(
            [df["date"].map(lambda d: d.year), df["date"].map(lambda d: d.month)]
        ):
            write = self.writer.write(
                group, P.OHLCV, self.paths.ohlcv_backfill_file(year, month), mode="merge"
            )
            result.record_write(write)

        result.details["index_days_written"] = len(df)
        result.details["indices"] = sorted(df["symbol"].unique())
        if backfilled:
            result.details["indices_backfilled"] = sorted(backfilled)

    def _window(
        self,
        frame: pd.DataFrame,
        symbol: str,
        first: dict[str, date],
        lookback: date,
        backfilled: list[str],
    ) -> pd.DataFrame:
        """The whole history the first time, only recent days thereafter.

        CBOE's files hold everything back to 1990, and rewriting four hundred
        monthly partitions every night to restate numbers that never change
        would be all cost. A symbol whose stored history begins inside the
        lookback window has not been backfilled yet and gets it all.
        """
        stored = first.get(symbol)
        if stored is None or stored >= lookback:
            backfilled.append(symbol)
            return frame
        return frame[frame["date"] >= lookback]

    @staticmethod
    def parse_cboe_index(text: str, symbol: str, now: datetime) -> pd.DataFrame:
        """CBOE's ``*_History.csv``: OHLC for most, a single value for some.

        VVIX, SKEW and OVX publish ``DATE,<NAME>`` -- one number a day -- which
        is stored as open = high = low = close rather than leaving three
        columns empty for one index family and full for the rest.
        """
        raw = pd.read_csv(io.StringIO(text))
        raw.columns = [str(c).strip().upper() for c in raw.columns]
        if "DATE" not in raw.columns:
            raise ValueError(f"unexpected CBOE header: {list(raw.columns)[:6]}")
        days = pd.to_datetime(raw["DATE"], format="%m/%d/%Y", errors="coerce").dt.date

        values = [c for c in raw.columns if c != "DATE"]
        if {"OPEN", "HIGH", "LOW", "CLOSE"} <= set(values):
            o, h, lo, c = (
                pd.to_numeric(raw[k], errors="coerce") for k in ("OPEN", "HIGH", "LOW", "CLOSE")
            )
        elif len(values) == 1:
            o = h = lo = c = pd.to_numeric(raw[values[0]], errors="coerce")
        else:
            raise ValueError(f"unexpected CBOE columns: {values[:6]}")

        out = pd.DataFrame(
            {
                "symbol": symbol,
                "date": days,
                "open": o,
                "high": h,
                "low": lo,
                "close": c,
                # An index has no dividends and no adjustment; the adjusted
                # close is the close.
                "adj_close": c,
                # Nor volume. Null says "not applicable"; zero would claim no
                # contracts changed hands.
                "volume": pd.array([pd.NA] * len(raw), dtype="Int64"),
                "source": SOURCE_CBOE,
                "ingested_at": now,
            }
        )
        # CBOE's early history carries zero placeholders for missing opens.
        for col in ("open", "high", "low"):
            out.loc[out[col] <= 0, col] = pd.NA
        return out.dropna(subset=["date", "close"]).reset_index(drop=True)

    def _yahoo_daily(
        self, symbol: str, start: date | None, run_date: date, now: datetime
    ) -> pd.DataFrame | None:
        kwargs: dict[str, Any] = {"interval": "1d", "progress": False, "auto_adjust": False}
        if start is None:
            kwargs["period"] = "max"
        else:
            kwargs.update(start=start, end=run_date + timedelta(days=1))
        raw = yf.download(symbol, **kwargs)
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        close = pd.to_numeric(raw["Close"], errors="coerce")
        out = pd.DataFrame(
            {
                "symbol": symbol,
                "date": pd.to_datetime(raw.index).date,
                "open": pd.to_numeric(raw["Open"], errors="coerce").to_numpy(),
                "high": pd.to_numeric(raw["High"], errors="coerce").to_numpy(),
                "low": pd.to_numeric(raw["Low"], errors="coerce").to_numpy(),
                "close": close.to_numpy(),
                "adj_close": close.to_numpy(),
                "volume": pd.array([pd.NA] * len(raw), dtype="Int64"),
                "source": SOURCE_YAHOO,
                "ingested_at": now,
            }
        )
        return out.dropna(subset=["close"]).reset_index(drop=True)

    # --------------------------------------------------------- minute bars

    def _minutes(self, run_date: date, symbols: list[str], result: FetchResult) -> None:
        """1-minute bars, backfilling the ~30 days Yahoo keeps on the first run."""
        if not symbols:
            return
        interval = str(self.cfg("intraday_interval", "1m"))
        per_request = MAX_REQUEST_DAYS.get(interval, 7)
        first = self._first_stored(P.INTRADAY_BARS, "date", symbols)
        have_history = all(s in first for s in symbols)

        depth = int(
            self.cfg("intraday_lookback_days", per_request)
            if have_history
            else self.cfg("intraday_backfill_days", 29)
        )
        end = run_date + timedelta(days=1)
        start = run_date - timedelta(days=depth)

        frames: list[pd.DataFrame] = []
        cursor = start
        while cursor < end:
            stop = min(cursor + timedelta(days=per_request), end)
            try:
                frame = self._download(symbols, cursor, stop, interval)
            except Exception as exc:
                result.add_warning(f"minute bars {cursor}..{stop}: {type(exc).__name__}: {exc}")
                frame = None
            if frame is not None and not frame.empty:
                frame["source"] = SOURCE_YAHOO
                frames.append(frame)
            cursor = stop

        if not frames:
            result.add_warning(f"no {interval} volatility bars returned")
            return
        df = pd.concat(frames, ignore_index=True).drop_duplicates(
            subset=["symbol", "datetime", "interval"]
        )
        self._write_by_day(df, interval, result)
        result.details["minute_bars"] = len(df)
        result.details["minute_symbols"] = int(df["symbol"].nunique())
        if not have_history:
            result.details["minute_backfill_days"] = depth

    # -------------------------------------------------------------- futures

    def _futures(self, run_date: date, client: HttpClient, result: FetchResult) -> None:
        """Monthly VX contracts: everything settled once, live ones every run.

        A contract that has settled never changes, so it is fetched exactly
        once. Anything expiring within the last week or later is re-fetched
        each run, because its history is still being written.
        """
        start = as_date(self.cfg("futures.history_start", "2013-01-01")) or date(2013, 1, 1)
        months = int(self.cfg("futures.months_ahead", 10))
        horizon = date(
            run_date.year + (run_date.month + months - 1) // 12,
            (run_date.month + months - 1) % 12 + 1,
            28,
        )
        expirations = monthly_expirations(start, horizon)

        stored = self._stored_expirations()
        live_from = run_date - timedelta(days=7)
        wanted = [e for e in expirations if e >= live_from or e not in stored]
        if not wanted:
            return

        now = datetime.now(UTC)
        frames: list[pd.DataFrame] = []
        missing: list[date] = []
        for expiry in wanted:
            try:
                text = client.get_text(VX_URL.format(expiry=expiry.isoformat()))
            except HttpError:
                # Not listed yet, or never existed: both are normal at the far
                # end of the horizon.
                missing.append(expiry)
                continue
            except Exception as exc:
                result.add_warning(f"VX {expiry}: {type(exc).__name__}: {exc}")
                continue
            frame = self.parse_vx(text, expiry, now)
            if not frame.empty:
                frames.append(frame)

        if not frames:
            if wanted and len(missing) == len(wanted):
                result.add_warning("no VX contract files available")
            return
        df = pd.concat(frames, ignore_index=True)
        write = self.writer.write(df, P.VIX_FUTURES, self.paths.vix_futures_file(), mode="merge")
        result.record_write(write)
        result.details["vx_contracts_fetched"] = int(df["expiration"].nunique())
        result.details["vx_rows"] = len(df)
        # Only past expirations are a real gap; future ones simply are not listed yet.
        gaps = [e for e in missing if e < run_date]
        if gaps:
            result.add_warning(f"{len(gaps)} past VX expirations had no file: {gaps[:5]}")

    @staticmethod
    def parse_vx(text: str, expiration: date, now: datetime) -> pd.DataFrame:
        """One contract's CBOE history file."""
        raw = pd.read_csv(io.StringIO(text))
        raw.columns = [str(c).strip() for c in raw.columns]
        if "Trade Date" not in raw.columns or "Settle" not in raw.columns:
            return pd.DataFrame()

        num = lambda col: pd.to_numeric(raw.get(col), errors="coerce")  # noqa: E731
        volume = num("Total Volume")
        out = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(raw["Trade Date"], errors="coerce").dt.date,
                "expiration": expiration,
                "contract": raw.get("Futures"),
                "open": num("Open"),
                "high": num("High"),
                "low": num("Low"),
                "close": num("Close"),
                "settle": num("Settle"),
                "change": num("Change"),
                "volume": volume.astype("Int64"),
                "efp": num("EFP").astype("Int64"),
                "open_interest": num("Open Interest").astype("Int64"),
                "source": SOURCE_CBOE,
                "ingested_at": now,
            }
        )
        # No trades, no prices. On such days CBOE reports 0 for open and close
        # and a quoted high/low that can invert; the settlement stands alone.
        untraded = volume.fillna(0) <= 0
        out.loc[untraded, ["open", "high", "low", "close"]] = pd.NA
        for col in ("open", "high", "low", "close"):
            out.loc[out[col] <= 0, col] = pd.NA

        out = out.dropna(subset=["trade_date", "settle"])
        out = out[out["settle"] > 0]
        out["days_to_expiration"] = [(expiration - d).days for d in out["trade_date"]]
        return out.reset_index(drop=True)

    # ------------------------------------------------------------- helpers

    def _first_stored(self, dataset: str, column: str, symbols: list[str]) -> dict[str, date]:
        """Earliest stored date per symbol; absent symbols are simply missing."""
        if not symbols:
            return {}
        from tickerlake.storage.query import LakeQuery

        try:
            with LakeQuery(self.paths.root) as q:
                rows = q.sql(
                    f"SELECT symbol, MIN({column}) AS first FROM {dataset} "
                    f"WHERE symbol IN ({','.join('?' * len(symbols))}) GROUP BY symbol",
                    list(symbols),
                )
        except Exception as exc:
            self.log.debug("no stored %s yet: %s", dataset, exc)
            return {}
        return {str(r.symbol): as_date(r.first) for r in rows.itertuples() if as_date(r.first)}

    def _stored_expirations(self) -> set[date]:
        path = self.paths.vix_futures_file()
        if not path.exists():
            return set()
        import pyarrow.parquet as pq

        col = pq.read_table(path, columns=["expiration"]).column("expiration").to_pylist()
        return {as_date(d) for d in col if as_date(d)}
