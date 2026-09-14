"""Tiingo: an independent check on *adjusted* prices.

Why adjusted rather than raw
----------------------------
The Finnhub cross-check already validates raw closes, and it agrees with
yfinance to the cent. Raw closes are the easy case: every vendor sees the same
print. Adjusted prices are where sources actually diverge, because adjustment is
a *computation* over the dividend and split history, and a vendor that misses a
corporate action produces a series that looks perfectly reasonable while being
silently wrong for every date before it.

That matters here specifically: ``adj_close`` is what any sane return
calculation uses, and the lake spans 2010-present across 654 symbols including
15 splits among large caps alone. A missed split is a 4x or 10x step in a return
series, and nothing else in the pipeline would notice.

Tiingo is well suited to this because it publishes ``adjOpen``/``adjHigh``/
``adjLow``/``adjClose`` plus per-bar ``divCash`` and ``splitFactor``. yfinance
gives only an adjusted close, so Tiingo can confirm both the level and the
corporate actions that produced it.

Free-tier sizing
----------------
The free tier caps *unique symbols per month*, not just daily requests, and the
tracked universe is larger than that cap. So this deliberately samples a
rotating slice rather than sweeping everything: the point is detecting a
systemic adjustment problem, which a sample finds just as well as a full sweep.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd

from tickerlake.fetchers.base import BaseFetcher, FetchResult
from tickerlake.storage import paths as P
from tickerlake.universe.sources import to_yahoo_symbol
from tickerlake.utils.dates import as_date
from tickerlake.utils.http import HttpClient

SOURCE = "tiingo"
BASE = "https://api.tiingo.com/tiingo/daily"

# Relative tolerance before a difference is called a mismatch. Adjusted prices
# legitimately differ in the last decimal between vendors because of rounding in
# the adjustment factor, so this is looser than an exact match but far tighter
# than a missed split (which shows up as 2x, 4x or 10x).
PRICE_TOLERANCE = 0.005


class TiingoFetcher(BaseFetcher):
    """Cross-checks stored adjusted prices against an independent vendor."""

    name = "tiingo"
    dataset = P.QUALITY
    requires_secret = "tiingo_api_key"
    requires_trading_day = True

    def collect(self, run_date: date, result: FetchResult) -> None:
        token = self.config.secrets.tiingo_api_key
        symbols = self._slice(run_date, int(self.cfg("symbols_per_run", 25)))
        if not symbols:
            result.add_warning("no symbols to cross-check")
            return

        lookback = int(self.cfg("lookback_days", 10))
        start = run_date - timedelta(days=lookback)
        client = HttpClient(
            requests_per_second=float(self.cfg("requests_per_second", 3.0)),
            user_agent="TickerLake/0.1",
            max_attempts=3,
        )
        client.session.headers.update({"Authorization": f"Token {token}"})

        ours = self._our_bars(start, run_date, symbols)
        if not ours:
            result.add_warning("no stored bars to compare against")
            client.close()
            return

        now = datetime.now(UTC)
        rows: list[dict[str, Any]] = []
        compared = close_bad = adj_bad = 0

        try:
            for symbol in symbols:
                mine = ours.get(symbol)
                if not mine:
                    continue
                try:
                    payload = client.get_json(
                        f"{BASE}/{to_yahoo_symbol(symbol)}/prices",
                        params={
                            "startDate": start.isoformat(),
                            "endDate": run_date.isoformat(),
                        },
                    )
                    result.items_succeeded += 1
                except Exception as exc:
                    result.items_failed += 1
                    self.log.debug("tiingo %s: %s", symbol, exc)
                    continue

                for bar in payload or []:
                    day = as_date(str(bar.get("date"))[:10])
                    theirs_close = _as_float(bar.get("close"))
                    theirs_adj = _as_float(bar.get("adjClose"))
                    if day is None or day not in mine:
                        continue

                    our_close, our_adj = mine[day]
                    compared += 1

                    close_diff = _rel(our_close, theirs_close)
                    adj_diff = _rel(our_adj, theirs_adj)
                    close_ok = close_diff is None or close_diff <= PRICE_TOLERANCE
                    adj_ok = adj_diff is None or adj_diff <= PRICE_TOLERANCE

                    if not close_ok:
                        close_bad += 1
                    if not adj_ok:
                        adj_bad += 1
                        self.log.warning(
                            "adjusted price mismatch %s %s: ours=%.4f tiingo=%.4f (%.2f%%) "
                            "- check for a missed split or dividend",
                            symbol,
                            day,
                            our_adj or 0,
                            theirs_adj or 0,
                            (adj_diff or 0) * 100,
                        )

                    rows.append(
                        _quality_row(
                            run_date,
                            symbol,
                            day,
                            "close",
                            our_close,
                            theirs_close,
                            close_diff,
                            close_ok,
                            now,
                        )
                    )
                    rows.append(
                        _quality_row(
                            run_date,
                            symbol,
                            day,
                            "adj_close",
                            our_adj,
                            theirs_adj,
                            adj_diff,
                            adj_ok,
                            now,
                        )
                    )
        finally:
            client.close()

        if rows:
            write = self.writer.write(
                pd.DataFrame(rows), P.QUALITY, self.paths.quality_file(run_date), mode="merge"
            )
            result.record_write(write)

        result.details.update(
            {
                "bars_compared": compared,
                "close_mismatches": close_bad,
                "adj_close_mismatches": adj_bad,
                "tolerance": PRICE_TOLERANCE,
            }
        )
        if compared and adj_bad / compared > 0.05:
            result.add_warning(
                f"{adj_bad}/{compared} adjusted closes disagree with Tiingo by "
                f">{PRICE_TOLERANCE:.1%} - likely a missed corporate action"
            )
        elif compared:
            self.log.info(
                "cross-check: %d/%d closes and %d/%d adjusted closes agree with Tiingo",
                compared - close_bad,
                compared,
                compared - adj_bad,
                compared,
            )

    # --------------------------------------------------------------- helpers

    def _our_bars(
        self, start: date, end: date, symbols: list[str]
    ) -> dict[str, dict[date, tuple[float | None, float | None]]]:
        """Stored (close, adj_close) per symbol and session."""
        from tickerlake.storage.query import LakeQuery

        try:
            with LakeQuery(self.paths.root) as q:
                df = q.ohlcv(symbols=symbols, start=start, end=end)
        except Exception as exc:
            self.log.debug("could not read stored bars: %s", exc)
            return {}
        if df.empty:
            return {}

        out: dict[str, dict[date, tuple[float | None, float | None]]] = {}
        for row in df.itertuples():
            # DuckDB hands back date32 as Timestamp, which is not equal to the
            # plain date the Tiingo payload parses to -- so keying on it matches
            # nothing while every request still succeeds.
            day = as_date(row.date)
            out.setdefault(row.symbol, {})[day] = (
                _as_float(row.close),
                _as_float(row.adj_close),
            )
        return out

    def _slice(self, run_date: date, per_run: int) -> list[str]:
        if self.symbols_override is not None:
            return self.symbols_override
        members = self.tracker.current_members() if self.tracker else []
        if not members or per_run <= 0:
            return []
        offset = (run_date.toordinal() * per_run) % len(members)
        return (members[offset:] + members[:offset])[:per_run]


def _quality_row(
    run_date: date,
    symbol: str,
    session: date,
    field: str,
    ours: float | None,
    theirs: float | None,
    diff: float | None,
    passed: bool,
    now: datetime,
) -> dict[str, Any]:
    return {
        "run_id": run_date.isoformat(),
        "run_date": run_date,
        "stage": "tiingo",
        "dataset": P.OHLCV,
        "metric": f"{field}_crosscheck:{symbol}:{session.isoformat()}",
        "value_num": diff,
        "value_text": (
            f"session={session} field={field} "
            f"ours={ours if ours is not None else 'NA'} "
            f"tiingo={theirs if theirs is not None else 'NA'}"
        ),
        "passed": passed,
        "recorded_at": now,
    }


def _rel(ours: float | None, theirs: float | None) -> float | None:
    if ours is None or theirs is None or ours == 0:
        return None
    return abs(theirs - ours) / abs(ours)


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if pd.notna(out) else None
