"""Insider transactions (SEC Form 4), via Finnhub's parsed feed.

Form 4 filings are XML ownership documents, not prose, so they do not fit the
EDGAR text pipeline: extracting them would mean writing a second parser for a
different schema. Finnhub already parses Form 4 and exposes it on the free tier,
which is the same data without that work.

The signal worth capturing is **open-market purchases** (transaction code ``P``)
clustering across several insiders at one company. Awards (``A``) and option
exercises (``M``) dominate the raw feed by count and carry almost no
information: they are compensation events on a vesting schedule, not decisions
about price. ``transaction_code`` is stored verbatim so that filtering stays the
caller's choice rather than being baked in here.
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


class InsiderFetcher(BaseFetcher):
    """Form 4 insider transactions for a rotating slice of the universe."""

    name = "insider"
    dataset = P.INSIDER
    requires_secret = "finnhub_api_key"

    def collect(self, run_date: date, result: FetchResult) -> None:
        token = self.config.secrets.finnhub_api_key
        symbols = self._slice(run_date, int(self.cfg("symbols_per_run", 60)))
        if not symbols:
            result.add_warning("no symbols to fetch")
            return

        lookback = int(self.cfg("lookback_days", 180))
        frm = (run_date - timedelta(days=lookback)).isoformat()
        client = HttpClient(
            requests_per_second=float(self.cfg("requests_per_second", 1.0)),
            user_agent="TickerLake/0.1",
            max_attempts=3,
        )
        now = datetime.now(UTC)
        rows: list[dict[str, Any]] = []

        try:
            for symbol in symbols:
                try:
                    payload = client.get_json(
                        f"{BASE}/stock/insider-transactions",
                        params={
                            "symbol": to_yahoo_symbol(symbol),
                            "from": frm,
                            "to": run_date.isoformat(),
                            "token": token,
                        },
                    )
                    result.items_succeeded += 1
                except Exception as exc:
                    result.items_failed += 1
                    self.log.debug("insider %s: %s", symbol, exc)
                    continue

                for entry in (payload or {}).get("data", []) or []:
                    rows.append(
                        {
                            "symbol": symbol,
                            "insider_name": entry.get("name"),
                            "transaction_date": _to_date(entry.get("transactionDate")),
                            "filing_date": _to_date(entry.get("filingDate")),
                            "transaction_code": entry.get("transactionCode"),
                            # `share` is the post-trade holding; `change` is the
                            # trade itself. Conflating them overstates notionals
                            # by orders of magnitude for large holders.
                            "shares_held_after": _as_float(entry.get("share")),
                            "shares_transacted": _as_float(entry.get("change")),
                            "transaction_price": _as_float(entry.get("transactionPrice")),
                            "transaction_value": _notional(
                                entry.get("change"), entry.get("transactionPrice")
                            ),
                            "accession_number": entry.get("id"),
                            "source": SOURCE,
                            "ingested_at": now,
                        }
                    )
        finally:
            client.close()

        if not rows:
            result.add_warning("no insider transactions returned")
            return

        df = pd.DataFrame(rows)
        for year, group in df.groupby(
            pd.to_datetime(df["filing_date"], errors="coerce").dt.year.fillna(run_date.year)
        ):
            write = self.writer.write(
                group, P.INSIDER, self.paths.insider_file(int(year)), mode="merge"
            )
            result.record_write(write)

        purchases = int((df["transaction_code"] == "P").sum())
        sales = int((df["transaction_code"] == "S").sum())
        buy_value = float(
            df.loc[df["transaction_code"] == "P", "transaction_value"].fillna(0).sum()
        )
        result.details.update(
            {
                "transactions": len(df),
                "symbols_with_activity": int(df["symbol"].nunique()),
                "open_market_purchases": purchases,
                "open_market_sales": sales,
                "open_market_buy_value": round(buy_value, 2),
            }
        )
        self.log.info(
            "%d transactions across %d symbols (%d open-market buys, %d sells)",
            len(df),
            df["symbol"].nunique(),
            purchases,
            sales,
        )

    def _slice(self, run_date: date, per_run: int) -> list[str]:
        if self.symbols_override is not None:
            return self.symbols_override
        members = self.tracker.current_members() if self.tracker else []
        if not members or per_run <= 0:
            return []
        offset = (run_date.toordinal() * per_run) % len(members)
        return (members[offset:] + members[:offset])[:per_run]


def _notional(shares: Any, price: Any) -> float | None:
    """Value of the trade itself, from shares *transacted*."""
    n, p = _as_float(shares), _as_float(price)
    if n is None or p is None:
        return None
    return n * p


def _to_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
