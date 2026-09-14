"""Finnhub: company news and an independent price cross-check.

Two jobs, both sized to the 60-calls/minute free tier:

* **Company news** for a rotating slice of the universe, stored alongside GDELT
  in ``news_events`` with ``source='finnhub'``.
* **Quote cross-check** on a random sample, compared against the same day's
  yfinance close. Disagreement between two independent sources is the cheapest
  bad-data detector available, and the results land in the ``quality`` dataset
  rather than being silently discarded.

The sample is random per run (not rotating) because the point is to detect
systemic breakage quickly, and a random sample surfaces that faster than a
deterministic cycle.
"""

from __future__ import annotations

import hashlib
import random
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd

from tickerlake.fetchers.base import BaseFetcher, FetchResult
from tickerlake.storage import paths as P
from tickerlake.universe.sources import to_yahoo_symbol
from tickerlake.utils.dates import as_date
from tickerlake.utils.http import HttpClient, HttpError

SOURCE = "finnhub"
BASE = "https://finnhub.io/api/v1"

# Relative close difference above which we record a cross-check mismatch.
PRICE_TOLERANCE = 0.02


class FinnhubFetcher(BaseFetcher):
    """Secondary source: company news plus price validation."""

    name = "finnhub"
    dataset = P.NEWS_EVENTS
    requires_secret = "finnhub_api_key"

    def collect(self, run_date: date, result: FetchResult) -> None:
        token = self.config.secrets.finnhub_api_key
        client = HttpClient(
            requests_per_second=float(self.cfg("requests_per_second", 1.0)),
            user_agent="TickerLake/0.1",
            max_attempts=3,
        )
        try:
            if self.cfg("fetch_company_news", True):
                self._company_news(client, token, run_date, result)
            if self.cfg("fetch_quote_crosscheck", True):
                self._crosscheck(client, token, run_date, result)
        finally:
            client.close()

    # ------------------------------------------------------------------ news

    def _company_news(
        self, client: HttpClient, token: str, run_date: date, result: FetchResult
    ) -> None:
        lookback = int(self.cfg("news_lookback_days", 1))
        symbols = self._slice(run_date, int(self.cfg("news_symbols_per_run", 40)))
        if not symbols:
            return

        frm = (run_date - timedelta(days=lookback)).isoformat()
        to = run_date.isoformat()
        now = datetime.now(UTC)
        rows: list[dict[str, Any]] = []

        for symbol in symbols:
            try:
                payload = client.get_json(
                    f"{BASE}/company-news",
                    params={
                        "symbol": to_yahoo_symbol(symbol),
                        "from": frm,
                        "to": to,
                        "token": token,
                    },
                )
                result.items_succeeded += 1
            except HttpError as exc:
                result.items_failed += 1
                result.add_warning(f"news {symbol}: {exc}")
                continue
            except Exception as exc:
                result.items_failed += 1
                result.add_warning(f"news {symbol}: {type(exc).__name__}: {exc}")
                continue

            for article in payload or []:
                url = article.get("url") or ""
                if not url:
                    continue
                published = _from_epoch(article.get("datetime"))
                rows.append(
                    {
                        "event_id": hashlib.sha1(f"{url}|{symbol}".encode()).hexdigest(),
                        "published_at": published,
                        "date": published.date() if published else run_date,
                        "symbol": symbol,
                        "title": article.get("headline"),
                        "summary": article.get("summary"),
                        "url": url,
                        "domain": article.get("source"),
                        "language": "en",
                        "country": None,
                        "theme": None,
                        # Finnhub supplies no tone; an embedding/sentiment pass
                        # can fill this in later.
                        "tone": None,
                        "sentiment": None,
                        "match_score": None,
                        "category": article.get("category") or "company",
                        "embedding_status": "pending",
                        "source": SOURCE,
                        "ingested_at": now,
                    }
                )

        if rows:
            write = self.writer.write(
                pd.DataFrame(rows),
                P.NEWS_EVENTS,
                self.paths.news_file(run_date, SOURCE),
                mode="merge",
            )
            result.record_write(write)
            result.details["news_articles"] = len(rows)

    # ------------------------------------------------------------ crosscheck

    def _crosscheck(
        self, client: HttpClient, token: str, run_date: date, result: FetchResult
    ) -> None:
        """Compare Finnhub's previous close against our stored close for that session.

        Aligning the *dates* is the whole difficulty here. Finnhub's ``pc`` is the
        close of the session before the quote's own timestamp, so it is always a
        completed session. Our stored OHLCV, by contrast, includes a provisional
        bar for the current day whenever the run happens while the market is open.

        Comparing "our latest row" to ``pc`` therefore compares two different
        sessions -- silently, and with a plausible-looking 0.5-2% drift that reads
        as a data-quality problem when nothing is wrong. So we resolve the session
        ``pc`` refers to from the quote timestamp, and compare against our stored
        close for exactly that date. Both sides are then completed closes, and
        genuine disagreement is the only thing left that can trip the check.
        """
        sample_size = int(self.cfg("crosscheck_sample_size", 50))
        candidates = self._current_members()
        if not candidates or sample_size <= 0:
            return
        sample = random.sample(candidates, min(sample_size, len(candidates)))

        ours = self._our_closes(run_date, sample)
        if not ours:
            result.add_warning("no stored closes to cross-check against")
            return

        now = datetime.now(UTC)
        rows: list[dict[str, Any]] = []
        mismatches = 0
        compared = 0
        unaligned = 0

        for symbol in sample:
            by_date = ours.get(symbol)
            if not by_date:
                continue
            try:
                quote = client.get_json(
                    f"{BASE}/quote", params={"symbol": to_yahoo_symbol(symbol), "token": token}
                )
            except Exception as exc:
                self.log.debug("quote %s failed: %s", symbol, exc)
                continue

            their_close = _safe_float(quote.get("pc"))
            if not their_close or their_close <= 0:
                continue

            # The session `pc` describes: the last one strictly before the quote.
            quote_dt = _from_epoch(quote.get("t"))
            quote_date = quote_dt.date() if quote_dt else run_date
            prior = [d for d in by_date if d < quote_date]
            if not prior:
                unaligned += 1
                continue
            session = max(prior)
            our_close = by_date[session]

            compared += 1
            diff = abs(their_close - our_close) / our_close
            passed = diff <= PRICE_TOLERANCE
            if not passed:
                mismatches += 1
                self.log.warning(
                    "price mismatch %s on %s: yfinance=%.2f finnhub=%.2f (%.2f%%)",
                    symbol,
                    session,
                    our_close,
                    their_close,
                    diff * 100,
                )
            rows.append(
                {
                    "run_id": run_date.isoformat(),
                    "run_date": run_date,
                    "stage": self.name,
                    "dataset": P.OHLCV,
                    "metric": f"close_crosscheck:{symbol}",
                    "value_num": diff,
                    "value_text": (
                        f"session={session} yfinance={our_close:.4f} finnhub={their_close:.4f}"
                    ),
                    "passed": passed,
                    "recorded_at": now,
                }
            )

        if unaligned:
            result.add_warning(
                f"{unaligned} symbol(s) had no stored close before the quote date; "
                "run the ohlcv stage first for a meaningful cross-check"
            )

        if rows:
            write = self.writer.write(
                pd.DataFrame(rows), P.QUALITY, self.paths.quality_file(run_date), mode="merge"
            )
            result.record_write(write)

        result.details["crosscheck"] = {
            "compared": compared,
            "mismatches": mismatches,
            "tolerance": PRICE_TOLERANCE,
        }
        if compared and mismatches / compared > 0.1:
            result.add_warning(
                f"{mismatches}/{compared} closes disagree with Finnhub by >{PRICE_TOLERANCE:.0%} "
                "- check for a split or a bad yfinance day"
            )
        elif compared:
            self.log.info(
                "cross-check: %d/%d closes agree with Finnhub", compared - mismatches, compared
            )

    def _our_closes(self, run_date: date, symbols: list[str]) -> dict[str, dict[date, float]]:
        """Stored closes per symbol, keyed by session date.

        Returns the whole recent window rather than just the newest row, so the
        caller can pick the session that actually lines up with the quote instead
        of assuming the latest stored bar is a completed close.
        """
        from tickerlake.storage.query import LakeQuery

        try:
            with LakeQuery(self.paths.root) as q:
                df = q.ohlcv(symbols=symbols, start=run_date - timedelta(days=10), end=run_date)
        except Exception as exc:
            self.log.debug("could not read stored closes: %s", exc)
            return {}
        if df.empty:
            return {}

        out: dict[str, dict[date, float]] = {}
        for row in df.itertuples():
            if pd.isna(row.close):
                continue
            out.setdefault(row.symbol, {})[as_date(row.date)] = float(row.close)
        return out

    # --------------------------------------------------------------- helpers

    def _current_members(self) -> list[str]:
        if self.symbols_override is not None:
            return list(self.symbols_override)
        if self.tracker is None:
            return []
        return self.tracker.current_members()

    def _slice(self, run_date: date, per_run: int) -> list[str]:
        members = self._current_members()
        if not members or per_run <= 0:
            return []
        offset = (run_date.toordinal() * per_run) % len(members)
        rotated = members[offset:] + members[:offset]
        return rotated[:per_run]


def _from_epoch(value: Any) -> datetime | None:
    try:
        return datetime.fromtimestamp(int(value), tz=UTC)
    except (TypeError, ValueError, OSError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
        return result
    except (TypeError, ValueError):
        return None
