"""Marketaux: financial news with per-entity sentiment and tagging confidence.

What it adds over GDELT
-----------------------
GDELT is broad, free and unlimited, but it is a general-purpose news graph: it
has no notion of a ticker, so symbol tagging here had to be reconstructed from
organisation salience offsets, and its tone is a document-level score covering
whatever the article is about.

Marketaux is built for markets and supplies both missing pieces directly:

* ``sentiment_score`` **per entity**, so an article discussing Apple favourably
  and a supplier unfavourably yields a different score for each rather than one
  blended document tone.
* ``match_score``, the publisher's own confidence that the article really is
  about that symbol -- the exact quantity the GDELT offset heuristic
  approximates.

Free-tier sizing
----------------
The free tier is the tightest of any source here: ~100 requests/day returning 3
articles each. That is a few hundred articles daily against GDELT's thousands,
so this is deliberately *not* a replacement. GDELT provides breadth; Marketaux
provides a smaller, cleanly-tagged, sentiment-scored stream over a rotating
slice of the universe. Both land in ``news_events`` distinguished by ``source``.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd

from tickerlake.fetchers.base import BaseFetcher, FetchResult
from tickerlake.storage import paths as P
from tickerlake.universe.sources import to_yahoo_symbol
from tickerlake.utils.http import HttpClient

SOURCE = "marketaux"
NEWS_URL = "https://api.marketaux.com/v1/news/all"

# Below this, Marketaux's own tagging confidence is too low to treat the article
# as being about the symbol. Their scores run well above 1 for solid matches.
MIN_MATCH_SCORE = 10.0


class MarketauxFetcher(BaseFetcher):
    """Symbol-tagged, sentiment-scored financial news."""

    name = "marketaux"
    dataset = P.NEWS_EVENTS
    requires_secret = "marketaux_api_key"

    def collect(self, run_date: date, result: FetchResult) -> None:
        token = self.config.secrets.marketaux_api_key
        # Batch symbols per request: the daily request cap is the binding
        # constraint, so asking about several tickers at once buys coverage.
        batch_size = int(self.cfg("symbols_per_request", 20))
        requests_budget = int(self.cfg("requests_per_run", 20))
        per_request = int(self.cfg("articles_per_request", 3))

        symbols = self._slice(run_date, batch_size * requests_budget)
        if not symbols:
            result.add_warning("no symbols to query")
            return

        lookback = int(self.cfg("lookback_days", 1))
        published_after = (run_date - timedelta(days=lookback)).isoformat() + "T00:00"
        min_match = float(self.cfg("min_match_score", MIN_MATCH_SCORE))

        client = HttpClient(
            requests_per_second=float(self.cfg("requests_per_second", 1.0)),
            user_agent="TickerLake/0.1",
            max_attempts=2,
        )
        now = datetime.now(UTC)
        rows: list[dict[str, Any]] = []
        batches = [symbols[i : i + batch_size] for i in range(0, len(symbols), batch_size)][
            :requests_budget
        ]

        try:
            for batch in batches:
                try:
                    payload = client.get_json(
                        NEWS_URL,
                        params={
                            "symbols": ",".join(to_yahoo_symbol(s) for s in batch),
                            "filter_entities": "true",
                            "language": self.cfg("language", "en"),
                            "published_after": published_after,
                            "limit": per_request,
                            "api_token": token,
                        },
                    )
                    result.items_succeeded += 1
                except Exception as exc:
                    result.items_failed += 1
                    result.add_warning(f"marketaux batch: {type(exc).__name__}: {exc}")
                    continue

                rows.extend(self._parse(payload, run_date, min_match, now))
        finally:
            client.close()

        if not rows:
            result.add_warning("no articles returned")
            return

        df = pd.DataFrame(rows).drop_duplicates(subset=["event_id"])
        write = self.writer.write(
            df, P.NEWS_EVENTS, self.paths.news_file(run_date, SOURCE), mode="merge"
        )
        result.record_write(write)

        result.details.update(
            {
                "articles": len(df),
                "symbols_tagged": int(df["symbol"].nunique()),
                "mean_sentiment": round(float(df["sentiment"].mean()), 4),
                "requests_used": result.items_succeeded,
            }
        )
        self.log.info(
            "%d tagged articles across %d symbols, mean sentiment %.3f (%d requests)",
            len(df),
            df["symbol"].nunique(),
            df["sentiment"].mean(),
            result.items_succeeded,
        )

    # --------------------------------------------------------------- parsing

    def _parse(
        self, payload: dict, run_date: date, min_match: float, now: datetime
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for article in (payload or {}).get("data", []) or []:
            url = article.get("url") or ""
            uuid = article.get("uuid") or url
            if not url:
                continue
            published = _to_datetime(article.get("published_at"))

            # One row per tagged entity, not per article: an article about a
            # merger genuinely concerns both companies, and each gets its own
            # sentiment. Collapsing to one row would discard that.
            for entity in article.get("entities") or []:
                if str(entity.get("type", "")).lower() != "equity":
                    continue
                match = _as_float(entity.get("match_score"))
                if match is not None and match < min_match:
                    continue
                symbol = str(entity.get("symbol") or "").strip().upper().replace("-", ".")
                if not symbol:
                    continue

                sentiment = _as_float(entity.get("sentiment_score"))
                out.append(
                    {
                        "event_id": hashlib.sha1(f"{uuid}|{symbol}".encode()).hexdigest(),
                        "published_at": published,
                        "date": published.date() if published else run_date,
                        "symbol": symbol,
                        "title": article.get("title"),
                        "summary": article.get("description") or article.get("snippet"),
                        "url": url,
                        "domain": article.get("source"),
                        "language": article.get("language"),
                        "country": entity.get("country"),
                        "theme": entity.get("industry"),
                        # No document-level tone from this source; the per-entity
                        # score is already on the shared -1..+1 scale.
                        "tone": None,
                        "sentiment": sentiment,
                        "match_score": match,
                        "category": "company",
                        "embedding_status": "pending",
                        "source": SOURCE,
                        "ingested_at": now,
                    }
                )
        return out

    def _slice(self, run_date: date, count: int) -> list[str]:
        if self.symbols_override is not None:
            return self.symbols_override
        members = self.tracker.current_members() if self.tracker else []
        if not members or count <= 0:
            return []
        offset = (run_date.toordinal() * count) % len(members)
        return (members[offset:] + members[:offset])[:count]


def _to_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return pd.to_datetime(value, utc=True).to_pydatetime()
    except Exception:
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
