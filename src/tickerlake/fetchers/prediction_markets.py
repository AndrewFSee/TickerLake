"""Prediction markets on US data releases, the Fed, indices and single stocks.

Kalshi (CFTC-regulated) and Polymarket both publish their markets through public,
keyless read APIs. What they say about a number before it is released -- the
odds of each CPI print, payrolls bucket or Fed outcome -- exists only while the
market is open, so the daily snapshot this stage keeps is the only record of how
those odds moved, exactly like the analyst estimate snapshots.

Kalshi: every open event of the series listed under ``econ_calendar.releases`` plus
``prediction_markets.kalshi.extra_series``. An event is a set of binary markets,
either a strike ladder ("Above 0.3%") or named outcomes ("Hike 25bps").

Polymarket: every open event under the configured tags, minus titles matching
``exclude`` (other countries' data) and events below ``min_volume``. Its strike
labels are ``$140`` (close above), ``↑ $150`` / ``↓ $120`` (touch before expiry).

``prob`` is the bid/ask midpoint when both sides exist and the spread is at most
``max_spread``, else the last trade (Kalshi) or Polymarket's displayed price.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd

from tickerlake.fetchers.base import BaseFetcher, FetchResult
from tickerlake.storage import paths as P
from tickerlake.utils.http import HttpClient

KALSHI_URL = "https://api.elections.kalshi.com/trade-api/v2"
POLYMARKET_URL = "https://gamma-api.polymarket.com"
MAX_SPREAD = 0.15
_NUM = r"-?\$?[\d,]*\.?\d+\s*[km]?"
_RANGE = re.compile(rf"^({_NUM})\s*[-–]\s*({_NUM})%?$", re.I)
_LESS = re.compile(rf"^(?:<|under|below|less than)\s*({_NUM})", re.I)
_MORE = re.compile(rf"^(?:>|over|above|more than)\s*({_NUM})|^({_NUM})\s*\+$", re.I)
_PLAIN = re.compile(rf"^({_NUM})$", re.I)


def _f(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def _ts(value: Any) -> pd.Timestamp | None:
    if not value:
        return None
    t = pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def mid_prob(bid: float | None, ask: float | None, fallback: float | None, max_spread: float = MAX_SPREAD) -> float | None:
    """Midpoint of a sane quote, else the fallback price."""
    if bid is not None and ask is not None and 0 <= ask - bid <= max_spread:
        return (bid + ask) / 2
    return fallback


def kalshi_rows(events: list[dict], series: str, snapshot: date, now: datetime) -> list[dict]:
    rows = []
    for e in events:
        for m in e.get("markets", []):
            if m.get("status") not in (None, "active", "open"):
                continue
            bid, ask, last = _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars")), _f(m.get("last_price_dollars"))
            rows.append({
                "snapshot_date": snapshot, "snapshot_at": now, "venue": "kalshi", "series": series,
                "exclusive": bool(e.get("mutually_exclusive")),
                "event_id": e["event_ticker"], "event_title": e.get("title"),
                "market_id": m["ticker"], "outcome": m.get("yes_sub_title") or m.get("subtitle"),
                "strike_type": m.get("strike_type"),
                "floor_strike": _f(m.get("floor_strike")), "cap_strike": _f(m.get("cap_strike")),
                "close_time": _ts(m.get("close_time")),
                "yes_bid": bid, "yes_ask": ask, "last_price": last, "prob": mid_prob(bid, ask, last),
                "volume": _f(m.get("volume_fp")), "volume_24h": _f(m.get("volume_24h_fp")),
                "open_interest": _f(m.get("open_interest_fp")), "liquidity": None,
                "url": f"https://kalshi.com/markets/{series.lower()}",
            })
    return rows


def _num(text: str) -> float:
    t = text.replace("$", "").replace(",", "").strip().lower()
    mult = 1e3 if t.endswith("k") else 1e6 if t.endswith("m") else 1
    return float(t.rstrip("km").strip()) * mult


def polymarket_strike(label: str | None, event_title: str) -> tuple[str, float | None, float | None]:
    """(strike_type, floor, cap) from a Polymarket outcome label: "↑ $150" / "↓ $120"
    (touch), "$700-$720" (range), "<$600", ">$780" or "150k+" (open tails), and a bare
    "$140" in a close-above event."""
    label = (label or "").strip()
    if label[:1] in ("↑", "↓"):
        m = _PLAIN.match(label[1:].strip())
        return ("touch_above" if label[0] == "↑" else "touch_below"), (_num(m.group(1)) if m else None), None
    if m := _RANGE.match(label):
        return "between", _num(m.group(1)), _num(m.group(2))
    if m := _LESS.match(label):
        return "less", None, _num(m.group(1))
    if m := _MORE.match(label):
        return "greater", _num(m.group(1) or m.group(2)), None
    if (m := _PLAIN.match(label)) and "$" in label and re.search(r"above|close|finish", event_title, re.I):
        return "greater", _num(m.group(1)), None
    return "custom", None, None


def polymarket_rows(events: list[dict], tag: str, snapshot: date, now: datetime) -> list[dict]:
    rows = []
    for e in events:
        for m in e.get("markets", []):
            if m.get("closed") or not m.get("active", True):
                continue  # a strike already hit, or a resolved outcome
            try:
                yes = _f(json.loads(m.get("outcomePrices") or "[]")[0])
            except (ValueError, IndexError, TypeError):
                yes = None
            bid, ask = _f(m.get("bestBid")), _f(m.get("bestAsk"))
            kind, floor, cap = polymarket_strike(m.get("groupItemTitle"), e.get("title", ""))
            rows.append({
                "snapshot_date": snapshot, "snapshot_at": now, "venue": "polymarket", "series": tag,
                "exclusive": bool(e.get("negRisk")),
                "event_id": e.get("slug") or str(e.get("id")), "event_title": e.get("title"),
                "market_id": m.get("conditionId") or str(m.get("id")),
                "outcome": m.get("groupItemTitle") or m.get("question"),
                "strike_type": kind, "floor_strike": floor, "cap_strike": cap,
                "close_time": _ts(m.get("endDate") or e.get("endDate")),
                "yes_bid": bid, "yes_ask": ask, "last_price": _f(m.get("lastTradePrice")),
                "prob": mid_prob(bid, ask, yes),
                "volume": _f(m.get("volume")), "volume_24h": _f(m.get("volume24hr")),
                "open_interest": None, "liquidity": _f(m.get("liquidity")),
                "url": f"https://polymarket.com/event/{e.get('slug')}" if e.get("slug") else None,
            })
    return rows


class PredictionMarketsFetcher(BaseFetcher):
    """Daily snapshot of the Kalshi and Polymarket markets we follow."""

    name = "prediction_markets"
    dataset = P.PREDICTION_MARKETS

    def kalshi_series(self) -> list[str]:
        releases = self.config.get("econ_calendar.releases", {}) or {}
        series = [s for r in releases.values() for s in (r.get("kalshi") or [])]
        series += list(self.cfg("kalshi.extra_series", []) or [])
        return list(dict.fromkeys(series))

    def collect(self, run_date: date, result: FetchResult) -> None:
        now = datetime.now(UTC)
        client = HttpClient(requests_per_second=float(self.cfg("requests_per_second", 3.0)),
                            user_agent="Mozilla/5.0 (TickerLake)")
        rows: list[dict] = []
        for series in self.kalshi_series():
            try:
                data = client.get_json(f"{KALSHI_URL}/events", params={
                    "series_ticker": series, "status": "open", "with_nested_markets": "true", "limit": 200})
            except Exception as exc:
                result.add_warning(f"kalshi {series}: {type(exc).__name__}: {exc}")
                continue
            got = kalshi_rows(data.get("events", []), series, run_date, now)
            rows += got
            result.details.setdefault("kalshi_markets", {})[series] = len(got)

        exclude = self.cfg("polymarket.exclude")
        min_volume = float(self.cfg("polymarket.min_volume", 0))
        seen: set[str] = set()
        for tag in self.cfg("polymarket.tags", []) or []:
            events = self._polymarket_events(client, tag, result)
            keep = [e for e in events
                    if e.get("slug") not in seen
                    and float(e.get("volume") or 0) >= min_volume
                    and not (exclude and re.search(exclude, e.get("title", ""), re.I))]
            seen.update(e.get("slug") for e in keep)
            got = polymarket_rows(keep, tag, run_date, now)
            rows += got
            result.details.setdefault("polymarket_events", {})[tag] = len(keep)

        if not rows:
            result.add_error("no prediction markets returned")
            return
        df = pd.DataFrame(rows).drop_duplicates(["venue", "market_id"], keep="first")
        df["source"] = df["venue"]
        df["ingested_at"] = now
        write = self.writer.write(df, P.PREDICTION_MARKETS,
                                  self.paths.prediction_markets_file(run_date.year, run_date.month), mode="merge")
        result.record_write(write)
        result.items_succeeded = len(df)
        self.log.info("prediction markets: %d outcomes (%d Kalshi, %d Polymarket)", len(df),
                      int((df["venue"] == "kalshi").sum()), int((df["venue"] == "polymarket").sum()))

    def _polymarket_events(self, client: HttpClient, tag: str, result: FetchResult) -> list[dict]:
        events: list[dict] = []
        page = 100
        for offset in range(0, int(self.cfg("polymarket.max_events_per_tag", 500)), page):
            try:
                batch = client.get_json(f"{POLYMARKET_URL}/events", params={
                    "tag_slug": tag, "closed": "false", "limit": page, "offset": offset})
            except Exception as exc:
                result.add_warning(f"polymarket {tag}: {type(exc).__name__}: {exc}")
                break
            events += batch
            if len(batch) < page:
                break
        return events
