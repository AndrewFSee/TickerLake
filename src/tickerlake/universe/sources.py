"""External sources for index constituents, current and historical.

Three sources, deliberately:

* ``fetch_seed_intervals`` -- fja05680/sp500's ``sp500_ticker_start_end.csv``.
  Already interval-shaped and, importantly, models index *re-entry* correctly
  (AAL appears twice: 1996-1997 and 2015-2024). This seeds history.
* ``fetch_live_constituents`` -- Wikipedia. Updated same-day, so this is what
  detects today's adds and drops. Also carries sector, sub-industry and CIK.
* ``fetch_snapshot_history`` -- fja05680's dated full-constituent snapshots.
  Only used by ``verify-membership`` to rebuild intervals independently and diff
  them against the seed, so a bad upstream commit does not silently corrupt the
  membership table.

Ticker dialects
---------------
Class shares are spelled differently by different sources: Wikipedia and SEC use
``BRK.B``, Yahoo uses ``BRK-B``. We store the **dot** form as canonical -- it is
what SEC and most vendors use -- and convert at request time with
``to_yahoo_symbol``. Storing one canonical form is what lets OHLCV, options,
filings and membership actually join.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd

from tickerlake.utils.http import HttpClient

log = logging.getLogger(__name__)

USER_AGENT = "TickerLake/0.1 (research data collection; contact via repo)"


def to_yahoo_symbol(symbol: str) -> str:
    """Canonical (dot) ticker -> Yahoo dialect. BRK.B -> BRK-B."""
    return symbol.strip().upper().replace(".", "-")


def from_yahoo_symbol(symbol: str) -> str:
    """Yahoo dialect -> canonical (dot). BRK-B -> BRK.B."""
    return symbol.strip().upper().replace("-", ".")


def normalize_symbol(symbol: str) -> str:
    """Canonicalise a ticker from any source: uppercase, trimmed, dot-form."""
    return str(symbol).strip().upper().replace("-", ".").replace(" ", "")


@dataclass
class LiveConstituent:
    symbol: str
    company_name: str | None = None
    gics_sector: str | None = None
    gics_sub_industry: str | None = None
    cik: str | None = None
    date_added: date | None = None


# ------------------------------------------------------------------- seeding


def fetch_seed_intervals(url: str, client: HttpClient | None = None) -> pd.DataFrame:
    """Historical membership intervals.

    Returns columns: symbol, start_date, end_date (NaT = still a member).
    """
    own = client is None
    client = client or HttpClient(requests_per_second=2.0, user_agent=USER_AGENT)
    try:
        text = client.get_text(url)
    finally:
        if own:
            client.close()

    df = pd.read_csv(io.StringIO(text))
    expected = {"ticker", "start_date", "end_date"}
    if not expected.issubset(df.columns):
        raise ValueError(
            f"membership seed at {url} has unexpected columns {list(df.columns)}; expected {expected}"
        )

    out = pd.DataFrame(
        {
            "symbol": df["ticker"].map(normalize_symbol),
            "start_date": pd.to_datetime(df["start_date"], errors="coerce").dt.date,
            "end_date": pd.to_datetime(df["end_date"], errors="coerce").dt.date,
        }
    )
    out = out.dropna(subset=["symbol", "start_date"])
    out = out[out["symbol"].str.len() > 0]
    log.info(
        "seed: %d intervals across %d unique symbols (%d currently open)",
        len(out),
        out["symbol"].nunique(),
        int(out["end_date"].isna().sum()),
    )
    return out.reset_index(drop=True)


def fetch_snapshot_history(url: str, client: HttpClient | None = None) -> pd.DataFrame:
    """Dated full-constituent snapshots. Columns: date, symbol."""
    own = client is None
    client = client or HttpClient(requests_per_second=2.0, user_agent=USER_AGENT)
    try:
        text = client.get_text(url)
    finally:
        if own:
            client.close()

    df = pd.read_csv(io.StringIO(text))
    if not {"date", "tickers"}.issubset(df.columns):
        raise ValueError(f"snapshot history at {url} has unexpected columns {list(df.columns)}")

    rows = []
    for _, row in df.iterrows():
        snap = pd.to_datetime(row["date"], errors="coerce")
        if pd.isna(snap):
            continue
        tickers = str(row["tickers"]).split(",")
        for t in tickers:
            sym = normalize_symbol(t)
            if sym:
                rows.append({"date": snap.date(), "symbol": sym})
    out = pd.DataFrame(rows)
    log.info("snapshot history: %d dated snapshots, %d symbol-days", df.shape[0], len(out))
    return out


def intervals_from_snapshots(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Rebuild membership intervals from dated snapshots, independently of the seed.

    A symbol's interval runs from the snapshot where it first appears until the
    last snapshot where it is still present. A gap re-opens a new interval, which
    is what makes index re-entry come out right.
    """
    if snapshots.empty:
        return pd.DataFrame(columns=["symbol", "start_date", "end_date"])

    dates = sorted(snapshots["date"].unique())
    last_date = dates[-1]
    by_date = {d: set(g["symbol"]) for d, g in snapshots.groupby("date")}

    open_start: dict[str, date] = {}
    prev_seen: dict[str, date] = {}
    intervals: list[dict] = []

    for d in dates:
        present = by_date[d]
        for sym in present:
            if sym not in open_start:
                open_start[sym] = d
            prev_seen[sym] = d
        # Anything with an open interval but absent from this snapshot has left.
        for sym in [s for s in open_start if s not in present]:
            intervals.append(
                {"symbol": sym, "start_date": open_start.pop(sym), "end_date": prev_seen[sym]}
            )

    for sym, start in open_start.items():
        # Still present in the newest snapshot: interval stays open.
        intervals.append({"symbol": sym, "start_date": start, "end_date": None})

    out = pd.DataFrame(intervals).sort_values(["symbol", "start_date"]).reset_index(drop=True)
    log.info("rebuilt %d intervals from snapshots (newest snapshot %s)", len(out), last_date)
    return out


# ---------------------------------------------------------------------- live


def fetch_live_constituents(url: str, client: HttpClient | None = None) -> list[LiveConstituent]:
    """Today's constituents from Wikipedia, with sector/CIK metadata.

    Fetched through our own client rather than ``pd.read_html(url)`` so the
    request carries a real User-Agent -- Wikipedia rejects some default clients.
    """
    own = client is None
    client = client or HttpClient(requests_per_second=1.0, user_agent=USER_AGENT)
    try:
        html = client.get_text(url)
    finally:
        if own:
            client.close()

    tables = pd.read_html(io.StringIO(html))
    table = _pick_constituents_table(tables)
    if table is None:
        raise ValueError(f"no constituents table found at {url} ({len(tables)} tables parsed)")

    cols = {str(c).strip().lower(): c for c in table.columns}

    def col(*names: str):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    sym_col = col("symbol", "ticker", "ticker symbol")
    if sym_col is None:
        raise ValueError(f"constituents table has no symbol column: {list(table.columns)}")

    name_col = col("security", "company", "name")
    sector_col = col("gics sector", "sector")
    sub_col = col("gics sub-industry", "gics sub industry", "sub-industry")
    cik_col = col("cik")
    added_col = col("date added", "date first added")

    out: list[LiveConstituent] = []
    for _, row in table.iterrows():
        symbol = normalize_symbol(row[sym_col])
        if not symbol or symbol.lower() == "nan":
            continue
        out.append(
            LiveConstituent(
                symbol=symbol,
                company_name=_clean_str(row[name_col]) if name_col else None,
                gics_sector=_clean_str(row[sector_col]) if sector_col else None,
                gics_sub_industry=_clean_str(row[sub_col]) if sub_col else None,
                cik=_clean_cik(row[cik_col]) if cik_col else None,
                date_added=_clean_date(row[added_col]) if added_col else None,
            )
        )

    # A parse that yields far too few names means the page structure changed;
    # better to fail loudly than to silently "remove" 400 symbols from the index.
    if len(out) < 400:
        raise ValueError(
            f"live universe parse returned only {len(out)} symbols from {url}; "
            "refusing to treat this as a valid S&P 500 snapshot"
        )

    log.info("live universe: %d constituents from Wikipedia", len(out))
    return out


def _pick_constituents_table(tables: list[pd.DataFrame]) -> pd.DataFrame | None:
    """Choose the constituents table by shape, not by index.

    Wikipedia reorders tables periodically; hardcoding ``tables[0]`` is how this
    breaks silently six months from now.
    """
    best, best_score = None, -1
    for t in tables:
        lower = {str(c).strip().lower() for c in t.columns}
        if not ({"symbol", "ticker"} & lower):
            continue
        score = len(t) + (50 if "gics sector" in lower else 0) + (25 if "cik" in lower else 0)
        if len(t) >= 400 and score > best_score:
            best, best_score = t, score
    return best


# ------------------------------------------------------------------- helpers


def _clean_str(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _clean_cik(value) -> str | None:
    text = _clean_str(value)
    if not text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(10) if digits else None


def _clean_date(value) -> date | None:
    ts = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(ts) else ts.date()
