"""Backfill earnings-release dates for every company in ``filings_facts`` from SEC submissions.

The ``announcements`` stage dates only the quarters that have a stored surprise (Finnhub, since
late 2024). Questions about the earnings cycle further back -- e.g. when companies are in their
buyback blackout -- need the release date of every quarter, so this reads each company's whole
submissions history (the ``recent`` block plus the older ``files`` pages) and applies the same
rules as ``AnnouncementFetcher``:

* the fiscal close of each quarter is the ``reportDate`` of its 10-Q / 10-K;
* the release is the *last* 8-K carrying Item 2.02 at least ``MIN_DAYS`` after the close and no
  later than the periodic report (+2 days of EDGAR stamping slack), at most ``MAX_DAYS`` out.

Quarters with no matching 8-K keep ``release_date`` empty and carry ``report_filed`` (the 10-Q /
10-K filing date) so callers can choose a fallback. Writes
``data/earnings_releases/earnings_releases.parquet`` (one row per symbol and fiscal close).

    python scripts/backfill_earnings_releases.py            # everything (~10 min at 8 requests/s)
    python scripts/backfill_earnings_releases.py --symbols AAPL,JPM
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from tickerlake.config import load_config  # noqa: E402
from tickerlake.fetchers.announcements import AnnouncementFetcher  # noqa: E402
from tickerlake.storage import paths as P  # noqa: E402
from tickerlake.storage.query import LakeQuery  # noqa: E402
from tickerlake.utils.http import HttpClient  # noqa: E402

DATA = PROJECT_ROOT / "data"
OUT = DATA / P.EARNINGS_RELEASES / "earnings_releases.parquet"
SUBMISSIONS = "https://data.sec.gov/submissions/{name}"
SINCE = pd.Timestamp("2008-01-01").date()
MIN_DAYS, MAX_DAYS, SLACK_DAYS = 7, 75, 2


def merged_filings(client: HttpClient, cik: str) -> dict[str, list]:
    """The company's full filing list: the ``recent`` block plus every older page."""
    payload = client.get_json(SUBMISSIONS.format(name=f"CIK{cik}.json"))
    filings = payload.get("filings") or {}
    blocks = [filings.get("recent") or {}]
    for page in filings.get("files") or []:
        blocks.append(client.get_json(SUBMISSIONS.format(name=page["name"])))
    keys = ("form", "items", "filingDate", "reportDate", "accessionNumber")
    out: dict[str, list] = {k: [] for k in keys}
    for b in blocks:
        n = len(b.get("form") or [])
        for k in keys:
            vals = list(b.get(k) or [])
            out[k].extend(vals + [""] * (n - len(vals)))
    return out


def releases(symbol: str, cik: str, filings: dict[str, list]) -> list[dict]:
    closes = AnnouncementFetcher._quarter_closes(filings)
    filed = AnnouncementFetcher._report_filed(filings)
    by_day: dict = {}
    for day, acc in AnnouncementFetcher._parse_202(filings):
        by_day.setdefault(day, acc)
    days = sorted(by_day)
    rows, claimed = [], set()
    for close, fq in closes:
        if close < SINCE:
            continue
        hi = close + timedelta(days=MAX_DAYS)
        if close in filed:
            hi = min(hi, filed[close] + timedelta(days=SLACK_DAYS))
        cands = [d for d in days if close + timedelta(days=MIN_DAYS) <= d <= hi and d not in claimed]
        pick = cands[-1] if cands else None
        if pick is not None:
            claimed.add(pick)
        rows.append({"symbol": symbol, "cik": cik, "fiscal_close": close, "fiscal_quarter": fq,
                     "release_date": pick, "release_accession": by_day.get(pick) if pick else None,
                     "report_filed": filed.get(close)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", help="comma-separated subset (merged into the existing file)")
    parser.add_argument("--rps", type=float, default=8.0, help="SEC requests per second (their limit is 10)")
    args = parser.parse_args()

    config = load_config()
    agent = config.secrets.sec_user_agent
    if not agent:
        sys.exit("No SEC user agent configured (secrets.sec_user_agent); SEC requires one.")
    with LakeQuery(DATA) as q:
        ciks = q.sql("SELECT symbol, any_value(cik) AS cik FROM filings_facts GROUP BY 1 ORDER BY 1")
    ciks["cik"] = ciks["cik"].astype(str).str.zfill(10)
    if args.symbols:
        ciks = ciks[ciks["symbol"].isin({s.strip().upper() for s in args.symbols.split(",")})]

    client = HttpClient(requests_per_second=args.rps, user_agent=agent, max_attempts=3, timeout=90)
    rows, failed, t0 = [], [], time.time()
    for i, r in enumerate(ciks.itertuples(), 1):
        try:
            rows += releases(r.symbol, r.cik, merged_filings(client, r.cik))
        except Exception as exc:  # noqa: BLE001 - one company's failure shouldn't stop the rest
            failed.append(f"{r.symbol}: {exc}")
        if i % 50 == 0:
            print(f"  {i}/{len(ciks)} companies ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    for c in ("fiscal_close", "release_date", "report_filed"):
        df[c] = pd.to_datetime(df[c])
    if args.symbols and OUT.exists():
        old = pd.read_parquet(OUT)
        df = pd.concat([old[~old["symbol"].isin(df["symbol"])], df], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    df.sort_values(["symbol", "fiscal_close"]).to_parquet(tmp, index=False)
    tmp.replace(OUT)
    dated = df["release_date"].notna().mean()
    print(f"Wrote {len(df):,} quarters for {df['symbol'].nunique()} companies; {dated:.0%} matched to an Item 2.02 "
          f"release ({time.time() - t0:.0f}s). {len(failed)} failed" + (": " + "; ".join(failed[:5]) if failed else "."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
