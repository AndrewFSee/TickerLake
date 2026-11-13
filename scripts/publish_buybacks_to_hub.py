"""Publish share repurchases and estimated buyback blackout windows into the Dashboard Hub.

The hub's ``run_jobs.py`` runs this with *this* project's venv and the hub folder on
``PYTHONPATH``. It reads the lake only, so ``--demo`` changes nothing. Method and caveats are
in ``tickerlake.analytics.buybacks``.

Inputs: SEC facts (``filings_facts``: buyback cash flow, remaining authorization, diluted share
count), split history (``ohlcv``), each quarter's release date (``earnings_releases``, from
``scripts/backfill_earnings_releases.py``, topped up with the ``announcements`` stage's dated
surprises), expected report dates (Finnhub calendar rows in ``earnings``) and members.

Writes to ``<HUB_DATA_DIR>/buybacks/``:

* ``stocks``          current members: last quarter's and trailing-year buybacks, buyback yield,
                      1-year change in share count, remaining authorization (and as % of cap),
                      when an increase was last reported, blackout status, reopen / shut dates,
                      next release and where that date comes from
* ``blackout_share``  per business day since 2009 and 90 days ahead: share of members (and of
                      today's market cap) estimated in blackout; ``projected`` marks future days
* ``summary``         (JSON) as-of date, totals, coverage, assumptions
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tickerlake.analytics import buybacks as B  # noqa: E402
from tickerlake.storage import paths as P  # noqa: E402
from tickerlake.storage.query import LakeQuery  # noqa: E402

try:
    from core import store  # the hub's storage module (hub folder is on PYTHONPATH)
except ImportError:
    sys.exit("Could not import the hub's core.store: run this through market-compass's run_jobs.py, "
             "or put the market-compass folder on PYTHONPATH.")

MODULE = "buybacks"
DATA = PROJECT_ROOT / "data"
RELEASES = DATA / P.EARNINGS_RELEASES / "earnings_releases.parquet"
HISTORY_START = "2010-01-04"  # the membership history starts in 2010
AHEAD_DAYS = 90


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="accepted for the hub; this script never fetches")
    parser.parse_args()
    if not RELEASES.exists():
        sys.exit("No release history: run `python scripts/backfill_earnings_releases.py` in TickerLake first.")

    concepts = ",".join(f"'{c}'" for c in B.CONCEPTS)
    with LakeQuery(DATA) as q:
        facts = q.sql(f"SELECT symbol, concept, unit, value, start_date, end_date, filed_date FROM filings_facts "
                      f"WHERE concept IN ({concepts}) AND form_type IN ('10-K','10-Q','10-K/A','10-Q/A')")
        splits = q.sql("SELECT symbol, date, stock_splits FROM ohlcv WHERE stock_splits > 0")
        members = q.sql("SELECT symbol, start_date, end_date_eff, is_current, gics_sector, company_name "
                        "FROM membership_pit WHERE index_name = 'SP500'")
        earn = q.sql(f"SELECT symbol, record_type, period, announcement_date, fiscal_period_end FROM {P.EARNINGS} "
                     "WHERE record_type IN ('calendar', 'surprise')")
        last_day = pd.Timestamp(q.sql("SELECT max(date) AS d FROM ohlcv")["d"].iloc[0])

    # release dates: the backfill, plus quarters the announcements stage has dated since
    rel = pd.read_parquet(RELEASES)
    dated = earn[(earn["record_type"] == "surprise") & earn["announcement_date"].notna() & earn["fiscal_period_end"].notna()]
    newer = pd.DataFrame({"symbol": dated["symbol"], "fiscal_close": pd.to_datetime(dated["fiscal_period_end"]),
                          "release_date": pd.to_datetime(dated["announcement_date"]), "report_filed": pd.NaT})
    have = set(zip(rel["symbol"], rel["fiscal_close"]))
    newer = newer[[(s, c) not in have for s, c in zip(newer["symbol"], newer["fiscal_close"])]]
    rel = pd.concat([rel, newer], ignore_index=True)
    cal = earn.loc[earn["record_type"] == "calendar", ["symbol", "period"]].rename(columns={"period": "date"})
    today = last_day.normalize()
    windows = B.blackout_windows(B.expected_releases(rel, cal, today, AHEAD_DAYS + 60))
    log(f"Windows for {windows['symbol'].nunique()} companies ({len(rel):,} quarters, {len(newer)} added from the "
        f"announcements stage).")

    # daily share in blackout: point-in-time members; future days use today's members
    days = pd.bdate_range(HISTORY_START, today + pd.Timedelta(days=AHEAD_DAYS))
    shut = B.blackout_matrix(windows, days)
    members["start_date"] = pd.to_datetime(members["start_date"])
    members["end"] = pd.to_datetime(members["end_date_eff"]).where(~members["is_current"].astype(bool))
    member = pd.DataFrame(False, index=days, columns=shut.columns)
    for r in members[members["symbol"].isin(shut.columns)].itertuples():
        member.loc[(days >= r.start_date) & ((days < r.end) if pd.notna(r.end) else True), r.symbol] = True
    member &= B.covered(windows, days)
    current = members[members["is_current"].astype(bool)].drop_duplicates("symbol").set_index("symbol")
    try:
        caps = store.load_frame("market", "market_caps")["market_cap"]
    except Exception:  # noqa: BLE001 - optional input
        caps = None
    share = pd.DataFrame({"share_count": B.blackout_share(shut, member)})
    if caps is not None:
        share["share_cap"] = B.blackout_share(shut, member, caps)
    share["projected"] = share.index > today
    share = share.reset_index(names="date")

    # per company
    rep = B.repurchase_table(facts, splits).set_index("symbol")
    status = B.status_on(windows, today).set_index("symbol")
    st = current[["company_name", "gics_sector"]].rename(columns={"company_name": "name", "gics_sector": "sector"})
    st = st.join(rep, how="left").join(status, how="left")
    st["mcap"] = st.index.map(caps) if caps is not None else np.nan
    stale = (today - pd.to_datetime(st["quarter_end"])).dt.days > B.STALE_DAYS
    for c in ("repurchased_q", "repurchased_ttm", "share_change_1y"):
        st.loc[stale, c] = np.nan
    old_auth = (today - pd.to_datetime(st["remaining_as_of"])).dt.days > B.STALE_DAYS
    st.loc[old_auth, "remaining_auth"] = np.nan
    st.loc[(today - pd.to_datetime(st["auth_added_reported"])).dt.days > 730, "auth_added_reported"] = pd.NaT
    st["buyback_yield"] = st["repurchased_ttm"] / st["mcap"]
    st["remaining_pct_cap"] = st["remaining_auth"] / st["mcap"]
    stocks = st.reset_index(names="symbol").sort_values("mcap", ascending=False)

    now = share[share["date"] == share.loc[~share["projected"], "date"].max()].iloc[0]
    store.save_frame(MODULE, "stocks", stocks.reset_index(drop=True))
    store.save_frame(MODULE, "blackout_share", share)
    store.save_json(MODULE, "summary", {
        "as_of": str(today.date()), "members": int(len(current)),
        "with_buybacks": int(stocks["repurchased_ttm"].notna().sum()),
        "buybacks_ttm": float(stocks["repurchased_ttm"].sum()),
        "remaining_auth": float(stocks["remaining_auth"].sum()),
        "with_auth": int(stocks["remaining_auth"].notna().sum()),
        "blackout_share_count": float(now["share_count"]),
        "blackout_share_cap": float(now["share_cap"]) if "share_cap" in now and pd.notna(now["share_cap"]) else None,
        "shut_days_before_close": B.SHUT_DAYS_BEFORE_CLOSE, "reopen_bdays": B.REOPEN_BDAYS,
        "release_sources": {k: int(v) for k, v in stocks["release_source"].value_counts().items()},
        "history_start": HISTORY_START,
    })
    log(f"Published buybacks for {len(stocks)} members ({today.date()}): ${stocks['repurchased_ttm'].sum() / 1e9:,.0f}bn "
        f"over the last year; {now['share_count']:.0%} of members estimated in blackout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
