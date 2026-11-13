"""Publish NYSE-wide breadth from the lake into the Dashboard Hub (market-compass).

The hub's ``run_jobs.py`` runs this with *this* project's venv and puts the hub folder on
``PYTHONPATH``. It reads the lake only (the daily ``listings`` and ``listed_ohlcv`` stages
fetch), and computes breadth with the hub's own ``modules.market.indicators.breadth`` so
the NYSE and S&P 500 panels are the same calculation on different universes.

Prices are dividend-adjusted (``LakeQuery.adjusted_ohlcv``, rebuilt from the raw close and
the stored dividends). The NYSE is full of high-payout closed-end funds, REITs and MLPs whose
raw price falls by every distribution: on raw closes an 8% payer sits about 5% "below" its
200-day average when its total return hasn't moved. Measured 2026-11-12, % above the
20/50/200-day: 50.8 / 47.8 / 41.9 on raw closes, 51.8 / 50.1 / 46.2 adjusted (S&P 500
members: 66.9 / 70.6 / 65.1).

Universe: NYSE common stock (ADRs, REITs, closed-end funds and partnerships included;
preferreds, notes, warrants, units, rights and ETFs excluded), as classified by the
``listings`` stage. From the first listings snapshot on, each day counts the stocks listed
that day. Before it, the stocks listed on that first snapshot: names that delisted earlier
are missing, so older history is survivorship-biased (it flatters breadth a little).

Writes to ``<HUB_DATA_DIR>/breadth/``:

* ``nyse``     daily breadth (advancing/declining, % above 20/50/200-day, 52-week highs/lows,
               A/D line) plus ``stocks`` (how many were counted) and ``provisional`` (how many
               of the last day's closes were built from hourly bars)
* ``summary``  (JSON) as-of date, universe, coverage and caveats
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from tickerlake.storage import paths as P  # noqa: E402
from tickerlake.storage.query import LakeQuery  # noqa: E402

try:
    from core import store  # the hub's storage module (hub folder is on PYTHONPATH)
    from modules.market import indicators as ind  # the hub's breadth calculation
except ImportError:
    sys.exit("Could not import the hub's modules: run this through market-compass's run_jobs.py, "
             "or put the market-compass folder on PYTHONPATH.")

MODULE = "breadth"
DATA = PROJECT_ROOT / "data"
EXCHANGE, CATEGORY = "N", "common"
MIN_COVERAGE = 0.9  # a session counts once 90% of the usual number of stocks have a bar
PROVISIONAL = "yfinance_1h"


def log(msg: str) -> None:
    print(msg, flush=True)


def listed_mask(closes: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    """True where a stock was listed on that day: each day uses the latest snapshot on or
    before it, and days before the first snapshot use the first one."""
    snaps = {d: set(g["symbol"]) for d, g in snapshots.groupby("snapshot_date")}
    dates = sorted(snaps)
    mask = pd.DataFrame(False, index=closes.index, columns=closes.columns)
    for i, d in enumerate(dates):
        start = pd.Timestamp(d) if i else closes.index.min()
        end = pd.Timestamp(dates[i + 1]) if i + 1 < len(dates) else None
        rows = (closes.index >= start) & ((closes.index < end) if end is not None else True)
        mask.loc[rows, mask.columns.isin(snaps[d])] = True
    return mask


def complete_sessions(closes: pd.DataFrame) -> pd.DataFrame:
    """Drop trailing sessions too thinly filled to count (Yahoo posts late)."""
    counts = closes.notna().sum(axis=1)
    usual = counts.rolling(20, min_periods=1).max().shift(1).fillna(counts)
    ok = counts >= MIN_COVERAGE * usual
    last_ok = ok[ok].index.max()
    return closes.loc[:last_ok]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="accepted for the hub; this script never fetches")
    parser.parse_args()

    if not any((DATA / P.OHLCV_LISTED).rglob("*.parquet")):
        sys.exit("No listed-stock bars in the lake yet: run `tickerlake backfill listed_ohlcv` first.")
    with LakeQuery(DATA) as q:
        snapshots = q.sql(f"SELECT snapshot_date, symbol FROM {P.LISTINGS} "
                          "WHERE exchange = ? AND category = ?", [EXCHANGE, CATEGORY])
        bars = q.adjusted_ohlcv(symbols=sorted(set(snapshots["symbol"])), dataset=P.OHLCV_LISTED)
        sources = q.sql(f"SELECT date, source FROM {P.OHLCV_LISTED} WHERE date = (SELECT MAX(date) "
                        f"FROM {P.OHLCV_LISTED})")
    bars["date"] = pd.to_datetime(bars["date"])
    closes = bars.pivot_table(index="date", columns="symbol", values="adj_close", aggfunc="last").sort_index()
    closes = closes.where(listed_mask(closes, snapshots))
    closes = complete_sessions(closes)

    out = ind.breadth(closes)
    out["stocks"] = closes.notna().sum(axis=1).reindex(out.index)
    last = closes.index[-1]
    sources["date"] = pd.to_datetime(sources["date"])
    provisional = int(((sources["date"] == last) & (sources["source"] == PROVISIONAL)).sum())
    out["provisional"] = 0
    out.loc[last, "provisional"] = provisional

    store.save_frame(MODULE, "nyse", out)
    store.save_json(MODULE, "summary", {
        "as_of": last.strftime("%Y-%m-%d"),
        "universe": "NYSE common stock",
        "stocks": int(out["stocks"].iloc[-1]),
        "provisional_last_day": provisional,
        "first_snapshot": pd.Timestamp(snapshots["snapshot_date"].min()).strftime("%Y-%m-%d"),
        "history_start": out.index[0].strftime("%Y-%m-%d"),
        "notes": ("Before the first listings snapshot, history covers only stocks still listed on it "
                  "(survivorship). Provisional closes are the last hourly trade before 16:00, replaced "
                  "by the official close the next evening."),
    })
    log(f"Published NYSE breadth through {last:%Y-%m-%d}: {int(out['stocks'].iloc[-1])} stocks, "
        f"{provisional} provisional closes, {out['pct_adv'].iloc[-1]:.0f}% advancing, "
        f"{out['above_200'].iloc[-1]:.0f}% above 200-day.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
