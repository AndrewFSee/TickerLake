"""Publish sector-level analyst readings from the lake into the Dashboard Hub (market-compass).

The hub's ``run_jobs.py`` runs this with *this* project's venv and puts the hub folder on
``PYTHONPATH`` so ``from core import store`` resolves to the hub's storage module. It reads
the lake only (the daily ``analyst`` stage does the fetching), so ``--demo`` changes nothing.

Tables written to ``<HUB_DATA_DIR>/revisions/``, by GICS sector of current S&P 500 members:

* ``sector_revisions``  latest snapshot: median 30/90-day change in next fiscal year's EPS
                        estimate, analysts' up-vs-down revision balance, and rating actions
                        (upgrades, downgrades, target raises/cuts) over the last 30 days
* ``revisions_history`` the same revision readings for every stored snapshot date
* ``revisions_names``   each member's revisions, for the hub's drill-down
* ``summary``           (JSON) snapshot date and coverage
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tickerlake.storage.query import LakeQuery  # noqa: E402

try:
    from core import store  # the hub's storage module (hub folder is on PYTHONPATH)
except ImportError:
    sys.exit("Could not import the hub's core.store: run this through market-compass's run_jobs.py, "
             "or put the market-compass folder on PYTHONPATH.")

MODULE = "revisions"
DATA = PROJECT_ROOT / "data"
ACTION_DAYS = 30
MIN_EPS = 0.10  # a % revision of an estimate near zero is meaningless


def log(msg: str) -> None:
    print(msg, flush=True)


def member_revisions(est: pd.DataFrame) -> pd.DataFrame:
    """Per symbol and snapshot: next-fiscal-year EPS revisions and the revision counts."""
    fy1 = est[est["period"] == "+1y"].set_index(["snapshot_date", "symbol"])
    out = pd.DataFrame(index=fy1.index)
    base = fy1["eps_trend_current"]
    for days in (30, 90):
        prior = fy1[f"eps_trend_{days}d"]
        out[f"rev_{days}d"] = np.where(prior.abs() >= MIN_EPS, (base / prior - 1) * 100, np.nan)
    years = est[est["period"].isin(["0y", "+1y"])].groupby(["snapshot_date", "symbol"])
    out = out.join(years[["up_30d", "down_30d"]].sum(min_count=1), how="outer")
    return out.reset_index()


def sector_view(rev: pd.DataFrame) -> pd.DataFrame:
    g = rev.groupby(["snapshot_date", "gics_sector"])
    up, down = g["up_30d"].sum(), g["down_30d"].sum()
    return pd.DataFrame({
        "names": g["symbol"].nunique(),
        "rev_30d": g["rev_30d"].median(),
        "rev_90d": g["rev_90d"].median(),
        "rising_share": g["rev_30d"].apply(lambda s: (s > 0).mean() * 100 if s.notna().any() else np.nan),
        "breadth": (up - down) / (up + down).replace(0, np.nan) * 100,
    }).reset_index()


def rating_actions(ratings: pd.DataFrame, sectors: pd.Series, end: pd.Timestamp) -> pd.DataFrame:
    recent = ratings[ratings["graded_at"] > end - pd.Timedelta(days=ACTION_DAYS)].copy()
    recent["gics_sector"] = recent["symbol"].map(sectors)
    pt = recent["price_target_action"].fillna("").str.lower()
    held = recent["action"].isin(["main", "reit"])
    flags = pd.DataFrame({
        "gics_sector": recent["gics_sector"],
        "upgrades": recent["action"].eq("up"), "downgrades": recent["action"].eq("down"),
        "targets_raised": held & pt.str.startswith("raise"), "targets_cut": held & pt.str.startswith("lower"),
    }).dropna(subset=["gics_sector"])
    return flags.groupby("gics_sector").sum().astype(int)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="accepted for the hub; this script never fetches")
    parser.parse_args()

    est_path = DATA / "analyst_estimates" / "analyst_estimates.parquet"
    rat_path = DATA / "analyst_ratings" / "analyst_ratings.parquet"
    if not est_path.exists():
        sys.exit("No analyst estimates in the lake yet: run `tickerlake run --stages analyst` first.")
    with LakeQuery(DATA) as q:
        members = q.sql("SELECT symbol, gics_sector, company_name FROM membership_pit "
                        "WHERE index_name = 'SP500' AND is_current")
        est = q.sql("SELECT * FROM read_parquet(?)", [str(est_path)])
        ratings = q.sql("SELECT * FROM read_parquet(?)", [str(rat_path)]) if rat_path.exists() else None
    sectors = members.set_index("symbol")["gics_sector"]
    est["snapshot_date"] = pd.to_datetime(est["snapshot_date"])

    rev = member_revisions(est[est["symbol"].isin(sectors.index)])
    rev["gics_sector"] = rev["symbol"].map(sectors)
    rev = rev.dropna(subset=["gics_sector"])
    history = sector_view(rev)
    latest_date = rev["snapshot_date"].max()
    latest = history[history["snapshot_date"] == latest_date].drop(columns="snapshot_date").set_index("gics_sector")
    if ratings is not None and not ratings.empty:
        latest = latest.join(rating_actions(ratings, sectors, pd.Timestamp(latest_date, tz="UTC")
                                            + pd.Timedelta(days=1)), how="left").fillna(
            {"upgrades": 0, "downgrades": 0, "targets_raised": 0, "targets_cut": 0})
    names = rev[rev["snapshot_date"] == latest_date].drop(columns="snapshot_date")
    names["name"] = names["symbol"].map(members.set_index("symbol")["company_name"])

    store.save_frame(MODULE, "sector_revisions", latest.reset_index())
    store.save_frame(MODULE, "revisions_history", history)
    store.save_frame(MODULE, "revisions_names", names.reset_index(drop=True))
    store.save_json(MODULE, "summary", {
        "snapshot_date": latest_date.strftime("%Y-%m-%d"),
        "snapshots": int(rev["snapshot_date"].nunique()),
        "first_snapshot": rev["snapshot_date"].min().strftime("%Y-%m-%d"),
        "members": int(len(sectors)), "covered": int(names["symbol"].nunique()),
        "action_days": ACTION_DAYS,
    })
    log(f"Published revisions for {names['symbol'].nunique()} of {len(sectors)} members "
        f"({latest_date:%Y-%m-%d}, {rev['snapshot_date'].nunique()} snapshot(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
