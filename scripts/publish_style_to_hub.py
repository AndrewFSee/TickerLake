"""Publish value / growth scores of S&P 500 members into the Dashboard Hub (market-compass).

The hub's ``run_jobs.py`` runs this with *this* project's venv and the hub folder on
``PYTHONPATH``. It reads the lake only (SEC facts from ``filings_facts``, prices and
dividends from ``ohlcv``, point-in-time members), so ``--demo`` changes nothing. The
method is in ``tickerlake.analytics.style``.

Scores are rebuilt for every month-end since 2012-12 and for the last complete session,
each from the members of that day and the filings known the day before. The month-end
history is also the backtest: does either score rank the next month's or quarter's
returns? The result goes into ``summary`` so the page can state it.

Writes to ``<HUB_DATA_DIR>/style/``:

* ``stocks``         latest scores per member: value, growth, quadrant, each factor raw and
                     as a z-score, market cap, sector, name
* ``stock_history``  value and growth per member per month-end
* ``sectors``        cap-weighted value and growth per sector per month-end and latest date
* ``gaps``           growth members vs value members (book / price split at 30 / 70%) per month-end and
                     latest date: cap-weighted asset growth, book / price and return on equity of each
                     group (``style.style_gaps``); the hub's growth_value job compares it with Ken
                     French's large growth / large value portfolios
* ``summary``        (JSON) as-of date, coverage, factor list and the backtest
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tickerlake.analytics import style  # noqa: E402
from tickerlake.storage.query import LakeQuery  # noqa: E402

try:
    from core import store  # the hub's storage module (hub folder is on PYTHONPATH)
except ImportError:
    sys.exit("Could not import the hub's core.store: run this through market-compass's run_jobs.py, "
             "or put the market-compass folder on PYTHONPATH.")

MODULE = "style"
DATA = PROJECT_ROOT / "data"
HISTORY_START = "2012-12-01"
PRICES_FROM = "2008-01-01"   # 3-year growth trends need filings back to ~2009
MIN_COVERAGE = 0.9


def log(msg: str) -> None:
    print(msg, flush=True)


def last_complete_session(close: pd.DataFrame, symbols: list[str]) -> pd.Timestamp:
    """Latest day on which at least 90% of the usual number of members have a close."""
    counts = close[[s for s in symbols if s in close.columns]].iloc[-30:].notna().sum(axis=1)
    usual = counts.max()
    return counts[counts >= MIN_COVERAGE * usual].index.max()


def latest_caps() -> pd.Series | None:
    """Today's market caps from the hub's market job, when it has run (filing share counts lag)."""
    try:
        caps = store.load_frame("market", "market_caps")
    except Exception:  # noqa: BLE001 - optional input
        return None
    if caps is None or caps.empty or "market_cap" not in caps:
        return None
    return caps["market_cap"].dropna()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="accepted for the hub; this script never fetches")
    parser.parse_args()
    t0 = time.time()

    concepts = ",".join(f"'{c}'" for c in style.CONCEPTS)
    forms = ",".join(f"'{f}'" for f in style.FORMS)
    with LakeQuery(DATA) as q:
        facts = q.sql(f"SELECT symbol, concept, unit, value, start_date, end_date, filed_date FROM filings_facts "
                      f"WHERE concept IN ({concepts}) AND form_type IN ({forms})")
        prices = q.sql("SELECT symbol, date, close, dividends, stock_splits FROM ohlcv WHERE date >= ?", [PRICES_FROM])
        adj = q.adjusted_ohlcv(start=PRICES_FROM)
        members = q.sql("SELECT symbol, start_date, end_date_eff, is_current, gics_sector, company_name "
                        "FROM membership_pit WHERE index_name = 'SP500'")
    if facts.empty:
        sys.exit("No SEC facts in the lake: run `tickerlake run --stages filings` first.")
    log(f"Read {len(facts):,} facts for {facts['symbol'].nunique()} symbols, {len(prices):,} price rows "
        f"({time.time() - t0:.0f}s).")

    prices["date"] = pd.to_datetime(prices["date"])
    close = prices.pivot_table(index="date", columns="symbol", values="close").sort_index()
    divs = prices.pivot_table(index="date", columns="symbol", values="dividends").reindex(close.index).fillna(0)
    adj["date"] = pd.to_datetime(adj["date"])
    adj_close = adj.pivot_table(index="date", columns="symbol", values="adj_close").sort_index()
    members["start_date"] = pd.to_datetime(members["start_date"])
    members["end_date"] = pd.to_datetime(members["end_date_eff"]).where(~members["is_current"].astype(bool))
    names = members.sort_values("start_date").drop_duplicates("symbol", keep="last").set_index("symbol")["company_name"]

    fund = style.fundamentals(facts, prices[["symbol", "date", "stock_splits"]])
    log(f"Quarterly fundamentals for {fund['symbol'].nunique()} symbols ({time.time() - t0:.0f}s).")

    current = members.loc[members["is_current"].astype(bool), "symbol"].tolist()
    latest = last_complete_session(close, current)
    # last session of each completed month (the latest session's month is still running)
    days = close.loc[HISTORY_START:latest].index
    days = days[days.to_period("M") < latest.to_period("M")]
    month_ends = pd.Series(days, index=days).groupby(days.to_period("M")).max().tolist()
    dates = sorted(set(month_ends) | {latest})

    panel = style.factor_panel(fund, close, divs, members, dates)
    panel = style.clean(panel, latest_caps())
    gaps = style.style_gaps(panel)
    scored = style.score_panel(panel)
    log(f"Scored {len(dates)} dates ({time.time() - t0:.0f}s).")

    # backtest on completed month-ends only (the latest session sits mid-month)
    history = scored[scored["date"].isin(month_ends)]
    test = style.backtest(history, adj_close)
    log(f"Backtest: value IC {test['stock']['value']['1m']['mean']:+.3f} (t {test['stock']['value']['1m']['t']:+.1f}), "
        f"growth IC {test['stock']['growth']['1m']['mean']:+.3f} (t {test['stock']['growth']['1m']['t']:+.1f}) "
        f"({time.time() - t0:.0f}s).")

    today = scored[scored["date"] == latest].copy()
    today["quadrant"] = [style.quadrant(v, g) for v, g in zip(today["value"], today["growth"])]
    today["name"] = today["symbol"].map(names)
    cols = ["symbol", "name", "sector", "mcap", "value", "growth", "quadrant", *style.FACTORS,
            *[f"z_{f}" for f in style.FACTORS]]
    stocks = today[cols].sort_values("mcap", ascending=False).reset_index(drop=True)
    sectors = style.sector_scores(scored)
    stock_history = history[["date", "symbol", "value", "growth"]].dropna(subset=["value", "growth"], how="all")

    scored_today = int(today[["value", "growth"]].notna().all(axis=1).sum())
    store.save_frame(MODULE, "stocks", stocks)
    store.save_frame(MODULE, "stock_history", stock_history.reset_index(drop=True))
    store.save_frame(MODULE, "sectors", sectors)
    store.save_frame(MODULE, "gaps", gaps)
    store.save_json(MODULE, "summary", {
        "as_of": str(latest.date()), "members": len(current), "priced": int(len(today)), "scored": scored_today,
        "history_start": str(pd.Timestamp(month_ends[0]).date()), "value_factors": list(style.VALUE),
        "growth_factors": list(style.GROWTH), "caps_from_hub": latest_caps() is not None,
        "quadrants": {k: int(v) for k, v in today["quadrant"].value_counts().items()},
        "backtest": _clean_json(test),
    })
    log(f"Published style scores for {scored_today} of {len(current)} members ({latest.date()}), "
        f"{len(month_ends)} month-ends of history ({time.time() - t0:.0f}s).")
    return 0


def _clean_json(x):
    """NaN -> None so the summary stays valid JSON."""
    if isinstance(x, dict):
        return {k: _clean_json(v) for k, v in x.items()}
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


if __name__ == "__main__":
    raise SystemExit(main())
