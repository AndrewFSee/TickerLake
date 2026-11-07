"""Did broker upgrades and downgrades carry an edge? Event study + calendar-time portfolios.

Reads the lake only (analyst_ratings, ohlcv, membership); run after
`tickerlake backfill analyst`:

    .venv\\Scripts\\python.exe scripts\\backtest_analyst_ratings.py [--since 2012-01-01]

Rules, chosen so nothing is known before it could have been:

* Universe: a rating change counts only if the company was an S&P 500 member on
  that date (point-in-time membership). Yahoo still serves history for many
  removed names, not all of them (see the coverage line in the output).
* Timing: a change is tradable at the first open after its timestamp. Yahoo
  stamps most changes pre-market; a change stamped exactly midnight or 16:00 ET
  carries a date only and is treated as after the close.
* Returns are beta-adjusted against SPY (beta from the 252 sessions before the
  event, at least 126), using split/dividend-adjusted prices.
* Event study: the announcement reaction (close before -> close of the first
  session after) is information, not a trade; the drift from the entry open is
  the tradable part. t-statistics cluster events by month.
* Calendar-time portfolios: hold every stock with a qualifying change for H
  sessions after entry, equal-weighted, and measure the daily beta-adjusted
  return of that portfolio. This handles events piling up on the same days.
  Holding starts at the entry session's close, so the entry day itself (open to
  close) is left out: a conservative choice.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tickerlake.storage.query import LakeQuery  # noqa: E402

BULLISH = {"buy", "strong buy", "outperform", "overweight", "positive", "market outperform",
           "sector outperform", "accumulate", "add", "conviction buy", "top pick", "long-term buy",
           "outperformer", "moderate buy", "speculative buy"}
BEARISH = {"sell", "strong sell", "underperform", "underweight", "negative", "reduce",
           "market underperform", "sector underperform", "moderate sell", "underperformer"}
HORIZONS = (1, 5, 21, 63)
HOLDS = (5, 21, 63)
COST_BPS = 10  # one-way cost assumed for break-even comparisons


def load(root: Path, since: str):
    with LakeQuery(root) as q:
        ratings = q.sql("SELECT * FROM read_parquet(?)",
                        [str(root / "analyst_ratings" / "analyst_ratings.parquet")])
        members = q.sql("SELECT symbol, start_date, end_date_eff, gics_sector FROM membership_pit "
                        "WHERE index_name = 'SP500'")
        symbols = sorted(set(ratings["symbol"]) | {"SPY"})
        start = (pd.Timestamp(since) - pd.DateOffset(years=1, months=2)).date()
        px = q.ohlcv(symbols=symbols, start=start)
    return ratings, members, px


def classify(r: pd.DataFrame) -> pd.Series:
    to = r["to_grade"].fillna("").str.strip().str.lower()
    pt_up = r["price_target_action"].fillna("").str.lower().str.startswith("raise")
    pt_down = r["price_target_action"].fillna("").str.lower().str.startswith("lower")
    held = r["action"].isin(["main", "reit"])
    return pd.Series(np.select(
        [r["action"] == "up", r["action"] == "down",
         (r["action"] == "init") & to.isin(BULLISH), (r["action"] == "init") & to.isin(BEARISH),
         (r["action"] == "init"), held & pt_up, held & pt_down],
        ["upgrade", "downgrade", "init bullish", "init bearish", "init neutral", "target raised",
         "target lowered"], default="other"), index=r.index)


def entry_session(stamps: pd.Series, sessions: pd.DatetimeIndex) -> np.ndarray:
    """Index of the first session whose 09:30 ET open comes after each timestamp."""
    et = stamps.dt.tz_convert("America/New_York")
    hm = et.dt.hour * 100 + et.dt.minute
    day = et.dt.tz_localize(None).dt.normalize()
    date_only = hm.isin([0, 1600])
    before_open = (hm < 930) & ~date_only
    first_on_or_after = sessions.searchsorted(day.to_numpy())
    # a session on the event day itself only counts if the change came before its open
    same_day = (first_on_or_after < len(sessions)) & (
        sessions[np.minimum(first_on_or_after, len(sessions) - 1)] == day.to_numpy())
    return np.where(same_day & ~before_open.to_numpy(), first_on_or_after + 1, first_on_or_after)


def month_clustered(values: pd.Series, months: pd.Series) -> tuple[float, float, int]:
    m = values.groupby(months).mean().dropna()
    if len(m) < 12:
        return float("nan"), float("nan"), len(m)
    return float(m.mean()), float(m.mean() / m.std() * np.sqrt(len(m))), len(m)  # mean of monthly means


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--since", default="2012-01-01")
    args = ap.parse_args()
    root = Path(args.data)

    ratings, members, px = load(root, args.since)
    ratings = ratings[ratings["graded_at"] >= pd.Timestamp(args.since, tz="UTC")].copy()
    close = px.pivot_table(index="date", columns="symbol", values="adj_close")
    raw_close = px.pivot_table(index="date", columns="symbol", values="close")
    open_ = px.pivot_table(index="date", columns="symbol", values="open") * (close / raw_close)
    close.index = open_.index = pd.to_datetime(close.index)
    sessions = close.index
    if "SPY" not in close:
        print("SPY prices missing from the lake", file=sys.stderr)
        return 1

    # point-in-time membership on the event date
    ratings["day"] = ratings["graded_at"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    m = ratings.reset_index().merge(members, on="symbol", how="inner")
    m = m[(pd.to_datetime(m["start_date"]) <= m["day"]) & (pd.to_datetime(m["end_date_eff"]) >= m["day"])]
    ev = ratings.loc[m["index"].unique()].copy()
    in_universe = members["symbol"].nunique()
    covered = ev["symbol"].nunique()
    ev["kind"] = classify(ev)
    ev = ev[ev["kind"] != "other"]
    ev = ev[ev["symbol"].isin(close.columns)]

    # returns and betas
    ret = close.pct_change(fill_method=None)
    mkt = ret["SPY"]
    beta = (ret.rolling(252, min_periods=126).cov(mkt).div(mkt.rolling(252, min_periods=126).var(), axis=0)
            .shift(1))
    col = {s: i for i, s in enumerate(close.columns)}
    C, OP, B = close.to_numpy(), open_.to_numpy(), beta.to_numpy()
    spy_c, spy_o = close["SPY"].to_numpy(), open_["SPY"].to_numpy()
    e_idx = entry_session(ev["graded_at"], sessions)
    ok = e_idx < len(sessions) - 1
    ev, e_idx = ev[ok].copy(), e_idx[ok]
    j = ev["symbol"].map(col).to_numpy()
    b = B[e_idx, j]
    ev["beta"] = b
    # reaction: close of the session before entry -> close of the entry session (pre-open
    # changes) -- the market's verdict, not a trade
    prev = np.maximum(e_idx - 1, 0)
    ev["reaction"] = (C[e_idx, j] / C[prev, j] - 1) - b * (spy_c[e_idx] / spy_c[prev] - 1)
    for h in HORIZONS:
        x = np.minimum(e_idx + h - 1, len(sessions) - 1)
        valid = e_idx + h - 1 < len(sessions)
        r = (C[x, j] / OP[e_idx, j] - 1) - b * (spy_c[x] / spy_o[e_idx] - 1)
        ev[f"drift_{h}"] = np.where(valid, r, np.nan)
    ev = ev[np.isfinite(ev["beta"])]
    ev["month"] = ev["graded_at"].dt.strftime("%Y-%m")
    ev["half"] = np.where(ev["graded_at"] < pd.Timestamp("2019-01-01", tz="UTC"), "2012-2018", "2019-2026")

    print(f"Rating changes while in the S&P 500: {len(ev):,} classified events, "
          f"{covered} of {in_universe} member companies covered by Yahoo, "
          f"{ev['graded_at'].min():%Y-%m} to {ev['graded_at'].max():%Y-%m}\n")

    # ---------------------------------------------------------------- event study
    kinds = ["upgrade", "downgrade", "init bullish", "init neutral", "init bearish", "target raised",
             "target lowered"]
    rows = []
    for k in kinds:
        g = ev[ev["kind"] == k]
        row = {"event": k, "n": len(g)}
        mean, t, _ = month_clustered(g["reaction"], g["month"])
        row["reaction %"], row["t(r)"] = mean * 100, t
        for h in HORIZONS:
            mean, t, _ = month_clustered(g[f"drift_{h}"].dropna(), g.loc[g[f"drift_{h}"].notna(), "month"])
            row[f"drift {h}d %"], row[f"t({h})"] = mean * 100, t
        rows.append(row)
    table = pd.DataFrame(rows).set_index("event")
    print("EVENT STUDY - beta-adjusted vs SPY; reaction = close before -> first close after;")
    print("drift = from the first open after the change (tradable); t clustered by month")
    print(table.round(2).to_string(), "\n")

    print("DRIFT 21d BY PERIOD (%, t)")
    for k in kinds:
        parts = []
        for half, g in ev[ev["kind"] == k].groupby("half"):
            mean, t, _ = month_clustered(g["drift_21"].dropna(), g.loc[g["drift_21"].notna(), "month"])
            parts.append(f"{half}: {mean * 100:+.2f} (t {t:+.1f}, n {len(g):,})")
        print(f"  {k:15s} " + "   ".join(parts))
    print()

    # ------------------------------------------------------- calendar-time portfolios
    ex = (ret - beta.mul(mkt, axis=0)).to_numpy()  # daily beta-adjusted returns
    dates = sessions
    # Baseline: every covered company while it was a member, equal-weighted. The covered set
    # leans towards survivors (Yahoo drops many removed names), so legs are shown against it.
    member = np.zeros_like(ex, dtype=bool)
    for r in members[members["symbol"].isin(ev["symbol"].unique())].itertuples():
        jj = col.get(r.symbol)
        if jj is not None:
            member[(dates >= pd.Timestamp(r.start_date)) & (dates <= pd.Timestamp(r.end_date_eff)), jj] = True
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        baseline = pd.Series(np.nanmean(np.where(member, ex, np.nan), axis=1), index=dates)
    b_ann = baseline[baseline.index >= pd.Timestamp(args.since)].mean() * 252 * 100
    print("CALENDAR-TIME PORTFOLIOS - equal-weight, daily beta-adjusted, held H sessions from the entry close.")
    print(f"Each leg is shown minus the baseline (all covered members, equal-weight: {b_ann:+.2f}%/yr vs "
          "beta x SPY);")
    print("annualized %, t, and the one-way cost per trade that would erase it")
    port = {}
    for k in ["upgrade", "downgrade", "init bullish", "init bearish", "target raised", "target lowered"]:
        g = ev[ev["kind"] == k]
        gi, gj = e_idx[ev.index.get_indexer(g.index)], g["symbol"].map(col).to_numpy()
        for hold in HOLDS:
            held = np.zeros_like(ex, dtype=bool)
            for d in range(1, hold + 1):
                rows_ = np.minimum(gi + d, len(dates) - 1)
                held[rows_, gj] = True
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # days with nothing held
                port[(k, hold)] = pd.Series(np.nanmean(np.where(held, ex, np.nan), axis=1), index=dates)
    for hold in HOLDS:
        print(f"  hold {hold} sessions:")
        for k in ["upgrade", "downgrade", "init bullish", "init bearish", "target raised", "target lowered"]:
            s = (port[(k, hold)] - baseline).dropna()
            s = s[s.index >= pd.Timestamp(args.since)]
            ann = s.mean() * 252 * 100
            t = s.mean() / s.std() * np.sqrt(len(s))
            turnover = 2 * 252 / hold  # full portfolio in and out every `hold` sessions
            print(f"    {k:15s} {ann:+6.2f}%/yr  t {t:+5.1f}  break-even cost {abs(ann) / turnover * 100:5.1f} bps/trade")
        # long-short: the clean test, since both legs share the size/coverage tilt
        for name, (lo, sh) in (("up - down", ("upgrade", "downgrade")),
                               ("target up - down", ("target raised", "target lowered"))):
            ls = (port[(lo, hold)] - port[(sh, hold)]).dropna()
            ls = ls[ls.index >= pd.Timestamp(args.since)]
            for label, part in (("all", ls), ("2012-2018", ls[ls.index < "2019-01-01"]),
                                ("2019-2026", ls[ls.index >= "2019-01-01"])):
                ann, t = part.mean() * 252 * 100, part.mean() / part.std() * np.sqrt(len(part))
                net = ann - 4 * 252 / hold * COST_BPS / 100
                print(f"    {name:16s} {label:10s} {ann:+6.2f}%/yr  t {t:+5.1f}  "
                      f"(net of {COST_BPS} bps/trade on both legs: {net:+.2f}%/yr)")
    print()

    # ------------------------------------------------------------------ by firm
    top = ev["firm"].value_counts().head(15).index
    rows = []
    for firm in top:
        g = ev[ev["firm"] == firm]
        up, dn = g[g["kind"] == "upgrade"], g[g["kind"] == "downgrade"]
        mu = month_clustered(up["drift_21"].dropna(), up.loc[up["drift_21"].notna(), "month"])
        md = month_clustered(dn["drift_21"].dropna(), dn.loc[dn["drift_21"].notna(), "month"])
        rows.append({"firm": firm, "upgrades": len(up), "drift21 up %": mu[0] * 100, "t up": mu[1],
                     "downgrades": len(dn), "drift21 down %": md[0] * 100, "t down": md[1],
                     "reaction up %": up["reaction"].mean() * 100, "reaction down %": dn["reaction"].mean() * 100})
    print("BY FIRM (15 most active) - 21-session drift from the entry open; reaction for comparison")
    print(pd.DataFrame(rows).set_index("firm").round(2).to_string())
    print("\nWith 15 firms x 2 sides tested, expect one or two |t| > 2 by chance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
