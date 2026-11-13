"""Value and growth scores for index members, built point in time from SEC filings.

The factors follow CRSP's style method, less the two that need analyst forecasts (those
snapshots only start 2026-11-07, so they have no history to test):

* value:  book / price, trailing earnings / price, sales / price, dividends / price
* growth: 3-year trend of EPS and of sales per share, capex / assets, return on assets

Each factor is clipped at its 2.5 / 97.5 percentiles and turned into a z-score around the
cap-weighted mean of that day's members. A stock's value (growth) score is the mean of its
value (growth) z-scores when at least two exist, standardised the same way again, so 0 is
the cap-weighted index and 1 is one cross-sectional standard deviation.

Point in time: every fact is the first value reported for its period and is usable the day
after it was filed, and a quarter counts only once every input it uses has been filed.
Quarterly flows come from 3-month facts or from year-to-date facts differenced against the
previous quarter of the same fiscal year. Share counts are period
averages (never differenced) put on today's split basis, because the stored closes and
dividends already are.

Pure functions on plain frames: no I/O.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VALUE = ("bp", "ep", "sp", "dp")
GROWTH = ("epsg3", "spsg3", "inv", "roa")
FACTORS = VALUE + GROWTH

# concept groups: per period, the first concept in the list that has a value wins
FLOW = {
    "ni": ["NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic", "ProfitLoss"],
    "rev": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax", "RevenuesNetOfInterestExpense"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
}
SHARES = ["WeightedAverageNumberOfDilutedSharesOutstanding", "WeightedAverageNumberOfSharesOutstandingBasic"]
INSTANT = {"equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
           "assets": ["Assets"]}
CONCEPTS = sorted({c for group in [*FLOW.values(), *INSTANT.values(), SHARES] for c in group})
FORMS = ("10-K", "10-Q", "10-K/A", "10-Q/A")

QUARTER_DAYS = 91.3
GROWTH_QUARTERS, GROWTH_MIN = 12, 8   # 3-year trend from >= 8 of the last 12 TTM values
MAX_FACT_AGE_DAYS = 200               # a snapshot older than this (no recent 10-Q) is dropped
SHARE_OUTLIER = 3.0                   # implied share count > 3x off the stock's median: bad data
CLIP = 0.025
MIN_FACTORS = 2


# ------------------------------------------------------------------ facts -> quarters
def first_reported(facts: pd.DataFrame) -> pd.DataFrame:
    """One row per (symbol, concept, period): the value as first filed, with parsed dates."""
    f = facts[facts["unit"].isin(["USD", "shares"])].copy()
    for c in ("start_date", "end_date", "filed_date"):
        f[c] = pd.to_datetime(f[c])
    f = f.sort_values("filed_date").drop_duplicates(["symbol", "concept", "start_date", "end_date"], keep="first")
    return f


def quarters(f: pd.DataFrame) -> pd.DataFrame:
    """Quarterly values (index: quarter end; columns value, known) of one flow concept, from
    3-month facts and year-to-date facts differenced against the previous quarter."""
    f = f.assign(days=(f["end_date"] - f["start_date"]).dt.days)
    out: dict = {}
    for start, g in f.groupby("start_date"):
        prev_v, prev_known, prev_end = 0.0, pd.Timestamp.min, None
        for r in g.sort_values("end_date").itertuples():
            n_q = round(r.days / QUARTER_DAYS)
            if not 1 <= n_q <= 4 or abs(r.days - n_q * QUARTER_DAYS) > 25:
                continue
            if n_q == 1:
                out.setdefault(r.end_date, (r.value, r.filed_date))
            elif (prev_end is not None and round((r.end_date - prev_end).days / QUARTER_DAYS) == 1
                  and round((prev_end - start).days / QUARTER_DAYS) == n_q - 1):
                out.setdefault(r.end_date, (r.value - prev_v, max(r.filed_date, prev_known)))
            prev_v, prev_known, prev_end = r.value, r.filed_date, r.end_date
    if not out:
        return pd.DataFrame(columns=["value", "known"])
    return pd.DataFrame.from_dict(out, orient="index", columns=["value", "known"]).sort_index()


def period_average(f: pd.DataFrame) -> pd.DataFrame:
    """Period averages (share counts): per end date, the fact closest to one quarter long.
    Averages can't be differenced like flows, so a year-to-date average is used as is."""
    f = f.assign(off=((f["end_date"] - f["start_date"]).dt.days - 91).abs())
    f = f[f["off"] < 300].sort_values(["end_date", "off", "filed_date"]).drop_duplicates("end_date")
    return f.set_index("end_date")[["value", "filed_date"]].rename(columns={"filed_date": "known"})


def instant(f: pd.DataFrame) -> pd.DataFrame:
    """Balance-sheet values per date, as first filed."""
    f = f.sort_values(["end_date", "filed_date"]).drop_duplicates("end_date")
    return f.set_index("end_date")[["value", "filed_date"]].rename(columns={"filed_date": "known"})


def pick(f: pd.DataFrame, group: list[str], fn) -> pd.DataFrame:
    """Per period, the first concept in ``group`` that has a value."""
    parts = []
    for rank, concept in enumerate(group):
        sub = f[f["concept"] == concept]
        if len(sub):
            x = fn(sub)
            if len(x):
                parts.append(x.assign(rank=rank))
    if not parts:
        return pd.DataFrame(columns=["value", "known"])
    x = pd.concat(parts).reset_index(names="end").sort_values(["end", "rank"]).drop_duplicates("end")
    return x.set_index("end")[["value", "known"]]


def growth_slope(values: pd.Series, window: int = GROWTH_QUARTERS, min_obs: int = GROWTH_MIN) -> pd.Series:
    """Trend per year of a quarterly series over the last ``window`` quarters, as a share of
    its mean absolute level (so it reads like a growth rate and copes with losses)."""
    years = np.arange(window) / 4.0

    def slope(w: np.ndarray) -> float:
        ok = ~np.isnan(w)
        if ok.sum() < min_obs or not ok[-1]:
            return np.nan
        level = np.abs(w[ok]).mean()
        return np.polyfit(years[-len(w):][ok], w[ok], 1)[0] / level if level > 0 else np.nan

    return values.rolling(window, min_periods=min_obs).apply(slope, raw=True)


def split_factor(known: pd.Series, splits: pd.Series) -> np.ndarray:
    """For each date in ``known``, the product of the splits (ratio per date) after it."""
    if splits.empty:
        return np.ones(len(known))
    s = splits.sort_index()
    after = np.cumprod(s.to_numpy(float)[::-1])[::-1]          # product from each split onward
    pos = np.searchsorted(s.index.to_numpy(), known.to_numpy(), side="right")
    return np.append(after, 1.0)[pos]


def symbol_quarters(f: pd.DataFrame, splits: pd.Series) -> pd.DataFrame | None:
    """One symbol's quarterly fundamentals with the date each row became known."""
    flows = {k: pick(f, v, quarters) for k, v in FLOW.items()}
    shares = pick(f, SHARES, period_average)
    if flows["ni"].empty or shares.empty:
        return None
    idx = flows["ni"].index
    df = pd.DataFrame(index=idx)
    consecutive = pd.Series(idx, index=idx).diff(3).dt.days.between(250, 300)
    for k, q in flows.items():
        df[f"{k}_ttm"] = q["value"].reindex(idx).astype(float).rolling(4).sum().where(consecutive)
    sh = shares.reindex(idx)
    df["shares"] = sh["value"].astype(float).ffill(limit=2)
    knowns = [pd.to_datetime(q["known"]).reindex(idx) for q in flows.values() if len(q)]
    knowns.append(pd.to_datetime(sh["known"]).ffill(limit=2))
    for k, group in INSTANT.items():
        x = pick(f, group, instant)
        if len(x):
            df[k] = x["value"].astype(float).reindex(idx, method="ffill", tolerance=pd.Timedelta(days=20))
            knowns.append(pd.to_datetime(x["known"]).reindex(idx, method="ffill", tolerance=pd.Timedelta(days=20)))
        else:
            df[k] = np.nan
    # a quarter counts once every input with a value for it has been filed: a fact can first appear
    # as a comparative in a later filing (22% of quarters had one, a median 282 days after net income)
    df["known"] = pd.concat(knowns, axis=1).max(axis=1).cummax()
    df["shares"] = df["shares"] * split_factor(df["known"], splits)
    df["epsg3"] = growth_slope(df["ni_ttm"] / df["shares"])
    df["spsg3"] = growth_slope(df["rev_ttm"] / df["shares"])
    return df.reset_index(names="qend")


def fundamentals(facts: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Quarterly fundamentals of every symbol. ``facts`` as in ``filings_facts``; ``splits``
    has symbol, date and stock_splits (the ratio, as Yahoo stores it)."""
    f = first_reported(facts[facts["concept"].isin(CONCEPTS)])
    sp = splits[splits["stock_splits"].fillna(0) > 0].assign(date=lambda d: pd.to_datetime(d["date"]))
    by_symbol = {s: g.set_index("date")["stock_splits"] for s, g in sp.groupby("symbol")}
    rows = []
    for sym, g in f.groupby("symbol"):
        q = symbol_quarters(g, by_symbol.get(sym, pd.Series(dtype=float)))
        if q is not None:
            rows.append(q.assign(symbol=sym))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ------------------------------------------------------------------ quarters -> factors
def factor_panel(fund: pd.DataFrame, close: pd.DataFrame, dividends: pd.DataFrame,
                 members: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Raw factors for each member on each date. ``close`` / ``dividends``: date x symbol,
    split-adjusted. ``members``: symbol, start_date, end_date (NaT while current), gics_sector.
    A filing counts from the day after it was filed."""
    fund = fund.sort_values(["symbol", "qend"])
    out = []
    for d in dates:
        m = members[(members["start_date"] <= d) & (members["end_date"].isna() | (members["end_date"] > d))]
        syms = [s for s in m["symbol"].unique() if s in close.columns]
        p = close.loc[d, syms].dropna()
        snap = fund[fund["known"] < d].groupby("symbol").tail(1).set_index("symbol")
        snap = snap[snap["qend"] > d - pd.Timedelta(days=MAX_FACT_AGE_DAYS)].reindex(p.index)
        cap = (p * snap["shares"]).where(snap["shares"] > 0)
        dps = dividends.loc[d - pd.Timedelta(days=365):d, p.index].sum()
        out.append(pd.DataFrame({
            "date": d, "symbol": p.index, "price": p.values, "mcap": cap.values,
            "bp": (snap["equity"] / cap).values, "ep": (snap["ni_ttm"] / cap).values,
            "sp": (snap["rev_ttm"] / cap).values, "dp": (dps / p).values,
            "epsg3": snap["epsg3"].values, "spsg3": snap["spsg3"].values,
            "inv": (snap["capex_ttm"] / snap["assets"]).values, "roa": (snap["ni_ttm"] / snap["assets"]).values,
            "sector": p.index.map(m.drop_duplicates("symbol", keep="last").set_index("symbol")["gics_sector"]),
        }))
    return pd.concat(out, ignore_index=True)


def clean(panel: pd.DataFrame, latest_caps: pd.Series | None = None) -> pd.DataFrame:
    """inf -> NaN; blank the cap-based factors where the implied share count is more than
    ``SHARE_OUTLIER`` times off the stock's own median (a units or split slip in a filing).
    ``latest_caps`` (symbol -> market cap) replaces the last date's caps, rescaling B/P, E/P, S/P."""
    panel = panel.replace([np.inf, -np.inf], np.nan).copy()
    shares = panel["mcap"] / panel["price"]
    rel = shares / shares.groupby(panel["symbol"]).transform("median")
    panel.loc[~rel.between(1 / SHARE_OUTLIER, SHARE_OUTLIER), ["mcap", "bp", "ep", "sp"]] = np.nan
    if latest_caps is not None:
        last = panel["date"] == panel["date"].max()
        real = panel.loc[last, "symbol"].map(latest_caps)
        have = last & real.reindex(panel.index).notna() & (real.reindex(panel.index) > 0)
        k = panel.loc[have, "mcap"] / real[have]
        for c in ("bp", "ep", "sp"):
            panel.loc[have, c] = panel.loc[have, c] * k
        panel.loc[have, "mcap"] = real[have]
    return panel


# ------------------------------------------------------------------ factors -> scores
def zscore(x: pd.Series, weights: pd.Series, clip: float = CLIP) -> pd.Series:
    """Clip at the ``clip`` / 1 - ``clip`` quantiles, centre on the cap-weighted mean,
    divide by the plain cross-sectional standard deviation."""
    if x.notna().sum() < 3:
        return pd.Series(np.nan, index=x.index)
    lo, hi = x.quantile([clip, 1 - clip])
    x = x.clip(lo, hi)
    ok = x.notna() & weights.notna() & (weights > 0)
    centre = np.average(x[ok], weights=weights[ok]) if ok.any() else x.mean()
    sd = x.std()
    return (x - centre) / sd if sd > 0 else pd.Series(np.nan, index=x.index)


def score(day: pd.DataFrame) -> pd.DataFrame:
    """One date's cross-section -> z_<factor>, value and growth columns added."""
    day = day.copy()
    w = day["mcap"]
    for f in FACTORS:
        day[f"z_{f}"] = zscore(day[f], w)
    for name, group in (("value", VALUE), ("growth", GROWTH)):
        z = day[[f"z_{f}" for f in group]]
        composite = z.mean(axis=1).where(z.notna().sum(axis=1) >= MIN_FACTORS)
        day[name] = zscore(composite, w, clip=0.0)
    return day


def score_panel(panel: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([score(g) for _, g in panel.groupby("date")], ignore_index=True)


def quadrant(value: float, growth: float) -> str | None:
    if pd.isna(value) or pd.isna(growth):
        return None
    if value > 0:
        return "cheap, growing" if growth > 0 else "cheap, slow"
    return "pricey, growing" if growth > 0 else "pricey, slow"


def sector_scores(scored: pd.DataFrame) -> pd.DataFrame:
    """Cap-weighted value and growth per date and sector (stocks with both scores and a cap)."""
    s = scored.dropna(subset=["value", "growth", "mcap", "sector"])
    s = s[s["mcap"] > 0].assign(wv=lambda d: d["value"] * d["mcap"], wg=lambda d: d["growth"] * d["mcap"])
    g = s.groupby(["date", "sector"]).agg(wv=("wv", "sum"), wg=("wg", "sum"), mcap=("mcap", "sum"),
                                          stocks=("symbol", "size")).reset_index()
    g["value"], g["growth"] = g["wv"] / g["mcap"], g["wg"] / g["mcap"]
    return g[["date", "sector", "value", "growth", "mcap", "stocks"]]


# ------------------------------------------------------------------ does it forecast?
def _tstat(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.mean() / s.std() * np.sqrt(len(s))) if len(s) > 2 and s.std() > 0 else float("nan")


def _summary(s: pd.Series, split: pd.Timestamp) -> dict:
    s = s.dropna()
    a, b = s[s.index < split], s[s.index >= split]
    return {"mean": float(s.mean()), "t": _tstat(s), "n": int(len(s)),
            "first": float(a.mean()), "first_t": _tstat(a), "second": float(b.mean()), "second_t": _tstat(b)}


def _rank_ic(frame: pd.DataFrame, col: str, ret: str, min_n: int) -> float:
    ok = frame[[col, ret]].dropna()
    return ok[col].rank().corr(ok[ret].rank()) if len(ok) >= min_n else np.nan


def forward_returns(scored: pd.DataFrame, adj_close: pd.DataFrame, horizons=(1, 3)) -> pd.DataFrame:
    """Add r<h>: % total return from each scoring date to the h-th next one."""
    dates = sorted(scored["date"].unique())
    px = adj_close.reindex(dates, method="ffill")
    out = scored.copy()
    for h in horizons:
        long = ((px.shift(-h) / px - 1) * 100).stack().reset_index()
        long.columns = ["date", "symbol", f"r{h}"]
        out = out.merge(long, on=["date", "symbol"], how="left")
    return out


def backtest(scored: pd.DataFrame, adj_close: pd.DataFrame, split: str = "2020-01-01") -> dict:
    """Do the scores rank next-month / next-quarter returns? Stock level (members at each date),
    sector level (cap-weighted), and the cheap-and-growing corner. Three-month tests use
    non-overlapping quarters."""
    split = pd.Timestamp(split)
    df = forward_returns(scored, adj_close)
    dates = sorted(df["date"].unique())
    quarterly = list(dates[::3])
    res: dict = {"start": str(pd.Timestamp(dates[0]).date()), "end": str(pd.Timestamp(dates[-1]).date()),
                 "months": len(dates), "split": str(split.date()),
                 "members_min": int(df.groupby("date").size().min()),
                 "members_max": int(df.groupby("date").size().max())}

    def ic(col: str, ret: str, frame=df, min_n=50, step=None) -> dict:
        f = frame[frame["date"].isin(quarterly)] if step == 3 else frame
        return _summary(f.groupby("date").apply(lambda g: _rank_ic(g, col, ret, min_n)), split)

    res["stock"] = {c: {"1m": ic(c, "r1"), "3m": ic(c, "r3", step=3)} for c in ("value", "growth")}
    res["factors"] = {f: {"1m": ic(f"z_{f}", "r1"), "3m": ic(f"z_{f}", "r3", step=3)} for f in FACTORS}

    def quintile(g: pd.DataFrame, col: str) -> float:
        g = g.dropna(subset=[col, "r1"])
        if len(g) < 50:
            return np.nan
        return g.loc[g[col] >= g[col].quantile(0.8), "r1"].mean() - g.loc[g[col] <= g[col].quantile(0.2), "r1"].mean()

    res["quintile"] = {c: _summary(df.groupby("date").apply(lambda g: quintile(g, c)), split)
                       for c in ("value", "growth")}

    def corner(g: pd.DataFrame) -> float:
        g = g.dropna(subset=["value", "growth", "r1"])
        sel = g[(g["value"] > 0) & (g["growth"] > 0)]
        return sel["r1"].mean() - g["r1"].mean() if len(sel) >= 5 else np.nan

    both = df.dropna(subset=["value", "growth"])
    res["corner"] = {**_summary(df.groupby("date").apply(corner), split),
                     "share": float(((both["value"] > 0) & (both["growth"] > 0)).mean())}

    s = df.dropna(subset=["value", "growth", "mcap", "sector"])
    s = s[s["mcap"] > 0]
    rows = []
    for (d, sec), g in s.groupby(["date", "sector"]):
        row = {"date": d, "sector": sec, "value": np.average(g["value"], weights=g["mcap"]),
               "growth": np.average(g["growth"], weights=g["mcap"])}
        for h in ("r1", "r3"):
            ok = g[h].notna()
            row[h] = np.average(g.loc[ok, h], weights=g.loc[ok, "mcap"]) if ok.any() else np.nan
        rows.append(row)
    sec = pd.DataFrame(rows)
    res["sector"] = {c: {"1m": ic(c, "r1", sec, 8), "3m": ic(c, "r3", sec, 8, step=3)} for c in ("value", "growth")}
    res["sectors"] = int(sec["sector"].nunique())
    return res
