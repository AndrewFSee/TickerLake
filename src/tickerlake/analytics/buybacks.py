"""Share repurchases from SEC filings, and estimated buyback blackout windows.

What is and isn't known
-----------------------
Companies announce a repurchase *authorization* (a ceiling, not a commitment). The buying itself
is reported only afterwards: the cash-flow statement gives the dollars spent each quarter, and
the 10-Q / 10-K repurchase table gives shares per month plus the authorization left. So every
number here trails by up to a quarter.

Blackout windows are never disclosed. They are company policy, commonly shutting about two weeks
before the fiscal quarter ends and reopening a day or two after the earnings release. Companies
with a 10b5-1 plan keep buying through them, so "blackout" means less discretionary buying, not
none. The windows here are estimates from each company's own fiscal closes and release dates.

Pure functions on plain frames: no I/O.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tickerlake.analytics import style

REPURCHASE = ["PaymentsForRepurchaseOfCommonStock", "PaymentsForRepurchaseOfEquity"]
REMAINING = ["StockRepurchaseProgramRemainingAuthorizedRepurchaseAmount1",
             "StockRepurchaseProgramRemainingAuthorizedRepurchaseAmount"]
CONCEPTS = REPURCHASE + REMAINING + style.SHARES

SHUT_DAYS_BEFORE_CLOSE = 14   # window shuts this many days before the fiscal quarter ends
REOPEN_BDAYS = 2              # ...and reopens this many business days after the release
MIN_RELEASE_DAYS, MAX_RELEASE_DAYS = 7, 75
STALE_DAYS = 200              # repurchase / authorization figures older than this are dropped


# ------------------------------------------------------------------ repurchases
def symbol_repurchases(f: pd.DataFrame, splits: pd.Series) -> dict | None:
    """Latest point-in-time buyback figures for one company (facts already first-reported)."""
    q = style.pick(f, REPURCHASE, style.quarters)
    if q.empty:
        return None
    q = q.sort_index()
    idx = q.index
    consecutive = pd.Series(idx, index=idx).diff(3).dt.days.between(250, 300)
    ttm = q["value"].astype(float).rolling(4).sum().where(consecutive)
    out = {"quarter_end": idx[-1], "repurchased_q": float(q["value"].iloc[-1]),
           "repurchased_ttm": float(ttm.iloc[-1]) if pd.notna(ttm.iloc[-1]) else np.nan,
           "known": pd.to_datetime(q["known"]).max()}
    sh = style.pick(f, style.SHARES, style.period_average)
    if len(sh) >= 2:
        sh = sh.sort_index()
        adj = sh["value"].astype(float) * style.split_factor(pd.to_datetime(sh["known"]), splits)
        prior = adj[adj.index <= sh.index[-1] - pd.Timedelta(days=330)]
        if len(prior) and prior.index[-1] >= sh.index[-1] - pd.Timedelta(days=400):
            out["share_change_1y"] = float(adj.iloc[-1] / prior.iloc[-1] - 1)
    rem = style.pick(f, REMAINING, style.instant)
    if len(rem):
        rem = rem.sort_index()
        out["remaining_auth"] = float(rem["value"].iloc[-1])
        out["remaining_as_of"] = rem.index[-1]
        if len(rem) >= 2:
            # remaining authorization goes up only when the board adds to the program
            rose = rem["value"].astype(float).diff() > 0
            if rose.any():
                out["auth_added_reported"] = pd.to_datetime(rem.loc[rose, "known"]).max()
    return out


def repurchase_table(facts: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """One row per company: last quarter's and trailing-year buybacks, 1-year change in the
    split-adjusted diluted share count, remaining authorization and when an increase in it was
    last reported. ``facts`` as in ``filings_facts``; ``splits``: symbol, date, stock_splits."""
    f = style.first_reported(facts[facts["concept"].isin(CONCEPTS)])
    sp = splits[splits["stock_splits"].fillna(0) > 0].assign(date=lambda d: pd.to_datetime(d["date"]))
    by_symbol = {s: g.set_index("date")["stock_splits"] for s, g in sp.groupby("symbol")}
    rows = []
    for sym, g in f.groupby("symbol"):
        r = symbol_repurchases(g, by_symbol.get(sym, pd.Series(dtype=float)))
        if r is not None:
            rows.append({"symbol": sym, **r})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ blackout windows
def expected_releases(releases: pd.DataFrame, calendar: pd.DataFrame, today: pd.Timestamp,
                      horizon_days: int = 120) -> pd.DataFrame:
    """Past quarters plus the next ones up to ``horizon_days`` ahead.

    ``releases``: symbol, fiscal_close, release_date (NaT when unmatched), report_filed.
    ``calendar``: symbol, date (expected report dates, e.g. Finnhub's).
    A missing past release falls back to the 10-Q / 10-K filing date. A future close is the same
    quarter a year earlier + 364 days; its release is the calendar date that falls 7-75 days
    after it, else the close plus the company's median lag over its last eight quarters."""
    r = releases.copy()
    r["source"] = np.where(r["release_date"].notna(), "8-K", "10-Q/10-K filed")
    r["release_date"] = r["release_date"].fillna(r["report_filed"])
    r = r.dropna(subset=["release_date"])
    cal = calendar.dropna(subset=["date"]).assign(date=lambda d: pd.to_datetime(d["date"]))
    cal_by = {s: sorted(g["date"]) for s, g in cal.groupby("symbol")}
    extra = []
    end = today + pd.Timedelta(days=horizon_days)
    for sym, g in r.groupby("symbol"):
        g = g.sort_values("fiscal_close")
        lag = (g["release_date"] - g["fiscal_close"]).dt.days.tail(8).median()
        closes = list(g["fiscal_close"])
        last_release = g["release_date"].max()
        while True:
            nxt = (closes[-4] + pd.Timedelta(days=364)) if len(closes) >= 4 else closes[-1] + pd.Timedelta(days=91)
            if nxt > end:
                break
            if nxt <= closes[-1]:
                nxt = closes[-1] + pd.Timedelta(days=91)
            closes.append(nxt)
            hits = [d for d in cal_by.get(sym, []) if nxt + pd.Timedelta(days=MIN_RELEASE_DAYS) <= d
                    <= nxt + pd.Timedelta(days=MAX_RELEASE_DAYS) and d > last_release]
            if hits:
                rel, src = hits[0], "calendar"
            else:
                rel, src = nxt + pd.Timedelta(days=round(float(lag)) if pd.notna(lag) else 30), "estimated"
            extra.append({"symbol": sym, "fiscal_close": nxt, "release_date": rel, "source": src})
    out = pd.concat([r[["symbol", "fiscal_close", "release_date", "source"]], pd.DataFrame(extra)], ignore_index=True)
    return out.sort_values(["symbol", "fiscal_close"]).reset_index(drop=True)


def blackout_windows(releases: pd.DataFrame) -> pd.DataFrame:
    """Each quarter's estimated window: shuts ``SHUT_DAYS_BEFORE_CLOSE`` days before the close,
    reopens ``REOPEN_BDAYS`` business days after the release (``reopen`` is the first open day)."""
    w = releases.copy()
    w["shut"] = w["fiscal_close"] - pd.Timedelta(days=SHUT_DAYS_BEFORE_CLOSE)
    w["reopen"] = w["release_date"] + pd.offsets.BDay(REOPEN_BDAYS)
    return w[w["reopen"] > w["shut"]]


def status_on(windows: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Per company on ``day``: in blackout or not; if shut, when it reopens and the release that
    reopens it; when the next window shuts; and the next release date on or after ``day``."""
    rows = []
    for sym, g in windows.groupby("symbol"):
        g = g.sort_values("shut")
        cur = g[(g["shut"] <= day) & (g["reopen"] > day)]
        nxt = g[g["shut"] > day]
        upcoming = g[g["release_date"] >= day]
        rows.append({
            "symbol": sym, "blackout": not cur.empty,
            "reopens": cur["reopen"].max() if not cur.empty else pd.NaT,
            "window_release": cur.sort_values("reopen")["release_date"].iloc[-1] if not cur.empty else pd.NaT,
            "window_source": cur.sort_values("reopen")["source"].iloc[-1] if not cur.empty else None,
            "next_shut": nxt["shut"].min() if not nxt.empty else pd.NaT,
            "next_release": upcoming["release_date"].min() if not upcoming.empty else pd.NaT,
            "release_source": upcoming.sort_values("release_date")["source"].iloc[0] if not upcoming.empty else None,
        })
    return pd.DataFrame(rows)


def blackout_matrix(windows: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """days x symbols: True while that company's window is estimated shut."""
    out = pd.DataFrame(False, index=days, columns=sorted(windows["symbol"].unique()))
    pos = {s: i for i, s in enumerate(out.columns)}
    arr = out.to_numpy().copy()
    d = days.to_numpy()
    for r in windows.itertuples():
        a, b = np.searchsorted(d, np.datetime64(r.shut)), np.searchsorted(d, np.datetime64(r.reopen))
        arr[a:b, pos[r.symbol]] = True
    return pd.DataFrame(arr, index=days, columns=out.columns)


def covered(windows: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """days x symbols: True between a company's first window and its last (so a company with no
    filings yet, or none any more, isn't counted as 'open')."""
    span = windows.groupby("symbol").agg(first=("shut", "min"), last=("reopen", "max"))
    d = days.to_numpy()[:, None]
    return pd.DataFrame((d >= span["first"].to_numpy()[None, :]) & (d < span["last"].to_numpy()[None, :]),
                        index=days, columns=span.index)


def blackout_share(shut: pd.DataFrame, member: pd.DataFrame, weights: pd.Series | None = None) -> pd.Series:
    """Share of members (or of their weight) estimated in blackout on each day. ``member`` is a
    days x symbols mask of who counts that day."""
    m = member.reindex(columns=shut.columns, fill_value=False)
    s = shut & m
    if weights is None:
        return s.sum(axis=1) / m.sum(axis=1).replace(0, np.nan)
    w = weights.reindex(shut.columns).fillna(0).to_numpy()
    return (s.to_numpy() * w).sum(axis=1) / pd.Series((m.to_numpy() * w).sum(axis=1), index=shut.index).replace(0, np.nan)
