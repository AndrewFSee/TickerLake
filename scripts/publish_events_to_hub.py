"""Publish the release calendar and prediction-market odds into the Dashboard Hub.

The hub's ``run_jobs.py`` runs this with *this* project's venv and the hub folder on
``PYTHONPATH``. It reads the lake only (the ``econ_calendar`` and ``prediction_markets``
stages fetch), so ``--demo`` changes nothing.

Writes to ``<HUB_DATA_DIR>/events/``:

* ``calendar``  upcoming releases (date, New York time, importance)
* ``odds``      each release's markets: Kalshi events whose markets close on the release
                date, Polymarket events matching the release's title patterns that end
                within ``MATCH_DAYS`` of it. A ladder gives the implied median; named
                outcomes (the Fed) give each outcome's probability.
* ``assets``    S&P 500, Nasdaq-100 and single-stock markets: close-above ladders (implied
                median), touch ladders (odds of trading at a price before expiry), ranges
* ``fed``       every upcoming FOMC meeting on either venue: P(cut), P(hold), P(hike)
* ``other``     the most-traded other US macro / Fed markets (recession, number of cuts, ...)
* ``history``   per snapshot: each event's implied median or leading outcome's probability
* ``summary``   (JSON) snapshot date and counts
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from tickerlake.analytics.prediction import event_view  # noqa: E402
from tickerlake.config import load_config  # noqa: E402
from tickerlake.storage import paths as P  # noqa: E402
from tickerlake.storage.query import LakeQuery  # noqa: E402

try:
    from core import store  # the hub's storage module (hub folder is on PYTHONPATH)
except ImportError:
    sys.exit("Could not import the hub's core.store: run this through market-compass's run_jobs.py, "
             "or put the market-compass folder on PYTHONPATH.")

MODULE = "events"
DATA = PROJECT_ROOT / "data"
MATCH_DAYS = 10
NEW_YORK = "America/New_York"
INDEX_SERIES = {"KXINX": "S&P 500", "KXNASDAQ100": "Nasdaq-100", "KXINXPOS": "S&P 500"}
OTHER_TAGS = ("economy", "fed-rates")
OTHER_KEPT = 15
# "other" keeps US macro and Fed questions; the same tags also carry "largest company" and the like
OTHER_PATTERN = r"Fed|rate|recession|inflation|CPI|PCE|yield|Treasury|GDP|jobs|unemploy|payroll|cut|hike|shutdown"


def log(msg: str) -> None:
    print(msg, flush=True)


def summarise(rows: pd.DataFrame) -> dict:
    """One event's row for the hub: identity, size and what its prices imply."""
    first = rows.iloc[0]
    view = event_view(rows)
    out = {
        "venue": first["venue"], "series": first["series"], "event_id": first["event_id"],
        "event_title": first["event_title"], "close_time": rows["close_time"].min(),
        "volume": rows["volume"].sum(min_count=1), "volume_24h": rows["volume_24h"].sum(min_count=1),
        "url": first["url"], "kind": view["kind"], "median": view.get("median"),
        "below_range": view.get("below_range"), "above_range": view.get("above_range"),
        "top": view.get("top"), "top_prob": view.get("top_prob"),
        "pct": bool(rows["outcome"].astype(str).str.contains("%").any()),
        "detail": json.dumps({k: v for k, v in view.items() if k in ("strikes", "p_above", "outcomes", "probs",
                                                                     "up", "down")}),
    }
    return out


def match_releases(cal: pd.DataFrame, events: pd.DataFrame, releases: dict) -> pd.DataFrame:
    """Attach each event to the release it trades on."""
    out = []
    for _, ev in events.iterrows():
        when = pd.Timestamp(ev["close_time"]).tz_convert(NEW_YORK).date() if pd.notna(ev["close_time"]) else None
        for key, spec in releases.items():
            dates = cal.loc[cal["release_key"] == key, "release_date"]
            if dates.empty or when is None:
                continue
            if ev["venue"] == "kalshi":
                hit = ev["series"] in (spec.get("kalshi") or []) and when in set(dates)
                rdate = when
            else:
                pats = spec.get("polymarket") or []
                if not any(re.search(p, ev["event_title"] or "", re.I) for p in pats):
                    continue
                gaps = {d: abs((pd.Timestamp(d) - pd.Timestamp(when)).days) for d in dates}
                rdate = min(gaps, key=gaps.get)
                hit = gaps[rdate] <= MATCH_DAYS
            if hit:
                out.append({**ev.to_dict(), "release_key": key, "release_date": rdate})
                break
    return pd.DataFrame(out)


def fed_side(label: str) -> str | None:
    t = (label or "").lower()
    if re.search(r"cut|decrease|lower", t):
        return "cut"
    if re.search(r"hike|increase|raise", t):
        return "hike"
    if re.search(r"maintain|no change|hold|unchanged|pause", t):
        return "hold"
    return None


def fed_meetings(today: pd.DataFrame) -> pd.DataFrame:
    """P(cut / hold / hike) per meeting and venue from the Fed-decision events."""
    fed = today[(today["series"] == "KXFEDDECISION")
                | (today["venue"].eq("polymarket") & today["event_title"].str.match(r"Fed Decision in", na=False))]
    out = []
    for (venue, eid), g in fed.groupby(["venue", "event_id"]):
        g = g.dropna(subset=["prob"])
        total = g["prob"].sum()
        if total <= 0:
            continue
        sides = g.assign(side=g["outcome"].map(fed_side)).groupby("side")["prob"].sum() / total
        out.append({"meeting": pd.Timestamp(g["close_time"].min()).tz_convert(NEW_YORK).date(), "venue": venue,
                    "event_id": eid, "event_title": g["event_title"].iloc[0], "url": g["url"].iloc[0],
                    "cut": float(sides.get("cut", 0.0)), "hold": float(sides.get("hold", 0.0)),
                    "hike": float(sides.get("hike", 0.0)), "volume": g["volume"].sum(min_count=1)})
    return pd.DataFrame(out).sort_values(["meeting", "venue"]) if out else pd.DataFrame()


def underlying(ev: pd.Series) -> str | None:
    if ev["venue"] == "kalshi":
        return INDEX_SERIES.get(ev["series"])
    m = re.search(r"\(([A-Z.]{1,6})\)", ev["event_title"] or "")
    return m.group(1) if m else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="accepted for the hub; this script never fetches")
    parser.parse_args()

    if not any((DATA / P.PREDICTION_MARKETS).rglob("*.parquet")):
        sys.exit("No prediction markets in the lake yet: run `tickerlake run --stages econ_calendar,prediction_markets`.")
    releases = load_config().get("econ_calendar.releases", {}) or {}
    with LakeQuery(DATA) as q:
        cal = q.sql(f"SELECT * FROM {P.ECON_CALENDAR} WHERE snapshot_date = (SELECT MAX(snapshot_date) "
                    f"FROM {P.ECON_CALENDAR})")
        pm = q.sql(f"SELECT * FROM {P.PREDICTION_MARKETS}")
    for col in ("snapshot_date", "release_date"):
        if col in cal:
            cal[col] = pd.to_datetime(cal[col]).dt.date
    pm["snapshot_date"] = pd.to_datetime(pm["snapshot_date"]).dt.date
    pm["close_time"] = pd.to_datetime(pm["close_time"], utc=True)
    latest = pm["snapshot_date"].max()
    today = pm[pm["snapshot_date"] == latest]

    events = pd.DataFrame([summarise(g) for _, g in today.groupby(["venue", "event_id"], sort=False)])
    cal = cal[cal["release_date"] >= latest].sort_values(["release_date", "time_et"])
    odds = match_releases(cal, events, releases)

    events["underlying"] = events.apply(underlying, axis=1)
    is_asset = (events["venue"].eq("kalshi") & events["series"].isin(INDEX_SERIES)) | \
               (events["venue"].eq("polymarket") & events["series"].eq("stocks") & events["underlying"].notna())
    assets = events[is_asset].sort_values(["underlying", "close_time"])
    matched = set(odds["event_id"]) if len(odds) else set()
    other = events[events["venue"].eq("polymarket") & events["series"].isin(OTHER_TAGS)
                   & events["event_title"].str.contains(OTHER_PATTERN, case=False, regex=True, na=False)
                   & ~events["event_title"].str.match(r"Fed Decision in", na=False)  # in the fed table
                   & ~events["event_id"].isin(matched)].sort_values("volume_24h", ascending=False).head(OTHER_KEPT)

    hist = []
    for (snap, venue, eid), g in pm.groupby(["snapshot_date", "venue", "event_id"]):
        v = event_view(g)
        hist.append({"snapshot_date": snap, "venue": venue, "event_id": eid, "median": v.get("median"),
                     "top": v.get("top"), "top_prob": v.get("top_prob")})

    store.save_frame(MODULE, "calendar", cal[["release_date", "time_et", "name", "importance", "release_key"]]
                     .reset_index(drop=True))
    store.save_frame(MODULE, "odds", odds.reset_index(drop=True))
    store.save_frame(MODULE, "assets", assets.reset_index(drop=True))
    store.save_frame(MODULE, "other", other.reset_index(drop=True))
    store.save_frame(MODULE, "fed", fed_meetings(today).reset_index(drop=True))
    store.save_frame(MODULE, "history", pd.DataFrame(hist))
    store.save_json(MODULE, "summary", {
        "snapshot_date": latest.isoformat(), "snapshots": int(pm["snapshot_date"].nunique()),
        "first_snapshot": pm["snapshot_date"].min().isoformat(),
        "releases": int(len(cal)), "matched_events": int(len(odds)), "asset_events": int(len(assets)),
    })
    log(f"Published {len(cal)} upcoming releases, {len(odds)} matched market events, "
        f"{len(assets)} index/stock events ({latest}, {pm['snapshot_date'].nunique()} snapshot(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
