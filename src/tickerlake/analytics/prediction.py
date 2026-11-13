"""What a set of prediction-market prices implies about a number.

A Kalshi CPI event is a ladder of binary markets, "Above 0.1%", "Above 0.2%", ...,
each priced as the probability the print exceeds its strike. Read together they are
the market's distribution for the print. Prices are quoted independently, so the
ladder can be slightly non-monotone; it is made monotone before use (a higher
strike can never be likelier to be exceeded).

Pure functions on plain frames: no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ABOVE = {"greater", "greater_or_equal"}
BELOW = {"less", "less_or_equal"}


@dataclass
class LadderView:
    median: float | None        # strike where P(above) crosses 50%
    below_range: bool           # the median sits under the lowest strike
    above_range: bool           # ...or over the highest
    strikes: list[float]
    p_above: list[float]        # monotone P(value > strike), one per strike


def _monotone_above(p: np.ndarray) -> np.ndarray:
    """Non-increasing in strike: average of the running min from the left and the running
    max from the right, which keeps a well-behaved ladder unchanged."""
    left = np.minimum.accumulate(p)
    right = np.maximum.accumulate(p[::-1])[::-1]
    return np.clip((left + right) / 2, 0, 1)


def ladder_from_above(strikes, p_above) -> LadderView:
    """Median and monotone curve from "above strike" probabilities."""
    df = pd.DataFrame({"k": strikes, "p": p_above}).dropna().sort_values("k").drop_duplicates("k")
    if df.empty:
        return LadderView(None, False, False, [], [])
    k, p = df["k"].to_numpy(float), _monotone_above(df["p"].to_numpy(float))
    if p[0] < 0.5:
        return LadderView(None, True, False, k.tolist(), p.tolist())
    if p[-1] > 0.5:
        return LadderView(None, False, True, k.tolist(), p.tolist())
    i = int(np.argmax(p <= 0.5))  # first strike at or below 50%
    if i == 0 or p[i] == p[i - 1]:
        median = float(k[i])
    else:
        median = float(k[i - 1] + (p[i - 1] - 0.5) / (p[i - 1] - p[i]) * (k[i] - k[i - 1]))
    return LadderView(median, False, False, k.tolist(), p.tolist())


def ladder_from_buckets(floors, caps, probs) -> LadderView:
    """Same view from range buckets (Kalshi's index ranges: "between", plus open tails).

    A bucket's probability counts as mass above its floor; tails with no floor sit
    below everything. Probabilities are normalised to sum to one first, since the
    buckets' prices are quoted independently."""
    df = pd.DataFrame({"lo": floors, "hi": caps, "p": probs}).dropna(subset=["p"])
    total = df["p"].sum()
    if df.empty or total <= 0:
        return LadderView(None, False, False, [], [])
    df["p"] = df["p"] / total
    df["edge"] = df["lo"].fillna(-np.inf)
    edges = sorted(e for e in df["edge"].unique() if np.isfinite(e))
    p_above = [float(df.loc[df["edge"] >= e, "p"].sum()) for e in edges]
    return ladder_from_above(edges, p_above)


def event_view(rows: pd.DataFrame) -> dict:
    """Summary of one event's outcomes: a ladder (median) or named outcomes (probabilities)."""
    rows = rows.dropna(subset=["prob"])
    kinds = set(rows["strike_type"].dropna())
    if rows.empty:
        return {"kind": "empty"}
    if len(rows) == 1:  # one yes/no market: its price is the answer
        return {"kind": "binary", "top": "Yes", "top_prob": float(rows["prob"].iloc[0])}
    if kinds and kinds <= ABOVE and rows["floor_strike"].notna().all():
        v = ladder_from_above(rows["floor_strike"], rows["prob"])
        return {"kind": "ladder", "median": v.median, "below_range": v.below_range,
                "above_range": v.above_range, "strikes": v.strikes, "p_above": v.p_above}
    if kinds and kinds <= {"touch_above", "touch_below"}:
        # "Will it trade at $X before expiry": each strike stands alone, no distribution
        up = rows[rows["strike_type"] == "touch_above"].sort_values("floor_strike")
        down = rows[rows["strike_type"] == "touch_below"].sort_values("floor_strike", ascending=False)
        return {"kind": "touch",
                "up": list(zip(up["floor_strike"].tolist(), up["prob"].tolist(), strict=True)),
                "down": list(zip(down["floor_strike"].tolist(), down["prob"].tolist(), strict=True))}
    if "between" in kinds or (kinds and kinds <= ABOVE | BELOW and len(kinds) > 1):
        v = ladder_from_buckets(rows["floor_strike"], rows["cap_strike"], rows["prob"])
        return {"kind": "ladder", "median": v.median, "below_range": v.below_range,
                "above_range": v.above_range, "strikes": v.strikes, "p_above": v.p_above}
    if "exclusive" in rows and not rows["exclusive"].fillna(False).any() and len(rows) > 1:
        # independent yes/no markets ("how high will the 10-year go"): no single answer
        each = rows.set_index("outcome")["prob"]
        return {"kind": "multi", "outcomes": each.index.tolist(), "probs": each.tolist()}
    total = rows["prob"].sum()
    probs = (rows.set_index("outcome")["prob"] / total) if total > 1.05 or total < 0.95 else rows.set_index("outcome")["prob"]
    probs = probs.groupby(level=0).sum().sort_values(ascending=False)
    return {"kind": "outcomes", "outcomes": probs.index.tolist(), "probs": probs.tolist(),
            "top": probs.index[0], "top_prob": float(probs.iloc[0]), "normalised": bool(total > 1.05 or total < 0.95)}
