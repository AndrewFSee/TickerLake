"""Information-driven bars (Lopez de Prado, AFML ch. 2) from 1-minute bars.

What is and is not possible from minute bars
--------------------------------------------
AFML samples bars on *information arrival* rather than clock time. Which of its
bar types can be reconstructed depends entirely on what the source carries.

============================  ==========================  ==================
Bar type                      Requires                    From 1m bars?
============================  ==========================  ==================
Time bars                     clock                       yes (they are one)
Volume bars                   cumulative shares           approximate
Dollar bars                   cumulative traded value     approximate
Tick bars                     trade count per period      no - not published
Imbalance bars (TIB/VIB/DIB)  signed trade *sequence*     no
Run bars (TRB/VRB/DRB)        signed trade *sequence*     no
============================  ==========================  ==================

Volume and dollar bars are *approximate* because a bar can only close on a
minute boundary. The realised bar overshoots its threshold by however much the
final minute contributed, and that overshoot is worst exactly where it hurts
most: intraday volume is roughly 19x heavier at the open than at midday, so a
threshold sized for ~50 bars/day is exceeded by the opening minute on its own.
``calibrate_threshold`` reports this, and every bar carries ``overshoot_pct`` so
the error is inspectable per row rather than assumed away.

Imbalance and run bars are not approximated here, deliberately. They rest on the
tick rule -- ``b_t = b_{t-1} if dp_t == 0 else sign(dp_t)`` applied per trade --
and a minute bar exposes only the net change across ~60 seconds. Signing a whole
minute by its close-to-close direction produces a number that looks like signed
order flow while discarding the intra-minute sequence those bars exist to
measure. ``signed_volume_proxy`` computes that bar-level approximation under a
name that does not pretend otherwise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BarKind = Literal["dollar", "volume", "time"]


@dataclass
class ThresholdReport:
    """What a chosen threshold will actually produce, before building anything."""

    symbol: str
    kind: BarKind
    threshold: float
    target_bars_per_day: int
    avg_daily_total: float
    median_minute: float
    max_minute: float
    typical_overshoot_pct: float
    open_overshoot_pct: float

    @property
    def open_minute_exceeds_threshold(self) -> bool:
        """True when a single opening minute can fill an entire bar by itself."""
        return self.max_minute >= self.threshold

    def summary(self) -> str:
        warn = (
            "  <-- opening minute alone exceeds the threshold"
            if self.open_minute_exceeds_threshold
            else ""
        )
        return (
            f"{self.symbol} {self.kind}: threshold={self.threshold:,.0f} "
            f"targets ~{self.target_bars_per_day} bars/day, "
            f"overshoot ~{self.typical_overshoot_pct:.1f}% typical / "
            f"{self.open_overshoot_pct:.0f}% at the open{warn}"
        )


def calibrate_threshold(
    minute_bars: pd.DataFrame, kind: BarKind = "dollar", target_bars_per_day: int = 50
) -> ThresholdReport:
    """Pick a threshold for a target bar count, and quantify the quantisation cost.

    Call this before building bars. A threshold that looks reasonable on daily
    averages can still be smaller than a single opening minute, in which case the
    most information-dense part of the session collapses into one bar.
    """
    df = _prepare(minute_bars)
    if df.empty:
        raise ValueError("no minute bars supplied")

    column = "dollar" if kind == "dollar" else "volume"
    per_day = df.groupby("session")[column].sum()
    per_minute = df[column]

    avg_daily = float(per_day.mean())
    threshold = avg_daily / max(target_bars_per_day, 1)
    median_minute = float(per_minute.median())
    max_minute = float(per_minute.max())

    report = ThresholdReport(
        symbol=str(df["symbol"].iloc[0]) if "symbol" in df.columns else "?",
        kind=kind,
        threshold=threshold,
        target_bars_per_day=target_bars_per_day,
        avg_daily_total=avg_daily,
        median_minute=median_minute,
        max_minute=max_minute,
        # Expected overshoot is about half the contributing minute.
        typical_overshoot_pct=100 * median_minute / (2 * threshold),
        open_overshoot_pct=100 * max_minute / (2 * threshold),
    )
    if report.open_minute_exceeds_threshold:
        log.warning(
            "%s: a single minute (%.0f) exceeds the %s threshold (%.0f); "
            "the open will collapse into one bar. Use fewer bars/day, or tick data.",
            report.symbol,
            max_minute,
            kind,
            threshold,
        )
    return report


def build_bars(
    minute_bars: pd.DataFrame,
    kind: BarKind = "dollar",
    threshold: float | None = None,
    target_bars_per_day: int | None = None,
    reset_daily: bool = True,
) -> pd.DataFrame:
    """Aggregate 1-minute bars into volume, dollar, or time bars.

    ``reset_daily`` closes any open bar at the session end rather than carrying
    the accumulator overnight. Carrying it would let a bar span the close, and
    the overnight gap would land *inside* a bar instead of between two -- which
    breaks the return series the bars exist to produce.
    """
    df = _prepare(minute_bars)
    if df.empty:
        return _empty_bars()

    if threshold is None:
        if target_bars_per_day is None:
            raise ValueError("supply either threshold or target_bars_per_day")
        threshold = calibrate_threshold(df, kind, target_bars_per_day).threshold

    if kind == "time":
        return _time_bars(df, int(threshold))

    column = "dollar" if kind == "dollar" else "volume"
    groups = df.groupby("session", sort=True) if reset_daily else [(None, df)]

    bars: list[dict] = []
    for _, session_df in groups:
        bars.extend(_accumulate(session_df, column, float(threshold), kind))

    out = pd.DataFrame(bars)
    if out.empty:
        return _empty_bars()

    out["bar_return"] = out.groupby("symbol")["close"].pct_change()
    out["log_return"] = np.log(out["close"] / out.groupby("symbol")["close"].shift(1))
    log.info(
        "built %d %s bars from %d minute bars (%.1f bars/session)",
        len(out),
        kind,
        len(df),
        len(out) / max(df["session"].nunique(), 1),
    )
    return out


def _accumulate(df: pd.DataFrame, column: str, threshold: float, kind: BarKind) -> list[dict]:
    """Walk minute bars, closing a bar each time the accumulator crosses."""
    bars: list[dict] = []
    start = 0
    total = 0.0

    values = df[column].to_numpy()
    for i in range(len(df)):
        total += values[i]
        if total >= threshold:
            bars.append(_make_bar(df.iloc[start : i + 1], total, threshold, kind))
            start = i + 1
            total = 0.0

    # Trailing partial bar: kept, but marked, because dropping it silently loses
    # the end of every session and keeping it unmarked pollutes the size
    # distribution with an undersized bar.
    if start < len(df):
        tail = df.iloc[start:]
        bar = _make_bar(tail, float(tail[column].sum()), threshold, kind)
        bar["incomplete"] = True
        bars.append(bar)
    return bars


def _make_bar(chunk: pd.DataFrame, realised: float, threshold: float, kind: BarKind) -> dict:
    volume = float(chunk["volume"].sum())
    dollar = float(chunk["dollar"].sum())
    return {
        "symbol": chunk["symbol"].iloc[0],
        "bar_kind": kind,
        "threshold": threshold,
        "bar_start": chunk["datetime"].iloc[0],
        "bar_end": chunk["datetime"].iloc[-1],
        "session": chunk["session"].iloc[0],
        "open": float(chunk["open"].iloc[0]),
        "high": float(chunk["high"].max()),
        "low": float(chunk["low"].min()),
        "close": float(chunk["close"].iloc[-1]),
        "volume": volume,
        "dollar_volume": dollar,
        # VWAP from minute typical prices. Not a true VWAP -- that needs trades --
        # but materially better than using close as the bar's representative price.
        "vwap": (dollar / volume) if volume > 0 else float(chunk["close"].iloc[-1]),
        "minutes": int(len(chunk)),
        # How far past the threshold this bar actually closed. The cost of
        # minute-boundary quantisation, per bar, rather than as an average.
        "overshoot_pct": 100.0 * (realised - threshold) / threshold if threshold > 0 else 0.0,
        "incomplete": False,
    }


def _time_bars(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Plain resampling, for a like-for-like baseline against the other kinds."""
    out = (
        df.set_index("datetime")
        .groupby("symbol")
        .resample(f"{minutes}min")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            dollar_volume=("dollar", "sum"),
            minutes=("close", "size"),
        )
        .dropna(subset=["close"])
        .reset_index()
    )
    out["bar_kind"] = "time"
    out["threshold"] = minutes
    out["bar_start"] = out["datetime"]
    out["bar_end"] = out["datetime"]
    out["vwap"] = np.where(out["volume"] > 0, out["dollar_volume"] / out["volume"], out["close"])
    out["overshoot_pct"] = 0.0
    out["incomplete"] = False
    out["bar_return"] = out.groupby("symbol")["close"].pct_change()
    out["log_return"] = np.log(out["close"] / out.groupby("symbol")["close"].shift(1))
    return out


def signed_volume_proxy(minute_bars: pd.DataFrame) -> pd.DataFrame:
    """Bar-level tick rule: a *proxy* for signed order flow, not the real thing.

    AFML's imbalance and run bars sign each **trade** and accumulate the
    imbalance. Here the finest unit available is a minute, so the sign comes from
    that minute's close-to-close direction (carrying the previous sign when
    unchanged, per the tick rule) and the whole minute's volume takes it.

    A minute in which 60% of volume lifted the offer and 40% hit the bid is
    recorded as 100% buy-initiated. The imbalance those bars are designed to
    detect happens *inside* that minute and cannot be recovered afterwards, so
    treat this as a coarse order-flow feature and not as VIB/DIB input.
    """
    df = _prepare(minute_bars)
    if df.empty:
        return df

    out = []
    for _symbol, group in df.groupby("symbol", sort=True):
        group = group.sort_values("datetime").copy()
        delta = group["close"].diff()
        sign = np.sign(delta).replace(0.0, np.nan).ffill().fillna(1.0)
        group["tick_sign"] = sign
        group["signed_volume"] = sign * group["volume"]
        group["signed_dollar"] = sign * group["dollar"]
        group["cum_signed_dollar"] = group.groupby("session")["signed_dollar"].cumsum()
        out.append(group)
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ helpers


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise input and derive per-minute traded value."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True)
    if "session" not in out.columns:
        source = out["date"] if "date" in out.columns else out["datetime"]
        out["session"] = (
            pd.to_datetime(source, utc=True, errors="coerce")
            .dt.tz_convert("America/New_York")
            .dt.date
        )
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Typical price rather than close: a minute's traded value is better
    # represented by (H+L+C)/3 than by its final print.
    typical = (out["high"] + out["low"] + out["close"]) / 3.0
    out["dollar"] = typical * out["volume"]

    out = out.dropna(subset=["datetime", "close", "volume"])
    return out.sort_values(["symbol", "datetime"]).reset_index(drop=True)


def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "symbol",
            "bar_kind",
            "threshold",
            "bar_start",
            "bar_end",
            "session",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "dollar_volume",
            "vwap",
            "minutes",
            "overshoot_pct",
            "incomplete",
            "bar_return",
            "log_return",
        ]
    )
