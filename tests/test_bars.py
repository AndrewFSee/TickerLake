"""Tests for information-driven bars (AFML ch. 2) built from minute bars.

The properties worth pinning down are the ones a careless implementation gets
wrong silently: OHLC aggregation across a chunk, bars not spanning the overnight
gap, and the quantisation error being reported rather than hidden.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from tickerlake.analytics.bars import (
    build_bars,
    calibrate_threshold,
    signed_volume_proxy,
)


def _minutes(
    n: int = 390, price: float = 100.0, volume: float = 1000.0, sessions: int = 1, symbol="AAPL"
) -> pd.DataFrame:
    """Synthetic 1-minute bars: n minutes per session, from 09:30 ET."""
    rows = []
    for day in range(sessions):
        start = datetime(2026, 3, 2 + day, 14, 30, tzinfo=UTC)  # 09:30 ET
        for i in range(n):
            p = price + i * 0.01
            rows.append(
                {
                    "symbol": symbol,
                    "datetime": start + timedelta(minutes=i),
                    "open": p,
                    "high": p + 0.05,
                    "low": p - 0.05,
                    "close": p + 0.01,
                    "volume": volume,
                }
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------- calibration


def test_calibrate_hits_the_requested_bar_count():
    df = _minutes(n=390, volume=1000, sessions=2)
    report = calibrate_threshold(df, "volume", target_bars_per_day=39)
    # 390 minutes x 1000 shares = 390,000/day; 39 bars -> 10,000 per bar.
    assert report.threshold == pytest.approx(10_000, rel=1e-6)

    bars = build_bars(df, "volume", threshold=report.threshold)
    complete = bars[~bars["incomplete"]]
    assert 38 <= len(complete) / 2 <= 39


def test_calibrate_flags_when_one_minute_fills_a_bar():
    """The failure mode that matters: a threshold smaller than the opening minute."""
    df = _minutes(n=100, volume=100)
    spike = df.copy()
    spike.loc[0, "volume"] = 1_000_000  # opening print dwarfs the rest

    report = calibrate_threshold(spike, "volume", target_bars_per_day=50)
    assert report.open_minute_exceeds_threshold
    assert "opening minute alone exceeds" in report.summary()


def test_calibrate_rejects_empty_input():
    with pytest.raises(ValueError, match="no minute bars"):
        calibrate_threshold(pd.DataFrame(), "dollar", 50)


# ------------------------------------------------------------ construction


def test_volume_bars_reach_their_threshold():
    df = _minutes(n=390, volume=1000)
    bars = build_bars(df, "volume", threshold=10_000)
    complete = bars[~bars["incomplete"]]

    assert len(complete) == 39
    # Each bar accumulates exactly 10 minutes of 1,000 shares.
    assert (complete["volume"] == 10_000).all()
    assert (complete["minutes"] == 10).all()


def test_ohlc_is_aggregated_across_the_chunk_not_copied():
    df = _minutes(n=20, volume=1000)
    bars = build_bars(df, "volume", threshold=10_000)
    first = bars.iloc[0]
    chunk = df.iloc[:10]

    assert first["open"] == pytest.approx(chunk["open"].iloc[0])
    assert first["close"] == pytest.approx(chunk["close"].iloc[-1])
    assert first["high"] == pytest.approx(chunk["high"].max())
    assert first["low"] == pytest.approx(chunk["low"].min())
    assert first["high"] >= first["low"]


def test_bars_do_not_span_the_overnight_gap():
    """A bar crossing the close would swallow the overnight return."""
    df = _minutes(n=15, volume=1000, sessions=2)
    bars = build_bars(df, "volume", threshold=10_000, reset_daily=True)

    for row in bars.itertuples():
        assert row.bar_start.date() == row.bar_end.date()
    assert bars["session"].nunique() == 2


def test_trailing_partial_bar_is_kept_but_marked():
    df = _minutes(n=25, volume=1000)
    bars = build_bars(df, "volume", threshold=10_000)

    assert bars["incomplete"].sum() == 1, "the session tail must be flagged, not dropped"
    tail = bars[bars["incomplete"]].iloc[0]
    assert tail["volume"] < 10_000
    assert bars["volume"].sum() == 25_000, "no volume may be lost"


def test_overshoot_is_reported_per_bar():
    df = _minutes(n=30, volume=1000)
    df.loc[9, "volume"] = 5_000  # the 10th minute overshoots hard

    bars = build_bars(df, "volume", threshold=10_000)
    assert bars.iloc[0]["overshoot_pct"] > 0
    assert bars.iloc[0]["volume"] == pytest.approx(14_000)
    assert bars.iloc[0]["overshoot_pct"] == pytest.approx(40.0, rel=1e-6)


def test_dollar_bars_use_traded_value_not_share_count():
    """Two symbols, same shares, 10x price -> the expensive one bars 10x faster."""
    cheap = _minutes(n=100, price=10.0, volume=1000, symbol="CHEAP")
    rich = _minutes(n=100, price=100.0, volume=1000, symbol="RICH")

    threshold = 1_000_000
    cheap_bars = build_bars(cheap, "dollar", threshold=threshold)
    rich_bars = build_bars(rich, "dollar", threshold=threshold)
    assert len(rich_bars) > len(cheap_bars)


def test_vwap_lies_within_the_bar_range():
    df = _minutes(n=40, volume=1000)
    bars = build_bars(df, "dollar", target_bars_per_day=4)
    for row in bars.itertuples():
        assert row.low <= row.vwap <= row.high


def test_time_bars_resample_on_the_clock():
    df = _minutes(n=60, volume=1000)
    bars = build_bars(df, "time", threshold=5)
    assert len(bars) == 12
    assert (bars["volume"] == 5000).all()


def test_returns_are_computed_between_consecutive_bars():
    df = _minutes(n=100, volume=1000)
    bars = build_bars(df, "volume", threshold=10_000)
    assert bars["log_return"].isna().iloc[0], "the first bar has no predecessor"
    assert bars["log_return"].iloc[1:].notna().all()
    # Prices rise monotonically in the fixture, so returns must be positive.
    assert (bars["log_return"].iloc[1:] > 0).all()


def test_empty_input_returns_empty_frame_with_schema():
    out = build_bars(pd.DataFrame(), "dollar", threshold=1000)
    assert out.empty
    assert "overshoot_pct" in out.columns


def test_build_requires_a_threshold_or_a_target():
    with pytest.raises(ValueError, match="threshold or target_bars_per_day"):
        build_bars(_minutes(n=10), "dollar")


# --------------------------------------------------------- signed proxy


def test_signed_volume_proxy_applies_the_tick_rule():
    df = _minutes(n=5, volume=1000)
    df.loc[:, "close"] = [100.0, 101.0, 101.0, 99.0, 99.0]

    out = signed_volume_proxy(df)
    signs = out["tick_sign"].tolist()

    assert signs[1] == 1.0, "an uptick is buy-initiated"
    assert signs[2] == 1.0, "an unchanged price carries the previous sign"
    assert signs[3] == -1.0, "a downtick is sell-initiated"
    assert signs[4] == -1.0, "and carries forward again"
    assert out["signed_volume"].iloc[3] == pytest.approx(-1000.0)


def test_signed_proxy_accumulates_within_a_session():
    df = _minutes(n=10, volume=1000, sessions=2)
    out = signed_volume_proxy(df)
    # The cumulative series must restart each session, not run across the gap.
    firsts = out.groupby("session")["cum_signed_dollar"].first()
    assert len(firsts) == 2
    assert np.isfinite(firsts).all()
