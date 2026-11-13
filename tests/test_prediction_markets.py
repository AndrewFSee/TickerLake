"""Tests for the prediction-market and release-calendar stages and the odds maths."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from tickerlake.analytics.prediction import event_view, ladder_from_above, ladder_from_buckets
from tickerlake.fetchers.econ_calendar import kalshi_release_dates
from tickerlake.fetchers.prediction_markets import (
    kalshi_rows,
    mid_prob,
    polymarket_rows,
    polymarket_strike,
)

DAY = date(2026, 11, 12)
NOW = datetime(2026, 11, 12, 22, 0, tzinfo=UTC)


def test_mid_prob_falls_back_on_wide_or_one_sided_quotes():
    assert mid_prob(0.40, 0.44, 0.9) == pytest.approx(0.42)
    assert mid_prob(0.10, 0.60, 0.33) == 0.33  # spread too wide to trust
    assert mid_prob(None, 0.05, 0.02) == 0.02


def test_kalshi_rows_read_dollar_fields():
    events = [{"event_ticker": "KXCPI-26NOV", "title": "CPI in November", "markets": [
        {"ticker": "KXCPI-26NOV-T0.2", "status": "active", "yes_sub_title": "Above 0.2%", "strike_type": "greater",
         "floor_strike": 0.2, "close_time": "2026-12-10T13:29:00Z", "yes_bid_dollars": "0.6100",
         "yes_ask_dollars": "0.6300", "last_price_dollars": "0.6200", "volume_fp": "1000.5",
         "open_interest_fp": "500"},
        {"ticker": "KXCPI-26NOV-T0.9", "status": "settled", "yes_sub_title": "Above 0.9%"},
    ]}]
    rows = kalshi_rows(events, "KXCPI", DAY, NOW)
    assert len(rows) == 1  # settled markets dropped
    r = rows[0]
    assert r["prob"] == pytest.approx(0.62) and r["floor_strike"] == 0.2 and r["volume"] == 1000.5
    assert r["close_time"] == pd.Timestamp("2026-12-10T13:29:00Z")


@pytest.mark.parametrize("label,title,expected", [
    ("↑ $150", "What will NVIDIA (NVDA) hit in November 2026?", ("touch_above", 150.0, None)),
    ("↓ $1,200", "What will X hit?", ("touch_below", 1200.0, None)),
    ("$140", "Will NVIDIA (NVDA) close above ___ end of November?", ("greater", 140.0, None)),
    ("$700-$720", "Micron (MU) closes week of Nov 9 at ___?", ("between", 700.0, 720.0)),
    ("<$600", "Micron (MU) closes week of Nov 9 at ___?", ("less", None, 600.0)),
    (">$780", "Micron (MU) closes week of Nov 9 at ___?", ("greater", 780.0, None)),
    ("150k+", "How many jobs added in November?", ("greater", 150000.0, None)),
    ("100k-125k", "How many jobs added in November?", ("between", 100000.0, 125000.0)),
    ("25 bps increase", "Fed Decision in December?", ("custom", None, None)),
    ("0.2%", "Core CPI MoM - November 2026", ("custom", None, None)),
])
def test_polymarket_strike(label, title, expected):
    assert polymarket_strike(label, title) == expected


def test_polymarket_rows_skip_resolved_strikes():
    events = [{"slug": "spy-week", "title": "What will S&P 500 (SPY) hit Week of November 9 2026?",
               "endDate": "2026-11-13T21:00:00Z", "markets": [
                   {"conditionId": "a", "groupItemTitle": "↑ $770", "outcomePrices": '["0.07", "0.93"]',
                    "bestBid": 0.05, "bestAsk": 0.09, "volume": "3518", "active": True, "closed": False},
                   {"conditionId": "b", "groupItemTitle": "↑ $760", "outcomePrices": '["1", "0"]',
                    "active": True, "closed": True}]}]
    rows = polymarket_rows(events, "stocks", DAY, NOW)
    assert [r["market_id"] for r in rows] == ["a"]
    assert rows[0]["prob"] == pytest.approx(0.07) and rows[0]["strike_type"] == "touch_above"


def test_ladder_median_interpolates_and_repairs_monotonicity():
    v = ladder_from_above([0.1, 0.2, 0.3, 0.4], [0.95, 0.70, 0.30, 0.05])
    assert v.median == pytest.approx(0.25)
    bumpy = ladder_from_above([1, 2, 3], [0.8, 0.82, 0.2])  # 2 priced above 1: quoting noise
    assert all(a >= b for a, b in zip(bumpy.p_above, bumpy.p_above[1:], strict=False))
    assert ladder_from_above([1, 2], [0.3, 0.1]).below_range
    assert ladder_from_above([1, 2], [0.9, 0.7]).above_range


def test_bucket_ladder_normalises_and_finds_median():
    # tails plus three ranges; prices sum to 1.1, as independently quoted buckets do
    v = ladder_from_buckets([None, 100, 110, 120, 130], [100, 110, 120, 130, None], [0.1, 0.2, 0.4, 0.3, 0.1])
    assert 110 < v.median < 120


def test_event_view_kinds():
    ladder = pd.DataFrame({"strike_type": "greater", "floor_strike": [0.1, 0.2, 0.3], "cap_strike": None,
                           "prob": [0.9, 0.5, 0.1], "outcome": ["a", "b", "c"]})
    assert event_view(ladder)["median"] == pytest.approx(0.2)
    fed = pd.DataFrame({"strike_type": "custom", "floor_strike": None, "cap_strike": None,
                        "prob": [0.88, 0.1, 0.02], "outcome": ["Hike 25bps", "Hike >25bps", "Maintains"]})
    view = event_view(fed)
    assert view["kind"] == "outcomes" and view["top"] == "Hike 25bps"
    single = pd.DataFrame({"strike_type": "greater", "floor_strike": [0.0], "cap_strike": None, "prob": [0.31],
                           "outcome": ["Above 0"]})
    assert event_view(single) == {"kind": "binary", "top": "Yes", "top_prob": 0.31}
    levels = pd.DataFrame({"strike_type": "custom", "floor_strike": None, "cap_strike": None, "exclusive": False,
                           "prob": [0.52, 0.2], "outcome": ["5.3%", "5.5%"]})
    assert event_view(levels) == {"kind": "multi", "outcomes": ["5.3%", "5.5%"], "probs": [0.52, 0.2]}
    buckets = pd.DataFrame({"strike_type": ["less", "between", "between", "greater"],
                            "floor_strike": [None, 600.0, 620.0, 640.0], "cap_strike": [600.0, 620.0, 640.0, None],
                            "prob": [0.1, 0.2, 0.3, 0.4], "outcome": ["<600", "600-620", "620-640", ">640"]})
    assert 620 <= event_view(buckets)["median"] <= 640
    touch = pd.DataFrame({"strike_type": ["touch_above", "touch_below"], "floor_strike": [770.0, 700.0],
                          "cap_strike": None, "prob": [0.07, 0.2], "outcome": ["↑ $770", "↓ $700"]})
    assert event_view(touch) == {"kind": "touch", "up": [(770.0, 0.07)], "down": [(700.0, 0.2)]}


def test_kalshi_release_dates_use_new_york_date():
    events = [{"markets": [{"close_time": "2026-12-09T18:59:00Z"}, {"close_time": "2026-12-09T19:00:00Z"}]},
              {"markets": [{"close_time": "2027-01-28T02:00:00Z"}]}]  # 9 PM New York on Jan 27
    assert kalshi_release_dates(events) == [date(2026, 12, 9), date(2027, 1, 27)]
