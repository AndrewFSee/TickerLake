"""Tests for the Yahoo analyst stage: estimate snapshots and broker rating changes.

The estimates matter only as a daily snapshot (Yahoo keeps no history), the
rating history matters most for names that have since left the index, and
neither may depend on the network here: yfinance is replaced by fakes.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pyarrow.parquet as pq
import pytest

from tickerlake.config import Config, Secrets
from tickerlake.fetchers import yf_analyst
from tickerlake.fetchers.base import FetchResult
from tickerlake.fetchers.yf_analyst import YFinanceAnalystFetcher, estimate_rows, rating_rows
from tickerlake.storage import paths as P
from tickerlake.storage.paths import DatasetPaths
from tickerlake.storage.writer import ParquetWriter
from tickerlake.universe.membership import MembershipTracker

RUN = date(2026, 11, 6)
NOW = datetime(2026, 11, 6, 22, 0, tzinfo=UTC)


def _frames(scale=1.0):
    idx = ["0q", "+1q", "0y", "+1y"]
    est = pd.DataFrame({"avg": [2.9, 2.1, 9.4, 10.5], "low": [2.8, 1.9, 8.3, 8.6],
                        "high": [3.2, 2.5, 10.0, 11.6], "yearAgoEps": [1.85, 2.0, 7.5, 8.8],
                        "numberOfAnalysts": [27, 23, 38, 34], "growth": [0.57, 0.07, 0.26, 0.19]}, index=idx)
    trend = pd.DataFrame({"current": [2.92 * scale, 2.15, 8.84, 10.53], "7daysAgo": [2.91, 2.22, 8.84, 10.83],
                          "30daysAgo": [2.90, 2.23, 8.83, 10.83], "60daysAgo": [2.91, 2.24, 8.82, 10.80],
                          "90daysAgo": [2.90, 2.21, 8.80, 10.73]}, index=idx)
    rev = pd.DataFrame({"upLast7days": [2, 1, 1, 2], "upLast30days": [3, 3, 3, 4],
                        "downLast30days": [3, 5, 5, 2], "downLast7Days": [1, 2, 2, 0]}, index=idx)
    return est, trend, rev


class FakeTicker:
    def __init__(self, symbol, ratings=None, fail=None):
        self.symbol = symbol
        self.earnings_estimate, self.eps_trend, self.eps_revisions = _frames()
        self._ratings, self._fail = ratings, fail

    @property
    def upgrades_downgrades(self):
        if self._fail:
            raise self._fail
        return self._ratings


def _ratings_frame(*stamps):
    return pd.DataFrame(
        {"Firm": ["Morgan Stanley"] * len(stamps), "ToGrade": ["Overweight"] * len(stamps),
         "FromGrade": ["Equal-Weight"] * len(stamps), "Action": ["up"] * len(stamps),
         "priceTargetAction": ["Raises"] * len(stamps), "currentPriceTarget": [24.0] * len(stamps),
         "priorPriceTarget": [22.0] * len(stamps)},
        index=pd.Index(pd.to_datetime(list(stamps)), name="GradeDate"),
    )


@pytest.fixture
def lake(tmp_path):
    paths = DatasetPaths(tmp_path)
    paths.ensure_layout()
    return paths, ParquetWriter()


@pytest.fixture
def tracker(lake):
    paths, writer = lake
    t = MembershipTracker(paths, writer, collect_indices=["SP500", "ETF"])
    t.seed(
        pd.DataFrame(
            [{"symbol": "AAPL", "start_date": date(2010, 1, 1), "end_date": None},
             {"symbol": "GAS", "start_date": date(2010, 1, 1), "end_date": date(2016, 7, 1)}],
            columns=["symbol", "start_date", "end_date"],
        ),
        force=True,
    )
    t.register_static({"VXX": "iPath VIX futures ETN"}, "ETF")
    return t


def _config(root, **analyst):
    return Config(raw={"analyst": {"enabled": True, "throttle_seconds": 0, **analyst}},
                  secrets=Secrets(), data_root=root, config_path=root / "config.yaml")


def test_estimate_rows_map_yahoo_columns():
    est, trend, rev = _frames()
    rows = estimate_rows("AAPL", est, trend, rev, {"0y": "2027-09-30"}, RUN, NOW)
    assert [r["period"] for r in rows] == ["0q", "+1q", "0y", "+1y"]
    fy = rows[2]
    assert fy["period_end"] == date(2027, 9, 30) and fy["analysts"] == 38
    assert fy["eps_trend_current"] == 8.84 and fy["eps_trend_90d"] == 8.80
    assert fy["up_30d"] == 3 and fy["down_30d"] == 5 and fy["down_7d"] == 2  # Yahoo's mixed casing
    assert estimate_rows("X", None, None, None, {}, RUN, NOW) == []  # nothing covered -> no rows


def test_rating_rows_are_utc_and_keep_targets():
    rows = rating_rows("AAL", _ratings_frame("2026-10-28 22:55:57"), NOW)
    assert rows[0]["graded_at"] == datetime(2026, 10, 28, 22, 55, 57, tzinfo=UTC)
    assert rows[0]["price_target"] == 24.0 and rows[0]["action"] == "up"
    assert rating_rows("AAL", pd.DataFrame(), NOW) == []
    no_target = _ratings_frame("2013-12-18 21:00").assign(currentPriceTarget=0.0, priorPriceTarget=0.0)
    assert rating_rows("AAL", no_target, NOW)[0]["price_target"] is None  # Yahoo's 0 = no target


def test_daily_run_snapshots_members_and_merges_ratings(lake, tracker, monkeypatch):
    paths, writer = lake
    calls = []

    def fake(symbol):
        calls.append(symbol)
        return FakeTicker(symbol, ratings=_ratings_frame("2026-10-01 12:00", "2026-11-02 13:30"))

    monkeypatch.setattr(yf_analyst.yf, "Ticker", fake)
    f = YFinanceAnalystFetcher(_config(paths.root), paths, writer, tracker=tracker)
    result = FetchResult(stage="analyst")
    f.collect(RUN, result)
    f.collect(RUN, result)  # a re-run the same day must not duplicate anything

    est = pq.read_table(paths.analyst_file(P.ANALYST_ESTIMATES)).to_pandas()
    assert set(est["symbol"]) == {"AAPL"}, "companies only: no removed names, no funds"
    assert len(est) == 4
    ratings = pq.read_table(paths.analyst_file(P.ANALYST_RATINGS)).to_pandas()
    assert len(ratings) == 2
    assert "VXX" not in calls


def test_backfill_reaches_removed_members_and_skips_snapshots(lake, tracker, monkeypatch):
    paths, writer = lake
    monkeypatch.setattr(yf_analyst.yf, "Ticker",
                        lambda s: FakeTicker(s, ratings=_ratings_frame("2014-03-03 14:00")))
    f = YFinanceAnalystFetcher(_config(paths.root), paths, writer, tracker=tracker, backfill=True)
    f.collect(RUN, FetchResult(stage="analyst"))
    ratings = pq.read_table(paths.analyst_file(P.ANALYST_RATINGS)).to_pandas()
    assert set(ratings["symbol"]) == {"AAPL", "GAS"}, "the removed name is what avoids survivorship"
    assert not paths.analyst_file(P.ANALYST_ESTIMATES).exists()


def test_no_coverage_is_an_answer_and_rate_limits_are_retried(lake, tracker, monkeypatch):
    paths, writer = lake
    attempts = {"n": 0}

    class RateLimited(Exception):
        pass

    def fake(symbol):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return FakeTicker(symbol, fail=RateLimited("Too Many Requests. Rate limited."))
        return FakeTicker(symbol, fail=ValueError("No upgrade/downgrade history found for AAPL"))

    monkeypatch.setattr(yf_analyst.yf, "Ticker", fake)
    monkeypatch.setattr("tickerlake.utils.throttle.time.sleep", lambda s: None)
    f = YFinanceAnalystFetcher(_config(paths.root, fetch_estimates=False), paths, writer,
                               tracker=tracker, symbols_override=["AAPL"])
    result = FetchResult(stage="analyst")
    f.collect(RUN, result)
    assert result.items_succeeded == 1 and result.items_failed == 0
    assert result.details["ratings_empty"] == 1
