"""Tests for the volatility stage: CBOE index levels and the VIX futures curve.

The expiration dates, curve values and file shapes below are CBOE's own, taken
from the files the stage reads.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from tickerlake.config import Config, Secrets
from tickerlake.fetchers.volatility import (
    VolatilityFetcher,
    lake_symbol,
    monthly_expirations,
    vix_expiration,
)
from tickerlake.storage.paths import DatasetPaths
from tickerlake.storage.writer import ParquetWriter

NOW = datetime(2026, 9, 25, 22, tzinfo=UTC)


@pytest.fixture
def fetcher(tmp_path):
    paths = DatasetPaths(tmp_path)
    paths.ensure_layout()
    config = Config(
        raw={"volatility": {"enabled": True, "daily_lookback_days": 10}},
        secrets=Secrets(),
        data_root=tmp_path,
        config_path=tmp_path / "config.yaml",
    )
    return VolatilityFetcher(config, paths, ParquetWriter())


# ------------------------------------------------------------ expirations


@pytest.mark.parametrize(
    ("year", "month", "expected"),
    [
        # Every one of these resolved to a real CBOE contract file.
        (2026, 10, date(2026, 10, 21)),
        (2026, 11, date(2026, 11, 18)),
        (2026, 12, date(2026, 12, 16)),
        (2027, 1, date(2027, 1, 20)),
        (2027, 2, date(2027, 2, 17)),
        (2027, 3, date(2027, 3, 17)),
        (2027, 4, date(2027, 4, 21)),
        (2015, 1, date(2015, 1, 21)),
        (2018, 2, date(2018, 2, 14)),
        (2020, 3, date(2020, 3, 18)),
        (2024, 3, date(2024, 3, 20)),
        (2026, 4, date(2026, 4, 15)),
    ],
)
def test_vix_expiration_matches_cboe(year, month, expected):
    assert vix_expiration(year, month) == expected


def test_a_good_friday_moves_expiration_to_a_tuesday():
    """March 2025: April's third Friday was Good Friday.

    The count starts from the Thursday before it instead, which lands the
    settlement on Tuesday 18 March rather than the usual Wednesday.
    """
    exp = vix_expiration(2025, 3)
    assert exp == date(2025, 3, 18)
    assert exp.weekday() == 1


def test_expirations_are_monthly_and_bounded():
    got = monthly_expirations(date(2026, 9, 25), date(2027, 1, 31))
    assert got == [date(2026, 10, 21), date(2026, 11, 18), date(2026, 12, 16), date(2027, 1, 20)]


def test_lake_symbol_marks_an_index():
    """The caret keeps an index from ever colliding with a ticker."""
    assert lake_symbol("vix") == "^VIX"
    assert lake_symbol("^VIX") == "^VIX"


# ------------------------------------------------------------ index files


def test_an_ohlc_history_file_parses():
    text = (
        "DATE,OPEN,HIGH,LOW,CLOSE\n"
        "09/24/2026,15.830000,16.570000,15.340000,15.670000\n"
        "09/25/2026,15.610000,15.940000,14.680000,14.870000\n"
    )
    df = VolatilityFetcher.parse_cboe_index(text, "^VIX", NOW)
    assert list(df["date"]) == [date(2026, 9, 24), date(2026, 9, 25)]
    last = df.iloc[-1]
    assert (last["open"], last["high"], last["low"], last["close"]) == (15.61, 15.94, 14.68, 14.87)
    assert last["adj_close"] == last["close"], "an index has nothing to adjust for"
    assert pd.isna(last["volume"]), "and no volume: null, not zero"
    assert last["source"] == "cboe"


def test_a_single_value_history_file_parses():
    """VVIX, SKEW and OVX publish one number a day."""
    text = "DATE,VVIX\n09/24/2026,86.100000\n09/25/2026,87.840000\n"
    df = VolatilityFetcher.parse_cboe_index(text, "^VVIX", NOW)
    last = df.iloc[-1]
    assert last["open"] == last["high"] == last["low"] == last["close"] == 87.84


def test_zero_placeholders_are_not_prices():
    text = "DATE,OPEN,HIGH,LOW,CLOSE\n01/02/1990,0,0,0,17.24\n"
    row = VolatilityFetcher.parse_cboe_index(text, "^VIX", NOW).iloc[0]
    assert pd.isna(row["open"]) and pd.isna(row["high"]) and pd.isna(row["low"])
    assert row["close"] == 17.24


def test_an_unexpected_header_is_refused():
    with pytest.raises(ValueError):
        VolatilityFetcher.parse_cboe_index("<html>Access Denied</html>\n", "^VIX", NOW)


def test_the_whole_history_is_written_only_the_first_time(fetcher):
    """Rewriting four hundred partitions a night to restate history is all cost."""
    frame = pd.DataFrame(
        {"symbol": "^VIX", "date": [date(1990, 1, 2), date(2026, 9, 20), date(2026, 9, 25)]}
    )
    lookback = date(2026, 9, 15)

    backfilled: list[str] = []
    fresh = fetcher._window(frame, "^VIX", {}, lookback, backfilled)
    assert len(fresh) == 3 and backfilled == ["^VIX"]

    backfilled = []
    known = fetcher._window(frame, "^VIX", {"^VIX": date(1990, 1, 2)}, lookback, backfilled)
    assert list(known["date"]) == [date(2026, 9, 20), date(2026, 9, 25)]
    assert backfilled == []


# ---------------------------------------------------------------- futures

VX_OCT_2026 = """Trade Date,Futures,Open,High,Low,Close,Settle,Change,Total Volume,EFP,Open Interest
2026-01-26,V (Oct 2026),0.0000,21.0000,22.0000,0.0000,21.50,0,0,0,0
2026-09-23,V (Oct 2026),17.40,17.90,17.33,17.65,17.7372,0.3765,78315,0,223185
2026-09-24,V (Oct 2026),17.65,18.20,17.60,17.91,17.8081,0.0709,81394,0,219147
"""


def test_a_contract_file_parses():
    df = VolatilityFetcher.parse_vx(VX_OCT_2026, date(2026, 10, 21), NOW)
    last = df.set_index("trade_date").loc[date(2026, 9, 24)]
    assert last["settle"] == pytest.approx(17.8081)
    assert last["volume"] == 81394 and last["open_interest"] == 219147
    assert last["days_to_expiration"] == 27
    assert last["contract"] == "V (Oct 2026)"


def test_an_untraded_day_keeps_its_settle_but_no_prices():
    """CBOE's first row: no trades, zero open/close, and a low above the high."""
    df = VolatilityFetcher.parse_vx(VX_OCT_2026, date(2026, 10, 21), NOW)
    first = df.set_index("trade_date").loc[date(2026, 1, 26)]
    assert first["settle"] == pytest.approx(21.50), "the settlement still stands"
    assert all(pd.isna(first[c]) for c in ("open", "high", "low", "close"))


def test_a_file_without_settlements_yields_nothing():
    assert VolatilityFetcher.parse_vx("<html>404</html>\n", date(2026, 10, 21), NOW).empty


# ------------------------------------------------------------------ scope


class _Untouchable:
    """A tracker that fails the test if anything reaches for it."""

    def __getattr__(self, name):
        raise AssertionError(f"volatility touched the membership tracker: .{name}")


def test_the_stage_runs_from_its_own_config_not_membership(tmp_path, monkeypatch):
    """Indices must stay out of everything membership drives.

    ETFs registered there were asked for earnings and 8-Ks. An index has no
    quote on Tiingo or Finnhub and files nothing, so registering one would fail
    in every stage at once. The stage reads its own config and never the
    tracker -- and keeps once-a-day indices out of the minute feed.
    """
    paths = DatasetPaths(tmp_path)
    paths.ensure_layout()
    config = Config(
        raw={
            "volatility": {
                "enabled": True,
                "indices": {"VIX": "x", "SKEW": "x"},
                "yahoo_indices": {"MOVE": "x"},
                "daily_only": ["SKEW", "MOVE"],
                "futures": {"enabled": True},
            }
        },
        secrets=Secrets(),
        data_root=tmp_path,
        config_path=tmp_path / "config.yaml",
    )
    f = VolatilityFetcher(config, paths, ParquetWriter(), tracker=_Untouchable())
    seen: dict = {}
    monkeypatch.setattr(
        f,
        "_daily",
        lambda run, cboe, yahoo, client, result: seen.update(daily=sorted([*cboe, *yahoo])),
    )
    monkeypatch.setattr(f, "_minutes", lambda run, symbols, result: seen.update(minute=symbols))
    monkeypatch.setattr(f, "_futures", lambda run, client, result: seen.update(futures=True))

    from tickerlake.fetchers.base import FetchResult

    f.collect(date(2026, 9, 25), FetchResult(stage="volatility", dataset="ohlcv"))
    assert seen["daily"] == ["MOVE", "SKEW", "VIX"], "every index gets daily history"
    assert seen["minute"] == ["^VIX"], "once-a-day indices get no minute feed"
    assert seen["futures"] is True
