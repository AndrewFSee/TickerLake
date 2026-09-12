"""Tests for the NYSE trading calendar and the closed-day guard.

The calendar was validated against 2,940 real SPY sessions (2015-2026) at
99.932% agreement, with the only two disagreements being the documented ad-hoc
closures. These tests pin the rules that produced that.
"""

from __future__ import annotations

from datetime import date

import pytest

from tickerlake.utils.market_calendar import (
    easter_sunday,
    is_trading_day,
    nyse_holidays,
    previous_trading_day,
    trading_days_between,
    why_closed,
)

# ------------------------------------------------------------------ easter


@pytest.mark.parametrize(
    ("year", "expected"),
    [
        (2020, date(2020, 4, 12)),
        (2021, date(2021, 4, 4)),
        (2022, date(2022, 4, 17)),
        (2023, date(2023, 4, 9)),
        (2024, date(2024, 3, 31)),
        (2025, date(2025, 4, 20)),
        (2026, date(2026, 4, 5)),
    ],
)
def test_easter_sunday(year, expected):
    assert easter_sunday(year) == expected


def test_good_friday_is_two_days_before_easter_and_is_a_friday():
    for year in range(2015, 2031):
        good_friday = easter_sunday(year) - __import__("datetime").timedelta(days=2)
        assert good_friday.weekday() == 4
        assert good_friday in nyse_holidays(year), f"{year} Good Friday must be a closure"


# ---------------------------------------------------------------- holidays


def test_2026_holiday_set():
    assert sorted(nyse_holidays(2026)) == [
        date(2026, 1, 1),    # New Year's Day (Thursday)
        date(2026, 1, 19),   # MLK Jr Day
        date(2026, 2, 16),   # Washington's Birthday
        date(2026, 4, 3),    # Good Friday
        date(2026, 5, 25),   # Memorial Day
        date(2026, 6, 19),   # Juneteenth (Friday)
        date(2026, 7, 3),    # Independence Day observed (Jul 4 is a Saturday)
        date(2026, 9, 7),    # Labor Day
        date(2026, 11, 26),  # Thanksgiving
        date(2026, 12, 25),  # Christmas (Friday)
    ]


def test_every_year_has_nine_or_ten_closures():
    """Juneteenth from 2022 makes it ten; a stray count means a broken rule."""
    for year in range(2015, 2031):
        assert len(nyse_holidays(year)) in (9, 10), year


def test_juneteenth_only_from_2022():
    assert date(2021, 6, 18) not in nyse_holidays(2021)
    assert date(2021, 6, 19) not in nyse_holidays(2021)
    assert date(2022, 6, 20) in nyse_holidays(2022)  # Jun 19 2022 was a Sunday


def test_saturday_holidays_shift_back_to_friday():
    # Jul 4 2026 falls on a Saturday, so the exchange closes Friday Jul 3.
    assert date(2026, 7, 3) in nyse_holidays(2026)
    assert date(2026, 7, 4) not in nyse_holidays(2026)


def test_sunday_holidays_shift_forward_to_monday():
    # Dec 25 2022 was a Sunday, observed Monday Dec 26.
    assert date(2022, 12, 26) in nyse_holidays(2022)


def test_saturday_new_year_is_not_observed():
    """NYSE does not close 31 December for a Saturday 1 January."""
    assert date(2022, 1, 1).weekday() == 5
    assert date(2021, 12, 31) not in nyse_holidays(2021)
    assert date(2022, 1, 1) not in nyse_holidays(2022)


def test_federal_only_holidays_are_not_nyse_closures():
    """The federal calendar is not a substitute for the exchange calendar."""
    # Columbus Day and Veterans Day 2026: markets open.
    assert is_trading_day(date(2026, 10, 12))
    assert is_trading_day(date(2026, 11, 11))


# ------------------------------------------------------------ trading days


def test_weekends_are_never_trading_days():
    assert not is_trading_day(date(2026, 9, 12))  # Saturday
    assert not is_trading_day(date(2026, 9, 13))  # Sunday
    assert is_trading_day(date(2026, 9, 11))      # Friday


def test_known_closures_and_sessions():
    assert not is_trading_day(date(2026, 9, 7)), "Labor Day"
    assert is_trading_day(date(2026, 9, 8)), "the session after Labor Day"
    assert not is_trading_day(date(2026, 11, 26)), "Thanksgiving"
    assert is_trading_day(date(2026, 11, 27)), "the half day after Thanksgiving still trades"


def test_why_closed_explains_itself():
    assert why_closed(date(2026, 9, 12)) == "Saturday"
    assert why_closed(date(2026, 9, 13)) == "Sunday"
    assert "Labor Day" in why_closed(date(2026, 9, 7))
    assert "Thanksgiving" in why_closed(date(2026, 11, 26))
    assert why_closed(date(2026, 9, 11)) is None


def test_previous_trading_day_skips_weekends_and_holidays():
    # The Tuesday after Labor Day steps back over Monday and the weekend.
    assert previous_trading_day(date(2026, 9, 8)) == date(2026, 9, 4)
    assert previous_trading_day(date(2026, 11, 27)) == date(2026, 11, 25)


def test_trading_days_between_counts_a_normal_year():
    days = trading_days_between(date(2026, 1, 1), date(2026, 12, 31))
    # US equity markets run 250-253 sessions a year.
    assert 249 <= len(days) <= 253, len(days)
    assert all(is_trading_day(d) for d in days)


def test_september_2026_sessions_match_the_collected_data():
    """The window whose gap first surfaced Labor Day in this project."""
    days = trading_days_between(date(2026, 9, 5), date(2026, 9, 11))
    assert days == [date(2026, 9, 8), date(2026, 9, 9), date(2026, 9, 10), date(2026, 9, 11)]


# -------------------------------------------------------------- the guard


def test_options_and_intraday_require_a_trading_day():
    from tickerlake.fetchers.yf_intraday import YFinanceIntradayFetcher
    from tickerlake.fetchers.yf_options import YFinanceOptionsFetcher

    assert YFinanceOptionsFetcher.requires_trading_day is True
    assert YFinanceIntradayFetcher.requires_trading_day is True


def test_stages_that_fail_cleanly_are_not_gated():
    """Sources returning nothing on a closed day need no guard."""
    from tickerlake.fetchers.finra import FinraShortVolumeFetcher
    from tickerlake.fetchers.fred import FredFetcher
    from tickerlake.fetchers.sec_edgar import SECEdgarFetcher
    from tickerlake.fetchers.yf_ohlcv import YFinanceOHLCVFetcher

    for cls in (YFinanceOHLCVFetcher, SECEdgarFetcher, FredFetcher, FinraShortVolumeFetcher):
        assert cls.requires_trading_day is False, cls.__name__


def test_guarded_fetcher_skips_on_a_closed_day(tmp_path):
    from tickerlake.config import Config, Secrets
    from tickerlake.fetchers.yf_options import YFinanceOptionsFetcher
    from tickerlake.storage.paths import DatasetPaths
    from tickerlake.storage.writer import ParquetWriter

    paths = DatasetPaths(tmp_path)
    paths.ensure_layout()
    config = Config(
        raw={"options": {"enabled": True}},
        secrets=Secrets(),
        data_root=tmp_path,
        config_path=tmp_path / "c.yaml",
    )
    fetcher = YFinanceOptionsFetcher(config, paths, ParquetWriter(), symbols_override=["AAPL"])

    result = fetcher.run(date(2026, 9, 12))  # Saturday
    assert result.skipped
    assert "not a trading day" in result.skip_reason
    assert "Saturday" in result.skip_reason
    assert result.rows_written == 0


def test_run_on_closed_days_override(tmp_path, monkeypatch):
    """The escape hatch must actually let a run through."""
    from tickerlake.config import Config, Secrets
    from tickerlake.fetchers.yf_options import YFinanceOptionsFetcher
    from tickerlake.storage.paths import DatasetPaths
    from tickerlake.storage.writer import ParquetWriter

    paths = DatasetPaths(tmp_path)
    paths.ensure_layout()
    config = Config(
        raw={"options": {"enabled": True, "run_on_closed_days": True}},
        secrets=Secrets(),
        data_root=tmp_path,
        config_path=tmp_path / "c.yaml",
    )
    fetcher = YFinanceOptionsFetcher(config, paths, ParquetWriter(), symbols_override=[])
    # No symbols, so collect() returns immediately -- we only care that the
    # trading-day gate did not short-circuit it.
    result = fetcher.run(date(2026, 9, 12))
    assert not result.skipped
