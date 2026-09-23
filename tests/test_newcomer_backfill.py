"""Tests for backfilling symbols that join the index mid-quarter.

A name added at a reconstitution arrives with only the bars collected since it
joined. Bloom Energy and P entered the S&P 500 on 2026-09-21 and had five bars
each, reaching back to the day the universe first saw them. Nothing failed --
that is what makes it worth a test. The symbols were simply short, so anything
needing a lookback had nothing to compute from, and the gap was visible only
because somebody went looking.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from tickerlake.config import Config, Secrets
from tickerlake.fetchers.yf_ohlcv import NEWCOMER_LABEL, YFinanceOHLCVFetcher
from tickerlake.storage import paths as P
from tickerlake.storage.paths import DatasetPaths
from tickerlake.storage.writer import ParquetWriter

RUN_DATE = date(2026, 9, 22)


def _bars(symbol: str, days: list[date]) -> pd.DataFrame:
    now = datetime.now(UTC)
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "date": d,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "adj_close": 100.0,
                "volume": 1_000_000,
                "dividends": 0.0,
                "stock_splits": 0.0,
                "repaired": False,
                "source": "test",
                "ingested_at": now,
            }
            for d in days
        ]
    )


@pytest.fixture
def fetcher(tmp_path):
    """A lake holding one long-standing symbol and one fresh arrival."""
    paths = DatasetPaths(tmp_path)
    paths.ensure_layout()
    w = ParquetWriter()

    # OLD has years of history.
    w.write(
        _bars("OLD", [date(2015, 6, 1), date(2015, 6, 2)]),
        P.OHLCV,
        paths.ohlcv_backfill_file(2015, 6),
        mode="overwrite",
    )
    w.write(
        _bars("OLD", [date(2026, 9, 21), date(2026, 9, 22)]),
        P.OHLCV,
        paths.ohlcv_backfill_file(2026, 9),
        mode="merge",
    )
    # NEW joined last week and has nothing before that.
    w.write(
        _bars("NEW", [date(2026, 9, 16), date(2026, 9, 21), date(2026, 9, 22)]),
        P.OHLCV,
        paths.ohlcv_backfill_file(2026, 9),
        mode="merge",
    )

    config = Config(
        raw={"ohlcv": {"enabled": True, "start_date": "2010-01-01"}},
        secrets=Secrets(),
        data_root=tmp_path,
        config_path=tmp_path / "config.yaml",
    )
    return YFinanceOHLCVFetcher(config, paths, w)


# ----------------------------------------------------------------- detection


def test_a_symbol_with_only_recent_bars_is_flagged(fetcher):
    assert fetcher._needs_history(["OLD", "NEW"], RUN_DATE) == ["NEW"]


def test_a_symbol_absent_from_the_lake_is_flagged(fetcher):
    """Never seen at all is the strongest case for a backfill."""
    assert "UNSEEN" in fetcher._needs_history(["OLD", "UNSEEN"], RUN_DATE)


def test_a_long_standing_symbol_is_not_flagged(fetcher):
    assert fetcher._needs_history(["OLD"], RUN_DATE) == []


def test_detection_stops_firing_once_the_history_lands(fetcher, tmp_path):
    """The point of testing recency rather than row count.

    After the backfill the earliest bar moves back years, so the symbol drops
    out on its own with no state to track and nothing to reset.
    """
    assert fetcher._needs_history(["NEW"], RUN_DATE) == ["NEW"]

    ParquetWriter().write(
        _bars("NEW", [date(2018, 7, 25)]),
        P.OHLCV,
        DatasetPaths(tmp_path).ohlcv_backfill_file(2018, 7),
        mode="overwrite",
    )
    assert fetcher._needs_history(["NEW"], RUN_DATE) == [], "one pass is enough"


def test_the_window_is_configurable(tmp_path, fetcher):
    fetcher.config.raw["ohlcv"]["recent_history_days"] = 1
    assert fetcher._needs_history(["NEW"], RUN_DATE) == [], "NEW is older than one day"


def test_an_empty_symbol_list_is_survivable(fetcher):
    assert fetcher._needs_history([], RUN_DATE) == []


def test_an_empty_lake_flags_everything_but_the_ceiling_stops_it(tmp_path):
    """A fresh install must not turn the nightly run into a full backfill.

    Detection alone says every symbol is new, which is true and useless. The
    ceiling is what distinguishes a reconstitution from an empty data directory.
    """
    from tickerlake.fetchers.base import FetchResult

    paths = DatasetPaths(tmp_path)
    paths.ensure_layout()
    config = Config(
        raw={"ohlcv": {"enabled": True, "max_newcomers": 3}},
        secrets=Secrets(),
        data_root=tmp_path,
        config_path=tmp_path / "config.yaml",
    )
    f = YFinanceOHLCVFetcher(config, paths, ParquetWriter())
    many = [f"SYM{i}" for i in range(10)]

    assert f._needs_history(many, RUN_DATE) == many, "all genuinely unseen"

    result = FetchResult(stage="ohlcv", dataset=P.OHLCV)
    assert f._newcomers(many, RUN_DATE, result) == [], "but the ceiling holds"
    assert any("backfill ohlcv" in w for w in result.warnings), "and says what to do"


def test_a_reconstitution_sized_group_passes_the_ceiling(fetcher):
    from tickerlake.fetchers.base import FetchResult

    result = FetchResult(stage="ohlcv", dataset=P.OHLCV)
    assert fetcher._newcomers(["OLD", "NEW"], RUN_DATE, result) == ["NEW"]
    assert not result.warnings


def test_a_delisted_symbol_is_never_treated_as_a_newcomer(fetcher):
    """AGL Resources (GAS) left the index in 2016 and has zero stored bars.

    It qualifies on recency -- there is nothing there at all -- but the vendor
    will not serve a delisted name's history, so requesting it would fail every
    night forever. Only current members are eligible.
    """

    class Tracker:
        def current_members(self):
            return ["OLD", "NEW"]

    from tickerlake.fetchers.base import FetchResult

    fetcher.tracker = Tracker()
    result = FetchResult(stage="ohlcv", dataset=P.OHLCV)

    assert "GAS" in fetcher._needs_history(["OLD", "NEW", "GAS"], RUN_DATE)
    assert fetcher._newcomers(["OLD", "NEW", "GAS"], RUN_DATE, result) == ["NEW"]
    assert not result.warnings, "a delisted name is expected, not worth warning about"


# -------------------------------------------------------------------- wiring


def test_newcomers_are_fetched_over_the_full_range(fetcher, monkeypatch):
    """The whole point: NEW gets 2010, OLD gets the five-day window."""
    calls: list[tuple[tuple[str, ...], date]] = []

    def fake_download(symbols, start, end):
        calls.append((tuple(symbols), start))
        return _bars(symbols[0], [RUN_DATE])

    monkeypatch.setattr(fetcher, "_download", fake_download)
    monkeypatch.setattr(fetcher, "_record_observations", lambda *a, **k: None)
    fetcher.symbols_override = ["OLD", "NEW"]

    result = fetcher.run(RUN_DATE)
    assert result.ok, result.errors

    ranges = {syms: start for syms, start in calls}
    assert ranges[("OLD",)] == RUN_DATE - pd.Timedelta(days=5).to_pytimedelta()
    assert ranges[("NEW",)] == date(2010, 1, 1), "full history, not the lookback"
    assert result.details["backfilled_newcomers"] == ["NEW"]


def test_no_newcomers_means_a_single_pass(fetcher, monkeypatch):
    """The ordinary day must not grow an extra request."""
    calls: list[tuple[str, ...]] = []

    def fake_download(symbols, start, end):
        calls.append(tuple(symbols))
        return _bars(symbols[0], [RUN_DATE])

    monkeypatch.setattr(fetcher, "_download", fake_download)
    monkeypatch.setattr(fetcher, "_record_observations", lambda *a, **k: None)
    fetcher.symbols_override = ["OLD"]

    result = fetcher.run(RUN_DATE)
    assert result.ok, result.errors
    assert calls == [("OLD",)]
    assert "backfilled_newcomers" not in result.details


def test_an_explicit_backfill_is_unaffected(fetcher, monkeypatch):
    """`backfill ohlcv` already asks for everything; it must not split in two."""
    calls: list[tuple[tuple[str, ...], date]] = []

    def fake_download(symbols, start, end):
        calls.append((tuple(symbols), start))
        return _bars(symbols[0], [RUN_DATE])

    monkeypatch.setattr(fetcher, "_download", fake_download)
    monkeypatch.setattr(fetcher, "_record_observations", lambda *a, **k: None)
    fetcher.backfill = True

    result = fetcher.run(RUN_DATE)
    assert result.ok, result.errors
    assert all(start == date(2010, 1, 1) for _, start in calls)
    assert "backfilled_newcomers" not in result.details


def test_the_newcomer_label_is_what_the_code_matches_on(fetcher):
    """Two places branch on it, so a typo would silently disable the feature."""
    assert NEWCOMER_LABEL == "backfilling newcomers"
