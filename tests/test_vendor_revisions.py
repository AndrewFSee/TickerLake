"""Tests for checking stored bars against Yahoo's current view of them.

``close`` is split-adjusted only as of the day a row was written. Corteva spun
off Vylor on 2026-10-01 and Yahoo recorded it, days later, as a 6.665 split --
rescaling its own history and none of ours, so the stored series fell 84% in a
session where nothing of the sort happened. The same refresh that lets Yahoo
correct a recent bar also let it overwrite Warner Bros. Discovery's last real
session with a broken copy, and Yahoo pads ended listings with flat bars that
never traded.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from tickerlake.config import Config, Secrets
from tickerlake.fetchers.base import FetchResult
from tickerlake.fetchers.yf_ohlcv import (
    YFinanceOHLCVFetcher,
    _drop_trailing_filler,
    _rescaled_symbols,
)
from tickerlake.storage import paths as P
from tickerlake.storage.paths import DatasetPaths
from tickerlake.storage.query import LakeQuery
from tickerlake.storage.writer import ParquetWriter

RUN_DATE = date(2026, 10, 16)
SPLIT = date(2026, 10, 1)
RATIO = 6.665


def _sessions(start: date, end: date) -> list[date]:
    days, d = [], start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _bars(symbol: str, closes: dict[date, float], volume: float = 1_000_000) -> pd.DataFrame:
    now = datetime.now(UTC)
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "date": d,
                "open": c,
                "high": c * 1.01,
                "low": c * 0.99,
                "close": c,
                "adj_close": c,
                "volume": volume,
                "dividends": 0.0,
                "stock_splits": RATIO if (symbol == "CTVA" and d == SPLIT) else 0.0,
                "repaired": False,
                "source": "test",
                "ingested_at": now,
            }
            for d, c in closes.items()
        ]
    )


def _ctva(scaled: bool, start: date = date(2026, 9, 1)) -> pd.DataFrame:
    """Corteva around the spin-off: 77-ish before, 12-ish after.

    ``scaled`` is Yahoo's view today (pre-split bars divided by the ratio);
    unscaled is what the lake stored before Yahoo got round to it.
    """
    closes = {}
    for d in _sessions(start, RUN_DATE):
        pre = 77.65 + 0.01 * d.day
        closes[d] = (pre / RATIO if scaled else pre) if d < SPLIT else 12.5 + 0.01 * d.day
    return _bars("CTVA", closes)


@pytest.fixture
def lake(tmp_path):
    paths = DatasetPaths(tmp_path)
    paths.ensure_layout()
    w = ParquetWriter()
    stored = pd.concat(
        [
            _bars("CTVA", {date(2015, 3, 2): 30.0}),
            _ctva(scaled=False),
            _bars("KEEP", {d: 50.0 for d in _sessions(date(2026, 9, 1), RUN_DATE)}),
            # An old bar too, or a history this short would look like a newcomer.
            _bars("WBD", {date(2022, 4, 11): 24.0}),
            _bars("WBD", {date(2026, 10, 2): 30.94, date(2026, 10, 5): 30.95}, 97_886_735),
        ]
    )
    for (y, m), group in stored.groupby(
        [pd.to_datetime(stored["date"]).dt.year, pd.to_datetime(stored["date"]).dt.month]
    ):
        w.write(group, P.OHLCV, paths.ohlcv_backfill_file(int(y), int(m)), mode="merge")

    config = Config(
        raw={"ohlcv": {"enabled": True, "start_date": "2010-01-01"}},
        secrets=Secrets(),
        data_root=tmp_path,
        config_path=tmp_path / "config.yaml",
    )
    fetcher = YFinanceOHLCVFetcher(config, paths, w)
    fetcher.symbols_override = ["CTVA", "KEEP", "WBD"]
    return fetcher, tmp_path


def _yahoo_today(symbols, start, end):
    """What Yahoo serves now: CTVA rescaled, WBD's last bar degraded."""
    parts = []
    for s in symbols:
        if s == "CTVA":
            hist = _ctva(scaled=True, start=max(start, date(2026, 9, 1)))
            if start.year < 2020:
                hist = pd.concat([_bars("CTVA", {date(2015, 3, 2): 30.0 / RATIO}), hist])
            parts.append(hist)
        elif s == "KEEP":
            parts.append(_bars("KEEP", {d: 50.0 for d in _sessions(start, RUN_DATE)}))
        elif s == "WBD":
            parts.append(_bars("WBD", {date(2026, 10, 5): 30.95}, volume=40))
    return pd.concat(parts, ignore_index=True)


def _stored(root, symbol):
    with LakeQuery(root) as q:
        df = q.sql("SELECT * FROM ohlcv WHERE symbol = ? ORDER BY date", [symbol])
    return df.assign(date=pd.to_datetime(df["date"]).dt.date).set_index("date")


# ------------------------------------------------------------------ detection


def _overlap(symbol, yahoo, stored):
    return pd.DataFrame({"symbol": symbol, "close": yahoo, "close_stored": stored})


def test_a_constant_ratio_across_the_older_bars_is_a_rescale():
    both = _overlap("CTVA", [11.65, 11.70, 11.60, 12.57], [77.65, 77.98, 77.31, 12.57])
    assert _rescaled_symbols(both, tolerance=0.02, min_bars=3) == ["CTVA"]


def test_scattered_disagreement_is_not_a_rescale():
    """PSKY: yfinance's repair pass moves an aliased ticker by 0.7%-7.6%."""
    both = _overlap("PSKY", [10.98, 10.69, 10.05, 10.41], [10.21, 10.62, 9.96, 10.33])
    assert _rescaled_symbols(both, tolerance=0.02, min_bars=3) == []


def test_too_few_disagreeing_bars_are_not_a_rescale():
    both = _overlap("X", [10.0, 10.0, 50.0], [60.0, 60.0, 50.0])
    assert _rescaled_symbols(both, tolerance=0.02, min_bars=3) == []


def test_sub_penny_ticks_are_not_a_rescale():
    """CPWR trades at fractions of a cent, where one tick is a 50% move."""
    both = _overlap("CPWR", [0.0051] * 4, [0.0077] * 4)
    assert _rescaled_symbols(both, tolerance=0.02, min_bars=3) == []


def test_trailing_filler_is_dropped():
    """Electronic Arts: four flat, volume-less days after its last session."""
    bars = _bars("EA", {date(2026, 8, 3): 209.98, date(2026, 8, 4): 209.70})
    filler = _bars("EA", {d: 209.70 for d in _sessions(date(2026, 8, 5), date(2026, 8, 10))})
    filler[["open", "high", "low"]] = 209.70
    filler["volume"] = 0
    out = _drop_trailing_filler(pd.concat([bars, filler], ignore_index=True))
    assert list(out["date"]) == [date(2026, 8, 3), date(2026, 8, 4)]


def test_a_flat_bar_mid_series_is_kept():
    bars = _bars("X", {date(2026, 8, 3): 10.0, date(2026, 8, 4): 10.0, date(2026, 8, 5): 10.5})
    bars.loc[1, ["open", "high", "low", "close", "volume"]] = [10.0, 10.0, 10.0, 10.0, 0]
    assert len(_drop_trailing_filler(bars)) == 3


def test_a_live_series_is_untouched():
    bars = _bars("X", {d: 10.0 for d in _sessions(date(2026, 8, 3), date(2026, 8, 7))})
    assert len(_drop_trailing_filler(bars)) == 5


# --------------------------------------------------------------------- wiring


def test_a_rescaled_symbol_gets_its_whole_history_again(lake, monkeypatch):
    fetcher, root = lake
    calls: list[tuple[tuple[str, ...], date]] = []

    def fake_download(symbols, start, end):
        calls.append((tuple(symbols), start))
        return _yahoo_today(symbols, start, end)

    monkeypatch.setattr(fetcher, "_download", fake_download)
    result = fetcher.run(RUN_DATE)
    assert result.ok, result.errors

    assert (("CTVA",), date(2010, 1, 1)) in calls, "full history requested"
    assert result.details["rescaled_history"] == ["CTVA"]

    ctva = _stored(root, "CTVA")
    before, after = ctva.loc[date(2026, 9, 30), "close"], ctva.loc[SPLIT, "close"]
    assert after / before > 0.5, "no 84% cliff at the spin-off"
    assert ctva.loc[date(2015, 3, 2), "close"] == pytest.approx(30.0 / RATIO), "old years too"
    assert result.items_succeeded == 3, "the refetch is not counted twice"


def test_an_unrescaled_symbol_writes_only_the_lookback(lake, monkeypatch):
    fetcher, root = lake
    monkeypatch.setattr(fetcher, "_download", _yahoo_today)
    before = _stored(root, "KEEP")["ingested_at"]
    fetcher.run(RUN_DATE)
    after = _stored(root, "KEEP")["ingested_at"]

    changed = sorted(d for d in after.index if after[d] != before[d])
    assert changed and min(changed) >= RUN_DATE - timedelta(days=5)


def test_a_degraded_refresh_keeps_the_stored_bar(lake, monkeypatch):
    """WBD's final session came back with a volume of 40 against ~98M."""
    fetcher, root = lake
    monkeypatch.setattr(fetcher, "_download", _yahoo_today)
    result = fetcher.run(RUN_DATE)

    assert _stored(root, "WBD").loc[date(2026, 10, 5), "volume"] == 97_886_735
    assert result.details["kept_stored_bars"] == ["WBD 2026-10-05"]


def test_a_mass_rescale_is_refused(lake, monkeypatch):
    fetcher, root = lake
    fetcher.config.raw["ohlcv"]["max_rescaled"] = 0
    monkeypatch.setattr(fetcher, "_download", _yahoo_today)
    result = fetcher.run(RUN_DATE)

    assert "rescaled_history" not in result.details
    assert any("not refetching" in w for w in result.warnings)
    assert _stored(root, "CTVA").loc[date(2015, 3, 2), "close"] == 30.0, "history untouched"


# ------------------------------------------------------- hundredfold dividends


@pytest.fixture
def emcor(tmp_path):
    """EMCOR with a year of $0.40 quarterlies stored, and Yahoo's 10/19 answer."""
    paths = DatasetPaths(tmp_path)
    paths.ensure_layout()
    w = ParquetWriter()
    history = _bars("EME", {d: 700.0 for d in _sessions(date(2025, 10, 1), date(2026, 10, 9))})
    for d in (date(2026, 1, 14), date(2026, 4, 16), date(2026, 7, 15), date(2025, 10, 15)):
        history.loc[history["date"] == d, "dividends"] = 0.40
    for (y, m), group in history.groupby(
        [pd.to_datetime(history["date"]).dt.year, pd.to_datetime(history["date"]).dt.month]
    ):
        w.write(group, P.OHLCV, paths.ohlcv_backfill_file(int(y), int(m)), mode="merge")
    config = Config(
        raw={"ohlcv": {"enabled": True, "start_date": "2010-01-01"}},
        secrets=Secrets(),
        data_root=tmp_path,
        config_path=tmp_path / "config.yaml",
    )
    return YFinanceOHLCVFetcher(config, paths, w)


def _emcor_window(dividend: float) -> pd.DataFrame:
    closes = {
        date(2026, 10, 13): 776.05,
        date(2026, 10, 14): 745.35,
        date(2026, 10, 15): 732.53,
        date(2026, 10, 16): 721.66,
    }
    bars = _bars("EME", closes)
    bars.loc[bars["date"] == date(2026, 10, 15), "dividends"] = dividend
    # Yahoo's Adj Close is built on whatever dividend it served.
    factor = 1 - dividend / 745.35
    early = bars["date"] < date(2026, 10, 15)
    bars.loc[early, "adj_close"] = bars.loc[early, "close"] * factor
    return bars


def test_a_hundredfold_dividend_is_divided_back_down(emcor):
    result = FetchResult(stage="ohlcv")
    out = emcor._check_dividends(_emcor_window(40.0), result)

    ex = out.set_index("date")
    assert ex.loc[date(2026, 10, 15), "dividends"] == pytest.approx(0.40)
    assert ex.loc[date(2026, 10, 14), "adj_close"] == pytest.approx(745.35 - 0.40, abs=0.01)
    assert ex.loc[date(2026, 10, 15), "adj_close"] == 732.53, "the ex-date itself is untouched"
    assert result.details["dividends_corrected"] == ["EME 2026-10-15: 40 -> 0.4"]


def test_an_ordinary_dividend_is_left_alone(emcor):
    window = _emcor_window(0.40)
    out = emcor._check_dividends(window, FetchResult(stage="ohlcv"))
    pd.testing.assert_frame_equal(out, window)


def test_a_special_dividend_that_is_not_exactly_100x_is_believed(emcor):
    """Costco-style specials are real; only the exact cents-for-dollars slip is not."""
    out = emcor._check_dividends(_emcor_window(15.0), FetchResult(stage="ohlcv"))
    assert out.set_index("date").loc[date(2026, 10, 15), "dividends"] == 15.0


def test_a_special_near_100x_is_believed(emcor):
    """Baker Hughes' real $17.50 special in 2017 was 103x its prior $0.17."""
    out = emcor._check_dividends(_emcor_window(0.40 * 103), FetchResult(stage="ohlcv"))
    assert out.set_index("date").loc[date(2026, 10, 15), "dividends"] == pytest.approx(41.2)


def test_a_first_ever_dividend_has_nothing_to_compare_with(lake):
    fetcher, _ = lake
    window = _bars("KEEP", {date(2026, 10, 14): 50.0, date(2026, 10, 15): 50.0})
    window.loc[1, "dividends"] = 40.0
    out = fetcher._check_dividends(window, FetchResult(stage="ohlcv"))
    assert out.loc[1, "dividends"] == 40.0
