"""Tests for dividend adjustment.

``adj_close`` is a running total over the *future*: the value for a 2015 session
depends on every dividend paid since. The fetcher writes a row when its session
is current -- when the adjustment is 1.0 by definition -- and the incremental
lookback only revisits the last few sessions, so the row then freezes and every
later dividend fails to reach it. Crown Castle's year-old rows were understated
by 1.45%, one missed dividend, across the symbol's whole history.

Each case below is drawn from a real symbol that broke an earlier attempt.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from tickerlake.config import Config, Secrets
from tickerlake.pipeline.adjustments import AdjustmentRepair
from tickerlake.storage import paths as P
from tickerlake.storage.paths import DatasetPaths
from tickerlake.storage.query import LakeQuery
from tickerlake.storage.writer import ParquetWriter


def _bars(symbol, rows, ingested):
    """rows: (date, close, adj_close, dividend, split)."""
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "date": d,
                "open": c,
                "high": c,
                "low": c,
                "close": c,
                "adj_close": adj,
                "volume": 1_000_000,
                "dividends": div,
                "stock_splits": split,
                "repaired": False,
                "source": "test",
                "ingested_at": ingested,
            }
            for d, c, adj, div, split in rows
        ]
    )


@pytest.fixture
def lake(tmp_path):
    """One payer whose stored column has gone stale, one that has not.

    PAYER was written before its 2026-02-02 dividend, so every row predating
    that dividend is missing the adjustment. FRESH was written after its own,
    so it is already correct and must be left alone.
    """
    paths = DatasetPaths(tmp_path)
    paths.ensure_layout()
    w = ParquetWriter()
    written_on = datetime(2026, 1, 15, tzinfo=UTC)

    # close 100 throughout; a $2 dividend on 2026-02-02 means every earlier row
    # should carry a factor of 1 - 2/100 = 0.98. January was backfilled on the
    # 15th and has been frozen since; February arrived with the daily run, so it
    # is already current. That split is the whole bug.
    january = [
        (date(2026, 1, 5), 100.0, 100.0, 0.0, 0.0),
        (date(2026, 1, 6), 100.0, 100.0, 0.0, 0.0),
    ]
    w.write(
        _bars("PAYER", january, written_on),
        P.OHLCV,
        paths.ohlcv_backfill_file(2026, 1),
        mode="overwrite",
    )
    february = [
        (date(2026, 2, 2), 100.0, 100.0, 2.0, 0.0),
        (date(2026, 2, 3), 100.0, 100.0, 0.0, 0.0),
    ]
    w.write(
        _bars("PAYER", february, datetime(2026, 2, 3, tzinfo=UTC)),
        P.OHLCV,
        paths.ohlcv_backfill_file(2026, 2),
        mode="merge",
    )

    fresh = [
        (date(2026, 1, 5), 50.0, 49.0, 0.0, 0.0),
        (date(2026, 1, 6), 50.0, 50.0, 1.0, 0.0),  # ex-date itself is unadjusted
        (date(2026, 1, 7), 50.0, 50.0, 0.0, 0.0),
    ]
    w.write(
        _bars("FRESH", fresh, datetime(2026, 3, 1, tzinfo=UTC)),
        P.OHLCV,
        paths.ohlcv_backfill_file(2026, 3),
        mode="overwrite",
    )
    return tmp_path


def _config(root):
    return Config(raw={}, secrets=Secrets(), data_root=root, config_path=root / "c.yaml")


# ------------------------------------------------------------ computed series


def test_a_dividend_scales_every_earlier_session(lake):
    with LakeQuery(lake) as q:
        df = q.adjusted_ohlcv(symbols=["PAYER"]).set_index("date")

    assert df.loc[pd.Timestamp("2026-01-05"), "adj_factor"] == pytest.approx(0.98)
    assert df.loc[pd.Timestamp("2026-01-05"), "adj_close"] == pytest.approx(98.0)
    # The ex-date itself and anything after already reflect the drop.
    assert df.loc[pd.Timestamp("2026-02-02"), "adj_factor"] == pytest.approx(1.0)
    assert df.loc[pd.Timestamp("2026-02-03"), "adj_factor"] == pytest.approx(1.0)


def test_splits_are_not_applied_because_close_already_carries_them(tmp_path):
    """The 3M lesson.

    yfinance is called with ``auto_adjust=False``, which back-adjusts Close for
    splits and puts only dividends into Adj Close. 3M's 2024 Solventum spin-off
    is recorded as ``stock_splits = 1.196``; applying it on top moved the series
    by 17%.
    """
    paths = DatasetPaths(tmp_path)
    paths.ensure_layout()
    rows = [
        (date(2026, 1, 5), 100.0, 100.0, 0.0, 0.0),
        (date(2026, 1, 6), 50.0, 50.0, 0.0, 2.0),  # 2:1, close already halved
        (date(2026, 1, 7), 50.0, 50.0, 0.0, 0.0),
    ]
    ParquetWriter().write(
        _bars("SPLIT", rows, datetime(2026, 2, 1, tzinfo=UTC)),
        P.OHLCV,
        paths.ohlcv_backfill_file(2026, 1),
        mode="overwrite",
    )
    with LakeQuery(tmp_path) as q:
        df = q.adjusted_ohlcv(symbols=["SPLIT"])
    assert df["adj_factor"].sub(1.0).abs().max() < 1e-9, "a split must not move the factor"


def test_as_of_applies_only_dividends_known_by_then(lake):
    """The conventional adjusted close is future-dependent, so it needs gating."""
    with LakeQuery(lake) as q:
        blind = q.adjusted_ohlcv(symbols=["PAYER"], end=date(2026, 1, 6), as_of=date(2026, 1, 20))
        after = q.adjusted_ohlcv(symbols=["PAYER"], end=date(2026, 1, 6), as_of=date(2026, 2, 10))

    assert blind.iloc[0]["adj_factor"] == pytest.approx(1.0), "February was not knowable"
    assert blind.iloc[0]["adj_close"] == pytest.approx(100.0)
    assert after.iloc[0]["adj_factor"] == pytest.approx(0.98), "now it is"


def test_the_scan_reaches_past_the_requested_end(lake):
    """A factor depends on dividends after the window, so the scan must too."""
    with LakeQuery(lake) as q:
        df = q.adjusted_ohlcv(symbols=["PAYER"], start=date(2026, 1, 5), end=date(2026, 1, 6))
    assert len(df) == 2, "output is clipped to the window"
    assert df.iloc[0]["adj_factor"] == pytest.approx(0.98), "but February still counts"


# ------------------------------------------------------------ corrected series


def test_a_stale_row_is_pulled_down_to_match_a_later_one(lake):
    """January froze before the February dividend and never caught up."""
    with LakeQuery(lake) as q:
        df = q.corrected_adjustments(symbols=["PAYER"]).set_index("date")

    row = df.loc[pd.Timestamp("2026-01-05")]
    assert row["stored"] == pytest.approx(100.0), "what the vendor left behind"
    assert row["ratio"] == pytest.approx(0.98), "the dividend it never saw"
    assert row["adj_close"] == pytest.approx(98.0)
    # February was written after the dividend and is already right.
    assert df.loc[pd.Timestamp("2026-02-02"), "adj_close"] == pytest.approx(100.0)


def test_an_adjustment_our_actions_cannot_explain_is_preserved(tmp_path):
    """The Danaher guard.

    Danaher's 2016 Fortive spin-off is booked as both a $24.56 dividend and a
    1.319 split, and Yahoo's factor across it matches neither reading of that
    pair. Rebuilding from our own actions moved its pre-2016 history by 13.4%.
    Here the vendor has adjusted a session by 40% with no recorded dividend at
    all, and that adjustment has to survive.
    """
    paths = DatasetPaths(tmp_path)
    paths.ensure_layout()
    rows = [
        (date(2026, 1, 5), 100.0, 60.0, 0.0, 0.0),  # vendor knows something we do not
        (date(2026, 1, 6), 100.0, 100.0, 0.0, 0.0),
    ]
    ParquetWriter().write(
        _bars("SPIN", rows, datetime(2026, 1, 6, tzinfo=UTC)),
        P.OHLCV,
        paths.ohlcv_backfill_file(2026, 1),
        mode="overwrite",
    )
    with LakeQuery(tmp_path) as q:
        df = q.corrected_adjustments(symbols=["SPIN"]).set_index("date")

    assert df.loc[pd.Timestamp("2026-01-05"), "adj_close"] == pytest.approx(60.0)


def test_a_row_already_current_is_left_alone(lake):
    with LakeQuery(lake) as q:
        df = q.corrected_adjustments(symbols=["FRESH"]).set_index("date")

    assert df.loc[pd.Timestamp("2026-01-05"), "adj_close"] == pytest.approx(49.0)
    assert df.loc[pd.Timestamp("2026-01-06"), "adj_close"] == pytest.approx(50.0)


def test_drift_reports_the_stale_symbol_only(lake):
    with LakeQuery(lake) as q:
        d = q.adjustment_drift()
    assert set(d["symbol"]) == {"PAYER"}
    assert d.iloc[0]["max_drift_pct"] == pytest.approx(2.0, abs=0.01)


# --------------------------------------------------------------------- repair


def test_repair_rewrites_the_stored_column(lake):
    repair = AdjustmentRepair(_config(lake))
    result = repair.run()

    assert not result.errors
    assert result.rows_corrected == 2, "the two sessions predating the dividend"
    assert result.symbols_corrected == 1

    with LakeQuery(lake) as q:
        df = q.ohlcv(symbols=["PAYER"]).set_index("date")
    assert df.loc[pd.Timestamp("2026-01-05"), "adj_close"] == pytest.approx(98.0)
    assert df.loc[pd.Timestamp("2026-02-03"), "adj_close"] == pytest.approx(100.0)


def test_repair_is_idempotent(lake):
    config = _config(lake)
    AdjustmentRepair(config).run()
    second = AdjustmentRepair(config).run()
    assert second.rows_corrected == 0, "a repaired lake has nothing left to correct"
    assert second.files_rewritten == 0


def test_dry_run_changes_nothing_on_disk(lake):
    config = _config(lake)
    before = AdjustmentRepair(config).run(dry_run=True)
    assert before.rows_corrected == 2

    with LakeQuery(lake) as q:
        df = q.ohlcv(symbols=["PAYER"]).set_index("date")
    assert df.loc[pd.Timestamp("2026-01-05"), "adj_close"] == pytest.approx(100.0), "untouched"

    assert AdjustmentRepair(config).run(dry_run=True).rows_corrected == 2


def test_repair_leaves_the_fresh_symbol_untouched(lake):
    AdjustmentRepair(_config(lake)).run()
    with LakeQuery(lake) as q:
        df = q.ohlcv(symbols=["FRESH"]).set_index("date")
    assert df.loc[pd.Timestamp("2026-01-05"), "adj_close"] == pytest.approx(49.0)


def test_other_columns_survive_the_rewrite(lake):
    AdjustmentRepair(_config(lake)).run()
    with LakeQuery(lake) as q:
        df = q.ohlcv(symbols=["PAYER"]).set_index("date")
    row = df.loc[pd.Timestamp("2026-02-02")]
    assert row["close"] == pytest.approx(100.0)
    assert row["dividends"] == pytest.approx(2.0)
    assert row["volume"] == 1_000_000
    assert row["source"] == "test"


def test_an_empty_lake_is_survivable(tmp_path):
    DatasetPaths(tmp_path).ensure_layout()
    result = AdjustmentRepair(_config(tmp_path)).run()
    assert result.rows_corrected == 0
    assert not result.errors


def test_a_dividend_larger_than_the_price_is_ignored_not_crashed(tmp_path):
    """A negative factor would make LN() undefined and take the whole scan down."""
    paths = DatasetPaths(tmp_path)
    paths.ensure_layout()
    rows = [
        (date(2026, 1, 5), 10.0, 10.0, 0.0, 0.0),
        (date(2026, 1, 6), 10.0, 10.0, 50.0, 0.0),  # nonsense: dividend > price
    ]
    ParquetWriter().write(
        _bars("BAD", rows, datetime(2026, 1, 1, tzinfo=UTC)),
        P.OHLCV,
        paths.ohlcv_backfill_file(2026, 1),
        mode="overwrite",
    )
    with LakeQuery(tmp_path) as q:
        df = q.adjusted_ohlcv(symbols=["BAD"])
    assert df["adj_factor"].sub(1.0).abs().max() < 1e-9


def test_the_first_session_has_no_previous_close(tmp_path):
    """A dividend on the very first row has nothing to divide by."""
    paths = DatasetPaths(tmp_path)
    paths.ensure_layout()
    rows = [
        (date(2026, 1, 5), 10.0, 10.0, 0.5, 0.0),
        (date(2026, 1, 6), 10.0, 10.0, 0.0, 0.0),
    ]
    ParquetWriter().write(
        _bars("FIRST", rows, datetime(2026, 1, 1, tzinfo=UTC)),
        P.OHLCV,
        paths.ohlcv_backfill_file(2026, 1),
        mode="overwrite",
    )
    with LakeQuery(tmp_path) as q:
        df = q.adjusted_ohlcv(symbols=["FIRST"])
    assert len(df) == 2
    assert df["adj_factor"].notna().all()


def test_empty_symbol_list_returns_an_empty_frame(lake):
    with LakeQuery(lake) as q:
        assert q.adjusted_ohlcv(symbols=[]).empty
        assert q.corrected_adjustments(symbols=[]).empty
