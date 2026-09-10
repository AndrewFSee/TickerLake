"""Tests for the storage layer: schema coercion, idempotency, validation, query."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pyarrow.parquet as pq
import pytest

from tickerlake.storage import paths as P
from tickerlake.storage import schemas as S
from tickerlake.storage.paths import DatasetPaths, safe_symbol
from tickerlake.storage.query import LakeQuery
from tickerlake.storage.writer import ParquetWriter


@pytest.fixture
def lake(tmp_path):
    paths = DatasetPaths(tmp_path)
    paths.ensure_layout()
    return paths, ParquetWriter()


def _ohlcv_rows(symbol="AAPL", days=3, close=100.0):
    now = datetime.now(UTC)
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "date": date(2024, 3, day),
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close + day,
                "adj_close": close + day,
                "volume": 1_000 * day,
                "dividends": 0.0,
                "stock_splits": 0.0,
                "repaired": False,
                "source": "test",
                "ingested_at": now,
            }
            for day in range(1, days + 1)
        ]
    )


# ------------------------------------------------------------------- paths


def test_safe_symbol_handles_awkward_tickers():
    assert safe_symbol("BRK.B") == "BRK.B"  # dots are legal in filenames
    assert safe_symbol("brk-b") == "BRK-B"
    assert safe_symbol("A/B") == "A_B"  # slashes would create directories
    assert safe_symbol("^GSPC") == "_GSPC"


# ------------------------------------------------------------------ writing


def test_write_is_idempotent(lake):
    paths, writer = lake
    path = paths.ohlcv_backfill_file(2024, 3)
    df = _ohlcv_rows()

    first = writer.write(df, P.OHLCV, path, mode="overwrite")
    second = writer.write(df, P.OHLCV, path, mode="overwrite")

    assert first.rows_written == second.rows_written == 3
    assert pq.read_metadata(path).num_rows == 3, "re-running a day must not duplicate rows"


def test_merge_deduplicates_on_natural_key(lake):
    paths, writer = lake
    path = paths.ohlcv_backfill_file(2024, 3)

    writer.write(_ohlcv_rows(close=100.0), P.OHLCV, path, mode="merge")
    # Same symbol/date, revised prices - the newer values must win.
    writer.write(_ohlcv_rows(close=200.0), P.OHLCV, path, mode="merge")

    out = pq.read_table(path).to_pandas()
    assert len(out) == 3, "merge must de-duplicate on (symbol, date)"
    assert out["open"].unique().tolist() == [200.0], "merge must keep the newest row"


def test_merge_preserves_other_symbols(lake):
    paths, writer = lake
    path = paths.ohlcv_backfill_file(2024, 3)

    writer.write(_ohlcv_rows("AAPL"), P.OHLCV, path, mode="merge")
    writer.write(_ohlcv_rows("MSFT"), P.OHLCV, path, mode="merge")

    out = pq.read_table(path).to_pandas()
    assert set(out["symbol"]) == {"AAPL", "MSFT"}
    assert len(out) == 6


def test_schema_is_enforced_and_extra_columns_dropped(lake):
    paths, writer = lake
    path = paths.ohlcv_backfill_file(2024, 3)

    df = _ohlcv_rows()
    df["surprise_column"] = "unexpected"
    df["volume"] = df["volume"].astype(str)  # source returns the wrong dtype

    writer.write(df, P.OHLCV, path, mode="overwrite")

    table = pq.read_table(path)
    assert "surprise_column" not in table.column_names
    assert table.schema.field("volume").type == S.OHLCV_SCHEMA.field("volume").type
    assert table.schema.equals(S.OHLCV_SCHEMA)


def test_uncoercible_values_become_null_not_an_exception(lake):
    paths, writer = lake
    path = paths.ohlcv_backfill_file(2024, 3)

    df = _ohlcv_rows()
    # Widen first: pandas refuses to set a string into a float64 column, but a
    # real source hands back object-dtype columns with junk mixed in.
    df["close"] = df["close"].astype(object)
    df.loc[0, "close"] = "not-a-number"

    result = writer.write(df, P.OHLCV, path, mode="overwrite")
    out = pq.read_table(path).to_pandas()

    assert result.rows_written == 3, "one bad cell must not discard the whole frame"
    assert pd.isna(out.loc[out["date"] == date(2024, 3, 1), "close"].iloc[0])


def test_atomic_write_leaves_no_temp_file(lake):
    paths, writer = lake
    path = paths.ohlcv_backfill_file(2024, 3)
    writer.write(_ohlcv_rows(), P.OHLCV, path, mode="overwrite")

    assert path.exists()
    assert not list(path.parent.glob("*.tmp"))


def test_empty_frame_is_skipped_not_written(lake):
    paths, writer = lake
    path = paths.ohlcv_backfill_file(2024, 3)
    result = writer.write(pd.DataFrame(), P.OHLCV, path, mode="overwrite")

    assert result.skipped
    assert not path.exists()


# --------------------------------------------------------------- validation


def test_validation_flags_nulls_in_required_columns():
    df = _ohlcv_rows()
    df.loc[0, "symbol"] = None
    report = S.validate_frame(df, P.OHLCV)

    assert not report.ok
    assert any("symbol" in e for e in report.errors)


def test_validation_flags_all_null_close():
    df = _ohlcv_rows()
    df["close"] = None
    report = S.validate_frame(df, P.OHLCV)

    assert not report.ok
    assert any("entirely null" in e for e in report.errors)


def test_validation_rejects_undersized_membership_table():
    df = pd.DataFrame(
        [{"symbol": "A", "index_name": "SP500", "start_date": date(2010, 1, 1), "source": "t"}]
    )
    report = S.validate_frame(df, P.MEMBERSHIP)

    assert not report.ok, "a 1-row S&P 500 membership table means a broken parse"


def test_strict_validation_raises():
    df = _ohlcv_rows()
    df["close"] = None
    with pytest.raises(S.ValidationError):
        S.validate_frame(df, P.OHLCV, strict=True)


# -------------------------------------------------------------------- query


def test_query_works_on_an_empty_lake(tmp_path):
    DatasetPaths(tmp_path).ensure_layout()
    with LakeQuery(tmp_path) as q:
        # Typed-empty views mean queries work before any data is collected.
        assert q.ohlcv().empty
        assert q.members_on(date(2024, 1, 1)) == []
        assert q.sql("SELECT COUNT(*) AS n FROM ohlcv")["n"].iloc[0] == 0


def test_query_reads_written_data(lake):
    paths, writer = lake
    writer.write(_ohlcv_rows("AAPL"), P.OHLCV, paths.ohlcv_backfill_file(2024, 3), mode="overwrite")

    with LakeQuery(paths.root) as q:
        df = q.ohlcv(symbols=["AAPL"], start=date(2024, 3, 1), end=date(2024, 3, 3))
        assert len(df) == 3
        assert set(df["symbol"]) == {"AAPL"}


def test_ohlcv_date_range_prunes_partitions(lake):
    paths, writer = lake
    for month in (1, 2, 3):
        rows = _ohlcv_rows()
        rows["date"] = [date(2024, month, d) for d in range(1, 4)]
        writer.write(rows, P.OHLCV, paths.ohlcv_backfill_file(2024, month), mode="overwrite")

    with LakeQuery(paths.root) as q:
        # Only February's partition should contribute.
        df = q.ohlcv(start=date(2024, 2, 1), end=date(2024, 2, 28))
        assert len(df) == 3
        assert all(d.month == 2 for d in df["date"])


def test_pit_ohlcv_excludes_pre_membership_rows(lake, tmp_path):
    from tickerlake.universe.membership import MembershipTracker

    paths, writer = lake
    tracker = MembershipTracker(paths, writer, index_name="SP500", history_start=date(2010, 1, 1))
    tracker.seed(
        pd.DataFrame(
            [{"symbol": "LATE", "start_date": date(2024, 3, 2), "end_date": None}],
            columns=["symbol", "start_date", "end_date"],
        ),
        force=True,
    )
    writer.write(_ohlcv_rows("LATE"), P.OHLCV, paths.ohlcv_backfill_file(2024, 3), mode="overwrite")

    with LakeQuery(paths.root) as q:
        raw = q.ohlcv(symbols=["LATE"])
        pit = q.pit_ohlcv(date(2024, 3, 1), date(2024, 3, 3))

    assert len(raw) == 3, "the raw table keeps everything"
    # March 1 predates membership, so a survivorship-safe panel must drop it.
    assert len(pit) == 2
    assert date(2024, 3, 1) not in set(pit["date"])
