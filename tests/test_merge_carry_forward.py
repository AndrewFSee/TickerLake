"""Tests for preserving stored columns across a merge.

Several stages write to the same dataset, and each knows only about its own
columns. ``earnings`` re-fetching a surprise row wiped the ``announcement_date``
that ``announcements`` had just resolved: pandas unions the columns on concat,
so the incoming row arrived carrying a NaN for a field it had never heard of,
and ``keep="last"`` handed it the win.

Only stage ordering hid the damage -- ``announcements`` runs later in the same
pipeline and repaired it each night. That is not a property worth relying on.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pyarrow.parquet as pq
import pytest

from tickerlake.storage import paths as P
from tickerlake.storage.paths import DatasetPaths
from tickerlake.storage.writer import ParquetWriter


@pytest.fixture
def lake(tmp_path):
    paths = DatasetPaths(tmp_path)
    paths.ensure_layout()
    return paths, ParquetWriter()


def _surprise(**overrides):
    row = {
        "symbol": "HON",
        "record_type": "surprise",
        "period": date(2026, 6, 30),
        "eps_estimate": 4.40,
        "eps_actual": 4.52,
        "source": "finnhub",
        "ingested_at": datetime.now(UTC),
    }
    row.update(overrides)
    return row


def _read(path):
    return pq.read_table(path).to_pandas()


# ------------------------------------------------------------------ absent


def test_a_column_the_incoming_frame_lacks_is_preserved(lake):
    """The exact production failure, end to end."""
    paths, w = lake
    target = paths.earnings_file("surprises")

    # announcements resolves the date.
    w.write(
        pd.DataFrame(
            [
                _surprise(
                    announcement_date=date(2026, 7, 23),
                    announcement_accession="0000773840-26-000055",
                )
            ]
        ),
        P.EARNINGS,
        target,
        mode="overwrite",
    )

    # earnings re-fetches the same period. Its frame has no announcement
    # columns at all -- earnings.py does not know they exist.
    w.write(pd.DataFrame([_surprise(eps_actual=4.55)]), P.EARNINGS, target, mode="merge")

    out = _read(target)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["announcement_date"] == date(2026, 7, 23), "the stored date must survive"
    assert row["announcement_accession"] == "0000773840-26-000055"
    assert row["eps_actual"] == pytest.approx(4.55), "but the fetched value still wins"


def test_the_pipeline_no_longer_depends_on_stage_order(lake):
    """Running earnings after announcements must be safe, not merely untested.

    Reversing the two stages used to lose every date. The whole point of fixing
    this at the writer is that ordering stops mattering.
    """
    paths, w = lake
    target = paths.earnings_file("surprises")
    dated = [
        _surprise(period=date(2026, 3, 31), announcement_date=date(2026, 4, 23)),
        _surprise(period=date(2026, 6, 30), announcement_date=date(2026, 7, 23)),
    ]
    w.write(pd.DataFrame(dated), P.EARNINGS, target, mode="overwrite")

    for _ in range(3):  # three more nightly earnings runs
        w.write(
            pd.DataFrame(
                [_surprise(period=date(2026, 3, 31)), _surprise(period=date(2026, 6, 30))]
            ),
            P.EARNINGS,
            target,
            mode="merge",
        )

    out = _read(target)
    assert len(out) == 2
    assert out["announcement_date"].notna().all(), "dates must not decay run by run"


# -------------------------------------------------------------------- null


def test_a_column_present_but_null_still_overwrites(lake):
    """Absent and null are different statements.

    A missing column is an absence of information, so the stored value stands.
    A column that is present and null says the value is unknown, and must be
    able to clear a stale one -- otherwise nothing could ever be un-set.
    """
    paths, w = lake
    target = paths.earnings_file("surprises")

    w.write(
        pd.DataFrame([_surprise(announcement_date=date(2026, 7, 23))]),
        P.EARNINGS,
        target,
        mode="overwrite",
    )
    w.write(
        pd.DataFrame([_surprise(announcement_date=None)]),
        P.EARNINGS,
        target,
        mode="merge",
    )

    out = _read(target)
    assert out["announcement_date"].isna().all(), "an explicit null is a real statement"


# ------------------------------------------------------------------- scope


def test_rows_whose_key_is_new_get_nothing(lake):
    """There is no stored value to preserve, so the column stays null."""
    paths, w = lake
    target = paths.earnings_file("surprises")

    w.write(
        pd.DataFrame([_surprise(announcement_date=date(2026, 7, 23))]),
        P.EARNINGS,
        target,
        mode="overwrite",
    )
    w.write(pd.DataFrame([_surprise(period=date(2026, 9, 30))]), P.EARNINGS, target, mode="merge")

    out = _read(target).sort_values("period")
    assert len(out) == 2
    assert out.iloc[0]["announcement_date"] == date(2026, 7, 23), "the old row keeps its date"
    assert pd.isna(out.iloc[1]["announcement_date"]), "the new row must not inherit one"


def test_values_are_matched_per_key_not_positionally(lake):
    """Two symbols, written back in a different order."""
    paths, w = lake
    target = paths.earnings_file("surprises")

    w.write(
        pd.DataFrame(
            [
                _surprise(symbol="AAA", announcement_date=date(2026, 7, 1)),
                _surprise(symbol="BBB", announcement_date=date(2026, 8, 2)),
            ]
        ),
        P.EARNINGS,
        target,
        mode="overwrite",
    )
    w.write(
        pd.DataFrame([_surprise(symbol="BBB"), _surprise(symbol="AAA")]),
        P.EARNINGS,
        target,
        mode="merge",
    )

    out = _read(target).set_index("symbol")
    assert out.loc["AAA", "announcement_date"] == date(2026, 7, 1)
    assert out.loc["BBB", "announcement_date"] == date(2026, 8, 2)


def test_the_three_column_earnings_key_matches_on_all_of_it(lake):
    """The key is (symbol, record_type, period), so the lookup is a MultiIndex.

    Two records differing only in ``record_type`` must not borrow each other's
    stored values.
    """
    paths, w = lake
    target = paths.earnings_file("surprises")

    w.write(
        pd.DataFrame(
            [
                _surprise(record_type="surprise", announcement_date=date(2026, 7, 23)),
                _surprise(record_type="calendar", announcement_date=date(2026, 9, 9)),
            ]
        ),
        P.EARNINGS,
        target,
        mode="overwrite",
    )
    w.write(
        pd.DataFrame([_surprise(record_type="surprise")]),
        P.EARNINGS,
        target,
        mode="merge",
    )

    out = _read(target).set_index("record_type")
    assert out.loc["surprise", "announcement_date"] == date(2026, 7, 23)
    assert out.loc["calendar", "announcement_date"] == date(2026, 9, 9), "untouched"


def test_a_dataset_without_a_natural_key_is_left_alone(lake):
    """filings_facts declares no unique_on, so there is nothing to match on.

    It is written with mode="overwrite" per symbol so this never arises in the
    pipeline, but the guard must not raise if it ever does.
    """
    paths, w = lake
    target = paths.filings_facts_file("TEST")
    now = datetime.now(UTC)
    fact = {
        "cik": "0000773840",
        "symbol": "HON",
        "taxonomy": "us-gaap",
        "concept": "Assets",
        "unit": "USD",
        "value": 1.0,
        "end_date": date(2026, 6, 30),
        "filed_date": date(2026, 7, 23),
        "form_type": "10-Q",
        "accession_number": "acc-1",
        "frame": "CY2026Q2I",
        "source": "test",
        "ingested_at": now,
    }
    w.write(pd.DataFrame([fact]), P.FILINGS_FACTS, target, mode="overwrite")
    w.write(
        pd.DataFrame([{**fact, "value": 9.0}]).drop(columns=["frame"]),
        P.FILINGS_FACTS,
        target,
        mode="merge",
    )
    assert len(_read(target)) == 2, "no key means no de-duplication, and no carry-forward"


def test_ordinary_columns_are_untouched(lake):
    """Nothing here should make a present column sticky."""
    paths, w = lake
    target = paths.earnings_file("surprises")

    w.write(pd.DataFrame([_surprise(eps_actual=4.52)]), P.EARNINGS, target, mode="overwrite")
    w.write(pd.DataFrame([_surprise(eps_actual=4.99)]), P.EARNINGS, target, mode="merge")

    assert _read(target).iloc[0]["eps_actual"] == pytest.approx(4.99)


def test_a_frame_missing_a_key_column_is_left_alone():
    """Without the full key there is no way to say which stored row matches.

    Exercised directly: an earnings frame with no ``period`` cannot reach here
    through ``write()``, because the column is non-nullable and Arrow rejects
    it first. The guard still has to hold for any dataset where it can.
    """
    from tickerlake.storage.writer import _carry_forward

    existing = pd.DataFrame([_surprise(announcement_date=date(2026, 7, 23))])
    incoming = pd.DataFrame([_surprise()]).drop(columns=["period"])

    out = _carry_forward(incoming, existing, P.EARNINGS)
    assert "announcement_date" not in out.columns, "nothing may be guessed"
    assert list(out.columns) == list(incoming.columns)


def test_an_empty_frame_on_either_side_is_survivable():
    from tickerlake.storage.writer import _carry_forward

    row = pd.DataFrame([_surprise(announcement_date=date(2026, 7, 23))])
    empty = pd.DataFrame(columns=list(row.columns))

    assert _carry_forward(empty, row, P.EARNINGS).empty
    assert (
        "announcement_date"
        not in _carry_forward(pd.DataFrame([_surprise()]), empty, P.EARNINGS).columns
    )
