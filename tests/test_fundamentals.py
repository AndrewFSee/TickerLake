"""Tests for point-in-time fundamentals.

The lake is survivorship-bias-free by construction; fundamentals are where
*lookahead* bias gets in instead. A period ending 2025-09-27 is not public until
the filing lands on 2025-10-31, so anything keyed on ``end_date`` hands a model
34 days of the future -- and far more for restatements, where the same figure
reappears in a filing 398 days after its period closed.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from tickerlake.storage import paths as P
from tickerlake.storage.paths import DatasetPaths
from tickerlake.storage.query import LakeQuery
from tickerlake.storage.writer import ParquetWriter
from tickerlake.utils.dates import as_date


@pytest.fixture
def lake(tmp_path):
    """A miniature filings_facts modelled on the real AAPL shape."""
    paths = DatasetPaths(tmp_path)
    paths.ensure_layout()
    now = datetime.now(UTC)

    rows = [
        # Q3 FY2025: period ends June, filed August.
        ("AAPL", "NetIncomeLoss", date(2025, 6, 28), date(2025, 8, 1), "10-Q", 2025, "Q3", 23.4e9),
        # FY2025: period ends September, not filed until 31 October.
        (
            "AAPL",
            "NetIncomeLoss",
            date(2025, 9, 27),
            date(2025, 10, 31),
            "10-K",
            2025,
            "FY",
            112.0e9,
        ),
        # The FY2024 comparative restated inside the FY2025 filing: note the
        # fiscal_year says 2025 while the period is 2024.
        (
            "AAPL",
            "NetIncomeLoss",
            date(2024, 9, 28),
            date(2025, 10, 31),
            "10-K",
            2025,
            "FY",
            93.7e9,
        ),
        # The original FY2024 filing.
        ("AAPL", "NetIncomeLoss", date(2024, 9, 28), date(2024, 11, 1), "10-K", 2024, "FY", 93.7e9),
        # Revenue under both tags, to exercise the ASC 606 coalesce.
        ("AAPL", "Revenues", date(2017, 9, 30), date(2017, 11, 3), "10-K", 2017, "FY", 229.2e9),
        (
            "AAPL",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            date(2025, 9, 27),
            date(2025, 10, 31),
            "10-K",
            2025,
            "FY",
            416.2e9,
        ),
        ("MSFT", "NetIncomeLoss", date(2025, 6, 30), date(2025, 7, 29), "10-K", 2025, "FY", 88.1e9),
    ]
    df = pd.DataFrame(
        [
            {
                "cik": "0000320193",
                "symbol": sym,
                "taxonomy": "us-gaap",
                "concept": concept,
                "unit": "USD",
                "value": value,
                "start_date": None,
                "end_date": end,
                "fiscal_year": fy,
                "fiscal_period": fp,
                "form_type": form,
                "filed_date": filed,
                "accession_number": f"{sym}-{end}-{filed}",
                "frame": None,
                "source": "test",
                "ingested_at": now,
            }
            for sym, concept, end, filed, form, fy, fp, value in rows
        ]
    )
    ParquetWriter().write(df, P.FILINGS_FACTS, paths.filings_facts_file("TEST"), mode="overwrite")
    return paths.root


# --------------------------------------------------------------- lookahead


def test_unfiled_figures_are_invisible(lake):
    """The core guarantee: a period is not knowable before it is filed."""
    with LakeQuery(lake) as q:
        # FY2025 ended 2025-09-27 but was not filed until 2025-10-31.
        before = q.pit_fundamentals("2025-10-15", ["NetIncomeLoss"], symbols=["AAPL"])
        after = q.pit_fundamentals("2025-11-15", ["NetIncomeLoss"], symbols=["AAPL"])

    assert before.iloc[0]["value"] == pytest.approx(23.4e9), "must still be the Q3 figure"
    # DuckDB returns date32 as Timestamp, which is not equal to a plain date -
    # the same trap utils.dates exists for. Coerce rather than compare raw.
    assert as_date(before.iloc[0]["end_date"]) == date(2025, 6, 28)
    assert after.iloc[0]["value"] == pytest.approx(112.0e9), "annual is known once filed"
    assert as_date(after.iloc[0]["end_date"]) == date(2025, 9, 27)


def test_the_day_before_filing_still_excludes_it(lake):
    with LakeQuery(lake) as q:
        eve = q.pit_fundamentals("2025-10-30", ["NetIncomeLoss"], symbols=["AAPL"])
        day = q.pit_fundamentals("2025-10-31", ["NetIncomeLoss"], symbols=["AAPL"])

    assert as_date(eve.iloc[0]["end_date"]) == date(2025, 6, 28)
    assert as_date(day.iloc[0]["end_date"]) == date(2025, 9, 27), "filed_date <= as_of is inclusive"


def test_nothing_is_returned_before_the_first_filing(lake):
    with LakeQuery(lake) as q:
        df = q.pit_fundamentals("2016-01-01", ["NetIncomeLoss"], symbols=["AAPL"])
    assert df.empty


def test_one_row_per_symbol_and_concept(lake):
    """Restatements must not multiply rows."""
    with LakeQuery(lake) as q:
        df = q.pit_fundamentals("2026-01-01", ["NetIncomeLoss"], symbols=["AAPL", "MSFT"])
    assert len(df) == 2
    assert set(df["symbol"]) == {"AAPL", "MSFT"}


def test_latest_period_wins_not_latest_filing(lake):
    """The FY2024 comparative was filed in 2025, but FY2025 is the newer period."""
    with LakeQuery(lake) as q:
        df = q.pit_fundamentals("2026-01-01", ["NetIncomeLoss"], symbols=["AAPL"])
    row = df.iloc[0]
    assert as_date(row["end_date"]) == date(2025, 9, 27)
    assert row["value"] == pytest.approx(112.0e9)
    assert row["fiscal_year"] == 2025


# ------------------------------------------------------------- asc 606


def test_revenue_coalesces_both_tags(lake):
    """Querying one revenue tag alone loses roughly half the universe."""
    with LakeQuery(lake) as q:
        df = q.pit_revenue("2026-01-01", symbols=["AAPL"])

    assert len(df) == 1
    row = df.iloc[0]
    assert row["value"] == pytest.approx(416.2e9), "the ASC 606 tag is the current one"
    assert row["concept"] == "RevenueFromContractWithCustomerExcludingAssessedTax"


def test_revenue_falls_back_to_the_legacy_tag(lake):
    """Before ASC 606 only the old tag exists, and it must still be found."""
    with LakeQuery(lake) as q:
        df = q.pit_revenue("2018-01-01", symbols=["AAPL"])

    assert len(df) == 1
    assert df.iloc[0]["concept"] == "Revenues"
    assert df.iloc[0]["value"] == pytest.approx(229.2e9)


def test_revenue_concept_list_covers_both_eras():
    assert "Revenues" in LakeQuery.REVENUE_CONCEPTS
    assert "RevenueFromContractWithCustomerExcludingAssessedTax" in LakeQuery.REVENUE_CONCEPTS


# ------------------------------------------------------------- filtering


def test_form_type_filter(lake):
    with LakeQuery(lake) as q:
        annual = q.pit_fundamentals(
            "2026-01-01", ["NetIncomeLoss"], symbols=["AAPL"], form_types=["10-K"]
        )
        quarterly = q.pit_fundamentals(
            "2025-10-15", ["NetIncomeLoss"], symbols=["AAPL"], form_types=["10-Q"]
        )
    assert annual.iloc[0]["form_type"] == "10-K"
    assert quarterly.iloc[0]["form_type"] == "10-Q"


def test_empty_inputs_return_empty_frames(lake):
    with LakeQuery(lake) as q:
        assert q.pit_fundamentals("2026-01-01", []).empty
        assert q.pit_fundamentals("2026-01-01", ["NetIncomeLoss"], symbols=[]).empty


def test_concept_discovery(lake):
    with LakeQuery(lake) as q:
        df = q.fundamentals_concepts(limit=10)
    assert "NetIncomeLoss" in set(df["concept"])
    assert df["symbols"].max() >= 2
