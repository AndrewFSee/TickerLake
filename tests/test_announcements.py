"""Tests for earnings announcement dates.

A surprise row carries ``period`` -- the fiscal period end -- which says nothing
about when the number became public. Dating it from the SEC 8-K that announced
it is what makes the dataset usable point-in-time, and the matching is where it
goes wrong: every case below is drawn from a real mis-pairing found in the lake.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from tickerlake.config import Config, Secrets
from tickerlake.fetchers.announcements import AnnouncementFetcher
from tickerlake.storage import paths as P
from tickerlake.storage.paths import DatasetPaths
from tickerlake.storage.query import LakeQuery
from tickerlake.storage.writer import ParquetWriter
from tickerlake.utils.dates import as_date


def _fetcher(tmp_path, **overrides):
    paths = DatasetPaths(tmp_path)
    paths.ensure_layout()
    config = Config(
        raw={"announcements": {"enabled": True, **overrides}},
        secrets=Secrets(sec_user_agent="TickerLake test test@example.com"),
        data_root=tmp_path,
        config_path=tmp_path / "config.yaml",
    )
    return AnnouncementFetcher(config, paths, ParquetWriter())


def _match(tmp_path, periods, filings, **overrides):
    """Run the matcher and return {period: announcement_date}."""
    fetcher = _fetcher(tmp_path, **overrides)
    rows = fetcher._match("TEST", periods, [(d, f"acc-{d}") for d in filings])
    return {as_date(r["period_key"]): r["announcement_date"] for r in rows}


# --------------------------------------------------- off-calendar fiscal years


def test_announcement_before_its_own_period_label_is_found(tmp_path):
    """The General Mills case, verbatim from SEC.

    GIS closes its quarters at the end of August, November, February and May.
    Finnhub files them under calendar quarter ends, so three of the four are
    announced *before* the label they carry. Requiring the filing to come
    strictly after the period end skipped every one of them and took the next
    quarter's instead -- dating all four rows one quarter late.
    """
    periods = [date(2025, 9, 30), date(2025, 12, 31), date(2026, 3, 31), date(2026, 6, 30)]
    filings = [
        date(2025, 6, 25),
        date(2025, 9, 17),
        date(2025, 12, 17),
        date(2026, 3, 18),
        date(2026, 7, 1),
    ]
    assert _match(tmp_path, periods, filings) == {
        date(2025, 9, 30): date(2025, 9, 17),
        date(2025, 12, 31): date(2025, 12, 17),
        date(2026, 3, 31): date(2026, 3, 18),
        date(2026, 6, 30): date(2026, 7, 1),
    }


def test_calendar_year_end_companies_are_unaffected(tmp_path):
    """Apple announces after each label, and must still pair the same way."""
    periods = [date(2025, 6, 30), date(2025, 9, 30), date(2025, 12, 31), date(2026, 3, 31)]
    filings = [
        date(2025, 7, 31),
        date(2025, 10, 30),
        date(2026, 1, 29),
        date(2026, 4, 30),
        date(2026, 7, 30),
    ]
    assert _match(tmp_path, periods, filings) == {
        date(2025, 6, 30): date(2025, 7, 31),
        date(2025, 9, 30): date(2025, 10, 30),
        date(2025, 12, 31): date(2026, 1, 29),
        date(2026, 3, 31): date(2026, 4, 30),
    }


# -------------------------------------------------------------------- decoys


def test_an_unrelated_item_202_filing_is_not_mistaken_for_the_release(tmp_path):
    """The Goldman Sachs case.

    GS filed an Item 2.02 8-K on 2026-01-08, a week before its actual Q4
    release on 2026-01-15. Taking the earliest filing after the period end
    picked the decoy and dated the quarter seven days early -- lookahead, which
    is the exact failure this dataset exists to prevent.
    """
    periods = [date(2025, 9, 30), date(2025, 12, 31), date(2026, 3, 31), date(2026, 6, 30)]
    filings = [
        date(2025, 10, 14),
        date(2026, 1, 8),
        date(2026, 1, 15),
        date(2026, 4, 13),
        date(2026, 7, 14),
    ]
    matched = _match(tmp_path, periods, filings)
    assert matched[date(2025, 12, 31)] == date(2026, 1, 15), "cadence, not earliness"
    assert date(2026, 1, 8) not in matched.values()


def test_a_filing_far_off_the_companys_cadence_leaves_the_period_undated(tmp_path):
    """Undated beats wrongly dated: a null is visible, a wrong date is not."""
    periods = [date(2025, 3, 31), date(2025, 6, 30), date(2025, 9, 30), date(2025, 12, 31)]
    filings = [
        date(2025, 4, 30),
        date(2025, 7, 30),
        date(2025, 10, 29),
        # Q4 never announced; the only later filing is an unrelated 2.02 four
        # months out, well beyond the ~30-day cadence of the other three.
        date(2026, 4, 20),
    ]
    matched = _match(tmp_path, periods, filings)
    assert date(2025, 12, 31) not in matched
    assert len(matched) == 3


def test_the_cadence_estimate_recovers_from_a_polluted_first_pass(tmp_path):
    """The Honeywell case, which needed more than one pass.

    HON filed an unrelated Item 2.02 on 2025-12-22 and another on 2026-06-29,
    on either side of its real releases. The seed pairing claimed the December
    decoy, dragging the measured offset from 23 days down to 11 -- which then
    put the June decoy and the real 2026-07-23 release exactly equidistant, and
    the tie-break handed it to the decoy. Re-measuring the offset from the
    corrected pairing pulls the median back to 23 and breaks the tie properly.
    """
    periods = [date(2025, 9, 30), date(2025, 12, 31), date(2026, 3, 31), date(2026, 6, 30)]
    filings = [
        date(2025, 7, 24),
        date(2025, 10, 23),
        date(2025, 12, 22),
        date(2026, 1, 29),
        date(2026, 4, 23),
        date(2026, 6, 29),
        date(2026, 7, 23),
    ]
    assert _match(tmp_path, periods, filings) == {
        date(2025, 9, 30): date(2025, 10, 23),
        date(2025, 12, 31): date(2026, 1, 29),
        date(2026, 3, 31): date(2026, 4, 23),
        date(2026, 6, 30): date(2026, 7, 23),
    }


def test_a_late_annual_report_is_not_mistaken_for_a_decoy(tmp_path):
    """Keurig Dr Pepper takes 55 days over Q4 and ~25 over the others.

    Audited annual results legitimately take longer than quarterlies, so the
    cadence tolerance has to be loose enough to keep them.
    """
    periods = [date(2025, 9, 30), date(2025, 12, 31), date(2026, 3, 31)]
    filings = [date(2025, 10, 27), date(2026, 2, 24), date(2026, 4, 23)]
    matched = _match(tmp_path, periods, filings)
    assert matched[date(2025, 12, 31)] == date(2026, 2, 24), "a slow annual is still real"
    assert len(matched) == 3


# --------------------------------------------------------------- exclusivity


def test_one_filing_cannot_serve_two_periods(tmp_path):
    """A single GIS 8-K was recorded as both Q3 and Q4."""
    periods = [date(2026, 3, 31), date(2026, 6, 30)]
    matched = _match(tmp_path, periods, [date(2026, 7, 1)])
    assert len(set(matched.values())) == len(matched)
    assert len(matched) == 1


def test_same_day_filings_collapse_to_one_event(tmp_path):
    """An 8-K and its amendment are one announcement, not two.

    Accession numbers are issued in sequence, so the lower one is the original
    filing and lexicographic order is filing order.
    """
    fetcher = _fetcher(tmp_path)
    rows = fetcher._match(
        "TEST",
        [date(2026, 3, 31)],
        [
            (date(2026, 4, 30), "0000320193-26-000042"),
            (date(2026, 4, 30), "0000320193-26-000041"),
        ],
    )
    assert len(rows) == 1
    assert rows[0]["announcement_accession"] == "0000320193-26-000041"


def test_no_filings_yields_no_rows(tmp_path):
    assert _match(tmp_path, [date(2026, 3, 31)], []) == {}


# ------------------------------------------------------------- item parsing


class _StubClient:
    def __init__(self, payload):
        self.payload = payload

    def get_json(self, url, params=None):
        return self.payload


def test_item_codes_are_matched_whole_not_as_substrings(tmp_path):
    """The code 12.02 contains 2.02, so substring matching would accept it."""
    payload = {
        "filings": {
            "recent": {
                "form": ["8-K", "8-K", "8-K", "10-Q"],
                "items": ["2.02,9.01", "12.02", "7.01", "2.02"],
                "filingDate": ["2026-04-30", "2026-05-01", "2026-05-02", "2026-05-03"],
                "accessionNumber": ["a", "b", "c", "d"],
            }
        }
    }
    found = _fetcher(tmp_path)._item_202_filings(_StubClient(payload), "0000000001")
    assert found == [(date(2026, 4, 30), "a")], "only the real 8-K carrying item 2.02"


def test_missing_filing_blocks_are_survivable(tmp_path):
    found = _fetcher(tmp_path)._item_202_filings(_StubClient({}), "0000000001")
    assert found == []


# ------------------------------------------------------------- enrichment


@pytest.fixture
def lake_with_surprises(tmp_path):
    """Stored surprises carrying EPS figures and no announcement date."""
    paths = DatasetPaths(tmp_path)
    paths.ensure_layout()
    rows = [
        ("AAPL", date(2025, 6, 30), 1.43, 1.57, 9.79),
        ("AAPL", date(2025, 9, 30), 1.77, 1.85, 4.52),
    ]
    df = pd.DataFrame(
        [
            {
                "symbol": sym,
                "record_type": "surprise",
                "period": period,
                "fiscal_year": 2025,
                "fiscal_quarter": 3,
                "eps_estimate": est,
                "eps_actual": actual,
                "eps_surprise_pct": pct,
                "announcement_date": None,
                "announcement_accession": None,
                "source": "finnhub",
                "ingested_at": datetime.now(UTC),
            }
            for sym, period, est, actual, pct in rows
        ]
    )
    ParquetWriter().write(df, P.EARNINGS, paths.earnings_file("surprises"), mode="overwrite")
    return tmp_path


def test_enrichment_preserves_every_eps_field(lake_with_surprises, monkeypatch):
    """The writer replaces on (symbol, period).

    Writing back only the key plus the new date would blank the stored EPS --
    turning the enrichment into data loss. The whole row is carried through for
    exactly this reason, so it is worth holding in place.
    """
    import tickerlake.fetchers.announcements as mod

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def get_json(self, url, params=None):
            if url == mod.TICKER_MAP_URL:
                return {"0": {"ticker": "AAPL", "cik_str": 320193}}
            return {
                "filings": {
                    "recent": {
                        "form": ["8-K", "8-K"],
                        "items": ["2.02,9.01", "2.02,9.01"],
                        "filingDate": ["2025-07-31", "2025-10-30"],
                        "accessionNumber": ["acc-q3", "acc-q4"],
                    }
                }
            }

        def close(self):
            pass

    monkeypatch.setattr(mod, "HttpClient", FakeClient)

    fetcher = _fetcher(lake_with_surprises)
    result = fetcher.run(date(2026, 1, 2))
    assert result.ok, result.errors

    with LakeQuery(lake_with_surprises) as q:
        out = q.sql("SELECT * FROM earnings WHERE record_type='surprise' ORDER BY period")

    assert len(out) == 2, "enrichment must not multiply rows"
    assert out["eps_actual"].notna().all(), "EPS survived the merge"
    assert out["announcement_date"].notna().all()
    assert as_date(out.iloc[0]["announcement_date"]) == date(2025, 7, 31)
    assert as_date(out.iloc[1]["announcement_date"]) == date(2025, 10, 30)
    assert out.iloc[0]["eps_actual"] == pytest.approx(1.57)


# ------------------------------------------------------------ point-in-time


@pytest.fixture
def dated_lake(tmp_path):
    """Surprises with announcement dates, plus one that could not be matched."""
    paths = DatasetPaths(tmp_path)
    paths.ensure_layout()
    rows = [
        ("AAPL", date(2025, 6, 30), 1.57, date(2025, 7, 31)),
        ("AAPL", date(2025, 9, 30), 1.85, date(2025, 10, 30)),
        ("MSFT", date(2025, 6, 30), 3.65, date(2025, 7, 29)),
        # No 8-K could be matched, so the date is unknown -- not assumed.
        ("MSFT", date(2025, 9, 30), 3.72, None),
    ]
    df = pd.DataFrame(
        [
            {
                "symbol": sym,
                "record_type": "surprise",
                "period": period,
                "fiscal_year": 2025,
                "fiscal_quarter": 3,
                "eps_estimate": actual - 0.05,
                "eps_actual": actual,
                "eps_surprise_pct": 2.0,
                "announcement_date": announced,
                "announcement_accession": "acc" if announced else None,
                "source": "finnhub",
                "ingested_at": datetime.now(UTC),
            }
            for sym, period, actual, announced in rows
        ]
    )
    ParquetWriter().write(df, P.EARNINGS, paths.earnings_file("surprises"), mode="overwrite")
    return tmp_path


def test_a_surprise_is_invisible_before_it_is_announced(dated_lake):
    """The core guarantee, mirroring pit_fundamentals."""
    with LakeQuery(dated_lake) as q:
        # The September quarter closed on the 30th but was not announced until
        # 30 October. A month of hindsight sits in between.
        before = q.pit_earnings("2025-10-15", symbols=["AAPL"])
        after = q.pit_earnings("2025-11-15", symbols=["AAPL"])

    assert as_date(before.iloc[0]["period"]) == date(2025, 6, 30), "still the June quarter"
    assert as_date(after.iloc[0]["period"]) == date(2025, 9, 30)


def test_the_announcement_day_itself_is_included(dated_lake):
    with LakeQuery(dated_lake) as q:
        eve = q.pit_earnings("2025-10-29", symbols=["AAPL"])
        day = q.pit_earnings("2025-10-30", symbols=["AAPL"])

    assert as_date(eve.iloc[0]["period"]) == date(2025, 6, 30)
    assert as_date(day.iloc[0]["period"]) == date(2025, 9, 30), "<= as_of is inclusive"


def test_undated_surprises_are_excluded_rather_than_assumed(dated_lake):
    """An unmatched 8-K means the date is unknown; guessing reintroduces bias."""
    with LakeQuery(dated_lake) as q:
        df = q.pit_earnings("2026-06-01", symbols=["MSFT"])

    assert len(df) == 1
    assert as_date(df.iloc[0]["period"]) == date(2025, 6, 30)


def test_one_row_per_symbol_by_default(dated_lake):
    with LakeQuery(dated_lake) as q:
        df = q.pit_earnings("2026-06-01")
    assert len(df) == 2
    assert set(df["symbol"]) == {"AAPL", "MSFT"}


def test_quarters_returns_a_surprise_history(dated_lake):
    with LakeQuery(dated_lake) as q:
        df = q.pit_earnings("2026-06-01", symbols=["AAPL"], quarters=4)
    assert len(df) == 2, "only two are stored"
    assert as_date(df.iloc[0]["period"]) == date(2025, 9, 30), "most recent first"


def test_nothing_is_known_before_the_first_announcement(dated_lake):
    with LakeQuery(dated_lake) as q:
        assert q.pit_earnings("2020-01-01").empty


def test_empty_symbol_list_returns_an_empty_frame(dated_lake):
    with LakeQuery(dated_lake) as q:
        assert q.pit_earnings("2026-06-01", symbols=[]).empty
