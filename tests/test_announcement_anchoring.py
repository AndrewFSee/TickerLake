"""Tests for dating announcements against each company's true fiscal close.

Finnhub labels every quarter with a calendar-quarter end, and the label follows
no single rule relative to the real close: Coca-Cola's quarter ending
2026-04-03 is labelled 2026-03-31, three days *before*; Applied Materials'
ending 2026-07-26 is labelled 2026-09-30, sixty-six days *after*. Every attempt
to infer the close from the label was wrong by a quarter for someone -- one of
them corrupted eighteen symbols with the previous quarter's announcement. These
tests pin the approach that replaced it, each against a real case.
"""

from __future__ import annotations

from datetime import date

import pytest

from tickerlake.config import Config, Secrets
from tickerlake.fetchers.announcements import AnnouncementFetcher
from tickerlake.storage.paths import DatasetPaths
from tickerlake.storage.writer import ParquetWriter
from tickerlake.utils.dates import as_date


@pytest.fixture
def fetcher(tmp_path):
    paths = DatasetPaths(tmp_path)
    paths.ensure_layout()
    config = Config(
        raw={"announcements": {"enabled": True}},
        secrets=Secrets(sec_user_agent="TickerLake test test@example.com"),
        data_root=tmp_path,
        config_path=tmp_path / "config.yaml",
    )
    return AnnouncementFetcher(config, paths, ParquetWriter())


def _recent(periodic: list[tuple[str, str, str]], eightk: list[str] = ()) -> dict:
    """A submissions ``recent`` block.

    periodic: (form, reportDate, filingDate); eightk: Item 2.02 filing dates.
    A 10-Q is numbered by its distance from the preceding 10-K, so every
    fixture carries a year end -- as every real record does.
    """
    forms, reports, filed, items, acc = [], [], [], [], []
    for n, (form, report, when) in enumerate(periodic):
        forms.append(form)
        reports.append(report)
        filed.append(when)
        items.append("")
        acc.append(f"p-{n}")
    for n, when in enumerate(eightk):
        forms.append("8-K")
        reports.append("")
        filed.append(when)
        items.append("2.02,9.01")
        acc.append(f"k-{n}")
    return {
        "form": forms,
        "reportDate": reports,
        "filingDate": filed,
        "items": items,
        "accessionNumber": acc,
    }


def _date(fetcher, recent, rows):
    """Run the full anchored path; return {label: announcement_date}."""
    anchors = fetcher._anchors(rows, fetcher._quarter_closes(recent), fetcher._report_filed(recent))
    assert anchors is not None, "every period should anchor"
    out = fetcher._match("X", [p for p, _ in rows], fetcher._parse_202(recent), anchors)
    return {as_date(r["period_key"]): r["announcement_date"] for r in out}


# ------------------------------------------------------------ fiscal closes


def test_quarter_numbers_come_from_position_after_the_year_end(fetcher):
    """Costco runs 12/12/12/16-week quarters; position still numbers them."""
    recent = _recent(
        [
            ("10-K", "2025-08-31", "2025-10-08"),
            ("10-Q", "2025-11-23", "2025-12-18"),
            ("10-Q", "2026-02-15", "2026-03-12"),
            ("10-Q", "2026-05-10", "2026-06-04"),
        ]
    )
    assert fetcher._quarter_closes(recent) == [
        (date(2025, 8, 31), 4),
        (date(2025, 11, 23), 1),
        (date(2026, 2, 15), 2),
        (date(2026, 5, 10), 3),
    ]


def test_an_amendment_does_not_add_a_second_close(fetcher):
    recent = _recent(
        [
            ("10-Q", "2026-03-31", "2026-05-01"),
            ("10-Q/A", "2026-03-31", "2026-06-15"),
            ("10-K", "2025-12-31", "2026-02-20"),
        ]
    )
    assert fetcher._quarter_closes(recent) == [(date(2025, 12, 31), 4), (date(2026, 3, 31), 1)]
    assert fetcher._report_filed(recent)[date(2026, 3, 31)] == date(2026, 5, 1), "the original"


# --------------------------------------------------------------- anchoring


def test_a_close_after_its_label_is_found(fetcher):
    """Coca-Cola: the Q1 close (Apr 3) falls three days after its label.

    Assuming the close always precedes the label anchored this quarter to the
    *previous* close, and dated it with the previous quarter's announcement.
    """
    recent = _recent(
        [
            ("10-K", "2025-12-31", "2026-02-20"),
            ("10-Q", "2026-04-03", "2026-04-30"),
            ("10-Q", "2026-07-03", "2026-07-30"),
        ],
        ["2026-02-10", "2026-04-28", "2026-07-28"],
    )
    got = _date(fetcher, recent, [(date(2026, 3, 31), 1), (date(2026, 6, 30), 2)])
    assert got == {date(2026, 3, 31): date(2026, 4, 28), date(2026, 6, 30): date(2026, 7, 28)}


def test_a_close_long_before_its_label_is_found(fetcher):
    """Applied Materials: the close sits 66 days before the label.

    A label-centred window just short of that missed the real release and took
    the next quarter's, dating every quarter one late.
    """
    recent = _recent(
        [
            ("10-K", "2025-10-26", "2025-12-12"),
            ("10-Q", "2026-01-25", "2026-02-19"),
            ("10-Q", "2026-04-26", "2026-05-21"),
            ("10-Q", "2026-07-26", "2026-08-20"),
        ],
        ["2025-11-13", "2026-02-12", "2026-05-14", "2026-08-13"],
    )
    rows = [
        (date(2025, 12, 31), 4),
        (date(2026, 3, 31), 1),
        (date(2026, 6, 30), 2),
        (date(2026, 9, 30), 3),
    ]
    assert _date(fetcher, recent, rows) == {
        date(2025, 12, 31): date(2025, 11, 13),
        date(2026, 3, 31): date(2026, 2, 12),
        date(2026, 6, 30): date(2026, 5, 14),
        date(2026, 9, 30): date(2026, 8, 13),
    }


def test_the_quarter_number_must_match(fetcher):
    """Two closes can sit near one label; only the right quarter qualifies."""
    closes = [(date(2026, 4, 3), 1), (date(2026, 7, 3), 2)]
    anchors = fetcher._anchors([(date(2026, 6, 30), 2)], closes, {})
    assert anchors[date(2026, 6, 30)][0] == date(2026, 7, 3)


def test_a_quarter_numbering_disagreement_leaves_it_unanchored(fetcher):
    """If Finnhub and SEC ever disagree on the quarter, fail safe.

    The wrong-numbered close is about a quarter away, outside the window, so
    the symbol falls back rather than borrowing a neighbour's announcement.
    """
    closes = [(date(2026, 3, 31), 1), (date(2026, 6, 30), 2)]
    assert fetcher._anchors([(date(2026, 6, 30), 1)], closes, {}) is None


def test_the_newest_close_is_projected_from_a_year_earlier(fetcher):
    """Paychex announced fiscal Q1 2027 before its 10-Q existed."""
    closes = [(date(2025, 8, 31), 1), (date(2025, 11, 30), 2), (date(2026, 5, 31), 4)]
    anchors = fetcher._anchors([(date(2026, 9, 30), 1)], closes, {})
    close, exact, filed = anchors[date(2026, 9, 30)]
    assert abs((close - date(2026, 8, 31)).days) <= 7
    assert exact is False, "a projection is not the company's statement"
    assert filed is None


def test_a_period_with_no_quarter_number_falls_back(fetcher):
    assert fetcher._anchors([(date(2026, 3, 31), None)], [(date(2026, 3, 31), 1)], {}) is None


# ---------------------------------------------------------------- decoys


def test_a_filing_days_after_the_close_is_not_the_release(fetcher):
    """Tesla files a delivery report two days into every quarter.

    Its earnings follow three weeks later. The delivery reports are perfectly
    regular, and the old cadence estimate locked onto them -- every Tesla
    surprise was dated about twenty days early.
    """
    recent = _recent(
        [("10-K", "2025-12-31", "2026-01-29"), ("10-Q", "2026-03-31", "2026-04-23")],
        ["2026-01-02", "2026-01-28", "2026-04-02", "2026-04-22"],
    )
    got = _date(fetcher, recent, [(date(2025, 12, 31), 4), (date(2026, 3, 31), 1)])
    assert got == {date(2025, 12, 31): date(2026, 1, 28), date(2026, 3, 31): date(2026, 4, 22)}


def test_the_last_filing_before_the_report_wins(fetcher):
    """Occidental pre-announces about ten days after every close.

    Its earnings come five weeks after, and the regular early stream beat the
    real release on cadence -- four weeks of lookahead on every quarter.
    """
    recent = _recent(
        [("10-K", "2025-12-31", "2026-02-18"), ("10-Q", "2026-03-31", "2026-05-05")],
        ["2026-01-20", "2026-02-18", "2026-04-10", "2026-05-05"],
    )
    got = _date(fetcher, recent, [(date(2025, 12, 31), 4), (date(2026, 3, 31), 1)])
    assert got == {date(2025, 12, 31): date(2026, 2, 18), date(2026, 3, 31): date(2026, 5, 5)}


def test_an_annual_release_later_than_the_quarterlies_is_kept(fetcher):
    """Target: Q4 at +31 with a pre-announcement at +11; other quarters +18.

    The cadence estimate preferred the decoy because it looked more typical.
    """
    recent = _recent([("10-K", "2026-01-31", "2026-03-11")], ["2026-02-11", "2026-03-03"])
    got = _date(fetcher, recent, [(date(2026, 3, 31), 4)])
    assert got == {date(2026, 3, 31): date(2026, 3, 3)}


def test_a_release_stamped_the_day_after_its_report_is_found(fetcher):
    """Grainger filed its 10-Q on May 7 and its release is dated May 8.

    EDGAR stamps after-hours filings with the next business day, so the bound
    carries a couple of days of slack.
    """
    recent = _recent(
        [("10-K", "2025-12-31", "2026-02-20"), ("10-Q", "2026-03-31", "2026-05-07")], ["2026-05-08"]
    )
    got = _date(fetcher, recent, [(date(2026, 3, 31), 1)])
    assert got == {date(2026, 3, 31): date(2026, 5, 8)}


def test_a_filing_after_the_report_is_not_the_release(fetcher):
    """Berkshire publishes results with its 10-Q, not in an 8-K.

    A 2.02 five days after the report is something else, and dating the quarter
    to it would only be a guess. Undated is the honest answer.
    """
    recent = _recent(
        [("10-K", "2025-12-31", "2026-02-20"), ("10-Q", "2026-03-31", "2026-05-02")], ["2026-05-07"]
    )
    anchors = fetcher._anchors(
        [(date(2026, 3, 31), 1)],
        fetcher._quarter_closes(recent),
        fetcher._report_filed(recent),
    )
    assert fetcher._match("X", [date(2026, 3, 31)], fetcher._parse_202(recent), anchors) == []


def test_only_exact_closes_are_recorded(fetcher):
    recent = _recent(
        [("10-K", "2025-12-31", "2026-02-20"), ("10-Q", "2026-03-31", "2026-05-01")], ["2026-04-28"]
    )
    exact = fetcher._anchors([(date(2026, 3, 31), 1)], fetcher._quarter_closes(recent), {})
    row = fetcher._match("X", [date(2026, 3, 31)], fetcher._parse_202(recent), exact)[0]
    assert row["fiscal_period_end"] == date(2026, 3, 31)

    projected = {date(2026, 3, 31): (date(2026, 3, 30), False, None)}
    row = fetcher._match("X", [date(2026, 3, 31)], fetcher._parse_202(recent), projected)[0]
    assert row["fiscal_period_end"] is None


# ------------------------------------------------- fiscal-year changes and spin-offs


def test_a_transition_report_closes_the_new_fiscal_year(fetcher):
    """Ferguson moved its year end from July to December in 2025.

    The change is marked by a 10-KT. Ignoring it numbered the March 2026
    quarter from the July year end -- Q3 rather than Q1 -- so nothing anchored,
    the symbol fell back to label matching, and two quarters were dated with
    the previous quarter's release, ten to eleven weeks early.
    """
    recent = _recent(
        [
            ("10-Q", "2025-04-30", "2025-06-03"),
            ("10-K", "2025-07-31", "2025-09-26"),
            ("10-Q", "2025-10-31", "2025-12-09"),
            ("10-KT", "2025-12-31", "2026-02-27"),
            ("10-Q", "2026-03-31", "2026-05-05"),
            ("10-Q", "2026-06-30", "2026-08-10"),
        ],
        ["2025-06-03", "2025-09-16", "2025-12-09", "2026-02-24", "2026-05-05", "2026-08-10"],
    )
    rows = [
        (date(2025, 6, 30), 3),
        (date(2025, 12, 31), 4),
        (date(2026, 3, 31), 1),
        (date(2026, 6, 30), 2),
    ]
    assert _date(fetcher, recent, rows) == {
        date(2025, 6, 30): date(2025, 6, 3),
        date(2025, 12, 31): date(2026, 2, 24),
        date(2026, 3, 31): date(2026, 5, 5),
        date(2026, 6, 30): date(2026, 8, 10),
    }


def test_a_spin_off_before_its_first_10k_still_anchors(fetcher):
    """Honeywell Aerospace has one 10-Q and no 10-K yet.

    Its quarter cannot be numbered, but it is the only close near the label,
    so it still anchors. Without that it fell back to label matching and took
    an Item 2.02 filed two days after the close -- a month before the release.
    """
    recent = _recent([("10-Q", "2026-06-27", "2026-08-05")], ["2026-06-29", "2026-08-05"])
    assert _date(fetcher, recent, [(date(2026, 6, 30), 2)]) == {date(2026, 6, 30): date(2026, 8, 5)}


def test_two_unnumbered_closes_near_one_label_are_not_guessed_between(fetcher):
    closes = [(date(2026, 6, 1), None), (date(2026, 6, 27), None)]
    assert fetcher._anchors([(date(2026, 6, 30), 2)], closes, {}) is None
