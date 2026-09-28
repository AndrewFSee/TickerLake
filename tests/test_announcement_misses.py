"""Tests for reporting undatable quarters once rather than every night.

Some quarters cannot be dated from Item 2.02 at all: Exxon, AES and ONEOK
publish earnings under Item 7.01, and Berkshire releases with its 10-Q. The
stage used to warn "no announcement dates could be matched" on every run in
which only those were pending -- a warning that fired when nothing new had gone
wrong, which is how a real failure ends up ignored.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pyarrow.parquet as pq
import pytest

import tickerlake.fetchers.announcements as mod
from tickerlake.config import Config, Secrets
from tickerlake.fetchers.announcements import AnnouncementFetcher
from tickerlake.storage import paths as P
from tickerlake.storage.paths import DatasetPaths
from tickerlake.storage.writer import ParquetWriter

RUN = date(2026, 9, 28)
CIKS = {"AAA": 1, "XOM": 2, "OLD": 3, "BAD": 4}


def _recent(periodic, eightk=()):
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
        "filings": {
            "recent": {
                "form": forms,
                "reportDate": reports,
                "filingDate": filed,
                "items": items,
                "accessionNumber": acc,
            }
        }
    }


YEAR = [("10-K", "2025-12-31", "2026-02-20"), ("10-Q", "2026-03-31", "2026-05-01")]
SUBMISSIONS = {
    # Dates cleanly: an earnings 8-K a month after the June close.
    "AAA": _recent([*YEAR, ("10-Q", "2026-06-30", "2026-08-05")], ["2026-07-28"]),
    # Publishes earnings under Item 7.01, so there is no 2.02 to find.
    "XOM": _recent([*YEAR, ("10-Q", "2026-06-30", "2026-08-04")]),
    # Undatable too, but its quarter closed long ago.
    "OLD": _recent(YEAR),
}


def _fake_client(submissions):
    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def get_json(self, url, params=None):
            if url == mod.TICKER_MAP_URL:
                return {
                    str(n): {"ticker": s, "cik_str": c} for n, (s, c) in enumerate(CIKS.items())
                }
            cik = int(url.split("CIK")[1].split(".")[0])
            symbol = next(s for s, c in CIKS.items() if c == cik)
            if symbol == "BAD":
                raise RuntimeError("connection reset")
            return submissions[symbol]

        def close(self):
            pass

    return FakeClient


def _row(symbol, period, fq):
    return {
        "symbol": symbol,
        "record_type": "surprise",
        "period": period,
        "fiscal_year": period.year,
        "fiscal_quarter": fq,
        "eps_actual": 1.0,
        "announcement_date": None,
        "source": "finnhub",
        "ingested_at": datetime(2026, 9, 20, tzinfo=UTC),
    }


@pytest.fixture
def lake(tmp_path):
    paths = DatasetPaths(tmp_path)
    paths.ensure_layout()
    rows = [
        _row("AAA", date(2026, 6, 30), 2),
        _row("XOM", date(2026, 6, 30), 2),
        _row("OLD", date(2025, 12, 31), 4),
        _row("BAD", date(2026, 6, 30), 2),
    ]
    ParquetWriter().write(
        pd.DataFrame(rows), P.EARNINGS, paths.earnings_file("surprises"), mode="overwrite"
    )
    return paths


def _run(paths, run_date, monkeypatch, submissions=SUBMISSIONS):
    monkeypatch.setattr(mod, "HttpClient", _fake_client(submissions))
    config = Config(
        raw={"announcements": {"enabled": True}},
        secrets=Secrets(sec_user_agent="TickerLake test test@example.com"),
        data_root=paths.root,
        config_path=paths.root / "config.yaml",
    )
    result = AnnouncementFetcher(config, paths, ParquetWriter()).run(run_date)
    stored = pq.read_table(paths.earnings_file("surprises")).to_pandas().set_index("symbol")
    return result, stored


def _checked(stored, symbol):
    value = stored.loc[symbol, "announcement_checked"]
    return None if pd.isna(value) else pd.Timestamp(value).date()


def test_a_new_undatable_quarter_warns_once(lake, monkeypatch):
    first, stored = _run(lake, RUN, monkeypatch)
    assert first.ok
    assert any("XOM 2026-06-30" in w for w in first.warnings), first.warnings
    assert first.details["undated_new"] == 1
    assert _checked(stored, "XOM") == RUN

    second, stored = _run(lake, date(2026, 9, 29), monkeypatch)
    assert second.ok
    assert not second.warnings, "already reported; silence the second time"
    assert second.details["undated_new"] == 0
    assert _checked(stored, "XOM") == date(2026, 9, 29), "still stamped as seen"


def test_a_quarter_whose_window_closed_long_ago_never_warns(lake, monkeypatch):
    result, stored = _run(lake, RUN, monkeypatch)
    assert not any("OLD" in w for w in result.warnings)
    assert _checked(stored, "OLD") == RUN, "recorded, but as history rather than news"


def test_the_warning_that_used_to_fire_every_night_is_gone(lake, monkeypatch):
    _run(lake, RUN, monkeypatch)
    result, _ = _run(lake, date(2026, 9, 29), monkeypatch)
    assert not any("no announcement dates could be matched" in w for w in result.warnings)


def test_a_failed_lookup_is_not_recorded_as_a_miss(lake, monkeypatch):
    """A lookup that failed says nothing about whether a release exists."""
    result, stored = _run(lake, RUN, monkeypatch)
    assert result.items_failed == 1
    assert _checked(stored, "BAD") is None
    assert not any("BAD" in w for w in result.warnings)


def test_a_datable_quarter_is_dated_not_stamped(lake, monkeypatch):
    _, stored = _run(lake, RUN, monkeypatch)
    assert pd.Timestamp(stored.loc["AAA", "announcement_date"]).date() == date(2026, 7, 28)
    assert _checked(stored, "AAA") is None


def test_a_quarter_that_becomes_datable_is_cleared(lake, monkeypatch):
    _, stored = _run(lake, RUN, monkeypatch)
    assert _checked(stored, "XOM") == RUN

    later = dict(SUBMISSIONS)
    later["XOM"] = _recent([*YEAR, ("10-Q", "2026-06-30", "2026-08-04")], ["2026-07-31"])
    _, stored = _run(lake, date(2026, 9, 29), monkeypatch, later)
    assert pd.Timestamp(stored.loc["XOM", "announcement_date"]).date() == date(2026, 7, 31)
    assert _checked(stored, "XOM") is None, "no longer a miss"


def test_stamping_a_miss_keeps_the_rest_of_the_row(lake, monkeypatch):
    """The stamp is written back through a merge; the EPS must survive it."""
    _, stored = _run(lake, RUN, monkeypatch)
    assert stored.loc["XOM", "eps_actual"] == pytest.approx(1.0)
    assert len(stored) == 4, "no rows added or lost"
