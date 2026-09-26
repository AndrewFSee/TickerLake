"""Tests for keeping the earnings datasets to real companies and real quarters.

Two faults, both found in the lake:

* **Funds asked about earnings.** Every per-symbol rotation spanned the 60
  tracked ETFs. Finnhub returned nothing for 59 and, for VXX -- a VIX-futures
  ETN -- four quarters of earnings belonging to nothing. The insider feed
  returned 21 "insider trades" in USO that were Hudson River Trading's
  market-making inventory, filed because it crossed 10% of the fund.

* **Quarters that change their label.** Finnhub revises ``period``: Paychex's
  fiscal Q1 2027 arrived as ``2027-03-31`` and was re-issued as ``2026-09-30``.
  Keyed on period, the corrected row lands beside the stale one. And Finnhub
  withdraws quarters outright -- Amcor's fiscal Q4 2026 vanished from its answer.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pyarrow.parquet as pq
import pytest

from tickerlake.config import Config, Secrets
from tickerlake.fetchers.base import FetchResult
from tickerlake.fetchers.earnings import EarningsFetcher
from tickerlake.storage import paths as P
from tickerlake.storage.paths import DatasetPaths
from tickerlake.storage.writer import ParquetWriter
from tickerlake.universe.membership import MembershipTracker

RUN = date(2026, 9, 25)


@pytest.fixture
def lake(tmp_path):
    paths = DatasetPaths(tmp_path)
    paths.ensure_layout()
    return paths, ParquetWriter()


def _config(root, **earnings):
    return Config(
        raw={"earnings": {"enabled": True, **earnings}},
        secrets=Secrets(finnhub_api_key="x"),
        data_root=root,
        config_path=root / "config.yaml",
    )


# -------------------------------------------------------------- companies only


@pytest.fixture
def tracker(lake):
    """Two constituents -- one since removed -- and two ETFs."""
    paths, writer = lake
    t = MembershipTracker(paths, writer, collect_indices=["SP500", "ETF"])
    t.seed(
        pd.DataFrame(
            [
                {"symbol": "AAPL", "start_date": date(2010, 1, 1), "end_date": None},
                {"symbol": "GAS", "start_date": date(2010, 1, 1), "end_date": date(2016, 7, 1)},
            ],
            columns=["symbol", "start_date", "end_date"],
        ),
        force=True,
    )
    t.register_static({"VXX": "iPath VIX futures ETN", "USO": "US Oil Fund"}, "ETF")
    return t


def test_collection_spans_funds_but_company_rotations_do_not(lake, tracker):
    paths, writer = lake
    f = EarningsFetcher(_config(paths.root), paths, writer, tracker=tracker)

    assert set(tracker.current_members()) == {"AAPL", "VXX", "USO"}, "prices want funds"
    assert f.company_members() == ["AAPL"], "earnings do not"


def test_the_rotation_never_reaches_a_fund(lake, tracker):
    paths, writer = lake
    f = EarningsFetcher(_config(paths.root), paths, writer, tracker=tracker)
    seen = set()
    for day in range(40):
        seen.update(f._slice(date.fromordinal(RUN.toordinal() + day), 1))
    assert seen == {"AAPL"}


def test_constituents_include_removed_names_but_never_funds(tracker):
    """A delisted company still has real filings; a fund never did."""
    assert tracker.constituent_symbols() == {"AAPL", "GAS"}


def test_an_explicit_symbol_list_still_wins(lake, tracker):
    """--symbols is a deliberate request and is honoured as given."""
    paths, writer = lake
    f = EarningsFetcher(
        _config(paths.root), paths, writer, tracker=tracker, symbols_override=["VXX"]
    )
    assert f.company_members() == ["VXX"]


# -------------------------------------------------------------- reconciliation


def _stored(symbol, period, fy, fq, eps):
    return {
        "symbol": symbol,
        "record_type": "surprise",
        "period": period,
        "fiscal_year": fy,
        "fiscal_quarter": fq,
        "eps_actual": eps,
        "source": "finnhub",
        "ingested_at": datetime(2026, 9, 17, tzinfo=UTC),
    }


def _finnhub(*quarters):
    """Payload shaped like /stock/earnings: (period, year, quarter, actual)."""
    return [
        {"period": p, "year": y, "quarter": q, "actual": a, "estimate": a}
        for p, y, q, a in quarters
    ]


class _Client:
    def __init__(self, answers):
        self.answers = answers

    def get_json(self, url, params=None):
        answer = self.answers[params["symbol"]]
        if isinstance(answer, Exception):
            raise answer
        return answer


def _run(lake, stored_rows, answers):
    paths, writer = lake
    path = paths.earnings_file("surprises")
    writer.write(pd.DataFrame(stored_rows), P.EARNINGS, path, mode="overwrite")
    f = EarningsFetcher(_config(paths.root), paths, writer, symbols_override=list(answers))
    result = FetchResult(stage="earnings", dataset=P.EARNINGS)
    f._surprises(_Client(answers), "x", RUN, result)
    out = pq.read_table(path).to_pandas()
    out["period"] = pd.to_datetime(out["period"]).dt.date
    return out, result


def test_a_relabelled_quarter_is_stored_once(lake):
    """Paychex, verbatim: same quarter, same EPS, new label."""
    out, result = _run(
        lake,
        [
            _stored("PAYX", date(2026, 6, 30), 2026, 4, 1.32),
            _stored("PAYX", date(2027, 3, 31), 2027, 1, 1.34),
        ],
        {
            "PAYX": _finnhub(
                ("2026-09-30", 2027, 1, 1.34),
                ("2026-06-30", 2026, 4, 1.32),
            )
        },
    )
    q1 = out[(out["fiscal_year"] == 2027) & (out["fiscal_quarter"] == 1)]
    assert len(q1) == 1, "one quarter, one row"
    assert q1.iloc[0]["period"] == date(2026, 9, 30), "the corrected label survives"
    assert result.details["surprises_relabelled"] == 1


def test_a_withdrawn_quarter_is_removed(lake):
    """Amcor: the lake held fiscal Q4 2026; Finnhub now tops out at Q3.

    The endpoint returns the latest quarters, so a stored quarter later than
    all of them cannot have aged out -- it was retracted.
    """
    out, result = _run(
        lake,
        [
            _stored("AMCR", date(2026, 3, 31), 2026, 3, 0.96),
            _stored("AMCR", date(2026, 12, 31), 2026, 4, 1.23),
        ],
        {
            "AMCR": _finnhub(
                ("2026-03-31", 2026, 3, 0.96),
                ("2025-12-31", 2026, 2, 0.86),
                ("2025-09-30", 2026, 1, 0.965),
            )
        },
    )
    assert not ((out["fiscal_year"] == 2026) & (out["fiscal_quarter"] == 4)).any()
    assert result.details["surprises_withdrawn"] == 1


def test_older_quarters_outside_the_window_are_kept(lake):
    """Finnhub's free tier returns four quarters; older history is not a retraction."""
    out, _ = _run(
        lake,
        [_stored("AAPL", date(2024, 6, 30), 2024, 3, 1.40)],
        {"AAPL": _finnhub(("2026-06-30", 2026, 3, 1.57), ("2026-03-31", 2026, 2, 1.65))},
    )
    assert (out["period"] == date(2024, 6, 30)).any()


def test_a_one_row_answer_is_too_thin_to_infer_a_withdrawal(lake):
    out, result = _run(
        lake,
        [_stored("AMCR", date(2026, 12, 31), 2026, 4, 1.23)],
        {"AMCR": _finnhub(("2026-03-31", 2026, 3, 0.96))},
    )
    assert (out["period"] == date(2026, 12, 31)).any()
    assert "surprises_withdrawn" not in result.details


def test_a_failed_request_reconciles_nothing(lake):
    """No answer says nothing about which quarters exist."""
    out, _ = _run(
        lake,
        [_stored("PAYX", date(2027, 3, 31), 2027, 1, 1.34)],
        {"PAYX": RuntimeError("rate limited")},
    )
    assert (out["period"] == date(2027, 3, 31)).any()


def test_other_symbols_are_untouched(lake):
    out, _ = _run(
        lake,
        [
            _stored("MSFT", date(2027, 3, 31), 2027, 1, 3.0),  # odd, but not ours to judge
            _stored("PAYX", date(2027, 3, 31), 2027, 1, 1.34),
        ],
        {"PAYX": _finnhub(("2026-09-30", 2027, 1, 1.34), ("2026-06-30", 2026, 4, 1.32))},
    )
    assert ((out["symbol"] == "MSFT") & (out["period"] == date(2027, 3, 31))).any()


def test_an_unchanged_label_is_left_alone(lake):
    out, result = _run(
        lake,
        [_stored("AAPL", date(2026, 6, 30), 2026, 3, 1.57)],
        {"AAPL": _finnhub(("2026-06-30", 2026, 3, 1.57), ("2026-03-31", 2026, 2, 1.65))},
    )
    assert len(out[out["symbol"] == "AAPL"]) == 2
    assert "surprises_relabelled" not in result.details
