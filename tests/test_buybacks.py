"""Buybacks from filings and estimated blackout windows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tickerlake.analytics import buybacks as B


def fact(concept, value, start, end, filed, unit="USD", symbol="AAA"):
    return {"symbol": symbol, "concept": concept, "unit": unit, "value": value,
            "start_date": start, "end_date": end, "filed_date": filed}


def _facts():
    rows = []
    ytd = 0.0
    for i, end in enumerate(pd.date_range("2025-03-31", "2025-12-31", freq="QE")):
        ytd += 10.0 * (i + 1)                                    # 10, 20, 30, 40 per quarter, reported year-to-date
        filed = (end + pd.Timedelta(days=35)).strftime("%Y-%m-%d")
        rows.append(fact("PaymentsForRepurchaseOfCommonStock", ytd, "2025-01-01", end.strftime("%Y-%m-%d"), filed))
        rows.append(fact("StockRepurchaseProgramRemainingAuthorizedRepurchaseAmount1", [500, 480, 900, 860][i],
                         None, end.strftime("%Y-%m-%d"), filed))
    for end, shares in [("2024-12-31", 100.0), ("2025-12-31", 50.0)]:   # a 2-for-1 split in between
        start = (pd.Timestamp(end) - pd.Timedelta(days=91)).strftime("%Y-%m-%d")
        rows.append(fact("WeightedAverageNumberOfDilutedSharesOutstanding", shares, start, end,
                         (pd.Timestamp(end) + pd.Timedelta(days=35)).strftime("%Y-%m-%d"), "shares"))
    return pd.DataFrame(rows)


def test_repurchases_from_year_to_date_cash_flow():
    splits = pd.DataFrame({"symbol": ["AAA"], "date": ["2025-06-02"], "stock_splits": [0.5]})  # reverse split 1:2
    t = B.repurchase_table(_facts(), splits).set_index("symbol").loc["AAA"]
    assert t["repurchased_q"] == 40 and t["repurchased_ttm"] == 100
    assert t["remaining_auth"] == 860
    assert t["auth_added_reported"] == pd.Timestamp("2025-11-04")     # 480 -> 900, reported with Q3
    # 100 shares before a 1-for-2 reverse split = 50 on today's basis, so the count is unchanged
    assert t["share_change_1y"] == pytest.approx(0.0)


def test_window_shuts_before_the_close_and_reopens_after_the_release():
    rel = pd.DataFrame({"symbol": ["AAA"], "fiscal_close": [pd.Timestamp("2026-09-30")],
                        "release_date": [pd.Timestamp("2026-10-22")], "source": ["8-K"]})   # a Thursday
    w = B.blackout_windows(rel).iloc[0]
    assert w["shut"] == pd.Timestamp("2026-09-16") and w["reopen"] == pd.Timestamp("2026-10-26")
    s = B.status_on(B.blackout_windows(rel), pd.Timestamp("2026-10-01")).iloc[0]
    assert s["blackout"] and s["reopens"] == pd.Timestamp("2026-10-26")
    assert s["window_release"] == pd.Timestamp("2026-10-22") and s["window_source"] == "8-K"
    s = B.status_on(B.blackout_windows(rel), pd.Timestamp("2026-10-27")).iloc[0]
    assert not s["blackout"] and pd.isna(s["next_shut"])


def test_next_quarters_use_the_calendar_else_the_usual_lag():
    closes = pd.date_range("2025-03-31", "2026-06-30", freq="QE")
    rel = pd.DataFrame({"symbol": "AAA", "fiscal_close": closes, "release_date": closes + pd.Timedelta(days=25),
                        "report_filed": closes + pd.Timedelta(days=40)})
    cal = pd.DataFrame({"symbol": ["AAA"], "date": ["2026-10-20"]})
    out = B.expected_releases(rel, cal, pd.Timestamp("2026-08-15"), horizon_days=150)
    nxt = out[out["fiscal_close"] > closes[-1]].reset_index(drop=True)
    assert nxt.loc[0, "release_date"] == pd.Timestamp("2026-10-20") and nxt.loc[0, "source"] == "calendar"
    assert nxt.loc[1, "source"] == "estimated"
    assert (nxt.loc[1, "release_date"] - nxt.loc[1, "fiscal_close"]).days == 25


def test_missing_release_falls_back_to_the_filing_date():
    rel = pd.DataFrame({"symbol": ["AAA"], "fiscal_close": [pd.Timestamp("2026-03-31")],
                        "release_date": [pd.NaT], "report_filed": [pd.Timestamp("2026-05-05")]})
    out = B.expected_releases(rel, pd.DataFrame(columns=["symbol", "date"]), pd.Timestamp("2026-05-10"), 0)
    assert out.loc[0, "release_date"] == pd.Timestamp("2026-05-05") and out.loc[0, "source"] == "10-Q/10-K filed"


def test_blackout_share_counts_members_only():
    days = pd.bdate_range("2026-09-01", "2026-10-30")
    rel = pd.DataFrame({"symbol": ["AAA", "BBB"], "fiscal_close": [pd.Timestamp("2026-09-30")] * 2,
                        "release_date": [pd.Timestamp("2026-10-22"), pd.Timestamp("2026-10-08")], "source": "8-K"})
    shut = B.blackout_matrix(B.blackout_windows(rel), days)
    member = pd.DataFrame(True, index=days, columns=shut.columns)
    share = B.blackout_share(shut, member)
    assert share[pd.Timestamp("2026-10-15")] == 0.5 and share[pd.Timestamp("2026-10-01")] == 1.0
    w = B.blackout_share(shut, member, pd.Series({"AAA": 3.0, "BBB": 1.0}))
    assert w[pd.Timestamp("2026-10-15")] == pytest.approx(0.75)
    member["BBB"] = False
    assert B.blackout_share(shut, member)[pd.Timestamp("2026-10-15")] == 1.0
    assert np.isnan(B.blackout_share(shut, member & False)).all()
