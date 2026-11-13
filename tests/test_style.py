"""Value / growth scores: quarterly flows from filings, point-in-time use, split basis, scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tickerlake.analytics import style


def fact(concept, value, start, end, filed, unit="USD", symbol="AAA"):
    return {"symbol": symbol, "concept": concept, "unit": unit, "value": value,
            "start_date": start, "end_date": end, "filed_date": filed}


def test_quarters_difference_year_to_date_facts():
    f = style.first_reported(pd.DataFrame([
        fact("NetIncomeLoss", 10, "2024-01-01", "2024-03-31", "2024-05-01"),
        fact("NetIncomeLoss", 25, "2024-01-01", "2024-06-30", "2024-08-01"),   # 6-month YTD
        fact("NetIncomeLoss", 45, "2024-01-01", "2024-09-30", "2024-11-01"),   # 9-month YTD
        fact("NetIncomeLoss", 70, "2024-01-01", "2024-12-31", "2025-02-15"),   # 10-K
    ]))
    q = style.quarters(f)
    assert q["value"].tolist() == [10, 15, 20, 25]
    assert q["known"].iloc[-1] == pd.Timestamp("2025-02-15")


def test_first_reported_value_wins_over_restatement():
    f = style.first_reported(pd.DataFrame([
        fact("NetIncomeLoss", 10, "2024-01-01", "2024-03-31", "2024-05-01"),
        fact("NetIncomeLoss", 99, "2024-01-01", "2024-03-31", "2025-05-01"),   # restated a year later
    ]))
    assert style.quarters(f)["value"].tolist() == [10]


def test_share_counts_are_not_differenced():
    f = style.first_reported(pd.DataFrame([
        fact("WeightedAverageNumberOfDilutedSharesOutstanding", 100, "2024-04-01", "2024-06-30", "2024-08-01", "shares"),
        fact("WeightedAverageNumberOfDilutedSharesOutstanding", 98, "2024-01-01", "2024-06-30", "2024-08-01", "shares"),
    ]))
    a = style.period_average(f)
    assert a.loc[pd.Timestamp("2024-06-30"), "value"] == 100   # the 3-month average, not 100 - 98


def test_growth_slope_reads_like_an_annual_rate():
    s = pd.Series(np.arange(1.0, 13.0))             # +1 per quarter = +4 per year, mean level 6.5
    assert style.growth_slope(s).iloc[-1] == pytest.approx(4 / 6.5)
    assert np.isnan(style.growth_slope(s.where(s != 12)).iloc[-1])   # latest missing: no reading


def test_split_factor_counts_only_later_splits():
    splits = pd.Series([2.0, 3.0], index=pd.to_datetime(["2020-06-01", "2024-06-01"]))
    known = pd.Series(pd.to_datetime(["2019-01-01", "2021-01-01", "2025-01-01"]))
    assert style.split_factor(known, splits).tolist() == [6.0, 3.0, 1.0]


def _quarterly_facts(symbol, filed_lag_days=30, shares=100.0):
    rows = []
    for end in pd.date_range("2021-03-31", "2024-12-31", freq="QE"):
        start = end - pd.offsets.QuarterBegin(startingMonth=1)
        filed = (end + pd.Timedelta(days=filed_lag_days)).strftime("%Y-%m-%d")
        s, e = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        rows += [fact("NetIncomeLoss", 10.0, s, e, filed, symbol=symbol),
                 fact("Revenues", 100.0, s, e, filed, symbol=symbol),
                 fact("PaymentsToAcquirePropertyPlantAndEquipment", 5.0, s, e, filed, symbol=symbol),
                 fact("WeightedAverageNumberOfDilutedSharesOutstanding", shares, s, e, filed, "shares", symbol),
                 fact("StockholdersEquity", 400.0, e, e, filed, symbol=symbol),
                 fact("Assets", 1000.0, e, e, filed, symbol=symbol)]
    return pd.DataFrame(rows)


def test_a_filing_counts_from_the_day_after_it_is_filed():
    facts = _quarterly_facts("AAA")
    fund = style.fundamentals(facts, pd.DataFrame(columns=["symbol", "date", "stock_splits"]))
    filed = pd.Timestamp("2024-12-31") + pd.Timedelta(days=30)
    days = pd.bdate_range("2024-12-01", "2025-03-31")
    close = pd.DataFrame({"AAA": 40.0}, index=days)
    members = pd.DataFrame({"symbol": ["AAA"], "start_date": [pd.Timestamp("2010-01-01")],
                            "end_date": [pd.NaT], "gics_sector": ["Energy"]})
    panel = style.factor_panel(fund, close, close * 0, members, [filed, filed + pd.Timedelta(days=1)])
    assert panel["date"].tolist() == [filed, filed + pd.Timedelta(days=1)]
    # both see 4 quarters of 10 earnings on 100 shares at 40: E/P = 40 / 4000
    assert panel["ep"].tolist() == pytest.approx([0.01, 0.01])
    fund_q = fund.set_index("qend")
    assert fund_q.loc[pd.Timestamp("2024-12-31"), "known"] == filed
    last_seen_before = fund[fund["known"] < filed]["qend"].max()
    assert last_seen_before == pd.Timestamp("2024-09-30")


def test_shares_filed_before_a_split_move_to_todays_basis():
    facts = _quarterly_facts("AAA")
    splits = pd.DataFrame({"symbol": ["AAA"], "date": ["2025-06-02"], "stock_splits": [2.0]})
    fund = style.fundamentals(facts, splits)
    assert fund["shares"].iloc[-1] == pytest.approx(200.0)


def test_scores_are_centred_on_the_cap_weighted_index():
    rng = np.random.default_rng(0)
    n = 300
    day = pd.DataFrame({"symbol": [f"S{i}" for i in range(n)], "mcap": rng.lognormal(10, 1, n)})
    for f in style.FACTORS:
        day[f] = rng.normal(size=n)
    out = style.score(day)
    assert np.average(out["value"], weights=out["mcap"]) == pytest.approx(0, abs=1e-9)
    assert np.average(out["growth"], weights=out["mcap"]) == pytest.approx(0, abs=1e-9)
    assert out["value"].std() == pytest.approx(1)


def test_one_factor_is_not_enough_for_a_score():
    day = pd.DataFrame({"symbol": list("ABCDE"), "mcap": [1.0] * 5, **{f: [1.0, 2, 3, 4, 5] for f in style.FACTORS}})
    day.loc[0, ["ep", "sp", "dp"]] = np.nan
    assert np.isnan(style.score(day).loc[0, "value"])


def test_quadrants():
    assert style.quadrant(0.5, 0.5) == "cheap, growing"
    assert style.quadrant(-0.5, -0.1) == "pricey, slow"
    assert style.quadrant(np.nan, 1) is None


def test_backtest_finds_a_planted_signal():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2015-01-01", "2022-12-31")
    month_ends = list(pd.Series(dates, index=dates).groupby(dates.to_period("M")).max())
    syms = [f"S{i}" for i in range(120)]
    rows = []
    level = pd.DataFrame(100.0, index=month_ends, columns=syms)
    for i, d in enumerate(month_ends):
        v = rng.normal(size=len(syms))
        rows.append(pd.DataFrame({"date": d, "symbol": syms, "mcap": 1.0, "sector": [f"X{j % 10}" for j in range(120)],
                                  "value": v, "growth": rng.normal(size=len(syms)),
                                  **{f"z_{f}": rng.normal(size=len(syms)) for f in style.FACTORS}}))
        if i + 1 < len(month_ends):
            level.iloc[i + 1] = level.iloc[i] * (1 + 0.02 * v + 0.01 * rng.normal(size=len(syms)))
    res = style.backtest(pd.concat(rows, ignore_index=True), level)
    assert res["stock"]["value"]["1m"]["mean"] > 0.5 and res["stock"]["value"]["1m"]["t"] > 5
    assert abs(res["stock"]["growth"]["1m"]["t"]) < 3
    assert res["months"] == len(month_ends)


def test_a_quarter_waits_for_its_latest_filed_input():
    """Assets for the last quarter first appear a year later (a comparative): the quarter can't be used before."""
    facts = _quarterly_facts("AAA")
    late = (facts["concept"] == "Assets") & (facts["end_date"] == "2024-12-31")
    facts.loc[late, "filed_date"] = "2026-01-30"
    fund = style.fundamentals(facts, pd.DataFrame(columns=["symbol", "date", "stock_splits"]))
    assert fund.set_index("qend").loc[pd.Timestamp("2024-12-31"), "known"] == pd.Timestamp("2026-01-30")
