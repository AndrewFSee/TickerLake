"""Tests for Fama-French, Treasury curve, CFTC COT, and insider parsing.

Each source has a format quirk that silently produces wrong numbers rather than
an error, so those are what these pin down.
"""

from __future__ import annotations

import csv
import io
from datetime import date

import pytest

from tickerlake.fetchers.reference import TENORS, _as_date, _as_float

# ------------------------------------------------------------------ parsing


def test_as_float_handles_publisher_null_markers():
    assert _as_float("3.72") == pytest.approx(3.72)
    assert _as_float("1,234.5") == pytest.approx(1234.5)
    assert _as_float("") is None
    assert _as_float("N/A") is None
    assert _as_float(".") is None, "Treasury and FRED both use a bare dot for missing"
    assert _as_float(None) is None


def test_as_date_accepts_iso_and_date():
    assert _as_date("1990-01-01") == date(1990, 1, 1)
    assert _as_date(date(2020, 5, 4)) == date(2020, 5, 4)


# ------------------------------------------------------------ fama-french


def _ff_fixture() -> str:
    """A file shaped exactly like French's: prose header, data, copyright tail."""
    return "\n".join(
        [
            "This file was created by using the 202607 CRSP database.",
            "The Tbill return is the simple daily rate that, over the trading days",
            "",
            ",Mkt-RF,SMB,HML,RMW,CMA,RF",
            "19630701,   -0.67,    0.00,   -0.33,   -0.01,    0.16,    0.01",
            "20260731,    0.68,   -0.38,   -0.59,    1.09,   -2.90,    0.02",
            "20260801,  -99.99,  -99.99,  -99.99,  -99.99,  -99.99,   -99.99",
            "",
            "Copyright 2026 Eugene F. Fama and Kenneth R. French",
        ]
    )


def _parse_ff(raw: str, start: date) -> list[dict]:
    """Mirror of FamaFrenchFetcher._fetch_set's parsing, minus the download."""
    from tickerlake.fetchers.reference import _FF_ROW

    header: list[str] = []
    out: list[dict] = []
    for line in raw.splitlines():
        match = _FF_ROW.match(line)
        if match is None:
            if line.startswith(",") and not header:
                header = [c.strip() for c in line.split(",")[1:]]
            continue
        if not header:
            continue
        stamp = match.group(1)
        day = date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]))
        if day < start:
            continue
        for name, value in zip(header, match.group(2).split(","), strict=False):
            parsed = _as_float(value)
            if parsed is None or parsed <= -99:
                continue
            out.append({"date": day, "factor": name, "value": parsed})
    return out


def test_fama_french_header_and_prose_are_not_parsed_as_data():
    rows = _parse_ff(_ff_fixture(), date(1900, 1, 1))
    assert {r["date"] for r in rows} == {date(1963, 7, 1), date(2026, 7, 31)}
    assert {r["factor"] for r in rows} == {"Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"}


def test_fama_french_missing_sentinel_is_dropped():
    """-99.99 is French's missing marker; storing it as a return poisons means."""
    rows = _parse_ff(_ff_fixture(), date(1900, 1, 1))
    assert all(r["value"] > -99 for r in rows)
    assert not any(r["date"] == date(2026, 8, 1) for r in rows)


def test_fama_french_start_date_filter():
    rows = _parse_ff(_ff_fixture(), date(2000, 1, 1))
    assert {r["date"] for r in rows} == {date(2026, 7, 31)}


def test_fama_french_values_are_signed_correctly():
    rows = _parse_ff(_ff_fixture(), date(1900, 1, 1))
    by_key = {(r["date"], r["factor"]): r["value"] for r in rows}
    assert by_key[(date(1963, 7, 1), "Mkt-RF")] == pytest.approx(-0.67)
    assert by_key[(date(2026, 7, 31), "CMA")] == pytest.approx(-2.90)


# --------------------------------------------------------------- treasury


def test_tenor_map_is_ordered_and_spans_the_curve():
    years = list(TENORS.values())
    assert years == sorted(years), "tenors must ascend for interpolation to work"
    assert min(years) == pytest.approx(1 / 12)
    assert max(years) == 30.0
    assert len(TENORS) == 14


def test_tenor_years_are_correct_fractions():
    assert TENORS["BC_3MONTH"] == pytest.approx(0.25)
    assert TENORS["BC_6MONTH"] == pytest.approx(0.5)
    assert TENORS["BC_1_5MONTH"] == pytest.approx(0.125)
    assert TENORS["BC_10YEAR"] == 10.0


def test_treasury_xml_entry_parsing():
    from tickerlake.fetchers.reference import _ENTRY, _FIELD

    xml = (
        "<feed><entry><content><m:properties>"
        "<d:NEW_DATE m:type='Edm.DateTime'>2026-01-02T00:00:00</d:NEW_DATE>"
        "<d:BC_1MONTH m:type='Edm.Double'>3.72</d:BC_1MONTH>"
        "<d:BC_10YEAR m:type='Edm.Double'>4.19</d:BC_10YEAR>"
        "<d:BC_30YEAR m:type='Edm.Double'></d:BC_30YEAR>"
        "</m:properties></content></entry></feed>"
    )
    blocks = _ENTRY.findall(xml)
    assert len(blocks) == 1
    fields = dict(_FIELD.findall(blocks[0]))
    assert fields["NEW_DATE"][:10] == "2026-01-02"
    assert _as_float(fields["BC_1MONTH"]) == pytest.approx(3.72)
    assert _as_float(fields["BC_30YEAR"]) is None, "an empty tenor must be null, not zero"


# -------------------------------------------------------------------- cot


def test_cot_market_and_exchange_split():
    """The CFTC packs both into one field separated by ' - '."""
    market = "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE"
    name, _, exchange = market.partition(" - ")
    assert name.strip() == "E-MINI S&P 500"
    assert exchange.strip() == "CHICAGO MERCANTILE EXCHANGE"


def test_cot_net_positioning_arithmetic():
    row = {"asset_mgr_long": 1_000_000.0, "asset_mgr_short": 87_639.0}
    net = row["asset_mgr_long"] - row["asset_mgr_short"]
    assert net == pytest.approx(912_361.0)


def test_cot_keyword_filter_selects_equity_markets():
    keywords = ["S&P", "NASDAQ", "E-MINI", "VIX"]
    markets = [
        "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE",
        "WHEAT-SRW - CHICAGO BOARD OF TRADE",
        "VIX FUTURES - CBOE FUTURES EXCHANGE",
        "LEAN HOGS - CHICAGO MERCANTILE EXCHANGE",
    ]
    kept = [m for m in markets if any(k in m.upper() for k in keywords)]
    assert len(kept) == 2
    assert all("WHEAT" not in m and "HOGS" not in m for m in kept)


def test_cot_row_parsing_from_csv():
    raw = (
        "Market_and_Exchange_Names,Report_Date_as_YYYY-MM-DD,Open_Interest_All,"
        "Asset_Mgr_Positions_Long_All,Asset_Mgr_Positions_Short_All\n"
        '"E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE",2026-09-08,2071836,1000000,87639\n'
    )
    row = next(csv.DictReader(io.StringIO(raw)))
    assert _as_float(row["Open_Interest_All"]) == pytest.approx(2_071_836)
    long_ = _as_float(row["Asset_Mgr_Positions_Long_All"])
    short = _as_float(row["Asset_Mgr_Positions_Short_All"])
    assert long_ - short == pytest.approx(912_361)


# ---------------------------------------------------------------- insider


def test_insider_notional_uses_shares_transacted_not_holdings():
    """The trap: Finnhub's `share` is the post-trade holding, not the trade.

    Cascade Investment holds ~114M RSG shares. Valuing that at the trade price
    reports a $25bn transaction instead of the real $51m.
    """
    from tickerlake.fetchers.insider import _notional

    entry = {"share": 113_886_505, "change": 235_978, "transactionPrice": 219.4076}

    correct = _notional(entry["change"], entry["transactionPrice"])
    assert correct == pytest.approx(235_978 * 219.4076)
    assert 5e7 < correct < 6e7, "a real institutional purchase, ~$52m"

    wrong = _notional(entry["share"], entry["transactionPrice"])
    assert wrong > 2e10, "using holdings would overstate this by ~480x"


def test_insider_notional_handles_missing_inputs():
    from tickerlake.fetchers.insider import _notional

    assert _notional(None, 10.0) is None
    assert _notional(100, None) is None
    assert _notional("not a number", 10.0) is None


def test_insider_schema_names_are_unambiguous():
    """Naming is the fix: `shares` alone invited the error above."""
    from tickerlake.storage import paths as P
    from tickerlake.storage import schemas as S

    names = set(S.SCHEMAS[P.INSIDER].names)
    assert {"shares_held_after", "shares_transacted", "transaction_value"} <= names
    assert "shares" not in names, "the ambiguous name must not come back"
    assert "share_change" not in names
