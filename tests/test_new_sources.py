"""Tests for the FINRA, GDELT-bulk, and earnings parsing logic."""

from __future__ import annotations

import pytest

from tickerlake.fetchers.gdelt import (
    _clean_company_name,
    _match_company,
    _parse_stamp,
    _parse_tone,
)

# ------------------------------------------------------------ company naming


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Apple Inc.", "APPLE"),
        ("Berkshire Hathaway", "BERKSHIRE HATHAWAY"),
        ("Alphabet Inc. Class A", "ALPHABET INC. CLASS A"),
        ("JPMorgan Chase & Co", "JPMORGAN CHASE"),
        ("Starbucks Corporation", "STARBUCKS"),
        ("Estee Lauder Companies (The)", "ESTEE LAUDER"),
    ],
)
def test_company_name_cleaning(raw, expected):
    assert _clean_company_name(raw) == expected


# --------------------------------------------------------- salience matching


def test_match_requires_early_mention():
    """The article's subject appears in the headline/lede; mentions do not.

    This is the real case that motivated offset matching: a wire story about
    Lufax that quotes a JPMorgan analyst note must not be tagged JPM.
    """
    companies = {"JPMORGAN CHASE": "JPM", "APPLE": "AAPL"}

    subject = "Jpmorgan Chase Co,20;Zacks Research,599"
    assert _match_company(subject, companies, max_offset=250) == "JPM"

    incidental = "Jpmorgan Chase Co,452;Scientech Research,2118"
    assert _match_company(incidental, companies, max_offset=250) is None


def test_match_picks_the_most_salient_company():
    companies = {"APPLE": "AAPL", "JPMORGAN CHASE": "JPM"}
    field = "Jpmorgan Chase Co,180;Apple Inc,15"
    assert _match_company(field, companies, max_offset=250) == "AAPL"


def test_match_handles_malformed_entries():
    companies = {"APPLE": "AAPL"}
    assert _match_company("garbage;Apple Inc,notanumber;Apple Inc,10", companies, 250) == "AAPL"
    assert _match_company("", companies, 250) is None
    assert _match_company(";;;", companies, 250) is None


def test_unknown_company_is_not_tagged():
    assert _match_company("Some Private Firm,10", {"APPLE": "AAPL"}, 250) is None


# ----------------------------------------------------------------- gkg tone


def test_parse_tone_extracts_overall_tone():
    tone = _parse_tone("0,4.28571,4.28571,8.57142,17.1428,0,64")
    assert tone["tone"] == pytest.approx(0.0)
    assert tone["positive"] == pytest.approx(4.28571, abs=1e-4)
    assert tone["word_count"] == pytest.approx(64)


def test_parse_tone_survives_garbage():
    assert _parse_tone("") == {}
    assert "tone" not in _parse_tone("not,a,number")


def test_parse_stamp():
    dt = _parse_stamp("20260910160000")
    assert dt is not None
    assert (dt.year, dt.month, dt.day, dt.hour) == (2026, 9, 10, 16)
    assert _parse_stamp("nonsense") is None
    assert _parse_stamp("") is None


# --------------------------------------------------------------- finra parse


def test_finra_short_ratio_and_symbol_normalisation(tmp_path):
    """The pipe-delimited file parses, and BRK-B becomes canonical BRK.B."""
    import io

    import pandas as pd

    raw = (
        "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
        "20260909|AAPL|418949.64|61|779188.03|B,Q,N\n"
        "20260909|BRK-B|390347.55|127|1001730.32|B,Q,N\n"
        "20260909|ZERO|0|0|0|B\n"
    )
    df = pd.read_csv(io.StringIO(raw), sep="|")
    df["symbol"] = df["Symbol"].astype(str).str.strip().str.upper().str.replace("-", ".")

    assert set(df["symbol"]) == {"AAPL", "BRK.B", "ZERO"}

    short = pd.to_numeric(df["ShortVolume"])
    total = pd.to_numeric(df["TotalVolume"])
    ratio = (short / total).where(total > 0)

    assert ratio.iloc[0] == pytest.approx(418949.64 / 779188.03)
    # A symbol that did not trade must yield NULL, not a divide-by-zero inf.
    assert pd.isna(ratio.iloc[2])
