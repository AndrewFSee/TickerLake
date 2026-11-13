"""Tests for exchange listings and the listed-universe bars behind exchange breadth.

NYSE breadth needs every NYSE common stock, about 2,100 names, which must stay out
of the membership table: every stage iterating it (options, intraday, filings)
would otherwise start fetching them. And Yahoo posts most of these names' daily
bars hours after the close, so the session is first filled from hourly bars.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from tickerlake.config import Config, Secrets
from tickerlake.fetchers.listings import classify, listed_symbols, parse_listings
from tickerlake.fetchers.yf_listed import PROVISIONAL_SOURCE, ListedOHLCVFetcher, hourly_to_daily
from tickerlake.storage import paths as P
from tickerlake.storage.paths import DatasetPaths
from tickerlake.storage.writer import ParquetWriter

DAY = date(2026, 11, 12)

NASDAQ = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N
QQQ|Invesco QQQ Trust, Series 1|G|N|N|100|Y|N
ZXZZT|NASDAQ TEST STOCK|G|Y|N|100|N|N
File Creation Time: 1112202621:01||||||
"""
OTHER = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
IBM|International Business Machines Corporation Common Stock|N|IBM|N|100|N|IBM
BRK.B|Berkshire Hathaway Inc. Class B|N|BRK.B|N|100|N|BRK=B
ABR$D|Arbor Realty Trust 6.375% Series D Cumulative Redeemable Preferred Stock|N|ABRpD|N|100|N|ABR-D
SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY
GLDA|Gold Co Common Stock|A|GLDA|N|100|N|GLDA
File Creation Time: 1112202621:01||||||||
"""


@pytest.mark.parametrize("symbol,name,etf,test,expected", [
    ("IBM", "International Business Machines Corporation Common Stock", False, False, "common"),
    ("TSM", "Taiwan Semiconductor American Depositary Shares, each representing 5 shares", False, False, "common"),
    ("FFC", "Flaherty & Crumrine Preferred and Income Fund Common Stock", False, False, "common"),
    ("ET", "Energy Transfer LP Common Units representing limited partner interests", False, False, "common"),
    ("PBT", "Permian Basin Royalty Trust Units of Beneficial Interest", False, False, "common"),
    ("AB", "AllianceBernstein Holding L.P.  Units", False, False, "common"),
    ("BSRT", "BSR Real Estate Investment Trust Units", False, False, "common"),
    ("DUKU", "Duke Energy Corporation Corporate Units", False, False, "unit"),
    ("AMX", "America Movil American Depositary Shares (each representing the right to receive twenty "
            "(20) Series B Shares", False, False, "common"),
    ("BSBR", "Banco Santander Brasil SA American Depositary Shares, each representing one unit", False, False,
     "common"),
    ("ABR$D", "Arbor Realty Trust Series D Preferred Stock", False, False, "preferred"),
    ("XYZ.U", "XYZ Acquisition Corp Units, each consisting of one share and one-half of one warrant", False, False, "unit"),
    ("XYZ.WS", "XYZ Acquisition Corp Redeemable Warrants", False, False, "warrant"),
    ("XYZ.R", "XYZ Acquisition Corp Rights", False, False, "right"),
    ("AOMN", "Angel Oak Mortgage 9.500% Senior Notes due 2029", False, False, "note"),
    ("PFD", "Some Bank Depositary Shares, each representing a 1/1000th interest in a share of Preferred Stock", False, False, "preferred"),
    ("SPY", "SPDR S&P 500 ETF Trust", True, False, "etf"),
    ("ZXZZT", "NASDAQ TEST STOCK", False, True, "test"),
])
def test_classify(symbol, name, etf, test, expected):
    assert classify(symbol, name, etf, test) == expected


def test_parse_listings_merges_both_directories():
    df = parse_listings(NASDAQ, OTHER, DAY)
    assert set(df["exchange"]) == {"Q", "N", "P", "A"}
    assert not df["symbol"].str.startswith("File").any()
    cat = df.set_index("symbol")["category"]
    assert cat["IBM"] == "common" and cat["BRK.B"] == "common" and cat["ABR$D"] == "preferred"
    assert cat["SPY"] == "etf" and cat["QQQ"] == "etf" and cat["ZXZZT"] == "test"
    assert (df["snapshot_date"] == DAY).all()


def _config(tmp_path, **listed):
    raw = {"listed_ohlcv": {"enabled": True, "exchanges": ["N"], "categories": ["common"], **listed}}
    return Config(raw=raw, secrets=Secrets(), data_root=tmp_path, config_path=tmp_path / "config.yaml")


@pytest.fixture
def lake(tmp_path):
    paths = DatasetPaths(tmp_path)
    paths.ensure_layout()
    w = ParquetWriter()
    w.write(parse_listings(NASDAQ, OTHER, DAY), P.LISTINGS, paths.listings_file(2026), mode="merge")
    return paths, w


def test_listed_symbols_picks_exchange_and_category(lake):
    paths, _ = lake
    assert listed_symbols(paths.root, DAY, ["N"], ["common"]) == ["BRK.B", "IBM"]
    assert listed_symbols(paths.root, date(2026, 12, 1), ["N", "A"], ["common"]) == ["BRK.B", "GLDA", "IBM"]


def test_listed_fetcher_never_touches_membership(lake, tmp_path):
    paths, w = lake
    f = ListedOHLCVFetcher(_config(tmp_path), paths, w)
    assert f._symbols(DAY) == ["BRK.B", "IBM"]
    assert f._current_symbols() == {"BRK.B", "IBM"}
    assert P.OHLCV_LISTED in str(f._month_file(2026, 11))
    f._record_observations({"IBM": None}, None)  # no tracker, no membership writes, no error


def _hourly(symbols, day, last_hour=15):
    idx = pd.date_range(f"{day} 09:30", f"{day} {last_hour}:30", freq="1h", tz="America/New_York")
    cols = pd.MultiIndex.from_product([symbols, ["Open", "High", "Low", "Close", "Volume"]])
    df = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for s in symbols:
        df[(s, "Open")] = range(10, 10 + len(idx))
        df[(s, "High")] = df[(s, "Open")] + 1
        df[(s, "Low")] = df[(s, "Open")] - 1
        df[(s, "Close")] = df[(s, "Open")] + 0.5
        df[(s, "Volume")] = 100
    return df


def test_hourly_to_daily_needs_the_finished_session():
    bars = hourly_to_daily(_hourly(["IBM"], DAY), ["IBM"], {"IBM": "IBM"}, DAY)
    row = bars.iloc[0]
    assert (row["open"], row["high"], row["low"], row["close"], row["volume"]) == (10, 17, 9, 16.5, 700)
    assert row["source"] == PROVISIONAL_SOURCE
    unfinished = hourly_to_daily(_hourly(["IBM"], DAY, last_hour=14), ["IBM"], {"IBM": "IBM"}, DAY)
    assert unfinished.empty  # run before the close: no bar


def test_provisional_bars_fill_only_missing_symbols(lake, tmp_path, monkeypatch):
    paths, w = lake
    official = pd.DataFrame([{
        "symbol": "IBM", "date": DAY, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "adj_close": 1.5,
        "volume": 10, "dividends": 0.0, "stock_splits": 0.0, "repaired": False, "source": "yfinance",
        "ingested_at": datetime.now(UTC),
    }])
    w.write(official, P.OHLCV_LISTED, paths.listed_ohlcv_file(2026, 11), mode="merge")
    f = ListedOHLCVFetcher(_config(tmp_path, throttle_seconds=0), paths, w)
    f._symbols(DAY)
    asked = []

    def fake_hourly(symbols, day):
        asked.extend(symbols)
        return hourly_to_daily(_hourly(symbols, day), symbols, {s: s for s in symbols}, day)

    monkeypatch.setattr(f, "_download_hourly", fake_hourly)
    from tickerlake.fetchers.base import FetchResult
    result = FetchResult(stage="listed_ohlcv", dataset=P.OHLCV_LISTED)
    f._provisional_session(DAY, result)
    assert asked == ["BRK.B"]  # IBM already has its official bar
    stored = pd.read_parquet(paths.listed_ohlcv_file(2026, 11)).set_index("symbol")
    assert stored.loc["BRK.B", "source"] == PROVISIONAL_SOURCE and stored.loc["IBM", "source"] == "yfinance"


def test_adjusted_ohlcv_reads_the_listed_dataset(lake, tmp_path):
    """Breadth uses dividend-adjusted closes: an 8% payer must not drift below its averages."""
    from tickerlake.storage.query import LakeQuery

    paths, w = lake
    now = datetime.now(UTC)
    bars = pd.DataFrame([{
        "symbol": "IBM", "date": d, "open": c, "high": c, "low": c, "close": c, "adj_close": c,
        "volume": 10, "dividends": div, "stock_splits": 0.0, "repaired": False, "source": "yfinance",
        "ingested_at": now} for d, c, div in [(date(2026, 11, 10), 100.0, 0.0), (date(2026, 11, 11), 98.0, 2.0)]])
    w.write(bars, P.OHLCV_LISTED, paths.listed_ohlcv_file(2026, 11), mode="merge")
    with LakeQuery(paths.root) as q:
        adj = q.adjusted_ohlcv(dataset=P.OHLCV_LISTED).set_index("date")["adj_close"]
        assert q.adjusted_ohlcv().empty  # the index dataset is untouched
    assert adj[pd.Timestamp("2026-11-10")] == pytest.approx(98.0)  # 100 x (1 - 2/100): flat on a total-return basis
