"""Exchange listings from Nasdaq Trader's symbol directories.

Two pipe-delimited files, refreshed every night, list everything listed on a US
exchange: ``nasdaqlisted.txt`` (Nasdaq) and ``otherlisted.txt`` (NYSE, NYSE
American, NYSE Arca, Cboe BZX, IEX). They describe only today, so the daily
snapshot this stage keeps is the only point-in-time record of what was listed,
which is what exchange-wide breadth needs to stay free of survivorship from the
day collection starts.

Each security is classified from its name. Breadth on "the NYSE" means common
stock; the directories also carry preferreds, warrants, units, rights, notes and
ETFs. ADRs, REITs, closed-end funds and partnerships count as common, as they do
in the exchange's own advance/decline figures.
"""

from __future__ import annotations

import io
import re
from datetime import UTC, date, datetime

import pandas as pd

from tickerlake.fetchers.base import BaseFetcher, FetchResult
from tickerlake.storage import paths as P
from tickerlake.utils.http import HttpClient

SOURCE = "nasdaqtrader"
NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

# A directory this thin is a broken download, not a market that delisted half of itself.
MIN_ROWS = 5000
MIN_NYSE_COMMON = 1000

_UNIT = re.compile(r"\beach consisting of\b|\bunits?\b")
_NOT_UNIT = re.compile(r"common units|beneficial interest|partnership|royalty|limited partner|\bl\.?p\.?(\s|$)|holding|investment trust")
_WARRANT = re.compile(r"\bwarrants?\b")
_RIGHT = re.compile(r"\brights?\b")
# "Preferred and Income Fund" is a closed-end fund's common shares: only the security
# itself being preferred counts.
_PREFERRED = re.compile(r"preferred (stock|shares|securities)|\bpfd\b|depositary shares.*preferred")
_NOTE = re.compile(r"\bnotes?\b|\bdebentures?\b")


def classify(symbol: str, name: str, etf: bool, test_issue: bool) -> str:
    """Security type from the directory's flags, symbol and name."""
    if test_issue:
        return "test"
    if etf:
        return "etf"
    if "$" in symbol:  # NYSE preferred series: ABR$D
        return "preferred"
    n = (name or "").lower()
    if "depositary" in n:  # ADRs: "each representing the right to receive", "one unit"
        return "preferred" if _PREFERRED.search(n) else "common"
    if "each consisting of" in n:  # SPAC units name their share and warrant parts
        return "unit"
    if _WARRANT.search(n):
        return "warrant"
    if _RIGHT.search(n) and not _UNIT.search(n):
        return "right"
    if _UNIT.search(n) and not _NOT_UNIT.search(n):
        return "unit"
    if _PREFERRED.search(n):
        return "preferred"
    if _NOTE.search(n):
        return "note"
    return "common"


def _read(text: str) -> pd.DataFrame:
    body = "\n".join(line for line in text.splitlines() if not line.startswith("File Creation Time"))
    return pd.read_csv(io.StringIO(body), sep="|", dtype=str, keep_default_na=False)


def parse_listings(nasdaq_text: str, other_text: str, snapshot: date) -> pd.DataFrame:
    """Both directories as one snapshot in the LISTINGS schema."""
    nq = _read(nasdaq_text)
    nq = pd.DataFrame({
        "symbol": nq["Symbol"], "exchange": "Q", "security_name": nq["Security Name"],
        "etf": nq["ETF"].eq("Y"), "test_issue": nq["Test Issue"].eq("Y"),
    })
    ot = _read(other_text)
    ot = pd.DataFrame({
        "symbol": ot["ACT Symbol"], "exchange": ot["Exchange"], "security_name": ot["Security Name"],
        "etf": ot["ETF"].eq("Y"), "test_issue": ot["Test Issue"].eq("Y"),
    })
    df = pd.concat([nq, ot], ignore_index=True)
    df["symbol"] = df["symbol"].str.strip().str.upper()
    df = df[df["symbol"].ne("")].drop_duplicates("symbol", keep="first")
    df["category"] = [classify(s, n, e, t) for s, n, e, t in
                      zip(df["symbol"], df["security_name"], df["etf"], df["test_issue"], strict=True)]
    df.insert(0, "snapshot_date", snapshot)
    df["source"] = SOURCE
    df["ingested_at"] = datetime.now(UTC)
    return df.reset_index(drop=True)


def listed_symbols(root, as_of: date, exchanges: list[str], categories: list[str]) -> list[str]:
    """Symbols in the latest stored snapshot on or before ``as_of``."""
    from tickerlake.storage.query import LakeQuery

    ex = ",".join("?" * len(exchanges))
    cat = ",".join("?" * len(categories))
    with LakeQuery(root) as q:
        df = q.sql(
            f"SELECT symbol FROM {P.LISTINGS} WHERE snapshot_date = "
            f"(SELECT MAX(snapshot_date) FROM {P.LISTINGS} WHERE snapshot_date <= ?) "
            f"AND exchange IN ({ex}) AND category IN ({cat})",
            [as_of, *exchanges, *categories],
        )
    return sorted(df["symbol"].astype(str))


class ListingsFetcher(BaseFetcher):
    """Daily snapshot of every US exchange listing."""

    name = "listings"
    dataset = P.LISTINGS

    def collect(self, run_date: date, result: FetchResult) -> None:
        client = HttpClient(requests_per_second=1.0, user_agent="Mozilla/5.0 (TickerLake)")
        df = parse_listings(client.get_text(NASDAQ_URL), client.get_text(OTHER_URL), run_date)
        nyse_common = int(((df["exchange"] == "N") & (df["category"] == "common")).sum())
        if len(df) < MIN_ROWS or nyse_common < MIN_NYSE_COMMON:
            result.add_error(
                f"implausible directory: {len(df)} listings, {nyse_common} NYSE common; not stored"
            )
            return
        write = self.writer.write(df, P.LISTINGS, self.paths.listings_file(run_date.year), mode="merge")
        result.record_write(write)
        result.items_succeeded = len(df)
        counts = df.groupby(["exchange", "category"]).size()
        result.details.update({
            "listings": len(df),
            "nyse_common": nyse_common,
            "by_exchange": {ex: int(n) for ex, n in df["exchange"].value_counts().items()},
            "common_by_exchange": {ex: int(counts.get((ex, "common"), 0)) for ex in sorted(df["exchange"].unique())},
        })
        self.log.info("listings: %d securities, %d NYSE common stocks", len(df), nyse_common)
