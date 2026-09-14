"""Dataset path layout for the lake.

Partitioning rationale
----------------------
Two competing layouts exist for OHLCV: symbol-first (`symbol=AAPL/year=2024/`) and
date-first (`year=2024/month=03/`). We use **date-first with `symbol` as a sorted
column**, for two reasons:

1. File count. Symbol-first at symbol/year/month granularity is
   500 symbols x 15 years x 12 months = ~90,000 files of ~21 rows each. Parquet
   metadata overhead then dominates the actual data and DuckDB spends its time
   opening files rather than reading them.
2. Query shape. ML feature engineering is overwhelmingly cross-sectional
   ("every symbol on date X"), which date-first prunes perfectly. The reverse
   query ("all of AAPL's history") stays fast because we sort each file by
   symbol, so Parquet row-group min/max statistics let DuckDB skip almost every
   row group without a directory partition.

Options use `snapshot_date` as the only partition key for the same reason: at
500 symbols x ~16 expirations, partitioning by expiration too would create
~8,000 files/day.

Within a partition there are two file kinds:
  - ``_daily_<date>.parquet`` / ``sym_<SYMBOL>.parquet``: incremental writes from a
    run. Small, and the unit of resumability.
  - ``data.parquet``: the compacted consolidation, produced by the weekly job.
Both are read together by the query layer; compaction deletes the deltas only
after the merged file is durably written.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# Dataset directory names, also used as DuckDB view names.
OHLCV = "ohlcv"
OPTIONS_CHAINS = "options_chains"
FILINGS_TEXT = "filings_text"
FILINGS_FACTS = "filings_facts"
MACRO_SERIES = "macro_series"
FACTORS = "factors"
YIELD_CURVE = "yield_curve"
COT = "cot"
INSIDER = "insider"
NEWS_EVENTS = "news_events"
INTRADAY_BARS = "intraday_bars"
BOOK_SNAPSHOTS = "book_snapshots"
OPTIONS_GREEKS = "options_greeks"
OPTIONS_FLOW = "options_flow"
SHORT_VOLUME = "short_volume"
EARNINGS = "earnings"
MEMBERSHIP = "membership"
UNIVERSE_HISTORY = "universe_history"
QUALITY = "quality"

DATASETS = [
    OHLCV,
    OPTIONS_CHAINS,
    INTRADAY_BARS,
    BOOK_SNAPSHOTS,
    OPTIONS_GREEKS,
    OPTIONS_FLOW,
    SHORT_VOLUME,
    EARNINGS,
    FILINGS_TEXT,
    FILINGS_FACTS,
    MACRO_SERIES,
    FACTORS,
    YIELD_CURVE,
    COT,
    INSIDER,
    NEWS_EVENTS,
    MEMBERSHIP,
    UNIVERSE_HISTORY,
    QUALITY,
]

# Internal (non-dataset) directories.
CHECKPOINTS = "_checkpoints"
LOGS = "_logs"
RUNS = "_runs"

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def safe_symbol(symbol: str) -> str:
    """Filesystem-safe rendering of a ticker.

    Tickers carry dots (BRK.B) and occasionally slashes or carets in historical
    data. Dots are legal in filenames; everything else becomes an underscore.
    """
    return _UNSAFE.sub("_", symbol.strip().upper())


@dataclass(frozen=True)
class DatasetPaths:
    """Resolves every path in the lake. All methods create parent dirs on demand."""

    root: Path

    def ensure_layout(self) -> None:
        for name in DATASETS:
            (self.root / name).mkdir(parents=True, exist_ok=True)
        for name in (CHECKPOINTS, LOGS, RUNS):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def dataset_dir(self, dataset: str) -> Path:
        return self.root / dataset

    # ---------------------------------------------------------------- ohlcv

    def ohlcv_partition(self, year: int, month: int) -> Path:
        return self.root / OHLCV / f"year={year:04d}" / f"month={month:02d}"

    def ohlcv_daily_file(self, day: date) -> Path:
        part = self.ohlcv_partition(day.year, day.month)
        part.mkdir(parents=True, exist_ok=True)
        return part / f"_daily_{day.isoformat()}.parquet"

    def ohlcv_backfill_file(self, year: int, month: int) -> Path:
        """Consolidated file written directly by the backfill (not a delta)."""
        part = self.ohlcv_partition(year, month)
        part.mkdir(parents=True, exist_ok=True)
        return part / "data.parquet"

    # -------------------------------------------------------------- options

    def options_partition(self, snapshot: date) -> Path:
        return self.root / OPTIONS_CHAINS / f"snapshot_date={snapshot.isoformat()}"

    def options_symbol_file(self, snapshot: date, symbol: str) -> Path:
        part = self.options_partition(snapshot)
        part.mkdir(parents=True, exist_ok=True)
        return part / f"sym_{safe_symbol(symbol)}.parquet"

    def options_compacted_file(self, snapshot: date) -> Path:
        part = self.options_partition(snapshot)
        part.mkdir(parents=True, exist_ok=True)
        return part / "data.parquet"

    def intraday_partition(self, interval: str, day: date) -> Path:
        return (
            self.root
            / INTRADAY_BARS
            / f"interval={safe_symbol(interval).lower()}"
            / f"date={day.isoformat()}"
        )

    def intraday_file(self, interval: str, day: date) -> Path:
        part = self.intraday_partition(interval, day)
        part.mkdir(parents=True, exist_ok=True)
        return part / "data.parquet"

    def book_snapshot_file(self, day: date) -> Path:
        """1-minute L2 snapshots. Same grid as intraday_bars, so they join."""
        part = self.root / BOOK_SNAPSHOTS / f"date={day.isoformat()}"
        part.mkdir(parents=True, exist_ok=True)
        return part / "data.parquet"

    def options_greeks_file(self, snapshot: date) -> Path:
        part = self.root / OPTIONS_GREEKS / f"snapshot_date={snapshot.isoformat()}"
        part.mkdir(parents=True, exist_ok=True)
        return part / "data.parquet"

    def options_flow_file(self, snapshot: date) -> Path:
        d = self.root / OPTIONS_FLOW
        d.mkdir(parents=True, exist_ok=True)
        return d / f"_daily_{snapshot.isoformat()}.parquet"

    # ---------------------------------------------------------- short volume

    def short_volume_file(self, day: date) -> Path:
        part = self.root / SHORT_VOLUME / f"year={day.year:04d}"
        part.mkdir(parents=True, exist_ok=True)
        return part / f"_daily_{day.isoformat()}.parquet"

    # -------------------------------------------------------------- earnings

    def earnings_file(self, kind: str) -> Path:
        """One file per kind: surprises, calendar, recommendations."""
        d = self.root / EARNINGS
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{safe_symbol(kind).lower()}.parquet"

    # ----------------------------------------------------------- membership

    def membership_parquet(self) -> Path:
        d = self.root / MEMBERSHIP
        d.mkdir(parents=True, exist_ok=True)
        return d / "sp500_membership.parquet"

    def membership_csv(self) -> Path:
        """Human-readable, diffable mirror. Small enough to version-control."""
        d = self.root / MEMBERSHIP
        d.mkdir(parents=True, exist_ok=True)
        return d / "sp500_membership.csv"

    def universe_history_file(self) -> Path:
        """Append-only log of every observed daily universe + detected changes."""
        d = self.root / UNIVERSE_HISTORY
        d.mkdir(parents=True, exist_ok=True)
        return d / "universe_history.parquet"

    # ------------------------------------------------------------- filings

    def filings_text_file(self, day: date) -> Path:
        part = self.root / FILINGS_TEXT / f"year={day.year:04d}"
        part.mkdir(parents=True, exist_ok=True)
        return part / f"_daily_{day.isoformat()}.parquet"

    def filings_facts_file(self, symbol: str) -> Path:
        d = self.root / FILINGS_FACTS
        d.mkdir(parents=True, exist_ok=True)
        return d / f"sym_{safe_symbol(symbol)}.parquet"

    # --------------------------------------------------------------- macro

    def macro_series_file(self, series_id: str) -> Path:
        d = self.root / MACRO_SERIES
        d.mkdir(parents=True, exist_ok=True)
        return d / f"series_{safe_symbol(series_id)}.parquet"

    # ------------------------------------------------------ macro extensions

    def factors_file(self, factor_set: str) -> Path:
        d = self.root / FACTORS
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{safe_symbol(factor_set).lower()}.parquet"

    def yield_curve_file(self, year: int) -> Path:
        d = self.root / YIELD_CURVE
        d.mkdir(parents=True, exist_ok=True)
        return d / f"year={year:04d}.parquet"

    def cot_file(self, year: int) -> Path:
        d = self.root / COT
        d.mkdir(parents=True, exist_ok=True)
        return d / f"year={year:04d}.parquet"

    def insider_file(self, year: int) -> Path:
        d = self.root / INSIDER
        d.mkdir(parents=True, exist_ok=True)
        return d / f"year={year:04d}.parquet"

    # ---------------------------------------------------------------- news

    def news_file(self, day: date, source: str) -> Path:
        part = self.root / NEWS_EVENTS / f"date={day.isoformat()}"
        part.mkdir(parents=True, exist_ok=True)
        return part / f"_{safe_symbol(source).lower()}.parquet"

    # ------------------------------------------------------------- quality

    def quality_file(self, day: date) -> Path:
        d = self.root / QUALITY
        d.mkdir(parents=True, exist_ok=True)
        return d / f"_daily_{day.isoformat()}.parquet"

    # ----------------------------------------------------------- internals

    def checkpoint_file(self, stage: str, day: date) -> Path:
        d = self.root / CHECKPOINTS
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{stage}_{day.isoformat()}.json"

    def run_summary_file(self, run_id: str) -> Path:
        d = self.root / RUNS
        d.mkdir(parents=True, exist_ok=True)
        return d / f"run_{run_id}.json"

    def logs_dir(self) -> Path:
        d = self.root / LOGS
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ----------------------------------------------------------- discovery

    def glob_dataset(self, dataset: str) -> list[Path]:
        """Every Parquet file in a dataset, deltas and compacted alike."""
        return sorted((self.root / dataset).rglob("*.parquet"))

    def dataset_glob_pattern(self, dataset: str) -> str:
        """DuckDB-friendly recursive glob for a dataset."""
        return (self.root / dataset / "**" / "*.parquet").as_posix()
