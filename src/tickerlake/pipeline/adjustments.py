"""Rewrites the stored ``adj_close`` from the corporate actions on disk.

Why the stored column needs repairing at all
--------------------------------------------
``adj_close`` is a *running total over the future*: the value for a 2015 session
depends on every dividend paid since. The fetcher writes a row when its session
is current -- at which point the adjustment is 1.0 by definition -- and the
incremental lookback only ever revisits the last few sessions, so the row then
freezes. Every later dividend should reach back and lower it, and none of them
do. Crown Castle's rows from a year ago were understated by 1.45%, a single
missed dividend, across the symbol's entire history.

The events themselves do not have this problem. A dividend is a fact about one
day, not a running total, so ``dividends`` and ``stock_splits`` stay correct
once written -- 32k dividend events and 342 splits, complete. That makes the
adjustment reconstructible from data already on disk, with no vendor call.

:meth:`LakeQuery.adjusted_ohlcv` computes it at query time and is always right.
This module exists for the other reader: anything opening the Parquet directly,
or calling :meth:`LakeQuery.ohlcv`, still sees the stored column. Running this
keeps that column honest between dividends rather than letting it decay
silently.

Two things are deliberately *not* done here.

Splits are not applied. ``close`` already carries them -- the fetcher uses
``auto_adjust=False``, which back-adjusts Close for splits and puts only
dividends into Adj Close. Re-applying the ratio double-counts it: 3M's 2024
Solventum spin-off is recorded as ``stock_splits = 1.196``, and applying it
moved the series by 17%.

And the series is not rebuilt from scratch. The stored column encodes vendor
handling this lake cannot reproduce: Danaher's 2016 Fortive spin-off is recorded
as a $24.56 dividend *and* a 1.319 split, and Yahoo's factor across it matches
neither reading of that pair. A full rebuild moved Danaher's pre-2016 history by
13.4% in the wrong direction.

So neither source is taken on faith. The adjustment ratio can only fall as you
go back in time, so the defensible value for a row is the most-adjusted one any
later row implies -- see :meth:`LakeQuery.corrected_adjustments`. Where the
vendor is stale our dividends win; where our actions are incomplete the vendor
wins; and running it twice changes nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from tickerlake.config import Config
from tickerlake.storage import paths as P
from tickerlake.storage.query import LakeQuery
from tickerlake.storage.writer import ParquetWriter

log = logging.getLogger(__name__)

# Below this the two series are the same number to any use a model would make
# of them, and rewriting the file would only churn bytes.
DEFAULT_TOLERANCE = 0.0005


@dataclass
class RepairResult:
    files_scanned: int = 0
    files_rewritten: int = 0
    rows_corrected: int = 0
    symbols_corrected: int = 0
    max_drift_pct: float = 0.0
    dry_run: bool = False
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        verb = "would correct" if self.dry_run else "corrected"
        return (
            f"adjustments: {verb} {self.rows_corrected:,} row(s) across "
            f"{self.symbols_corrected} symbol(s) in {self.files_rewritten}/"
            f"{self.files_scanned} file(s); worst drift {self.max_drift_pct:.4f}%"
        )


class AdjustmentRepair:
    """Recomputes ``adj_close`` for every stored OHLCV row."""

    def __init__(self, config: Config):
        self.config = config
        self.paths = P.DatasetPaths(config.data_root)
        self.writer = ParquetWriter(
            compression=config.get("storage.compression", "zstd"),
            compression_level=config.get("storage.compression_level", 3),
        )

    def run(self, tolerance: float = DEFAULT_TOLERANCE, dry_run: bool = False) -> RepairResult:
        result = RepairResult(dry_run=dry_run)

        with LakeQuery(self.paths.root) as q:
            correct = q.corrected_adjustments()
        if correct.empty:
            log.info("no OHLCV rows to repair")
            return result

        # One lookup for the whole lake. Keyed on (symbol, date) as plain
        # values: the frames come from DuckDB and from Parquet respectively,
        # which disagree on whether a date32 is a Timestamp or a date object.
        correct["date"] = pd.to_datetime(correct["date"])
        correct = correct.set_index([correct["symbol"].astype(str), correct["date"]])["adj_close"]

        base = self.paths.root / P.OHLCV
        if not base.exists():
            return result

        touched: set[str] = set()
        for path in sorted(base.rglob("*.parquet")):
            result.files_scanned += 1
            try:
                self._repair_file(path, correct, tolerance, result, touched, dry_run)
            except Exception as exc:
                result.errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
                log.error("could not repair %s: %s", path.name, exc)

        result.symbols_corrected = len(touched)
        log.info(result.summary())
        return result

    # ------------------------------------------------------------- internals

    def _repair_file(
        self,
        path: Path,
        correct: pd.Series,
        tolerance: float,
        result: RepairResult,
        touched: set[str],
        dry_run: bool,
    ) -> None:
        df = pq.read_table(path).to_pandas()
        if df.empty or "adj_close" not in df.columns:
            return

        index = pd.MultiIndex.from_arrays([df["symbol"].astype(str), pd.to_datetime(df["date"])])
        target = correct.reindex(index).to_numpy()

        stored = pd.to_numeric(df["adj_close"], errors="coerce").to_numpy(dtype=float)
        # Only rows the recomputation actually covers and actually disagrees
        # with. A row with no computed value -- a symbol absent from the scan --
        # is left exactly as it was rather than blanked.
        usable = ~np.isnan(target) & ~np.isnan(stored) & (target > 0)
        drift = np.zeros(len(df), dtype=float)
        np.divide(np.abs(target - stored), np.abs(target), out=drift, where=usable)
        changed = usable & (drift >= tolerance)

        if not changed.any():
            return

        worst = float(drift[changed].max() * 100)
        result.max_drift_pct = max(result.max_drift_pct, worst)
        result.rows_corrected += int(changed.sum())
        touched.update(df.loc[changed, "symbol"].astype(str).unique())
        result.files_rewritten += 1

        if dry_run:
            return

        df.loc[changed, "adj_close"] = target[changed]
        self.writer.write(df, P.OHLCV, path, mode="overwrite")
