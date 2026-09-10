"""Weekly compaction: merge small daily files into consolidated partitions.

The daily run writes one small file per symbol (options) or per run (everything
else), which is what makes it resumable. Left alone, that produces ~500 files a
day and DuckDB ends up spending its time opening files instead of reading them.

Compaction merges each partition's deltas into a single sorted ``data.parquet``.
Sorting matters as much as merging: it is what makes Parquet's row-group
statistics able to skip most of a file when you filter by symbol.

Safety: the merged file is written and verified before any delta is deleted, so
an interrupted compaction can only ever leave *duplicate* data (which the reader
tolerates, and the next compaction removes), never missing data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from tickerlake.config import Config
from tickerlake.storage import paths as P
from tickerlake.storage.writer import ParquetWriter

log = logging.getLogger(__name__)

# Files a partition considers its consolidated output.
COMPACTED_NAME = "data.parquet"


@dataclass
class CompactionResult:
    dataset: str
    partitions_compacted: int = 0
    files_merged: int = 0
    files_removed: int = 0
    rows: int = 0
    bytes_before: int = 0
    bytes_after: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.partitions_compacted:
            return f"{self.dataset}: nothing to compact"
        saved = self.bytes_before - self.bytes_after
        return (
            f"{self.dataset}: {self.partitions_compacted} partition(s), "
            f"{self.files_merged} -> {self.partitions_compacted} files, "
            f"{self.rows:,} rows, "
            f"{self.bytes_before / 1_048_576:.1f} -> {self.bytes_after / 1_048_576:.1f} MB "
            f"({saved / 1_048_576:+.1f} MB)"
        )


class Compactor:
    """Merges delta files into consolidated per-partition Parquet files."""

    def __init__(self, config: Config):
        self.config = config
        self.paths = P.DatasetPaths(config.data_root)
        self.writer = ParquetWriter(
            compression=config.get("storage.compression", "zstd"),
            compression_level=config.get("storage.compression_level", 3),
        )
        self.min_age_days = int(config.get("compaction.min_age_days", 2))

    def run(self, datasets: list[str] | None = None, force: bool = False) -> list[CompactionResult]:
        targets = datasets or list(
            self.config.get("compaction.datasets", [P.OHLCV, P.OPTIONS_CHAINS])
        )
        results = []
        for dataset in targets:
            try:
                results.append(self._compact_dataset(dataset, force))
            except Exception as exc:
                result = CompactionResult(dataset=dataset)
                result.errors.append(f"{type(exc).__name__}: {exc}")
                log.error("compaction of %s failed: %s", dataset, exc)
                results.append(result)
        for result in results:
            log.info(result.summary())
        return results

    # ------------------------------------------------------------ internals

    def _compact_dataset(self, dataset: str, force: bool) -> CompactionResult:
        result = CompactionResult(dataset=dataset)
        base = self.paths.dataset_dir(dataset)
        if not base.exists():
            return result

        cutoff = datetime.now(UTC) - timedelta(days=self.min_age_days)

        for partition in _leaf_partitions(base):
            files = sorted(p for p in partition.glob("*.parquet") if not p.name.endswith(".tmp"))
            deltas = [f for f in files if f.name != COMPACTED_NAME]
            if not deltas:
                continue
            # One delta and no existing consolidated file is already optimal.
            if len(files) == 1 and not force and files[0].name != COMPACTED_NAME:
                if _newest_mtime(files) >= cutoff:
                    continue

            if not force and _newest_mtime(deltas) >= cutoff:
                log.debug("%s: deltas still inside the %d-day window", partition, self.min_age_days)
                continue

            try:
                self._compact_partition(dataset, partition, files, deltas, result)
            except Exception as exc:
                result.errors.append(f"{partition.name}: {exc}")
                log.error("could not compact %s: %s", partition, exc)

        return result

    def _compact_partition(
        self,
        dataset: str,
        partition: Path,
        files: list[Path],
        deltas: list[Path],
        result: CompactionResult,
    ) -> None:
        bytes_before = sum(f.stat().st_size for f in files)

        frames = []
        for path in files:
            try:
                frames.append(pq.read_table(path).to_pandas())
            except Exception as exc:
                # Never delete a delta whose contents we could not read.
                raise RuntimeError(f"unreadable file {path.name}: {exc}") from exc
        if not frames:
            return

        combined = pd.concat(frames, ignore_index=True)
        target = partition / COMPACTED_NAME

        # Write to a staging name so the live data.parquet stays valid until the
        # merged file is complete.
        staging = partition / "_compacting.parquet"
        write = self.writer.write(combined, dataset, staging, mode="overwrite")
        if write.rows_written == 0:
            staging.unlink(missing_ok=True)
            return

        verified = pq.read_metadata(staging).num_rows
        if verified != write.rows_written:
            staging.unlink(missing_ok=True)
            raise RuntimeError(
                f"verification failed for {partition.name}: "
                f"wrote {write.rows_written} rows, file reports {verified}"
            )

        staging.replace(target)

        removed = 0
        for path in deltas:
            if path.name == COMPACTED_NAME:
                continue
            try:
                path.unlink()
                removed += 1
            except OSError as exc:
                log.warning("could not remove delta %s: %s", path.name, exc)

        result.partitions_compacted += 1
        result.files_merged += len(files)
        result.files_removed += removed
        result.rows += write.rows_written
        result.bytes_before += bytes_before
        result.bytes_after += target.stat().st_size
        log.info(
            "compacted %s: %d files -> 1 (%d rows, %.1f -> %.1f MB)",
            partition.relative_to(self.paths.root),
            len(files),
            write.rows_written,
            bytes_before / 1_048_576,
            target.stat().st_size / 1_048_576,
        )


def _leaf_partitions(base: Path) -> list[Path]:
    """Directories that directly contain Parquet files."""
    out = {p.parent for p in base.rglob("*.parquet")}
    return sorted(out)


def _newest_mtime(files: list[Path]) -> datetime:
    newest = max((f.stat().st_mtime for f in files), default=0)
    return datetime.fromtimestamp(newest, tz=UTC)
