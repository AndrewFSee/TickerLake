"""Parquet writing: schema coercion, validation, de-duplication, atomic replace.

Idempotency model
-----------------
Every write targets a *deterministic* path derived from (dataset, date, symbol).
Re-running the same day therefore overwrites rather than appends, which is what
makes the whole pipeline safe to re-run. Where a dataset genuinely accumulates
into one file (macro series, XBRL facts, membership), ``mode="merge"`` reads the
existing file, concatenates, and de-duplicates on the dataset's natural key
keeping the newest row.

Writes go to a ``.tmp`` sibling and are then ``os.replace``d into place, so a
crash mid-write leaves the previous good file intact rather than a truncated
Parquet that poisons every later scan.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tickerlake.storage import schemas as S

log = logging.getLogger(__name__)

WriteMode = Literal["overwrite", "merge", "skip_if_exists"]


@dataclass
class WriteResult:
    path: Path
    dataset: str
    rows_written: int
    rows_input: int
    bytes_written: int
    skipped: bool = False
    validation: S.ValidationReport | None = None

    def __str__(self) -> str:
        if self.skipped:
            return f"skipped {self.path.name} (already exists)"
        mb = self.bytes_written / 1_048_576
        return f"wrote {self.rows_written:,} rows -> {self.path.name} ({mb:.2f} MB)"


class ParquetWriter:
    """Writes validated, schema-conformant Parquet into the lake."""

    def __init__(
        self,
        compression: str = "zstd",
        compression_level: int = 3,
        strict_validation: bool = False,
    ) -> None:
        self.compression = compression
        self.compression_level = compression_level
        self.strict_validation = strict_validation

    # ------------------------------------------------------------------ write

    def write(
        self,
        df: pd.DataFrame,
        dataset: str,
        path: Path,
        mode: WriteMode = "overwrite",
        sort_by: list[str] | None = None,
        allow_empty: bool = False,
    ) -> WriteResult:
        """Validate, coerce, and atomically write ``df`` to ``path``."""
        rows_input = len(df)

        if mode == "skip_if_exists" and path.exists():
            return WriteResult(path, dataset, 0, rows_input, 0, skipped=True)

        if df.empty and not allow_empty:
            log.debug("%s: nothing to write to %s", dataset, path.name)
            return WriteResult(path, dataset, 0, 0, 0, skipped=True)

        report = S.validate_frame(df, dataset, strict=self.strict_validation)
        if not report.ok:
            log.warning("validation issues: %s", report.summary())
        elif report.warnings:
            log.debug("validation notes: %s", report.summary())

        if mode == "merge" and path.exists():
            df = self._merge_with_existing(df, dataset, path)

        df = self._dedupe(df, dataset)

        sort_cols = sort_by or self._default_sort(dataset)
        present = [c for c in sort_cols if c in df.columns]
        if present:
            # Sorting is what makes row-group min/max statistics useful, which is
            # how DuckDB skips row groups without a directory partition per symbol.
            df = df.sort_values(present, kind="stable").reset_index(drop=True)

        table = self._to_table(df, dataset)
        bytes_written = self._atomic_write(table, path)

        return WriteResult(
            path=path,
            dataset=dataset,
            rows_written=table.num_rows,
            rows_input=rows_input,
            bytes_written=bytes_written,
            validation=report,
        )

    # ------------------------------------------------------------- internals

    def _merge_with_existing(self, df: pd.DataFrame, dataset: str, path: Path) -> pd.DataFrame:
        try:
            existing = pq.read_table(path).to_pandas()
        except Exception as exc:  # corrupt/partial file: prefer new data over crashing
            log.error("could not read %s for merge (%s); overwriting instead", path.name, exc)
            return df
        # New rows last so keep="last" in _dedupe prefers freshly fetched values.
        return pd.concat([existing, df], ignore_index=True)

    def _dedupe(self, df: pd.DataFrame, dataset: str) -> pd.DataFrame:
        rule = S.RULES.get(dataset)
        if not rule or not rule.unique_on:
            return df
        keys = [c for c in rule.unique_on if c in df.columns]
        if not keys:
            return df
        before = len(df)
        df = df.drop_duplicates(subset=keys, keep="last")
        if len(df) != before:
            log.debug("%s: dropped %d duplicate row(s) on %s", dataset, before - len(df), keys)
        return df.reset_index(drop=True)

    @staticmethod
    def _default_sort(dataset: str) -> list[str]:
        from tickerlake.storage import paths as P

        return {
            P.OHLCV: ["symbol", "date"],
            P.OPTIONS_CHAINS: ["symbol", "expiration", "option_type", "strike"],
            P.INTRADAY_BARS: ["symbol", "datetime"],
            P.BOOK_SNAPSHOTS: ["symbol", "datetime"],
            P.OPTIONS_GREEKS: ["symbol", "expiration", "option_type", "strike"],
            P.OPTIONS_FLOW: ["symbol"],
            P.SHORT_VOLUME: ["symbol", "date"],
            P.EARNINGS: ["symbol", "period"],
            P.MEMBERSHIP: ["symbol", "start_date"],
            P.UNIVERSE_HISTORY: ["observed_date", "symbol"],
            P.FILINGS_TEXT: ["filing_date", "cik"],
            P.FILINGS_FACTS: ["cik", "concept", "end_date"],
            P.MACRO_SERIES: ["series_id", "date"],
            P.NEWS_EVENTS: ["date", "symbol"],
            P.QUALITY: ["stage", "metric"],
        }.get(dataset, [])

    def _to_table(self, df: pd.DataFrame, dataset: str) -> pa.Table:
        schema = S.SCHEMAS.get(dataset)
        if schema is None:
            return pa.Table.from_pandas(df, preserve_index=False)

        arrays = []
        for field in schema:
            series = (
                df[field.name]
                if field.name in df.columns
                else pd.Series([None] * len(df), index=df.index, dtype="object")
            )
            arrays.append(_cast_series(series, field.type, f"{dataset}.{field.name}"))
        return pa.Table.from_arrays(arrays, schema=schema)

    def _atomic_write(self, table: pa.Table, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        try:
            pq.write_table(
                table,
                tmp,
                compression=self.compression,
                compression_level=self.compression_level,
                use_dictionary=True,
                write_statistics=True,
                # 128k rows keeps row groups big enough for good compression while
                # staying granular enough for statistics-based skipping to pay off.
                row_group_size=131_072,
            )
            size = tmp.stat().st_size
            os.replace(tmp, path)  # atomic on both Windows and POSIX
            return size
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass


# --------------------------------------------------------------- casting


def _cast_series(series: pd.Series, typ: pa.DataType, label: str) -> pa.Array:
    """Best-effort coercion of a pandas Series to an Arrow type.

    Anything uncoercible becomes null rather than raising: one malformed cell
    from an unofficial API should not discard an otherwise good 800k-row day.
    """
    try:
        if pa.types.is_date32(typ):
            dt = pd.to_datetime(series, errors="coerce")
            if isinstance(dt.dtype, pd.DatetimeTZDtype):
                dt = dt.dt.tz_localize(None)
            return pa.array(dt.dt.date, type=pa.date32(), from_pandas=True)

        if pa.types.is_timestamp(typ):
            dt = pd.to_datetime(series, errors="coerce", utc=True)
            return pa.array(dt, type=typ, from_pandas=True)

        if pa.types.is_integer(typ):
            num = pd.to_numeric(series, errors="coerce")
            return pa.array(num.astype("Int64"), type=typ, from_pandas=True)

        if pa.types.is_floating(typ):
            num = pd.to_numeric(series, errors="coerce")
            return pa.array(num.astype("float64"), type=typ, from_pandas=True)

        if pa.types.is_boolean(typ):
            return pa.array(series.astype("boolean"), type=typ, from_pandas=True)

        if pa.types.is_string(typ):
            obj = series.astype("object")
            obj = obj.map(
                lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
            )
            return pa.array(obj, type=pa.string(), from_pandas=True)

        return pa.array(series, type=typ, from_pandas=True)

    except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError, TypeError) as exc:
        log.warning("cast failed for %s -> %s (%s); writing nulls", label, typ, exc)
        return pa.nulls(len(series), type=typ)
