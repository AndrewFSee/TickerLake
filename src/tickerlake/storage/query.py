"""DuckDB query layer over the Parquet lake.

No database server: DuckDB reads the Parquet files directly, so the lake stays a
pile of portable files you can copy, sync, or hand to someone else.

Partition pruning
-----------------
Partition columns (``snapshot_date``, ``year``, ``month``, ``date``) are stored
*both* in the directory name and as real columns in the file, so every file is
self-describing. That means Hive partitioning is deliberately switched **off**
when reading (it would collide with the in-file column). To still get pruning,
the typed helpers below build explicit globs covering only the partitions in the
requested range instead of scanning the dataset. Use them rather than the
convenience views whenever you have a date filter.

Survivorship bias
-----------------
``members_on()`` and ``pit_ohlcv()`` are the survivorship-safe entry points: they
resolve the universe from the point-in-time membership table, never from today's
constituents. ``ohlcv`` on its own is the raw table and will happily give you
delisted names -- which is correct, that data is retained on purpose.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa

from tickerlake.storage import paths as P
from tickerlake.storage import schemas as S

log = logging.getLogger(__name__)

# end_date IS NULL means "still a member"; this sentinel keeps range predicates simple.
OPEN_END_SENTINEL = "9999-12-31"


class LakeQuery:
    """Thin, well-typed wrapper around a DuckDB connection over the lake."""

    def __init__(self, data_root: Path, read_only: bool = True, threads: int | None = None):
        self.root = Path(data_root)
        self.paths = P.DatasetPaths(self.root)
        self.con = duckdb.connect(database=":memory:", read_only=False)
        if threads:
            self.con.execute(f"SET threads TO {threads}")
        self._register_views()

    # ------------------------------------------------------------- lifecycle

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> LakeQuery:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ----------------------------------------------------------------- setup

    def _dataset_has_files(self, dataset: str) -> bool:
        d = self.root / dataset
        if not d.exists():
            return False
        return any(d.rglob("*.parquet"))

    def _register_views(self) -> None:
        """Create one view per dataset, typed-empty when no files exist yet."""
        for dataset in P.DATASETS:
            try:
                if self._dataset_has_files(dataset):
                    pattern = self.paths.dataset_glob_pattern(dataset)
                    self.con.execute(
                        f"CREATE OR REPLACE VIEW {dataset} AS "
                        f"SELECT * FROM read_parquet('{pattern}', "
                        f"hive_partitioning=false, union_by_name=true)"
                    )
                else:
                    self.con.execute(
                        f"CREATE OR REPLACE VIEW {dataset} AS {_empty_select(dataset)}"
                    )
            except duckdb.Error as exc:
                log.warning("could not register view '%s': %s", dataset, exc)

        # Point-in-time membership: the survivorship-safe source of truth.
        try:
            self.con.execute(
                f"""
                CREATE OR REPLACE VIEW membership_pit AS
                SELECT
                    symbol,
                    index_name,
                    start_date,
                    COALESCE(end_date, DATE '{OPEN_END_SENTINEL}') AS end_date_eff,
                    end_date IS NULL                               AS is_current,
                    company_name, gics_sector, gics_sub_industry, cik,
                    reason_added, reason_removed,
                    suspected_delisted, last_data_date, consecutive_empty_runs
                FROM membership
                """
            )
        except duckdb.Error as exc:
            log.warning("could not register membership_pit view: %s", exc)

    def refresh(self) -> None:
        """Re-scan the filesystem. Call after a write if the connection is long-lived."""
        self._register_views()

    # ------------------------------------------------------------------- sql

    def sql(self, query: str, params: list[Any] | None = None) -> pd.DataFrame:
        """Run arbitrary SQL and return a DataFrame."""
        return self.con.execute(query, params or []).df()

    def arrow(self, query: str, params: list[Any] | None = None) -> pa.Table:
        """Run SQL and return Arrow, for zero-copy hand-off to ML tooling."""
        return self.con.execute(query, params or []).arrow()

    # ------------------------------------------------------------ membership

    def members_on(self, as_of: date | str, index_name: str = "SP500") -> list[str]:
        """Symbols that were in the index on ``as_of``. The anti-survivorship primitive."""
        as_of = _to_date(as_of)
        df = self.sql(
            """
            SELECT DISTINCT symbol
            FROM membership_pit
            WHERE index_name = ?
              AND start_date <= ?
              AND end_date_eff >= ?
            ORDER BY symbol
            """,
            [index_name, as_of, as_of],
        )
        return df["symbol"].tolist()

    def current_members(self, index_name: str = "SP500") -> list[str]:
        df = self.sql(
            "SELECT symbol FROM membership_pit WHERE index_name = ? AND is_current ORDER BY symbol",
            [index_name],
        )
        return df["symbol"].tolist()

    def tracked_symbols(
        self, index_name: str = "SP500", since: date | str | None = None
    ) -> list[str]:
        """Every symbol ever tracked -- current and historical members alike.

        This is the set we keep collecting data for. Nothing is ever removed from it.
        """
        query = "SELECT DISTINCT symbol FROM membership_pit WHERE index_name = ?"
        params: list[Any] = [index_name]
        if since is not None:
            query += " AND end_date_eff >= ?"
            params.append(_to_date(since))
        return self.sql(query + " ORDER BY symbol", params)["symbol"].tolist()

    def membership_changes(
        self, start: date | str, end: date | str, index_name: str = "SP500"
    ) -> pd.DataFrame:
        """Additions and removals within a window."""
        start, end = _to_date(start), _to_date(end)
        return self.sql(
            """
            SELECT symbol, 'added' AS change_type, start_date AS event_date, reason_added AS reason
            FROM membership_pit
            WHERE index_name = ? AND start_date BETWEEN ? AND ?
            UNION ALL
            SELECT symbol, 'removed' AS change_type, end_date_eff AS event_date, reason_removed AS reason
            FROM membership_pit
            WHERE index_name = ? AND NOT is_current AND end_date_eff BETWEEN ? AND ?
            ORDER BY event_date, symbol
            """,
            [index_name, start, end, index_name, start, end],
        )

    def suspected_delistings(self, index_name: str = "SP500") -> pd.DataFrame:
        return self.sql(
            """
            SELECT symbol, last_data_date, consecutive_empty_runs, is_current, end_date_eff
            FROM membership_pit
            WHERE index_name = ? AND suspected_delisted
            ORDER BY consecutive_empty_runs DESC, symbol
            """,
            [index_name],
        )

    # ----------------------------------------------------------------- ohlcv

    def ohlcv(
        self,
        symbols: Iterable[str] | None = None,
        start: date | str | None = None,
        end: date | str | None = None,
    ) -> pd.DataFrame:
        """Raw OHLCV. Includes delisted names -- that retention is intentional."""
        pattern = self._ohlcv_globs(start, end)
        if pattern is None:
            return _empty_df(P.OHLCV)

        where, params = ["1=1"], []
        if symbols is not None:
            syms = [s.upper() for s in symbols]
            if not syms:
                return _empty_df(P.OHLCV)
            where.append(f"symbol IN ({','.join('?' * len(syms))})")
            params.extend(syms)
        if start is not None:
            where.append("date >= ?")
            params.append(_to_date(start))
        if end is not None:
            where.append("date <= ?")
            params.append(_to_date(end))

        return self.sql(
            f"SELECT * FROM read_parquet({pattern}, hive_partitioning=false, union_by_name=true) "
            f"WHERE {' AND '.join(where)} ORDER BY symbol, date",
            params,
        )

    def pit_ohlcv(
        self,
        start: date | str,
        end: date | str,
        index_name: str = "SP500",
    ) -> pd.DataFrame:
        """Survivorship-bias-free OHLCV panel.

        Each row is kept only if that symbol was an index member *on that row's own
        date*, so a name added in 2018 contributes nothing to 2015 and a name
        removed in 2020 still contributes its pre-removal history.
        """
        start, end = _to_date(start), _to_date(end)
        pattern = self._ohlcv_globs(start, end)
        if pattern is None:
            return _empty_df(P.OHLCV)

        return self.sql(
            f"""
            SELECT o.*
            FROM read_parquet({pattern}, hive_partitioning=false, union_by_name=true) o
            JOIN membership_pit m
              ON m.symbol = o.symbol
             AND m.index_name = ?
             AND o.date BETWEEN m.start_date AND m.end_date_eff
            WHERE o.date BETWEEN ? AND ?
            ORDER BY o.date, o.symbol
            """,
            [index_name, start, end],
        )

    def _ohlcv_globs(self, start: date | str | None, end: date | str | None) -> str | None:
        """Explicit per-month globs so DuckDB opens only the relevant partitions."""
        base = self.root / P.OHLCV
        if not base.exists():
            return None
        if start is None or end is None:
            files = sorted(base.rglob("*.parquet"))
            return _sql_str_list([f.as_posix() for f in files]) if files else None

        s, e = _to_date(start), _to_date(end)
        globs: list[str] = []
        y, m = s.year, s.month
        while (y, m) <= (e.year, e.month):
            part = base / f"year={y:04d}" / f"month={m:02d}"
            if part.exists():
                globs.append((part / "*.parquet").as_posix())
            m += 1
            if m > 12:
                y, m = y + 1, 1
        return _sql_str_list(globs) if globs else None

    # --------------------------------------------------------------- options

    def options_snapshot(
        self,
        snapshot_date: date | str,
        symbols: Iterable[str] | None = None,
        min_dte: int | None = None,
        max_dte: int | None = None,
        option_type: str | None = None,
    ) -> pd.DataFrame:
        """A single day's option chains, pruned to that day's partition."""
        snap = _to_date(snapshot_date)
        part = self.paths.options_partition(snap)
        if not part.exists():
            return _empty_df(P.OPTIONS_CHAINS)

        where, params = ["1=1"], []
        if symbols is not None:
            syms = [s.upper() for s in symbols]
            if not syms:
                return _empty_df(P.OPTIONS_CHAINS)
            where.append(f"symbol IN ({','.join('?' * len(syms))})")
            params.extend(syms)
        if min_dte is not None:
            where.append("dte >= ?")
            params.append(min_dte)
        if max_dte is not None:
            where.append("dte <= ?")
            params.append(max_dte)
        if option_type is not None:
            where.append("option_type = ?")
            params.append(option_type.lower())

        pattern = (part / "*.parquet").as_posix()
        return self.sql(
            f"SELECT * FROM read_parquet('{pattern}', hive_partitioning=false, union_by_name=true) "
            f"WHERE {' AND '.join(where)} ORDER BY symbol, expiration, option_type, strike",
            params,
        )

    def options_history(
        self,
        symbol: str,
        start: date | str,
        end: date | str,
        max_dte: int | None = None,
    ) -> pd.DataFrame:
        """One symbol's chain snapshots across a date range.

        This is the dataset that only exists because we snapshot daily -- Yahoo
        exposes the live chain only, so history is what we accumulate ourselves.
        """
        s, e = _to_date(start), _to_date(end)
        globs = []
        base = self.root / P.OPTIONS_CHAINS
        if base.exists():
            for part in sorted(base.glob("snapshot_date=*")):
                try:
                    d = date.fromisoformat(part.name.split("=", 1)[1])
                except (ValueError, IndexError):
                    continue
                if s <= d <= e:
                    globs.append((part / "*.parquet").as_posix())
        if not globs:
            return _empty_df(P.OPTIONS_CHAINS)

        where = ["symbol = ?"]
        params: list[Any] = [symbol.upper()]
        if max_dte is not None:
            where.append("dte <= ?")
            params.append(max_dte)

        return self.sql(
            f"SELECT * FROM read_parquet({_sql_str_list(globs)}, hive_partitioning=false, "
            f"union_by_name=true) WHERE {' AND '.join(where)} "
            f"ORDER BY snapshot_date, expiration, option_type, strike",
            params,
        )

    def options_coverage(self) -> pd.DataFrame:
        """Per-snapshot-date row and symbol counts. The first thing to check after a run."""
        base = self.root / P.OPTIONS_CHAINS
        if not base.exists() or not any(base.rglob("*.parquet")):
            return pd.DataFrame(columns=["snapshot_date", "symbols", "rows", "expirations"])
        return self.sql(
            """
            SELECT snapshot_date,
                   COUNT(DISTINCT symbol)     AS symbols,
                   COUNT(*)                   AS rows,
                   COUNT(DISTINCT expiration) AS expirations
            FROM options_chains
            GROUP BY snapshot_date
            ORDER BY snapshot_date DESC
            """
        )

    # ------------------------------------------------------------- utilities

    def dataset_stats(self) -> pd.DataFrame:
        """File count and on-disk size per dataset."""
        rows = []
        for dataset in P.DATASETS:
            files = list((self.root / dataset).rglob("*.parquet"))
            rows.append(
                {
                    "dataset": dataset,
                    "files": len(files),
                    "size_mb": round(sum(f.stat().st_size for f in files) / 1_048_576, 2),
                }
            )
        return pd.DataFrame(rows)

    def table_counts(self) -> pd.DataFrame:
        rows = []
        for dataset in P.DATASETS:
            try:
                n = self.con.execute(f"SELECT COUNT(*) FROM {dataset}").fetchone()[0]
            except duckdb.Error:
                n = None
            rows.append({"dataset": dataset, "rows": n})
        return pd.DataFrame(rows)


@contextmanager
def open_lake(data_root: Path, **kwargs):
    q = LakeQuery(data_root, **kwargs)
    try:
        yield q
    finally:
        q.close()


# ----------------------------------------------------------------- helpers


def _to_date(value: date | str | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _sql_str_list(items: list[str]) -> str:
    """Render a Python list of paths as a DuckDB list literal."""
    escaped = [i.replace("'", "''") for i in items]
    return "[" + ", ".join(f"'{i}'" for i in escaped) + "]"


_PA_TO_DUCKDB = {
    "string": "VARCHAR",
    "date32[day]": "DATE",
    "bool": "BOOLEAN",
    "int32": "INTEGER",
    "int64": "BIGINT",
    "double": "DOUBLE",
}


def _duckdb_type(typ: pa.DataType) -> str:
    if pa.types.is_timestamp(typ):
        return "TIMESTAMP WITH TIME ZONE" if typ.tz else "TIMESTAMP"
    return _PA_TO_DUCKDB.get(str(typ), "VARCHAR")


def _empty_select(dataset: str) -> str:
    """A typed, zero-row SELECT so queries work against a lake with no data yet."""
    schema = S.SCHEMAS.get(dataset)
    if schema is None:
        return "SELECT NULL WHERE 1=0"
    cols = ", ".join(f"CAST(NULL AS {_duckdb_type(f.type)}) AS {f.name}" for f in schema)
    return f"SELECT {cols} WHERE 1=0"


def _empty_df(dataset: str) -> pd.DataFrame:
    schema = S.SCHEMAS.get(dataset)
    return pd.DataFrame(columns=list(schema.names) if schema else [])
