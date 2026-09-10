"""PyArrow schemas and pre-write validation.

Every dataset has an explicit schema. Writes are cast to it before hitting disk,
so a source that silently changes a dtype (yfinance returns object-dtype columns
more often than you would like) cannot corrupt a partition or break a later
DuckDB scan with a schema mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pyarrow as pa

from tickerlake.storage import paths as P

# --------------------------------------------------------------------- schemas

OHLCV_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("date", pa.date32(), nullable=False),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("adj_close", pa.float64()),
        pa.field("volume", pa.int64()),
        pa.field("dividends", pa.float64()),
        pa.field("stock_splits", pa.float64()),
        # True where yfinance's repair pass corrected a bad Yahoo tick.
        # Worth keeping: a corrected bar is not the same evidence as a clean one.
        pa.field("repaired", pa.bool_()),
        pa.field("source", pa.string(), nullable=False),
        pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

OPTIONS_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("snapshot_date", pa.date32(), nullable=False),
        pa.field("expiration", pa.date32(), nullable=False),
        pa.field("option_type", pa.string(), nullable=False),  # 'call' | 'put'
        pa.field("strike", pa.float64(), nullable=False),
        pa.field("contract_symbol", pa.string()),
        pa.field("last_price", pa.float64()),
        pa.field("bid", pa.float64()),
        pa.field("ask", pa.float64()),
        pa.field("change", pa.float64()),
        pa.field("percent_change", pa.float64()),
        pa.field("volume", pa.int64()),
        pa.field("open_interest", pa.int64()),
        pa.field("implied_volatility", pa.float64()),
        pa.field("in_the_money", pa.bool_()),
        pa.field("contract_size", pa.string()),
        pa.field("currency", pa.string()),
        pa.field("last_trade_date", pa.timestamp("us", tz="UTC")),
        # Denormalised for convenience: recomputing DTE in every query is tedious
        # and this costs 4 bytes that compress to almost nothing.
        pa.field("dte", pa.int32()),
        pa.field("underlying_price", pa.float64()),
        pa.field("source", pa.string(), nullable=False),
        pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

MEMBERSHIP_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("index_name", pa.string(), nullable=False),
        pa.field("start_date", pa.date32(), nullable=False),
        # NULL end_date == still a member. The DuckDB layer exposes a view that
        # coalesces this to 9999-12-31 so range predicates stay simple.
        pa.field("end_date", pa.date32(), nullable=True),
        pa.field("company_name", pa.string()),
        pa.field("gics_sector", pa.string()),
        pa.field("gics_sub_industry", pa.string()),
        pa.field("cik", pa.string()),
        pa.field("reason_added", pa.string()),
        pa.field("reason_removed", pa.string()),
        pa.field("source", pa.string(), nullable=False),
        pa.field("first_seen_utc", pa.timestamp("us", tz="UTC")),
        pa.field("last_seen_utc", pa.timestamp("us", tz="UTC")),
        # Silent-delisting tracking. Never used to delete, only to flag.
        pa.field("consecutive_empty_runs", pa.int32()),
        pa.field("suspected_delisted", pa.bool_()),
        pa.field("last_data_date", pa.date32(), nullable=True),
    ]
)

UNIVERSE_HISTORY_SCHEMA = pa.schema(
    [
        pa.field("observed_date", pa.date32(), nullable=False),
        pa.field("index_name", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("change_type", pa.string(), nullable=False),  # present|added|removed
        pa.field("source", pa.string(), nullable=False),
        pa.field("observed_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

FILINGS_TEXT_SCHEMA = pa.schema(
    [
        pa.field("accession_number", pa.string(), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("cik", pa.string(), nullable=False),
        pa.field("company_name", pa.string()),
        pa.field("form_type", pa.string(), nullable=False),
        pa.field("filing_date", pa.date32(), nullable=False),
        pa.field("report_date", pa.date32(), nullable=True),
        pa.field("primary_doc_url", pa.string()),
        pa.field("text", pa.string()),
        pa.field("text_chars", pa.int64()),
        pa.field("truncated", pa.bool_()),
        # Hook for the future embedding step (LanceDB/Chroma). Left NULL by the
        # fetcher; an embedding job fills it in and can then filter on it.
        pa.field("embedding_status", pa.string()),
        pa.field("embedding_model", pa.string()),
        pa.field("source", pa.string(), nullable=False),
        pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

FILINGS_FACTS_SCHEMA = pa.schema(
    [
        pa.field("cik", pa.string(), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("taxonomy", pa.string()),
        pa.field("concept", pa.string(), nullable=False),
        pa.field("unit", pa.string()),
        pa.field("value", pa.float64()),
        pa.field("start_date", pa.date32(), nullable=True),
        pa.field("end_date", pa.date32(), nullable=True),
        pa.field("fiscal_year", pa.int32()),
        pa.field("fiscal_period", pa.string()),
        pa.field("form_type", pa.string()),
        pa.field("filed_date", pa.date32(), nullable=True),
        pa.field("accession_number", pa.string()),
        pa.field("frame", pa.string()),
        pa.field("source", pa.string(), nullable=False),
        pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

MACRO_SCHEMA = pa.schema(
    [
        pa.field("series_id", pa.string(), nullable=False),
        pa.field("date", pa.date32(), nullable=False),
        pa.field("value", pa.float64()),
        pa.field("realtime_start", pa.date32(), nullable=True),
        pa.field("realtime_end", pa.date32(), nullable=True),
        pa.field("title", pa.string()),
        pa.field("units", pa.string()),
        pa.field("frequency", pa.string()),
        pa.field("seasonal_adjustment", pa.string()),
        pa.field("source", pa.string(), nullable=False),
        pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

NEWS_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("published_at", pa.timestamp("us", tz="UTC")),
        pa.field("date", pa.date32(), nullable=False),
        pa.field("symbol", pa.string(), nullable=True),
        pa.field("title", pa.string()),
        pa.field("summary", pa.string()),
        pa.field("url", pa.string()),
        pa.field("domain", pa.string()),
        pa.field("language", pa.string()),
        pa.field("country", pa.string()),
        pa.field("theme", pa.string()),
        pa.field("tone", pa.float64()),
        pa.field("category", pa.string()),
        pa.field("embedding_status", pa.string()),
        pa.field("source", pa.string(), nullable=False),
        pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

QUALITY_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("run_date", pa.date32(), nullable=False),
        pa.field("stage", pa.string(), nullable=False),
        pa.field("dataset", pa.string()),
        pa.field("metric", pa.string(), nullable=False),
        pa.field("value_num", pa.float64()),
        pa.field("value_text", pa.string()),
        pa.field("passed", pa.bool_()),
        pa.field("recorded_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

SCHEMAS: dict[str, pa.Schema] = {
    P.OHLCV: OHLCV_SCHEMA,
    P.OPTIONS_CHAINS: OPTIONS_SCHEMA,
    P.MEMBERSHIP: MEMBERSHIP_SCHEMA,
    P.UNIVERSE_HISTORY: UNIVERSE_HISTORY_SCHEMA,
    P.FILINGS_TEXT: FILINGS_TEXT_SCHEMA,
    P.FILINGS_FACTS: FILINGS_FACTS_SCHEMA,
    P.MACRO_SERIES: MACRO_SCHEMA,
    P.NEWS_EVENTS: NEWS_SCHEMA,
    P.QUALITY: QUALITY_SCHEMA,
}


# ------------------------------------------------------------------ validation


class ValidationError(ValueError):
    """Raised when a frame fails validation and the caller asked to be strict."""


@dataclass
class ValidationRule:
    """Declarative expectations checked before every write."""

    min_rows: int = 1
    max_rows: int | None = None
    non_null: tuple[str, ...] = ()
    unique_on: tuple[str, ...] = ()
    # Columns where an all-null result means the source returned junk, even
    # though individual nulls are legitimate.
    not_all_null: tuple[str, ...] = ()
    positive: tuple[str, ...] = ()
    non_negative: tuple[str, ...] = ()


@dataclass
class ValidationReport:
    dataset: str
    rows: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        if self.ok and not self.warnings:
            return f"{self.dataset}: {self.rows} rows, validation passed"
        bits = [f"{self.dataset}: {self.rows} rows"]
        if self.errors:
            bits.append(f"{len(self.errors)} error(s): " + "; ".join(self.errors[:5]))
        if self.warnings:
            bits.append(f"{len(self.warnings)} warning(s): " + "; ".join(self.warnings[:5]))
        return " | ".join(bits)


RULES: dict[str, ValidationRule] = {
    P.OHLCV: ValidationRule(
        min_rows=1,
        non_null=("symbol", "date", "source", "ingested_at"),
        unique_on=("symbol", "date"),
        not_all_null=("close",),
        non_negative=("volume",),
    ),
    P.OPTIONS_CHAINS: ValidationRule(
        min_rows=1,
        non_null=("symbol", "snapshot_date", "expiration", "option_type", "strike"),
        unique_on=("symbol", "expiration", "option_type", "strike"),
        not_all_null=("strike",),
        positive=("strike",),
        non_negative=("volume", "open_interest"),
    ),
    P.MEMBERSHIP: ValidationRule(
        # An S&P 500 membership table with under 100 rows means a broken parse,
        # not a small index.
        min_rows=100,
        non_null=("symbol", "index_name", "start_date", "source"),
        unique_on=("symbol", "index_name", "start_date"),
    ),
    P.UNIVERSE_HISTORY: ValidationRule(
        min_rows=1,
        non_null=("observed_date", "symbol", "change_type", "source"),
    ),
    P.FILINGS_TEXT: ValidationRule(
        min_rows=0,
        non_null=("accession_number", "cik", "form_type", "filing_date"),
        unique_on=("accession_number",),
    ),
    P.FILINGS_FACTS: ValidationRule(
        min_rows=0,
        non_null=("cik", "concept"),
    ),
    P.MACRO_SERIES: ValidationRule(
        min_rows=0,
        non_null=("series_id", "date"),
        unique_on=("series_id", "date"),
    ),
    P.NEWS_EVENTS: ValidationRule(
        min_rows=0,
        non_null=("event_id", "date"),
        unique_on=("event_id",),
    ),
    P.QUALITY: ValidationRule(min_rows=0, non_null=("run_id", "stage", "metric")),
}


def validate_frame(df, dataset: str, strict: bool = False) -> ValidationReport:
    """Check a pandas DataFrame against its dataset's schema and rules.

    Column-set mismatches and rule breaches become errors; recoverable oddities
    (duplicates that we de-duplicate on write, optional columns that are absent)
    become warnings. Returns a report rather than raising, unless ``strict``.
    """
    rule = RULES.get(dataset, ValidationRule())
    schema = SCHEMAS.get(dataset)
    report = ValidationReport(dataset=dataset, rows=len(df))

    if len(df) < rule.min_rows:
        report.errors.append(f"row count {len(df)} below minimum {rule.min_rows}")
    if rule.max_rows is not None and len(df) > rule.max_rows:
        report.errors.append(f"row count {len(df)} above maximum {rule.max_rows}")

    if schema is not None:
        expected = set(schema.names)
        actual = set(df.columns)
        missing = expected - actual
        extra = actual - expected
        # Nullable columns may legitimately be absent; the writer fills them.
        hard_missing = {name for name in missing if not schema.field(name).nullable}
        if hard_missing:
            report.errors.append(f"missing required columns: {sorted(hard_missing)}")
        soft_missing = missing - hard_missing
        if soft_missing:
            report.warnings.append(
                f"optional columns absent (will be null): {sorted(soft_missing)}"
            )
        if extra:
            report.warnings.append(f"unexpected columns dropped: {sorted(extra)}")

    if len(df) == 0:
        return _finish(report, strict)

    for col in rule.non_null:
        if col in df.columns:
            n = int(df[col].isna().sum())
            if n:
                report.errors.append(f"column '{col}' has {n} null(s) but must not")

    for col in rule.not_all_null:
        if col in df.columns and bool(df[col].isna().all()):
            report.errors.append(
                f"column '{col}' is entirely null - source likely returned no data"
            )

    # Range checks coerce first: a source that hands back strings where numbers
    # belong is exactly the case validation exists to catch, so comparing raw
    # would crash on the very input we most need a report about.
    for col in rule.positive:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            bad = int((values <= 0).sum())
            if bad:
                report.errors.append(f"column '{col}' has {bad} non-positive value(s)")

    for col in rule.non_negative:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            bad = int((values < 0).sum())
            if bad:
                report.errors.append(f"column '{col}' has {bad} negative value(s)")

    if rule.unique_on and all(c in df.columns for c in rule.unique_on):
        dupes = int(df.duplicated(subset=list(rule.unique_on)).sum())
        if dupes:
            report.warnings.append(
                f"{dupes} duplicate row(s) on {list(rule.unique_on)} - de-duplicated on write"
            )

    return _finish(report, strict)


def _finish(report: ValidationReport, strict: bool) -> ValidationReport:
    if strict and not report.ok:
        raise ValidationError(report.summary())
    return report
