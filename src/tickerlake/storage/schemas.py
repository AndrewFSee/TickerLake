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

INTRADAY_BARS_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("datetime", pa.timestamp("us", tz="UTC"), nullable=False),
        # Session date in US Eastern. A 16:00 ET bar is 20:00 UTC, so deriving
        # this from the UTC date would misfile the afternoon session.
        pa.field("date", pa.date32(), nullable=False),
        pa.field("interval", pa.string(), nullable=False),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.int64()),
        pa.field("source", pa.string(), nullable=False),
        pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)


def _book_snapshot_schema(depth: int = 5) -> pa.Schema:
    """L2 snapshots. Level count is configurable, so the schema is generated."""
    fields = [
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("datetime", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("date", pa.date32(), nullable=False),
        pa.field("best_bid", pa.float64()),
        pa.field("best_ask", pa.float64()),
        pa.field("spread", pa.float64()),
        pa.field("mid", pa.float64()),
        pa.field("microprice", pa.float64()),
        pa.field("imbalance_l1", pa.float64()),
        pa.field("bid_depth", pa.float64()),
        pa.field("ask_depth", pa.float64()),
        # Time-weighted across the minute rather than sampled at its boundary:
        # a single instant can land on a momentary spread that never held.
        pa.field("twa_spread", pa.float64()),
        pa.field("twa_mid", pa.float64()),
        pa.field("twa_imbalance", pa.float64()),
        pa.field("quoted_seconds", pa.float64()),
        pa.field("n_updates", pa.int32()),
        pa.field("n_trades", pa.int32()),
        pa.field("trade_volume", pa.int64()),
        pa.field("trade_notional", pa.float64()),
    ]
    for i in range(1, depth + 1):
        fields += [
            pa.field(f"bid_px_{i}", pa.float64()),
            pa.field(f"bid_sz_{i}", pa.float64()),
            pa.field(f"ask_px_{i}", pa.float64()),
            pa.field(f"ask_sz_{i}", pa.float64()),
        ]
    fields += [
        pa.field("source", pa.string(), nullable=False),
        pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
    return pa.schema(fields)


BOOK_SNAPSHOTS_SCHEMA = _book_snapshot_schema(5)

OPTIONS_GREEKS_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("snapshot_date", pa.date32(), nullable=False),
        pa.field("expiration", pa.date32(), nullable=False),
        pa.field("option_type", pa.string(), nullable=False),
        pa.field("strike", pa.float64(), nullable=False),
        pa.field("contract_symbol", pa.string()),
        pa.field("dte", pa.int32()),
        pa.field("underlying_price", pa.float64()),
        pa.field("mid_price", pa.float64()),
        pa.field("intrinsic_value", pa.float64()),
        pa.field("time_value", pa.float64()),
        # Our own solved IV. NULL wherever the quote could not support one --
        # that absence is the point, versus a vendor field that always has a value.
        pa.field("iv", pa.float64()),
        pa.field("iv_uncertainty", pa.float64()),
        # Yahoo's field, retained for comparison rather than for use.
        pa.field("iv_vendor", pa.float64()),
        pa.field("delta", pa.float64()),
        pa.field("gamma", pa.float64()),
        pa.field("vega", pa.float64()),
        pa.field("theta", pa.float64()),
        pa.field("rho", pa.float64()),
        pa.field("moneyness", pa.float64()),
        pa.field("log_moneyness", pa.float64()),
        pa.field("risk_free_rate", pa.float64()),
        pa.field("dividend_yield", pa.float64()),
        pa.field("quality_flags", pa.string()),
        pa.field("iv_usable", pa.bool_()),
        pa.field("volume", pa.int64()),
        pa.field("open_interest", pa.int64()),
        pa.field("source", pa.string(), nullable=False),
        pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

OPTIONS_FLOW_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("snapshot_date", pa.date32(), nullable=False),
        pa.field("call_volume", pa.int64()),
        pa.field("put_volume", pa.int64()),
        pa.field("put_call_volume_ratio", pa.float64()),
        pa.field("call_open_interest", pa.int64()),
        pa.field("put_open_interest", pa.int64()),
        pa.field("put_call_oi_ratio", pa.float64()),
        pa.field("atm_iv_30d", pa.float64()),
        pa.field("skew_25d", pa.float64()),
        pa.field("contracts", pa.int32()),
        pa.field("expirations", pa.int32()),
        pa.field("underlying_price", pa.float64()),
        pa.field("source", pa.string(), nullable=False),
        pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

SHORT_VOLUME_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("date", pa.date32(), nullable=False),
        pa.field("short_volume", pa.float64()),
        pa.field("short_exempt_volume", pa.float64()),
        pa.field("total_volume", pa.float64()),
        # The headline feature: short volume as a share of total reported volume.
        pa.field("short_volume_ratio", pa.float64()),
        pa.field("market", pa.string()),
        pa.field("source", pa.string(), nullable=False),
        pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

EARNINGS_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("record_type", pa.string(), nullable=False),  # surprise|calendar|recommendation
        pa.field("period", pa.date32(), nullable=False),
        pa.field("fiscal_year", pa.int32()),
        pa.field("fiscal_quarter", pa.int32()),
        pa.field("eps_estimate", pa.float64()),
        pa.field("eps_actual", pa.float64()),
        pa.field("eps_surprise", pa.float64()),
        pa.field("eps_surprise_pct", pa.float64()),
        pa.field("revenue_estimate", pa.float64()),
        pa.field("revenue_actual", pa.float64()),
        pa.field("report_hour", pa.string()),
        # When the result actually became public, from the SEC 8-K carrying
        # Item 2.02 (Results of Operations and Financial Condition).
        #
        # `period` is the fiscal period end and says nothing about when the
        # number was known: Apple's June quarter was announced on 30 July, a
        # month later. Keying a model on `period` leaks the surprise into the
        # weeks before it existed. This is the field to filter on.
        pa.field("announcement_date", pa.date32(), nullable=True),
        pa.field("announcement_accession", pa.string()),
        pa.field("strong_buy", pa.int32()),
        pa.field("buy", pa.int32()),
        pa.field("hold", pa.int32()),
        pa.field("sell", pa.int32()),
        pa.field("strong_sell", pa.int32()),
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

FACTORS_SCHEMA = pa.schema(
    [
        pa.field("date", pa.date32(), nullable=False),
        # FF3 and FF5 both publish Mkt-RF/SMB/HML but construct them differently,
        # so the set is part of the key rather than a label.
        pa.field("factor_set", pa.string(), nullable=False),
        pa.field("factor", pa.string(), nullable=False),
        # Daily return in percent, as published.
        pa.field("value", pa.float64()),
        pa.field("source", pa.string(), nullable=False),
        pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

YIELD_CURVE_SCHEMA = pa.schema(
    [
        pa.field("date", pa.date32(), nullable=False),
        pa.field("tenor", pa.string(), nullable=False),
        # Numeric maturity, so the curve can be interpolated without parsing labels.
        pa.field("tenor_years", pa.float64(), nullable=False),
        pa.field("yield_pct", pa.float64()),
        pa.field("source", pa.string(), nullable=False),
        pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

COT_SCHEMA = pa.schema(
    [
        pa.field("report_date", pa.date32(), nullable=False),
        pa.field("market", pa.string(), nullable=False),
        pa.field("exchange", pa.string()),
        pa.field("contract_code", pa.string()),
        pa.field("open_interest", pa.float64()),
        # Trader categories from the CFTC financial futures report.
        pa.field("dealer_long", pa.float64()),
        pa.field("dealer_short", pa.float64()),
        pa.field("asset_mgr_long", pa.float64()),
        pa.field("asset_mgr_short", pa.float64()),
        pa.field("lev_money_long", pa.float64()),
        pa.field("lev_money_short", pa.float64()),
        pa.field("other_rept_long", pa.float64()),
        pa.field("other_rept_short", pa.float64()),
        pa.field("nonrept_long", pa.float64()),
        pa.field("nonrept_short", pa.float64()),
        # Net positioning is the feature people actually use; precomputed so the
        # sign convention is fixed once rather than re-derived per query.
        pa.field("asset_mgr_net", pa.float64()),
        pa.field("lev_money_net", pa.float64()),
        pa.field("dealer_net", pa.float64()),
        pa.field("source", pa.string(), nullable=False),
        pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

INSIDER_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("insider_name", pa.string()),
        pa.field("transaction_date", pa.date32(), nullable=True),
        pa.field("filing_date", pa.date32(), nullable=True),
        # SEC Form 4 codes: P = open-market purchase, S = sale, A = award,
        # M = option exercise. P clusters are the signal; A and M are
        # compensation events and carry little information.
        pa.field("transaction_code", pa.string()),
        # Named for what they are. Finnhub's `share` is the insider's TOTAL
        # holding after the trade, not the trade size - multiplying it by price
        # values the whole position and produces absurd notionals (Cascade
        # Investment's 114M RSG shares came out at $25bn per transaction).
        # `change` is the actual number of shares transacted.
        pa.field("shares_held_after", pa.float64()),
        pa.field("shares_transacted", pa.float64()),
        pa.field("transaction_price", pa.float64()),
        # Precomputed from shares_transacted so the correct figure is the one
        # closest to hand.
        pa.field("transaction_value", pa.float64()),
        pa.field("accession_number", pa.string()),
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
        # GDELT's native tone, roughly -100..+100. Source-specific scale.
        pa.field("tone", pa.float64()),
        # Normalised sentiment in -1..+1: GDELT tone divided by 100, Marketaux's
        # per-entity score as published.
        #
        # The bounds match but the empirical distributions do not. GDELT tone
        # rarely leaves -10..+10, so normalised it clusters within +/-0.1, while
        # Marketaux routinely reaches +/-0.8. Pooling the two without
        # standardising per source lets Marketaux dominate any model using both.
        # The transform is left honest rather than fudged to match; standardise
        # per source before combining.
        pa.field("sentiment", pa.float64()),
        # Publisher's confidence that the article is really about this symbol.
        # Marketaux supplies it directly; GDELT has no equivalent, so it stays
        # null there and salience offsets do the same job at ingest time.
        pa.field("match_score", pa.float64()),
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
    P.INTRADAY_BARS: INTRADAY_BARS_SCHEMA,
    P.BOOK_SNAPSHOTS: BOOK_SNAPSHOTS_SCHEMA,
    P.OPTIONS_GREEKS: OPTIONS_GREEKS_SCHEMA,
    P.OPTIONS_FLOW: OPTIONS_FLOW_SCHEMA,
    P.SHORT_VOLUME: SHORT_VOLUME_SCHEMA,
    P.EARNINGS: EARNINGS_SCHEMA,
    P.MEMBERSHIP: MEMBERSHIP_SCHEMA,
    P.UNIVERSE_HISTORY: UNIVERSE_HISTORY_SCHEMA,
    P.FILINGS_TEXT: FILINGS_TEXT_SCHEMA,
    P.FILINGS_FACTS: FILINGS_FACTS_SCHEMA,
    P.MACRO_SERIES: MACRO_SCHEMA,
    P.FACTORS: FACTORS_SCHEMA,
    P.YIELD_CURVE: YIELD_CURVE_SCHEMA,
    P.COT: COT_SCHEMA,
    P.INSIDER: INSIDER_SCHEMA,
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
    # (a, b) pairs where a >= b must hold, e.g. ("high", "low"). Yahoo does
    # emit bars that violate this on broken tickers, and nothing else here
    # would notice.
    at_least: tuple[tuple[str, str], ...] = ()


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
        at_least=(("high", "low"), ("high", "close"), ("close", "low")),
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
    P.INTRADAY_BARS: ValidationRule(
        min_rows=1,
        non_null=("symbol", "datetime", "date", "interval"),
        unique_on=("symbol", "datetime", "interval"),
        not_all_null=("close",),
        non_negative=("volume",),
        at_least=(("high", "low"),),
    ),
    P.BOOK_SNAPSHOTS: ValidationRule(
        min_rows=1,
        non_null=("symbol", "datetime", "date"),
        unique_on=("symbol", "datetime"),
        non_negative=("bid_depth", "ask_depth", "n_updates", "trade_volume"),
    ),
    P.OPTIONS_GREEKS: ValidationRule(
        min_rows=1,
        non_null=("symbol", "snapshot_date", "expiration", "option_type", "strike"),
        unique_on=("symbol", "expiration", "option_type", "strike"),
        positive=("strike",),
    ),
    P.OPTIONS_FLOW: ValidationRule(
        min_rows=1,
        non_null=("symbol", "snapshot_date"),
        unique_on=("symbol", "snapshot_date"),
    ),
    P.SHORT_VOLUME: ValidationRule(
        min_rows=1,
        non_null=("symbol", "date"),
        unique_on=("symbol", "date", "market"),
        non_negative=("short_volume", "total_volume"),
    ),
    P.EARNINGS: ValidationRule(
        min_rows=0,
        non_null=("symbol", "record_type", "period"),
        unique_on=("symbol", "record_type", "period"),
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
    P.FACTORS: ValidationRule(
        min_rows=1,
        non_null=("date", "factor_set", "factor"),
        unique_on=("date", "factor_set", "factor"),
    ),
    P.YIELD_CURVE: ValidationRule(
        min_rows=1,
        non_null=("date", "tenor"),
        unique_on=("date", "tenor"),
        positive=("tenor_years",),
    ),
    P.COT: ValidationRule(
        min_rows=1,
        non_null=("report_date", "market"),
        unique_on=("report_date", "market"),
        non_negative=("open_interest",),
    ),
    P.INSIDER: ValidationRule(
        min_rows=0,
        non_null=("symbol",),
        unique_on=(
            "symbol",
            "insider_name",
            "transaction_date",
            "transaction_code",
            "shares_transacted",
        ),
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

    # Cross-column ordering. Reported as a warning, not an error: these are
    # genuine source defects on a handful of broken tickers, and discarding an
    # otherwise good 500-row day over seven bad cells would lose more than it
    # protects. Flagging makes them findable.
    for higher, lower in rule.at_least:
        if higher in df.columns and lower in df.columns:
            a = pd.to_numeric(df[higher], errors="coerce")
            b = pd.to_numeric(df[lower], errors="coerce")
            bad = int((a < b).sum())
            if bad:
                report.warnings.append(
                    f"{bad} row(s) violate {higher} >= {lower} - malformed source bars"
                )

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
