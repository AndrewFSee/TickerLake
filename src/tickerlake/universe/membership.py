"""Point-in-time index membership: the survivorship-bias defence.

Invariants this module enforces
-------------------------------
1. **Nothing is ever deleted.** Removal from the index closes an interval; it
   never drops a row. Every symbol ever tracked stays queryable forever.
2. **History is never rewritten from today's list.** Today's constituents only
   ever *open* new intervals or *close* existing ones as of today. Past
   intervals are immutable.
3. **Re-entry produces a new interval**, never a mutation of the old one, so
   ``start_date <= X <= end_date`` stays a correct membership test for any X.
4. **Implausible changes fail loudly.** A parse that would remove dozens of
   symbols at once is rejected rather than applied, because the likeliest cause
   is an upstream page change, not an index event.

The table is written as Parquet (queried) plus a CSV mirror (small, diffable,
worth version-controlling so index history has a reviewable audit trail).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from tickerlake.storage import paths as P
from tickerlake.storage.writer import ParquetWriter
from tickerlake.universe.sources import LiveConstituent

log = logging.getLogger(__name__)

MEMBERSHIP_COLUMNS = [
    "symbol",
    "index_name",
    "start_date",
    "end_date",
    "company_name",
    "gics_sector",
    "gics_sub_industry",
    "cik",
    "reason_added",
    "reason_removed",
    "source",
    "first_seen_utc",
    "last_seen_utc",
    "consecutive_empty_runs",
    "suspected_delisted",
    "last_data_date",
]

# More than this many same-day removals is implausible for the S&P 500 (even a
# quarterly rebalance moves a handful) and almost certainly means a bad parse.
MAX_PLAUSIBLE_DAILY_REMOVALS = 25


@dataclass
class UniverseDiff:
    """What changed between the stored universe and today's live universe."""

    observed_date: date
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    re_entered: list[str] = field(default_factory=list)
    unchanged_count: int = 0
    live_count: int = 0
    rejected: bool = False
    rejection_reason: str | None = None

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed)

    def summary(self) -> str:
        if self.rejected:
            return f"universe diff REJECTED: {self.rejection_reason}"
        if not self.has_changes:
            return f"universe unchanged: {self.live_count} constituents"
        bits = [f"universe: {self.live_count} constituents"]
        if self.added:
            bits.append(f"+{len(self.added)} added ({', '.join(sorted(self.added)[:10])})")
        if self.removed:
            bits.append(f"-{len(self.removed)} removed ({', '.join(sorted(self.removed)[:10])})")
        if self.re_entered:
            bits.append(
                f"{len(self.re_entered)} re-entered ({', '.join(sorted(self.re_entered)[:5])})"
            )
        return " | ".join(bits)

    def to_dict(self) -> dict:
        return {
            "observed_date": self.observed_date.isoformat(),
            "added": sorted(self.added),
            "removed": sorted(self.removed),
            "re_entered": sorted(self.re_entered),
            "live_count": self.live_count,
            "unchanged_count": self.unchanged_count,
            "rejected": self.rejected,
            "rejection_reason": self.rejection_reason,
        }


class MembershipTracker:
    """Loads, updates, and persists the point-in-time membership table."""

    def __init__(
        self,
        paths: P.DatasetPaths,
        writer: ParquetWriter,
        index_name: str = "SP500",
        history_start: date | None = None,
        silent_delist_threshold: int = 5,
        post_removal_grace_days: int = 30,
    ) -> None:
        self.paths = paths
        self.writer = writer
        self.index_name = index_name
        self.history_start = history_start
        self.silent_delist_threshold = silent_delist_threshold
        self.post_removal_grace_days = post_removal_grace_days
        self._df: pd.DataFrame | None = None

    # ------------------------------------------------------------------ load

    def load(self, reload: bool = False) -> pd.DataFrame:
        if self._df is not None and not reload:
            return self._df
        path = self.paths.membership_parquet()
        if path.exists():
            df = pd.read_parquet(path)
            df = _normalize_dates(df)
        else:
            df = pd.DataFrame(columns=MEMBERSHIP_COLUMNS)
        self._df = df
        return df

    @property
    def exists(self) -> bool:
        return self.paths.membership_parquet().exists()

    def save(self) -> None:
        """Persist to Parquet and refresh the CSV mirror."""
        df = self.load()
        result = self.writer.write(
            df, P.MEMBERSHIP, self.paths.membership_parquet(), mode="overwrite"
        )
        log.info("membership saved: %s", result)

        csv_df = df.copy()
        for col in ("first_seen_utc", "last_seen_utc"):
            if col in csv_df.columns:
                csv_df[col] = pd.to_datetime(csv_df[col], errors="coerce", utc=True)
        csv_df.sort_values(["symbol", "start_date"]).to_csv(
            self.paths.membership_csv(), index=False
        )

    # ------------------------------------------------------------------ seed

    def seed(
        self,
        seed_intervals: pd.DataFrame,
        live: list[LiveConstituent] | None = None,
        force: bool = False,
    ) -> int:
        """Build the initial membership table from historical intervals.

        Refuses to run over an existing table unless ``force``, since re-seeding
        would discard locally observed history that upstream does not have.
        """
        existing = self.load()
        if len(existing) and not force:
            raise RuntimeError(
                f"membership table already has {len(existing)} rows. "
                "Re-seeding would discard locally observed history. Pass force=True to override."
            )

        df = seed_intervals.copy()
        if self.history_start is not None:
            before = len(df)
            # Keep any interval that was still open at or after the cutoff, and
            # clamp its start so the table does not claim history we did not seed.
            df = df[df["end_date"].isna() | (df["end_date"] >= self.history_start)]
            df["start_date"] = df["start_date"].map(
                lambda d: max(d, self.history_start) if pd.notna(d) else d
            )
            log.info(
                "seed filtered to history_start=%s: %d -> %d intervals",
                self.history_start,
                before,
                len(df),
            )

        now = datetime.now(UTC)
        df["index_name"] = self.index_name
        df["reason_added"] = "seed:historical"
        df["reason_removed"] = df["end_date"].map(
            lambda d: None if pd.isna(d) else "seed:index_removal"
        )
        df["source"] = "fja05680/sp500"
        df["first_seen_utc"] = now
        df["last_seen_utc"] = now
        df["consecutive_empty_runs"] = 0
        df["suspected_delisted"] = False
        # Object dtype, not datetime64: these hold plain ``datetime.date``,
        # and a datetime64 column rejects a bare date on assignment.
        df["last_data_date"] = pd.Series([None] * len(df), index=df.index, dtype="object")
        for col in ("company_name", "gics_sector", "gics_sub_industry", "cik"):
            df[col] = None

        df = df[MEMBERSHIP_COLUMNS]
        self._df = df.reset_index(drop=True)

        if live:
            self._apply_live_metadata(live)
            # The seed mirror may lag Wikipedia by days; reconcile immediately so
            # the first run starts from a correct current universe.
            diff = self.refresh(live, observed_date=date.today(), allow_large_change=True)
            log.info("seed reconciliation with live universe -> %s", diff.summary())

        self.save()
        log.info(
            "seeded %d intervals (%d unique symbols, %d currently open)",
            len(self._df),
            self._df["symbol"].nunique(),
            int(self._df["end_date"].isna().sum()),
        )
        return len(self._df)

    # --------------------------------------------------------------- refresh

    def refresh(
        self,
        live: list[LiveConstituent],
        observed_date: date | None = None,
        allow_large_change: bool = False,
    ) -> UniverseDiff:
        """Reconcile the stored universe against today's live constituents."""
        observed_date = observed_date or date.today()
        df = self.load()
        live_by_symbol = {c.symbol: c for c in live}
        live_symbols = set(live_by_symbol)

        mask_index = df["index_name"] == self.index_name if len(df) else pd.Series(dtype=bool)
        open_mask = mask_index & df["end_date"].isna() if len(df) else pd.Series(dtype=bool)
        open_symbols = set(df.loc[open_mask, "symbol"]) if len(df) else set()

        added = sorted(live_symbols - open_symbols)
        removed = sorted(open_symbols - live_symbols)

        diff = UniverseDiff(
            observed_date=observed_date,
            added=added,
            removed=removed,
            unchanged_count=len(live_symbols & open_symbols),
            live_count=len(live_symbols),
        )

        if len(removed) > MAX_PLAUSIBLE_DAILY_REMOVALS and not allow_large_change:
            diff.rejected = True
            diff.rejection_reason = (
                f"{len(removed)} removals in one observation exceeds the plausible "
                f"maximum of {MAX_PLAUSIBLE_DAILY_REMOVALS}; refusing to apply. "
                "Likely an upstream page/format change rather than an index event. "
                "Re-run with allow_large_change=True if this is genuinely correct."
            )
            log.error(diff.rejection_reason)
            return diff

        prior_observed = self._last_observed_date()
        # A symbol removed today was last confirmed present at the previous
        # observation; falling back to yesterday when we have no prior run.
        end_date = prior_observed or (observed_date - timedelta(days=1))
        if end_date >= observed_date:
            end_date = observed_date - timedelta(days=1)

        now = datetime.now(UTC)
        rows = df.to_dict("records") if len(df) else []
        closed_symbols = (
            set(df.loc[mask_index & df["end_date"].notna(), "symbol"]) if len(df) else set()
        )

        for symbol in added:
            constituent = live_by_symbol[symbol]
            if symbol in closed_symbols:
                diff.re_entered.append(symbol)
            rows.append(
                {
                    "symbol": symbol,
                    "index_name": self.index_name,
                    # Wikipedia's "date added" is authoritative when it is not in
                    # the future and not before our history window.
                    "start_date": _best_start_date(constituent, observed_date, self.history_start),
                    "end_date": None,
                    "company_name": constituent.company_name,
                    "gics_sector": constituent.gics_sector,
                    "gics_sub_industry": constituent.gics_sub_industry,
                    "cik": constituent.cik,
                    "reason_added": "detected:live_universe",
                    "reason_removed": None,
                    "source": "wikipedia",
                    "first_seen_utc": now,
                    "last_seen_utc": now,
                    "consecutive_empty_runs": 0,
                    "suspected_delisted": False,
                    "last_data_date": pd.NaT,
                }
            )

        removed_set = set(removed)
        for row in rows:
            if row.get("index_name") != self.index_name:
                continue
            symbol = row["symbol"]
            if symbol in removed_set and pd.isna(row.get("end_date")):
                row["end_date"] = end_date
                row["reason_removed"] = "detected:absent_from_live_universe"
                row["last_seen_utc"] = now
                log.info("universe removal: %s closed with end_date=%s", symbol, end_date)
            elif symbol in live_symbols and pd.isna(row.get("end_date")):
                row["last_seen_utc"] = now
                constituent = live_by_symbol[symbol]
                # Refresh metadata that Wikipedia may have corrected since.
                for key, value in (
                    ("company_name", constituent.company_name),
                    ("gics_sector", constituent.gics_sector),
                    ("gics_sub_industry", constituent.gics_sub_industry),
                    ("cik", constituent.cik),
                ):
                    if value:
                        row[key] = value

        self._df = pd.DataFrame(rows, columns=MEMBERSHIP_COLUMNS)
        self._df = _normalize_dates(self._df)
        self._append_universe_history(live_symbols, added, removed, observed_date)

        if diff.has_changes:
            log.warning("UNIVERSE CHANGE on %s -> %s", observed_date, diff.summary())
        else:
            log.info(diff.summary())
        return diff

    def _apply_live_metadata(self, live: list[LiveConstituent]) -> None:
        df = self.load()
        if df.empty:
            return
        by_symbol = {c.symbol: c for c in live}
        for col, attr in (
            ("company_name", "company_name"),
            ("gics_sector", "gics_sector"),
            ("gics_sub_industry", "gics_sub_industry"),
            ("cik", "cik"),
        ):
            df[col] = df.apply(
                lambda r, c=col, a=attr: (
                    getattr(by_symbol[r["symbol"]], a)
                    if r["symbol"] in by_symbol and getattr(by_symbol[r["symbol"]], a)
                    else r.get(c)
                ),
                axis=1,
            )
        self._df = df

    def _last_observed_date(self) -> date | None:
        path = self.paths.universe_history_file()
        if not path.exists():
            return None
        try:
            hist = pd.read_parquet(path, columns=["observed_date", "index_name"])
        except Exception as exc:
            log.warning("could not read universe history: %s", exc)
            return None
        hist = hist[hist["index_name"] == self.index_name]
        if hist.empty:
            return None
        return pd.to_datetime(hist["observed_date"]).max().date()

    def _append_universe_history(
        self, live_symbols: set[str], added: list[str], removed: list[str], observed_date: date
    ) -> None:
        now = datetime.now(UTC)
        added_set, removed_set = set(added), set(removed)
        records = [
            {
                "observed_date": observed_date,
                "index_name": self.index_name,
                "symbol": symbol,
                "change_type": "added" if symbol in added_set else "present",
                "source": "wikipedia",
                "observed_at": now,
            }
            for symbol in sorted(live_symbols)
        ]
        records.extend(
            {
                "observed_date": observed_date,
                "index_name": self.index_name,
                "symbol": symbol,
                "change_type": "removed",
                "source": "wikipedia",
                "observed_at": now,
            }
            for symbol in sorted(removed_set)
        )
        if not records:
            return

        new = pd.DataFrame(records)
        path = self.paths.universe_history_file()
        if path.exists():
            try:
                old = pd.read_parquet(path)
                old = old[
                    ~(
                        (pd.to_datetime(old["observed_date"]).dt.date == observed_date)
                        & (old["index_name"] == self.index_name)
                    )
                ]
                new = pd.concat([old, new], ignore_index=True)
            except Exception as exc:
                log.warning("could not merge universe history (%s); overwriting", exc)
        self.writer.write(new, P.UNIVERSE_HISTORY, path, mode="overwrite")

    # ----------------------------------------------- silent delisting tracking

    def record_data_observations(self, observations: dict[str, date | None]) -> list[str]:
        """Update per-symbol data-availability state. Returns newly flagged symbols.

        A symbol that stops returning data while still nominally in the index is
        the earliest warning that a company is about to vanish from free sources.
        We flag it; we never act on it by deleting anything.
        """
        df = self.load()
        if df.empty:
            return []

        # Guarantee the columns we are about to assign into accept their values.
        df["last_data_date"] = df["last_data_date"].astype("object")
        df["consecutive_empty_runs"] = (
            pd.to_numeric(df["consecutive_empty_runs"], errors="coerce").fillna(0).astype(int)
        )
        df["suspected_delisted"] = df["suspected_delisted"].fillna(False).astype(bool)

        newly_flagged: list[str] = []
        open_by_symbol = {}
        for idx, row in df.iterrows():
            if row["index_name"] == self.index_name and pd.isna(row["end_date"]):
                open_by_symbol[row["symbol"]] = idx

        for symbol, data_date in observations.items():
            idx = open_by_symbol.get(symbol)
            if idx is None:
                # Also track closed intervals: we keep pulling delisted names
                # until the free sources genuinely stop answering.
                matches = df.index[df["symbol"] == symbol]
                if len(matches) == 0:
                    continue
                idx = matches[-1]

            if data_date is not None:
                df.at[idx, "consecutive_empty_runs"] = 0
                prior = df.at[idx, "last_data_date"]
                if pd.isna(prior) or data_date > _as_date(prior):
                    df.at[idx, "last_data_date"] = data_date
                if bool(df.at[idx, "suspected_delisted"]):
                    log.info("%s is returning data again; clearing delisting flag", symbol)
                    df.at[idx, "suspected_delisted"] = False
            else:
                runs = int(df.at[idx, "consecutive_empty_runs"] or 0) + 1
                df.at[idx, "consecutive_empty_runs"] = runs
                if runs >= self.silent_delist_threshold and not bool(
                    df.at[idx, "suspected_delisted"]
                ):
                    df.at[idx, "suspected_delisted"] = True
                    newly_flagged.append(symbol)
                    log.warning(
                        "SILENT DELISTING SUSPECTED: %s returned no data for %d consecutive runs "
                        "(last data %s). History retained; flagged for review.",
                        symbol,
                        runs,
                        df.at[idx, "last_data_date"],
                    )

        self._df = df
        return newly_flagged

    # -------------------------------------------------------- universe views

    def current_members(self) -> list[str]:
        df = self.load()
        if df.empty:
            return []
        mask = (df["index_name"] == self.index_name) & df["end_date"].isna()
        return sorted(df.loc[mask, "symbol"].unique())

    def members_on(self, as_of: date) -> list[str]:
        df = self.load()
        if df.empty:
            return []
        end = df["end_date"].map(lambda d: date(9999, 12, 31) if pd.isna(d) else _as_date(d))
        start = df["start_date"].map(_as_date)
        mask = (df["index_name"] == self.index_name) & (start <= as_of) & (end >= as_of)
        return sorted(df.loc[mask, "symbol"].unique())

    def tracked_symbols(self, as_of: date | None = None) -> list[str]:
        """Every symbol we should still attempt to fetch.

        Current members, plus recently removed names inside the grace window,
        plus any historical member that has not yet been flagged as silently
        delisted -- because free sources often keep serving a delisted ticker for
        a while, and that tail is exactly what keeps the dataset unbiased.
        """
        as_of = as_of or date.today()
        df = self.load()
        if df.empty:
            return []

        index_mask = df["index_name"] == self.index_name
        current = df["end_date"].isna()
        cutoff = as_of - timedelta(days=self.post_removal_grace_days)
        recently_removed = df["end_date"].notna() & (
            df["end_date"].map(lambda d: _as_date(d) if pd.notna(d) else date.min) >= cutoff
        )
        still_responding = ~df["suspected_delisted"].fillna(False).astype(bool)

        mask = index_mask & (current | recently_removed | still_responding)
        return sorted(df.loc[mask, "symbol"].unique())

    def suspected_delisted(self) -> list[str]:
        df = self.load()
        if df.empty:
            return []
        mask = (df["index_name"] == self.index_name) & df["suspected_delisted"].fillna(
            False
        ).astype(bool)
        return sorted(df.loc[mask, "symbol"].unique())

    def stats(self) -> dict:
        df = self.load()
        if df.empty:
            return {"intervals": 0, "symbols": 0, "current": 0, "historical": 0, "flagged": 0}
        index_df = df[df["index_name"] == self.index_name]
        return {
            "intervals": len(index_df),
            "symbols": int(index_df["symbol"].nunique()),
            "current": int(index_df["end_date"].isna().sum()),
            "historical": int(index_df["end_date"].notna().sum()),
            "flagged": int(index_df["suspected_delisted"].fillna(False).astype(bool).sum()),
        }


# ------------------------------------------------------------------- helpers


def _as_date(value) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.Timestamp(value).date()


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Force date columns to plain ``datetime.date`` (or None) for stable comparisons."""
    for col in ("start_date", "end_date", "last_data_date"):
        if col in df.columns:
            df[col] = df[col].map(lambda v: None if pd.isna(v) else _as_date(v)).astype("object")
    for col in ("consecutive_empty_runs",):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    if "suspected_delisted" in df.columns:
        df["suspected_delisted"] = df["suspected_delisted"].fillna(False).astype(bool)
    return df


def _best_start_date(
    constituent: LiveConstituent, observed_date: date, history_start: date | None
) -> date:
    """Prefer Wikipedia's stated 'date added' when it is sane, else today."""
    candidate = constituent.date_added
    if candidate is None or candidate > observed_date:
        return observed_date
    if history_start is not None and candidate < history_start:
        return history_start
    return candidate
