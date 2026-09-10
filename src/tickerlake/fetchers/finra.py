"""FINRA daily short sale volume.

Free, no key, one flat file per trading day covering every US equity. Short
volume as a share of total reported volume is a well-documented cross-sectional
signal and is absent from every other source wired up here.

An important caveat to record alongside the data: this is *short volume*, not
*short interest*. It counts shares sold short during the session, including
market-maker hedging that is flattened minutes later, so a high ratio does not
by itself mean accumulated bearish positioning. It is a flow measure. Short
interest proper is reported bi-monthly and lives elsewhere.

FINRA publishes several files per day; ``CNMS`` is the consolidated tape across
all venues, which is the one worth having. The per-venue files (Nasdaq, NYSE,
ADF) are the same trades sliced by reporting facility.
"""

from __future__ import annotations

import io
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from tickerlake.fetchers.base import BaseFetcher, FetchResult
from tickerlake.storage import paths as P
from tickerlake.utils.http import HttpClient, HttpError

SOURCE = "finra"

# Consolidated NMS tape: all venues combined.
FILE_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{stamp}.txt"


class FinraShortVolumeFetcher(BaseFetcher):
    """Daily short sale volume for the tracked universe."""

    name = "finra"
    dataset = P.SHORT_VOLUME

    def collect(self, run_date: date, result: FetchResult) -> None:
        client = HttpClient(
            requests_per_second=float(self.cfg("requests_per_second", 2.0)),
            user_agent="TickerLake/0.1 (research)",
            max_attempts=3,
        )
        lookback = int(self.cfg("lookback_days", 5))
        tracked = set(self._tracked_symbols())
        if not tracked:
            result.add_warning("no tracked symbols; is the membership table seeded?")

        try:
            for offset in range(lookback + 1):
                day = run_date - timedelta(days=offset)
                if day.weekday() >= 5:
                    continue
                path = self.paths.short_volume_file(day)
                if path.exists() and not self.cfg("force_refetch", False):
                    result.items_skipped += 1
                    continue
                if self._fetch_day(client, day, tracked, result):
                    result.items_succeeded += 1
                else:
                    result.items_skipped += 1
        finally:
            client.close()

    def _fetch_day(
        self, client: HttpClient, day: date, tracked: set[str], result: FetchResult
    ) -> bool:
        url = FILE_URL.format(stamp=day.strftime("%Y%m%d"))
        try:
            text = client.get_text(url)
        except HttpError:
            # Holidays and not-yet-published days legitimately 404.
            self.log.debug("no FINRA short volume file for %s", day)
            return False
        except Exception as exc:
            result.add_warning(f"finra {day}: {type(exc).__name__}: {exc}")
            return False

        df = pd.read_csv(io.StringIO(text), sep="|")
        expected = {"Date", "Symbol", "ShortVolume", "TotalVolume"}
        if not expected.issubset(df.columns):
            result.add_warning(f"finra {day}: unexpected columns {list(df.columns)}")
            return False

        # The file ends with a footer row that is not data.
        df = df[df["Symbol"].notna() & (df["Symbol"].astype(str).str.len() > 0)]
        df["symbol"] = df["Symbol"].astype(str).str.strip().str.upper().str.replace("-", ".")
        if tracked:
            df = df[df["symbol"].isin(tracked)]
        if df.empty:
            self.log.debug("no tracked symbols in the FINRA file for %s", day)
            return False

        short_vol = pd.to_numeric(df["ShortVolume"], errors="coerce")
        total_vol = pd.to_numeric(df["TotalVolume"], errors="coerce")

        out = pd.DataFrame(
            {
                "symbol": df["symbol"],
                "date": pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce").dt.date,
                "short_volume": short_vol,
                "short_exempt_volume": pd.to_numeric(df.get("ShortExemptVolume"), errors="coerce"),
                "total_volume": total_vol,
                # The headline feature. Guarded against the zero-volume rows that
                # appear for symbols that did not trade.
                "short_volume_ratio": (short_vol / total_vol).where(total_vol > 0),
                "market": df.get("Market"),
                "source": SOURCE,
                "ingested_at": datetime.now(UTC),
            }
        )
        out = out.dropna(subset=["symbol", "date"])

        write = self.writer.write(
            out, P.SHORT_VOLUME, self.paths.short_volume_file(day), mode="overwrite"
        )
        result.record_write(write)
        self.log.info(
            "%s: %d tracked symbols, mean short ratio %.1f%%",
            day,
            len(out),
            100 * out["short_volume_ratio"].mean(),
        )
        return True

    def _tracked_symbols(self) -> list[str]:
        if self.symbols_override is not None:
            return self.symbols_override
        if self.tracker is None:
            return []
        return self.tracker.tracked_symbols()
