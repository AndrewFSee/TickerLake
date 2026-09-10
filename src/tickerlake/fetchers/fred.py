"""FRED macro time series.

One Parquet file per series, merged on each run. FRED revises history (GDP and
payrolls get restated for years), so every run re-pulls the full series rather
than appending only new observations -- it is a few hundred KB per series and it
means the stored history always matches what FRED currently believes.

``realtime_start``/``realtime_end`` are carried through so a later vintage-aware
backtest can ask what was *known* on a date, not just what is true now.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pandas as pd

from tickerlake.fetchers.base import BaseFetcher, FetchResult
from tickerlake.storage import paths as P
from tickerlake.utils.http import HttpClient, HttpError

SOURCE = "fred"
OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
SERIES_URL = "https://api.stlouisfed.org/fred/series"


class FredFetcher(BaseFetcher):
    """Macro series from the St. Louis Fed."""

    name = "fred"
    dataset = P.MACRO_SERIES
    requires_secret = "fred_api_key"

    def collect(self, run_date: date, result: FetchResult) -> None:
        api_key = self.config.secrets.fred_api_key
        series_ids = list(self.cfg("series", []))
        if not series_ids:
            result.add_warning("no FRED series configured")
            return

        start = str(self.cfg("start_date", self.config.get("ohlcv.start_date", "2010-01-01")))[:10]
        client = HttpClient(
            requests_per_second=float(self.cfg("requests_per_second", 5.0)),
            user_agent="TickerLake/0.1",
        )
        self.log.info("fetching %d FRED series from %s", len(series_ids), start)

        try:
            for series_id in series_ids:
                try:
                    meta = self._series_metadata(client, series_id, api_key)
                    frame = self._observations(client, series_id, api_key, start, meta)
                except HttpError as exc:
                    result.items_failed += 1
                    result.add_warning(f"{series_id}: {exc}")
                    continue
                except Exception as exc:
                    result.items_failed += 1
                    result.add_warning(f"{series_id}: {type(exc).__name__}: {exc}")
                    continue

                if frame.empty:
                    result.items_skipped += 1
                    continue

                write = self.writer.write(
                    frame,
                    P.MACRO_SERIES,
                    self.paths.macro_series_file(series_id),
                    mode="overwrite",
                )
                result.record_write(write)
                result.items_succeeded += 1
                self.log.debug("%s: %d observations", series_id, len(frame))
        finally:
            client.close()

        result.details["series_fetched"] = result.items_succeeded

    def _series_metadata(self, client: HttpClient, series_id: str, api_key: str) -> dict[str, Any]:
        try:
            payload = client.get_json(
                SERIES_URL,
                params={"series_id": series_id, "api_key": api_key, "file_type": "json"},
            )
            entries = payload.get("seriess") or []
            return entries[0] if entries else {}
        except Exception:
            # Metadata is a nicety; never let it block the observations.
            return {}

    def _observations(
        self,
        client: HttpClient,
        series_id: str,
        api_key: str,
        start: str,
        meta: dict[str, Any],
    ) -> pd.DataFrame:
        payload = client.get_json(
            OBSERVATIONS_URL,
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "observation_start": start,
            },
        )
        observations = payload.get("observations") or []
        if not observations:
            return pd.DataFrame()

        df = pd.DataFrame(observations)
        # FRED encodes missing observations as "."; coerce turns those into NaN.
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["series_id"] = series_id
        df["title"] = meta.get("title")
        df["units"] = meta.get("units")
        df["frequency"] = meta.get("frequency")
        df["seasonal_adjustment"] = meta.get("seasonal_adjustment")
        df["source"] = SOURCE
        df["ingested_at"] = datetime.now(UTC)
        return df
