"""Tests for telling a FRED outage apart from a bad series.

On 2026-10-21 every FRED request came back as the St. Louis Fed's "Site
Maintenance" page. All 31 series failed and the stage still reported OK, so
the run summary listed no failed stages and the outage was visible only to
someone reading thirty-one HTML-stuffed warnings.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from tickerlake.config import Config, Secrets
from tickerlake.fetchers.fred import FredFetcher
from tickerlake.storage.paths import DatasetPaths
from tickerlake.storage.writer import ParquetWriter
from tickerlake.utils.http import HttpError

MAINTENANCE = HttpError(
    "non-JSON response from https://api.stlouisfed.org/fred/series/observations: "
    '<!doctype html>\n<html lang="en">\n<head>\n<title>Site Maintenance | St. Louis Fed</title>'
)


@pytest.fixture
def fetcher(tmp_path):
    paths = DatasetPaths(tmp_path)
    paths.ensure_layout()
    config = Config(
        raw={"fred": {"enabled": True, "series": ["DGS10", "UNRATE", "VIXCLS"]}},
        secrets=Secrets(fred_api_key="test"),
        data_root=tmp_path,
        config_path=tmp_path / "config.yaml",
    )
    return FredFetcher(config, paths, ParquetWriter())


def _series(series_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-10-20"],
            "value": [4.1],
            "realtime_start": ["2026-10-21"],
            "realtime_end": ["2026-10-21"],
            "series_id": [series_id],
            "title": [series_id],
            "units": ["Percent"],
            "frequency": ["Daily"],
            "seasonal_adjustment": ["NSA"],
            "source": ["fred"],
            "ingested_at": [pd.Timestamp.now(tz="UTC")],
        }
    )


def test_every_series_failing_fails_the_stage(fetcher, monkeypatch):
    def down(client, series_id, api_key):
        raise MAINTENANCE

    monkeypatch.setattr(fetcher, "_series_metadata", down)
    result = fetcher.run(date(2026, 10, 21))

    assert not result.ok
    assert result.items_failed == 3
    assert any("all 3 FRED series failed" in e for e in result.errors)


def test_one_series_failing_is_only_a_warning(fetcher, monkeypatch):
    monkeypatch.setattr(fetcher, "_series_metadata", lambda client, sid, key: {"title": sid})

    def observations(client, series_id, api_key, start, meta):
        if series_id == "UNRATE":
            raise MAINTENANCE
        return _series(series_id)

    monkeypatch.setattr(fetcher, "_observations", observations)
    result = fetcher.run(date(2026, 10, 21))

    assert result.ok, result.errors
    assert (result.items_succeeded, result.items_failed) == (2, 1)


def test_an_html_page_is_reported_by_its_first_line(fetcher, monkeypatch):
    def down(client, series_id, api_key):
        raise MAINTENANCE

    monkeypatch.setattr(fetcher, "_series_metadata", down)
    result = fetcher.run(date(2026, 10, 21))

    assert all("\n" not in w and "<head>" not in w for w in result.warnings)
