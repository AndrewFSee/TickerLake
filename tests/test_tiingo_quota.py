"""Tests for Tiingo refusing a request with a JSON body instead of bars.

On 2026-10-27 the free tier's 500-unique-symbols-a-month cap ran out four
symbols into the slice. Tiingo answers that with HTTP 200 and
{"detail": "You have run over your 500 symbol look up for this month..."},
the loop iterated the dict's keys as if they were bars, and the stage died on
``'str' object has no attribute 'get'``.
"""

from __future__ import annotations

from datetime import date

import pytest

from tickerlake.config import Config, Secrets
from tickerlake.fetchers.tiingo import TiingoFetcher
from tickerlake.storage.paths import DatasetPaths
from tickerlake.storage.writer import ParquetWriter
from tickerlake.utils.http import HttpClient

RUN_DATE = date(2026, 10, 27)
OVER_QUOTA = {
    "detail": "You have run over your 500 symbol look up for this month. "
    "Please upgrade at https://api.tiingo.com/pricing to have your limits increased."
}
SYMBOLS = ["WRB", "WSM", "WST", "WTW", "WY", "WYNN"]


@pytest.fixture
def fetcher(tmp_path, monkeypatch):
    paths = DatasetPaths(tmp_path)
    paths.ensure_layout()
    config = Config(
        raw={"tiingo": {"enabled": True}},
        secrets=Secrets(tiingo_api_key="test"),
        data_root=tmp_path,
        config_path=tmp_path / "config.yaml",
    )
    f = TiingoFetcher(config, paths, ParquetWriter(), symbols_override=SYMBOLS)
    ours = {s: {RUN_DATE: (100.0, 99.0)} for s in SYMBOLS}
    monkeypatch.setattr(f, "_our_bars", lambda start, end, symbols: ours)
    return f


def _answer(monkeypatch, refused_from: str | None, body=OVER_QUOTA) -> list[str]:
    asked: list[str] = []

    def get_json(self, url, params=None):
        symbol = url.split("/")[-2]
        asked.append(symbol)
        if refused_from and SYMBOLS.index(symbol) >= SYMBOLS.index(refused_from):
            return body
        return [{"date": "2026-10-27T00:00:00.000Z", "close": 100.0, "adjClose": 99.0}]

    monkeypatch.setattr(HttpClient, "get_json", get_json)
    return asked


def test_running_out_of_symbols_is_a_warning_not_a_crash(fetcher, monkeypatch):
    asked = _answer(monkeypatch, refused_from="WTW")
    result = fetcher.run(RUN_DATE)

    assert result.ok, result.errors
    assert (result.items_succeeded, result.items_failed) == (3, 1)
    assert result.details["bars_compared"] == 3
    assert any("monthly symbol cap" in w for w in result.warnings)
    assert asked == ["WRB", "WSM", "WST", "WTW"], "the cap lasts the month; stop asking"


def test_another_refusal_skips_only_that_symbol(fetcher, monkeypatch):
    def get_json(self, url, params=None):
        if "/WY/" in url:
            return {"detail": "Error: Ticker 'WY' not found"}
        return [{"date": "2026-10-27", "close": 100.0, "adjClose": 99.0}]

    monkeypatch.setattr(HttpClient, "get_json", get_json)
    result = fetcher.run(RUN_DATE)

    assert result.ok, result.errors
    assert (result.items_succeeded, result.items_failed) == (5, 1)
    assert not result.warnings
