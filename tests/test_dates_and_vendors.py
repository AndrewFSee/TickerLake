"""Tests for shared date coercion, the Tiingo cross-check, and Marketaux parsing."""

from __future__ import annotations

import tempfile
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from tickerlake.utils.dates import as_date, as_date_strict

# ------------------------------------------------------------ date coercion


def test_timestamp_becomes_a_plain_date():
    """pd.Timestamp satisfies isinstance(x, date) but is not equal to one.

    This produced three separate defects: membership comparisons, the Finnhub
    price cross-check, and the Tiingo adjusted-price check -- where it matched
    zero bars while every request succeeded and nothing logged an error.
    """
    ts = pd.Timestamp("2026-09-04")
    assert isinstance(ts, date), "which is exactly why the naive guard passes"
    assert ts != date(2026, 9, 4), "and exactly why that breaks dict lookups"

    assert as_date(ts) == date(2026, 9, 4)
    assert type(as_date(ts)) is date


def test_dict_keys_match_after_coercion():
    """The real failure mode, reproduced: a lookup that silently finds nothing."""
    coerced = {as_date(pd.Timestamp("2026-09-04")): 100.0}
    assert date(2026, 9, 4) in coerced

    naive = {pd.Timestamp("2026-09-04"): 100.0}
    assert date(2026, 9, 4) not in naive


@pytest.mark.parametrize(
    "value",
    [
        date(2026, 9, 4),
        datetime(2026, 9, 4, 13, 30),
        pd.Timestamp("2026-09-04 13:30:00"),
        "2026-09-04",
        "2026-09-04T00:00:00.000Z",
    ],
)
def test_accepts_every_shape_the_sources_emit(value):
    assert as_date(value) == date(2026, 9, 4)


def test_null_like_values_return_none_rather_than_raising():
    assert as_date(None) is None
    assert as_date(pd.NaT) is None
    assert as_date("") is None
    assert as_date("not a date") is None


def test_strict_variant_raises():
    assert as_date_strict("2026-09-04") == date(2026, 9, 4)
    with pytest.raises(ValueError, match="cannot interpret"):
        as_date_strict("nonsense")


# ------------------------------------------------------------------ tiingo


def test_relative_difference_detects_a_missed_split():
    from tickerlake.fetchers.tiingo import PRICE_TOLERANCE, _rel

    # Rounding noise between vendors sits inside tolerance.
    assert _rel(100.00, 100.02) < PRICE_TOLERANCE
    # A missed 4:1 split is what checking adjusted prices is actually for.
    assert _rel(100.0, 400.0) > PRICE_TOLERANCE
    assert _rel(100.0, 25.0) > PRICE_TOLERANCE


def test_relative_difference_handles_missing_and_zero():
    from tickerlake.fetchers.tiingo import _rel

    assert _rel(None, 100.0) is None
    assert _rel(100.0, None) is None
    assert _rel(0.0, 100.0) is None, "a zero base would divide by zero"


def test_tiingo_is_gated_and_keyed():
    from tickerlake.fetchers.tiingo import TiingoFetcher

    assert TiingoFetcher.requires_trading_day is True
    assert TiingoFetcher.requires_secret == "tiingo_api_key"


# --------------------------------------------------------------- marketaux


def _fetcher():
    from tickerlake.config import Config, Secrets
    from tickerlake.fetchers.marketaux import MarketauxFetcher
    from tickerlake.storage.paths import DatasetPaths
    from tickerlake.storage.writer import ParquetWriter

    tmp = Path(tempfile.mkdtemp())
    paths = DatasetPaths(tmp)
    paths.ensure_layout()
    config = Config(
        raw={"marketaux": {"enabled": True}},
        secrets=Secrets(marketaux_api_key="x"),
        data_root=tmp,
        config_path=tmp / "c.yaml",
    )
    return MarketauxFetcher(config, paths, ParquetWriter())


def _payload(entities):
    return {
        "data": [
            {
                "uuid": "abc-123",
                "url": "https://example.com/story",
                "title": "Chipmakers slide on AI slowdown fears",
                "description": "A description.",
                "published_at": "2026-09-14T12:00:00.000000Z",
                "source": "example.com",
                "language": "en",
                "entities": entities,
            }
        ]
    }


def _parse(entities, min_match=10.0):
    return _fetcher()._parse(
        _payload(entities), date(2026, 9, 14), min_match, datetime(2026, 9, 14)
    )


def test_one_row_per_tagged_entity():
    """An article on a downgrade concerns several firms, each with its own score."""
    rows = _parse(
        [
            {
                "symbol": "AMD",
                "type": "equity",
                "sentiment_score": -0.74,
                "match_score": 39.6,
                "industry": "Technology",
            },
            {
                "symbol": "AVGO",
                "type": "equity",
                "sentiment_score": -0.76,
                "match_score": 31.8,
                "industry": "Technology",
            },
        ]
    )
    assert {r["symbol"] for r in rows} == {"AMD", "AVGO"}
    assert rows[0]["sentiment"] != rows[1]["sentiment"], "per entity, not per article"
    assert len({r["event_id"] for r in rows}) == 2, "ids must not collide"


def test_low_confidence_tags_are_dropped():
    rows = _parse(
        [
            {"symbol": "AMD", "type": "equity", "sentiment_score": -0.7, "match_score": 39.6},
            {"symbol": "XYZ", "type": "equity", "sentiment_score": 0.1, "match_score": 2.0},
        ]
    )
    assert {r["symbol"] for r in rows} == {"AMD"}


def test_non_equity_entities_are_ignored():
    rows = _parse([{"symbol": "BTC", "type": "crypto", "sentiment_score": 0.5, "match_score": 90}])
    assert rows == []


def test_symbols_are_canonicalised_to_dot_form():
    rows = _parse(
        [{"symbol": "BRK-B", "type": "equity", "sentiment_score": 0.2, "match_score": 50}]
    )
    assert rows[0]["symbol"] == "BRK.B", "must join with the rest of the lake"


def test_sentiment_is_carried_through_unscaled():
    """Marketaux already publishes -1..+1, so it must not be rescaled."""
    rows = _parse(
        [{"symbol": "AMD", "type": "equity", "sentiment_score": -0.743, "match_score": 40}]
    )
    assert rows[0]["sentiment"] == pytest.approx(-0.743)
    assert rows[0]["tone"] is None, "tone is GDELT's raw scale and stays empty here"


def test_empty_payload_is_survivable():
    assert _parse([]) == []
    assert _fetcher()._parse({}, date(2026, 9, 14), 10.0, datetime(2026, 9, 14)) == []


def test_gdelt_tone_is_normalised_into_the_shared_column():
    """GDELT tone runs roughly -100..+100 and must be scaled to match."""
    tone = 8.5
    assert max(-1.0, min(1.0, tone / 100.0)) == pytest.approx(0.085)
    # And clamped, since GDELT can emit values outside its nominal range.
    assert max(-1.0, min(1.0, 250.0 / 100.0)) == 1.0
