"""Tests for throttling, checkpointing, expiration filters, and compaction."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import pandas as pd
import pyarrow.parquet as pq
import pytest

from tickerlake.storage import paths as P
from tickerlake.storage.paths import DatasetPaths
from tickerlake.storage.writer import ParquetWriter
from tickerlake.utils.checkpoint import Checkpoint
from tickerlake.utils.throttle import AdaptiveThrottle, RateLimiter, is_rate_limit_error

# ------------------------------------------------------------------ throttle


def test_rate_limit_detection_across_error_shapes():
    class YFRateLimitError(Exception):
        pass

    class FakeResponse:
        status_code = 429

    class WithResponse(Exception):
        response = FakeResponse()

    assert is_rate_limit_error(YFRateLimitError("boom"))
    assert is_rate_limit_error(Exception("Too Many Requests. Rate limited."))
    assert is_rate_limit_error(Exception("HTTP 429 returned"))
    assert is_rate_limit_error(WithResponse())
    assert not is_rate_limit_error(ValueError("unrelated parsing problem"))


def test_rate_limiter_enforces_minimum_interval():
    limiter = RateLimiter(min_interval=0.05, jitter_pct=0.0)
    limiter.wait()
    slept = limiter.wait()
    assert slept > 0, "the second call must wait for the interval to elapse"


def test_adaptive_throttle_slows_down_then_recovers():
    throttle = AdaptiveThrottle(
        base_interval=1.0,
        slowdown_factor=2.0,
        backoff_seconds=0.0,  # no real sleeping in tests
        decay_after_successes=2,
    )
    assert throttle.current_interval == 1.0

    throttle.record_rate_limit()
    assert throttle.current_interval == 2.0, "a rate limit must widen the interval"
    assert throttle.rate_limit_hits == 1

    throttle.record_success()
    throttle.record_success()
    assert throttle.current_interval < 2.0, "sustained success must decay the penalty"
    assert throttle.current_interval >= 1.0, "and never go below the configured baseline"


def test_adaptive_throttle_multiplier_is_capped():
    throttle = AdaptiveThrottle(
        base_interval=1.0, slowdown_factor=4.0, backoff_seconds=0.0, max_multiplier=8.0
    )
    for _ in range(10):
        throttle.record_rate_limit()
    assert throttle.multiplier == 8.0


# ---------------------------------------------------------------- checkpoint


def test_checkpoint_roundtrips(tmp_path):
    path = tmp_path / "cp.json"
    cp = Checkpoint.load_or_create(path, "options", date(2024, 5, 2))
    cp.mark_completed("AAPL", rows=100)
    cp.mark_failed("MSFT", "timeout")
    cp.mark_skipped("BRK.B", "no options")
    cp.save()

    reloaded = Checkpoint.load_or_create(path, "options", date(2024, 5, 2))
    assert reloaded.is_done("AAPL")
    assert reloaded.is_done("BRK.B")
    assert not reloaded.is_done("MSFT")
    assert reloaded.attempts("MSFT") == 1
    assert reloaded.rows_written() == 100


def test_checkpoint_pending_skips_done_work(tmp_path):
    cp = Checkpoint.load_or_create(tmp_path / "cp.json", "options", date(2024, 5, 2))
    cp.mark_completed("A")
    cp.mark_skipped("B", "no options")
    assert cp.pending(["A", "B", "C", "D"]) == ["C", "D"]


def test_checkpoint_counts_attempts_for_retry_budget(tmp_path):
    cp = Checkpoint.load_or_create(tmp_path / "cp.json", "options", date(2024, 5, 2))
    for _ in range(3):
        cp.mark_failed("X", "boom")
    assert cp.attempts("X") == 3
    assert cp.retryable(max_attempts=3) == []
    assert cp.retryable(max_attempts=4) == ["X"]


def test_corrupt_checkpoint_starts_fresh_rather_than_crashing(tmp_path):
    path = tmp_path / "cp.json"
    path.write_text("{not valid json", encoding="utf-8")

    cp = Checkpoint.load_or_create(path, "options", date(2024, 5, 2))
    assert cp.completed == {}
    assert cp.total_seen == 0


def test_checkpoint_save_is_atomic(tmp_path):
    path = tmp_path / "cp.json"
    cp = Checkpoint.load_or_create(path, "options", date(2024, 5, 2))
    cp.mark_completed("AAPL", rows=5)
    cp.save()

    assert not list(tmp_path.glob("*.tmp"))
    assert json.loads(path.read_text(encoding="utf-8"))["counts"]["completed"] == 1


# --------------------------------------------------------- expiration filters


@pytest.fixture
def options_fetcher(tmp_path):
    from tickerlake.config import Config, Secrets
    from tickerlake.fetchers.yf_options import YFinanceOptionsFetcher

    paths = DatasetPaths(tmp_path)
    paths.ensure_layout()
    config = Config(
        raw={"options": {"enabled": True}},
        secrets=Secrets(),
        data_root=tmp_path,
        config_path=tmp_path / "config.yaml",
    )
    return YFinanceOptionsFetcher(config, paths, ParquetWriter())


def test_default_filters_keep_every_expiration(options_fetcher):
    run_date = date(2024, 5, 1)
    expirations = ["2024-05-03", "2024-05-17", "2024-06-21", "2026-01-16"]
    assert options_fetcher._filter_expirations(expirations, run_date) == expirations


def test_max_dte_filter(options_fetcher):
    options_fetcher.config.raw["options"]["max_dte"] = 60
    run_date = date(2024, 5, 1)
    kept = options_fetcher._filter_expirations(["2024-05-03", "2024-06-21", "2026-01-16"], run_date)
    assert kept == ["2024-05-03", "2024-06-21"]


def test_monthlies_only_filter_keeps_third_fridays(options_fetcher):
    options_fetcher.config.raw["options"]["monthlies_only"] = True
    run_date = date(2024, 5, 1)
    # 2024-05-17 and 2024-06-21 are third Fridays; the others are weeklies.
    kept = options_fetcher._filter_expirations(
        ["2024-05-03", "2024-05-17", "2024-05-24", "2024-06-21"], run_date
    )
    assert kept == ["2024-05-17", "2024-06-21"]


def test_unparseable_expiration_is_skipped_not_fatal(options_fetcher):
    kept = options_fetcher._filter_expirations(["garbage", "2024-05-17"], date(2024, 5, 1))
    assert kept == ["2024-05-17"]


# --------------------------------------------------------------- compaction


def _options_rows(symbol, snapshot, n=5):
    now = datetime.now(UTC)
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "snapshot_date": snapshot,
                "expiration": date(2024, 6, 21),
                "option_type": "call",
                "strike": 100.0 + i,
                "contract_symbol": f"{symbol}240621C{i:05d}",
                "bid": 1.0,
                "ask": 1.1,
                "volume": 10,
                "open_interest": 100,
                "implied_volatility": 0.25,
                "in_the_money": False,
                "dte": 30,
                "underlying_price": 105.0,
                "source": "test",
                "ingested_at": now,
            }
            for i in range(n)
        ]
    )


def test_compaction_merges_and_removes_deltas(tmp_path):
    from tickerlake.config import Config, Secrets
    from tickerlake.pipeline.compact import Compactor

    paths = DatasetPaths(tmp_path)
    paths.ensure_layout()
    writer = ParquetWriter()
    snapshot = date(2024, 5, 2)

    for symbol in ("AAPL", "MSFT", "NVDA"):
        writer.write(
            _options_rows(symbol, snapshot),
            P.OPTIONS_CHAINS,
            paths.options_symbol_file(snapshot, symbol),
            mode="overwrite",
        )

    partition = paths.options_partition(snapshot)
    assert len(list(partition.glob("*.parquet"))) == 3

    config = Config(
        raw={"compaction": {"min_age_days": 0, "datasets": [P.OPTIONS_CHAINS]}},
        secrets=Secrets(),
        data_root=tmp_path,
        config_path=tmp_path / "config.yaml",
    )
    results = Compactor(config).run(force=True)

    files = list(partition.glob("*.parquet"))
    assert len(files) == 1 and files[0].name == "data.parquet"
    assert pq.read_metadata(files[0]).num_rows == 15
    assert results[0].partitions_compacted == 1
    assert results[0].files_merged == 3


def test_compaction_output_is_sorted_for_row_group_pruning(tmp_path):
    from tickerlake.config import Config, Secrets
    from tickerlake.pipeline.compact import Compactor

    paths = DatasetPaths(tmp_path)
    paths.ensure_layout()
    writer = ParquetWriter()
    snapshot = date(2024, 5, 2)

    for symbol in ("NVDA", "AAPL", "MSFT"):
        writer.write(
            _options_rows(symbol, snapshot),
            P.OPTIONS_CHAINS,
            paths.options_symbol_file(snapshot, symbol),
            mode="overwrite",
        )

    config = Config(
        raw={"compaction": {"min_age_days": 0, "datasets": [P.OPTIONS_CHAINS]}},
        secrets=Secrets(),
        data_root=tmp_path,
        config_path=tmp_path / "config.yaml",
    )
    Compactor(config).run(force=True)

    out = pq.read_table(paths.options_compacted_file(snapshot)).to_pandas()
    assert out["symbol"].tolist() == sorted(out["symbol"].tolist())


def test_compaction_is_a_no_op_on_an_empty_lake(tmp_path):
    from tickerlake.config import Config, Secrets
    from tickerlake.pipeline.compact import Compactor

    DatasetPaths(tmp_path).ensure_layout()
    config = Config(
        raw={"compaction": {"min_age_days": 0, "datasets": [P.OHLCV, P.OPTIONS_CHAINS]}},
        secrets=Secrets(),
        data_root=tmp_path,
        config_path=tmp_path / "config.yaml",
    )
    results = Compactor(config).run(force=True)
    assert all(r.partitions_compacted == 0 and not r.errors for r in results)
