"""Daily orchestration.

Order is not arbitrary: the universe stage runs first because every other stage
asks the membership table who to fetch. After that, stages are independent and a
failure in one is contained -- a Finnhub outage must not cost the day's options
snapshot.

Each run writes a JSON summary to ``_runs/`` recording what succeeded, what
failed, any universe changes, and any silent-delisting flags raised. That file is
the artefact to look at when a run went wrong, and the thing an alert hook reads.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import time
import traceback
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from tickerlake.config import Config
from tickerlake.fetchers.base import FetchResult
from tickerlake.fetchers.earnings import EarningsFetcher
from tickerlake.fetchers.finnhub import FinnhubFetcher
from tickerlake.fetchers.finra import FinraShortVolumeFetcher
from tickerlake.fetchers.fred import FredFetcher
from tickerlake.fetchers.gdelt import GdeltFetcher
from tickerlake.fetchers.sec_edgar import SECEdgarFetcher
from tickerlake.fetchers.yf_intraday import YFinanceIntradayFetcher
from tickerlake.fetchers.yf_ohlcv import YFinanceOHLCVFetcher
from tickerlake.fetchers.yf_options import YFinanceOptionsFetcher
from tickerlake.storage import paths as P
from tickerlake.storage.writer import ParquetWriter
from tickerlake.universe import sources as SRC
from tickerlake.universe.membership import MembershipTracker, UniverseDiff

log = logging.getLogger(__name__)

FETCHERS = {
    "ohlcv": YFinanceOHLCVFetcher,
    "intraday": YFinanceIntradayFetcher,
    "options": YFinanceOptionsFetcher,
    "sec_edgar": SECEdgarFetcher,
    "fred": FredFetcher,
    "gdelt": GdeltFetcher,
    "finnhub": FinnhubFetcher,
    "finra": FinraShortVolumeFetcher,
    "earnings": EarningsFetcher,
}


@dataclass
class RunSummary:
    """Everything worth knowing about one daily run."""

    run_id: str
    run_date: date
    started_at: datetime
    finished_at: datetime | None = None
    stages: list[FetchResult] = field(default_factory=list)
    universe_diff: UniverseDiff | None = None
    newly_flagged_delisted: list[str] = field(default_factory=list)
    fatal_error: str | None = None

    @property
    def duration_seconds(self) -> float:
        if self.finished_at is None:
            return 0.0
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def failed_stages(self) -> list[str]:
        return [s.stage for s in self.stages if not s.ok and not s.skipped]

    @property
    def ok(self) -> bool:
        return not self.failed_stages and self.fatal_error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_date": self.run_date.isoformat(),
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_seconds": round(self.duration_seconds, 1),
            "duration_minutes": round(self.duration_seconds / 60, 1),
            "ok": self.ok,
            "fatal_error": self.fatal_error,
            "failed_stages": self.failed_stages,
            "host": platform.node(),
            "totals": {
                "rows_written": sum(s.rows_written for s in self.stages),
                "bytes_written": sum(s.bytes_written for s in self.stages),
                "files_written": sum(s.files_written for s in self.stages),
            },
            "universe": self.universe_diff.to_dict() if self.universe_diff else None,
            "newly_flagged_delisted": self.newly_flagged_delisted,
            "stages": [s.to_dict() for s in self.stages],
        }

    def render(self) -> str:
        """Human-readable block for the end of the log."""
        lines = [
            "=" * 78,
            f"TickerLake run {self.run_id}  ({self.run_date})",
            f"status: {'OK' if self.ok else 'FAILED'}   duration: {self.duration_seconds / 60:.1f} min",
            "-" * 78,
        ]
        if self.universe_diff:
            lines.append(f"universe: {self.universe_diff.summary()}")
        if self.newly_flagged_delisted:
            lines.append(f"SILENT DELISTING FLAGGED: {', '.join(self.newly_flagged_delisted[:20])}")
        lines.append("-" * 78)
        for stage in self.stages:
            lines.append("  " + stage.summary_line())
        lines.append("-" * 78)
        totals = self.to_dict()["totals"]
        lines.append(
            f"totals: {totals['rows_written']:,} rows, {totals['files_written']} files, "
            f"{totals['bytes_written'] / 1_048_576:.1f} MB"
        )
        if self.failed_stages:
            lines.append(f"FAILED STAGES: {', '.join(self.failed_stages)}")
        if self.fatal_error:
            lines.append(f"FATAL: {self.fatal_error}")
        lines.append("=" * 78)
        return "\n".join(lines)


class DailyPipeline:
    """Runs the universe refresh followed by every enabled fetcher."""

    def __init__(self, config: Config, limit: int | None = None, symbols: list[str] | None = None):
        self.config = config
        self.paths = P.DatasetPaths(config.data_root)
        self.paths.ensure_layout()
        self.writer = ParquetWriter(
            compression=config.get("storage.compression", "zstd"),
            compression_level=config.get("storage.compression_level", 3),
        )
        self.tracker = _build_tracker(config, self.paths, self.writer)
        self.limit = limit
        self.symbols = symbols

    # ------------------------------------------------------------------ run

    def run(self, run_date: date | None = None, stages: list[str] | None = None) -> RunSummary:
        run_date = run_date or date.today()
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        summary = RunSummary(run_id=run_id, run_date=run_date, started_at=datetime.now(UTC))

        configured = stages or list(self.config.get("pipeline.stages", []))
        log.info("starting run %s for %s; stages: %s", run_id, run_date, ", ".join(configured))

        try:
            for stage in configured:
                if stage == "universe":
                    summary.universe_diff = self._universe_stage(run_date, summary)
                    continue
                if stage == "options_analytics":
                    self._analytics_stage(run_date, summary)
                    continue
                self._fetcher_stage(stage, run_date, summary)
        except KeyboardInterrupt:
            summary.fatal_error = "interrupted by user"
            log.warning("run interrupted; partial progress is checkpointed")
        except Exception as exc:
            summary.fatal_error = f"{type(exc).__name__}: {exc}"
            log.error("fatal pipeline error: %s", exc)
            log.debug("traceback:\n%s", traceback.format_exc())

        summary.finished_at = datetime.now(UTC)
        self._write_summary(summary)
        log.info("\n%s", summary.render())
        return summary

    # --------------------------------------------------------------- stages

    def _universe_stage(self, run_date: date, summary: RunSummary) -> UniverseDiff | None:
        """Refresh the live universe and reconcile the membership table."""
        log.info("=== universe: refreshing live constituents ===")
        try:
            if not self.tracker.exists:
                log.warning("membership table missing; seeding it now")
                seed = SRC.fetch_seed_intervals(self.config.get("universe.membership_seed_url"))
                live = SRC.fetch_live_constituents(self.config.get("universe.live_url"))
                self.tracker.seed(seed, live=live, force=True)
                return None

            live = SRC.fetch_live_constituents(self.config.get("universe.live_url"))
            diff = self.tracker.refresh(live, observed_date=run_date)
            self._register_etfs()
            if diff.rejected:
                # A rejected diff means the parse looked wrong; the stored table
                # is deliberately left untouched.
                log.error("universe refresh rejected; membership left unchanged")
            else:
                self.tracker.save()
            stats = self.tracker.stats()
            log.info(
                "membership: %d intervals, %d symbols, %d current, %d historical, %d flagged",
                stats["intervals"],
                stats["symbols"],
                stats["current"],
                stats["historical"],
                stats["flagged"],
            )
            return diff
        except Exception as exc:
            log.error("universe stage failed: %s", exc)
            log.debug("traceback:\n%s", traceback.format_exc())
            result = FetchResult(stage="universe", dataset=P.MEMBERSHIP)
            result.add_error(f"{type(exc).__name__}: {exc}")
            summary.stages.append(result)
            return None

    def _register_etfs(self) -> None:
        """Add any newly configured ETFs to the collection universe.

        Cheap and idempotent: existing tickers are left alone, so this just
        picks up edits to the config list. ETFs are never removed here -- taking
        one out of the config stops future collection but its stored history
        stays, exactly as a delisted constituent's does.
        """
        if not self.config.get("etfs.enabled", False):
            return
        symbols = self.config.get("etfs.symbols") or {}
        if not symbols:
            return

        history_start = self.config.get("etfs.history_start")
        added, present = self.tracker.register_static(
            symbols,
            index_name=self.config.get("etfs.index_name", "ETF"),
            start_date=date.fromisoformat(history_start) if history_start else None,
        )
        if added:
            log.warning("ETF universe: %d new (%s)", len(added), ", ".join(added[:15]))
        else:
            log.info("ETF universe: %d tracked, no changes", len(present))

    def _analytics_stage(self, run_date: date, summary: RunSummary) -> None:
        """Derive IV, Greeks, and flow ratios from the chains just collected.

        A transform rather than a fetch: it reads only what is already stored, so
        it can be re-run at any time against any past snapshot without touching
        a network.
        """
        result = FetchResult(stage="options_analytics", dataset=P.OPTIONS_GREEKS)
        if not self.config.get("options_analytics.enabled", True):
            result.skipped = True
            result.skip_reason = "disabled in config"
            summary.stages.append(result)
            return

        started = time.monotonic()
        try:
            from tickerlake.analytics.options_analytics import OptionsAnalytics

            stats = OptionsAnalytics(self.paths.root, self.writer).run(run_date)
            result.rows_written = stats.get("rows", 0)
            result.details.update(stats)
            if stats.get("contracts") and stats.get("iv_solved_pct", 0) < 20:
                result.add_warning(
                    f"implied volatility solved for only {stats['iv_solved_pct']}% of contracts; "
                    "the underlying snapshot was probably taken outside market hours"
                )
        except Exception as exc:
            result.add_error(f"{type(exc).__name__}: {exc}")
            log.error("options analytics failed: %s", exc)
            log.debug("traceback:\n%s", traceback.format_exc())
        finally:
            result.duration_seconds = time.monotonic() - started
        summary.stages.append(result)
        log.info("=== %s ===", result.summary_line())

    def _fetcher_stage(self, stage: str, run_date: date, summary: RunSummary) -> None:
        fetcher_cls = FETCHERS.get(stage)
        if fetcher_cls is None:
            log.warning("unknown stage '%s' in pipeline.stages; skipping", stage)
            return

        kwargs: dict[str, Any] = {
            "config": self.config,
            "paths": self.paths,
            "writer": self.writer,
            "tracker": self.tracker,
            "symbols_override": self.symbols,
        }
        if stage in ("options", "intraday"):
            kwargs["limit"] = self.limit

        fetcher = fetcher_cls(**kwargs)
        started = time.monotonic()
        result = fetcher.run(run_date)
        summary.stages.append(result)

        flagged = result.details.get("newly_flagged_delisted") or []
        summary.newly_flagged_delisted.extend(flagged)

        budget = self.config.get(f"pipeline.stage_timeout_minutes.{stage}")
        if budget and (time.monotonic() - started) / 60 > float(budget):
            result.add_warning(
                f"stage exceeded its {budget} minute budget "
                f"({(time.monotonic() - started) / 60:.0f} min)"
            )

        if not result.ok and not self.config.get("pipeline.continue_on_stage_failure", True):
            raise RuntimeError(f"stage '{stage}' failed and continue_on_stage_failure is false")

    # -------------------------------------------------------------- summary

    def _write_summary(self, summary: RunSummary) -> None:
        path = self.paths.run_summary_file(summary.run_id)
        try:
            path.write_text(json.dumps(summary.to_dict(), indent=2, default=str), encoding="utf-8")
            log.info("run summary written to %s", path)
        except OSError as exc:
            log.error("could not write run summary: %s", exc)

        # A stable filename for alerting and for "how did last night go?".
        latest = self.paths.root / P.RUNS / "latest.json"
        try:
            latest.write_text(
                json.dumps(summary.to_dict(), indent=2, default=str), encoding="utf-8"
            )
        except OSError:
            pass

        if not summary.ok:
            self._alert(summary)

    def _alert(self, summary: RunSummary) -> None:
        """Failure notification hook.

        Writes a marker file always, and posts to a webhook when
        ``TICKERLAKE_ALERT_WEBHOOK`` is set, so this works with Slack, Discord,
        or anything else that accepts a JSON POST -- without adding a dependency.
        """
        marker = self.paths.root / P.RUNS / "LAST_FAILURE.txt"
        try:
            marker.write_text(summary.render(), encoding="utf-8")
        except OSError:
            pass

        webhook = os.getenv("TICKERLAKE_ALERT_WEBHOOK")
        if not webhook:
            return
        try:
            import requests

            requests.post(
                webhook,
                json={
                    "text": f"TickerLake run {summary.run_id} FAILED\n"
                    f"stages: {', '.join(summary.failed_stages) or 'n/a'}\n"
                    f"{summary.fatal_error or ''}"
                },
                timeout=15,
            )
            log.info("failure alert posted to webhook")
        except Exception as exc:
            log.error("could not post failure alert: %s", exc)


def _build_tracker(
    config: Config, paths: P.DatasetPaths, writer: ParquetWriter
) -> MembershipTracker:
    history_start = config.get("universe.history_start")
    index_name = config.get("universe.index", "SP500")

    # ETFs are collected alongside the index but are not part of it. Only
    # collect_indices widens; index_name stays the survivorship scope.
    collect = [index_name]
    if config.get("etfs.enabled", False):
        collect.append(config.get("etfs.index_name", "ETF"))

    return MembershipTracker(
        paths=paths,
        writer=writer,
        index_name=index_name,
        history_start=date.fromisoformat(history_start) if history_start else None,
        silent_delist_threshold=int(config.get("universe.silent_delist_threshold_runs", 5)),
        post_removal_grace_days=int(config.get("universe.post_removal_grace_days", 30)),
        collect_indices=collect,
    )
