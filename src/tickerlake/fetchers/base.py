"""Common fetcher contract: fetch -> validate -> write.

Every source subclasses ``BaseFetcher`` and implements ``collect()``. The base
class owns everything that should be identical across sources: enablement
checks, timing, checkpoint plumbing, exception containment, and result
reporting. That is what makes adding Alpaca or Polygon later a matter of writing
one ``collect()`` rather than touching the orchestrator.

A fetcher must never raise out of ``run()``. A source that is down is a recorded
partial failure, not a dead pipeline -- the other five sources still have a job
to do.
"""

from __future__ import annotations

import logging
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any, ClassVar

from tickerlake.config import Config
from tickerlake.storage import paths as P
from tickerlake.storage.writer import ParquetWriter

if TYPE_CHECKING:
    from tickerlake.universe.membership import MembershipTracker

log = logging.getLogger(__name__)


@dataclass
class FetchResult:
    """Outcome of one fetcher's run. Serialised into the daily run summary."""

    stage: str
    dataset: str = ""
    ok: bool = True
    skipped: bool = False
    skip_reason: str | None = None
    rows_written: int = 0
    bytes_written: int = 0
    files_written: int = 0
    items_succeeded: int = 0
    items_failed: int = 0
    items_skipped: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        self.errors.append(str(message)[:1000])
        self.ok = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(str(message)[:1000])

    def record_write(self, write_result) -> None:
        if getattr(write_result, "skipped", False):
            return
        self.rows_written += write_result.rows_written
        self.bytes_written += write_result.bytes_written
        self.files_written += 1

    def summary_line(self) -> str:
        if self.skipped:
            return f"{self.stage}: SKIPPED ({self.skip_reason})"
        status = "OK" if self.ok else "FAILED"
        parts = [f"{self.stage}: {status}"]
        if self.rows_written:
            parts.append(f"{self.rows_written:,} rows")
        if self.files_written:
            parts.append(f"{self.files_written} file(s)")
        if self.bytes_written:
            parts.append(f"{self.bytes_written / 1_048_576:.1f} MB")
        if self.items_succeeded or self.items_failed or self.items_skipped:
            parts.append(
                f"{self.items_succeeded} ok / {self.items_failed} failed / {self.items_skipped} skipped"
            )
        parts.append(f"{self.duration_seconds:.1f}s")
        if self.errors:
            parts.append(f"errors: {self.errors[0][:160]}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "dataset": self.dataset,
            "ok": self.ok,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "rows_written": self.rows_written,
            "bytes_written": self.bytes_written,
            "files_written": self.files_written,
            "items_succeeded": self.items_succeeded,
            "items_failed": self.items_failed,
            "items_skipped": self.items_skipped,
            "duration_seconds": round(self.duration_seconds, 2),
            # Truncated: a systemic failure would otherwise write a 500-entry list.
            "errors": self.errors[:20],
            "warnings": self.warnings[:20],
            "details": self.details,
        }


class BaseFetcher(ABC):
    """Base class for all data source fetchers."""

    #: Stage name, matching the config section and pipeline.stages entry.
    name: ClassVar[str] = "base"
    #: Primary dataset this fetcher writes.
    dataset: ClassVar[str] = ""
    #: Attribute on Config.secrets that must be present, if any.
    requires_secret: ClassVar[str | None] = None

    def __init__(
        self,
        config: Config,
        paths: P.DatasetPaths,
        writer: ParquetWriter,
        tracker: MembershipTracker | None = None,
        symbols_override: list[str] | None = None,
    ) -> None:
        self.config = config
        self.paths = paths
        self.writer = writer
        self.tracker = tracker
        #: Explicit symbol list, bypassing the tracker. Used by --symbols and by
        #: the --limit smoke tests that validate a fetcher before a full run.
        self.symbols_override = (
            [s.strip().upper() for s in symbols_override] if symbols_override else None
        )
        self.log = logging.getLogger(f"tickerlake.fetchers.{self.name}")

    # ------------------------------------------------------------- interface

    @abstractmethod
    def collect(self, run_date: date, result: FetchResult) -> None:
        """Fetch, validate, and write this source's data for ``run_date``.

        Implementations mutate ``result`` as they go so that partial progress is
        still reported when something fails midway.
        """

    def is_enabled(self) -> tuple[bool, str | None]:
        """Whether this fetcher should run, and why not if it should not."""
        if not self.config.get(f"{self.name}.enabled", False):
            return False, "disabled in config"
        missing = self.config.missing_secret_for(self.name)
        if missing:
            return False, f"missing {missing} in environment"
        return True, None

    def cfg(self, key: str, default: Any = None) -> Any:
        """Read a key from this fetcher's own config section."""
        return self.config.get(f"{self.name}.{key}", default)

    # --------------------------------------------------------------- runner

    def run(self, run_date: date) -> FetchResult:
        """Template method. Never raises."""
        result = FetchResult(stage=self.name, dataset=self.dataset)

        enabled, reason = self.is_enabled()
        if not enabled:
            result.skipped = True
            result.skip_reason = reason
            self.log.info("skipping %s: %s", self.name, reason)
            return result

        started = time.monotonic()
        self.log.info("=== %s: starting for %s ===", self.name, run_date)
        try:
            self.collect(run_date, result)
        except KeyboardInterrupt:
            # Deliberate operator interrupt: record it and re-raise so the
            # orchestrator can stop cleanly rather than march on to the next stage.
            result.duration_seconds = time.monotonic() - started
            result.add_error("interrupted by user")
            self.log.warning("%s interrupted after %.1fs", self.name, result.duration_seconds)
            raise
        except Exception as exc:
            result.add_error(f"{type(exc).__name__}: {exc}")
            self.log.error("%s failed: %s", self.name, exc)
            self.log.debug("traceback:\n%s", traceback.format_exc())
        finally:
            result.duration_seconds = time.monotonic() - started

        self.log.info("=== %s ===", result.summary_line())
        return result
