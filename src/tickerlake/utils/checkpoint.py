"""Resumable run state.

A 500-symbol options pull is long enough that failing at symbol 380 and starting
over is unacceptable. Two independent mechanisms make a run resumable:

1. **Output-file existence.** A symbol whose Parquet file already exists for
   today's snapshot is done, full stop. This survives even losing the checkpoint
   file, and is the authoritative signal.
2. **This checkpoint.** Records *why* symbols are not done -- failure reasons,
   attempt counts, deliberate skips -- which file existence cannot express, and
   drives the end-of-run retry pass and the summary.

Saves are atomic (tmp + replace) so a crash during a checkpoint write cannot
leave unparseable JSON that breaks the next resume.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class Checkpoint:
    """JSON-backed progress record for one stage on one run date."""

    path: Path
    stage: str
    run_date: date
    completed: dict[str, dict[str, Any]] = field(default_factory=dict)
    failed: dict[str, dict[str, Any]] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    started_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------ persistence

    @classmethod
    def load_or_create(cls, path: Path, stage: str, run_date: date) -> Checkpoint:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                cp = cls(
                    path=path,
                    stage=data.get("stage", stage),
                    run_date=date.fromisoformat(data.get("run_date", run_date.isoformat())),
                    completed=data.get("completed", {}),
                    failed=data.get("failed", {}),
                    skipped=data.get("skipped", {}),
                    started_at=data.get("started_at", _utcnow()),
                    meta=data.get("meta", {}),
                )
                log.info(
                    "resuming %s from checkpoint: %d done, %d failed, %d skipped",
                    stage,
                    len(cp.completed),
                    len(cp.failed),
                    len(cp.skipped),
                )
                return cp
            except (json.JSONDecodeError, ValueError, OSError) as exc:
                log.warning("checkpoint %s unreadable (%s); starting fresh", path.name, exc)
        return cls(path=path, stage=stage, run_date=run_date)

    def save(self) -> None:
        self.updated_at = _utcnow()
        payload = {
            "stage": self.stage,
            "run_date": self.run_date.isoformat(),
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "counts": {
                "completed": len(self.completed),
                "failed": len(self.failed),
                "skipped": len(self.skipped),
            },
            "completed": self.completed,
            "failed": self.failed,
            "skipped": self.skipped,
            "meta": self.meta,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:
            log.error("failed to save checkpoint %s: %s", self.path.name, exc)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    # --------------------------------------------------------------- marking

    def mark_completed(self, key: str, **details: Any) -> None:
        self.completed[key] = {"at": _utcnow(), **details}
        self.failed.pop(key, None)
        self.skipped.pop(key, None)

    def mark_failed(self, key: str, error: str, **details: Any) -> None:
        prior = self.failed.get(key, {})
        self.failed[key] = {
            "at": _utcnow(),
            "error": str(error)[:500],
            "attempts": int(prior.get("attempts", 0)) + 1,
            **details,
        }

    def mark_skipped(self, key: str, reason: str) -> None:
        self.skipped[key] = reason
        self.failed.pop(key, None)

    # --------------------------------------------------------------- queries

    def is_done(self, key: str) -> bool:
        """Completed or deliberately skipped -- either way, do not retry this run."""
        return key in self.completed or key in self.skipped

    def attempts(self, key: str) -> int:
        return int(self.failed.get(key, {}).get("attempts", 0))

    def pending(self, keys: Iterable[str]) -> list[str]:
        return [k for k in keys if not self.is_done(k)]

    def retryable(self, max_attempts: int) -> list[str]:
        """Failed keys that have not yet exhausted their attempt budget."""
        return [k for k, v in self.failed.items() if int(v.get("attempts", 0)) < max_attempts]

    @property
    def total_seen(self) -> int:
        return len(self.completed) + len(self.failed) + len(self.skipped)

    def rows_written(self) -> int:
        return sum(int(v.get("rows", 0) or 0) for v in self.completed.values())

    def summary(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "run_date": self.run_date.isoformat(),
            "completed": len(self.completed),
            "failed": len(self.failed),
            "skipped": len(self.skipped),
            "rows": self.rows_written(),
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            # Cap the echoed detail: 500 failures should not bloat the run summary.
            "failed_symbols": sorted(self.failed)[:50],
            "skipped_symbols": sorted(self.skipped)[:50],
            **self.meta,
        }

    def clear_failures(self) -> None:
        self.failed.clear()
