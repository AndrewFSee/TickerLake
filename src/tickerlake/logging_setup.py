"""Logging: console + rotating file, with per-symbol detail kept out of the console."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

_CONFIGURED = False


class _ConsoleFormatter(logging.Formatter):
    """Compact console output; full detail still goes to the file handler."""

    LEVEL_TAG = {
        logging.DEBUG: "DBG",
        logging.INFO: "   ",
        logging.WARNING: "WRN",
        logging.ERROR: "ERR",
        logging.CRITICAL: "CRT",
    }

    def format(self, record: logging.LogRecord) -> str:
        tag = self.LEVEL_TAG.get(record.levelno, "???")
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        name = record.name.replace("tickerlake.", "")
        return f"{ts} {tag} {name:<22} {record.getMessage()}"


def setup_logging(
    data_root: Path,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    retention_days: int = 90,
    run_id: str | None = None,
) -> Path:
    """Configure root logging. Returns the path of the run's log file.

    Safe to call more than once; only the first call installs handlers.
    """
    global _CONFIGURED

    log_dir = data_root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = run_id or datetime.now(UTC).strftime("%Y%m%d")
    log_path = log_dir / f"tickerlake_{stamp}.log"

    if _CONFIGURED:
        return log_path

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    console.setFormatter(_ConsoleFormatter())
    root.addHandler(console)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=64 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s %(funcName)s:%(lineno)d | %(message)s"
        )
    )
    root.addHandler(file_handler)

    # These libraries are extremely chatty at DEBUG and drown out our own logs.
    # charset_normalizer in particular emits a line per HTTP response body, which
    # buries a 10,000-request options run in encoding-detection noise.
    for noisy in (
        "urllib3",
        "yfinance",
        "peewee",
        "curl_cffi",
        "matplotlib",
        "numexpr",
        "charset_normalizer",
        "requests",
        "bs4",
        "asyncio",
        "fsspec",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _prune_old_logs(log_dir, retention_days)
    _CONFIGURED = True
    return log_path


def _prune_old_logs(log_dir: Path, retention_days: int) -> None:
    if retention_days <= 0:
        return
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    for path in log_dir.glob("tickerlake_*.log*"):
        try:
            if datetime.fromtimestamp(path.stat().st_mtime, tz=UTC) < cutoff:
                path.unlink()
        except OSError:
            # A locked or already-removed log file is not worth failing a run over.
            continue


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
