"""yfinance options chain fetcher -- daily full-chain snapshots.

Why snapshots
-------------
Yahoo exposes only the *live* chain. There is no history endpoint. The only way
to build an options history is to snapshot every day and never lose one, which
makes this the fetcher where resumability actually matters: ~500 symbols x ~20
expirations is ~10,000 requests against an unauthenticated, undocumented API.

Resumability
------------
Two independent signals, so a crash never costs more than the symbol in flight:

* The per-symbol output file existing means that symbol is done. This survives
  losing the checkpoint entirely.
* The checkpoint records failures, attempt counts and deliberate skips, which
  file existence cannot express, and drives the end-of-run retry pass.

Failure policy
--------------
A symbol that fails is logged, recorded, and stepped over -- one delisted ticker
must not cost the other 499. Failures are retried once at the end of the run,
after the throttle has had time to recover. But if failures exceed
``abort_failure_rate``, the run stops: that pattern means Yahoo has blocked the
IP, and continuing only deepens the block.
"""

from __future__ import annotations

import sys
import time
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from tickerlake.fetchers.base import BaseFetcher, FetchResult
from tickerlake.storage import paths as P
from tickerlake.universe.sources import to_yahoo_symbol
from tickerlake.utils.checkpoint import Checkpoint
from tickerlake.utils.throttle import AdaptiveThrottle, is_rate_limit_error

SOURCE = "yfinance"

_COLUMN_MAP = {
    "contractSymbol": "contract_symbol",
    "lastTradeDate": "last_trade_date",
    "strike": "strike",
    "lastPrice": "last_price",
    "bid": "bid",
    "ask": "ask",
    "change": "change",
    "percentChange": "percent_change",
    "volume": "volume",
    "openInterest": "open_interest",
    "impliedVolatility": "implied_volatility",
    "inTheMoney": "in_the_money",
    "contractSize": "contract_size",
    "currency": "currency",
}


class SymbolHasNoOptions(Exception):
    """The symbol exists but lists no option expirations. A skip, not a failure."""


class RunAborted(Exception):
    """Failure rate crossed the abort threshold -- almost certainly an IP block."""


class YFinanceOptionsFetcher(BaseFetcher):
    """Full daily option chain snapshots for the current index members."""

    name = "options"
    requires_trading_day = True
    dataset = P.OPTIONS_CHAINS

    def __init__(self, *args: Any, limit: int | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        #: Cap the symbol count. Used to smoke-test on 10 before committing to 500.
        self.limit = limit
        #: Running bid-coverage tally, used to detect off-hours snapshots.
        self._bid_total = 0
        self._bid_present = 0

    # ------------------------------------------------------------------ main

    def collect(self, run_date: date, result: FetchResult) -> None:
        symbols = self._symbols(run_date)
        if not symbols:
            result.add_warning("no symbols to fetch; is the membership table seeded?")
            return

        checkpoint = Checkpoint.load_or_create(
            self.paths.checkpoint_file(self.name, run_date), self.name, run_date
        )
        throttle = AdaptiveThrottle(
            base_interval=float(self.cfg("throttle_seconds", 0.6)),
            jitter_pct=float(self.cfg("jitter_pct", 0.35)),
            slowdown_factor=float(self.cfg("rate_limit_slowdown_factor", 2.0)),
            backoff_seconds=float(self.cfg("rate_limit_backoff_seconds", 90)),
            max_backoff_seconds=float(self.cfg("rate_limit_max_backoff_seconds", 900)),
        )
        inter_symbol = float(self.cfg("inter_symbol_seconds", 0.4))

        pending = self._resolve_pending(symbols, checkpoint, run_date)
        already_done = len(symbols) - len(pending)
        if already_done:
            self.log.info(
                "resuming: %d/%d symbols already have a snapshot for %s",
                already_done,
                len(symbols),
                run_date,
            )

        started = time.monotonic()
        aborted = False
        try:
            self._process(pending, run_date, checkpoint, throttle, inter_symbol, result)
        except RunAborted as exc:
            aborted = True
            result.add_error(str(exc))
            self.log.error("ABORTED: %s", exc)
        except KeyboardInterrupt:
            checkpoint.save()
            self.log.warning("interrupted; checkpoint saved, re-run to resume")
            raise
        finally:
            checkpoint.save()

        # Retry pass: failures are often transient, and the throttle has by now
        # backed off, so a second attempt at the end is usually cheap and works.
        if not aborted and self.cfg("retry_failed_at_end", True):
            retryable = checkpoint.retryable(int(self.cfg("max_retries_per_symbol", 3)))
            if retryable:
                self.log.info("retrying %d failed symbol(s) at end of run", len(retryable))
                try:
                    self._process(
                        sorted(retryable),
                        run_date,
                        checkpoint,
                        throttle,
                        inter_symbol,
                        result,
                        retry_pass=True,
                    )
                except RunAborted as exc:
                    result.add_error(f"retry pass aborted: {exc}")
                finally:
                    checkpoint.save()

        self._finalize(symbols, checkpoint, throttle, result, run_date, time.monotonic() - started)

    # -------------------------------------------------------------- symbols

    def _symbols(self, run_date: date) -> list[str]:
        """Current index members only.

        Unlike OHLCV, options are not fetched for delisted names: a removed
        company has no listed chain, so requesting one is pure rate-limit budget
        spent on a guaranteed empty response.
        """
        if self.symbols_override is not None:
            symbols = list(self.symbols_override)
        elif self.tracker is not None:
            symbols = self.tracker.current_members()
        else:
            return []
        if self.limit:
            symbols = symbols[: self.limit]
        return symbols

    def _resolve_pending(
        self, symbols: list[str], checkpoint: Checkpoint, run_date: date
    ) -> list[str]:
        """Drop symbols already done, trusting the output file over the checkpoint."""
        pending = []
        for symbol in symbols:
            if self.paths.options_symbol_file(run_date, symbol).exists():
                if symbol not in checkpoint.completed:
                    checkpoint.mark_completed(symbol, rows=None, note="output file already present")
                continue
            if checkpoint.is_done(symbol):
                continue
            pending.append(symbol)
        return pending

    # ------------------------------------------------------------ processing

    def _process(
        self,
        symbols: list[str],
        run_date: date,
        checkpoint: Checkpoint,
        throttle: AdaptiveThrottle,
        inter_symbol: float,
        result: FetchResult,
        retry_pass: bool = False,
    ) -> None:
        checkpoint_every = int(self.cfg("checkpoint_every", 10))
        abort_rate = float(self.cfg("abort_failure_rate", 0.4))
        max_retries = int(self.cfg("max_retries_per_symbol", 3))

        # Under Task Scheduler there is no TTY, and a progress bar would just
        # fill the log with control codes; fall back to periodic INFO lines.
        interactive = sys.stderr.isatty()
        total = len(symbols)
        started_at = time.monotonic()
        bar = tqdm(
            symbols,
            desc="options (retry)" if retry_pass else "options",
            unit="sym",
            ncols=100,
            leave=False,
            disable=not interactive,
        )
        for n, symbol in enumerate(bar, 1):
            if interactive:
                bar.set_postfix_str(
                    f"{symbol} ok={len(checkpoint.completed)} fail={len(checkpoint.failed)}"
                )
            elif n == 1 or n % 25 == 0 or n == total:
                rate = (time.monotonic() - started_at) / max(n - 1, 1)
                remaining = (total - n) * rate / 60
                self.log.info(
                    "options progress %d/%d (%.0f%%) ok=%d fail=%d skip=%d ~%.0f min left",
                    n,
                    total,
                    100 * n / total,
                    len(checkpoint.completed),
                    len(checkpoint.failed),
                    len(checkpoint.skipped),
                    remaining,
                )
            throttle.wait()
            try:
                rows = self._fetch_symbol(symbol, run_date, throttle)
            except SymbolHasNoOptions:
                checkpoint.mark_skipped(symbol, "no listed option expirations")
                self.log.debug("%s: no options listed", symbol)
            except KeyboardInterrupt:
                bar.close()
                raise
            except Exception as exc:
                if is_rate_limit_error(exc):
                    throttle.record_rate_limit()
                    checkpoint.mark_failed(symbol, f"rate limited: {exc}")
                else:
                    checkpoint.mark_failed(symbol, f"{type(exc).__name__}: {exc}")
                    self.log.debug("%s failed: %s", symbol, exc)
            else:
                throttle.record_success()
                checkpoint.mark_completed(
                    symbol, rows=rows["rows"], expirations=rows["expirations"]
                )

            if n % checkpoint_every == 0:
                checkpoint.save()

            # Only abort on a *persistent* pattern: transient failures that the
            # retry pass will clear should not kill a 500-symbol run.
            seen = checkpoint.total_seen
            if seen >= 25 and not retry_pass:
                hard_failed = len(
                    [
                        k
                        for k, v in checkpoint.failed.items()
                        if int(v.get("attempts", 0)) >= max_retries
                    ]
                )
                rate = (len(checkpoint.failed) + hard_failed) / max(seen, 1)
                if rate > abort_rate:
                    bar.close()
                    raise RunAborted(
                        f"failure rate {rate:.0%} exceeds abort threshold {abort_rate:.0%} "
                        f"after {seen} symbols ({len(checkpoint.failed)} failed). "
                        "This pattern means Yahoo is blocking, not that the tickers are bad. "
                        "Progress is checkpointed; re-run later to resume."
                    )

            if inter_symbol > 0:
                time.sleep(inter_symbol)

        bar.close()

    def _fetch_symbol(
        self, symbol: str, run_date: date, throttle: AdaptiveThrottle
    ) -> dict[str, int]:
        """Fetch every expiration for one symbol and write its snapshot file."""
        ticker = yf.Ticker(to_yahoo_symbol(symbol))

        expirations = list(ticker.options or ())
        if not expirations:
            raise SymbolHasNoOptions(symbol)

        wanted = self._filter_expirations(expirations, run_date)
        if not wanted:
            raise SymbolHasNoOptions(f"{symbol}: no expirations passed filters")

        now = datetime.now(UTC)
        frames: list[pd.DataFrame] = []
        underlying_price: float | None = None

        for expiry in wanted:
            throttle.wait()
            chain = ticker.option_chain(expiry)

            if underlying_price is None:
                underlying_price = _underlying_price(chain)

            expiry_date = date.fromisoformat(expiry)
            for side, frame in (("call", chain.calls), ("put", chain.puts)):
                if frame is None or frame.empty:
                    continue
                frames.append(self._normalize(frame, symbol, run_date, expiry_date, side, now))

        if not frames:
            raise SymbolHasNoOptions(f"{symbol}: all expirations returned empty chains")

        out = pd.concat(frames, ignore_index=True)
        out["underlying_price"] = underlying_price

        write = self.writer.write(
            out,
            P.OPTIONS_CHAINS,
            self.paths.options_symbol_file(run_date, symbol),
            mode="overwrite",
        )

        # Bid coverage is the single best signal that a snapshot was taken in a
        # usable window. Yahoo serves the full strike ladder around the clock but
        # only populates bids while quotes are live, so an off-hours snapshot
        # looks complete -- right row count, right strikes, plausible last prices
        # -- while carrying almost no tradeable quotes. Implied volatility is
        # then unrecoverable from it, and nothing else in the row says why.
        bids = pd.to_numeric(out["bid"], errors="coerce")
        with_bid = int((bids > 0).sum())
        self._bid_total += len(out)
        self._bid_present += with_bid

        return {
            "rows": write.rows_written,
            "expirations": len(wanted),
            "bid_coverage": round(with_bid / max(len(out), 1), 3),
        }

    def _normalize(
        self,
        frame: pd.DataFrame,
        symbol: str,
        run_date: date,
        expiry: date,
        side: str,
        now: datetime,
    ) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        for src, dst in _COLUMN_MAP.items():
            out[dst] = frame[src] if src in frame.columns else pd.NA
        out["symbol"] = symbol
        out["snapshot_date"] = run_date
        out["expiration"] = expiry
        out["option_type"] = side
        out["dte"] = (expiry - run_date).days
        out["source"] = SOURCE
        out["ingested_at"] = now
        return out

    # ---------------------------------------------------------- expirations

    def _filter_expirations(self, expirations: list[str], run_date: date) -> list[str]:
        """Apply the configured expiration filters. Defaults keep everything."""
        max_dte = self.cfg("max_dte")
        monthlies_only = bool(self.cfg("monthlies_only", False)) or not bool(
            self.cfg("include_weeklies", True)
        )

        kept = []
        for raw in expirations:
            try:
                expiry = date.fromisoformat(raw)
            except ValueError:
                self.log.debug("unparseable expiration %r; skipping", raw)
                continue
            if max_dte is not None and (expiry - run_date).days > int(max_dte):
                continue
            if monthlies_only and not _is_monthly(expiry):
                continue
            kept.append(raw)
        return kept

    # ---------------------------------------------------------------- report

    def _finalize(
        self,
        symbols: list[str],
        checkpoint: Checkpoint,
        throttle: AdaptiveThrottle,
        result: FetchResult,
        run_date: date,
        elapsed: float,
    ) -> None:
        result.items_succeeded = len(checkpoint.completed)
        result.items_failed = len(checkpoint.failed)
        result.items_skipped = len(checkpoint.skipped)

        # Count what actually landed on disk rather than trusting the tally: a
        # resumed run's earlier files were written by a previous process.
        files = sorted(self.paths.options_partition(run_date).glob("sym_*.parquet"))
        result.files_written = len(files)
        result.bytes_written = sum(f.stat().st_size for f in files)
        result.rows_written = checkpoint.rows_written()

        coverage = self._bid_present / self._bid_total if self._bid_total else None
        if coverage is not None:
            result.details["bid_coverage"] = round(coverage, 3)
            floor = float(self.cfg("min_bid_coverage", 0.25))
            if coverage < floor:
                result.add_warning(
                    f"only {coverage:.1%} of contracts carry a bid (expected >{floor:.0%}). "
                    "Yahoo populates bid/ask only while quotes are live, so this snapshot was "
                    "almost certainly taken outside market hours. The strikes and last prices "
                    "are still stored, but implied volatility cannot be solved from it."
                )
                self.log.warning(
                    "LOW BID COVERAGE %.1f%% for %s - snapshot likely taken outside "
                    "market hours (equity options quote 09:30-16:00 ET)",
                    coverage * 100,
                    run_date,
                )
            else:
                self.log.info(
                    "bid coverage %.1f%% (%d/%d contracts)",
                    coverage * 100,
                    self._bid_present,
                    self._bid_total,
                )

        result.details.update(
            {
                "symbols_requested": len(symbols),
                "snapshot_date": run_date.isoformat(),
                "throttle": throttle.stats(),
                "elapsed_minutes": round(elapsed / 60, 1),
                "failed_symbols": sorted(checkpoint.failed)[:50],
                "skipped_symbols": sorted(checkpoint.skipped)[:50],
                "checkpoint": str(checkpoint.path),
            }
        )
        if checkpoint.failed:
            result.add_warning(
                f"{len(checkpoint.failed)} symbol(s) failed: {', '.join(sorted(checkpoint.failed)[:15])}"
            )

        self.log.info(
            "options summary for %s: %d succeeded, %d failed, %d skipped, "
            "%d files, %.1f MB, %.1f min, %d rate-limit hit(s)",
            run_date,
            result.items_succeeded,
            result.items_failed,
            result.items_skipped,
            result.files_written,
            result.bytes_written / 1_048_576,
            elapsed / 60,
            throttle.rate_limit_hits,
        )


# ------------------------------------------------------------------ helpers


def _is_monthly(expiry: date) -> bool:
    """Standard monthly expiration: the third Friday of the month."""
    return expiry.weekday() == 4 and 15 <= expiry.day <= 21


def _underlying_price(chain: Any) -> float | None:
    """Pull the spot price out of the chain's underlying payload.

    Storing spot alongside the chain is what makes moneyness computable later
    without a second join -- and without the risk of joining to a *revised*
    close that was not what the chain was quoted against.
    """
    underlying = getattr(chain, "underlying", None)
    if not isinstance(underlying, dict):
        return None
    for key in ("regularMarketPrice", "lastPrice", "previousClose", "regularMarketPreviousClose"):
        value = underlying.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return None
