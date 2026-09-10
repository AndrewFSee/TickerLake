"""IEX DEEP: reconstruct order books and consolidate to 1-minute snapshots.

Deliberately **not** part of the daily pipeline. Each trading day is an 11.5 GB
compressed pcapng file, so a nightly stage would cost ~345 GB/month of bandwidth
for depth data describing ~2.5% of consolidated volume. Instead this is an
on-demand research tool: name a date and some symbols, pay ~20 minutes once, and
keep a few MB of 1-minute book snapshots.

There is no urgency to run it daily. Unlike option chains -- which Yahoo serves
live-only, so an uncaptured day is lost forever -- the IEX archive goes back to
2017-05-15 and does not expire. Any past day can be reconstructed whenever a
question actually needs it.

How it works
------------
Stream-decompress the gzip, walk the pcapng blocks, pull IEX-TP payloads out of
each UDP packet, decode DEEP messages, and apply Price Level Updates to a book
per tracked symbol. Nothing is written to disk except the final snapshots; the
48 GB of decompressed stream passes through memory in chunks.

Why time-weighted, not point-in-time
------------------------------------
A book snapshot taken exactly at 09:31:00.000 is one instant out of ~60 seconds
of quoting, and can easily land on a momentary wide spread that was never
representative. Every minute therefore carries **both**: the state at the
boundary, and time-weighted averages across the whole minute, where each book
state is weighted by how long it actually stood. The message count comes free
and is a genuine microstructure activity measure available nowhere else here.

Format notes, learned the hard way
----------------------------------
The files are **pcapng**, not classic pcap -- the magic is ``0x0a0d0d0a`` and
the structure is typed blocks, not a flat header plus records. Reading them as
classic pcap yields plausible-looking garbage rather than an error.
"""

from __future__ import annotations

import logging
import struct
import time
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from tickerlake.storage import paths as P
from tickerlake.storage.writer import ParquetWriter

log = logging.getLogger(__name__)

SOURCE = "iex_deep"
HIST_INDEX_URL = "https://iextrading.com/api/1.0/hist"
USER_AGENT = "TickerLake/0.1 (research)"

# pcapng block types.
BLOCK_ENHANCED_PACKET = 0x00000006
PCAPNG_MAGIC = 0x0A0D0D0A

# DEEP message types.
MSG_PRICE_LEVEL_BUY = 0x38
MSG_PRICE_LEVEL_SELL = 0x35
MSG_TRADE_REPORT = 0x54
MSG_SYSTEM_EVENT = 0x53

# Ethernet(14) + IPv4(20) + UDP(8). IEX multicast is IPv4/UDP throughout.
UDP_PAYLOAD_OFFSET = 42
# IEX-TP v1 header length, before the first message block.
IEXTP_HEADER_LEN = 40

PRICE_SCALE = 10_000  # DEEP prices are signed integers with 4 implied decimals


@dataclass
class Book:
    """One symbol's resting depth, price -> size, per side."""

    bids: dict[int, int] = field(default_factory=dict)
    asks: dict[int, int] = field(default_factory=dict)

    def apply(self, is_buy: bool, price: int, size: int) -> None:
        side = self.bids if is_buy else self.asks
        if size == 0:
            side.pop(price, None)
        else:
            side[price] = size

    def levels(self, depth: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        bids = sorted(self.bids.items(), key=lambda kv: -kv[0])[:depth]
        asks = sorted(self.asks.items(), key=lambda kv: kv[0])[:depth]
        return bids, asks

    @property
    def best(self) -> tuple[int | None, int | None]:
        return (max(self.bids) if self.bids else None, min(self.asks) if self.asks else None)


@dataclass
class MinuteAccumulator:
    """Time-weighted integrals across one minute, plus activity counters."""

    weighted_spread: float = 0.0
    weighted_mid: float = 0.0
    weighted_imbalance: float = 0.0
    weighted_time: float = 0.0
    updates: int = 0
    trades: int = 0
    trade_volume: int = 0
    trade_notional: float = 0.0
    last_ts: int | None = None

    def observe(self, book: Book, ts_ns: int) -> None:
        """Credit the interval since the last change to the book state that held."""
        if self.last_ts is not None and ts_ns > self.last_ts:
            dt = (ts_ns - self.last_ts) / 1e9
            bid, ask = book.best
            if bid is not None and ask is not None and ask > bid:
                bid_sz = book.bids.get(bid, 0)
                ask_sz = book.asks.get(ask, 0)
                total = bid_sz + ask_sz
                self.weighted_spread += (ask - bid) / PRICE_SCALE * dt
                self.weighted_mid += (ask + bid) / (2 * PRICE_SCALE) * dt
                if total > 0:
                    self.weighted_imbalance += ((bid_sz - ask_sz) / total) * dt
                self.weighted_time += dt
        self.last_ts = ts_ns

    def twa(self, attr: str) -> float | None:
        if self.weighted_time <= 0:
            return None
        return getattr(self, attr) / self.weighted_time


class IEXDeepArchive:
    """Index of IEX's free historical feed files."""

    def __init__(self) -> None:
        self._index: dict[str, list[dict[str, Any]]] | None = None

    def index(self) -> dict[str, list[dict[str, Any]]]:
        if self._index is None:
            log.info("fetching IEX historical index")
            resp = requests.get(HIST_INDEX_URL, headers={"User-Agent": USER_AGENT}, timeout=120)
            resp.raise_for_status()
            self._index = resp.json()
        return self._index

    def deep_url(self, day: date) -> str | None:
        entries = self.index().get(day.strftime("%Y%m%d"))
        if not entries:
            return None
        for entry in entries:
            if entry.get("feed") == "DEEP":
                return entry.get("link")
        return None

    def available_dates(self) -> list[date]:
        out = []
        for key, entries in self.index().items():
            if any(e.get("feed") == "DEEP" for e in entries):
                try:
                    out.append(datetime.strptime(key, "%Y%m%d").date())
                except ValueError:
                    continue
        return sorted(out)


class DeepReconstructor:
    """Streams one DEEP file and emits 1-minute book snapshots."""

    def __init__(
        self,
        symbols: list[str],
        depth: int = 5,
        market_hours_only: bool = True,
        writer: ParquetWriter | None = None,
    ) -> None:
        # DEEP pads symbols to 8 bytes, so compare in that form and skip the
        # decode entirely for the ~10,400 symbols we do not care about.
        self.symbols = {s.strip().upper() for s in symbols}
        self.wanted = {s.ljust(8).encode("ascii"): s for s in self.symbols}
        self.depth = depth
        self.market_hours_only = market_hours_only
        self.writer = writer or ParquetWriter()

        self.books: dict[bytes, Book] = {k: Book() for k in self.wanted}
        self.acc: dict[bytes, MinuteAccumulator] = {k: MinuteAccumulator() for k in self.wanted}
        self.current_minute: int | None = None
        self.rows: list[dict[str, Any]] = []
        self.stats = {"messages": 0, "updates": 0, "trades": 0, "bytes": 0}

    # ------------------------------------------------------------------ run

    def run(self, url: str, day: date, progress_every: int = 25_000_000) -> pd.DataFrame:
        started = time.monotonic()
        last_report = 0

        for payload in self._iter_iextp_payloads(url):
            self._handle_payload(payload)
            if self.stats["messages"] - last_report >= progress_every:
                last_report = self.stats["messages"]
                elapsed = time.monotonic() - started
                log.info(
                    "%.0fM messages, %.0fM updates, %.1f GB read, %.0f min elapsed",
                    self.stats["messages"] / 1e6,
                    self.stats["updates"] / 1e6,
                    self.stats["bytes"] / 1e9,
                    elapsed / 60,
                )

        self._flush_minute()  # close the final minute
        df = pd.DataFrame(self.rows)
        log.info(
            "done in %.1f min: %d snapshots for %d symbols from %.0fM messages",
            (time.monotonic() - started) / 60,
            len(df),
            df["symbol"].nunique() if not df.empty else 0,
            self.stats["messages"] / 1e6,
        )
        return df

    # -------------------------------------------------------------- streaming

    def _iter_iextp_payloads(self, url: str) -> Iterator[bytes]:
        """Stream gzip -> pcapng blocks -> IEX-TP payloads, without buffering the file."""
        decompressor = zlib.decompressobj(zlib.MAX_WBITS | 16)
        buffer = bytearray()
        checked_magic = False

        with requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=600, stream=True
        ) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(1024 * 1024):
                if not chunk:
                    continue
                self.stats["bytes"] += len(chunk)
                buffer.extend(decompressor.decompress(chunk))

                if not checked_magic and len(buffer) >= 4:
                    magic = struct.unpack("<I", buffer[:4])[0]
                    if magic != PCAPNG_MAGIC:
                        raise ValueError(
                            f"expected pcapng (0x{PCAPNG_MAGIC:08x}), got 0x{magic:08x}. "
                            "IEX changed format, or this is not a DEEP file."
                        )
                    checked_magic = True

                offset = 0
                while len(buffer) - offset >= 12:
                    block_type, block_len = struct.unpack_from("<II", buffer, offset)
                    if block_len < 12 or block_len > 10_000_000:
                        # Desynchronised; the stream is unusable from here.
                        raise ValueError(f"implausible pcapng block length {block_len}")
                    if len(buffer) - offset < block_len:
                        break  # wait for more bytes
                    if block_type == BLOCK_ENHANCED_PACKET:
                        caplen = struct.unpack_from("<I", buffer, offset + 20)[0]
                        packet = bytes(buffer[offset + 28 : offset + 28 + caplen])
                        if len(packet) > UDP_PAYLOAD_OFFSET + IEXTP_HEADER_LEN:
                            yield packet[UDP_PAYLOAD_OFFSET:]
                    offset += block_len
                del buffer[:offset]

    # --------------------------------------------------------------- decoding

    def _handle_payload(self, payload: bytes) -> None:
        message_count = struct.unpack_from("<H", payload, 10)[0]
        pos = IEXTP_HEADER_LEN

        for _ in range(message_count):
            if pos + 2 > len(payload):
                return
            length = struct.unpack_from("<H", payload, pos)[0]
            pos += 2
            if length == 0 or pos + length > len(payload):
                return
            body = payload[pos : pos + length]
            pos += length
            self.stats["messages"] += 1

            kind = body[0]
            if kind in (MSG_PRICE_LEVEL_BUY, MSG_PRICE_LEVEL_SELL) and length >= 30:
                self._price_level(body, kind == MSG_PRICE_LEVEL_BUY)
            elif kind == MSG_TRADE_REPORT and length >= 30:
                self._trade(body)

    def _price_level(self, body: bytes, is_buy: bool) -> None:
        self.stats["updates"] += 1
        key = body[10:18]
        if key not in self.wanted:
            return

        ts_ns = struct.unpack_from("<q", body, 2)[0]
        self._maybe_roll_minute(ts_ns)

        size = struct.unpack_from("<I", body, 18)[0]
        price = struct.unpack_from("<q", body, 22)[0]

        book = self.books[key]
        accumulator = self.acc[key]
        # Credit the elapsed interval to the state that held *before* this
        # update, then apply it. Doing it the other way round attributes the new
        # state to time it had not yet existed for.
        accumulator.observe(book, ts_ns)
        book.apply(is_buy, price, size)
        accumulator.updates += 1

    def _trade(self, body: bytes) -> None:
        key = body[10:18]
        if key not in self.wanted:
            return
        ts_ns = struct.unpack_from("<q", body, 2)[0]
        self._maybe_roll_minute(ts_ns)
        size = struct.unpack_from("<I", body, 18)[0]
        price = struct.unpack_from("<q", body, 22)[0]

        accumulator = self.acc[key]
        accumulator.trades += 1
        accumulator.trade_volume += size
        accumulator.trade_notional += size * price / PRICE_SCALE
        self.stats["trades"] += 1

    # ---------------------------------------------------------------- minutes

    def _maybe_roll_minute(self, ts_ns: int) -> None:
        minute = ts_ns // 60_000_000_000
        if self.current_minute is None:
            self.current_minute = minute
        elif minute > self.current_minute:
            self._flush_minute()
            self.current_minute = minute

    def _flush_minute(self) -> None:
        if self.current_minute is None:
            return
        stamp = datetime.fromtimestamp(self.current_minute * 60, tz=UTC)
        eastern = stamp.astimezone(_EASTERN)

        if self.market_hours_only and not _in_session(eastern):
            self._reset_accumulators()
            return

        for key, symbol in self.wanted.items():
            book = self.books[key]
            accumulator = self.acc[key]
            # A minute with no activity and no resting book is genuinely absent,
            # not zero -- emitting a row would invent quiet where there was none.
            if accumulator.updates == 0 and not book.bids and not book.asks:
                continue
            self.rows.append(self._snapshot(symbol, stamp, eastern, book, accumulator))
        self._reset_accumulators()

    def _snapshot(
        self, symbol: str, stamp: datetime, eastern: datetime, book: Book, acc: MinuteAccumulator
    ) -> dict[str, Any]:
        bids, asks = book.levels(self.depth)
        best_bid = bids[0][0] / PRICE_SCALE if bids else None
        best_ask = asks[0][0] / PRICE_SCALE if asks else None

        row: dict[str, Any] = {
            "symbol": symbol,
            "datetime": stamp,
            "date": eastern.date(),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": (best_ask - best_bid) if (best_bid and best_ask) else None,
            "mid": ((best_ask + best_bid) / 2) if (best_bid and best_ask) else None,
            "bid_depth": float(sum(size for _, size in bids)),
            "ask_depth": float(sum(size for _, size in asks)),
            # Time-weighted across the whole minute, not sampled at its edge.
            "twa_spread": acc.twa("weighted_spread"),
            "twa_mid": acc.twa("weighted_mid"),
            "twa_imbalance": acc.twa("weighted_imbalance"),
            "quoted_seconds": round(acc.weighted_time, 3),
            "n_updates": acc.updates,
            "n_trades": acc.trades,
            "trade_volume": acc.trade_volume,
            "trade_notional": round(acc.trade_notional, 4),
            "source": SOURCE,
            "ingested_at": datetime.now(UTC),
        }

        top_bid_size = bids[0][1] if bids else 0
        top_ask_size = asks[0][1] if asks else 0
        total = top_bid_size + top_ask_size
        row["imbalance_l1"] = ((top_bid_size - top_ask_size) / total) if total else None
        # Microprice: the size-weighted mid. Leans toward the side with less
        # resting size, and predicts the next trade price better than the mid.
        if best_bid and best_ask and total:
            row["microprice"] = (best_bid * top_ask_size + best_ask * top_bid_size) / total
        else:
            row["microprice"] = None

        for i in range(self.depth):
            row[f"bid_px_{i + 1}"] = bids[i][0] / PRICE_SCALE if i < len(bids) else None
            row[f"bid_sz_{i + 1}"] = float(bids[i][1]) if i < len(bids) else None
            row[f"ask_px_{i + 1}"] = asks[i][0] / PRICE_SCALE if i < len(asks) else None
            row[f"ask_sz_{i + 1}"] = float(asks[i][1]) if i < len(asks) else None
        return row

    def _reset_accumulators(self) -> None:
        for key in self.wanted:
            self.acc[key] = MinuteAccumulator()


# ------------------------------------------------------------------ helpers

try:
    from zoneinfo import ZoneInfo

    _EASTERN = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - tzdata always present via pandas
    _EASTERN = UTC


def _in_session(eastern: datetime) -> bool:
    """Regular US equity session, 09:30-16:00 ET."""
    minutes = eastern.hour * 60 + eastern.minute
    return 9 * 60 + 30 <= minutes < 16 * 60


def reconstruct_day(
    day: date,
    symbols: list[str],
    data_root: Path,
    depth: int = 5,
    market_hours_only: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Download one DEEP day, rebuild books, and write 1-minute snapshots."""
    paths = P.DatasetPaths(data_root)
    out_path = paths.book_snapshot_file(day)
    if out_path.exists() and not overwrite:
        existing = pd.read_parquet(out_path)
        log.info(
            "%s already reconstructed (%d rows); pass overwrite=True to redo", day, len(existing)
        )
        return {"date": day.isoformat(), "rows": len(existing), "skipped": True}

    archive = IEXDeepArchive()
    url = archive.deep_url(day)
    if url is None:
        available = archive.available_dates()
        raise ValueError(
            f"no DEEP file for {day}. The archive covers {available[0]} to {available[-1]} "
            "(trading days only)."
        )

    log.info(
        "reconstructing %s for %d symbol(s) at depth %d - expect ~20 min and ~11 GB of download",
        day,
        len(symbols),
        depth,
    )
    reconstructor = DeepReconstructor(symbols, depth=depth, market_hours_only=market_hours_only)
    df = reconstructor.run(url, day)

    if df.empty:
        log.warning("no snapshots produced for %s - were the symbols traded on IEX that day?", day)
        return {"date": day.isoformat(), "rows": 0, "skipped": False}

    write = reconstructor.writer.write(df, P.BOOK_SNAPSHOTS, out_path, mode="overwrite")
    return {
        "date": day.isoformat(),
        "rows": write.rows_written,
        "symbols": int(df["symbol"].nunique()),
        "bytes_downloaded": reconstructor.stats["bytes"],
        "messages": reconstructor.stats["messages"],
        "path": str(out_path),
        "skipped": False,
    }
