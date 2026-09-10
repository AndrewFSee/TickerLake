"""Tests for IEX DEEP book reconstruction and 1-minute consolidation.

The DEEP binary layout is exercised with synthetic messages built to the same
spec the parser reads, so a wrong offset fails here rather than silently
producing plausible-looking prices from a real 11 GB file.
"""

from __future__ import annotations

import struct
from datetime import UTC, datetime

import pytest

from tickerlake.fetchers.iex_deep import (
    MSG_PRICE_LEVEL_BUY,
    MSG_PRICE_LEVEL_SELL,
    MSG_TRADE_REPORT,
    PRICE_SCALE,
    Book,
    DeepReconstructor,
    MinuteAccumulator,
    _in_session,
)

MINUTE_NS = 60_000_000_000


def _price_level(symbol: str, price: float, size: int, ts_ns: int, is_buy: bool) -> bytes:
    """A DEEP Price Level Update, 30 bytes."""
    return struct.pack(
        "<BBq8sIq",
        MSG_PRICE_LEVEL_BUY if is_buy else MSG_PRICE_LEVEL_SELL,
        0,
        ts_ns,
        symbol.ljust(8).encode("ascii"),
        size,
        int(round(price * PRICE_SCALE)),
    )


def _trade(symbol: str, price: float, size: int, ts_ns: int) -> bytes:
    return struct.pack(
        "<BBq8sIq",
        MSG_TRADE_REPORT,
        0,
        ts_ns,
        symbol.ljust(8).encode("ascii"),
        size,
        int(round(price * PRICE_SCALE)),
    )


def _payload(messages: list[bytes]) -> bytes:
    """Wrap messages in an IEX-TP segment: 40-byte header, then length-prefixed bodies."""
    header = bytearray(40)
    struct.pack_into("<H", header, 10, len(messages))
    body = b"".join(struct.pack("<H", len(m)) + m for m in messages)
    return bytes(header) + body


# ------------------------------------------------------------------- book


def test_book_tracks_both_sides():
    book = Book()
    book.apply(True, 1000000, 100)
    book.apply(True, 999900, 200)
    book.apply(False, 1000100, 150)

    assert book.best == (1000000, 1000100)
    bids, asks = book.levels(5)
    assert [p for p, _ in bids] == [1000000, 999900], "bids descend"
    assert [p for p, _ in asks] == [1000100]


def test_zero_size_removes_a_level():
    """Size 0 is a deletion in DEEP, not a level resting with no shares."""
    book = Book()
    book.apply(True, 1000000, 100)
    assert book.best[0] == 1000000

    book.apply(True, 1000000, 0)
    assert book.best[0] is None
    assert 1000000 not in book.bids


def test_resend_replaces_rather_than_accumulates():
    book = Book()
    book.apply(True, 1000000, 100)
    book.apply(True, 1000000, 250)
    assert book.bids[1000000] == 250, "a price level update is absolute, not a delta"


def test_levels_are_capped_at_requested_depth():
    book = Book()
    for i in range(20):
        book.apply(True, 1000000 - i * 100, 10)
        book.apply(False, 1000100 + i * 100, 10)
    bids, asks = book.levels(5)
    assert len(bids) == 5 and len(asks) == 5


# ------------------------------------------------------- time weighting


def test_time_weighting_favours_the_longer_standing_state():
    """A spread that held 9 seconds must dominate one that held 1."""
    book = Book()
    acc = MinuteAccumulator()

    book.apply(True, 1000000, 100)
    book.apply(False, 1000100, 100)  # $0.01 spread
    acc.observe(book, 0)
    acc.observe(book, 9_000_000_000)  # that spread holds 9s

    twa = acc.twa("weighted_spread")
    assert twa == pytest.approx(0.01, abs=1e-9)

    # Widening means *removing* the inside ask, not adding a level behind it:
    # the book legitimately holds both, and the best ask would not move.
    book.apply(False, 1000100, 0)
    book.apply(False, 1010000, 100)  # now $1.00 wide
    acc.observe(book, 10_000_000_000)  # holds 1s

    # 9s at $0.01 and 1s at $1.00 -> (0.09 + 1.00) / 10
    assert acc.twa("weighted_spread") == pytest.approx(0.109, abs=1e-6)

    acc.observe(book, 19_000_000_000)  # 9 more seconds at $1.00
    assert acc.twa("weighted_spread") > 0.5, "the wide state now dominates"


def test_no_quoted_time_yields_none_not_zero():
    acc = MinuteAccumulator()
    assert acc.twa("weighted_spread") is None, "absent is not the same as zero"


def test_crossed_book_is_not_credited():
    """A crossed book (bid >= ask) is a transient artefact, not a real spread."""
    book = Book()
    acc = MinuteAccumulator()
    book.apply(True, 1000100, 100)
    book.apply(False, 1000000, 100)  # crossed
    acc.observe(book, 0)
    acc.observe(book, 5_000_000_000)
    assert acc.twa("weighted_spread") is None


# ------------------------------------------------------------- decoding


def _reconstructor(symbols=("AAPL",), **kwargs):
    return DeepReconstructor(list(symbols), depth=5, market_hours_only=False, **kwargs)


def test_price_level_message_decodes_to_the_right_fields():
    r = _reconstructor()
    ts = 14 * 3600 * 1_000_000_000  # 14:00 UTC
    r._handle_payload(_payload([_price_level("AAPL", 315.07, 100, ts, True)]))

    book = r.books[b"AAPL    "]
    assert book.bids == {int(315.07 * PRICE_SCALE): 100}
    assert r.stats["messages"] == 1
    assert r.stats["updates"] == 1


def test_untracked_symbols_are_counted_but_not_booked():
    r = _reconstructor(("AAPL",))
    ts = 14 * 3600 * 1_000_000_000
    r._handle_payload(
        _payload([
            _price_level("AAPL", 315.0, 100, ts, True),
            _price_level("TSLA", 400.0, 100, ts, True),
        ])
    )
    assert r.stats["updates"] == 2, "every update is counted"
    assert len(r.books) == 1, "but only tracked symbols get a book"
    assert r.books[b"AAPL    "].bids


def test_trade_reports_accumulate_volume_and_notional():
    r = _reconstructor()
    ts = 14 * 3600 * 1_000_000_000
    r._handle_payload(
        _payload([_trade("AAPL", 315.50, 100, ts), _trade("AAPL", 315.60, 200, ts + 1000)])
    )
    acc = r.acc[b"AAPL    "]
    assert acc.trades == 2
    assert acc.trade_volume == 300
    assert acc.trade_notional == pytest.approx(315.50 * 100 + 315.60 * 200)


def test_truncated_payload_does_not_raise():
    r = _reconstructor()
    good = _payload([_price_level("AAPL", 315.0, 100, 0, True)])
    r._handle_payload(good[:-5])  # cut mid-message
    assert r.stats["messages"] <= 1


def test_message_count_larger_than_payload_is_survivable():
    header = bytearray(40)
    struct.pack_into("<H", header, 10, 99)  # claims 99 messages, supplies none
    _reconstructor()._handle_payload(bytes(header))


# ------------------------------------------------------------- snapshots


def test_minute_rollover_emits_a_snapshot():
    r = _reconstructor()
    base = 14 * 3600 * 1_000_000_000  # 14:00 UTC

    r._handle_payload(
        _payload([
            _price_level("AAPL", 315.00, 100, base, True),
            _price_level("AAPL", 315.10, 200, base + 1_000_000_000, False),
        ])
    )
    assert r.rows == [], "nothing emitted until the minute turns"

    r._handle_payload(_payload([_price_level("AAPL", 315.01, 100, base + MINUTE_NS, True)]))
    assert len(r.rows) == 1

    row = r.rows[0]
    assert row["symbol"] == "AAPL"
    assert row["best_bid"] == pytest.approx(315.00)
    assert row["best_ask"] == pytest.approx(315.10)
    assert row["spread"] == pytest.approx(0.10)
    assert row["mid"] == pytest.approx(315.05)
    assert row["n_updates"] == 2


def test_microprice_leans_toward_the_thinner_side():
    r = _reconstructor()
    base = 14 * 3600 * 1_000_000_000
    r._handle_payload(
        _payload([
            _price_level("AAPL", 100.00, 900, base, True),   # heavy bid
            _price_level("AAPL", 100.10, 100, base, False),  # thin ask
        ])
    )
    r._handle_payload(_payload([_price_level("AAPL", 100.00, 900, base + MINUTE_NS, True)]))

    row = r.rows[0]
    # Size-weighted toward the thin side, so above the mid of 100.05.
    assert row["microprice"] > row["mid"]
    assert row["imbalance_l1"] == pytest.approx((900 - 100) / 1000)


def test_depth_columns_are_padded_when_the_book_is_shallow():
    r = _reconstructor()
    base = 14 * 3600 * 1_000_000_000
    r._handle_payload(
        _payload([
            _price_level("AAPL", 100.00, 100, base, True),
            _price_level("AAPL", 100.10, 100, base, False),
        ])
    )
    r._handle_payload(_payload([_price_level("AAPL", 100.00, 100, base + MINUTE_NS, True)]))

    row = r.rows[0]
    assert row["bid_px_1"] == pytest.approx(100.00)
    assert row["bid_px_2"] is None, "absent levels are null, not zero"
    assert row["ask_px_5"] is None


def test_symbols_with_no_activity_are_omitted():
    r = _reconstructor(("AAPL", "MSFT"))
    base = 14 * 3600 * 1_000_000_000
    r._handle_payload(_payload([_price_level("AAPL", 100.0, 100, base, True)]))
    r._handle_payload(_payload([_price_level("AAPL", 100.0, 100, base + MINUTE_NS, True)]))

    # MSFT never quoted; a zero row would invent quiet that was never observed.
    assert {row["symbol"] for row in r.rows} == {"AAPL"}


def test_market_hours_filter_drops_overnight_minutes():
    r = DeepReconstructor(["AAPL"], depth=5, market_hours_only=True)
    overnight = 6 * 3600 * 1_000_000_000  # 06:00 UTC = 01:00 ET
    r._handle_payload(_payload([_price_level("AAPL", 100.0, 100, overnight, True)]))
    r._handle_payload(_payload([_price_level("AAPL", 100.0, 100, overnight + MINUTE_NS, True)]))
    assert r.rows == []


def test_session_bounds():
    def et(hour, minute):
        return datetime(2026, 9, 9, hour, minute, tzinfo=UTC)

    assert not _in_session(et(9, 29))
    assert _in_session(et(9, 30)), "the open is inside the session"
    assert _in_session(et(15, 59))
    assert not _in_session(et(16, 0)), "the close is exclusive"
