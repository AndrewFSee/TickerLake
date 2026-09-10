"""Tests for Black-Scholes pricing, IV solving, and quality gating.

The gating matters as much as the math here: the entire reason for computing our
own IV is to refuse to emit a number when the quote cannot support one.
"""

from __future__ import annotations

import math

import pytest

from tickerlake.analytics.black_scholes import (
    MAX_PLAUSIBLE_IV,
    MIN_VEGA_FOR_IV,
    QualityFlag,
    analyze_contract,
    bs_price,
    implied_volatility,
    price_and_greeks,
)

# ------------------------------------------------------------------- pricing


def test_matches_textbook_values():
    """Hull, S=100 K=100 T=1 r=5% q=0 sigma=20%."""
    assert bs_price(100, 100, 1, 0.05, 0, 0.20, True) == pytest.approx(10.4506, abs=1e-4)
    assert bs_price(100, 100, 1, 0.05, 0, 0.20, False) == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize(
    ("S", "K", "T", "r", "q"),
    [(100, 100, 1.0, 0.05, 0.0), (250, 200, 0.5, 0.04, 0.02), (37, 45, 2.0, 0.03, 0.015)],
)
def test_put_call_parity(S, K, T, r, q):
    call = bs_price(S, K, T, r, q, 0.3, True)
    put = bs_price(S, K, T, r, q, 0.3, False)
    parity = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert call - put == pytest.approx(parity, abs=1e-10)


def test_price_is_monotonic_in_volatility():
    prices = [bs_price(100, 100, 1, 0.04, 0, s, True) for s in (0.1, 0.2, 0.3, 0.5)]
    assert prices == sorted(prices)


def test_zero_time_collapses_to_intrinsic():
    assert bs_price(120, 100, 0, 0.05, 0, 0.3, True) == pytest.approx(20.0)
    assert bs_price(80, 100, 0, 0.05, 0, 0.3, False) == pytest.approx(20.0)


# -------------------------------------------------------------------- greeks


def test_greeks_match_finite_differences():
    S, K, T, r, q, sig = 100, 100, 1.0, 0.05, 0.0, 0.2
    g = price_and_greeks(S, K, T, r, q, sig, True)

    h = 1e-4
    fd_delta = (bs_price(S + h, K, T, r, q, sig, True) - bs_price(S - h, K, T, r, q, sig, True)) / (
        2 * h
    )
    assert g["delta"] == pytest.approx(fd_delta, abs=1e-6)

    # vega is reported per volatility *point*, hence the /100.
    fd_vega = (
        (bs_price(S, K, T, r, q, sig + h, True) - bs_price(S, K, T, r, q, sig - h, True))
        / (2 * h)
        / 100
    )
    assert g["vega"] == pytest.approx(fd_vega, abs=1e-6)


def test_call_and_put_delta_differ_by_one():
    call = price_and_greeks(100, 100, 1, 0.05, 0, 0.2, True)
    put = price_and_greeks(100, 100, 1, 0.05, 0, 0.2, False)
    assert call["delta"] - put["delta"] == pytest.approx(1.0, abs=1e-9)
    assert call["gamma"] == pytest.approx(put["gamma"], abs=1e-12)


def test_long_option_theta_is_negative():
    assert price_and_greeks(100, 100, 0.25, 0.04, 0, 0.3, True)["theta"] < 0


# ---------------------------------------------------- implied volatility


@pytest.mark.parametrize("sigma", [0.12, 0.25, 0.45, 0.80])
@pytest.mark.parametrize("K", [90, 100, 110])
@pytest.mark.parametrize("is_call", [True, False])
def test_iv_round_trip(sigma, K, is_call):
    price = bs_price(100, K, 0.5, 0.045, 0.015, sigma, is_call)
    solved = implied_volatility(100, K, 0.5, 0.045, 0.015, price, is_call)
    assert solved == pytest.approx(sigma, abs=1e-5)


def test_iv_returns_none_when_price_is_unreachable():
    # Far below the no-arbitrage floor: no volatility reproduces it.
    assert implied_volatility(100, 100, 1.0, 0.05, 0.0, 0.0001, True) is None


# --------------------------------------------------------- quality gating


def _tight_market(S, K, T, sigma, is_call, r=0.045, q=0.015):
    px = bs_price(S, K, T, r, q, sigma, is_call)
    return analyze_contract(S, K, T, r, q, is_call, bid=px * 0.999, ask=px * 1.001)


def test_healthy_contract_is_usable_and_accurate():
    res = _tight_market(100, 100, 0.25, 0.30, True)
    assert res.usable
    assert res.quality is QualityFlag.OK
    assert res.implied_volatility == pytest.approx(0.30, abs=1e-4)
    assert res.iv_uncertainty is not None and res.iv_uncertainty > 0


def test_zero_bid_is_refused():
    res = analyze_contract(100, 200, 0.02, 0.045, 0.0, True, bid=0.0, ask=0.05)
    assert not res.usable
    assert QualityFlag.ZERO_BID in res.quality
    assert res.implied_volatility is None, "a fictional mid must not produce an IV"


def test_crossed_market_is_refused():
    res = analyze_contract(100, 100, 0.25, 0.045, 0.0, True, bid=5.0, ask=4.5)
    assert QualityFlag.CROSSED_MARKET in res.quality
    assert not res.usable


def test_wide_spread_is_refused():
    res = analyze_contract(100, 100, 0.25, 0.045, 0.0, True, bid=1.0, ask=5.0)
    assert QualityFlag.WIDE_SPREAD in res.quality
    assert not res.usable


def test_below_intrinsic_is_refused():
    # A call quoted under S-K has no time value to solve against.
    res = analyze_contract(150, 100, 0.25, 0.045, 0.0, True, bid=40.0, ask=41.0)
    assert QualityFlag.BELOW_INTRINSIC in res.quality
    assert res.implied_volatility is None


def test_unidentifiable_when_vega_is_negligible():
    """Deep ITM, days to expiry: volatility cannot move the price.

    This is the exact failure mode that makes vendor IV fields untrustworthy --
    a solver will happily return its lower bound and present it as an answer.
    """
    res = _tight_market(100, 70, 0.019, 0.25, True)
    assert res.implied_volatility is None
    assert QualityFlag.UNIDENTIFIABLE in res.quality
    assert res.vega is not None and res.vega < MIN_VEGA_FOR_IV
    # Delta remains meaningful even though IV does not.
    assert res.delta is not None and res.delta > 0.95


def test_implausible_iv_is_flagged_not_silently_returned():
    """A deep-ITM American put quoted far above intrinsic.

    Black-Scholes is European, so the early-exercise premium has nowhere to go
    except into implied volatility, producing an absurd number.
    """
    res = analyze_contract(320.31, 420.0, 8 / 365, 0.042, 0.005, False, bid=155.90, ask=158.95)
    assert not res.usable, "an implausible IV must not be marked usable"
    if res.implied_volatility is not None:
        assert res.implied_volatility > MAX_PLAUSIBLE_IV
        assert QualityFlag.IMPLAUSIBLE_IV in res.quality


def test_expired_contract_is_refused():
    res = analyze_contract(100, 100, 0.0, 0.045, 0.0, True, bid=1.0, ask=1.1)
    assert QualityFlag.EXPIRED in res.quality
    assert not res.usable


def test_bad_input_is_refused_not_raised():
    res = analyze_contract(0, 0, 0.25, 0.045, 0.0, True, bid=1.0, ask=1.1)
    assert QualityFlag.BAD_INPUT in res.quality
    assert res.implied_volatility is None


def test_quality_flags_compose():
    """Flags are a set: a contract can fail in several ways at once."""
    res = analyze_contract(320, 200, 0.08, 0.045, 0.0, True, bid=0.0, ask=125.0)
    label = res.quality.to_label()
    assert "zero_bid" in label
    assert "|" in label, f"expected multiple flags, got {label!r}"


def test_iv_uncertainty_is_wider_when_vega_is_smaller():
    atm_long = _tight_market(100, 100, 1.0, 0.30, True)
    atm_short = _tight_market(100, 100, 7 / 365, 0.30, True)
    assert atm_short.vega < atm_long.vega
    assert atm_short.iv_uncertainty > atm_long.iv_uncertainty
