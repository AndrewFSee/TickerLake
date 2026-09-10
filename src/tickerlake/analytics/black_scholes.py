"""Black-Scholes pricing, implied volatility, and Greeks.

Why compute our own IV
----------------------
Yahoo's ``impliedVolatility`` is unusable on illiquid contracts -- a single
snapshot contains values of 0.00001 and 9.45 side by side. This is not a Yahoo
defect so much as an intrinsic one: Alpha Vantage reports 4.21 on comparable
deep-ITM contracts. When a contract has a zero bid, a 30%-wide spread, or trades
below intrinsic value, there is no well-defined implied volatility to report, and
every vendor papers over that with a number anyway.

Computing it ourselves gives three things a vendor field cannot:

1. **A quality verdict per contract.** ``QualityFlag`` says *why* a value is
   untrustworthy instead of silently emitting a number.
2. **Greeks**, which are better ML features than raw IV and which Yahoo does not
   provide at all.
3. **Reproducibility.** Chains are live-only, so a stored snapshot can never be
   re-fetched -- but it can be re-analysed. Recomputing with a better model later
   is a rerun over Parquet, not a lost opportunity.

Model choice and its limits
---------------------------
This is Black-Scholes-Merton with a continuous dividend yield: a **European**
model applied to **American** equity options. The gap is the early-exercise
premium, which is negligible for most out-of-the-money contracts and material
for deep-ITM puts and for calls immediately before a dividend. Those cases are
flagged (``DEEP_ITM``), not silently priced. A binomial American solver would
close the gap at roughly 50x the compute cost; if that trade becomes worthwhile,
only this module changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Flag, auto

import numpy as np
from scipy.optimize import brentq

# scipy.stats.norm.cdf is array-aware and pays a large per-call cost for it.
# The IV solver evaluates these two functions on scalars tens of times per
# contract, times ~850k contracts a day, so that overhead dominates the entire
# analytics stage. math.erf is a C builtin and gives identical values to double
# precision. Measured: ~8x faster end to end.
_INV_SQRT_2 = 1.0 / math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x * _INV_SQRT_2))


def _norm_pdf(x: float) -> float:
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


# Solver bounds. Below 0.1% vol a quote is indistinguishable from a stale print;
# above 500% the model has stopped describing anything real.
MIN_VOL = 1e-3
MAX_VOL = 5.0
_SOLVER_TOL = 1e-6
_SOLVER_MAXITER = 100

# A spread wider than this fraction of mid means the mid is not a real price.
MAX_SPREAD_RATIO = 0.50

# Minimum vega (price change per 1 volatility point) for IV to be identifiable.
# Below this, the option is priced at pure intrinsic and volatility has no
# measurable effect: any sigma reproduces the quote, so a solver will happily
# return its lower bound and call it an answer. That is exactly how vendors end
# up publishing 0.00001 as an implied volatility. At 0.001 a one-cent quote
# error moves IV by ~10 vol points, which is already the edge of usefulness.
MIN_VEGA_FOR_IV = 1e-3

# Typical minimum price increment for US equity options, used to translate vega
# into an IV uncertainty estimate.
OPTION_TICK = 0.01

# Ceiling above which a solved IV is not believable for a listed equity. Even
# biotech binaries and 0DTE rarely clear 300%. In practice a number above this
# means the European model is being asked to price something it cannot: most
# often a deep-ITM American put quoted above intrinsic, where the early-exercise
# premium has nowhere to go except into implied volatility.
MAX_PLAUSIBLE_IV = 3.0
# |log(K/S)| beyond this is deep enough that early exercise and quote quality
# both become serious concerns.
DEEP_ITM_LOG_MONEYNESS = 0.35


class QualityFlag(Flag):
    """Why a contract's implied volatility should or should not be trusted.

    A flag set, not a single verdict: a contract can be both zero-bid and
    below-intrinsic, and knowing which applies is the point of computing this.
    """

    OK = 0
    ZERO_BID = auto()  # no bid: nobody is buying, mid is fictional
    CROSSED_MARKET = auto()  # bid >= ask: stale or erroneous quote
    WIDE_SPREAD = auto()  # spread too wide for mid to mean anything
    BELOW_INTRINSIC = auto()  # price under intrinsic value: no time value to solve for
    NO_TIME_VALUE = auto()  # time value rounds to zero
    EXPIRED = auto()  # dte <= 0
    DEEP_ITM = auto()  # early-exercise premium likely material
    NO_SOLUTION = auto()  # solver found no root in [MIN_VOL, MAX_VOL]
    BAD_INPUT = auto()  # missing or nonsensical spot/strike/price
    UNIDENTIFIABLE = auto()  # vega ~ 0: price carries no information about vol
    IMPLAUSIBLE_IV = auto()  # solved, but outside any credible range for equities

    @property
    def is_tradeable_quote(self) -> bool:
        """True when the quote itself is sound enough to model."""
        bad = (
            QualityFlag.ZERO_BID
            | QualityFlag.CROSSED_MARKET
            | QualityFlag.WIDE_SPREAD
            | QualityFlag.BELOW_INTRINSIC
            | QualityFlag.NO_TIME_VALUE
            | QualityFlag.EXPIRED
            | QualityFlag.BAD_INPUT
            | QualityFlag.IMPLAUSIBLE_IV
        )
        return not (self & bad)

    def to_label(self) -> str:
        if self is QualityFlag.OK:
            return "ok"
        return "|".join(
            f.name.lower() for f in QualityFlag if f is not QualityFlag.OK and f in self
        )


@dataclass
class GreeksResult:
    """Implied volatility plus first- and second-order sensitivities."""

    implied_volatility: float | None = None
    delta: float | None = None
    gamma: float | None = None
    vega: float | None = None
    theta: float | None = None
    rho: float | None = None
    mid_price: float | None = None
    intrinsic_value: float | None = None
    time_value: float | None = None
    moneyness: float | None = None
    log_moneyness: float | None = None
    #: Approximate IV uncertainty in volatility points implied by one tick of
    #: quote error. Low vega means a wide band; useful as a sample weight.
    iv_uncertainty: float | None = None
    quality: QualityFlag = QualityFlag.OK

    @property
    def usable(self) -> bool:
        return self.implied_volatility is not None and self.quality.is_tradeable_quote


# ------------------------------------------------------------------- pricing


def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> tuple[float, float]:
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def bs_price(
    S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool
) -> float:
    """Black-Scholes-Merton price with continuous dividend yield ``q``."""
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if is_call else (K - S))
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_r, disc_q = math.exp(-r * T), math.exp(-q * T)
    if is_call:
        return S * disc_q * _norm_cdf(d1) - K * disc_r * _norm_cdf(d2)
    return K * disc_r * _norm_cdf(-d2) - S * disc_q * _norm_cdf(-d1)


def price_and_greeks(
    S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool
) -> dict[str, float]:
    """Analytic Greeks at a given volatility.

    Conventions chosen to match how these are quoted in practice: vega per 1
    volatility *point* (not per unit), theta per calendar day (not per year), and
    rho per 1% rate move. Getting these scalings wrong is the usual reason two
    systems' Greeks disagree by a factor of 100.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_r, disc_q = math.exp(-r * T), math.exp(-q * T)
    pdf_d1 = _norm_pdf(d1)
    sqrt_t = math.sqrt(T)

    gamma = disc_q * pdf_d1 / (S * sigma * sqrt_t)
    vega = S * disc_q * pdf_d1 * sqrt_t / 100.0

    common_theta = -(S * disc_q * pdf_d1 * sigma) / (2 * sqrt_t)
    if is_call:
        delta = disc_q * _norm_cdf(d1)
        theta = common_theta - r * K * disc_r * _norm_cdf(d2) + q * S * disc_q * _norm_cdf(d1)
        rho = K * T * disc_r * _norm_cdf(d2) / 100.0
    else:
        delta = -disc_q * _norm_cdf(-d1)
        theta = common_theta + r * K * disc_r * _norm_cdf(-d2) - q * S * disc_q * _norm_cdf(-d1)
        rho = -K * T * disc_r * _norm_cdf(-d2) / 100.0

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta / 365.0),
        "rho": float(rho),
    }


# ------------------------------------------------------- implied volatility


def _assess_quote(
    S: float,
    K: float,
    T: float,
    bid: float | None,
    ask: float | None,
    last: float | None,
    is_call: bool,
) -> tuple[float | None, float, QualityFlag]:
    """Decide the price to solve against, and judge whether it means anything."""
    flag = QualityFlag.OK

    if not S or not K or S <= 0 or K <= 0 or not np.isfinite(S) or not np.isfinite(K):
        return None, 0.0, QualityFlag.BAD_INPUT
    if T <= 0:
        return None, 0.0, QualityFlag.EXPIRED

    has_bid = bid is not None and np.isfinite(bid) and bid > 0
    has_ask = ask is not None and np.isfinite(ask) and ask > 0

    if has_bid and has_ask:
        if bid >= ask:
            flag |= QualityFlag.CROSSED_MARKET
        mid = 0.5 * (bid + ask)
        if mid > 0 and (ask - bid) / mid > MAX_SPREAD_RATIO:
            flag |= QualityFlag.WIDE_SPREAD
        price = mid
    elif has_ask:
        # A zero bid is a real market state, not missing data: it means no buyer.
        flag |= QualityFlag.ZERO_BID
        price = ask / 2.0
    elif last is not None and np.isfinite(last) and last > 0:
        flag |= QualityFlag.ZERO_BID
        price = float(last)
    else:
        return None, 0.0, flag | QualityFlag.BAD_INPUT

    intrinsic = max(0.0, (S - K) if is_call else (K - S))
    if price < intrinsic - 1e-9:
        flag |= QualityFlag.BELOW_INTRINSIC
    elif price - intrinsic < 1e-4:
        flag |= QualityFlag.NO_TIME_VALUE

    if abs(math.log(K / S)) > DEEP_ITM_LOG_MONEYNESS:
        flag |= QualityFlag.DEEP_ITM

    return float(price), float(intrinsic), flag


def implied_volatility(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    price: float,
    is_call: bool,
) -> float | None:
    """Solve Black-Scholes for sigma via Brent's method.

    Brent rather than Newton-Raphson: vega collapses toward zero for deep ITM/OTM
    contracts, and Newton divides by it. Brent needs only a sign change over the
    bracket and cannot diverge, which matters when the input is an unvetted quote
    from a free API.
    """
    if T <= 0 or S <= 0 or K <= 0 or price <= 0:
        return None

    def objective(sigma: float) -> float:
        return bs_price(S, K, T, r, q, sigma, is_call) - price

    try:
        lo, hi = objective(MIN_VOL), objective(MAX_VOL)
        if lo * hi > 0:
            # No sign change: the price lies outside what any volatility in the
            # bracket can produce (usually below intrinsic or above the bound).
            return None
        return float(brentq(objective, MIN_VOL, MAX_VOL, xtol=_SOLVER_TOL, maxiter=_SOLVER_MAXITER))
    except (ValueError, RuntimeError):
        return None


def analyze_contract(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    is_call: bool,
    bid: float | None = None,
    ask: float | None = None,
    last: float | None = None,
) -> GreeksResult:
    """Full analysis of one contract: quote quality, IV, then Greeks at that IV."""
    price, intrinsic, flag = _assess_quote(S, K, T, bid, ask, last, is_call)

    result = GreeksResult(
        mid_price=price,
        intrinsic_value=intrinsic if price is not None else None,
        time_value=(price - intrinsic) if price is not None else None,
        moneyness=(S / K) if K else None,
        log_moneyness=math.log(S / K) if (S > 0 and K > 0) else None,
        quality=flag,
    )

    # Solving against a fictional price yields a confident, meaningless number.
    if price is None or not flag.is_tradeable_quote:
        return result

    sigma = implied_volatility(S, K, T, r, q, price, is_call)
    if sigma is None:
        result.quality = flag | QualityFlag.NO_SOLUTION
        return result

    greeks = price_and_greeks(S, K, T, r, q, sigma, is_call)
    result.delta = greeks["delta"]
    result.gamma = greeks["gamma"]
    result.vega = greeks["vega"]
    result.theta = greeks["theta"]
    result.rho = greeks["rho"]

    # Greeks stay -- delta and gamma remain meaningful on a deep-ITM contract --
    # but the volatility itself does not survive this check. Vega near zero means
    # the solver's answer was determined by its own bracket, not by the quote.
    if greeks["vega"] < MIN_VEGA_FOR_IV:
        result.quality = flag | QualityFlag.UNIDENTIFIABLE
        return result

    result.implied_volatility = sigma
    result.iv_uncertainty = OPTION_TICK / greeks["vega"] / 100.0

    if sigma > MAX_PLAUSIBLE_IV:
        # Recorded, not discarded: the value is still evidence of *something*
        # (usually an American early-exercise premium the model cannot express),
        # and seeing it beats having it quietly averaged into a vol surface.
        result.quality = flag | QualityFlag.IMPLAUSIBLE_IV
    return result
