"""Derive implied volatility, Greeks, and flow ratios from stored option chains.

This is a *transform*, not a fetch. It reads option chains, the risk-free curve
(FRED), and dividend history (OHLCV) already in the lake, and writes a derived
``options_greeks`` dataset alongside them.

Keeping it separate from the options fetcher is deliberate. Yahoo serves only the
live chain, so a snapshot can never be re-fetched -- but it can be re-analysed
without limit. Storing derived values in their own dataset means improving the
model later is a rerun over Parquet rather than history you can never get back.

Inputs assembled per contract:
  * spot -- from the snapshot itself, so it matches what the chain was quoted
    against rather than a later revised close.
  * risk-free rate -- interpolated from the FRED curve at the contract's own
    maturity, not one flat rate for every tenor.
  * dividend yield -- trailing 12-month dividends over spot, from stored OHLCV.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from tickerlake.analytics.black_scholes import QualityFlag, analyze_contract
from tickerlake.storage import paths as P
from tickerlake.storage.query import LakeQuery
from tickerlake.storage.writer import ParquetWriter

log = logging.getLogger(__name__)

SOURCE = "tickerlake.analytics"

# FRED series -> maturity in years, used to build the discount curve.
CURVE_POINTS = {
    "DGS3MO": 0.25,
    "DGS2": 2.0,
    "DGS10": 10.0,
}
# Used for the very short end when available; SOFR is effectively overnight.
OVERNIGHT_SERIES = "SOFR"
FALLBACK_RATE = 0.04


class OptionsAnalytics:
    """Computes IV, Greeks, and per-symbol flow ratios for a snapshot date."""

    def __init__(self, data_root, writer: ParquetWriter | None = None):
        self.paths = P.DatasetPaths(data_root)
        self.writer = writer or ParquetWriter()

    # ------------------------------------------------------------------ main

    def run(self, snapshot_date: date, symbols: list[str] | None = None) -> dict:
        with LakeQuery(self.paths.root) as q:
            chains = q.options_snapshot(snapshot_date, symbols=symbols)
            if chains.empty:
                log.warning("no option chains stored for %s", snapshot_date)
                return {"rows": 0, "snapshot_date": snapshot_date.isoformat()}

            curve = self._risk_free_curve(q, snapshot_date)
            dividends = self._dividend_yields(q, snapshot_date, chains["symbol"].unique().tolist())

        log.info(
            "analysing %d contracts across %d symbols for %s",
            len(chains),
            chains["symbol"].nunique(),
            snapshot_date,
        )
        derived = self._analyze(chains, curve, dividends, snapshot_date)

        write = self.writer.write(
            derived,
            P.OPTIONS_GREEKS,
            self.paths.options_greeks_file(snapshot_date),
            mode="overwrite",
        )
        log.info("options_greeks: %s", write)

        flows = self._flow_ratios(chains, derived, snapshot_date)
        if not flows.empty:
            flow_write = self.writer.write(
                flows, P.OPTIONS_FLOW, self.paths.options_flow_file(snapshot_date), mode="overwrite"
            )
            log.info("options_flow: %s", flow_write)

        usable = int(derived["iv_usable"].sum())
        stats = {
            "snapshot_date": snapshot_date.isoformat(),
            "contracts": len(derived),
            "iv_solved": usable,
            "iv_solved_pct": round(100 * usable / max(len(derived), 1), 1),
            "symbols": int(derived["symbol"].nunique()),
            "rows": write.rows_written,
        }
        log.info(
            "IV solved for %d/%d contracts (%.1f%%); %d gated as unusable",
            usable,
            len(derived),
            stats["iv_solved_pct"],
            len(derived) - usable,
        )
        return stats

    # -------------------------------------------------------------- analysis

    def _analyze(
        self,
        chains: pd.DataFrame,
        curve: list[tuple[float, float]],
        dividends: dict[str, float],
        snapshot_date: date,
    ) -> pd.DataFrame:
        now = datetime.now(UTC)
        out = []

        for row in chains.itertuples():
            spot = _f(row.underlying_price)
            strike = _f(row.strike)
            dte = int(row.dte) if pd.notna(row.dte) else 0
            # Act/365 with a floor: a same-day expiry still has hours of life, and
            # T=0 makes every Greek collapse to zero or divide by zero.
            T = max(dte, 0.5) / 365.0

            rate = _interp_rate(curve, T)
            div_yield = dividends.get(row.symbol, 0.0)
            is_call = str(row.option_type).lower() == "call"

            if spot is None or strike is None:
                res = analyze_contract(0, 0, 0, rate, div_yield, is_call)
            else:
                res = analyze_contract(
                    S=spot,
                    K=strike,
                    T=T,
                    r=rate,
                    q=div_yield,
                    is_call=is_call,
                    bid=_f(row.bid),
                    ask=_f(row.ask),
                    last=_f(row.last_price),
                )

            out.append(
                {
                    "symbol": row.symbol,
                    "snapshot_date": snapshot_date,
                    "expiration": row.expiration,
                    "option_type": row.option_type,
                    "strike": strike,
                    "contract_symbol": row.contract_symbol,
                    "dte": dte,
                    "underlying_price": spot,
                    "mid_price": res.mid_price,
                    "intrinsic_value": res.intrinsic_value,
                    "time_value": res.time_value,
                    "iv": res.implied_volatility,
                    "iv_uncertainty": res.iv_uncertainty,
                    # Kept for comparison: this is the field the README warns about.
                    "iv_vendor": _f(row.implied_volatility),
                    "delta": res.delta,
                    "gamma": res.gamma,
                    "vega": res.vega,
                    "theta": res.theta,
                    "rho": res.rho,
                    "moneyness": res.moneyness,
                    "log_moneyness": res.log_moneyness,
                    "risk_free_rate": rate,
                    "dividend_yield": div_yield,
                    "quality_flags": res.quality.to_label(),
                    "iv_usable": bool(res.usable),
                    "volume": _i(row.volume),
                    "open_interest": _i(row.open_interest),
                    "source": SOURCE,
                    "ingested_at": now,
                }
            )

        return pd.DataFrame(out)

    # ------------------------------------------------------------ flow ratios

    def _flow_ratios(
        self, chains: pd.DataFrame, derived: pd.DataFrame, snapshot_date: date
    ) -> pd.DataFrame:
        """Per-symbol put/call ratios and an ATM IV summary.

        CBOE's published put/call ratios stopped being freely available, but this
        is the same statistic computed from our own chains -- and per symbol
        rather than one market-wide number, which is strictly more useful.
        """
        now = datetime.now(UTC)
        df = chains.copy()
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce").fillna(0)

        rows = []
        # Reference contracts for the vol surface summary, restricted to the
        # 20-45 day tenor where quotes are most reliable.
        ref = derived[
            derived["iv_usable"] & derived["iv"].notna() & (derived["dte"].between(20, 45))
        ]

        for symbol, group in df.groupby("symbol"):
            calls = group[group["option_type"] == "call"]
            puts = group[group["option_type"] == "put"]
            call_vol, put_vol = calls["volume"].sum(), puts["volume"].sum()
            call_oi, put_oi = calls["open_interest"].sum(), puts["open_interest"].sum()

            sub = ref[ref["symbol"] == symbol]
            atm_iv = skew_25d = None
            if not sub.empty:
                # ATM is selected by log-moneyness, not by delta. Delta looks like
                # a natural moneyness proxy but it is computed *from* the solved
                # IV, so a contract with an absurd IV has its delta pulled toward
                # 0.5 and masquerades as at-the-money -- which is precisely how a
                # deep-ITM put at 470% vol ends up setting a symbol's "ATM" level.
                # Log-moneyness depends only on spot and strike.
                near = sub[sub["log_moneyness"].abs() < 0.03]
                atm_iv = float(near["iv"].median()) if not near.empty else None

                # Wings by moneyness for the same reason: ~5-12% OTM either side.
                put_wing = sub[
                    (sub["option_type"] == "put") & sub["log_moneyness"].between(0.05, 0.12)
                ]
                call_wing = sub[
                    (sub["option_type"] == "call") & sub["log_moneyness"].between(-0.12, -0.05)
                ]
                if not put_wing.empty and not call_wing.empty:
                    # Median, not mean: one bad quote should not define the skew.
                    skew_25d = float(put_wing["iv"].median() - call_wing["iv"].median())

            rows.append(
                {
                    "symbol": symbol,
                    "snapshot_date": snapshot_date,
                    "call_volume": int(call_vol),
                    "put_volume": int(put_vol),
                    "put_call_volume_ratio": float(put_vol / call_vol) if call_vol else None,
                    "call_open_interest": int(call_oi),
                    "put_open_interest": int(put_oi),
                    "put_call_oi_ratio": float(put_oi / call_oi) if call_oi else None,
                    "atm_iv_30d": atm_iv,
                    "skew_25d": skew_25d,
                    "contracts": int(len(group)),
                    "expirations": int(group["expiration"].nunique()),
                    "underlying_price": _f(group["underlying_price"].iloc[0]),
                    "source": SOURCE,
                    "ingested_at": now,
                }
            )
        return pd.DataFrame(rows)

    # ----------------------------------------------------------- curve inputs

    def _risk_free_curve(self, q: LakeQuery, as_of: date) -> list[tuple[float, float]]:
        """(maturity_years, rate) points from the most recent FRED observations."""
        points: list[tuple[float, float]] = []
        try:
            wanted = list(CURVE_POINTS) + [OVERNIGHT_SERIES]
            placeholders = ",".join("?" * len(wanted))
            df = q.sql(
                f"""
                SELECT series_id, value
                FROM macro_series
                WHERE series_id IN ({placeholders})
                  AND value IS NOT NULL AND date <= ?
                QUALIFY ROW_NUMBER() OVER (PARTITION BY series_id ORDER BY date DESC) = 1
                """,
                [*wanted, as_of],
            )
            latest = dict(zip(df["series_id"], df["value"], strict=False))
            for series, maturity in CURVE_POINTS.items():
                if series in latest:
                    points.append((maturity, float(latest[series]) / 100.0))
            if OVERNIGHT_SERIES in latest:
                points.append((1 / 365.0, float(latest[OVERNIGHT_SERIES]) / 100.0))
        except Exception as exc:
            log.warning(
                "could not read the risk-free curve (%s); using %.1f%% flat",
                exc,
                FALLBACK_RATE * 100,
            )

        if not points:
            log.warning(
                "no FRED rate data available; using a flat %.1f%%. Run the fred stage "
                "for maturity-matched discounting.",
                FALLBACK_RATE * 100,
            )
            return [(1.0, FALLBACK_RATE)]

        points.sort()
        log.info(
            "risk-free curve: %s",
            ", ".join(f"{m:.2f}y={r * 100:.2f}%" for m, r in points),
        )
        return points

    def _dividend_yields(self, q: LakeQuery, as_of: date, symbols: list[str]) -> dict[str, float]:
        """Trailing 12-month dividend yield per symbol, from stored OHLCV."""
        try:
            df = q.ohlcv(symbols=symbols, start=as_of - timedelta(days=370), end=as_of)
        except Exception as exc:
            log.warning("could not read dividends (%s); assuming zero yield", exc)
            return {}
        if df.empty or "dividends" not in df.columns:
            return {}

        df["dividends"] = pd.to_numeric(df["dividends"], errors="coerce").fillna(0.0)
        paid = df.groupby("symbol")["dividends"].sum()
        latest = df.sort_values("date").groupby("symbol")["close"].last()

        out = {}
        for symbol, total in paid.items():
            price = latest.get(symbol)
            if price and price > 0 and total > 0:
                # Cap at 25%: anything above that is a special dividend or a data
                # error, and feeding it into the model would distort every Greek.
                out[symbol] = float(min(total / price, 0.25))
        log.info("dividend yields computed for %d/%d symbols", len(out), len(symbols))
        return out


# ------------------------------------------------------------------ helpers


def _interp_rate(curve: list[tuple[float, float]], T: float) -> float:
    """Linear interpolation on the curve, flat beyond either end."""
    if not curve:
        return FALLBACK_RATE
    if len(curve) == 1:
        return curve[0][1]
    maturities = [m for m, _ in curve]
    rates = [r for _, r in curve]
    return float(np.interp(T, maturities, rates))


def _f(value) -> float | None:
    try:
        out = float(value)
        return out if np.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _i(value) -> int | None:
    try:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = ["OptionsAnalytics", "QualityFlag"]
