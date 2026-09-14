"""Reference series: Fama-French factors, the Treasury curve, and CFTC COT.

Three free, no-key sources that share a shape: small, slow-moving, market-wide
series that are re-fetched whole each run rather than incrementally appended.
Each publisher revises history (French rebuilds factors when CRSP updates,
Treasury restates, CFTC reissues corrected reports), so an overwrite is both
simpler and more correct than trying to detect changed rows.

Why these three
---------------
* **Fama-French factors** are the reference against which a signal is judged. A
  strategy that looks profitable is usually just market, size, value,
  profitability, investment or momentum exposure wearing a different name, and
  without these series there is no cheap way to tell. Canonical and free.
* **The Treasury par yield curve** gives all 13 official tenors daily. FRED
  carries a handful of them and publishes on a lag; this is the source Treasury
  itself puts out, and the full curve makes slope and curvature computable
  rather than approximated from three points.
* **CFTC Commitments of Traders** is weekly positioning by trader category --
  asset managers, leveraged money, dealers -- for E-mini S&P 500 and the sector
  index futures. Nothing else here describes who is positioned which way.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from datetime import UTC, date, datetime

import pandas as pd

from tickerlake.fetchers.base import BaseFetcher, FetchResult
from tickerlake.storage import paths as P
from tickerlake.utils.http import HttpClient

# --------------------------------------------------------------------- F-F

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
FACTOR_SETS = {
    "FF3": "F-F_Research_Data_Factors_daily_CSV.zip",
    "FF5": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "MOM": "F-F_Momentum_Factor_daily_CSV.zip",
}
# A data row is an 8-digit date followed by numbers. Everything else in these
# files is prose: a provenance header, blank lines, and a copyright footer.
_FF_ROW = re.compile(r"^\s*(\d{8})\s*,(.*)$")


class FamaFrenchFetcher(BaseFetcher):
    """Daily factor returns from the Ken French data library."""

    name = "factors"
    dataset = P.FACTORS

    def collect(self, run_date: date, result: FetchResult) -> None:
        wanted = self.cfg("sets", ["FF5", "MOM"]) or []
        client = HttpClient(
            requests_per_second=float(self.cfg("requests_per_second", 1.0)),
            user_agent="TickerLake/0.1 (research)",
            timeout=120,
        )
        try:
            for factor_set in wanted:
                filename = FACTOR_SETS.get(factor_set)
                if filename is None:
                    result.add_warning(f"unknown factor set {factor_set!r}")
                    continue
                try:
                    frame = self._fetch_set(client, factor_set, filename)
                except Exception as exc:
                    result.items_failed += 1
                    result.add_warning(f"{factor_set}: {type(exc).__name__}: {exc}")
                    continue

                if frame.empty:
                    result.items_skipped += 1
                    continue
                write = self.writer.write(
                    frame, P.FACTORS, self.paths.factors_file(factor_set), mode="overwrite"
                )
                result.record_write(write)
                result.items_succeeded += 1
                self.log.info(
                    "%s: %d rows, %s to %s",
                    factor_set,
                    len(frame),
                    frame["date"].min(),
                    frame["date"].max(),
                )
        finally:
            client.close()

    def _fetch_set(self, client: HttpClient, factor_set: str, filename: str) -> pd.DataFrame:
        payload = client.get(f"{FRENCH_BASE}/{filename}").content
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            # These CSVs are latin-1, not UTF-8; decoding strictly raises.
            raw = zf.read(zf.namelist()[0]).decode("latin-1")

        header: list[str] = []
        rows: list[dict] = []
        now = datetime.now(UTC)
        start = _as_date(self.cfg("start_date", "1990-01-01"))

        for line in raw.splitlines():
            match = _FF_ROW.match(line)
            if match is None:
                # The column header is the last comma-led line before the data.
                if line.startswith(",") and not header:
                    header = [c.strip() for c in line.split(",")[1:]]
                continue
            if not header:
                continue

            stamp = match.group(1)
            try:
                day = date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]))
            except ValueError:
                continue
            if day < start:
                continue

            for name, value in zip(header, match.group(2).split(","), strict=False):
                parsed = _as_float(value)
                # -99.99 and -999 are French's missing-value sentinels; storing
                # them as numbers would poison any mean or regression.
                if parsed is None or parsed <= -99:
                    continue
                rows.append(
                    {
                        "date": day,
                        "factor_set": factor_set,
                        "factor": name,
                        "value": parsed,
                        "source": "ken_french",
                        "ingested_at": now,
                    }
                )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------- Treasury

TREASURY_XML = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value={year}"
)
# Treasury's field names -> maturity in years, for interpolation.
TENORS = {
    "BC_1MONTH": 1 / 12,
    "BC_1_5MONTH": 1.5 / 12,
    "BC_2MONTH": 2 / 12,
    "BC_3MONTH": 0.25,
    "BC_4MONTH": 4 / 12,
    "BC_6MONTH": 0.5,
    "BC_1YEAR": 1.0,
    "BC_2YEAR": 2.0,
    "BC_3YEAR": 3.0,
    "BC_5YEAR": 5.0,
    "BC_7YEAR": 7.0,
    "BC_10YEAR": 10.0,
    "BC_20YEAR": 20.0,
    "BC_30YEAR": 30.0,
}
_ENTRY = re.compile(r"<m:properties>(.*?)</m:properties>", re.S)
_FIELD = re.compile(r"<d:([A-Za-z_0-9]+)[^>]*>([^<]*)</d:")


class TreasuryCurveFetcher(BaseFetcher):
    """Official daily par yield curve, all published tenors."""

    name = "yield_curve"
    dataset = P.YIELD_CURVE

    def collect(self, run_date: date, result: FetchResult) -> None:
        back = int(self.cfg("years_back", 1))
        years = range(run_date.year - back, run_date.year + 1)
        client = HttpClient(
            requests_per_second=float(self.cfg("requests_per_second", 1.0)),
            user_agent="TickerLake/0.1 (research)",
            timeout=120,
        )
        try:
            for year in years:
                try:
                    frame = self._fetch_year(client, year)
                except Exception as exc:
                    result.items_failed += 1
                    result.add_warning(f"{year}: {type(exc).__name__}: {exc}")
                    continue
                if frame.empty:
                    result.items_skipped += 1
                    continue
                write = self.writer.write(
                    frame, P.YIELD_CURVE, self.paths.yield_curve_file(year), mode="overwrite"
                )
                result.record_write(write)
                result.items_succeeded += 1
                self.log.info(
                    "%d: %d observations across %d sessions",
                    year,
                    len(frame),
                    frame["date"].nunique(),
                )
        finally:
            client.close()

    def _fetch_year(self, client: HttpClient, year: int) -> pd.DataFrame:
        xml = client.get_text(TREASURY_XML.format(year=year))
        now = datetime.now(UTC)
        rows = []
        for block in _ENTRY.findall(xml):
            fields = dict(_FIELD.findall(block))
            raw_date = fields.get("NEW_DATE", "")[:10]
            try:
                day = date.fromisoformat(raw_date)
            except ValueError:
                continue
            for name, years_out in TENORS.items():
                value = _as_float(fields.get(name))
                if value is None:
                    continue
                rows.append(
                    {
                        "date": day,
                        "tenor": name.replace("BC_", "").replace("_", "."),
                        "tenor_years": years_out,
                        "yield_pct": value,
                        "source": "us_treasury",
                        "ingested_at": now,
                    }
                )
        return pd.DataFrame(rows)


# --------------------------------------------------------------------- COT

COT_URL = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"
_COT_COLUMNS = {
    "open_interest": "Open_Interest_All",
    "dealer_long": "Dealer_Positions_Long_All",
    "dealer_short": "Dealer_Positions_Short_All",
    "asset_mgr_long": "Asset_Mgr_Positions_Long_All",
    "asset_mgr_short": "Asset_Mgr_Positions_Short_All",
    "lev_money_long": "Lev_Money_Positions_Long_All",
    "lev_money_short": "Lev_Money_Positions_Short_All",
    "other_rept_long": "Other_Rept_Positions_Long_All",
    "other_rept_short": "Other_Rept_Positions_Short_All",
    "nonrept_long": "NonRept_Positions_Long_All",
    "nonrept_short": "NonRept_Positions_Short_All",
}


class CftcCotFetcher(BaseFetcher):
    """Weekly Commitments of Traders positioning for financial futures."""

    name = "cot"
    dataset = P.COT

    def collect(self, run_date: date, result: FetchResult) -> None:
        back = int(self.cfg("years_back", 1))
        keywords = [k.upper() for k in (self.cfg("market_keywords") or [])]
        client = HttpClient(
            requests_per_second=float(self.cfg("requests_per_second", 1.0)),
            user_agent="TickerLake/0.1 (research)",
            timeout=180,
        )
        try:
            for year in range(run_date.year - back, run_date.year + 1):
                try:
                    frame = self._fetch_year(client, year, keywords)
                except Exception as exc:
                    result.items_failed += 1
                    result.add_warning(f"{year}: {type(exc).__name__}: {exc}")
                    continue
                if frame.empty:
                    result.items_skipped += 1
                    continue
                write = self.writer.write(frame, P.COT, self.paths.cot_file(year), mode="overwrite")
                result.record_write(write)
                result.items_succeeded += 1
                self.log.info(
                    "%d: %d rows across %d markets, %d report dates",
                    year,
                    len(frame),
                    frame["market"].nunique(),
                    frame["report_date"].nunique(),
                )
        finally:
            client.close()

    def _fetch_year(self, client: HttpClient, year: int, keywords: list[str]) -> pd.DataFrame:
        payload = client.get(COT_URL.format(year=year)).content
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            raw = zf.read(zf.namelist()[0]).decode("latin-1")

        reader = csv.DictReader(io.StringIO(raw))
        now = datetime.now(UTC)
        rows = []
        for record in reader:
            record = {(k or "").strip(): v for k, v in record.items()}
            market = (record.get("Market_and_Exchange_Names") or "").strip()
            if not market:
                continue
            # The financial report covers 107 markets; keep only the ones that
            # bear on equities unless the filter is left empty.
            if keywords and not any(k in market.upper() for k in keywords):
                continue
            try:
                report_date = date.fromisoformat(
                    (record.get("Report_Date_as_YYYY-MM-DD") or "").strip()[:10]
                )
            except ValueError:
                continue

            name, _, exchange = market.partition(" - ")
            row = {
                "report_date": report_date,
                "market": name.strip(),
                "exchange": exchange.strip() or None,
                "contract_code": (record.get("CFTC_Contract_Market_Code") or "").strip() or None,
                "source": "cftc",
                "ingested_at": now,
            }
            for out_name, column in _COT_COLUMNS.items():
                row[out_name] = _as_float(record.get(column))

            for side in ("asset_mgr", "lev_money", "dealer"):
                long_, short_ = row.get(f"{side}_long"), row.get(f"{side}_short")
                row[f"{side}_net"] = (
                    long_ - short_ if long_ is not None and short_ is not None else None
                )
            rows.append(row)
        return pd.DataFrame(rows)


# ------------------------------------------------------------------ helpers


def _as_float(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text in {".", "-", "N/A", "NA"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _as_date(value: str | date) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value)[:10])
