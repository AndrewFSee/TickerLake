"""SEC EDGAR fetcher: filing text plus structured XBRL fundamentals.

Discovery strategy
------------------
The obvious approach -- poll ``/submissions/CIK*.json`` for each tracked symbol
-- costs ~850 requests and well over a gigabyte of JSON *every day*, almost all
of it re-reading filing history that has not changed.

Instead the daily run reads EDGAR's **daily index** (``master.idx``), which lists
every filing accepted that day in a single request, and intersects it with our
tracked CIKs. A typical weekday yields a few dozen relevant filings, so the daily
cost is roughly ``1 + 2 x filings`` requests rather than 850.

The per-symbol submissions API is still used, but only by ``backfill``, where
reading full filing history is the actual point.

SEC requires a descriptive User-Agent with a real contact address and asks for
under 10 requests/second. Both are enforced here.
"""

from __future__ import annotations

import re
import warnings
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from tickerlake.fetchers.base import BaseFetcher, FetchResult
from tickerlake.storage import paths as P
from tickerlake.utils.http import HttpClient, HttpError

SOURCE = "sec_edgar"

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
DAILY_INDEX_URL = (
    "https://www.sec.gov/Archives/edgar/daily-index/{year}/QTR{qtr}/master.{stamp}.idx"
)
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
FILING_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/index.json"
ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"

# Document extensions we can usefully extract text from.
_TEXT_EXTENSIONS = (".htm", ".html", ".txt")
_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")


class SECEdgarFetcher(BaseFetcher):
    """10-K / 10-Q / 8-K text and XBRL company facts for the tracked universe."""

    name = "sec_edgar"
    dataset = P.FILINGS_TEXT
    requires_secret = "sec_user_agent"

    def __init__(self, *args: Any, backfill: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.backfill = backfill
        self._client: HttpClient | None = None
        self._cik_to_symbol: dict[str, str] = {}
        self._symbol_to_cik: dict[str, str] = {}

    # ------------------------------------------------------------------ main

    def collect(self, run_date: date, result: FetchResult) -> None:
        user_agent = self.config.secrets.sec_user_agent or "TickerLake/0.1"
        self._client = HttpClient(
            requests_per_second=float(self.cfg("requests_per_second", 5.0)),
            user_agent=user_agent,
            max_attempts=4,
        )
        try:
            self._load_cik_map(result)
            if not self._symbol_to_cik:
                result.add_error("could not build ticker->CIK map; skipping EDGAR")
                return

            if self.backfill:
                self._backfill_filings(run_date, result)
            else:
                self._daily_filings(run_date, result)

            if self.cfg("fetch_companyfacts", True):
                self._refresh_companyfacts(run_date, result)
        finally:
            self._client.close()
            self._client = None

    # -------------------------------------------------------------- cik map

    def _load_cik_map(self, result: FetchResult) -> None:
        """Ticker -> zero-padded CIK, restricted to the tracked universe."""
        try:
            payload = self._client.get_json(TICKER_MAP_URL)
        except (HttpError, ValueError) as exc:
            result.add_error(f"could not load SEC ticker map: {exc}")
            return

        tracked = set(self._tracked_symbols())
        for entry in payload.values():
            raw = str(entry.get("ticker", "")).strip().upper()
            if not raw:
                continue
            # SEC spells class shares with a dash (BRK-B); we store dots.
            symbol = raw.replace("-", ".")
            if tracked and symbol not in tracked:
                continue
            cik = str(entry.get("cik_str", "")).zfill(10)
            self._symbol_to_cik[symbol] = cik
            self._cik_to_symbol[cik] = symbol

        self.log.info(
            "mapped %d/%d tracked symbols to CIKs", len(self._symbol_to_cik), len(tracked)
        )

    def _tracked_symbols(self) -> list[str]:
        if self.symbols_override is not None:
            return self.symbols_override
        if self.tracker is None:
            return []
        return self.tracker.tracked_symbols()

    # --------------------------------------------------------- daily filings

    def _daily_filings(self, run_date: date, result: FetchResult) -> None:
        """Discover via the daily index, then fetch each relevant filing."""
        lookback = int(self.cfg("lookback_days", 4))
        form_types = {f.upper() for f in self.cfg("form_types", ["10-K", "10-Q", "8-K"])}

        discovered: list[dict[str, Any]] = []
        for offset in range(lookback + 1):
            day = run_date - timedelta(days=offset)
            if day.weekday() >= 5:  # EDGAR publishes no weekend index
                continue
            discovered.extend(self._read_daily_index(day, form_types, result))

        if not discovered:
            result.add_warning(f"no matching filings found in the {lookback}-day index window")
            return

        # Skip filings already stored, so a re-run costs nothing.
        known = self._known_accessions(run_date)
        fresh = [f for f in discovered if f["accession_number"] not in known]
        self.log.info(
            "%d filings discovered, %d new (%d already stored)",
            len(discovered),
            len(fresh),
            len(discovered) - len(fresh),
        )
        if not fresh:
            return

        rows = []
        for filing in fresh:
            try:
                rows.append(self._fetch_filing_text(filing))
                result.items_succeeded += 1
            except Exception as exc:
                result.items_failed += 1
                self.log.debug("filing %s failed: %s", filing["accession_number"], exc)
                result.add_warning(f"{filing['accession_number']} ({filing.get('symbol')}): {exc}")

        if rows:
            write = self.writer.write(
                pd.DataFrame(rows),
                P.FILINGS_TEXT,
                self.paths.filings_text_file(run_date),
                mode="merge",
            )
            result.record_write(write)
            result.details["forms"] = (
                pd.DataFrame(rows)["form_type"].value_counts().to_dict() if rows else {}
            )

    def _read_daily_index(
        self, day: date, form_types: set[str], result: FetchResult
    ) -> list[dict[str, Any]]:
        """Parse one day's master.idx into filing records for tracked CIKs."""
        url = DAILY_INDEX_URL.format(
            year=day.year, qtr=(day.month - 1) // 3 + 1, stamp=day.strftime("%Y%m%d")
        )
        try:
            text = self._client.get_text(url)
        except HttpError:
            # Holidays and not-yet-published days legitimately 404.
            self.log.debug("no daily index for %s", day)
            return []

        out = []
        for line in text.splitlines():
            parts = line.split("|")
            if len(parts) != 5:
                continue
            cik_raw, company, form_type, filed, filename = (p.strip() for p in parts)
            if not cik_raw.isdigit():
                continue
            form_type = form_type.upper()
            if form_type not in form_types:
                continue
            cik = cik_raw.zfill(10)
            symbol = self._cik_to_symbol.get(cik)
            if symbol is None:
                continue
            try:
                filing_date = date.fromisoformat(filed)
            except ValueError:
                continue
            accession = filename.rsplit("/", 1)[-1].replace(".txt", "")
            out.append(
                {
                    "accession_number": accession,
                    "cik": cik,
                    "symbol": symbol,
                    "company_name": company,
                    "form_type": form_type,
                    "filing_date": filing_date,
                }
            )
        if out:
            self.log.info("daily index %s: %d tracked filings", day, len(out))
        return out

    def _known_accessions(self, run_date: date) -> set[str]:
        """Accession numbers already stored, so re-runs do not re-download."""
        known: set[str] = set()
        base = self.paths.dataset_dir(P.FILINGS_TEXT)
        if not base.exists():
            return known
        # Only this year and last, which is where the lookback window can land.
        for year in {run_date.year, (run_date - timedelta(days=400)).year}:
            for path in (base / f"year={year:04d}").glob("*.parquet"):
                try:
                    known.update(
                        pd.read_parquet(path, columns=["accession_number"])["accession_number"]
                    )
                except Exception as exc:
                    self.log.debug("could not read %s: %s", path.name, exc)
        return known

    def _fetch_filing_text(self, filing: dict[str, Any]) -> dict[str, Any]:
        """Resolve the primary document for a filing and extract its text."""
        cik_int = int(filing["cik"])
        acc_nodash = filing["accession_number"].replace("-", "")
        doc_name = self._primary_document(cik_int, acc_nodash, filing["form_type"])
        url = ARCHIVE_BASE.format(cik_int=cik_int, acc_nodash=acc_nodash, doc=doc_name)

        raw = self._client.get_text(url)
        text = _extract_text(raw, doc_name)

        max_chars = int(self.cfg("max_text_chars", 2_000_000))
        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars]

        return {
            **filing,
            # The daily index carries no report date; the submissions API (used by
            # backfill) does, so preserve it rather than blanking it.
            "report_date": filing.get("report_date"),
            "primary_doc_url": url,
            "text": text if self.cfg("store_text", True) else None,
            "text_chars": len(text),
            "truncated": truncated,
            # Left for a later embedding job to claim.
            "embedding_status": "pending",
            "embedding_model": None,
            "source": SOURCE,
            "ingested_at": datetime.now(UTC),
        }

    def _primary_document(self, cik_int: int, acc_nodash: str, form_type: str) -> str:
        """Pick the filing's primary document from its index.json.

        Falls back to the complete submission text file, which always exists.
        """
        items: list[dict[str, Any]] = []
        try:
            payload = self._client.get_json(
                FILING_INDEX_URL.format(cik_int=cik_int, acc_nodash=acc_nodash)
            )
            # `.get(key, {})` is not enough here: EDGAR sometimes returns the key
            # present with an explicit null, which yields None rather than the
            # default. Coerce at every level, and keep only dict entries.
            directory = (payload or {}).get("directory") or {}
            raw_items = directory.get("item") or []
            items = [i for i in raw_items if isinstance(i, dict)]
        except (HttpError, ValueError, AttributeError, TypeError):
            items = []

        candidates = [
            i.get("name", "")
            for i in items
            if str(i.get("name", "")).lower().endswith(_TEXT_EXTENSIONS)
        ]
        # Exhibits and XBRL viewer artefacts are not the filing itself.
        candidates = [
            c
            for c in candidates
            if not c.lower().startswith(("ex-", "ex_", "r", "report"))
            and "ex" != c.lower().split("-")[0]
        ]
        if candidates:
            # The primary document is reliably the largest of the main documents.
            sizes = {i.get("name"): int(i.get("size", 0) or 0) for i in items}
            return max(candidates, key=lambda c: sizes.get(c, 0))
        return f"{_with_dashes(acc_nodash)}.txt"

    # ------------------------------------------------------------- backfill

    def _backfill_filings(self, run_date: date, result: FetchResult) -> None:
        """Full filing history per symbol, via the submissions API."""
        start = _as_date(self.cfg("start_date", "2010-01-01"))
        form_types = {f.upper() for f in self.cfg("form_types", ["10-K", "10-Q", "8-K"])}
        symbols = self._tracked_symbols()
        self.log.info("backfilling filings for %d symbols since %s", len(symbols), start)

        known = self._known_accessions(run_date)
        rows: list[dict[str, Any]] = []

        for symbol in symbols:
            cik = self._symbol_to_cik.get(symbol)
            if cik is None:
                result.items_skipped += 1
                continue
            try:
                payload = self._client.get_json(SUBMISSIONS_URL.format(cik=cik))
                recent = payload.get("filings", {}).get("recent", {})
                for filing in _iter_submissions(recent, cik, symbol, payload.get("name")):
                    if filing["form_type"] not in form_types:
                        continue
                    if filing["filing_date"] < start:
                        continue
                    if filing["accession_number"] in known:
                        continue
                    rows.append(self._fetch_filing_text(filing))
                result.items_succeeded += 1
            except Exception as exc:
                result.items_failed += 1
                result.add_warning(f"{symbol} backfill failed: {exc}")

            # Flush periodically so a long backfill does not hold everything in RAM.
            if len(rows) >= 200:
                self._flush_backfill(rows, result)
                rows = []

        if rows:
            self._flush_backfill(rows, result)

    def _flush_backfill(self, rows: list[dict[str, Any]], result: FetchResult) -> None:
        df = pd.DataFrame(rows)
        for year, group in df.groupby(pd.to_datetime(df["filing_date"]).dt.year):
            path = self.paths.dataset_dir(P.FILINGS_TEXT) / f"year={int(year):04d}"
            path.mkdir(parents=True, exist_ok=True)
            write = self.writer.write(
                group, P.FILINGS_TEXT, path / "backfill.parquet", mode="merge"
            )
            result.record_write(write)

    # --------------------------------------------------------- company facts

    def _refresh_companyfacts(self, run_date: date, result: FetchResult) -> None:
        """Pull XBRL fundamentals for symbols whose facts file is missing or stale."""
        stale_days = int(self.cfg("companyfacts_refresh_days", 7))
        cutoff = datetime.now(UTC) - timedelta(days=stale_days)
        symbols = self._tracked_symbols()

        due = []
        for symbol in symbols:
            if symbol not in self._symbol_to_cik:
                continue
            path = self.paths.filings_facts_file(symbol)
            if not path.exists() or datetime.fromtimestamp(path.stat().st_mtime, tz=UTC) < cutoff:
                due.append(symbol)

        if not due:
            self.log.info("company facts are current for all tracked symbols")
            return

        # Amortise the ~850-symbol sweep across the week rather than doing it all
        # in one run; each symbol's payload is several MB.
        per_run = int(self.cfg("companyfacts_per_run", 120))
        batch = due[:per_run]
        self.log.info("refreshing XBRL company facts for %d/%d due symbols", len(batch), len(due))

        for symbol in batch:
            cik = self._symbol_to_cik[symbol]
            try:
                payload = self._client.get_json(COMPANYFACTS_URL.format(cik=cik))
                facts = _flatten_companyfacts(payload, cik, symbol)
                if facts.empty:
                    continue
                write = self.writer.write(
                    facts, P.FILINGS_FACTS, self.paths.filings_facts_file(symbol), mode="overwrite"
                )
                result.record_write(write)
            except HttpError as exc:
                # Plenty of tickers have no XBRL facts (foreign issuers, trusts).
                self.log.debug("no company facts for %s: %s", symbol, exc)
            except Exception as exc:
                result.add_warning(f"company facts {symbol}: {exc}")


# ------------------------------------------------------------------ helpers


def _with_dashes(acc_nodash: str) -> str:
    return f"{acc_nodash[:10]}-{acc_nodash[10:12]}-{acc_nodash[12:]}"


def _as_date(value: str | date) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value)[:10])


def _iter_submissions(recent: dict, cik: str, symbol: str, company: str | None):
    """Yield filing dicts from the submissions API's column-oriented payload."""
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    reports = recent.get("reportDate", [])
    for i, form in enumerate(forms):
        try:
            filing_date = date.fromisoformat(dates[i])
        except (ValueError, IndexError):
            continue
        report_date = None
        if i < len(reports) and reports[i]:
            try:
                report_date = date.fromisoformat(reports[i])
            except ValueError:
                report_date = None
        yield {
            "accession_number": accessions[i],
            "cik": cik,
            "symbol": symbol,
            "company_name": company,
            "form_type": str(form).upper(),
            "filing_date": filing_date,
            "report_date": report_date,
        }


def _flatten_companyfacts(payload: dict, cik: str, symbol: str) -> pd.DataFrame:
    """Flatten the nested XBRL companyfacts JSON into long rows."""
    now = datetime.now(UTC)
    rows = []
    for taxonomy, concepts in (payload.get("facts") or {}).items():
        for concept, body in concepts.items():
            for unit, entries in (body.get("units") or {}).items():
                for entry in entries:
                    rows.append(
                        {
                            "cik": cik,
                            "symbol": symbol,
                            "taxonomy": taxonomy,
                            "concept": concept,
                            "unit": unit,
                            "value": entry.get("val"),
                            "start_date": entry.get("start"),
                            "end_date": entry.get("end"),
                            "fiscal_year": entry.get("fy"),
                            "fiscal_period": entry.get("fp"),
                            "form_type": entry.get("form"),
                            "filed_date": entry.get("filed"),
                            "accession_number": entry.get("accn"),
                            "frame": entry.get("frame"),
                            "source": SOURCE,
                            "ingested_at": now,
                        }
                    )
    return pd.DataFrame(rows)


def _extract_text(raw: str, doc_name: str) -> str:
    """Strip markup and collapse whitespace, keeping paragraph structure.

    Modern filings are **inline XBRL**: the visible prose is wrapped around a
    hidden ``<ix:header>`` block holding the structured facts. Naively calling
    ``get_text()`` returns that machine-readable header instead of the document
    -- a stream of context refs and member tags that looks like text but is
    useless for NLP. The hidden layer is removed before extraction; the same
    facts are already collected properly via the XBRL companyfacts endpoint.
    """
    if doc_name.lower().endswith((".htm", ".html")) or "<html" in raw[:2000].lower():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
            soup = BeautifulSoup(raw, "lxml")

        for tag in soup(["script", "style", "ix:header", "ix:hidden", "xbrl"]):
            tag.decompose()
        # Inline XBRL also hides its header behind display:none rather than a
        # dedicated tag in some filer templates.
        for tag in soup.find_all(style=True):
            style = str(tag.get("style", "")).replace(" ", "").lower()
            if "display:none" in style or "visibility:hidden" in style:
                tag.decompose()
        text = soup.get_text("\n")
    else:
        text = re.sub(r"<[^>]+>", " ", raw)

    text = _WHITESPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return _BLANK_LINES.sub("\n\n", text).strip()
