"""Earnings announcement dates, from SEC 8-K Item 2.02 filings.

The problem this solves
-----------------------
Earnings surprise rows carry ``period`` -- the fiscal period end -- and nothing
about when the number became public. Apple's June quarter closed 2026-06-30 and
was announced 2026-07-30: a model keyed on ``period`` sees the surprise a month
before it existed. That is the same lookahead bias ``pit_fundamentals`` guards
against, in a dataset where it is easier to miss because the gap is only weeks
rather than months.

Finnhub's calendar cannot fix it. On the free tier it returns zero entries for
any past range, so it is forward-only and supplies no history at all.

SEC 8-K item codes can. A company announcing results files an 8-K carrying
**Item 2.02, Results of Operations and Financial Condition**, and that filing
date is the announcement date. The submissions API exposes an ``items`` field
per filing, so they are directly identifiable -- 45 of Apple's 103 recent 8-Ks
carry 2.02, on exactly the quarterly cadence.

Matching
--------
An 8-K does not state which fiscal period it reports, so the pairing has to be
inferred -- and two properties of the real data make the obvious rule wrong.

**Period labels are calendar quarters, not fiscal ones.** Finnhub files General
Mills' quarter ending 2025-11-30 under ``period = 2025-12-31``; it was announced
on 2025-12-17, *before* its own label. Requiring the filing to fall strictly
after the period end therefore skipped the real announcement and took the next
quarter's, dating every General Mills and Costco period one quarter late. So the
window opens ``max_lead_days`` *before* the label.

**Not every Item 2.02 filing is an earnings release.** Goldman Sachs filed one
on 2026-01-08, a week before its actual Q4 release on 2026-01-15. Taking the
earliest filing in the window picks the decoy.

What separates them is cadence: a company announces at a near-constant offset
from its period label, so the offset is measured from the company's own filing
history and each period then takes the filing closest to its expected date. A
filing is claimed by at most one period -- without that, a single 8-K was
recorded as both Q3 and Q4 -- and one too far off the company's own cadence is
left unmatched, because a wrong date reintroduces the bias this exists to
remove.

Reach
-----
Two limits are structural rather than bugs. The submissions API returns only a
filer's most recent ~1000 filings, and Item 2.02 did not exist before the SEC
renumbered 8-K items in August 2004 -- earnings releases were Item 12 then. So
periods from the early 2000s stay undated, which is the correct outcome: an
undated surprise is visibly unusable, a wrongly dated one is not.
"""

from __future__ import annotations

import statistics
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd

from tickerlake.fetchers.base import BaseFetcher, FetchResult
from tickerlake.storage import paths as P
from tickerlake.utils.dates import as_date
from tickerlake.utils.http import HttpClient, HttpError

SOURCE = "sec_edgar"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

# The 8-K item meaning "we are reporting results".
EARNINGS_ITEM = "2.02"


class AnnouncementFetcher(BaseFetcher):
    """Attaches announcement dates to stored earnings surprises."""

    name = "announcements"
    dataset = P.EARNINGS
    requires_secret = "sec_user_agent"

    def collect(self, run_date: date, result: FetchResult) -> None:
        unmatched = self._unmatched_surprises()
        if unmatched.empty:
            self.log.info("every stored surprise already has an announcement date")
            return

        # Merge key as datetime64 on both sides. Mapping to plain dates is not
        # enough: assigning them to a Series makes pandas re-infer datetime64,
        # so one side ends up holding Timestamps and the other date objects, and
        # the merge raises while sorting because the two do not compare. Name it
        # without a leading underscore too - itertuples() mangles those into
        # positional names like _5.
        unmatched["period_key"] = pd.to_datetime(unmatched["period"], errors="coerce")
        # Plain dates inside this module, Timestamps only where pandas needs
        # them (the merge key above). Mixing the two is what every failure in
        # this area has come down to: they satisfy each other's isinstance
        # checks but refuse to compare.
        pending: dict[str, list[date]] = {}
        for row in unmatched.itertuples():
            period = as_date(row.period_key)
            if period is not None:
                pending.setdefault(str(row.symbol), []).append(period)

        limit = int(self.cfg("symbols_per_run", 60))
        symbols = sorted(pending)[:limit]
        self.log.info(
            "resolving announcement dates for %d/%d symbols with pending surprises",
            len(symbols),
            len(pending),
        )

        client = HttpClient(
            requests_per_second=float(self.cfg("requests_per_second", 5.0)),
            user_agent=self.config.secrets.sec_user_agent,
            max_attempts=3,
            timeout=90,
        )
        resolved: list[dict[str, Any]] = []
        try:
            cik_map = self._cik_map(client, symbols, result)

            for symbol in symbols:
                cik = cik_map.get(symbol)
                if cik is None:
                    result.items_skipped += 1
                    continue
                try:
                    announcements = self._item_202_filings(client, cik)
                    result.items_succeeded += 1
                except Exception as exc:
                    result.items_failed += 1
                    self.log.debug("submissions %s: %s", symbol, exc)
                    continue
                if announcements:
                    resolved.extend(self._match(symbol, pending[symbol], announcements))
        finally:
            client.close()

        if not resolved:
            result.add_warning("no announcement dates could be matched")
            return

        # Join the resolved dates onto the full stored rows so every EPS field
        # survives the write.
        dates = pd.DataFrame(resolved)
        # Normalise both merge keys on both frames. Every failure in this area
        # has come from a key whose two sides looked identical but were not:
        # DuckDB returns strings as `category`, and a column of date objects is
        # re-inferred to datetime64 on assignment while a freshly built one
        # stays object. Either mismatch makes pandas raise while factorising the
        # key rather than compare by value.
        for frame in (unmatched, dates):
            frame["symbol"] = frame["symbol"].astype(str)
            frame["period_key"] = pd.to_datetime(frame["period_key"], errors="coerce")
        df = unmatched.merge(dates, on=["symbol", "period_key"], how="inner", suffixes=("", "_new"))
        if df.empty:
            result.add_warning("resolved dates did not join back onto any stored row")
            return

        df["announcement_date"] = df["announcement_date_new"]
        df["announcement_accession"] = df["announcement_accession_new"]
        df = df.drop(columns=[c for c in df.columns if c.endswith("_new") or c == "period_key"])
        df["ingested_at"] = datetime.now(UTC)

        write = self.writer.write(
            df, P.EARNINGS, self.paths.earnings_file("surprises"), mode="merge"
        )
        result.record_write(write)

        lags = (pd.to_datetime(df["announcement_date"]) - pd.to_datetime(df["period"])).dt.days
        result.details.update(
            {
                "periods_dated": len(df),
                "symbols": int(df["symbol"].nunique()),
                "median_lag_days": int(lags.median()),
                "max_lag_days": int(lags.max()),
            }
        )
        self.log.info(
            "dated %d periods across %d symbols; announcements land a median of %d days "
            "after period end",
            len(df),
            df["symbol"].nunique(),
            int(lags.median()),
        )

    # ---------------------------------------------------------------- inputs

    def _unmatched_surprises(self) -> pd.DataFrame:
        """Whole surprise rows needing an announcement date.

        The complete row is carried through deliberately. The writer merges by
        replacing on the natural key, so writing back only
        (symbol, record_type, period, announcement_date) would overwrite the
        stored EPS figures with nulls -- turning an enrichment into data loss.

        Normally only undated rows are considered, which makes this expensive on
        the first pass and nearly free afterwards. ``resolve_all`` re-examines
        dated rows too: a change to the matching rule leaves already-stored
        dates wrong, and nothing else would ever revisit them.
        """
        from tickerlake.storage.query import LakeQuery

        predicate = "" if self.cfg("resolve_all", False) else " AND announcement_date IS NULL"
        # Deliberately not caught. An unreadable earnings table is a real
        # failure, and swallowing it here made the stage report "every stored
        # surprise already has an announcement date" when the query had in fact
        # errored -- a silent wrong answer rather than a visible fault.
        with LakeQuery(self.paths.root) as q:
            return q.sql(f"SELECT * FROM earnings WHERE record_type = 'surprise'{predicate}")

    def _cik_map(
        self, client: HttpClient, symbols: list[str], result: FetchResult
    ) -> dict[str, str]:
        try:
            payload = client.get_json(TICKER_MAP_URL)
        except Exception as exc:
            result.add_error(f"could not load SEC ticker map: {exc}")
            return {}

        wanted = set(symbols)
        out: dict[str, str] = {}
        for entry in payload.values():
            # SEC spells class shares with a dash; the lake uses dots.
            symbol = str(entry.get("ticker", "")).strip().upper().replace("-", ".")
            if symbol in wanted:
                out[symbol] = str(entry.get("cik_str", "")).zfill(10)
        return out

    def _item_202_filings(self, client: HttpClient, cik: str) -> list[tuple[date, str]]:
        """(filing_date, accession) for every 8-K carrying Item 2.02."""
        try:
            payload = client.get_json(SUBMISSIONS_URL.format(cik=cik))
        except HttpError:
            return []

        recent = (payload.get("filings") or {}).get("recent") or {}
        forms = recent.get("form") or []
        items = recent.get("items") or []
        dates = recent.get("filingDate") or []
        accessions = recent.get("accessionNumber") or []

        out: list[tuple[date, str]] = []
        for i, form in enumerate(forms):
            if form != "8-K":
                continue
            codes = items[i] if i < len(items) else ""
            # Codes arrive comma-separated, e.g. "2.02,9.01". Substring matching
            # would also hit codes like 12.02, so split on the separator.
            if EARNINGS_ITEM not in {c.strip() for c in str(codes).split(",")}:
                continue
            day = as_date(dates[i] if i < len(dates) else None)
            if day is None:
                continue
            out.append((day, accessions[i] if i < len(accessions) else ""))
        return sorted(out)

    # -------------------------------------------------------------- matching

    def _match(
        self, symbol: str, periods: list[date], announcements: list[tuple[date, str]]
    ) -> list[dict[str, Any]]:
        """Pair each period with the Item 2.02 filing that announced it.

        See the module docstring for why this is not simply "the first filing
        after the period ends".
        """
        max_lag = int(self.cfg("max_lag_days", 120))
        max_lead = int(self.cfg("max_lead_days", 45))
        tolerance = int(self.cfg("lag_tolerance_days", 45))
        default_lag = float(self.cfg("typical_lag_days", 30))

        ordered = sorted(set(periods))
        # Collapse same-day filings: an 8-K and its amendment are one event.
        # Accession numbers are issued in sequence, so sorting the pairs keeps
        # the original filing rather than the amendment.
        by_day: dict[date, str] = {}
        for day, accession in sorted(announcements):
            by_day.setdefault(day, accession)
        days = sorted(by_day)

        def window(period: date) -> list[date]:
            lo = period - timedelta(days=max_lead)
            hi = period + timedelta(days=max_lag)
            return [d for d in days if lo <= d <= hi]

        def assign(typical: float | None) -> dict[date, date]:
            """One-to-one period -> filing pairing at a given expected offset.

            Exclusivity is the point: letting one filing serve two periods is
            what recorded a single General Mills 8-K as both Q3 and Q4.
            """
            claimed: set[date] = set()
            out: dict[date, date] = {}
            for period in ordered:
                candidates = [d for d in window(period) if d not in claimed]
                if not candidates:
                    continue
                if typical is None:
                    # Seed pass: earliest in window, just to get a first read on
                    # the cadence.
                    pick = candidates[0]
                else:
                    # Ties break to the earlier filing: results are public from
                    # their first disclosure, and a later duplicate 8-K does not
                    # undo that.
                    _, pick = min((abs((d - period).days - typical), d) for d in candidates)
                    if abs((pick - period).days - typical) > tolerance:
                        # Further off the company's own cadence than a late
                        # report plausibly explains, so more likely an unrelated
                        # 2.02. Leave the period undated rather than date it
                        # wrongly.
                        continue
                claimed.add(pick)
                out[period] = pick
            return out

        # The offset between a period label and its announcement is a property
        # of the company's fiscal calendar, so read it off the company's own
        # filings rather than assuming a December year-end. It is negative for
        # anyone whose fiscal quarters close before the calendar ones.
        #
        # Iterate, because the first estimate is taken from a pairing that may
        # itself have claimed a decoy, and fixing the pairing fixes the
        # estimate. Honeywell needed this: an unrelated Item 2.02 in December
        # dragged the median down to 11 days, which then let a June decoy tie
        # with the real July release and win on the earlier-filing tie-break.
        # One more pass put the median back at 23 and the tie disappeared.
        assignment = assign(None)
        for _ in range(int(self.cfg("cadence_passes", 5))):
            lags = [(pick - period).days for period, pick in assignment.items()]
            typical = statistics.median(lags) if lags else default_lag
            nxt = assign(typical)
            if nxt == assignment:
                break
            assignment = nxt

        out: list[dict[str, Any]] = []
        for period, pick in sorted(assignment.items()):
            out.append(
                {
                    "symbol": symbol,
                    "period_key": pd.Timestamp(period),
                    "announcement_date": pick,
                    "announcement_accession": by_day[pick],
                }
            )
        return out
