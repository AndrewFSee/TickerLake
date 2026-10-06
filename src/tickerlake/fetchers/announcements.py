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
inferred, and every shortcut here has been wrong for somebody.

**Finnhub's ``period`` is not the fiscal close, and follows no single rule
relative to it.** Coca-Cola's quarter ending 2026-04-03 is labelled
2026-03-31, three days *before* the close; Applied Materials' ending
2026-07-26 is labelled 2026-09-30, sixty-six days *after*. A window around the
label is therefore ambiguous by one quarter in one direction or the other --
widen it enough for Applied Materials and Apple takes the previous quarter's
announcement, which is lookahead. Inferring the close from the label corrupted
eighteen symbols on one attempt.

So the close comes from SEC instead. Every 10-Q and 10-K carries a
``reportDate`` -- the period of report, exactly the fiscal close -- and its
quarter number follows from position: a 10-K closes Q4, and each 10-Q is Q1-Q3
by its distance from the preceding year end. A Finnhub row is tied to the close
of the *same fiscal quarter* near its label. Two closes of one quarter are a
year apart, so at most one qualifies, and the window is kept tight enough that
a numbering disagreement leaves the row undated rather than borrowing a
neighbour. The newest quarter is usually announced before its 10-Q exists, so
its close is projected from the same quarter a year earlier.

**Not every Item 2.02 filing is an earnings release.** Tesla files a delivery
report two days into every quarter; Occidental, APA, Prudential and Super
Micro pre-announce a week or two after each close; Goldman filed preliminary
numbers a week before its Q4 release. These streams are as regular as the real
thing, which is how a cadence estimate seeded on the earliest filing locked onto
them and dated those surprises two to six weeks early.

With the true close known, two facts pick the release. Nobody announces within
a week of closing the books. And a company releases earnings before, or with,
its 10-Q or 10-K -- true in 1,684 of 1,686 quarters checked -- while
pre-announcements come earlier still. The release is the *last* Item 2.02 at
least a week after the close and no later than the periodic report.

That rule errs late rather than early: Robinhood furnishes monthly metrics
under Item 2.02, so its date can land a few days after the real release. Late
costs timeliness; early hands a model a surprise before it existed.

Where no close can be established the older label-based matcher still runs,
with its cadence estimate. It covers two symbols.

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

# The SEC renumbered 8-K items on this date; before it, earnings releases were
# Item 12 and no filing carries 2.02. Periods older than this can never be
# matched, so asking about them costs requests and reports a warning on every
# run in perpetuity -- noise that would mask a real failure later.
ITEM_202_INTRODUCED = date(2004, 8, 23)


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
        # Finnhub's fiscal quarter per row: the stable half of the quarter's
        # identity, and what ties a row to its real close below.
        quarter_of: dict[tuple[str, date], int | None] = {}
        for row in unmatched.itertuples():
            period = as_date(row.period_key)
            if period is not None:
                pending.setdefault(str(row.symbol), []).append(period)
                fq = getattr(row, "fiscal_quarter", None)
                quarter_of[(str(row.symbol), period)] = None if pd.isna(fq) else int(fq)

        # Only companies. A fund has no 8-K to find, and asking the SEC about one
        # just spends a request to learn that.
        if self.tracker is not None and self.symbols_override is None:
            companies = self.tracker.constituent_symbols()
            pending = {s: p for s, p in pending.items() if s in companies}

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
        # Symbols whose filings were actually read. Only these can be said to
        # have no release on record: a lookup that failed says nothing.
        examined: set[str] = set()
        anchored = 0
        try:
            cik_map = self._cik_map(client, symbols, result)

            for symbol in symbols:
                cik = cik_map.get(symbol)
                if cik is None:
                    result.items_skipped += 1
                    continue
                try:
                    recent = self._recent_filings(client, cik)
                    result.items_succeeded += 1
                except Exception as exc:
                    result.items_failed += 1
                    self.log.debug("submissions %s: %s", symbol, exc)
                    continue
                if recent:
                    examined.add(symbol)
                announcements = self._parse_202(recent)
                anchors = self._anchors(
                    [(p, quarter_of.get((symbol, p))) for p in pending[symbol]],
                    self._quarter_closes(recent),
                    self._report_filed(recent),
                )
                anchored += anchors is not None
                if announcements:
                    resolved.extend(self._match(symbol, pending[symbol], announcements, anchors))
        finally:
            client.close()

        result.details["symbols_anchored_on_fiscal_close"] = anchored
        result.details["symbols_on_label_fallback"] = len(symbols) - anchored

        # Normalise both merge keys on both frames. Every failure in this area
        # has come from a key whose two sides looked identical but were not:
        # DuckDB returns strings as `category`, and a column of date objects is
        # re-inferred to datetime64 on assignment while a freshly built one
        # stays object. Either mismatch makes pandas raise while factorising the
        # key rather than compare by value.
        unmatched["symbol"] = unmatched["symbol"].astype(str)
        unmatched["period_key"] = pd.to_datetime(unmatched["period_key"], errors="coerce")

        dated = self._apply_dates(unmatched, resolved, result)
        self._record_misses(unmatched, dated, examined, run_date, result)

    def _apply_dates(
        self, unmatched: pd.DataFrame, resolved: list[dict[str, Any]], result: FetchResult
    ) -> set[tuple[str, pd.Timestamp]]:
        """Write the resolved dates onto the full stored rows; return their keys."""
        if not resolved:
            return set()
        # Join onto the full stored rows so every EPS field survives the write.
        dates = pd.DataFrame(resolved)
        dates["symbol"] = dates["symbol"].astype(str)
        dates["period_key"] = pd.to_datetime(dates["period_key"], errors="coerce")
        df = unmatched.merge(dates, on=["symbol", "period_key"], how="inner", suffixes=("", "_new"))
        if df.empty:
            result.add_warning("resolved dates did not join back onto any stored row")
            return set()

        keys = set(zip(df["symbol"], df["period_key"], strict=True))
        df["announcement_date"] = df["announcement_date_new"]
        df["announcement_accession"] = df["announcement_accession_new"]
        if "fiscal_period_end_new" in df.columns:
            df["fiscal_period_end"] = df["fiscal_period_end_new"]
        # A dated quarter is no longer a miss.
        df["announcement_checked"] = None
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
        return keys

    def _record_misses(
        self,
        unmatched: pd.DataFrame,
        dated: set[tuple[str, pd.Timestamp]],
        examined: set[str],
        run_date: date,
        result: FetchResult,
    ) -> None:
        """Note quarters that could not be dated, warning only about new ones.

        Some quarters are undatable for good: Exxon, AES and ONEOK publish
        earnings under Item 7.01 rather than 2.02, and Berkshire releases with
        its 10-Q. Retrying them costs a request a night and is harmless, but
        warning about them every night made the warning meaningless -- it fired
        on a run where nothing new had gone wrong, which is how a real failure
        gets ignored.

        So a miss warns once: the first time a quarter is found undatable, and
        only while its release could still be recent. A quarter whose window
        closed long ago is history rather than news. Every miss is stamped
        with ``announcement_checked`` so the next run knows it has been seen;
        a broken matcher still shows up at once, as a burst of new misses
        across companies that normally date cleanly.
        """
        if not examined or unmatched.empty:
            return
        # resolve_all re-examines rows that already carry a date; failing to
        # re-find one is not a miss, and must not be recorded as one.
        looked = unmatched["symbol"].isin(examined) & unmatched["announcement_date"].isna()
        missed = looked & ~pd.Series(
            [
                (s, k) in dated
                for s, k in zip(unmatched["symbol"], unmatched["period_key"], strict=True)
            ],
            index=unmatched.index,
        )
        if not missed.any():
            return

        # The latest a release can fall after its label, on either matching path.
        window = max(
            int(self.cfg("close_max_days_after_label", 10))
            + int(self.cfg("max_days_after_quarter_end", 75)),
            int(self.cfg("max_lag_days", 120)),
        )
        labels = pd.to_datetime(unmatched["period"]).dt.date
        recent = labels.map(lambda d: (run_date - d).days <= window)
        if "announcement_checked" in unmatched.columns:
            first_time = unmatched["announcement_checked"].isna()
        else:
            first_time = pd.Series(True, index=unmatched.index)

        new = unmatched[missed & recent & first_time]
        known = int(missed.sum()) - len(new)
        result.details["undated_known"] = known
        result.details["undated_new"] = len(new)
        if len(new):
            listing = ", ".join(
                f"{r.symbol} {pd.Timestamp(r.period).date()}" for r in new.head(8).itertuples()
            )
            result.add_warning(
                f"{len(new)} recently reported quarter(s) could not be dated: {listing}"
                + (" ..." if len(new) > 8 else "")
            )
        else:
            self.log.info("%d quarter(s) remain undatable, none of them new", known)

        stamped = unmatched[missed].drop(columns=["period_key"]).copy()
        stamped["announcement_checked"] = run_date
        write = self.writer.write(
            stamped, P.EARNINGS, self.paths.earnings_file("surprises"), mode="merge"
        )
        result.record_write(write)

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

        Periods predating :data:`ITEM_202_INTRODUCED` are never requested, since
        no filing can carry the item code that identifies them.
        """
        from tickerlake.storage.query import LakeQuery

        where = ["record_type = 'surprise'", "period >= ?"]
        if not self.cfg("resolve_all", False):
            where.append("announcement_date IS NULL")
        # Deliberately not caught. An unreadable earnings table is a real
        # failure, and swallowing it here made the stage report "every stored
        # surprise already has an announcement date" when the query had in fact
        # errored -- a silent wrong answer rather than a visible fault.
        with LakeQuery(self.paths.root) as q:
            return q.sql(
                f"SELECT * FROM earnings WHERE {' AND '.join(where)}",
                [ITEM_202_INTRODUCED],
            )

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

    def _recent_filings(self, client: HttpClient, cik: str) -> dict[str, Any]:
        """The ``filings.recent`` block of a company's submissions record."""
        try:
            payload = client.get_json(SUBMISSIONS_URL.format(cik=cik))
        except HttpError:
            return {}
        return (payload.get("filings") or {}).get("recent") or {}

    def _item_202_filings(self, client: HttpClient, cik: str) -> list[tuple[date, str]]:
        """(filing_date, accession) for every 8-K carrying Item 2.02."""
        return self._parse_202(self._recent_filings(client, cik))

    @staticmethod
    def _parse_202(recent: dict[str, Any]) -> list[tuple[date, str]]:
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

    @staticmethod
    def _quarter_closes(recent: dict[str, Any]) -> list[tuple[date, int | None]]:
        """(close, fiscal quarter) for every 10-Q and 10-K on record.

        ``reportDate`` is SEC's period of report: exactly the fiscal period end,
        one per filing. It is the authoritative close. An earlier attempt read
        the latest date in a filing's XBRL facts instead, and got cover-page
        dates -- "shares outstanding as of 16 January" -- which put Apple's
        December quarter in the middle of January.

        The quarter number comes from position, which needs no naming
        convention: a 10-K closes Q4, and each 10-Q is Q1, Q2 or Q3 by how far
        it falls after the preceding year end. Costco's 12/12/12/16-week
        quarters land on 84, 168 and 252 days, which still round cleanly.
        """
        forms = recent.get("form") or []
        reports = recent.get("reportDate") or []
        years: set[date] = set()
        quarters: set[date] = set()
        for i, form in enumerate(forms):
            kind = str(form).split("/")[0]  # 10-K/A reports the same period
            if kind not in ("10-K", "10-KT", "10-Q"):
                continue
            day = as_date(reports[i] if i < len(reports) else None)
            if day is not None:
                # A 10-KT is the transition report a company files when it
                # changes fiscal year, and it closes the new year exactly as a
                # 10-K would. Ignoring it numbered every later quarter from the
                # old year end: Ferguson moved from July to December in 2025,
                # its March 2026 quarter came out as Q3 rather than Q1, and the
                # whole symbol fell back to label matching -- which dated two
                # quarters with the previous quarter's release.
                (years if kind in ("10-K", "10-KT") else quarters).add(day)

        ends = sorted(years)
        out: list[tuple[date, int | None]] = [(d, 4) for d in ends]
        for day in sorted(quarters - years):
            prior = [e for e in ends if e < day]
            later = [e for e in ends if e > day]
            if prior:
                q: int | None = round((day - prior[-1]).days / 91.3)
            elif later:
                q = 4 - round((later[0] - day).days / 91.3)
            else:
                # No year end on record at all -- a spin-off before its first
                # 10-K, like Honeywell Aerospace. The close is still a fact;
                # it just cannot be numbered.
                q = None
            if q is None or 1 <= q <= 3:
                out.append((day, q))
            # Anything else means a missing filing; better to leave that
            # quarter out than to number it wrongly.
        return sorted(out, key=lambda c: c[0])

    @staticmethod
    def _report_filed(recent: dict[str, Any]) -> dict[date, date]:
        """When each 10-Q / 10-K was first filed, keyed by the period it closes.

        The earliest filing per period, so a later amendment does not stand in
        for the original.
        """
        forms = recent.get("form") or []
        reports = recent.get("reportDate") or []
        filed = recent.get("filingDate") or []
        out: dict[date, date] = {}
        for i, form in enumerate(forms):
            if str(form).split("/")[0] not in ("10-K", "10-KT", "10-Q"):
                continue
            close = as_date(reports[i] if i < len(reports) else None)
            day = as_date(filed[i] if i < len(filed) else None)
            if close is not None and day is not None:
                out[close] = min(day, out.get(close, day))
        return out

    def _anchors(
        self,
        rows: list[tuple[date, int | None]],
        closes: list[tuple[date, int | None]],
        filed: dict[date, date] | None = None,
    ) -> dict[date, tuple[date, bool, date | None]] | None:
        """Each period's real fiscal close, or None to fall back on the label.

        Finnhub's label follows no single rule relative to the close. Coca-Cola's
        quarter ending 2026-04-03 is labelled 2026-03-31, three days *before*;
        Applied Materials' ending 2026-07-26 is labelled 2026-09-30, 66 days
        after. So a window around the label is ambiguous by one quarter in one
        direction or the other, and anchoring on "the close nearest the label"
        corrupted eighteen symbols with the previous quarter's announcement.

        The fiscal quarter number removes that. A close must match the row's
        quarter *and* sit near its label, and two closes of the same quarter
        are a year apart, so at most one qualifies. The window is kept tight on
        purpose: if Finnhub and SEC ever disagree on quarter numbering, the
        wrong-quarter close is ~91 days away and falls outside it, leaving the
        period undated rather than dated a quarter early.

        The newest quarter is normally announced before its 10-Q exists. Its
        close is projected from the same quarter a year earlier -- which copes
        with uneven quarters like Costco's -- and is marked inexact.

        Returns None unless every period anchors, because the matcher's cadence
        estimate needs one consistent basis.
        """
        before = int(self.cfg("close_max_days_before_label", 75))
        after = int(self.cfg("close_max_days_after_label", 10))
        filed = filed or {}
        out: dict[date, tuple[date, bool, date | None]] = {}
        for label, fq in rows:
            if fq is None:
                return None
            lo, hi = label - timedelta(days=before), label + timedelta(days=after)
            same = [d for d, q in closes if q == fq]
            inside = [d for d in same if lo <= d <= hi]
            if not inside:
                # An unnumbered close may stand in when it is the only one near
                # the label. The window is shorter than a quarter, so two
                # regular closes cannot both fall inside it.
                loose = [d for d, q in closes if q is None and lo <= d <= hi]
                if len(loose) == 1:
                    inside = loose
            if inside:
                close = max(inside)
                out[label] = (close, True, filed.get(close))
                continue
            earlier = [d for d in same if d < lo]
            found = None
            for years in (1, 2):
                if not earlier:
                    break
                guess = earlier[-1] + timedelta(days=364 * years)
                if lo <= guess <= hi:
                    found = guess
                    break
            if found is None:
                return None
            out[label] = (found, False, None)
        return out

    # -------------------------------------------------------------- matching

    def _match(
        self,
        symbol: str,
        periods: list[date],
        announcements: list[tuple[date, str]],
        anchors: dict[date, tuple[date, bool, date | None]] | None = None,
    ) -> list[dict[str, Any]]:
        """Pair each period with the Item 2.02 filing that announced it.

        With ``anchors`` -- each period's true fiscal close -- this defers to
        :meth:`_match_anchored`. What follows is the fallback for a company whose
        closes cannot be established: it works from Finnhub's label, which is
        ambiguous by a quarter, and so leans on the cadence estimate below.
        """
        if anchors:
            return self._match_anchored(symbol, periods, announcements, anchors)

        tolerance = int(self.cfg("lag_tolerance_days", 45))
        default_lag = float(self.cfg("typical_lag_days", 30))

        lead = int(self.cfg("max_lead_days", 45))
        lag = int(self.cfg("max_lag_days", 120))

        ordered = sorted(set(periods))
        # Collapse same-day filings: an 8-K and its amendment are one event.
        # Accession numbers are issued in sequence, so sorting the pairs keeps
        # the original filing rather than the amendment.
        by_day: dict[date, str] = {}
        for day, accession in sorted(announcements):
            by_day.setdefault(day, accession)
        days = sorted(by_day)

        def window(period: date) -> list[date]:
            lo = period - timedelta(days=lead)
            hi = period + timedelta(days=lag)
            return [d for d in days if lo <= d <= hi]

        def offset(pick: date, period: date) -> int:
            return (pick - period).days

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
                    _, pick = min((abs(offset(d, period) - typical), d) for d in candidates)
                    if abs(offset(pick, period) - typical) > tolerance:
                        # Further off the company's own cadence than a late
                        # report plausibly explains, so more likely an unrelated
                        # 2.02. Leave the period undated rather than date it
                        # wrongly.
                        continue
                claimed.add(pick)
                out[period] = pick
            return out

        # The offset between a period and its announcement is a property of the
        # company, so read it off the company's own filings. Iterate, because
        # the first estimate comes from a pairing that may itself have claimed a
        # decoy, and fixing the pairing fixes the estimate. Honeywell needed
        # this: an unrelated Item 2.02 dragged the median to 11 days, which then
        # let a June decoy tie with the real July release. Goldman still needs
        # it even when anchored -- its decoy on 2026-01-08 falls inside the
        # window, a week ahead of the real release on the 15th.
        assignment = assign(None)
        for _ in range(int(self.cfg("cadence_passes", 5))):
            lags = [offset(pick, period) for period, pick in assignment.items()]
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
                    # Only an exact close is recorded; a projected one is at
                    # most a week out, but it is not the company's statement.
                    "fiscal_period_end": None,
                }
            )
        return out

    def _match_anchored(
        self,
        symbol: str,
        periods: list[date],
        announcements: list[tuple[date, str]],
        anchors: dict[date, tuple[date, bool, date | None]],
    ) -> list[dict[str, Any]]:
        """The latest Item 2.02 between the close and the periodic report.

        With the true close known, the only question left is which of several
        Item 2.02 filings in a quarter is the earnings release, and two facts
        answer it. Nobody announces within days of closing the books, so a
        filing under a week after the close is not the release -- Tesla files
        its delivery report two days into every quarter. And a company releases
        earnings before, or with, its 10-Q or 10-K: that held in 1,684 of 1,686
        quarters checked. Pre-announcements come earlier still, so the release
        is the *last* 2.02 on or before the periodic report.

        Seeding on the earliest filing instead -- what the cadence estimate did
        -- is exactly wrong for a company that pre-announces every quarter:
        Occidental files an Item 2.02 about ten days after each close and its
        earnings about five weeks after, and the regular early stream won,
        dating every Occidental surprise four weeks early. APA, Prudential,
        Super Micro and The Trade Desk had the same shape.

        The rule can err the other way -- Robinhood furnishes monthly metrics
        under Item 2.02, so it may pick one of those a few days after the real
        release. Late is the safe direction here: it costs a few days of
        timeliness, where early hands a model the surprise before it existed.
        """
        floor = timedelta(days=int(self.cfg("min_days_after_quarter_end", 7)))
        cap = timedelta(days=int(self.cfg("max_days_after_quarter_end", 75)))
        # EDGAR stamps an after-hours filing with the next business day, so a
        # release can carry a date one day after the report it preceded.
        slack = timedelta(days=int(self.cfg("report_slack_days", 2)))

        by_day: dict[date, str] = {}
        for day, accession in sorted(announcements):
            by_day.setdefault(day, accession)
        days = sorted(by_day)

        claimed: set[date] = set()
        out: list[dict[str, Any]] = []
        for period in sorted(set(periods)):
            close, exact, filed = anchors[period]
            hi = close + cap
            if filed is not None:
                hi = min(hi, filed + slack)
            candidates = [d for d in days if close + floor <= d <= hi and d not in claimed]
            if not candidates:
                continue
            pick = candidates[-1]
            claimed.add(pick)
            out.append(
                {
                    "symbol": symbol,
                    "period_key": pd.Timestamp(period),
                    "announcement_date": pick,
                    "announcement_accession": by_day[pick],
                    # Only an exact close is recorded; a projected one is at
                    # most a week out, but it is not the company's statement.
                    "fiscal_period_end": close if exact else None,
                }
            )
        return out
