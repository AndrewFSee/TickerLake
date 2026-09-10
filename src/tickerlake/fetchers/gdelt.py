"""GDELT news and tone, via the bulk file feed.

Why not the DOC query API
-------------------------
GDELT's ``/api/v2/doc/doc`` query endpoint returns HTTP 429 for minutes at a time
regardless of pacing or User-Agent -- it was throttling continuously during
development, and a circuit breaker was needed just to stop it stalling the
pipeline. It is a free public service under heavy query load.

The **bulk file feed** is a different system: static ZIPs on a CDN, published
every 15 minutes, with no query cost to GDELT and no rate limiting. Same data,
delivered the way GDELT actually wants you to consume it at volume. It is also
strictly richer -- the Global Knowledge Graph carries themes, organisations, and
a full tone breakdown per article, where the DOC API returns a single tone
scalar.

Volume management
-----------------
Each 15-minute GKG file is ~6 MB zipped and holds ~1,400 articles worldwide, so
a full day is 96 files and roughly 600 MB. That is more bandwidth than a daily
job should spend on a sentiment feature, so ``max_files`` caps how far back each
run reaches (default 16 files = the most recent 4 hours, which for an
after-close run covers the US trading afternoon). Raise it for wider coverage.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd

from tickerlake.fetchers.base import BaseFetcher, FetchResult
from tickerlake.storage import paths as P
from tickerlake.utils.http import HttpClient

SOURCE = "gdelt"

LAST_UPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
MASTER_LIST_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"

# GKG 2.1 column positions we use. The file has 27 tab-delimited fields and no
# header, so these indices are the schema.
COL_RECORD_ID = 0
COL_DATE = 1
COL_SOURCE_NAME = 3
COL_DOCUMENT_ID = 4
COL_THEMES = 7
COL_ORGANISATIONS = 13
COL_V2ORGANISATIONS = 14
COL_V2TONE = 15
COL_ALL_NAMES = 23
GKG_MIN_COLUMNS = 24


class GdeltFetcher(BaseFetcher):
    """Article-level tone and themes from the GDELT Global Knowledge Graph."""

    name = "gdelt"
    dataset = P.NEWS_EVENTS

    def collect(self, run_date: date, result: FetchResult) -> None:
        client = HttpClient(
            requests_per_second=float(self.cfg("requests_per_second", 2.0)),
            user_agent="TickerLake/0.1 (research)",
            max_attempts=int(self.cfg("max_attempts", 3)),
            timeout=120,
        )
        max_files = int(self.cfg("max_files", 16))
        self._max_offset = int(self.cfg("max_organisation_offset", 250))

        try:
            urls = self._recent_gkg_urls(client, max_files, result)
            if not urls:
                result.add_error("could not list GDELT bulk files")
                return

            themes = {t.upper() for t in (self.cfg("themes", []) or [])}
            companies = self._company_index()
            self.log.info(
                "processing %d GKG file(s); matching %d themes and %d company names",
                len(urls),
                len(themes),
                len(companies),
            )

            rows: list[dict[str, Any]] = []
            for url in urls:
                try:
                    rows.extend(self._process_file(client, url, themes, companies, run_date))
                    result.items_succeeded += 1
                except Exception as exc:
                    result.items_failed += 1
                    result.add_warning(f"{url.rsplit('/', 1)[-1]}: {type(exc).__name__}: {exc}")
        finally:
            client.close()

        if not rows:
            result.add_warning("no articles matched the configured themes or companies")
            return

        df = pd.DataFrame(rows).drop_duplicates(subset=["event_id"])
        write = self.writer.write(
            df, P.NEWS_EVENTS, self.paths.news_file(run_date, SOURCE), mode="merge"
        )
        result.record_write(write)
        result.details.update(
            {
                "files_processed": result.items_succeeded,
                "articles_matched": len(df),
                "company_tagged": int(df["symbol"].notna().sum()),
                "mean_tone": round(float(df["tone"].mean()), 3) if len(df) else None,
            }
        )
        self.log.info(
            "matched %d articles (%d tagged to a symbol), mean tone %.2f",
            len(df),
            int(df["symbol"].notna().sum()),
            df["tone"].mean(),
        )

    # ----------------------------------------------------------- file listing

    def _recent_gkg_urls(
        self, client: HttpClient, max_files: int, result: FetchResult
    ) -> list[str]:
        """Most recent GKG file URLs, newest first."""
        try:
            latest = client.get_text(LAST_UPDATE_URL)
        except Exception as exc:
            result.add_warning(f"lastupdate.txt unreachable: {exc}")
            return []

        newest = None
        for line in latest.strip().splitlines():
            parts = line.split()
            if len(parts) == 3 and parts[2].endswith("gkg.csv.zip"):
                newest = parts[2]
        if not newest:
            return []
        if max_files <= 1:
            return [newest]

        # Walk backwards in 15-minute steps from the newest stamp rather than
        # downloading the 127 MB master list to find neighbours.
        stamp = newest.rsplit("/", 1)[-1].split(".")[0]
        try:
            cursor = datetime.strptime(stamp, "%Y%m%d%H%M%S")
        except ValueError:
            return [newest]

        base = newest.rsplit("/", 1)[0]
        urls = []
        for i in range(max_files):
            t = cursor - pd.Timedelta(minutes=15 * i)
            urls.append(f"{base}/{t.strftime('%Y%m%d%H%M%S')}.gkg.csv.zip")
        return urls

    # -------------------------------------------------------------- parsing

    def _process_file(
        self,
        client: HttpClient,
        url: str,
        themes: set[str],
        companies: dict[str, str],
        run_date: date,
    ) -> list[dict[str, Any]]:
        payload = client.get(url).content
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            name = zf.namelist()[0]
            raw = zf.read(name).decode("utf-8", "replace")

        now = datetime.now(UTC)
        out = []
        for line in raw.split("\n"):
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) < GKG_MIN_COLUMNS:
                continue

            article_themes = fields[COL_THEMES].upper()
            matched_theme = next((t for t in themes if t in article_themes), None)

            matched_symbol = _match_company(
                fields[COL_V2ORGANISATIONS], companies, self._max_offset
            )

            # Keep an article only if it is about a tracked company or one of the
            # configured macro themes; GKG is global and mostly not financial.
            if matched_theme is None and matched_symbol is None:
                continue

            url_doc = fields[COL_DOCUMENT_ID].strip()
            if not url_doc:
                continue

            tone = _parse_tone(fields[COL_V2TONE])
            published = _parse_stamp(fields[COL_DATE])
            out.append(
                {
                    "event_id": hashlib.sha1(
                        f"{url_doc}|{matched_symbol or matched_theme or ''}".encode()
                    ).hexdigest(),
                    "published_at": published,
                    "date": published.date() if published else run_date,
                    "symbol": matched_symbol,
                    "title": None,
                    "summary": None,
                    "url": url_doc,
                    "domain": fields[COL_SOURCE_NAME].strip() or None,
                    "language": None,
                    "country": None,
                    "theme": matched_theme,
                    "tone": tone.get("tone"),
                    "category": "company" if matched_symbol else "theme",
                    "embedding_status": "pending",
                    "source": SOURCE,
                    "ingested_at": now,
                }
            )
        return out

    # ------------------------------------------------------- company matching

    def _company_index(self) -> dict[str, str]:
        """UPPERCASE company name -> symbol, for matching GKG organisation fields.

        Short names are dropped: matching "V" or "GM" as substrings against a
        global news corpus produces far more noise than signal.
        """
        if self.tracker is None:
            return {}
        df = self.tracker.load()
        if df.empty or "company_name" not in df.columns:
            return {}

        current = df[df["end_date"].isna() & df["company_name"].notna()]
        index: dict[str, str] = {}
        for row in current.itertuples():
            name = _clean_company_name(str(row.company_name))
            # Reject names that cannot identify a company on their own: too
            # short to be distinctive, or an ordinary word that appears in
            # unrelated news constantly.
            if len(name) < 6 or name in _GENERIC_NAMES:
                continue
            index[name] = row.symbol
        return index


# ------------------------------------------------------------------ helpers

# Sorted longest-first at module load. Order is not cosmetic: matching " CO"
# before " & CO" turns "JPMorgan Chase & Co" into "JPMorgan Chase &" rather than
# "JPMorgan Chase", and the dangling ampersand then never matches anything GDELT
# emits.
_SUFFIXES = tuple(
    sorted(
        (
            " INC",
            " INC.",
            " CORP",
            " CORP.",
            " CORPORATION",
            " CO",
            " CO.",
            " COMPANY",
            " COMPANIES",
            " PLC",
            " LTD",
            " LTD.",
            " LLC",
            " GROUP",
            " HOLDINGS",
            " HOLDING",
            " (THE)",
            " & CO",
            " AND CO",
            " N.V.",
            " NV",
            " SA",
            " AG",
        ),
        key=len,
        reverse=True,
    )
)

# Single-word company names that are also ordinary English or ubiquitous in
# financial copy. Matching these by name tags an enormous amount of unrelated
# news: "Southern" appears in any regional story, and "Nasdaq" in essentially
# every equity article regardless of subject.
_GENERIC_NAMES = {
    "SOUTHERN",
    "NASDAQ",
    "TARGET",
    "GAP",
    "APA",
    "ALLY",
    "GENERAL",
    "UNION",
    "PUBLIC",
    "AMERICAN",
    "NATIONAL",
    "GLOBAL",
    "UNITED",
    "FIRST",
    "MARKET",
    "CAPITAL",
    "PROGRESSIVE",
    "PRINCIPAL",
    "FRANKLIN",
    "NEWS",
    "MATCH",
    "HOST",
    "SEALED AIR",
    "AIR",
    "CHARTER",
    "CROWN",
    "EQUITY",
    "PARAMOUNT",
}


def _match_company(v2_organisations: str, companies: dict[str, str], max_offset: int) -> str | None:
    """Tag an article to a symbol using GDELT's organisation *salience offsets*.

    ``V2Organizations`` is ``Name,charOffset;Name,charOffset;...`` where the
    offset is where the mention occurs in the article body. That distinction is
    what separates "this article is about JPMorgan" from "this article about
    Calix quotes a JPMorgan analyst" -- the subject of a story appears in the
    headline or lede (offset under ~100), while incidental mentions land at 400+.

    Naive substring matching over the whole field cannot tell those apart, and
    ends up tagging a wire story about any small-cap to whichever bank published
    a note on it. Requiring an early mention, and taking the earliest when
    several match, fixes it.
    """
    if not v2_organisations:
        return None

    best_symbol, best_offset = None, None
    for entry in v2_organisations.split(";"):
        name, _, offset_raw = entry.rpartition(",")
        if not name:
            continue
        try:
            offset = int(offset_raw)
        except ValueError:
            continue
        if offset > max_offset:
            continue

        symbol = companies.get(_clean_company_name(name))
        if symbol is not None and (best_offset is None or offset < best_offset):
            best_symbol, best_offset = symbol, offset

    return best_symbol


def _clean_company_name(name: str) -> str:
    """Strip corporate suffixes so 'Apple Inc' matches GDELT's 'Apple'."""
    cleaned = name.strip().upper()
    changed = True
    while changed:
        changed = False
        for suffix in _SUFFIXES:
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)].strip()
                changed = True
    # Trailing "&" is left behind by names like "Estee Lauder Companies (The)"
    # once the suffix comes off, and would never match GDELT output.
    return cleaned.strip(" ,.&-")


def _parse_tone(value: str) -> dict[str, float]:
    """V2Tone is a comma-separated tuple; the first element is overall tone."""
    parts = value.split(",")
    keys = (
        "tone",
        "positive",
        "negative",
        "polarity",
        "activity_density",
        "self_ref_density",
        "word_count",
    )
    out: dict[str, float] = {}
    for key, raw in zip(keys, parts, strict=False):
        try:
            out[key] = float(raw)
        except (TypeError, ValueError):
            continue
    return out


def _parse_stamp(value: str) -> datetime | None:
    try:
        return datetime.strptime(value.strip(), "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    except (ValueError, AttributeError):
        return None
