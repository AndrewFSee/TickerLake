# TickerLake

Automated daily collection of free financial data into a queryable Parquet + DuckDB
lake, designed for ML feature engineering. Idempotent, resumable, and
**survivorship-bias-free by construction**.

```
yfinance ─┐
SEC EDGAR ─┤
FRED      ─┼──> fetchers ──> validate ──> Parquet (zstd) ──> DuckDB SQL
GDELT     ─┤                                    │
Finnhub   ─┘                                    └──> weekly compaction
                    ▲
        point-in-time membership table
        (decides who to fetch, and who was a member on date X)
```

---

## Quick start

```powershell
# 1. Environment
uv venv
uv pip install -e ".[dev]"

# 2. Secrets
copy .env.example .env
#    Fill in FRED_API_KEY and FINNHUB_API_KEY (both free).
#    SEC_USER_AGENT is required and must contain a real contact address.

# 3. Check everything is reachable
.venv\Scripts\python.exe -m tickerlake.cli doctor

# 4. Seed the membership table (do this before anything else)
.venv\Scripts\python.exe -m tickerlake.cli init

# 5. Smoke-test the expensive stage on 10 symbols first
.venv\Scripts\python.exe -m tickerlake.cli options --limit 10

# 6. Backfill price history (one-time, ~30-60 min for 2010-present)
.venv\Scripts\python.exe -m tickerlake.cli backfill ohlcv

# 7. Full daily run
.venv\Scripts\python.exe -m tickerlake.cli run
```

---

## Why the membership table comes first

Survivorship bias is not a post-processing concern here; it is the schema.

Most free datasets give you *today's* S&P 500 and nothing else. Backtest against
that and you have quietly assumed that every company in your universe survived to
the present — which is exactly the companies that did well. TickerLake's
`membership` table stores **intervals**, not a list:

| symbol | index | start_date | end_date | reason_removed |
|---|---|---|---|---|
| AAPL | SP500 | 2010-01-01 | *null* | |
| AAL  | SP500 | 2015-03-23 | 2024-09-23 | seed:index_removal |
| AAL  | SP500 | 2010-01-01 | 2013-12-09 | seed:index_removal |

`end_date IS NULL` means still a member. Four invariants are enforced in code and
covered by tests:

1. **Nothing is ever deleted.** Index removal *closes an interval*; it never drops
   a row. `tickerlake` keeps collecting a delisted name's data for as long as any
   free source still answers.
2. **History is never rewritten from today's list.** Today's constituents can only
   open new intervals or close current ones as of today.
3. **Re-entry creates a second interval.** 18 symbols in the 2010+ window have
   left and rejoined; collapsing those into one interval would silently claim
   membership during the gap.
4. **Implausible changes are rejected, not applied.** A refresh that would remove
   more than 25 symbols in one day is refused and logged — the likely cause is
   Wikipedia changing its page layout, not an index event. The stored table is
   left untouched.

Use the survivorship-safe query entry points:

```python
from tickerlake.storage.query import LakeQuery

with LakeQuery("data") as q:
    q.members_on("2015-06-30")           # who was actually in the index that day
    q.pit_ohlcv("2015-01-01", "2020-12-31")   # panel filtered per-row by membership
    q.ohlcv(symbols=["AAL"])             # raw table: includes delisted names on purpose
```

`pit_ohlcv` joins each row to the membership interval covering **that row's own
date**, so a name added in 2018 contributes nothing to 2015 and a name removed in
2020 still contributes its pre-removal history.

### Silent delisting detection

A company usually stops returning data from free sources *before* any official
index removal shows up. Each run records whether every tracked symbol returned
data. After `silent_delist_threshold_runs` consecutive empty runs a symbol is
flagged `suspected_delisted` — flagged, never deleted — and surfaced in the run
summary and `tickerlake status`. If it starts returning data again the flag
clears automatically.

---

## Project layout

```
TickerLake/
├── config/config.yaml          # all tunables; no secrets
├── .env                        # secrets only (gitignored)
├── src/tickerlake/
│   ├── config.py               # YAML + .env, dotted access, env overrides
│   ├── logging_setup.py        # console + rotating file logs
│   ├── cli.py                  # command line entry point
│   ├── storage/
│   │   ├── paths.py            # dataset path layout & partitioning
│   │   ├── schemas.py          # PyArrow schemas + pre-write validation rules
│   │   ├── writer.py           # atomic, schema-coercing, de-duplicating writer
│   │   └── query.py            # DuckDB helpers, incl. point-in-time joins
│   ├── universe/
│   │   ├── sources.py          # Wikipedia + fja05680 loaders, ticker dialects
│   │   └── membership.py       # the point-in-time membership tracker
│   ├── fetchers/
│   │   ├── base.py             # fetch -> validate -> write contract
│   │   ├── yf_ohlcv.py         # daily OHLCV, batched
│   │   ├── yf_options.py       # full chain snapshots, throttled + resumable
│   │   ├── sec_edgar.py        # filing text + XBRL fundamentals
│   │   ├── fred.py             # macro series
│   │   ├── gdelt.py            # news + tone
│   │   └── finnhub.py          # company news + price cross-check
│   ├── utils/
│   │   ├── throttle.py         # rate limiter + adaptive backoff
│   │   ├── checkpoint.py       # resumable run state
│   │   └── http.py             # shared session with retry
│   └── pipeline/
│       ├── daily.py            # orchestrator + run summary + alert hook
│       └── compact.py          # weekly file consolidation
├── scripts/
│   ├── run_daily.ps1                # Task Scheduler wrapper
│   └── install_scheduled_tasks.ps1  # registers both scheduled tasks
├── tests/
└── data/                       # the lake (gitignored, except membership CSV)
```

---

## Storage layout

```
data/
├── ohlcv/year=2026/month=09/data.parquet        # + _daily_<date>.parquet deltas
├── options_chains/snapshot_date=2026-09-10/
│       ├── sym_AAPL.parquet ...                 # one per symbol during a run
│       └── data.parquet                         # after compaction
├── filings_text/year=2026/_daily_<date>.parquet
├── filings_facts/sym_AAPL.parquet               # XBRL fundamentals
├── macro_series/series_DGS10.parquet
├── news_events/date=2026-09-10/_gdelt.parquet
├── membership/sp500_membership.parquet + .csv   # CSV mirror is worth committing
├── universe_history/universe_history.parquet    # every observed daily universe
├── quality/                                     # cross-source validation results
├── _checkpoints/  _logs/  _runs/
```

### A deliberate deviation from symbol-first partitioning

The brief asked for OHLCV partitioned by `symbol/year/month`. This uses
`year/month` with `symbol` as a **sorted column** instead, for two reasons:

- **File count.** Symbol-first at month granularity is 500 symbols × 15 years ×
  12 months ≈ **90,000 files of ~21 rows each**. Parquet's per-file metadata then
  costs more than the data, and DuckDB spends its time opening files.
- **Query shape.** ML feature engineering is overwhelmingly cross-sectional
  ("every symbol on date X"), which date-first prunes perfectly. The reverse
  query stays fast because each file is sorted by symbol, so row-group min/max
  statistics let DuckDB skip nearly every row group without a directory
  partition.

Options use `snapshot_date` alone for the same reason: adding `expiration` as a
partition key would create ~8,000 files per day.

Partition values are stored **both** in the directory name and as real columns, so
every file is self-describing. Hive partitioning is therefore switched off when
reading (it would collide with the in-file column), and the typed query helpers
build explicit globs to get pruning instead. Use `q.ohlcv(start=, end=)` and
`q.options_snapshot(date)` rather than raw `SELECT * FROM ohlcv` when you have a
date filter.

---

## Measured performance

Benchmarked on this machine against the live APIs:

| Stage | Scope | Time | Output |
|---|---|---|---|
| Options snapshot | 10 symbols, every expiration | 1.9 min | 16,990 rows / 0.4 MB |
| Options snapshot | **503 symbols (projected)** | **~95 min** | ~850k rows / ~20 MB |
| OHLCV incremental | 848 tracked symbols | 7.1 min | 637 returned data, 211 empty |
| SEC EDGAR daily | 60 filings + 120 XBRL refreshes | 2.7 min | 3,039,680 rows / 15.5 MB |
| FRED | 31 series, 2010-present | 28 s | 65,325 rows / 0.5 MB |
| Finnhub | 40 news + 50 price cross-checks | 1.5 min | 251 rows, 50/50 agree |
| GDELT (while 429ing) | circuit breaker trip | 1.5 min | 0 rows, run continues |

A full daily run with all stages enabled lands at roughly **2 hours**, dominated
by the options snapshot.

At ~20 MB/day the options dataset lands around **5 GB/year** — zstd compresses
option chains extremely well. **Local disk is sufficient for years**; no cloud
storage backend is needed, and none is configured.

---

## Options collection at scale

~500 symbols × ~20 expirations ≈ **10,000 requests/day** against an unofficial,
unauthenticated API. The design assumes that will occasionally go wrong.

- **Throttling** is per-*expiration*, not just per-symbol (`throttle_seconds`,
  `inter_symbol_seconds`), with proportional **jitter** — a metronomic request
  every 600 ms is a stronger bot signal than the same average rate with noise.
- **Adaptive backoff.** On a rate-limit signal the throttle sleeps a penalty
  period and then multiplies its interval; sustained success decays it back
  toward baseline. A run that hits a wall finishes slower rather than failing.
- **Resume is dual-signal.** A symbol's output file existing means it is done —
  this survives losing the checkpoint entirely. The checkpoint separately records
  failure reasons, attempt counts and deliberate skips.
  *Verified:* deleting 3 of 10 symbol files **and** the whole checkpoint, then
  re-running, re-fetched only those 3 (30.7 s vs 112 s).
- **Failures are contained.** A delisted or optionless symbol is logged, skipped
  and retried once at the end of the run. It never kills the other 499.
- **Abort on systemic failure.** If the failure rate crosses
  `abort_failure_rate` (default 40%), the run stops. That pattern means Yahoo is
  blocking the IP, and continuing only deepens the block. Progress is
  checkpointed, so resuming later costs nothing.
- **Progress reporting** uses a `tqdm` bar interactively and falls back to
  periodic log lines with an ETA under Task Scheduler, where there is no TTY.

### On Yahoo rate limits

There is no published limit; Yahoo infers abuse from cadence and IP. yfinance
1.7.0 depends on `curl_cffi`, which impersonates browser TLS fingerprints and
resolves most of the HTTP 429 reports that plagued earlier versions. In testing,
a full 10-symbol / 200-expiration sweep at 0.6 s spacing drew **zero** rate-limit
responses. Defaults are deliberately conservative; if you do get throttled, raise
`options.throttle_seconds` rather than adding retries.

---

## Scheduling (Windows Task Scheduler)

From an **elevated** PowerShell:

```powershell
.\scripts\install_scheduled_tasks.ps1
```

Registers:

- **TickerLake-Daily** — weekdays 17:30 local. Options markets close at 16:15 ET
  and Yahoo needs time to settle end-of-day chains, so this leaves margin.
- **TickerLake-Compact** — Sundays 03:00 local.

Both wake the machine, run whether or not you are logged in, restart twice on
failure, and cap at 6 hours.

```powershell
Get-ScheduledTask -TaskName 'TickerLake-*'
Start-ScheduledTask -TaskName 'TickerLake-Daily'    # run once now
```

**Why not GitHub Actions.** Two blockers, both decisive at this volume: GitHub
runners use Azure datacenter IP ranges that Yahoo throttles far more aggressively
than residential ones, and ~95 min × 21 weekdays ≈ 2,000 minutes/month sits right
at the free-tier ceiling for a private repo (public repos are unlimited). The
6-hour per-job cap is survivable; the IP reputation problem is not. If you later
want unattended operation independent of this machine, a cheap VPS is the better
move — the code is scheduler-agnostic, and only `scripts/` would change.

---

## Querying

```powershell
tickerlake query "SELECT symbol, date, close FROM ohlcv ORDER BY date DESC LIMIT 10"
tickerlake query --file analysis.sql --output results.parquet
```

```python
from tickerlake.storage.query import LakeQuery

with LakeQuery("data") as q:
    # Options surface for one day
    chains = q.options_snapshot("2026-09-10", symbols=["AAPL"], max_dte=45)

    # The dataset that only exists because we snapshot daily
    history = q.options_history("AAPL", "2026-09-01", "2026-09-30")

    # Put/call volume ratio by symbol
    q.sql("""
        SELECT symbol,
               SUM(volume) FILTER (WHERE option_type='put')
                 / NULLIF(SUM(volume) FILTER (WHERE option_type='call'), 0) AS pc_ratio
        FROM options_chains
        WHERE snapshot_date = DATE '2026-09-10' AND dte BETWEEN 20 AND 40
        GROUP BY symbol ORDER BY pc_ratio DESC LIMIT 20
    """)

    # 30-delta-ish IV skew proxy against spot
    q.sql("""
        SELECT symbol, expiration,
               AVG(implied_volatility) FILTER (WHERE strike < underlying_price*0.95) AS otm_put_iv,
               AVG(implied_volatility) FILTER (WHERE strike > underlying_price*1.05) AS otm_call_iv
        FROM options_chains
        WHERE snapshot_date = DATE '2026-09-10' AND dte BETWEEN 25 AND 35
        GROUP BY symbol, expiration
    """)
```

Note: Yahoo's `implied_volatility` is unreliable on deep-ITM and zero-bid
contracts (values of 0.00001 and 9.45 both appear in real output). Filter on
`bid > 0 AND volume > 0` before using IV as a feature.

---

## Commands

| Command | Purpose |
|---|---|
| `init` | Seed the point-in-time membership table. Run first. |
| `run` | Full daily pipeline. `--stages`, `--limit`, `--symbols`, `--date`. |
| `options` | Options snapshot only. `--limit 10` to smoke-test. |
| `backfill ohlcv\|filings` | One-time history load. |
| `status` | Dataset sizes, membership stats, delisting flags, last run. |
| `query` | SQL against the lake. `--output` to write CSV/Parquet. |
| `compact` | Merge small files. `--force` ignores the age window. |
| `verify-membership` | Rebuild intervals from dated snapshots and diff. |
| `doctor` | Check config, credentials, and source reachability. |

---

## Operations

- **Run summaries** land in `data/_runs/run_<id>.json`, with `latest.json` always
  pointing at the most recent. A failure also writes `LAST_FAILURE.txt`.
- **Alerts**: set `TICKERLAKE_ALERT_WEBHOOK` to any URL accepting a JSON POST
  (Slack, Discord, ntfy) and failed runs post to it. No extra dependency.
- **Logs**: `data/_logs/`, rotating, 90-day retention. Console is terse; the file
  handler gets full per-symbol detail.
- **Config overrides** without editing YAML:
  `TICKERLAKE_OPTIONS__THROTTLE_SECONDS=1.5`.
- **Everything degrades gracefully.** A stage with a missing API key is skipped
  with a reason, not a crash. A stage that fails does not stop the others.

---

## Adding a data source

Subclass `BaseFetcher`, implement `collect()`, register it in
`pipeline/daily.py::FETCHERS`, add a config section. The base class handles
enablement, secrets, timing, exception containment and result reporting.

```python
class AlpacaFetcher(BaseFetcher):
    name = "alpaca"
    dataset = P.OHLCV
    requires_secret = "alpaca_api_key"

    def collect(self, run_date: date, result: FetchResult) -> None:
        df = ...                                  # fetch
        write = self.writer.write(df, self.dataset, path, mode="merge")
        result.record_write(write)                # validate + write happen inside
```

---

## Additional data sources worth adding

Ordered by value-per-effort for ML features. All free unless noted.

**High value, no key required**

1. **CBOE daily volume & put/call ratios** — free CSVs of index and equity
   put/call ratios. The single best free options-sentiment feature, and it
   predates your own snapshot history, so it gives you a usable series on day one.
2. **FINRA short sale volume** — daily short volume per symbol, free flat files.
   Short interest is a well-documented cross-sectional predictor and is not in
   any source currently wired up.
3. **Nasdaq Data Link / Quandl free tables** — notably Fed and Treasury series
   that FRED lags on.
4. **US Treasury yield curve XML** — official daily curve, no key, more timely
   than the FRED mirror.
5. **SEC Financial Statement Data Sets** — quarterly bulk ZIPs of *all* XBRL
   filings. Far cheaper than per-company `companyfacts` calls if you ever want
   full-market fundamentals rather than just the index.
6. **SEC Form 4** (insider transactions) — same EDGAR pipeline you already have;
   only the form type changes. Insider buying clusters are a genuine signal.
7. **SEC 13F holdings** — quarterly institutional positions. Heavy to parse, but
   ownership-change features are hard to get free anywhere else.

**Already wired up as of the latest run:** FRED now collects 31 series including
inflation breakevens (`T5YIE`, `T10YIE`), the 10Y real yield (`DFII10`), core PCE,
Fed balance sheet, initial claims, and two financial-conditions indices.

**High value, free key required**

8. **Alpha Vantage** — free tier includes earnings *estimates* and surprises,
   which nothing in the current stack provides. Earnings surprise is one of the
   strongest short-horizon features available.
9. **Tiingo** — free tier gives clean, adjusted EOD prices; a third opinion for
   the cross-check that already runs against Finnhub.
10. **NewsAPI / Marketaux** — better structured financial headlines than GDELT,
    with more reliable symbol tagging.

**Worth knowing about**

11. **Fama-French factor returns** (Ken French's data library) — free, canonical
    daily factor series. Essential if you ever want to check whether a signal is
    just repackaged market/size/value/momentum exposure.
12. **CFTC Commitments of Traders** — weekly positioning in index futures.
13. **Wikipedia / Google Trends pageview APIs** — retail attention proxies, free
    and surprisingly predictive at short horizons.
14. **Options-implied dividend and borrow rates** — derivable from the put-call
    parity violations already sitting in your own chain snapshots. No new source
    needed; just a feature-engineering job over data you are already collecting.

The two I would add next are **CBOE put/call ratios** (immediate history, no key,
directly complements the options snapshots) and **FINRA short volume** (free,
daily, and genuinely absent from everything else here).

---

## Known limitations

- **GDELT** returns HTTP 429 for minutes at a time regardless of User-Agent or
  pacing — it is a free public service under heavy load, and it was throttling
  throughout development. A **circuit breaker** trips after 5 consecutive
  failures and skips the rest of that run's queries: without it, 44 queries each
  burning a retry budget stalled the pipeline for ~45 minutes to collect nothing.
  Treat GDELT coverage as genuinely best-effort.
- **Yahoo implied volatility** is unreliable on illiquid contracts (see above).
- **623 of 848** tracked symbols map to a current SEC CIK; the remainder are
  delisted tickers absent from SEC's current ticker file. Their filing history is
  still reachable by CIK if you need it.
- **`BAMLH0A0HYM2` (high-yield OAS) only has ~3 years of history.** This is a
  FRED restriction on licensed ICE BofA data, not a collection bug — FRED's own
  `observation_start` for that series is 2023-09-11. `BAA10Y` (1986–) and
  `AAA10Y` (1983–) are configured alongside it as full-history credit-spread
  substitutes.
- **211 of 848** tracked symbols returned no OHLCV on the first full run. These
  are long-delisted historical members (WFM, XLNX, WCG, ...) that Yahoo no longer
  serves. They are counted, not dropped — after
  `silent_delist_threshold_runs` they get flagged and fall out of the active
  fetch list, while their stored history stays queryable forever.
- **No intraday or tick data.** yfinance offers 1-minute bars for only the
  trailing ~30 days; that belongs in a separate fetcher with its own retention
  policy rather than bolted onto the daily job.
- **`repair=True`** is on for OHLCV, which catches Yahoo's 100x price errors and
  missed splits. Repaired bars are flagged in the `repaired` column — a corrected
  bar is not the same evidence as a clean one.

---

## Testing

```powershell
.venv\Scripts\python.exe -m pytest tests/ -q
.venv\Scripts\python.exe -m ruff check .
```

43 tests, focused on the invariants that actually matter: point-in-time
membership correctness, the never-delete guarantee, re-entry handling,
implausible-diff rejection, silent-delisting flag lifecycle, writer idempotency
and de-duplication, schema coercion of bad dtypes, validation, partition pruning,
checkpoint resume, throttle backoff/recovery, expiration filters, and compaction.
