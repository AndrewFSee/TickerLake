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
    q.members_on("2015-06-30")  # who was actually in the index that day
    q.pit_ohlcv("2015-01-01", "2020-12-31")  # panel filtered per-row by membership
    q.ohlcv(symbols=["AAL"])  # raw table: includes delisted names on purpose
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

### ETFs

60 ETFs are collected alongside the index constituents, giving factor, sector,
duration and commodity exposure that individual equities cannot: broad US
equity (including `RSP` for the equal-weight vs cap-weight breadth signal),
style and factor funds, all 11 GICS sector SPDRs, high-beta industry proxies, a
fixed-income duration ladder plus credit and inflation, commodities,
international, real estate, and VIX futures. Edit `etfs.symbols` in the config
to change the list.

**They are deliberately not index members.** Two different questions live in the
membership table:

| Question | Scope |
|---|---|
| "who was in the S&P 500 on date X" | `index_name` -- the survivorship primitive |
| "whose data do we fetch" | `collect_indices` |

ETFs belong only to the second, and are registered under `index_name=ETF`.
Filing them under `SP500` would make `members_on()` return ~563 constituents on
every historical date and silently corrupt every survivorship-sensitive query --
precisely the bias this project exists to prevent. Verified: `members_on()`
still returns 497 / 505 / 503 for 2012 / 2020 / 2026 with zero ETF leakage,
while `current_members()` grows to 563.

Removing a ticker from the config stops future collection but retains its
history, exactly as a delisted constituent's history is retained.

**Cost of adding them:** +8 minutes on the options stage (74,977 extra contracts
against 388,773 for the S&P 500) and 241,638 extra OHLCV rows. All 60 backfilled
cleanly to 2010, and the inception dates check out against reality -- `XLC` from
2018-06-19 when Communication Services was created, `XLRE` from 2015-10-08 ahead
of Real Estate becoming a GICS sector, `VXX` from 2018-01-25.

The collected implied volatilities order themselves the way theory says they
should, which is useful independent evidence that both the ETF data and the IV
solver are sound:

| Lowest ATM IV | | Highest ATM IV | |
|---|---|---|---|
| SHY (1-3y Treasury) | 3.95% | VXX (VIX futures) | 60.1% |
| AGG (aggregate bond) | 4.61% | USO (crude oil) | 56.5% |
| IEF (7-10y Treasury) | 7.27% | SLV (silver) | 44.1% |

### Fundamentals and lookahead bias

`filings_facts` holds **14.7M XBRL facts across 598 symbols** (477 of 503 S&P
members) and 9,259 concepts, pulled from SEC companyfacts. Coverage of the
headline financials:

| Concept | Symbols |
|---|---|
| `Assets` | 598 |
| `NetIncomeLoss` | 589 |
| `StockholdersEquity` | 586 |
| `EarningsPerShareDiluted` | 582 |
| `OperatingIncomeLoss` | 501 |
| `Revenues` + ASC 606 tag | 466 / 428 |

`earnings` adds estimate-vs-actual surprises back to 2000, a forward calendar,
and analyst recommendations.

**The lake is survivorship-bias-free by construction; fundamentals are where
*lookahead* bias gets in instead.** Apple's FY2025 period ended 2025-09-27 but
was not public until the 10-K was filed on 2025-10-31. Anything keyed on
`end_date` therefore hands a model 34 days of the future -- and far more on
restatement, where the FY2024 figure reappears inside the FY2025 filing **398
days** after its period closed.

Use `pit_fundamentals()`, which filters on `filed_date` and never `end_date`:

```python
q.pit_fundamentals("2025-10-15", ["NetIncomeLoss"], symbols=["AAPL"])
#  -> $23.4bn, period 2025-06-28, filed 2025-08-01   (the quarterly figure)

q.pit_fundamentals("2025-11-15", ["NetIncomeLoss"], symbols=["AAPL"])
#  -> $112.0bn, period 2025-09-27, filed 2025-10-31  (annual, now filed)
```

Two further traps worth knowing:

- **`fiscal_year` is the filing's year, not the period's.** The FY2024
  comparative inside the FY2025 10-K carries `fiscal_year = 2025`. Group by
  `end_date`.
- **Revenue is split across two tags.** ASC 606 introduced
  `RevenueFromContractWithCustomerExcludingAssessedTax` in 2018 and both remain
  in active use -- as of 2026, 260 symbols report the old `Revenues` tag and 375
  the new one. Querying either alone silently loses about half the universe.
  `pit_revenue()` coalesces them.

### Earnings announcement dates

Surprise rows carry `period`, the fiscal period end, which says nothing about
when the number became public. That is the same lookahead bias, in a dataset
where it is easier to miss because the gap is weeks rather than months. Finnhub
cannot close it: on the free tier its calendar returns nothing for past ranges.

SEC 8-K **Item 2.02 (Results of Operations and Financial Condition)** can. The
submissions API exposes item codes per filing, so earnings releases are directly
identifiable, and `announcements` matches them onto stored periods. **690 of 695
surprises across all 175 symbols are now dated**, median 29 days after period
end.

Getting the match right took two corrections that are worth recording, because
both produced plausible-looking wrong answers rather than errors:

- **`period` is a *calendar* quarter end, not a fiscal one.** General Mills
  closes its quarters in August, November, February and May; Finnhub files them
  under calendar quarter ends, so three of four are announced *before* the label
  they carry. Requiring the 8-K to fall strictly after the period end skipped
  every one of them and silently took the next quarter's instead -- dating all
  four rows one quarter late, and recording a single 8-K as both Q3 and Q4.
  **19 of 175 symbols (11%) have this shape**, and 73 rows legitimately carry a
  negative lag.
- **Not every Item 2.02 filing is an earnings release.** Goldman Sachs filed one
  on 2026-01-08, a week before its actual Q4 release on 2026-01-15; Honeywell
  filed decoys on either side of two real releases. Taking the earliest filing
  in the window picks the decoy -- seven days of lookahead.

What separates a release from a decoy is cadence: a company announces at a
near-constant offset from its period label. So the offset is measured from the
company's own filing history, each period takes the filing closest to its
expected date, no filing may serve two periods, and the estimate is iterated --
because the first pass can itself claim a decoy, and correcting the pairing
corrects the estimate. A filing too far off the company's own cadence is left
unmatched, since a wrong date reintroduces the bias this exists to remove.

Validation after the fix, against filings read directly from SEC:

| | |
|---|---|
| Rows deviating >30 days from their own symbol's median lag | **0** of 690 |
| Filings claimed by two periods | **0** |
| Lag band | −43 to +57 days |
| Still undated | 5 rows, all periods from 2000 |

The undated five are the system working: Item 2.02 did not exist before the SEC
renumbered 8-K items in August 2004, and the submissions API returns only a
filer's most recent ~1000 filings. An undated surprise is visibly unusable; a
wrongly dated one is not.

```python
q.pit_earnings("2026-07-01", symbols=["HON"])
#  -> period 2026-03-31, announced 2026-04-23, EPS 4.90
#     (the June quarter had closed, but was not announced until 23 July)

q.pit_earnings("2026-08-01", symbols=["HON"])
#  -> period 2026-06-30, announced 2026-07-23, EPS 4.52
```

`pit_earnings()` filters on `announcement_date` and never `period`, and excludes
undated rows rather than assuming a date. Pass `quarters=N` for a surprise
history instead of a single snapshot.

### Adjusted prices decay, so they are re-derived

`adj_close` is a running total over the *future*: the value for a 2015 session
depends on every dividend paid since. The fetcher writes a row when its session
is current -- when the adjustment is 1.0 by definition -- and the incremental
lookback only revisits the last few sessions, so the row then **freezes**. Every
later dividend should reach back and lower it, and none of them do.

The cross-source check is what caught it: three Tiingo comparisons failed on CCI
with an identical 1.43% gap. Crown Castle's rows from a *year* earlier were
understated by the same 1.45% -- one missed dividend, propagated across the
symbol's entire history. At the point it was found, **300k rows across 75
symbols** were stale, and it compounds every ex-dividend date.

Two things make it fixable. The events themselves do not decay -- a dividend is
a fact about one day, not a running total -- and 32k dividend events and 342
splits are stored per bar. And `close` is already split-adjusted, because the
fetcher uses `auto_adjust=False`, which back-adjusts Close for splits and puts
only dividends into Adj Close.

That second point is a trap. Re-applying the split ratio double-counts it: 3M's
2024 Solventum spin-off is recorded as `stock_splits = 1.196`, and applying it
moved the series by **17%**. But rebuilding purely from our own dividends is
wrong too -- Danaher's 2016 Fortive spin-off is booked as a $24.56 dividend
*and* a 1.319 split, and Yahoo's factor across it matches neither reading of
that pair, so a full rebuild moved Danaher's pre-2016 history by **13.4%**.

So neither source is taken on faith. The adjustment ratio can only fall as you
go back in time -- each earlier session carries strictly more future dividends --
so the defensible value for a row is the **most adjusted one any later row
implies**:

```
correct(t) = min over u >= t of [ stored(u) x product of dividend factors over (t, u] ]
```

Where the vendor is stale, a later row plus the dividends between them wins.
Where our actions are incomplete, the vendor's own deeper adjustment wins. One
rule corrects both failures, it is monotone by construction, and running it
twice changes nothing.

Measured against Tiingo after the repair:

| symbol | stored error before | after |
|---|---|---|
| CCI | 1.3130% | **0.1703%** |
| KO | 0.6142% | **0.0173%** |
| NVDA | 0.1121% | **0.0004%** |
| MMM | 0.9304% | 0.9304% (spin-off convention, preserved) |

Provably-stale rows fell from **9,725 across 149 symbols to 477 across 74**, and
every one that remains is below the 0.05% write tolerance. `repair-adjustments`
runs nightly as a second action on the daily task, so the column is never more
than one session out of date.

**`adj_close` is also lookahead-contaminated by construction** -- its value for
2025-09-15 encodes dividends that had not happened yet on that date. For
point-in-time work use `adjusted_ohlcv(as_of=...)`, which builds the factor from
dividends known by then and nothing later, the same discipline as
`pit_fundamentals` and `pit_earnings`:

```python
q.adjusted_ohlcv(symbols=["CCI"], start="2025-09-15", end="2025-09-15", as_of="2025-10-01")
#  -> adj_close 93.57  (no later dividend was knowable yet)

q.adjusted_ohlcv(symbols=["CCI"], start="2025-09-15", end="2025-09-15")
#  -> adj_close 88.99  (every dividend since)
```

`adjustment_drift()` reports which symbols have decayed, if you want to watch it.

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
| Options analytics | 14,201 contracts -> IV + Greeks | 2.6 s | 5,441 contracts/sec |
| FINRA short volume | 2 days, 632 symbols each | 2.4 s | 1,263 rows |
| Earnings (Finnhub) | calendar + 60 symbols | 2.1 min | 641 rows |
| GDELT (bulk feed) | 16 GKG files | 26 s | 3,947 articles, 0 failures |
| Intraday 1m bars | 6 symbols x 3 sessions | 2.0 s | 7,020 bars (390/symbol/session) |

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

Both wake the machine, restart twice on failure, and cap at 6 hours.

They are registered with an **S4U principal**, which is what makes them run
whether or not anyone is logged on. The default principal
(`LogonType Interactive`) runs a task *only* while the user is signed in and
silently skips it from the lock screen afterwards -- a gap you would not notice
until the missed options chains were permanently gone. The installer prints the
logon type it ended up with; if it says anything other than `S4U`, the tasks
only fire while you are logged on.

**Non-trading days are guarded.** The trigger is weekdays-only, so weekends
never fire -- but market holidays fall on weekdays, and the sources do not all
fail cleanly on them. Measured on a closed day:

| Source | Behaviour when the market is closed |
|---|---|
| OHLCV, intraday | Return nothing for the closed date. Safe. |
| SEC, FINRA | No file published, HTTP 403/404. Handled. |
| FRED, GDELT, earnings, Finnhub | Idempotent on their natural keys. Safe. |
| **Yahoo option chains** | **Serve a full chain carrying the previous session's quotes.** |

That last one is the trap. Against the stored prior-session snapshot, a
closed-day chain came back with bid, ask and volume **100% identical across all
2,412 matched contracts**. Those rows are not duplicates by key -- a different
`snapshot_date` makes them distinct -- so nothing would reject them. They would
sit in the dataset as a plausible session whose every quote happens to match the
day before, which reads as a real zero-change day rather than a non-event.

`options`, `intraday` and `options_analytics` therefore carry
`requires_trading_day` and skip with a stated reason. A Saturday run finishes in
~9 minutes instead of 88, collecting only what is genuinely new:

```
  ohlcv:             OK | 5,576 rows | 697 ok / 211 skipped
  intraday:          SKIPPED (2026-09-12 is not a trading day (Saturday))
  options:           SKIPPED (2026-09-12 is not a trading day (Saturday))
  options_analytics: SKIPPED (2026-09-12 is not a trading day (Saturday))
  sec_edgar:         OK | 2,682,272 rows
  fred:              OK | 65,356 rows
```

The calendar is self-contained (no new dependency) and computes NYSE closures
including Good Friday, which the federal calendar omits, while excluding
Columbus Day and Veterans Day, which it wrongly includes. Validated against
**2,940 real SPY sessions (2015-2026) at 99.932% agreement** -- the only two
disagreements being the ad-hoc closures for the George H.W. Bush and Jimmy
Carter national days of mourning, which no rule set predicts. Set
`options.run_on_closed_days: true` to override.

**The 17:30 slot is verified, not assumed.** Yahoo populates option bid/ask only
while quotes are live, and it retains them after the close. Measured on the same
symbols in one day:

| Snapshot time | Bid coverage |
|---|---|
| 09:14 ET (pre-open) | 8.6% - unusable, no solvable IV |
| 11:50 ET (mid-session) | 87.5% |
| 16:06 ET (post-close) | 65.6% - comfortably usable |

So an after-close run is fine. A pre-open run is not, and the options fetcher
warns when coverage drops below `options.min_bid_coverage`.

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
        df = ...  # fetch
        write = self.writer.write(df, self.dataset, path, mode="merge")
        result.record_write(write)  # validate + write happen inside
```

### What `mode="merge"` does to columns you omit

Several stages write to the same dataset, each knowing only about its own
columns, so the writer distinguishes **absent** from **null**:

| Incoming frame | Result |
|---|---|
| Column missing entirely | Stored value is **kept** |
| Column present, value null | Stored value is **overwritten** with null |
| Column present, has a value | Stored value is overwritten |

A missing column is an absence of information; a null is a statement that the
value is unknown. Only the second can clear a stored value.

This is not academic. `earnings` and `announcements` both write surprise rows
keyed on `(symbol, record_type, period)`, and `earnings.py` has never heard of
`announcement_date`. Without the distinction, `pandas.concat` unions the
columns, the incoming row arrives carrying a NaN for a field it does not know
about, and `keep="last"` hands it the win -- **replaying one night's earnings
run against the real file destroys 237 announcement dates.** Only stage
ordering hid it: `announcements` runs later in the same pipeline and repaired
the damage each night, so every integrity check passed and the enrichment would
have silently decayed the moment that stage failed or moved.

Matching is per natural key, so a row whose key is new to the file gets null --
there is nothing stored to preserve. Datasets that declare no `unique_on`
(`filings_facts`, `quality`, `universe_history`) have nothing to match on and
are passed through untouched; they are written with `mode="overwrite"` anyway.

---

## Implied volatility and Greeks

Yahoo's `impliedVolatility` field is unusable on illiquid contracts: a single
snapshot contains 0.00001 and 20.8 side by side. That is not a Yahoo defect -
Alpha Vantage reports 4.21 on comparable deep-ITM contracts. When a contract has
a zero bid or trades below intrinsic value, there is no well-defined implied
volatility, and every vendor emits a number anyway.

So `options_analytics` solves its own, entirely from data already in the lake:
spot from the snapshot itself, the risk-free rate interpolated from the FRED
curve at each contract's own maturity, and dividend yield from stored OHLCV.
Black-Scholes-Merton, solved with Brent's method rather than Newton-Raphson -
vega collapses toward zero on the wings and Newton divides by it.

The point is not a better number; it is **a verdict per contract**. Every row
carries `quality_flags` saying why a value is or is not trustworthy:

| Flag | Meaning |
|---|---|
| `zero_bid` | No bid: the mid is fictional |
| `crossed_market` | bid >= ask, stale or erroneous quote |
| `wide_spread` | Spread wider than 50% of mid |
| `below_intrinsic` | No time value left to solve against |
| `unidentifiable` | Vega ~ 0: any sigma reproduces the quote |
| `implausible_iv` | Solved above 300%, effectively always an American early-exercise artefact |
| `deep_itm` | Early-exercise premium likely material |

Measured against the real chains:

- **Correlation with Yahoo's IV on healthy contracts: 0.952**, median absolute
  difference 3.8 vol points - the implementation agrees where agreement is
  meaningful.
- On contracts we gate, Yahoo's field spans **0.0 to 20.8**, with 818 values
  under 1% vol and 167 over 300%. None survive the gate.
- Greeks match finite differences to 1e-9; put-call parity holds to 1e-15.

Two things that only became visible by computing it ourselves:

**Snapshot timing decides everything.** An early run showed a 4.1% solve rate.
The cause was not the model: that snapshot was taken at 09:14 ET, before the
09:30 open, and Yahoo serves the full strike ladder around the clock while
populating bid/ask only during market hours. Bid coverage was 8.6%. Re-run
mid-session, the same 8 symbols gave **87.5% bid coverage and a 64.4% solve
rate**. The options fetcher now measures bid coverage on every run and warns
below `min_bid_coverage`, because an off-hours snapshot looks complete - right
row count, right strikes, plausible last prices - while being useless for
volatility.

**Delta is not a moneyness proxy.** The first flow-ratio implementation picked
ATM contracts by `|delta|` in [0.45, 0.55]. But delta is computed *from* the
solved IV, so a deep-ITM put sitting at 470% IV has its delta dragged toward 0.5
and passes as at-the-money - which pushed AAPL's reported 8-day ATM IV to 140%
against ~27% at every neighbouring tenor. ATM and the skew wings now use
log-moneyness, which depends only on spot and strike. The resulting term
structure is textbook: 55% at 1DTE decaying to a flat 25.4%.

```sql
-- Only contracts whose quote can actually support a volatility
SELECT symbol, expiration, strike, iv, delta, vega, iv_uncertainty
FROM options_greeks
WHERE iv_usable AND ABS(log_moneyness) < 0.05;
```

`options_flow` holds the per-symbol daily summary: put/call volume and open
interest ratios, 30-day ATM IV, and 25-delta skew.

---

## Tick data: what is actually available

Short answer: **consolidated (SIP) tick data is not free from anywhere.** Every
"free tick data" option is either a single-exchange sample, a bar series, or a
throttled snapshot stream. Tested directly rather than taken from documentation:

| Source | Trade-level? | Verdict |
|---|---|---|
| Finnhub `/stock/tick` | No | HTTP 403 on the free tier |
| Finnhub `/stock/bidask`, `/stock/candle` | No | HTTP 403 on the free tier |
| Finnhub WebSocket | No | Connects and pings, delivers zero trades |
| Yahoo WebSocket (`yf.WebSocket`) | No | Price only. No size, volume, or bid/ask; ~0.25 Hz per symbol |
| yfinance 1m bars | No | Bars, not ticks - but the finest free granularity |
| Alpaca free (Basic) | **Yes, IEX only** | Needs a free key. IEX is ~2.5% of US volume |
| Polygon/Massive free | Yes, but capped | 1,000 req/day, 500 symbols/month, 1 GB - cannot cover 500 symbols daily |
| IEX Cloud | n/a | Shut down August 2024 |

The Yahoo WebSocket is worth calling out because it looks like a tick feed and
is not: over 25 seconds it delivered 6 messages for AAPL, each carrying `price`,
`time` and `exchange` but no size and no volume. Real AAPL tape is hundreds of
trades per second.

### What we collect instead

`intraday_bars` captures 1-minute OHLCV daily, for the same reason we snapshot
option chains: **Yahoo's intraday history is a rolling window that expires.**
Measured per-request limits:

| Interval | Max span per request | Retention |
|---|---|---|
| 1m | 8 days | ~30 days |
| 2m / 5m / 15m / 30m | 60 days | 60 days |
| 1h | 730 days | 2 years |

A 1-minute day that is not captured is unrecoverable. At 390 bars per symbol per
session that is ~196k rows/day for the full universe, about 1.9 MB/day
(~480 MB/year) and ~3 minutes of runtime.

These are bars: no trade-level detail, no bid/ask, no trade conditions, no size
beyond per-bar volume. Anything needing genuine order-flow microstructure needs a
real tape.

**Decision: not collecting ticks, for now.** Alpaca's free IEX feed was
considered and declined, on the grounds that 1-minute bars carry most of the
signal at daily-to-hourly horizons and a 2.5% single-exchange sample is
misleading for volume-weighted features because it is not a random 2.5%.

That reasoning holds for volume and dollar bars, which the measurements above
confirm work well from minute data. It does **not** hold for AFML's imbalance
bars, run bars, or ch. 19 microstructural features: those need a signed trade
sequence, and no amount of bar aggregation recovers one. If that part of the
book becomes the goal, tick data stops being optional and the honest options are
Alpaca IEX with the sampling bias understood, or a paid SIP feed.

---

### Information-driven bars (AFML ch. 2)

`tickerlake bars` builds volume, dollar, and time bars from the stored 1-minute
series. What is reconstructable depends entirely on what the source carries:

| AFML bar type | Requires | From 1m bars? |
|---|---|---|
| Time bars | clock | Yes |
| **Volume bars** | cumulative shares | **Approximate** |
| **Dollar bars** | cumulative traded value | **Approximate** |
| Tick bars | trade count per period | No - yfinance publishes no trade count |
| Imbalance bars (TIB/VIB/DIB) | signed trade *sequence* | No |
| Run bars (TRB/VRB/DRB) | signed trade *sequence* | No |

Imbalance and run bars rest on the tick rule -- `b_t = b_{t-1} if dp=0 else
sign(dp)` applied **per trade**. A minute bar exposes only the net change across
~60 seconds, so the sequence inside it is unrecoverable. The same blocker rules
out ch. 19's microstructural features (Kyle's lambda, VPIN, Roll measure).
`signed_volume_proxy()` computes a bar-level approximation under a name that
does not pretend to be VIB/DIB.

#### Which bar type should you actually use?

Information-driven bars are widely assumed to be an HFT technique. They are not.
The problem they solve is **statistical, not latency-related**: markets do not
process information at a constant rate, so a time bar spanning a quiet August
Tuesday and one spanning an earnings release are treated as equivalent
observations by any model, despite containing wildly different amounts of
information. That distortion exists at every horizon.

Ranked for non-HFT use:

1. **Dollar bars - the default.** Invariant to price level and to splits. Across
   just ten large caps there have been 15 splits since 2010, including CMG 50:1,
   AMZN 20:1 and NVDA 10:1. A split multiplies share volume overnight without
   changing traded value, so a fixed *volume* threshold would suddenly sample
   50x more often for CMG in June 2024. Dollar thresholds are untouched. Over a
   2010-2026 sample this is decisive.
2. **Volume bars - situational.** Reasonable when share flow is what you
   actually model (market-impact work), but the split and price-drift problems
   above make them a poor default for long samples.
3. **Time bars - still the right choice sometimes.** Necessary whenever
   observations must align to a calendar, which in this lake is most of the
   time: daily OHLCV, macro series, earnings dates and option snapshots are all
   calendar-indexed.
4. **Tick bars - weakest outside HFT.** Trade count is badly non-stationary over
   long samples because algorithmic order-slicing has steadily shrunk average
   trade size; one parent order becomes hundreds of child prints. The unit of
   measurement drifts underneath you.
5. **Imbalance and run bars - genuinely short-horizon.** They detect order-flow
   imbalance, which mean-reverts on timescales far shorter than a daily
   rebalance. This is the part of ch. 2 most tied to execution and
   short-horizon alpha.

Which is a convenient result: the bars that matter most away from HFT (dollar)
are exactly the ones reconstructable from minute data, and the ones that need
tick data (imbalance, run) are the ones most specific to HFT.

**The honest caveat.** Dollar bars break calendar alignment. Every join in this
lake - macro series, earnings, filings, option snapshots - is date-indexed, so
using them means as-of joins against everything else. If the model is
fundamentally daily-horizon, that friction can outweigh the statistical gain.
Dollar bars earn their keep for intraday-to-multiday horizons.

**Where the returns actually are.** For most non-HFT work, AFML chapters 3-7
matter more than chapter 2, and all of them operate on whatever bars you choose:
triple-barrier labelling and meta-labelling (ch. 3), concurrency-adjusted sample
weights for overlapping labels (ch. 4), fractional differentiation (ch. 5), and
purged K-fold CV with embargo (ch. 7). Getting labelling and cross-validation
right beats perfecting the bar definition.

**The approximation still delivers the benefit.** AFML's central claim is that
information-driven bars have better statistical properties than time bars.
Measured on real AAPL minute data, matched bar counts:

| Bar type | n | Serial corr | Excess kurtosis | Jarque-Bera | Bar-size CV |
|---|---|---|---|---|---|
| time | 132 | -0.1120 | 16.05 | 1534.2 | 0.811 |
| volume | 134 | -0.0206 | 3.16 | 55.5 | 0.179 |
| **dollar** | 134 | **-0.0215** | **2.80** | **43.6** | **0.174** |

Dollar bars cut serial correlation ~5x, excess kurtosis ~5.7x, and the
Jarque-Bera statistic ~35x. Lower is better on all four.

**The cost is quantisation at the open.** A bar can only close on a minute
boundary, and intraday volume is ~19x heavier at the open than at midday, so a
threshold sized for 50 bars/day is exceeded by the opening minute alone:

| Target bars/day | AAPL threshold | Typical overshoot | At the open |
|---|---|---|---|
| 20 | $764M | 2.1% | 40% |
| 50 | $305M | 5.2% | 101% |
| 100 | $153M | 10.4% | 201% |

In practice at 30 bars/day: mean overshoot 9.4%, and 5 of 134 bars were filled
by a single minute -- all at the open. `calibrate_threshold()` warns when a
threshold is smaller than one opening minute, and every bar carries
`overshoot_pct` so the error is inspectable per row rather than assumed away.
**Prefer 20-30 bars/day over AFML's suggested ~50** when working from minute
data; the quantisation is what forces that, not the theory.

```powershell
tickerlake bars --kind dollar --target 30 --symbols AAPL,NVDA
tickerlake bars --kind dollar --target 30 --output bars.parquet
```

Bars are an on-demand export, not a collected dataset: the threshold is a
modelling choice, and freezing one into storage would fix a decision that
belongs to whoever builds the features. The 1-minute bars are the durable
artefact.

---

### Level 2 (order book depth)

Free L2 for US equities exists, is current, and is almost certainly impractical
to collect here. Measured directly:

**IEX DEEP** -- the IEX exchange publishes its full depth-of-book feed as free
historical pcap. `https://iextrading.com/api/1.0/hist` lists every daily file.

| | |
|---|---|
| Coverage | 2,357 trading days, 2017-05-15 to present (current through yesterday) |
| Feeds | DEEP (L2 depth), DPLC/DPLS (deep + auction), TOPS (L1) |
| Recent size | **12.5 GB/day compressed** |
| Full archive | **9.34 TB compressed** |
| One year forward | **~3.16 TB/year** |
| Format | gzipped pcap of the IEX-TP multicast feed |
| Access | HTTP 206 range requests supported |

Three things make it impractical as a daily collection stage:

1. **Volume.** 3.16 TB/year against a lake that is presently ~5 GB/year for
   options. Even streaming and discarding, that is 12.5 GB/day of download,
   roughly 375 GB/month.
2. **You cannot cheaply extract one symbol.** Range requests return 206, but the
   file is gzip, and gzip is not seekable -- reaching byte N means decompressing
   everything before it. Filtering to the S&P 500 still means streaming the
   whole 12.5 GB through a parser each day.
3. **It is IEX's book, not the market's.** IEX is ~2.5% of consolidated volume,
   so displayed depth is thin and is not the NBBO. Book-imbalance or depth
   features built from it describe one venue, not the market.

Plus the binary IEX-TP/DEEP protocol needs a parser; the available libraries are
lightly maintained.

#### Consolidating L2 to 1-minute snapshots

The 3.16 TB/year figure is the cost of storing the *raw message stream*. It is
not the cost of storing what you would actually use. Downsampling to 1-minute
book snapshots -- matching the `intraday_bars` grid -- changes the storage
arithmetic by roughly 300x, though it changes bandwidth not at all.

Feasibility was verified end to end against a real DEEP file, not assumed:

| Step | Measured |
|---|---|
| Format | **pcapng**, not classic pcap (magic `0x0a0d0d0a`) |
| Download | 11.5 GB compressed, **~20 min** at 9.5 MB/s |
| Decompressed | ~48 GB streamed (4.2x ratio) |
| Parse | **~23 min** at 0.29M msg/s in pure Python (overlappable with download) |
| Volume | ~389M messages/day, ~372M price-level updates |
| Symbols | 10,967 distinct |

Book reconstruction from `Price Level Update` messages (`0x38` buy / `0x35`
sell) was confirmed working -- decoded books for SPY, AAPL and MSFT came out
with sane bids, asks and sizes.

**Storage after consolidation:** 503 symbols x 390 minutes = ~196k snapshots/day.
With top-10 levels per side plus derived features that is roughly **20-40 MB/day
(~5-10 GB/year)**, against 3.16 TB/year raw.

**What does not improve: bandwidth.** Consolidation happens *after* the download,
so the full 11.5 GB still crosses the wire every day -- about **345 GB/month**.
That, plus ~45 minutes of nightly runtime, is the real price.

**And the representativeness caveat is unchanged.** IEX is ~2.5% of consolidated
volume. A reconstructed IEX book is one venue's resting liquidity, not the NBBO,
and depth or imbalance features built from it should be read that way.

#### `tickerlake l2` -- the selective reconstructor

Built as an **on-demand tool, deliberately not a pipeline stage**. Name a date
and some symbols; it streams that day's DEEP file, rebuilds the books, writes
1-minute snapshots, and discards the rest.

```powershell
tickerlake l2 --list-dates
tickerlake l2 --date 2026-09-09 --symbols AAPL,MSFT,NVDA,SPY --depth 5
```

There is no urgency to run it daily. Unlike option chains -- which Yahoo serves
live-only, so an uncaptured day is gone -- **the IEX archive reaches back to
2017-05-15 and does not expire**. Any past day can be rebuilt whenever a
question actually needs it, which is what makes on-demand the right shape.

Each snapshot carries the book state at the minute boundary *and* time-weighted
averages across the whole minute. That distinction matters: a snapshot taken at
exactly 09:31:00.000 is one instant out of ~60 seconds of quoting and can land
on a momentary wide spread that never represented anything. Each book state is
weighted by how long it actually stood.

Columns include `best_bid`/`best_ask`, `spread`, `mid`, `microprice` (size-
weighted mid, which leans toward the thinner side and predicts the next trade
better than the mid), `imbalance_l1`, `bid_depth`/`ask_depth`, the time-weighted
`twa_spread`/`twa_mid`/`twa_imbalance`, `quoted_seconds`, `n_updates`,
`n_trades`, `trade_volume`, `trade_notional`, and `bid_px_1..N`/`bid_sz_1..N`
plus the ask side. `n_updates` is a genuine microstructure activity measure
available nowhere else in the lake.

Snapshots land on the same 1-minute grid as `intraday_bars`, so they join
directly on `(symbol, datetime)`.

#### Verified on a real day

One full reconstruction of 2026-09-09 for AAPL/MSFT/NVDA/SPY:

| | |
|---|---|
| Runtime | **23.4 min** end to end |
| Processed | 395M messages, 11.5 GB streamed |
| Output | **1,560 snapshots = 390 minutes x 4 symbols**, full session |
| Size | **176 KB** -> ~22 MB/day extrapolated to all 503 members |

**The time-weighting decision paid off, measurably.** Comparing each minute's
boundary snapshot against its time-weighted average:

| Symbol | Mean abs. gap | Max gap | Minutes where the point sample is off by >50% |
|---|---|---|---|
| AAPL | $0.197 | $5.17 | **69.0%** |
| MSFT | $0.513 | $25.57 | **50.3%** |
| SPY | $0.030 | $0.33 | 34.9% |
| NVDA | $0.023 | $2.51 | 14.9% |

For AAPL, a naive boundary snapshot misrepresents the minute's spread by more
than half **69% of the time**. The opening minute is the clearest case: the
09:30 boundary showed a $7.97 spread while the time-weighted average across
that minute was $2.80.

**The 2.5% caveat, measured rather than quoted.** Joining IEX trade volume
against the consolidated volume already in `ohlcv`:

| Symbol | IEX shares | Consolidated | IEX share of tape |
|---|---|---|---|
| AAPL | 2,458,440 | 65,371,600 | **3.76%** |
| NVDA | 2,176,742 | 82,737,000 | **2.63%** |
| MSFT | 327,680 | 12,875,900 | **2.54%** |

Coverage is also very uneven per symbol: NVDA generated 2.86M book updates that
day against MSFT's 108K, a 26x difference. Depth features should be read as
describing IEX's book, and their reliability varies by name.

**Practical verdict:** selective pulls are clearly worthwhile and are what is
built. Nightly collection is technically viable but was declined -- 345 GB/month
and 45 min/night is a steep recurring price for depth describing 2.5% of the
market, and the archive's permanence means nothing is lost by waiting.

**Crypto L2 is free and complete.** Verified working: Coinbase
(`api.exchange.coinbase.com/products/{pair}/book?level=2`) returns a full 1.1 MB
book, and Kraken's `Depth` endpoint works. Binance returns HTTP 451 from US IPs.
If crypto ever enters scope, full-depth L2 is free, small, and needs no key --
the opposite of the equities situation.

**Paid consolidated L2:** Databento is the usual route (MBP-10 is their L2
schema, Standard plan around $199/month). That buys the consolidated book rather
than a single venue's.

---

## Data sources

### Collected

| Source | What it gives | Key |
|---|---|---|
| **yfinance** | OHLCV (2010-), 1m intraday bars, full option chains | none |
| **SEC EDGAR** | 10-K/10-Q/8-K text, XBRL company facts | contact UA |
| **FRED** | 31 macro series incl. breakevens, real yields, core PCE, NFCI | free |
| **Finnhub** | Company news, price cross-check, earnings, insider Form 4 | free |
| **FINRA** | Daily short sale volume | none |
| **GDELT** | News tone and themes, via the bulk GKG feed | none |
| **Ken French** | Fama-French daily factors (FF5 + momentum, 1990-) | none |
| **US Treasury** | Official daily par yield curve, all 14 tenors | none |
| **CFTC** | Commitments of Traders, weekly positioning by category | none |
| **Tiingo** | Independent check on *adjusted* prices and corporate actions | free |
| **Marketaux** | Symbol-tagged news with per-entity sentiment | free |
| **IEX DEEP** | L2 order book, on demand (`tickerlake l2`) | none |
| *derived* | IV + Greeks, put/call ratios, ATM IV, skew | - |

**Fama-French** is the reference against which a signal gets judged: most
apparent alpha is market, size, value, profitability, investment or momentum
exposure under another name. Collected values reproduce the literature -- Mkt-RF
at 9.4% annualised on 18.1% vol, RF at 0.1% vol, SMB at 0.65% (the size premium
really has been absent since 1990), momentum strongest at 6.2%.

**The Treasury curve** supplies all 14 official tenors daily, so slope and
curvature are exact rather than approximated from FRED's three points.

**CFTC COT** is the only source here describing *who* is positioned: asset
managers, leveraged money and dealers, weekly, for E-mini S&P 500 and the sector
index futures.

**Tiingo** checks *adjusted* prices rather than raw ones. Raw closes are the
easy case - every vendor sees the same print, and Finnhub already agrees with
yfinance to the cent. Adjustment is a *computation* over dividend and split
history, so a vendor that misses a corporate action produces a series that looks
entirely reasonable while being wrong for every prior date. With 15 splits among
large caps since 2010, a missed one is a 4x or 10x step in a return series that
nothing else here would catch. Tiingo publishes `adjOpen/High/Low/Close` plus
per-bar `divCash` and `splitFactor`, where yfinance gives only an adjusted
close. Current result: **125/125 closes and 125/125 adjusted closes agree**.

**Marketaux** supplies the two things GDELT structurally cannot: per-entity
`sentiment_score`, so an article can score differently for each company it
mentions, and `match_score`, the publisher's own tagging confidence - the exact
quantity the GDELT salience-offset heuristic approximates. The precision
difference is measurable: of 6,646 GDELT articles, 5,559 are theme-only with no
symbol; of 48 Marketaux articles, **0 are untagged**. It does not replace GDELT
- the free tier is ~100 requests/day at 3 articles each - so GDELT provides
breadth and Marketaux provides precision.

One caveat on combining them: `sentiment` is normalised to -1..+1 for both, but
the empirical distributions differ sharply. GDELT tone rarely leaves -10..+10,
so normalised it clusters within +/-0.1, while Marketaux routinely reaches
+/-0.8. **Standardise per source before pooling**, or Marketaux will dominate
any model using both.

**Insider transactions** carry a trap worth knowing about. Finnhub's `share`
field is the insider's holding *after* the trade, not the trade size; `change`
is the transaction. Valuing the former at the trade price reported $957bn of
buying in RSG, because Cascade Investment holds ~114M shares. The columns are
named `shares_held_after` and `shares_transacted`, with `transaction_value`
precomputed from the right one.

### Ruled out, with reasons

- **CBOE put/call ratios** -- no longer free. The legacy `totalpc.csv` endpoints
  return a JavaScript app and the daily-statistics page exposes no API; it is
  behind their paid DataShop. `options_flow` computes the same statistic from
  our own chains, per symbol rather than one market-wide number, at the cost of
  no history before collection started.
- **Alpha Vantage** -- unnecessary. Finnhub's free tier already provides
  earnings surprises, the forward calendar, recommendations and insider
  transactions. Alpha Vantage's free tier is 25 requests/day, which cannot cover
  500 symbols under any rotation, and its options IV shows the same implausible
  values on illiquid contracts.
- **Consolidated tick / L2** -- not free anywhere. See "Tick data" and "Level 2"
  above. IEX DEEP is free and real but covers ~2.5% of volume.

### Still available, not built

**Free, no key:**

- **SEC Financial Statement Data Sets** -- quarterly bulk XBRL ZIPs (~60 MB
  each). Only worth it for full-market fundamentals; `filings_facts` already
  covers the tracked universe.
- **SEC 13F** -- quarterly institutional holdings. Ownership-change features are
  hard to get free, but the parse is heavy and the frequency low.
- **Wikipedia / Google Trends pageviews** -- retail attention proxies, verified
  reachable, surprisingly predictive at short horizons.

**Free key required:**

- **NewsAPI** -- another tagged-news option, though Marketaux now covers this.

**Derivable from data already collected:**

- **Options-implied dividend and borrow rates** -- from put-call parity
  violations in the stored chains. No new source, just a feature job.

## Known limitations

- **GDELT** returns HTTP 429 for minutes at a time regardless of User-Agent or
  pacing — it is a free public service under heavy load, and it was throttling
  throughout development. A **circuit breaker** trips after 5 consecutive
  failures and skips the rest of that run's queries: without it, 44 queries each
  burning a retry budget stalled the pipeline for ~45 minutes to collect nothing.
  Treat GDELT coverage as genuinely best-effort.
- **Yahoo implied volatility is not used.** See "Implied volatility and Greeks"
  above - we solve our own and gate it per contract.
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
- **No true tick data**, and none is available free - see "Tick data" above.
  `intraday_bars` collects 1-minute bars daily because Yahoo's ~30-day window
  expires, but bars are not ticks.
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
