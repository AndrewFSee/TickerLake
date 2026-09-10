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

Both wake the machine, run whether or not you are logged in, restart twice on
failure, and cap at 6 hours.

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

**Now wired up:** FINRA short volume, Finnhub earnings (surprises, forward
calendar, analyst recommendations), and FRED's 31 series including inflation
breakevens, the 10Y real yield, core PCE, Fed balance sheet, initial claims and
two financial-conditions indices. Put/call ratios are computed from our own
chains in `options_flow`.

**CBOE put/call ratios are no longer freely available.** The legacy
`totalpc.csv` / `equitypc.csv` endpoints now return a JavaScript app, and the
daily-statistics page exposes no API path and no embedded data - it is all
behind their paid DataShop. `options_flow` computes the same statistic from our
own chains instead, per symbol rather than as one market-wide number, at the
cost of having no history before collection started.

**Alpha Vantage turned out to be unnecessary.** Finnhub's free tier - already in
use here - provides earnings surprises, a forward earnings calendar, analyst
recommendations, and insider transactions. Alpha Vantage's free tier is 25
requests/day, which cannot cover a 500-symbol universe under any rotation. Its
`HISTORICAL_OPTIONS` endpoint does return IV and Greeks, but shows the same
implausible values on illiquid contracts (4.21 where Yahoo shows 9.45), so it is
not a fix for the IV problem either.

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
