"""TickerLake command line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

from tickerlake import __version__
from tickerlake.config import Config, ConfigError, load_config
from tickerlake.logging_setup import setup_logging
from tickerlake.storage import paths as P
from tickerlake.storage.query import LakeQuery
from tickerlake.storage.writer import ParquetWriter

log = logging.getLogger("tickerlake.cli")


# ----------------------------------------------------------------- bootstrap


def _bootstrap(args: argparse.Namespace) -> Config:
    config = load_config(args.config)
    setup_logging(
        config.data_root,
        console_level="DEBUG"
        if getattr(args, "verbose", False)
        else config.get("logging.level", "INFO"),
        file_level=config.get("logging.file_level", "DEBUG"),
        retention_days=int(config.get("logging.retention_days", 90)),
    )
    P.DatasetPaths(config.data_root).ensure_layout()
    return config


def _run_date(args: argparse.Namespace) -> date:
    value = getattr(args, "date", None)
    return date.fromisoformat(value) if value else date.today()


def _symbols(args: argparse.Namespace) -> list[str] | None:
    value = getattr(args, "symbols", None)
    if not value:
        return None
    return [s.strip().upper() for s in value.split(",") if s.strip()]


def _tracker(config: Config):
    from tickerlake.pipeline.daily import _build_tracker

    paths = P.DatasetPaths(config.data_root)
    writer = ParquetWriter(
        compression=config.get("storage.compression", "zstd"),
        compression_level=config.get("storage.compression_level", 3),
    )
    return _build_tracker(config, paths, writer), paths, writer


# ------------------------------------------------------------------ commands


def cmd_init(args: argparse.Namespace) -> int:
    """Seed the point-in-time membership table. Run this first."""
    from tickerlake.universe import sources as SRC

    config = _bootstrap(args)
    tracker, _, _ = _tracker(config)

    if tracker.exists and not args.force:
        stats = tracker.stats()
        print(
            f"Membership table already exists: {stats['intervals']} intervals, "
            f"{stats['symbols']} symbols, {stats['current']} current members."
        )
        print("Re-seeding would discard locally observed history. Pass --force to override.")
        return 1

    print("Fetching historical membership intervals...")
    seed = SRC.fetch_seed_intervals(config.get("universe.membership_seed_url"))
    print("Fetching today's live constituents...")
    live = SRC.fetch_live_constituents(config.get("universe.live_url"))

    tracker.seed(seed, live=live, force=args.force)
    stats = tracker.stats()
    print(
        f"\nSeeded {stats['intervals']} intervals across {stats['symbols']} symbols "
        f"({stats['current']} current, {stats['historical']} historical)."
    )
    print(f"  Parquet: {P.DatasetPaths(config.data_root).membership_parquet()}")
    print(f"  CSV:     {P.DatasetPaths(config.data_root).membership_csv()}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run the daily pipeline."""
    from tickerlake.pipeline.daily import DailyPipeline

    config = _bootstrap(args)
    stages = [s.strip() for s in args.stages.split(",")] if args.stages else None
    pipeline = DailyPipeline(config, limit=args.limit, symbols=_symbols(args))
    summary = pipeline.run(run_date=_run_date(args), stages=stages)
    print(summary.render())
    return 0 if summary.ok else 1


def cmd_options(args: argparse.Namespace) -> int:
    """Run only the options snapshot. Use --limit to smoke-test on a subset."""
    from tickerlake.fetchers.yf_options import YFinanceOptionsFetcher

    config = _bootstrap(args)
    tracker, paths, writer = _tracker(config)
    fetcher = YFinanceOptionsFetcher(
        config, paths, writer, tracker=tracker, symbols_override=_symbols(args), limit=args.limit
    )
    result = fetcher.run(_run_date(args))
    print(result.summary_line())
    if result.details:
        print(json.dumps(result.details, indent=2, default=str))
    return 0 if result.ok else 1


def cmd_backfill(args: argparse.Namespace) -> int:
    """Backfill history for a dataset."""
    config = _bootstrap(args)
    tracker, paths, writer = _tracker(config)

    if not tracker.exists:
        print("No membership table. Run `tickerlake init` first.", file=sys.stderr)
        return 1

    if args.dataset == "ohlcv":
        from tickerlake.fetchers.yf_ohlcv import YFinanceOHLCVFetcher

        fetcher = YFinanceOHLCVFetcher(
            config, paths, writer, tracker=tracker, symbols_override=_symbols(args), backfill=True
        )
    elif args.dataset == "filings":
        from tickerlake.fetchers.sec_edgar import SECEdgarFetcher

        fetcher = SECEdgarFetcher(
            config, paths, writer, tracker=tracker, symbols_override=_symbols(args), backfill=True
        )
    else:
        print(f"Unknown backfill dataset: {args.dataset}", file=sys.stderr)
        return 1

    print(f"Backfilling {args.dataset}. This will take a while.")
    result = fetcher.run(_run_date(args))
    print(result.summary_line())
    return 0 if result.ok else 1


def cmd_status(args: argparse.Namespace) -> int:
    """Show lake contents, membership state, and the last run's outcome."""
    config = _bootstrap(args)
    paths = P.DatasetPaths(config.data_root)

    print(f"TickerLake {__version__}")
    print(f"data root: {config.data_root}")
    print(f"config:    {config.config_path}")
    print(f"secrets:   {config.secrets}")

    print("\n--- datasets ---")
    with LakeQuery(config.data_root) as q:
        stats = q.dataset_stats()
        counts = q.table_counts()
        merged = stats.merge(counts, on="dataset", how="left")
        print(merged.to_string(index=False))

        tracker, _, _ = _tracker(config)
        if tracker.exists:
            print("\n--- membership ---")
            m = tracker.stats()
            print(
                f"{m['intervals']} intervals | {m['symbols']} symbols | "
                f"{m['current']} current | {m['historical']} historical | {m['flagged']} flagged"
            )
            flagged = q.suspected_delistings()
            if not flagged.empty:
                print("\nsuspected silent delistings:")
                print(flagged.head(20).to_string(index=False))

        cov = q.options_coverage()
        if not cov.empty:
            print("\n--- options snapshots (most recent 10) ---")
            print(cov.head(10).to_string(index=False))

    latest = paths.root / P.RUNS / "latest.json"
    if latest.exists():
        data = json.loads(latest.read_text(encoding="utf-8"))
        print("\n--- last run ---")
        print(
            f"{data['run_id']} ({data['run_date']}): "
            f"{'OK' if data['ok'] else 'FAILED'} in {data['duration_minutes']} min"
        )
        for stage in data.get("stages", []):
            flag = "SKIP" if stage["skipped"] else ("OK" if stage["ok"] else "FAIL")
            print(f"  [{flag:4}] {stage['stage']:<10} {stage['rows_written']:>10,} rows")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    """Run SQL against the lake."""
    config = _bootstrap(args)
    sql = args.sql
    if args.file:
        sql = Path(args.file).read_text(encoding="utf-8")
    if not sql:
        print("Provide SQL as an argument or via --file.", file=sys.stderr)
        return 1

    with LakeQuery(config.data_root) as q:
        df = q.sql(sql)
        if args.output:
            out = Path(args.output)
            if out.suffix == ".parquet":
                df.to_parquet(out, index=False)
            else:
                df.to_csv(out, index=False)
            print(f"{len(df):,} rows -> {out}")
        else:
            import pandas as pd

            with pd.option_context("display.max_rows", args.limit, "display.width", 200):
                print(df.head(args.limit).to_string(index=False))
            print(f"\n({len(df):,} rows)")
    return 0


def cmd_bars(args: argparse.Namespace) -> int:
    """Build information-driven bars (AFML ch. 2) from stored 1-minute bars.

    Deliberately an on-demand export rather than a collected dataset: the
    threshold is a modelling choice, and baking one into storage would freeze a
    decision that belongs to whoever is building features. The 1-minute bars are
    the durable artefact; these are derived from them.
    """
    import pandas as pd

    from tickerlake.analytics.bars import build_bars, calibrate_threshold

    config = _bootstrap(args)
    symbols = _symbols(args)

    with LakeQuery(config.data_root) as q:
        query = "SELECT symbol, datetime, date, open, high, low, close, volume FROM intraday_bars WHERE interval = ?"
        params: list = [args.interval]
        if symbols:
            query += f" AND symbol IN ({','.join('?' * len(symbols))})"
            params.extend(symbols)
        minute_bars = q.sql(query, params)

    if minute_bars.empty:
        print(
            f"No {args.interval} bars stored. Run: tickerlake run --stages intraday",
            file=sys.stderr,
        )
        return 1

    frames = []
    for _symbol, group in minute_bars.groupby("symbol", sort=True):
        if args.kind != "time":
            report = calibrate_threshold(group, args.kind, args.target)
            print(report.summary())
            bars = build_bars(group, args.kind, threshold=report.threshold)
        else:
            bars = build_bars(group, "time", threshold=args.minutes)
        frames.append(bars)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        print("No bars produced.", file=sys.stderr)
        return 1

    complete = out[~out["incomplete"]]
    print(
        f"\n{len(out):,} {args.kind} bars across {out['symbol'].nunique()} symbol(s), "
        f"{out['session'].nunique()} session(s)"
    )
    if args.kind != "time" and not complete.empty:
        print(
            f"quantisation: mean overshoot {complete['overshoot_pct'].mean():.1f}%, "
            f"max {complete['overshoot_pct'].max():.1f}%, "
            f"{int((complete['minutes'] == 1).sum())} bar(s) filled by a single minute"
        )

    if args.output:
        path = Path(args.output)
        if path.suffix == ".parquet":
            out.to_parquet(path, index=False)
        else:
            out.to_csv(path, index=False)
        print(f"written to {path}")
    else:
        cols = [
            "symbol",
            "bar_start",
            "bar_end",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
            "minutes",
            "overshoot_pct",
        ]
        print(out[cols].head(args.limit).to_string(index=False))
    return 0


def cmd_l2(args: argparse.Namespace) -> int:
    """Reconstruct IEX order books for one day into 1-minute snapshots.

    On demand, never nightly: each day is ~11.5 GB of download for depth that
    covers ~2.5% of consolidated volume. The IEX archive reaches back to 2017
    and does not expire, so any past day can be rebuilt when a question needs
    it -- unlike option chains, nothing is lost by not collecting daily.
    """
    from tickerlake.fetchers.iex_deep import IEXDeepArchive, reconstruct_day

    config = _bootstrap(args)

    if args.list_dates:
        available = IEXDeepArchive().available_dates()
        print(f"{len(available)} trading days with DEEP data")
        print(f"earliest: {available[0]}    latest: {available[-1]}")
        print("most recent 10:", ", ".join(str(d) for d in available[-10:]))
        return 0

    symbols = _symbols(args)
    if not symbols:
        print(
            "Specify --symbols; reconstructing all 10,000+ IEX symbols is not the intent.",
            file=sys.stderr,
        )
        return 1
    if not args.date:
        print("Specify --date YYYY-MM-DD.", file=sys.stderr)
        return 1

    day = date.fromisoformat(args.date)
    print(f"Reconstructing {day} for {len(symbols)} symbol(s) at depth {args.depth}.")
    print("This streams ~11.5 GB and takes roughly 20-45 minutes. Ctrl-C is safe.")
    try:
        stats = reconstruct_day(
            day=day,
            symbols=symbols,
            data_root=config.data_root,
            depth=args.depth,
            market_hours_only=not args.include_extended,
            overwrite=args.overwrite,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Nothing partial was written.", file=sys.stderr)
        return 130

    print(json.dumps(stats, indent=2, default=str))
    return 0


def cmd_compact(args: argparse.Namespace) -> int:
    """Merge small daily files into consolidated partitions."""
    from tickerlake.pipeline.compact import Compactor

    config = _bootstrap(args)
    datasets = [d.strip() for d in args.datasets.split(",")] if args.datasets else None
    results = Compactor(config).run(datasets=datasets, force=args.force)
    for result in results:
        print(result.summary())
        for err in result.errors:
            print(f"  ERROR: {err}")
    return 0 if all(not r.errors for r in results) else 1


def cmd_repair_adjustments(args: argparse.Namespace) -> int:
    """Bring the stored adj_close up to date with dividends paid since."""
    from tickerlake.pipeline.adjustments import AdjustmentRepair

    config = _bootstrap(args)
    result = AdjustmentRepair(config).run(tolerance=args.tolerance, dry_run=args.dry_run)
    print(result.summary())
    for err in result.errors[:10]:
        print(f"  ERROR: {err}")
    return 0 if not result.errors else 1


def cmd_verify_membership(args: argparse.Namespace) -> int:
    """Rebuild membership from dated snapshots and diff against the stored table."""
    from tickerlake.universe import sources as SRC

    config = _bootstrap(args)
    tracker, _, _ = _tracker(config)
    if not tracker.exists:
        print("No membership table. Run `tickerlake init` first.", file=sys.stderr)
        return 1

    print("Rebuilding intervals from dated snapshots (independent of the seed)...")
    snapshots = SRC.fetch_snapshot_history(config.get("universe.membership_snapshot_url"))
    rebuilt = SRC.intervals_from_snapshots(snapshots)

    stored = tracker.load()
    history_start = config.get("universe.history_start")
    if history_start:
        cutoff = date.fromisoformat(history_start)
        rebuilt = rebuilt[rebuilt["end_date"].isna() | (rebuilt["end_date"] >= cutoff)]

    stored_current = set(tracker.current_members())
    rebuilt_current = set(rebuilt.loc[rebuilt["end_date"].isna(), "symbol"])

    only_stored = sorted(stored_current - rebuilt_current)
    only_rebuilt = sorted(rebuilt_current - stored_current)

    print(f"\nstored current members:   {len(stored_current)}")
    print(f"rebuilt current members:  {len(rebuilt_current)}")
    print(f"stored intervals:         {len(stored)}")
    print(f"rebuilt intervals:        {len(rebuilt)}")

    if only_stored:
        print(f"\nIn stored table but not in rebuilt snapshots ({len(only_stored)}):")
        print("  " + ", ".join(only_stored[:40]))
    if only_rebuilt:
        print(f"\nIn rebuilt snapshots but not in stored table ({len(only_rebuilt)}):")
        print("  " + ", ".join(only_rebuilt[:40]))
    if not only_stored and not only_rebuilt:
        print("\nCurrent membership agrees across both sources.")

    # Differences are expected and usually benign: the GitHub mirror lags
    # Wikipedia by days around an index change.
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check configuration, credentials, and source reachability."""
    config = _bootstrap(args)
    ok = True

    print("--- configuration ---")
    print(f"config file: {config.config_path}")
    print(
        f"data root:   {config.data_root} ({'exists' if config.data_root.exists() else 'MISSING'})"
    )

    print("\n--- credentials ---")
    for stage in ("fred", "finnhub", "sec_edgar"):
        enabled = config.get(f"{stage}.enabled", False)
        missing = config.missing_secret_for(stage)
        if not enabled:
            print(f"  {stage:<10} disabled in config")
        elif missing:
            print(f"  {stage:<10} ENABLED but {missing} is not set -> stage will be skipped")
        else:
            print(f"  {stage:<10} ready")

    print("\n--- source reachability ---")
    from tickerlake.utils.http import HttpClient

    checks = [
        ("Wikipedia (live universe)", config.get("universe.live_url")),
        ("GitHub (membership seed)", config.get("universe.membership_seed_url")),
        ("SEC EDGAR", "https://www.sec.gov/files/company_tickers.json"),
        (
            "GDELT",
            "https://api.gdeltproject.org/api/v2/doc/doc?query=test&mode=artlist&format=json&maxrecords=1",
        ),
    ]
    client = HttpClient(
        requests_per_second=2.0,
        user_agent=config.secrets.sec_user_agent or "TickerLake/0.1",
        max_attempts=1,
    )
    try:
        for label, url in checks:
            try:
                resp = client.get(url)
                print(f"  {label:<28} OK ({resp.status_code}, {len(resp.content):,} bytes)")
            except Exception as exc:
                ok = False
                print(f"  {label:<28} FAILED: {type(exc).__name__}: {exc}")
    finally:
        client.close()

    print("\n--- yfinance ---")
    try:
        import yfinance as yf

        t = yf.Ticker("AAPL")
        exps = t.options
        print(f"  yfinance {yf.__version__}: AAPL has {len(exps)} expirations -> OK")
    except Exception as exc:
        ok = False
        print(f"  yfinance FAILED: {type(exc).__name__}: {exc}")

    print(f"\n{'All checks passed.' if ok else 'Some checks FAILED (see above).'}")
    return 0 if ok else 1


# -------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tickerlake",
        description="Automated collection of free financial data into a Parquet/DuckDB lake.",
    )
    parser.add_argument("--version", action="version", version=f"tickerlake {__version__}")
    parser.add_argument("--config", help="path to config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging on the console")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="seed the point-in-time membership table (run this first)")
    p.add_argument("--force", action="store_true", help="re-seed over an existing table")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("run", help="run the daily pipeline")
    p.add_argument("--date", help="run date (YYYY-MM-DD), default today")
    p.add_argument("--stages", help="comma-separated subset of stages to run")
    p.add_argument("--limit", type=int, help="cap symbols in the options stage (for testing)")
    p.add_argument("--symbols", help="comma-separated symbol override (for testing)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("options", help="run only the options snapshot")
    p.add_argument("--date", help="snapshot date (YYYY-MM-DD), default today")
    p.add_argument("--limit", type=int, help="only the first N symbols")
    p.add_argument("--symbols", help="comma-separated symbols")
    p.set_defaults(func=cmd_options)

    p = sub.add_parser("backfill", help="backfill history")
    p.add_argument("dataset", choices=["ohlcv", "filings"])
    p.add_argument("--date", help="end date (YYYY-MM-DD), default today")
    p.add_argument("--symbols", help="comma-separated symbols")
    p.set_defaults(func=cmd_backfill)

    p = sub.add_parser("status", help="show lake contents and last run")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("query", help="run SQL against the lake")
    p.add_argument("sql", nargs="?", help="SQL text")
    p.add_argument("--file", help="read SQL from a file")
    p.add_argument("--output", help="write results to .csv or .parquet instead of stdout")
    p.add_argument("--limit", type=int, default=25, help="rows to print (default 25)")
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("bars", help="build volume/dollar/time bars from stored 1m bars (AFML ch.2)")
    p.add_argument("--kind", choices=["dollar", "volume", "time"], default="dollar")
    p.add_argument("--target", type=int, default=50, help="target bars per day (dollar/volume)")
    p.add_argument("--minutes", type=int, default=5, help="bar width when --kind time")
    p.add_argument("--interval", default="1m", help="source interval (default 1m)")
    p.add_argument("--symbols", help="comma-separated symbols")
    p.add_argument("--output", help="write to .csv or .parquet instead of stdout")
    p.add_argument("--limit", type=int, default=15, help="rows to print (default 15)")
    p.set_defaults(func=cmd_bars)

    p = sub.add_parser("l2", help="reconstruct IEX order books into 1-minute snapshots (on demand)")
    p.add_argument("--date", help="trading day to reconstruct (YYYY-MM-DD)")
    p.add_argument("--symbols", help="comma-separated symbols (required)")
    p.add_argument("--depth", type=int, default=5, help="book levels per side (default 5)")
    p.add_argument("--include-extended", action="store_true", help="keep pre/post-market minutes")
    p.add_argument("--overwrite", action="store_true", help="rebuild even if the day exists")
    p.add_argument("--list-dates", action="store_true", help="show archive coverage and exit")
    p.set_defaults(func=cmd_l2)

    p = sub.add_parser("compact", help="merge small daily files into consolidated partitions")
    p.add_argument("--datasets", help="comma-separated datasets, default from config")
    p.add_argument("--force", action="store_true", help="ignore the minimum-age window")
    p.set_defaults(func=cmd_compact)

    p = sub.add_parser(
        "repair-adjustments",
        help="bring stored adj_close up to date with dividends paid since it was written",
    )
    p.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.0005,
        help="relative difference below which a row is left alone (default 0.0005)",
    )
    p.set_defaults(func=cmd_repair_adjustments)

    p = sub.add_parser("verify-membership", help="cross-check membership against dated snapshots")
    p.set_defaults(func=cmd_verify_membership)

    p = sub.add_parser("doctor", help="check config, credentials, and source reachability")
    p.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is checkpointed; re-run to resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
