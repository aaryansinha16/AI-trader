"""
Daily EOD incremental collector for BANKNIFTY / FINNIFTY (Plan D).

Run this once after market close every weekday. It:

  1. Looks at the last `--lookback-days` weekdays
  2. For each symbol's spot + futures, finds days where the candle count
     is below the "complete day" threshold and refetches them
  3. For each historical day, computes that day's ATM from the futures
     close, builds the ATM±N option list, and fills any missing 1-min
     candles AND tick data for those options
  4. Optionally also runs a sanity backfill on NIFTY-I if it has gaps in
     the last 5 days (the live collector usually covers this, but a
     missed-startup day will leave a hole that this catches)

Idempotent: only fetches what's missing. Safe to re-run multiple times.

Usage:
  python scripts/eod_collect_other_symbols.py             # default lookback
  python scripts/eod_collect_other_symbols.py --lookback-days 7
  python scripts/eod_collect_other_symbols.py --dry-run   # report only
  python scripts/eod_collect_other_symbols.py --include-today
  python scripts/eod_collect_other_symbols.py --also-nifty
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import argparse
import time
from datetime import date, datetime, timedelta

from data.truedata_adapter import TrueDataAdapter
from config.settings import (
    TD_INDEX_SPOT_SYMBOLS,
    TD_INDEX_FUTURES_SYMBOLS,
    STRIKE_GAP,
    EOD_ONLY_SYMBOLS,
    ATM_RANGE,
)
from utils.logger import get_logger

from scripts._market_data_lib import (
    trading_days_back,
    candle_coverage,
    tick_coverage,
    find_missing_candle_days,
    find_missing_tick_days,
    fill_candle_day,
    fill_tick_day,
    get_atm_strike,
    get_current_monthly_expiry,
    get_futures_close_on,
    build_option_symbols,
    is_trading_day,
    MIN_CANDLES_PER_DAY,
    MIN_OPTION_CANDLES_PER_DAY,
    MIN_FUTURES_TICKS_PER_DAY,
    MIN_OPTION_TICKS_PER_DAY,
)

logger = get_logger("eod_collect")


def report(label: str, sym: str, cov: dict, threshold: int):
    print(f"\n  {label}: {sym}  (threshold {threshold:,})")
    for d in sorted(cov.keys()):
        n = cov[d]
        flag = "OK    " if n >= threshold else "MISSING"
        print(f"    {d}  {n:>7,}  [{flag}]")


def fix_symbol_candles(td: TrueDataAdapter, sym: str, days: list[date], dry_run: bool):
    """Refetch candles for any day below MIN_CANDLES_PER_DAY."""
    cov = candle_coverage(sym, days)
    missing = [d for d in days if cov.get(d, 0) < MIN_CANDLES_PER_DAY]
    if not missing:
        print(f"    [{sym}] candles OK for all {len(days)} days")
        return 0
    print(f"    [{sym}] missing candles on {len(missing)} day(s): {missing}")
    if dry_run:
        return 0
    total = 0
    for d in missing:
        n = fill_candle_day(td, sym, d)
        print(f"      {d}: +{n} candles")
        total += n
        time.sleep(1)
    return total


def fix_symbol_ticks(td: TrueDataAdapter, sym: str, days: list[date], threshold: int, dry_run: bool):
    """Replace ticks for any day below threshold (TrueData REST 5-day window)."""
    today = date.today()
    window_start = today - timedelta(days=5)
    cov = tick_coverage(sym, days)
    missing = [d for d in days if cov.get(d, 0) < threshold and d >= window_start]
    if not missing:
        print(f"    [{sym}] ticks OK or beyond 5-day TrueData window")
        return 0
    print(f"    [{sym}] missing ticks (in 5d window) on {len(missing)} day(s): {missing}")
    if dry_run:
        return 0
    total = 0
    for d in missing:
        n = fill_tick_day(td, sym, d)
        print(f"      {d}: +{n} ticks")
        total += n
        time.sleep(1)
    return total


def process_underlying(td: TrueDataAdapter, underlying: str, days: list[date], atm_range: int, dry_run: bool):
    print(f"\n{'─' * 60}")
    print(f"  {underlying}")
    print(f"{'─' * 60}")

    spot_sym = TD_INDEX_SPOT_SYMBOLS.get(underlying)
    fut_sym = TD_INDEX_FUTURES_SYMBOLS.get(underlying)
    gap = STRIKE_GAP.get(underlying, 50)

    # ── 1. Spot + Futures candles ────────────────────────────────────
    if spot_sym:
        fix_symbol_candles(td, spot_sym, days, dry_run)
    if fut_sym:
        fix_symbol_candles(td, fut_sym, days, dry_run)

    # ── 2. Futures ticks (5-day window) ──────────────────────────────
    if fut_sym:
        fix_symbol_ticks(td, fut_sym, days, MIN_FUTURES_TICKS_PER_DAY, dry_run)

    # ── 3. Resolve current monthly expiry ────────────────────────────
    expiry = get_current_monthly_expiry(underlying, td)
    if not expiry:
        print(f"    [{underlying}] no expiry resolved — skipping options")
        return
    print(f"    expiry: {expiry}")

    # ── 4. For each day, walk that day's ATM and check option coverage
    for d in sorted(days):
        fut_close = get_futures_close_on(fut_sym, d) if fut_sym else None
        if fut_close is None or fut_close <= 0:
            print(f"    [{d}] no futures close in DB — fetch futures candles first")
            continue

        atm = get_atm_strike(fut_close, gap)
        opts = build_option_symbols(underlying, atm, expiry, atm_range, gap)
        c_total, t_total, c_miss, t_miss = 0, 0, 0, 0

        for opt in opts:
            # Option candles: only refetch if truly empty for that day.
            # Illiquid OTM strikes legitimately have <50 candles, so any non-zero
            # count means we already have what TrueData has.
            cov = candle_coverage(opt, [d]).get(d, 0)
            if cov < MIN_OPTION_CANDLES_PER_DAY:
                c_miss += 1
                if not dry_run:
                    n = fill_candle_day(td, opt, d)
                    c_total += n
                    time.sleep(1)

            # Option ticks (only within 5-day TrueData window).
            # Same reasoning: only refetch if zero.
            if d >= date.today() - timedelta(days=5):
                tcov = tick_coverage(opt, [d]).get(d, 0)
                if tcov < MIN_OPTION_TICKS_PER_DAY:
                    t_miss += 1
                    if not dry_run:
                        n = fill_tick_day(td, opt, d)
                        t_total += n
                        time.sleep(1)

        print(f"    [{d}] ATM={atm}  options_missing: candles={c_miss}/{len(opts)}, "
              f"ticks={t_miss}/{len(opts)}  filled: +{c_total}c, +{t_total}t")


def process_nifty_safety(td: TrueDataAdapter, days: list[date], dry_run: bool):
    """Optional: catch any NIFTY-I gaps the live collector missed."""
    print(f"\n{'─' * 60}")
    print(f"  NIFTY (safety check)")
    print(f"{'─' * 60}")
    fix_symbol_candles(td, "NIFTY-I", days, dry_run)
    fix_symbol_ticks(td, "NIFTY-I", days, MIN_FUTURES_TICKS_PER_DAY, dry_run)


def main():
    parser = argparse.ArgumentParser(description="Daily EOD incremental data collector (Plan D)")
    parser.add_argument("--lookback-days", type=int, default=5,
                        help="How many trading days to scan/fix (default 5 = TrueData REST cap)")
    parser.add_argument("--symbols", nargs="+", default=EOD_ONLY_SYMBOLS,
                        help=f"Underlyings to process (default: {EOD_ONLY_SYMBOLS})")
    parser.add_argument("--atm-range", type=int, default=ATM_RANGE,
                        help=f"Strikes above/below ATM (default {ATM_RANGE})")
    parser.add_argument("--include-today", action="store_true",
                        help="Include today in the lookback (use after market close)")
    parser.add_argument("--also-nifty", action="store_true",
                        help="Also run a safety backfill on NIFTY-I")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report gaps only — no fetches or writes")
    args = parser.parse_args()

    days = trading_days_back(args.lookback_days, include_today=args.include_today)
    days.sort()

    print(f"\n{'#' * 60}")
    print(f"#  EOD INCREMENTAL — Plan D")
    print(f"#  Symbols:   {args.symbols}")
    print(f"#  Lookback:  {len(days)} days  ({days[0]} → {days[-1]})")
    print(f"#  ATM range: ±{args.atm_range}")
    print(f"#  Dry-run:   {args.dry_run}")
    print(f"{'#' * 60}")

    td = TrueDataAdapter()
    if not td.authenticate():
        print("AUTH FAILED")
        return 1

    started = datetime.now()
    for ul in args.symbols:
        process_underlying(td, ul, days, args.atm_range, args.dry_run)

    if args.also_nifty:
        process_nifty_safety(td, days, args.dry_run)

    elapsed = (datetime.now() - started).total_seconds()
    print(f"\n{'=' * 60}")
    print(f"  EOD COLLECTION DONE in {elapsed/60:.1f} minutes")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
