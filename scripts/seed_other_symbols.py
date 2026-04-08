"""
One-shot historical seed for BANKNIFTY + FINNIFTY (Plan D).

Fetches via TrueData REST only — does NOT touch the live websocket
subscription, so it's safe to run alongside the NIFTY collector.

What it pulls:
  • 6 months of 1-min candles for index spot + futures
  • For each symbol, the current monthly expiry's ATM ± 3 strikes:
      - Last 5 trading days of 1-min option candles
      - Last 5 trading days of tick data (with bid/ask/OI)

For each historical day in the 5-day window, the script computes the
ATM from THAT DAY'S futures close (not today's), so the option strikes
collected always reflect the actual at-the-money window for that
session — which is what a live system would have subscribed to.

Idempotent: re-running only inserts new rows (uses upsert for candles
and DELETE+INSERT per (symbol, day) for ticks).

Usage:
  python scripts/seed_other_symbols.py                  # 6mo + 5d ticks
  python scripts/seed_other_symbols.py --candles-days 90  # custom depth
  python scripts/seed_other_symbols.py --no-ticks       # candles only
  python scripts/seed_other_symbols.py --symbols BANKNIFTY  # one underlying
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
    seed_candles_for_symbol,
    fill_candle_day,
    fill_tick_day,
    get_current_monthly_expiry,
    get_atm_strike,
    build_option_symbols,
    get_futures_close_on,
    trading_days_back,
    is_trading_day,
)

logger = get_logger("seed_other_symbols")


def seed_underlying(
    underlying: str,
    td: TrueDataAdapter,
    candles_days: int,
    tick_days: int,
    do_ticks: bool,
    do_options: bool,
    atm_range: int,
):
    print(f"\n{'=' * 60}")
    print(f"  SEEDING {underlying}")
    print(f"{'=' * 60}")

    spot_sym = TD_INDEX_SPOT_SYMBOLS.get(underlying)
    fut_sym = TD_INDEX_FUTURES_SYMBOLS.get(underlying)
    gap = STRIKE_GAP.get(underlying, 50)

    # ── 1. Spot index 1-min candles ──────────────────────────────────
    if spot_sym:
        print(f"\n  [SPOT] {spot_sym}")
        seed_candles_for_symbol(td, spot_sym, days=candles_days)
        time.sleep(1)

    # ── 2. Futures 1-min candles ─────────────────────────────────────
    if fut_sym:
        print(f"\n  [FUT]  {fut_sym}")
        seed_candles_for_symbol(td, fut_sym, days=candles_days)
        time.sleep(1)

    if not do_options:
        print(f"\n  [skip options — --no-options]")
        return

    # ── 3. Resolve current monthly expiry ────────────────────────────
    expiry = get_current_monthly_expiry(underlying, td)
    if not expiry:
        logger.error(f"  Could not resolve current expiry for {underlying} — skipping options")
        return
    print(f"\n  Current monthly expiry: {expiry}")

    # ── 4. For each of the last `tick_days` weekdays:
    #       (a) determine ATM from that day's futures close
    #       (b) build ATM±N option list
    #       (c) fetch 1-min candles + ticks for each
    days = trading_days_back(tick_days)
    days.sort()
    print(f"  Walking historical option ATM for: {days}")

    for d in days:
        # Get futures close for that day from DB (just seeded above)
        fut_close = get_futures_close_on(fut_sym, d) if fut_sym else None
        if fut_close is None or fut_close <= 0:
            print(f"    [{d}] no futures close in DB — skipping options for this day")
            continue

        atm = get_atm_strike(fut_close, gap)
        opts = build_option_symbols(underlying, atm, expiry, atm_range, gap)
        print(f"    [{d}] futures_close={fut_close:.1f} ATM={atm} → {len(opts)} option contracts")

        c_total, t_total, c_skip = 0, 0, 0
        for opt in opts:
            # 1-min option candles for this single day
            try:
                n_c = fill_candle_day(td, opt, d)
                c_total += n_c
            except Exception as e:
                logger.warning(f"      candle fail {opt} {d}: {e}")
            time.sleep(1)  # respect 1 req/sec

            if do_ticks:
                try:
                    n_t = fill_tick_day(td, opt, d)
                    t_total += n_t
                except Exception as e:
                    logger.warning(f"      tick fail {opt} {d}: {e}")
                time.sleep(1)

        print(f"      → upserted {c_total} candles, {t_total} ticks across {len(opts)} contracts")

    # ── 5. Backfill 5d of tick data for the futures itself ───────────
    if do_ticks and fut_sym:
        print(f"\n  [{fut_sym}] backfilling {tick_days}d of futures ticks")
        for d in days:
            try:
                n = fill_tick_day(td, fut_sym, d)
                print(f"    {d}: {n} ticks")
            except Exception as e:
                logger.warning(f"    {d} fail: {e}")
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="One-shot 6mo seed for BANKNIFTY/FINNIFTY")
    parser.add_argument("--candles-days", type=int, default=180,
                        help="How many days of 1-min candles to fetch (default 180 = 6mo)")
    parser.add_argument("--tick-days", type=int, default=5,
                        help="How many trading days of tick data to fetch (default 5 = TrueData REST cap)")
    parser.add_argument("--no-ticks", action="store_true", help="Skip tick fetches")
    parser.add_argument("--no-options", action="store_true", help="Skip option fetches (futures+spot only)")
    parser.add_argument("--symbols", nargs="+", default=EOD_ONLY_SYMBOLS,
                        help=f"Underlyings to seed (default: {EOD_ONLY_SYMBOLS})")
    parser.add_argument("--atm-range", type=int, default=ATM_RANGE,
                        help=f"Strikes above/below ATM to fetch (default {ATM_RANGE})")
    args = parser.parse_args()

    print(f"\n{'#' * 60}")
    print(f"#  TRUEDATA REST SEED — Plan D")
    print(f"#  Symbols:        {args.symbols}")
    print(f"#  Candle depth:   {args.candles_days} days")
    print(f"#  Tick depth:     {args.tick_days} days  (REST max ≈ 5)")
    print(f"#  ATM range:      ±{args.atm_range} strikes")
    print(f"#  Skip options:   {args.no_options}")
    print(f"#  Skip ticks:     {args.no_ticks}")
    print(f"{'#' * 60}")

    td = TrueDataAdapter()
    if not td.authenticate():
        print("AUTH FAILED — check TRUEDATA_USER/TRUEDATA_PASSWORD in .env")
        return 1

    started = datetime.now()
    for ul in args.symbols:
        seed_underlying(
            underlying=ul,
            td=td,
            candles_days=args.candles_days,
            tick_days=args.tick_days,
            do_ticks=not args.no_ticks,
            do_options=not args.no_options,
            atm_range=args.atm_range,
        )

    elapsed = (datetime.now() - started).total_seconds()
    print(f"\n{'=' * 60}")
    print(f"  SEED COMPLETE in {elapsed/60:.1f} minutes")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
