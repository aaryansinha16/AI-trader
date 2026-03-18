"""
Historical Data Ingest Script
──────────────────────────────
Fetches all historical data needed for ML training from TrueData REST API
and saves to CSV files in data/historical/.

No database required — pure CSV output for verification before DB load.

Run: python scripts/ingest_historical.py

Data collected:
  1. NIFTY-I 1-min bars (6 months, chunked by month)
  2. NIFTY-I tick data (5 days)
  3. Option 1-min bars for ALL 26 weekly expiries in past 6 months
     - ATM±3 strikes (CE+PE) = 14 symbols per expiry
     - Each option fetched for its full active lifespan (~7 days before expiry)
  4. Option tick data for the nearest active expiry

Output structure:
  data/historical/
    nifty_index_1m.csv          # 6 months of 1-min bars for NIFTY-I
    nifty_index_ticks.csv       # 5 days of tick data for NIFTY-I
    nifty_expiry_list.csv       # Future expiry dates from API
    nifty_historical_expiries.csv  # All 26 discovered historical expiry dates
    nifty_options_1m/           # 1-min bars per option symbol (all expiries)
      NIFTY25092325000CE.csv
      ...
    nifty_options_ticks/        # Tick data per option symbol (nearest expiry)
      NIFTY26032423700CE.csv
      ...
    ingest_summary.json         # Summary of what was fetched
"""

import os
import sys
import json
import time
from datetime import datetime, date, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from io import StringIO

from config.settings import (
    TRUEDATA_USER, TRUEDATA_PASSWORD, SYMBOLS,
    TD_AUTH_URL, TD_HISTORY_URL, TD_SYMBOL_MASTER_URL,
    STRIKE_GAP, ATM_RANGE,
)
from data.truedata_adapter import TrueDataAdapter
from data.symbol_manager import SymbolManager
from utils.logger import get_logger

logger = get_logger("ingest")

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "historical"
INDEX_BAR_DAYS = 180          # 6 months of 1-min bars
INDEX_TICK_DAYS = 5           # 5 days of tick data for index
OPTION_TICK_DAYS = 5          # 5 days of tick data for current-week options
OPTION_LOOKBACK_DAYS = 10     # fetch option bars starting this many days before expiry

# ── Discovered Historical Expiry Dates ────────────────────────────────────────
# NIFTY weekly expiries (mostly Tuesdays, some Mondays due to holidays).
# Discovered by probing the TrueData history API for each week in the 6-month window.
NIFTY_HISTORICAL_EXPIRIES = [
    date(2025, 9, 23),
    date(2025, 9, 30),
    date(2025, 10, 7),
    date(2025, 10, 14),
    date(2025, 10, 20),   # Monday
    date(2025, 10, 28),
    date(2025, 11, 4),
    date(2025, 11, 11),
    date(2025, 11, 18),
    date(2025, 11, 25),
    date(2025, 12, 2),
    date(2025, 12, 9),
    date(2025, 12, 16),
    date(2025, 12, 23),
    date(2025, 12, 30),
    date(2026, 1, 6),
    date(2026, 1, 13),
    date(2026, 1, 20),
    date(2026, 1, 27),
    date(2026, 2, 3),
    date(2026, 2, 10),
    date(2026, 2, 17),
    date(2026, 2, 24),
    date(2026, 3, 2),    # Monday
    date(2026, 3, 10),
    date(2026, 3, 17),
]


def ensure_dirs():
    """Create output directories."""
    for sub in ["", "nifty_options_1m", "nifty_options_ticks"]:
        (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {OUTPUT_DIR}")


def fetch_and_save_index_bars(td: TrueDataAdapter, underlying: str):
    """Fetch 6 months of 1-min bars for the index continuous future."""
    from config.settings import TD_INDEX_FUTURES_SYMBOLS
    index_sym = TD_INDEX_FUTURES_SYMBOLS.get(underlying, f"{underlying}-I")

    logger.info(f"Fetching {INDEX_BAR_DAYS} days of 1-min bars for {index_sym}...")
    df = td.fetch_historical_minute_bars(index_sym, days=INDEX_BAR_DAYS)

    if df.empty:
        logger.error(f"No bars returned for {index_sym}")
        return df

    out_path = OUTPUT_DIR / f"{underlying.lower()}_index_1m.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"  Saved {len(df)} bars to {out_path.name}")
    logger.info(f"  Range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    return df


def fetch_and_save_index_ticks(td: TrueDataAdapter, underlying: str):
    """Fetch 5 days of tick data for index."""
    from config.settings import TD_INDEX_FUTURES_SYMBOLS
    index_sym = TD_INDEX_FUTURES_SYMBOLS.get(underlying, f"{underlying}-I")

    logger.info(f"Fetching {INDEX_TICK_DAYS} days of ticks for {index_sym}...")
    df = td.fetch_historical_ticks(index_sym, days=INDEX_TICK_DAYS)

    if df.empty:
        logger.warning(f"No ticks returned for {index_sym}")
        return df

    out_path = OUTPUT_DIR / f"{underlying.lower()}_index_ticks.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"  Saved {len(df)} ticks to {out_path.name}")
    return df


def fetch_expiry_list(td: TrueDataAdapter, sym_mgr: SymbolManager, underlying: str):
    """Fetch and save expiry list."""
    expiries = sym_mgr.fetch_expiry_list(underlying)
    if not expiries:
        logger.error(f"No expiries found for {underlying}")
        return []

    df = pd.DataFrame({"expiry": [str(e) for e in expiries]})
    out_path = OUTPUT_DIR / f"{underlying.lower()}_expiry_list.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"  {len(expiries)} expiries saved to {out_path.name}")
    logger.info(f"  Next 5: {expiries[:5]}")
    return expiries


def compute_atm_for_expiry(index_df: pd.DataFrame, expiry: date, underlying: str):
    """
    Compute the ATM strike for a given expiry date using the index close
    from the trading day just before that expiry (or on the expiry day itself).
    """
    if index_df.empty:
        return None, []

    gap = STRIKE_GAP.get(underlying, 50)
    n = ATM_RANGE

    idf = index_df.copy()
    idf["dt"] = pd.to_datetime(idf["timestamp"])
    idf["date"] = idf["dt"].dt.date

    # Use close from the expiry day or the nearest prior trading day
    daily = idf.groupby("date")["close"].last().reset_index()
    daily.columns = ["date", "close"]
    prior = daily[daily["date"] <= expiry]

    if prior.empty:
        return None, []

    ref_close = prior.iloc[-1]["close"]
    atm = round(ref_close / gap) * gap
    strikes = [atm + i * gap for i in range(-n, n + 1)]
    return atm, strikes


def fetch_option_bars_for_expiry(
    td: TrueDataAdapter,
    sym_mgr: SymbolManager,
    underlying: str,
    expiry: date,
    strikes: list,
    out_dir: Path,
):
    """
    Fetch 1-min bars for ATM±N strikes (CE+PE) for a specific expiry.
    Uses a date window from OPTION_LOOKBACK_DAYS before expiry to expiry EOD.
    Returns count of symbols fetched.
    """
    fetched = 0
    end_dt = datetime(expiry.year, expiry.month, expiry.day, 15, 30, 0)
    start_dt = end_dt - timedelta(days=OPTION_LOOKBACK_DAYS)

    for strike in strikes:
        for opt_type in ["CE", "PE"]:
            sym_name = sym_mgr.build_option_symbol_name(
                underlying, expiry, strike, opt_type
            )

            out_path = out_dir / f"{sym_name}.csv"
            if out_path.exists():
                fetched += 1
                continue

            df = td.fetch_historical_bars(sym_name, start_dt, end_dt, "1min")
            if not df.empty:
                df.to_csv(out_path, index=False)
                fetched += 1
                logger.info(f"    {sym_name}: {len(df)} bars")
            else:
                logger.debug(f"    {sym_name}: no data")

    return fetched


def fetch_option_ticks_for_current_week(
    td: TrueDataAdapter,
    sym_mgr: SymbolManager,
    underlying: str,
    expiry: date,
    strikes: list,
    out_dir: Path,
):
    """Fetch 5 days of tick data for current week's option strikes."""
    fetched = 0
    for strike in strikes:
        for opt_type in ["CE", "PE"]:
            sym_name = sym_mgr.build_option_symbol_name(
                underlying, expiry, strike, opt_type
            )

            out_path = out_dir / f"{sym_name}.csv"
            if out_path.exists():
                fetched += 1
                continue

            df = td.fetch_historical_ticks(sym_name, days=OPTION_TICK_DAYS)
            if not df.empty:
                df.to_csv(out_path, index=False)
                fetched += 1
                logger.info(f"    {sym_name}: {len(df)} ticks")
            else:
                logger.debug(f"    {sym_name}: no tick data")

    return fetched


def main():
    print("\n" + "=" * 60)
    print("  TrueData Historical Data Ingest")
    print("=" * 60)
    print(f"  Time: {datetime.now()}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Symbols: {SYMBOLS}")
    print(f"  Index bars: {INDEX_BAR_DAYS} days, Ticks: {INDEX_TICK_DAYS} days")
    print()

    ensure_dirs()
    summary = {"start_time": str(datetime.now()), "symbols": {}}

    # 1. Authenticate
    td = TrueDataAdapter()
    if not td.authenticate():
        logger.error("Authentication failed. Check .env credentials.")
        return

    # 2. Set up SymbolManager
    sym_mgr = SymbolManager()

    for underlying in SYMBOLS:
        logger.info(f"\n{'='*50}")
        logger.info(f"  INGESTING: {underlying}")
        logger.info(f"{'='*50}")

        sym_summary = {}

        # 3. Fetch future expiry list (for reference)
        future_expiries = fetch_expiry_list(td, sym_mgr, underlying)
        sym_summary["future_expiries"] = len(future_expiries)

        # 4. Fetch 6 months of index 1-min bars (skip if already exists)
        index_path = OUTPUT_DIR / f"{underlying.lower()}_index_1m.csv"
        if index_path.exists():
            logger.info(f"  Index bars already exist at {index_path.name}, loading...")
            index_df = pd.read_csv(index_path)
            index_df["timestamp"] = pd.to_datetime(index_df["timestamp"])
        else:
            index_df = fetch_and_save_index_bars(td, underlying)
        sym_summary["index_bars"] = len(index_df)

        if index_df.empty:
            continue

        # 5. Fetch 5 days of index ticks (skip if already exists)
        tick_path = OUTPUT_DIR / f"{underlying.lower()}_index_ticks.csv"
        if tick_path.exists():
            logger.info(f"  Index ticks already exist at {tick_path.name}")
            tick_df = pd.read_csv(tick_path)
        else:
            tick_df = fetch_and_save_index_ticks(td, underlying)
        sym_summary["index_ticks"] = len(tick_df)

        # 6. Save discovered historical expiry dates
        hist_expiries = NIFTY_HISTORICAL_EXPIRIES  # TODO: add BANKNIFTY when ready
        exp_df = pd.DataFrame({"expiry": [str(e) for e in hist_expiries]})
        exp_path = OUTPUT_DIR / f"{underlying.lower()}_historical_expiries.csv"
        exp_df.to_csv(exp_path, index=False)
        logger.info(f"  {len(hist_expiries)} historical expiries saved")

        # 7. Fetch option 1-min bars for ALL 26 historical expiries
        #    For each expiry, compute ATM from the index close near that date,
        #    then fetch ATM±3 × CE+PE = 14 symbols.
        logger.info(f"\n  Fetching option bars for {len(hist_expiries)} historical expiries...")
        total_option_files = 0
        total_option_symbols = 0
        out_dir = OUTPUT_DIR / f"{underlying.lower()}_options_1m"

        for i, expiry in enumerate(hist_expiries):
            atm, strikes = compute_atm_for_expiry(index_df, expiry, underlying)
            if atm is None:
                logger.warning(f"  [{i+1}/{len(hist_expiries)}] Expiry {expiry}: no index data, skipping")
                continue

            n_syms = len(strikes) * 2  # CE + PE
            total_option_symbols += n_syms
            logger.info(
                f"  [{i+1}/{len(hist_expiries)}] Expiry {expiry}: "
                f"ATM={int(atm)}, strikes={int(strikes[0])}-{int(strikes[-1])}, "
                f"{n_syms} symbols"
            )

            count = fetch_option_bars_for_expiry(
                td, sym_mgr, underlying, expiry, strikes, out_dir
            )
            total_option_files += count

        sym_summary["historical_expiries"] = len(hist_expiries)
        sym_summary["option_symbols_attempted"] = total_option_symbols
        sym_summary["option_files_saved"] = total_option_files

        # 8. Also fetch option bars for upcoming future expiries (next 4 weeks)
        near_future = [e for e in future_expiries if e >= date.today()][:4]
        if near_future and not index_df.empty:
            logger.info(f"\n  Fetching option bars for {len(near_future)} future expiries...")
            latest_close = index_df.iloc[-1]["close"]
            gap = STRIKE_GAP.get(underlying, 50)
            atm_now = round(latest_close / gap) * gap
            strikes_now = [atm_now + i * gap for i in range(-ATM_RANGE, ATM_RANGE + 1)]

            for expiry in near_future:
                logger.info(f"  Future expiry {expiry}: ATM={int(atm_now)}")
                count = fetch_option_bars_for_expiry(
                    td, sym_mgr, underlying, expiry, strikes_now, out_dir
                )
                total_option_files += count

            sym_summary["future_expiries_fetched"] = [str(e) for e in near_future]

        # 9. Fetch tick data for nearest active expiry
        nearest_expiry = near_future[0] if near_future else None
        if nearest_expiry and not index_df.empty:
            latest_close = index_df.iloc[-1]["close"]
            gap = STRIKE_GAP.get(underlying, 50)
            atm_now = round(latest_close / gap) * gap
            strikes_now = [atm_now + i * gap for i in range(-ATM_RANGE, ATM_RANGE + 1)]

            logger.info(
                f"\n  Fetching option ticks for nearest expiry "
                f"(expiry={nearest_expiry}, ATM={int(atm_now)})"
            )
            tick_out_dir = OUTPUT_DIR / f"{underlying.lower()}_options_ticks"
            tick_count = fetch_option_ticks_for_current_week(
                td, sym_mgr, underlying, nearest_expiry, strikes_now, tick_out_dir
            )
            sym_summary["option_tick_files"] = tick_count

        summary["symbols"][underlying] = sym_summary

    # 9. Save summary
    summary["end_time"] = str(datetime.now())
    summary_path = OUTPUT_DIR / "ingest_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    td.disconnect()

    # Print final report
    print("\n" + "=" * 60)
    print("  INGEST COMPLETE")
    print("=" * 60)
    for sym, info in summary["symbols"].items():
        print(f"  {sym}:")
        for k, v in info.items():
            print(f"    {k}: {v}")

    # Count files
    csv_count = len(list(OUTPUT_DIR.rglob("*.csv")))
    total_size = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*.csv"))
    print(f"\n  Total CSV files: {csv_count}")
    print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")
    print(f"  Summary: {summary_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
