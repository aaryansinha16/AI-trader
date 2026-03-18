"""
Fetch 5+ working days of tick data for NIFTY-I and ATM options.
Explicitly requests from Mar 9 → Mar 18 (8 trading days).
Loads directly into TimescaleDB.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import text

from data.truedata_adapter import TrueDataAdapter
from data.symbol_manager import SymbolManager
from database.db import get_engine, execute_sql
from config.settings import STRIKE_GAP
from utils.logger import get_logger

logger = get_logger("fetch_ticks")


def main():
    print("=" * 60)
    print("  Fetching 5+ working days of tick data")
    print("=" * 60)

    td = TrueDataAdapter()
    td.authenticate()
    engine = get_engine()

    # Mar 9 (Monday) → Mar 18 (Wednesday) = 8 trading days
    end = datetime(2026, 3, 18, 15, 30)
    start = datetime(2026, 3, 9, 9, 15)

    # Truncate existing tick data and reload
    print("\nTruncating existing tick_data...")
    execute_sql("TRUNCATE tick_data")

    # ── 1. Index ticks ────────────────────────────────────────────────────
    print(f"\nFetching NIFTY-I ticks: {start.date()} → {end.date()}")
    idx_ticks = td.fetch_historical_ticks("NIFTY-I", start=start, end=end, days=10)
    if not idx_ticks.empty:
        idx_ticks["timestamp"] = pd.to_datetime(idx_ticks["timestamp"], utc=True)
        idx_ticks.to_sql("tick_data", engine, if_exists="append", index=False, method="multi", chunksize=5000)
        days = idx_ticks["timestamp"].dt.date.nunique()
        print(f"  NIFTY-I: {len(idx_ticks):,} ticks across {days} trading days")
        print(f"  Range: {idx_ticks['timestamp'].min()} → {idx_ticks['timestamp'].max()}")
    else:
        print("  WARNING: No index ticks returned!")

    # ── 2. ATM option ticks (nearest expiry) ──────────────────────────────
    sym_mgr = SymbolManager()
    sym_mgr.load_symbol_master("NIFTY")
    expiries = sym_mgr.fetch_expiry_list("NIFTY")

    # Find nearest expiry to our end date
    from datetime import date
    ref = end.date()
    nearest_exp = None
    for e in expiries:
        if e >= ref:
            nearest_exp = e
            break
    if nearest_exp is None and expiries:
        nearest_exp = expiries[-1]

    if nearest_exp:
        print(f"\nFetching option ticks for expiry {nearest_exp}...")
        # Get ATM from latest index price
        latest_price = idx_ticks["price"].iloc[-1] if not idx_ticks.empty else 23400
        atm = round(latest_price / 50) * 50
        strikes = [atm + i * 50 for i in range(-3, 4)]

        total_opt_ticks = 0
        for strike in strikes:
            for opt_type in ["CE", "PE"]:
                exp_str = nearest_exp.strftime("%y%m%d")
                sym = f"NIFTY{exp_str}{strike}{opt_type}"
                try:
                    opt_df = td.fetch_historical_ticks(sym, start=start, end=end, days=10)
                    if not opt_df.empty:
                        opt_df["timestamp"] = pd.to_datetime(opt_df["timestamp"], utc=True)
                        opt_df.to_sql("tick_data", engine, if_exists="append", index=False, method="multi", chunksize=5000)
                        total_opt_ticks += len(opt_df)
                        print(f"    {sym}: {len(opt_df):,} ticks")
                except Exception as e:
                    print(f"    {sym}: error - {e}")

        print(f"\n  Total option ticks: {total_opt_ticks:,}")

    # ── Verify ────────────────────────────────────────────────────────────
    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM tick_data")).scalar()
        symbols = conn.execute(text("SELECT COUNT(DISTINCT symbol) FROM tick_data")).scalar()
        days = conn.execute(text("SELECT COUNT(DISTINCT timestamp::date) FROM tick_data")).scalar()
        date_range = conn.execute(text("SELECT MIN(timestamp)::date, MAX(timestamp)::date FROM tick_data")).fetchone()

    print(f"\n{'='*60}")
    print(f"  TICK DATA LOADED")
    print(f"{'='*60}")
    print(f"  Total rows:    {total:,}")
    print(f"  Symbols:       {symbols}")
    print(f"  Trading days:  {days}")
    print(f"  Date range:    {date_range[0]} → {date_range[1]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
