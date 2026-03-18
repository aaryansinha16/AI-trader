"""
Load Historical CSV Data into TimescaleDB
──────────────────────────────────────────
Reads all CSV files from data/historical/ and bulk-inserts them into
the appropriate TimescaleDB hypertables.

Mapping:
  nifty_index_1m.csv           → minute_candles
  nifty_options_1m/*.csv       → minute_candles
  nifty_index_ticks.csv        → tick_data
  nifty_options_ticks/*.csv    → tick_data

Run: python scripts/load_csv_to_db.py
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
from sqlalchemy import text

from database.db import get_engine, init_db, execute_sql
from utils.logger import get_logger

logger = get_logger("csv_to_db")

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "historical"


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_minute_bars(engine, csv_path: Path, batch_label: str):
    """Load a 1-min bar CSV into the minute_candles hypertable."""
    df = pd.read_csv(csv_path)
    if df.empty:
        logger.warning(f"  {batch_label}: empty CSV, skipping")
        return 0

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Compute VWAP if not present
    if "vwap" not in df.columns:
        typical = (df["high"] + df["low"] + df["close"]) / 3
        cum_tp_vol = (typical * df["volume"]).cumsum()
        cum_vol = df["volume"].cumsum().replace(0, np.nan)
        df["vwap"] = cum_tp_vol / cum_vol

    cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume", "vwap"]
    if "oi" in df.columns:
        cols.append("oi")

    df[cols].to_sql(
        "minute_candles", engine,
        if_exists="append", index=False, method="multi",
        chunksize=5000,
    )
    return len(df)


def load_tick_data(engine, csv_path: Path, batch_label: str):
    """Load a tick CSV into the tick_data hypertable."""
    df = pd.read_csv(csv_path)
    if df.empty:
        logger.warning(f"  {batch_label}: empty CSV, skipping")
        return 0

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Map CSV columns to tick_data schema
    cols_map = {
        "timestamp": "timestamp",
        "symbol": "symbol",
        "price": "price",
        "volume": "volume",
        "oi": "oi",
        "bid_price": "bid_price",
        "ask_price": "ask_price",
        "bid_qty": "bid_qty",
        "ask_qty": "ask_qty",
    }

    out = pd.DataFrame()
    for csv_col, db_col in cols_map.items():
        if csv_col in df.columns:
            out[db_col] = df[csv_col]
        else:
            out[db_col] = None

    out.to_sql(
        "tick_data", engine,
        if_exists="append", index=False, method="multi",
        chunksize=5000,
    )
    return len(out)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  CSV → TimescaleDB Loader")
    print("=" * 60)
    print(f"  Data dir: {DATA_DIR}")

    engine = get_engine()

    # Verify connection
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        result.fetchone()
    logger.info("Database connection verified.")

    # Initialize schema (idempotent)
    init_db()

    # Check if tables already have data
    with engine.connect() as conn:
        mc_count = conn.execute(text("SELECT COUNT(*) FROM minute_candles")).scalar()
        td_count = conn.execute(text("SELECT COUNT(*) FROM tick_data")).scalar()
    logger.info(f"Current DB state: minute_candles={mc_count:,} rows, tick_data={td_count:,} rows")

    if mc_count > 0 or td_count > 0:
        logger.warning("Tables already contain data. Truncating before reload...")
        execute_sql("TRUNCATE minute_candles")
        execute_sql("TRUNCATE tick_data")
        logger.info("Tables truncated.")

    stats = {
        "minute_bars_loaded": 0,
        "minute_files": 0,
        "tick_rows_loaded": 0,
        "tick_files": 0,
    }
    t0 = time.time()

    # ── 1. Load index 1-min bars ──────────────────────────────────────────────
    index_csv = DATA_DIR / "nifty_index_1m.csv"
    if index_csv.exists():
        logger.info(f"\n  Loading index 1-min bars: {index_csv.name}")
        n = load_minute_bars(engine, index_csv, "index_1m")
        stats["minute_bars_loaded"] += n
        stats["minute_files"] += 1
        logger.info(f"    → {n:,} rows inserted")

    # ── 2. Load option 1-min bars ─────────────────────────────────────────────
    opt_dir = DATA_DIR / "nifty_options_1m"
    if opt_dir.exists():
        opt_files = sorted(opt_dir.glob("*.csv"))
        logger.info(f"\n  Loading {len(opt_files)} option 1-min bar files...")
        for i, f in enumerate(opt_files):
            n = load_minute_bars(engine, f, f.stem)
            stats["minute_bars_loaded"] += n
            stats["minute_files"] += 1
            if (i + 1) % 50 == 0 or (i + 1) == len(opt_files):
                logger.info(f"    Progress: {i+1}/{len(opt_files)} files, {stats['minute_bars_loaded']:,} total rows")

    # ── 3. Load index ticks ───────────────────────────────────────────────────
    tick_csv = DATA_DIR / "nifty_index_ticks.csv"
    if tick_csv.exists():
        logger.info(f"\n  Loading index ticks: {tick_csv.name}")
        n = load_tick_data(engine, tick_csv, "index_ticks")
        stats["tick_rows_loaded"] += n
        stats["tick_files"] += 1
        logger.info(f"    → {n:,} rows inserted")

    # ── 4. Load option ticks ──────────────────────────────────────────────────
    opt_tick_dir = DATA_DIR / "nifty_options_ticks"
    if opt_tick_dir.exists():
        tick_files = sorted(opt_tick_dir.glob("*.csv"))
        logger.info(f"\n  Loading {len(tick_files)} option tick files...")
        for i, f in enumerate(tick_files):
            n = load_tick_data(engine, f, f.stem)
            stats["tick_rows_loaded"] += n
            stats["tick_files"] += 1
            if (i + 1) % 5 == 0 or (i + 1) == len(tick_files):
                logger.info(f"    Progress: {i+1}/{len(tick_files)} files, {stats['tick_rows_loaded']:,} total rows")

    elapsed = time.time() - t0

    # ── Verify ────────────────────────────────────────────────────────────────
    with engine.connect() as conn:
        mc_final = conn.execute(text("SELECT COUNT(*) FROM minute_candles")).scalar()
        td_final = conn.execute(text("SELECT COUNT(*) FROM tick_data")).scalar()
        mc_symbols = conn.execute(text("SELECT COUNT(DISTINCT symbol) FROM minute_candles")).scalar()
        td_symbols = conn.execute(text("SELECT COUNT(DISTINCT symbol) FROM tick_data")).scalar()
        mc_range = conn.execute(text(
            "SELECT MIN(timestamp), MAX(timestamp) FROM minute_candles"
        )).fetchone()

    print("\n" + "=" * 60)
    print("  LOAD COMPLETE")
    print("=" * 60)
    print(f"  Time: {elapsed:.1f}s")
    print(f"\n  minute_candles:")
    print(f"    Rows: {mc_final:,}")
    print(f"    Symbols: {mc_symbols}")
    print(f"    Range: {mc_range[0]} → {mc_range[1]}")
    print(f"    Files loaded: {stats['minute_files']}")
    print(f"\n  tick_data:")
    print(f"    Rows: {td_final:,}")
    print(f"    Symbols: {td_symbols}")
    print(f"    Files loaded: {stats['tick_files']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
