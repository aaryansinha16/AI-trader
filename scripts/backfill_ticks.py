"""
Backfill missing/sparse tick data from TrueData REST API.

Checks tick coverage from a start date, identifies days below the threshold,
and fills gaps for dates within the last 5-day TrueData window.

Usage:
    python scripts/backfill_ticks.py                         # auto-detect gaps, default symbols
    python scripts/backfill_ticks.py --from-date 2026-03-10  # check from specific date
    python scripts/backfill_ticks.py --symbols NIFTY-I       # specific symbol only
    python scripts/backfill_ticks.py --threshold 5000        # sparse threshold (ticks/day)
    python scripts/backfill_ticks.py --dry-run               # report gaps only, no fetch
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import argparse
from datetime import datetime, date, timedelta

import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy import Table, MetaData

from data.truedata_adapter import TrueDataAdapter
from database.db import read_sql, engine
from utils.logger import get_logger

logger = get_logger("backfill_ticks")

# Market session times (IST — stored as naive local in the DB)
MARKET_OPEN  = (9, 15)
MARKET_CLOSE = (15, 30)

# Default minimum tick count to consider a day "complete"
DEFAULT_THRESHOLD = 5000

# TrueData only keeps ~5 days of tick history
TRUEDATA_TICK_WINDOW_DAYS = 5


def get_tick_counts(from_date: date, symbols: list[str]) -> pd.DataFrame:
    """Return per-(date, symbol) tick counts from DB."""
    symbol_list = ", ".join(f"'{s}'" for s in symbols)
    sql = f"""
        SELECT
            DATE(timestamp AT TIME ZONE 'UTC') AS dt,
            symbol,
            COUNT(*) AS ticks
        FROM tick_data
        WHERE symbol IN ({symbol_list})
          AND timestamp >= :from_ts
        GROUP BY 1, 2
        ORDER BY 1, 2
    """
    df = read_sql(sql, {"from_ts": datetime.combine(from_date, datetime.min.time())})
    return df


def identify_gaps(
    tick_counts: pd.DataFrame,
    symbols: list[str],
    threshold: int,
    window_start: date,
) -> dict[tuple[date, str], int]:
    """
    Return {(date, symbol): current_tick_count} for days that are sparse
    AND within the TrueData 5-day historical window.
    """
    # Build a set of all trading days we've seen data for (any symbol)
    all_dates = set(tick_counts["dt"].dt.date if hasattr(tick_counts["dt"], "dt") else
                    [d.date() if hasattr(d, "date") else d for d in tick_counts["dt"]])

    gaps = {}
    for sym in symbols:
        sym_df = tick_counts[tick_counts["symbol"] == sym].copy()
        sym_df["dt"] = sym_df["dt"].apply(lambda d: d.date() if hasattr(d, "date") else d)
        sym_counts = dict(zip(sym_df["dt"], sym_df["ticks"]))

        for d in all_dates:
            if d < window_start:
                continue
            count = sym_counts.get(d, 0)
            if count < threshold:
                gaps[(d, sym)] = count

    return gaps


def upsert_ticks(df: pd.DataFrame) -> int:
    """Insert ticks into tick_data with ON CONFLICT DO NOTHING on (timestamp, symbol)."""
    if df.empty:
        return 0

    required = ["timestamp", "symbol", "price", "volume", "oi",
                "bid_price", "ask_price", "bid_qty", "ask_qty"]

    for col in required:
        if col not in df.columns:
            if col in ("bid_price", "ask_price"):
                df[col] = df.get("price", 0)
            else:
                df[col] = 0

    df = df[required].copy()

    meta = MetaData()
    meta.reflect(bind=engine, only=["tick_data"])
    tbl = meta.tables["tick_data"]

    rows = df.to_dict(orient="records")
    inserted = 0
    with engine.begin() as conn:
        for chunk_start in range(0, len(rows), 500):
            chunk = rows[chunk_start : chunk_start + 500]
            stmt = (
                pg_insert(tbl)
                .values(chunk)
                .on_conflict_do_nothing(index_elements=["timestamp", "symbol"])
            )
            result = conn.execute(stmt)
            inserted += result.rowcount

    return inserted


def backfill_day(td: TrueDataAdapter, target_date: date, symbol: str) -> int:
    """Fetch a full trading day of ticks and upsert into DB. Returns inserted count."""
    start_dt = datetime.combine(target_date, datetime.min.time().replace(
        hour=MARKET_OPEN[0], minute=MARKET_OPEN[1]))
    end_dt = datetime.combine(target_date, datetime.min.time().replace(
        hour=MARKET_CLOSE[0], minute=MARKET_CLOSE[1]))

    logger.info(f"  Fetching {symbol} on {target_date} ({start_dt.strftime('%H:%M')}–{end_dt.strftime('%H:%M')} IST)")

    ticks = td.fetch_historical_ticks(symbol=symbol, start=start_dt, end=end_dt)
    if ticks.empty:
        logger.warning(f"  No ticks returned for {symbol} on {target_date}")
        return 0

    logger.info(f"  Got {len(ticks)} ticks from API")
    inserted = upsert_ticks(ticks)
    logger.info(f"  Inserted {inserted} new ticks (skipped {len(ticks) - inserted} duplicates)")
    return inserted


def main():
    parser = argparse.ArgumentParser(description="Backfill missing tick data from TrueData")
    parser.add_argument(
        "--from-date", default="2026-03-10",
        help="Check coverage from this date (YYYY-MM-DD, default: 2026-03-10)"
    )
    parser.add_argument(
        "--symbols", nargs="+", default=["NIFTY-I"],
        help="Symbols to check (default: NIFTY-I)"
    )
    parser.add_argument(
        "--threshold", type=int, default=DEFAULT_THRESHOLD,
        help=f"Minimum ticks/day to consider complete (default: {DEFAULT_THRESHOLD})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report gaps only — do not fetch or insert"
    )
    args = parser.parse_args()

    from_date = datetime.strptime(args.from_date, "%Y-%m-%d").date()
    today = date.today()
    window_start = today - timedelta(days=TRUEDATA_TICK_WINDOW_DAYS)

    logger.info("=" * 60)
    logger.info("Tick Data Coverage Report")
    logger.info("=" * 60)
    logger.info(f"Checking from:       {from_date}")
    logger.info(f"Symbols:             {args.symbols}")
    logger.info(f"Sparse threshold:    < {args.threshold:,} ticks/day")
    logger.info(f"TrueData 5-day window: {window_start} → {today}")
    logger.info("")

    # 1. Get current tick counts
    tick_counts = get_tick_counts(from_date, args.symbols)

    if tick_counts.empty:
        logger.warning("No tick data found at all from the given start date.")
    else:
        logger.info("Current tick coverage:")
        for _, row in tick_counts.iterrows():
            dt = row["dt"].date() if hasattr(row["dt"], "date") else row["dt"]
            status = "OK" if row["ticks"] >= args.threshold else "SPARSE"
            in_window = "(in 5d window)" if dt >= window_start else "(beyond window)"
            logger.info(f"  {dt}  {row['symbol']:20s}  {int(row['ticks']):>7,} ticks  [{status}] {in_window}")

    logger.info("")

    # 2. Identify gaps within the refillable window
    gaps = identify_gaps(tick_counts, args.symbols, args.threshold, window_start)

    if not gaps:
        logger.info("No sparse days found within the 5-day TrueData window. All good!")
        return 0

    logger.info(f"Found {len(gaps)} gap(s) within the refillable window:")
    for (d, sym), count in sorted(gaps.items()):
        logger.info(f"  {d}  {sym}  — {count:,} ticks (below {args.threshold:,})")

    if args.dry_run:
        logger.info("\n--dry-run specified. Skipping fetch.")
        return 0

    # 3. Backfill each gap
    logger.info("\nStarting backfill...")
    td = TrueDataAdapter()

    total_inserted = 0
    for (target_date, symbol) in sorted(gaps.keys()):
        logger.info(f"\n[{target_date}] {symbol}")
        inserted = backfill_day(td, target_date, symbol)
        total_inserted += inserted

    # 4. Summary
    logger.info("\n" + "=" * 60)
    logger.info("BACKFILL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Gaps processed:  {len(gaps)}")
    logger.info(f"New ticks added: {total_inserted:,}")

    # Re-check counts for filled days
    if total_inserted > 0:
        logger.info("\nUpdated counts for backfilled days:")
        filled_dates = sorted({d for (d, _) in gaps.keys()})
        updated = get_tick_counts(min(filled_dates), args.symbols)
        updated["dt_date"] = updated["dt"].apply(lambda d: d.date() if hasattr(d, "date") else d)
        for _, row in updated[updated["dt_date"].isin(filled_dates)].iterrows():
            logger.info(f"  {row['dt_date']}  {row['symbol']:20s}  {int(row['ticks']):>7,} ticks")

    return 0


if __name__ == "__main__":
    sys.exit(main())
