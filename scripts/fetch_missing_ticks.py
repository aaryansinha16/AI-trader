"""
Fetch missing tick data for specific dates from TrueData API.
Usage: python scripts/fetch_missing_ticks.py --dates 2026-03-19 2026-03-20
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime, date, timedelta
import pandas as pd
from data.truedata_adapter import TrueDataAdapter
from database.db import get_engine, write_df, upsert_candles
from utils.logger import get_logger

logger = get_logger("fetch_missing_ticks")


def fetch_ticks_for_date(td: TrueDataAdapter, symbol: str, target_date: date) -> pd.DataFrame:
    """Fetch all ticks for a symbol on a specific date."""
    start_dt = datetime.combine(target_date, datetime.min.time().replace(hour=9, minute=15))
    end_dt = datetime.combine(target_date, datetime.min.time().replace(hour=15, minute=30))
    
    logger.info(f"Fetching ticks for {symbol} on {target_date} ({start_dt} to {end_dt})")
    
    try:
        ticks = td.fetch_historical_ticks(
            symbol=symbol,
            start=start_dt,
            end=end_dt
        )
        
        if ticks.empty:
            logger.warning(f"No ticks returned for {symbol} on {target_date}")
            return pd.DataFrame()
        
        # Add symbol column if not present
        if 'symbol' not in ticks.columns:
            ticks['symbol'] = symbol
        
        logger.info(f"Fetched {len(ticks)} ticks for {symbol} on {target_date}")
        return ticks
    
    except Exception as e:
        logger.error(f"Error fetching ticks for {symbol} on {target_date}: {e}")
        return pd.DataFrame()


def fetch_candles_for_date(td: TrueDataAdapter, symbol: str, target_date: date) -> pd.DataFrame:
    """Fetch 1-minute candles for a symbol on a specific date."""
    start_dt = datetime.combine(target_date, datetime.min.time().replace(hour=9, minute=15))
    end_dt = datetime.combine(target_date, datetime.min.time().replace(hour=15, minute=30))
    
    logger.info(f"Fetching candles for {symbol} on {target_date}")
    
    try:
        candles = td.fetch_historical_bars(
            symbol=symbol,
            start=start_dt,
            end=end_dt,
            interval='1min'
        )
        
        if candles.empty:
            logger.warning(f"No candles returned for {symbol} on {target_date}")
            return pd.DataFrame()
        
        # Add symbol column if not present
        if 'symbol' not in candles.columns:
            candles['symbol'] = symbol
        
        # Add vwap column if missing (calculate from OHLC)
        if 'vwap' not in candles.columns:
            candles['vwap'] = (candles['high'] + candles['low'] + candles['close']) / 3
        
        logger.info(f"Fetched {len(candles)} candles for {symbol} on {target_date}")
        return candles
    
    except Exception as e:
        logger.error(f"Error fetching candles for {symbol} on {target_date}: {e}")
        return pd.DataFrame()


def store_ticks(ticks_df: pd.DataFrame) -> int:
    """Store ticks in the database."""
    if ticks_df.empty:
        return 0
    
    # Prepare columns for tick_data table
    required_cols = ['timestamp', 'symbol', 'price', 'volume', 'oi', 
                     'bid_price', 'ask_price', 'bid_qty', 'ask_qty']
    
    # Fill missing columns with defaults
    for col in required_cols:
        if col not in ticks_df.columns:
            if col in ['bid_price', 'ask_price']:
                ticks_df[col] = ticks_df['price']
            elif col in ['bid_qty', 'ask_qty']:
                ticks_df[col] = ticks_df.get('volume', 0)
            elif col == 'oi':
                ticks_df[col] = 0
            else:
                ticks_df[col] = 0
    
    # Select only required columns
    ticks_df = ticks_df[required_cols].copy()
    
    # Use pandas to_sql with append mode
    initial_count = len(ticks_df)
    try:
        write_df(ticks_df, 'tick_data', if_exists='append')
        logger.info(f"Inserted {initial_count} ticks into tick_data")
        return initial_count
    except Exception as e:
        logger.error(f"Error inserting ticks: {e}")
        return 0


def store_candles(candles_df: pd.DataFrame) -> int:
    """Store candles in the database."""
    if candles_df.empty:
        return 0
    
    # Prepare columns for minute_candles table
    required_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'oi']
    
    # Fill missing columns with defaults
    for col in required_cols:
        if col not in candles_df.columns:
            if col == 'vwap':
                candles_df[col] = (candles_df['high'] + candles_df['low'] + candles_df['close']) / 3
            elif col in ['oi', 'volume']:
                candles_df[col] = 0
    
    # Select only required columns
    candles_df = candles_df[required_cols].copy()
    
    initial_count = len(candles_df)
    try:
        upsert_candles(candles_df)
        logger.info(f"Upserted {initial_count} candles into minute_candles")
        return initial_count
    except Exception as e:
        logger.error(f"Error upserting candles: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description='Fetch missing tick data from TrueData')
    parser.add_argument('--dates', nargs='+', required=True, 
                       help='Dates to fetch in YYYY-MM-DD format (e.g., 2026-03-19 2026-03-20)')
    parser.add_argument('--symbol', default='NIFTY-I', help='Symbol to fetch (default: NIFTY-I)')
    
    args = parser.parse_args()
    
    # Parse dates
    target_dates = []
    for date_str in args.dates:
        try:
            target_dates.append(datetime.strptime(date_str, '%Y-%m-%d').date())
        except ValueError:
            logger.error(f"Invalid date format: {date_str}. Use YYYY-MM-DD")
            return 1
    
    logger.info(f"Fetching data for {len(target_dates)} dates: {target_dates}")
    
    # Initialize TrueData adapter
    td = TrueDataAdapter()
    
    total_ticks = 0
    total_candles = 0
    
    for target_date in target_dates:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {target_date}")
        logger.info(f"{'='*60}")
        
        # Fetch ticks
        ticks = fetch_ticks_for_date(td, args.symbol, target_date)
        if not ticks.empty:
            inserted = store_ticks(ticks)
            total_ticks += inserted
        
        # Fetch candles
        candles = fetch_candles_for_date(td, args.symbol, target_date)
        if not candles.empty:
            inserted = store_candles(candles)
            total_candles += inserted
    
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total ticks inserted: {total_ticks}")
    logger.info(f"Total candles inserted: {total_candles}")
    logger.info(f"Dates processed: {len(target_dates)}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
