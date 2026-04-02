#!/usr/bin/env python3
"""
Automated Tick Data Collector
─────────────────────────────
Connects to TrueData WebSocket at market open and collects tick data
for NIFTY-I (futures) + ATM option strikes into TimescaleDB.

Designed to run unattended via cron/launchd:
  - Waits until 9:14 IST if started early
  - Connects WebSocket, subscribes to symbols
  - Collects ticks from 9:15 to 15:30 IST
  - Aggregates 1-min candles and persists both ticks + candles
  - Gracefully shuts down after market close
  - Logs everything to logs/tick_collector_YYYYMMDD.log

Usage:
  python scripts/collect_ticks.py           # run for today
  python scripts/collect_ticks.py --test    # test connection only (no wait)

Automate (macOS launchd):
  See scripts/setup_launchd.py to install the launch agent.

Automate (cron):
  55 8 * * 1-5 cd /path/to/ai-trader && .venv/bin/python scripts/collect_ticks.py >> logs/cron.log 2>&1
"""

import os
import sys
import time
import signal
import logging
import argparse
from datetime import datetime, date, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import json
import threading
import pandas as pd

from config.settings import (
    SYMBOLS, TD_INDEX_FUTURES_SYMBOLS, TD_INDEX_SPOT_SYMBOLS,
    STRIKE_GAP, ATM_RANGE, MAX_SYMBOLS,
    MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE,
    MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE,
)
from data.truedata_adapter import TrueDataAdapter
from data.tick_collector import TickCollector
from database.db import write_df, upsert_candles, read_sql, get_engine
from utils.logger import get_logger

# ── Setup logging to file ────────────────────────────────────────────────────
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"tick_collector_{date.today().strftime('%Y%m%d')}.log"

file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logging.getLogger().addHandler(file_handler)

logger = get_logger("auto_collector")

# ── Globals ──────────────────────────────────────────────────────────────────
td: TrueDataAdapter = None
collector: TickCollector = None
candle_buffer: dict = {}   # symbol -> list of ticks for current minute
last_minute: dict = {}     # symbol -> last completed minute timestamp
running = True

# Live price cache: symbol -> {price, ts} updated on every tick
live_price_cache: dict = {}
LIVE_CACHE_FILE = Path("/tmp/td_live_prices.json")

# Watchdog: track when we last received a real tick (not a heartbeat)
last_tick_received_time: float = time.time()


def signal_handler(signum, frame):
    global running
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    running = False


def _atexit_cleanup():
    """Ensure WebSocket is disconnected on exit to avoid 'User Already Connected'."""
    global running
    running = False
    try:
        if td and hasattr(td, '_ws_connected') and td._ws_connected:
            logger.info("atexit: disconnecting WebSocket...")
            td.ws_stop_streaming()
            td.ws_disconnect()
    except Exception as e:
        logger.debug(f"atexit cleanup error: {e}")
    # Clean up cache file
    try:
        LIVE_CACHE_FILE.unlink(missing_ok=True)
    except Exception:
        pass


import atexit
atexit.register(_atexit_cleanup)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ── Helpers ──────────────────────────────────────────────────────────────────

def is_trading_day() -> bool:
    """Check if today is a weekday (Mon-Fri). Does not check holidays."""
    return date.today().weekday() < 5


def market_open_time() -> datetime:
    today = date.today()
    return datetime(today.year, today.month, today.day,
                    MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE, 0)


def market_close_time() -> datetime:
    today = date.today()
    return datetime(today.year, today.month, today.day,
                    MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE, 0)


def wait_for_market_open():
    """Sleep until 1 minute before market open."""
    target = market_open_time() - timedelta(minutes=1)
    now = datetime.now()
    if now >= target:
        logger.info("Market is already open or about to open.")
        return

    wait_secs = (target - now).total_seconds()
    logger.info(f"Waiting {wait_secs/60:.1f} minutes until {target.strftime('%H:%M')} IST...")

    while datetime.now() < target and running:
        remaining = (target - datetime.now()).total_seconds()
        if remaining > 60:
            time.sleep(60)
        elif remaining > 0:
            time.sleep(remaining)
        else:
            break


def get_option_symbols_for_today(index_price: float) -> list:
    """
    Build list of ATM ± N option symbols to subscribe.
    Returns symbols like NIFTY26032024300CE, NIFTY26032024300PE, etc.
    """
    from backtest.option_resolver import get_nearest_expiry

    today = date.today()
    expiry = get_nearest_expiry(today)
    if not expiry:
        # Fallback: next Tuesday (NIFTY weekly expiry day)
        days_ahead = 1 - today.weekday()  # Tuesday = 1
        if days_ahead <= 0:
            days_ahead += 7
        expiry = today + timedelta(days=days_ahead)

    gap = STRIKE_GAP.get("NIFTY", 50)
    atm = round(index_price / gap) * gap
    exp_code = expiry.strftime("%y%m%d")

    symbols = []
    for i in range(-ATM_RANGE, ATM_RANGE + 1):
        strike = int(atm + i * gap)
        symbols.append(f"NIFTY{exp_code}{strike}CE")
        symbols.append(f"NIFTY{exp_code}{strike}PE")

    logger.info(f"Option symbols: ATM={atm}, Expiry={expiry}, {len(symbols)} contracts")
    return symbols


def aggregate_candle(symbol: str, ticks: list) -> dict:
    """Build a 1-min candle from a list of tick dicts."""
    prices = [t["price"] for t in ticks if t.get("price", 0) > 0]
    volumes = [t.get("volume", 0) for t in ticks]
    ois = [t.get("oi", 0) for t in ticks]

    if not prices:
        return None

    ts = ticks[0].get("timestamp", datetime.now())
    minute_ts = ts.replace(second=0, microsecond=0)

    return {
        "timestamp": minute_ts,
        "symbol": symbol,
        "open": prices[0],
        "high": max(prices),
        "low": min(prices),
        "close": prices[-1],
        "volume": sum(volumes),
        "vwap": sum(p * v for p, v in zip(prices, volumes)) / max(sum(volumes), 1),
        "oi": ois[-1] if ois else 0,
    }


def _flush_price_cache():
    """Write live_price_cache to disk every second for Flask to read."""
    while running:
        try:
            tmp = LIVE_CACHE_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(live_price_cache))
            tmp.replace(LIVE_CACHE_FILE)
        except Exception:
            pass
        time.sleep(1)


def on_tick(tick: dict):
    """Process each incoming tick: buffer for candle aggregation."""
    global candle_buffer, last_minute, live_price_cache, last_tick_received_time
    last_tick_received_time = time.time()

    symbol = tick.get("symbol", "")
    ts = tick.get("timestamp", datetime.now())
    minute_ts = ts.replace(second=0, microsecond=0)

    # Map spot symbol back to futures symbol for consistency
    original_symbol = symbol
    symbol_map = {v: k for k, v in TD_INDEX_SPOT_SYMBOLS.items()}
    if symbol in symbol_map:
        # Store as NIFTY-I internally
        tick["symbol"] = TD_INDEX_FUTURES_SYMBOLS.get(symbol_map[symbol], symbol)
        symbol = tick["symbol"]

    # Update live price cache immediately (cache both original and remapped names)
    # Use wall-clock time as "ts" so Flask's freshness check sees when WE received the tick,
    # not the market timestamp (which can be hours old for illiquid options snapshots).
    price = tick.get("price", 0)
    if price and price > 0:
        bid = tick.get("bid_price") or price
        ask = tick.get("ask_price") or price
        entry = {
            "price": price,
            "bid": bid,
            "ask": ask,
            "ts": datetime.now().isoformat(),
        }
        live_price_cache[symbol] = entry
        if original_symbol != symbol:
            live_price_cache[original_symbol] = entry

    # Initialize buffer for new symbols
    if symbol not in candle_buffer:
        candle_buffer[symbol] = []
        last_minute[symbol] = minute_ts

    # New minute → aggregate previous minute into candle
    if minute_ts > last_minute.get(symbol, minute_ts):
        prev_ticks = candle_buffer[symbol]
        if prev_ticks:
            candle = aggregate_candle(symbol, prev_ticks)
            if candle:
                try:
                    upsert_candles(pd.DataFrame([candle]))
                except Exception as e:
                    logger.error(f"Failed to write candle for {symbol}: {e}")
        candle_buffer[symbol] = []
        last_minute[symbol] = minute_ts

    candle_buffer[symbol].append(tick)


def flush_remaining_candles():
    """Flush any remaining tick buffers into candles at shutdown."""
    for symbol, ticks in candle_buffer.items():
        if ticks:
            candle = aggregate_candle(symbol, ticks)
            if candle:
                try:
                    upsert_candles(pd.DataFrame([candle]))
                except Exception:
                    pass
    candle_buffer.clear()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global td, collector, running, last_tick_received_time

    parser = argparse.ArgumentParser(description="Automated tick collector")
    parser.add_argument("--test", action="store_true", help="Test connection only")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  AUTOMATED TICK DATA COLLECTOR")
    logger.info(f"  Date: {date.today()}")
    logger.info("=" * 60)

    if not is_trading_day():
        logger.info("Not a trading day (weekend). Exiting.")
        return

    # ── Test DB connection ────────────────────────────────────────────────
    try:
        engine = get_engine()
        import sqlalchemy
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("SELECT 1"))
        logger.info("Database connection OK.")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return

    # ── Initialize TrueData ──────────────────────────────────────────────
    td = TrueDataAdapter()
    if not td.authenticate():
        logger.error("TrueData authentication failed. Check credentials.")
        return
    logger.info("TrueData authenticated.")

    # ── Initialize tick collector ────────────────────────────────────────
    collector = TickCollector(buffer_size=200)

    if args.test:
        # Quick connection test
        logger.info("Testing WebSocket connection...")
        if td.ws_connect():
            logger.info("WebSocket connection successful!")
            td.ws_disconnect()
        else:
            logger.error("WebSocket connection failed.")
        return

    # ── Wait for market open ─────────────────────────────────────────────
    wait_for_market_open()

    if not running:
        return

    # ── Get current NIFTY price for option symbol selection ──────────────
    logger.info("Fetching current NIFTY price for option symbol selection...")
    last_bars = td.fetch_last_n_bars("NIFTY-I", n=1, interval="1min")
    if not last_bars.empty:
        current_price = float(last_bars.iloc[-1]["close"])
    else:
        # Fallback: use last known price from DB
        db_price = read_sql(
            "SELECT close FROM minute_candles WHERE symbol = 'NIFTY-I' "
            "ORDER BY timestamp DESC LIMIT 1"
        )
        current_price = float(db_price.iloc[0]["close"]) if not db_price.empty else 24000
    logger.info(f"Current NIFTY price: {current_price:.1f}")

    # ── Build subscription list ──────────────────────────────────────────
    subscribe_symbols = []

    # Index futures (for tick data)
    for sym_key in SYMBOLS:
        futures_sym = TD_INDEX_FUTURES_SYMBOLS.get(sym_key)
        if futures_sym:
            subscribe_symbols.append(futures_sym)

    # ATM options
    option_symbols = get_option_symbols_for_today(current_price)
    subscribe_symbols.extend(option_symbols)

    # Respect max symbol limit
    if len(subscribe_symbols) > MAX_SYMBOLS:
        logger.warning(f"Trimming {len(subscribe_symbols)} symbols to {MAX_SYMBOLS}")
        subscribe_symbols = subscribe_symbols[:MAX_SYMBOLS]

    logger.info(f"Subscribing to {len(subscribe_symbols)} symbols")

    # ── Connect WebSocket ────────────────────────────────────────────────
    if not td.ws_connect():
        logger.error("WebSocket connection failed. Exiting.")
        return

    # Subscribe
    td.ws_subscribe(subscribe_symbols)

    # Start streaming with our tick handler
    def combined_handler(tick):
        """Feed tick to both collector (DB persistence) and candle aggregator."""
        collector.on_tick(tick)
        on_tick(tick)

    td.ws_start_streaming(combined_handler)

    # Start background thread to flush live price cache to disk every second
    cache_thread = threading.Thread(target=_flush_price_cache, daemon=True)
    cache_thread.start()
    logger.info(f"Live price cache flushing to {LIVE_CACHE_FILE}")

    logger.info("Streaming started. Collecting ticks until market close...")
    print(f"  Collecting ticks... (log: {log_file})")
    print(f"  Market close: {MARKET_CLOSE_HOUR}:{MARKET_CLOSE_MINUTE:02d} IST")
    print(f"  Press Ctrl+C to stop early.")

    # ── Main loop: wait until market close ───────────────────────────────
    close_time = market_close_time() + timedelta(minutes=1)  # 1 min buffer
    tick_count_last = 0
    WATCHDOG_TIMEOUT = 60  # seconds — force reconnect if no real tick received

    # Track subscribed ATM for dynamic re-subscription when NIFTY drifts
    subscribed_atm = round(current_price / 50) * 50
    subscribed_symbols_set = set(subscribe_symbols)
    last_atm_check = time.time()
    ATM_RECHECK_SECS = 120  # check every 2 minutes

    while running and datetime.now() < close_time:
        time.sleep(30)

        # ── Dynamic ATM re-subscription ───────────────────────────────────
        # If NIFTY moves 100+ pts (2 strikes) from the subscribed ATM, subscribe
        # to the new ATM range so open positions always have live tick prices.
        if time.time() - last_atm_check >= ATM_RECHECK_SECS:
            live_price = live_price_cache.get("NIFTY-I", {}).get("price", 0)
            if live_price > 0:
                new_atm = round(live_price / 50) * 50
                if abs(new_atm - subscribed_atm) >= 100:
                    new_opt_symbols = get_option_symbols_for_today(live_price)
                    to_add = [s for s in new_opt_symbols if s not in subscribed_symbols_set]
                    if to_add:
                        logger.info(
                            f"NIFTY ATM drifted {subscribed_atm} → {new_atm} "
                            f"(live={live_price:.1f}). Adding {len(to_add)} new symbols: {to_add}"
                        )
                        td.ws_subscribe(to_add)
                        subscribed_symbols_set.update(to_add)
                        subscribed_atm = new_atm
            last_atm_check = time.time()

        # ── Watchdog: detect silent WebSocket stall ───────────────────────
        # The _stream_loop in truedata_adapter handles auto-reconnect on exceptions.
        # We just need to force-close the socket; _stream_loop will detect the error
        # and reconnect + re-subscribe automatically.
        secs_since_tick = time.time() - last_tick_received_time
        if secs_since_tick > WATCHDOG_TIMEOUT:
            logger.warning(
                f"No ticks for {secs_since_tick:.0f}s — force-closing socket; "
                "stream loop will auto-reconnect."
            )
            try:
                if td._ws:
                    td._ws.close()
            except Exception as e:
                logger.debug(f"Watchdog socket close error: {e}")
            last_tick_received_time = time.time()  # reset to avoid tight retry loop

        # Periodic status
        current_count = len(collector._buffer)
        if current_count > 0 or tick_count_last > 0:
            logger.info(
                f"Status: buffer={current_count} ticks, "
                f"candle_symbols={len(candle_buffer)}, "
                f"secs_since_last_tick={secs_since_tick:.0f}, "
                f"time={datetime.now().strftime('%H:%M:%S')}"
            )
        tick_count_last = current_count

    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("Market closed. Shutting down...")

    td.ws_stop_streaming()
    td.ws_disconnect()

    # Flush remaining data
    collector.flush()
    flush_remaining_candles()

    # ── Summary ──────────────────────────────────────────────────────────
    # Count what we collected today
    today_str = date.today().isoformat()
    tick_count = read_sql(
        "SELECT COUNT(*) as cnt FROM tick_data WHERE timestamp::date = :dt",
        {"dt": today_str},
    )
    candle_count = read_sql(
        "SELECT COUNT(DISTINCT symbol) as syms, COUNT(*) as bars "
        "FROM minute_candles WHERE timestamp::date = :dt",
        {"dt": today_str},
    )

    tc = int(tick_count.iloc[0]["cnt"]) if not tick_count.empty else 0
    if not candle_count.empty:
        cs = int(candle_count.iloc[0]["syms"])
        cb = int(candle_count.iloc[0]["bars"])
    else:
        cs, cb = 0, 0

    logger.info("=" * 60)
    logger.info("  COLLECTION SUMMARY")
    logger.info(f"  Date:           {today_str}")
    logger.info(f"  Ticks stored:   {tc:,}")
    logger.info(f"  Candle symbols: {cs}")
    logger.info(f"  Candle bars:    {cb:,}")
    logger.info("=" * 60)

    print(f"\n  Done! Collected {tc:,} ticks, {cb:,} candles for {cs} symbols.")
    print(f"  Log: {log_file}")


if __name__ == "__main__":
    main()
