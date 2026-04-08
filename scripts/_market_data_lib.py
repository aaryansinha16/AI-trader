"""
Shared helpers for fetching, gap-detecting, and upserting market data
from TrueData REST.

Used by:
  - scripts/seed_other_symbols.py     (one-shot 6mo historical seed)
  - scripts/eod_collect_other_symbols.py (daily incremental EOD runner)

Key conventions:
  - minute_candles uses upsert_candles (has UNIQUE (timestamp, symbol))
  - tick_data has NO unique constraint, so we DELETE then INSERT per
    (symbol, date). This makes re-runs idempotent without duplicates.
  - Timestamps in DB are stored as IST mislabelled +00:00 for legacy
    compatibility. fetch_historical_ticks/bars return naive IST values
    which we keep as-is (don't add tzinfo).
  - All operations are gap-aware: no-ops when data already complete.
"""

from __future__ import annotations

import sys
import os
from datetime import datetime, date, timedelta
from typing import Iterable

import pandas as pd
import requests
from sqlalchemy import text

# Make project root importable when called as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.truedata_adapter import TrueDataAdapter
from database.db import engine, read_sql, upsert_candles
from config.settings import (
    TD_INDEX_SPOT_SYMBOLS,
    TD_INDEX_FUTURES_SYMBOLS,
    STRIKE_GAP,
    EXPIRY_CADENCE,
)
from utils.logger import get_logger

logger = get_logger("market_data_lib")

# Trading session (IST)
MARKET_OPEN_HM = (9, 15)
MARKET_CLOSE_HM = (15, 30)

# A "complete" day for futures/spot is ~375 1-min bars (9:15→15:30).
# We accept anything ≥ 250 as good enough (covers data feed glitches).
FULL_CANDLES_PER_DAY = 375
MIN_CANDLES_PER_DAY = 250

# Options: an illiquid OTM strike can legitimately trade <50 candles per day.
# Use this threshold instead of MIN_CANDLES_PER_DAY when checking option contracts.
# We only flag a strike "missing" if it has literally zero candles — that's the only
# case we can act on (the strike was never fetched, vs. just lightly traded).
MIN_OPTION_CANDLES_PER_DAY = 1

# Tick coverage threshold for futures: ~5,000 ticks/day is normal NIFTY-I.
# Options can be much sparser, so the option tick check uses a smaller floor.
MIN_FUTURES_TICKS_PER_DAY = 3000
MIN_OPTION_TICKS_PER_DAY = 1   # only refetch when truly empty


# ─── Date helpers ────────────────────────────────────────────────────────────


def is_trading_day(d: date) -> bool:
    """Mon–Fri (NSE holidays not handled — caller can skip empty fetches)."""
    return d.weekday() < 5


def trading_days_back(n: int, end: date | None = None, include_today: bool = False) -> list[date]:
    """
    Return the last `n` trading days (most recent first), inclusive of end.

    By default `today` is EXCLUDED — the live collector handles today, and
    a mid-session fetch would only get partial data. Pass include_today=True
    to override (used by EOD runner that runs after market close).
    """
    end = end or date.today()
    if not include_today:
        end -= timedelta(days=1)
    out: list[date] = []
    d = end
    while len(out) < n:
        if is_trading_day(d):
            out.append(d)
        d -= timedelta(days=1)
    return out


def session_window(d: date) -> tuple[datetime, datetime]:
    return (
        datetime.combine(d, datetime.min.time().replace(hour=MARKET_OPEN_HM[0], minute=MARKET_OPEN_HM[1])),
        datetime.combine(d, datetime.min.time().replace(hour=MARKET_CLOSE_HM[0], minute=MARKET_CLOSE_HM[1])),
    )


# ─── TrueData expiry discovery ───────────────────────────────────────────────


def get_current_monthly_expiry(symbol_underlying: str, td: TrueDataAdapter) -> date | None:
    """
    Fetch the next available expiry for a symbol via TrueData REST.
    Returns the *earliest expiry that is >= today*.

    Endpoint: https://history.truedata.in/getSymbolExpiryList
    """
    if not td.authenticate():
        return None

    try:
        r = requests.get(
            "https://history.truedata.in/getSymbolExpiryList",
            params={"symbol": symbol_underlying, "response": "csv"},
            headers=td._auth_header(),
            timeout=15,
        )
        r.raise_for_status()
        lines = [ln.strip() for ln in r.text.strip().split("\n") if ln.strip()]
        if not lines or lines[0].lower() != "expiry":
            return None
        today = date.today()
        for ln in lines[1:]:
            try:
                d = datetime.strptime(ln, "%Y-%m-%d").date()
                # Skip placeholder dates (TrueData uses 2099-12-31 for non-expiring)
                if d.year >= 2099:
                    continue
                if d >= today:
                    return d
            except ValueError:
                continue
    except Exception as e:
        logger.warning(f"  expiry list fetch failed for {symbol_underlying}: {e}")
    return None


# ─── ATM and option symbol building ──────────────────────────────────────────


def get_atm_strike(price: float, gap: int) -> int:
    return int(round(price / gap) * gap)


def build_option_symbols(
    underlying: str,
    atm: int,
    expiry: date,
    atm_range: int,
    gap: int,
) -> list[str]:
    """E.g. underlying='BANKNIFTY', atm=55000, expiry=2026-04-28, range=3
    → [BANKNIFTY26042854700CE, BANKNIFTY26042854700PE, ..., BANKNIFTY26042855300PE]
    """
    exp_code = expiry.strftime("%y%m%d")
    syms: list[str] = []
    for i in range(-atm_range, atm_range + 1):
        strike = atm + i * gap
        syms.append(f"{underlying}{exp_code}{strike}CE")
        syms.append(f"{underlying}{exp_code}{strike}PE")
    return syms


def get_futures_close_on(symbol: str, day: date) -> float | None:
    """Return the last close of `symbol` on `day` from minute_candles, or None."""
    df = read_sql(
        """
        SELECT close FROM minute_candles
        WHERE symbol = :sym
          AND timestamp >= :start AND timestamp < :end
        ORDER BY timestamp DESC LIMIT 1
        """,
        {
            "sym": symbol,
            "start": datetime.combine(day, datetime.min.time()),
            "end": datetime.combine(day + timedelta(days=1), datetime.min.time()),
        },
    )
    return float(df.iloc[0]["close"]) if not df.empty else None


# ─── Coverage / gap detection ────────────────────────────────────────────────


def candle_coverage(symbol: str, days: list[date]) -> dict[date, int]:
    """Return {day: candle_count} for the given symbol/days."""
    if not days:
        return {}
    df = read_sql(
        """
        SELECT DATE(timestamp) AS dt, COUNT(*) AS n
        FROM minute_candles
        WHERE symbol = :sym
          AND timestamp >= :start
          AND timestamp < :end
        GROUP BY 1
        """,
        {
            "sym": symbol,
            "start": datetime.combine(min(days), datetime.min.time()),
            "end": datetime.combine(max(days) + timedelta(days=1), datetime.min.time()),
        },
    )
    counts = {row["dt"] if isinstance(row["dt"], date) else row["dt"].date(): int(row["n"])
              for _, row in df.iterrows()}
    return {d: counts.get(d, 0) for d in days}


def tick_coverage(symbol: str, days: list[date]) -> dict[date, int]:
    """Return {day: tick_count} for the given symbol/days."""
    if not days:
        return {}
    df = read_sql(
        """
        SELECT DATE(timestamp) AS dt, COUNT(*) AS n
        FROM tick_data
        WHERE symbol = :sym
          AND timestamp >= :start
          AND timestamp < :end
        GROUP BY 1
        """,
        {
            "sym": symbol,
            "start": datetime.combine(min(days), datetime.min.time()),
            "end": datetime.combine(max(days) + timedelta(days=1), datetime.min.time()),
        },
    )
    counts = {row["dt"] if isinstance(row["dt"], date) else row["dt"].date(): int(row["n"])
              for _, row in df.iterrows()}
    return {d: counts.get(d, 0) for d in days}


def find_missing_candle_days(symbol: str, days: list[date], min_count: int = MIN_CANDLES_PER_DAY) -> list[date]:
    """Return weekdays where the symbol has < min_count candles."""
    cov = candle_coverage(symbol, days)
    return [d for d in days if is_trading_day(d) and cov.get(d, 0) < min_count]


def find_missing_tick_days(symbol: str, days: list[date], min_count: int) -> list[date]:
    """Return weekdays where the symbol has < min_count ticks."""
    cov = tick_coverage(symbol, days)
    return [d for d in days if is_trading_day(d) and cov.get(d, 0) < min_count]


# ─── Persistence ─────────────────────────────────────────────────────────────


def upsert_minute_candles(df: pd.DataFrame, symbol: str) -> int:
    """Normalize candles and upsert via the existing upsert_candles helper."""
    if df.empty:
        return 0
    out = df.copy()
    out["symbol"] = symbol
    out["timestamp"] = pd.to_datetime(out["timestamp"])

    # Required columns for minute_candles
    required = ["timestamp", "symbol", "open", "high", "low", "close", "volume", "vwap", "oi"]
    if "vwap" not in out.columns:
        out["vwap"] = (out["high"] + out["low"] + out["close"]) / 3.0
    if "oi" not in out.columns:
        out["oi"] = 0
    out = out[[c for c in required if c in out.columns]]
    return upsert_candles(out, table="minute_candles")


def replace_day_ticks(df: pd.DataFrame, symbol: str, day: date) -> int:
    """
    Replace one day's ticks for a symbol. Used because tick_data has no unique
    constraint, so we DELETE-then-INSERT to keep re-runs idempotent.
    """
    if df.empty:
        return 0
    out = df.copy()
    out["symbol"] = symbol
    out["timestamp"] = pd.to_datetime(out["timestamp"])

    required = ["timestamp", "symbol", "price", "volume", "oi",
                "bid_price", "ask_price", "bid_qty", "ask_qty"]
    for col in required:
        if col not in out.columns:
            if col in ("bid_price", "ask_price"):
                out[col] = out.get("price", 0)
            else:
                out[col] = 0
    out = out[required]

    # Delete the day's existing rows for this symbol, then insert fresh ones
    start = datetime.combine(day, datetime.min.time())
    end = datetime.combine(day + timedelta(days=1), datetime.min.time())
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM tick_data WHERE symbol = :sym AND timestamp >= :start AND timestamp < :end"),
            {"sym": symbol, "start": start, "end": end},
        )
        # Bulk insert (no method='multi' — slower but avoids parameter limits on huge days)
        out.to_sql("tick_data", conn, if_exists="append", index=False, method=None, chunksize=2000)
    return len(out)


# ─── Fetch + persist primitives ──────────────────────────────────────────────


def seed_candles_for_symbol(
    td: TrueDataAdapter,
    truedata_symbol: str,
    days: int = 180,
) -> int:
    """Fetch `days` of 1-min bars in chunks and upsert. Returns rows inserted."""
    end = datetime.now()
    logger.info(f"  [{truedata_symbol}] fetching {days}d of 1-min bars (chunked)")
    df = td.fetch_historical_minute_bars(truedata_symbol, days=days, end_date=end)
    if df.empty:
        logger.warning(f"  [{truedata_symbol}] no candles returned")
        return 0
    inserted = upsert_minute_candles(df, truedata_symbol)
    logger.info(f"  [{truedata_symbol}] {len(df)} candles fetched, {inserted} new rows upserted")
    return inserted


def fill_candle_day(td: TrueDataAdapter, truedata_symbol: str, day: date) -> int:
    """Fetch a single day of 1-min bars and upsert. Returns rows inserted."""
    start, end = session_window(day)
    df = td.fetch_historical_bars(truedata_symbol, start, end, "1min")
    if df.empty:
        return 0
    return upsert_minute_candles(df, truedata_symbol)


def fill_tick_day(td: TrueDataAdapter, truedata_symbol: str, day: date) -> int:
    """Fetch and replace one day of ticks. Returns rows inserted."""
    start, end = session_window(day)
    df = td.fetch_historical_ticks(truedata_symbol, start=start, end=end)
    if df.empty:
        return 0
    return replace_day_ticks(df, truedata_symbol, day)
