"""
Option Contract Resolver
─────────────────────────
Resolves ATM option contracts from the DB for a given index price and timestamp.
Used by the backtest engine to trade actual option premiums instead of delta approximations.
"""

import re
from datetime import date, datetime
from typing import Optional, Tuple

import pandas as pd

from database.db import read_sql
from utils.logger import get_logger

logger = get_logger("option_resolver")

# All historical expiry dates from our option data
_EXPIRY_DATES = None
_OPTION_PREMIUM_CACHE = {}


def _load_expiry_dates():
    """Load all expiry dates from option symbols in DB (cached)."""
    global _EXPIRY_DATES
    if _EXPIRY_DATES is not None:
        return _EXPIRY_DATES

    syms = read_sql("""
        SELECT DISTINCT SUBSTRING(symbol FROM 6 FOR 6) as exp_code
        FROM minute_candles
        WHERE symbol LIKE 'NIFTY______%%CE'
        ORDER BY 1
    """)
    dates = []
    for _, r in syms.iterrows():
        try:
            d = datetime.strptime(r["exp_code"], "%y%m%d").date()
            dates.append(d)
        except ValueError:
            pass
    _EXPIRY_DATES = sorted(dates)
    logger.info(f"Loaded {len(_EXPIRY_DATES)} expiry dates from DB")
    return _EXPIRY_DATES


def get_nearest_expiry(ref_date: date) -> Optional[date]:
    """Find the nearest expiry on or after ref_date."""
    expiries = _load_expiry_dates()
    for e in expiries:
        if e >= ref_date:
            return e
    return expiries[-1] if expiries else None


def get_days_to_expiry(ref_date: date, expiry: date) -> int:
    """Trading days to expiry (approximate, weekdays only)."""
    if ref_date >= expiry:
        return 0
    from datetime import timedelta
    days = 0
    current = ref_date
    while current < expiry:
        current += timedelta(days=1)
        if current.weekday() < 5:
            days += 1
    return days


def build_option_symbol(expiry: date, strike: int, opt_type: str) -> str:
    """Build NIFTY option symbol string."""
    exp_code = expiry.strftime("%y%m%d")
    return f"NIFTY{exp_code}{strike}{opt_type}"


def get_atm_strike(index_price: float, strike_gap: int = 50) -> int:
    """Round to nearest ATM strike."""
    return int(round(index_price / strike_gap) * strike_gap)


def load_option_premiums_for_day(symbol: str, trading_date: date) -> pd.DataFrame:
    """Load all 1-min premium bars for an option symbol on a given day."""
    cache_key = (symbol, str(trading_date))
    if cache_key in _OPTION_PREMIUM_CACHE:
        return _OPTION_PREMIUM_CACHE[cache_key]

    df = read_sql(
        "SELECT timestamp, open, high, low, close as premium, volume, oi "
        "FROM minute_candles "
        "WHERE symbol = :sym AND timestamp::date = :dt "
        "ORDER BY timestamp",
        {"sym": symbol, "dt": str(trading_date)},
    )
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    _OPTION_PREMIUM_CACHE[cache_key] = df
    return df


def preload_option_premiums(expiry_dates: list, index_df: pd.DataFrame, strike_gap: int = 50):
    """
    Preload all option premium data needed for a backtest run.
    Returns a dict: {timestamp -> {CE_premium_df, PE_premium_df, expiry, atm, symbol_ce, symbol_pe}}
    
    This is much faster than loading per-trade.
    """
    logger.info("Preloading option premium data for backtest...")

    # For each trading day in the index data, determine expiry + ATM
    index_df = index_df.copy()
    index_df["timestamp"] = pd.to_datetime(index_df["timestamp"])
    index_df["date"] = index_df["timestamp"].dt.date

    # Group by day to batch load
    day_info = {}
    for day, group in index_df.groupby("date"):
        first_close = group.iloc[0]["close"]
        atm = get_atm_strike(first_close, strike_gap)
        expiry = get_nearest_expiry(day)
        if expiry is None:
            continue
        day_info[day] = {"atm": atm, "expiry": expiry}

    # Load all needed option bars in bulk
    all_option_bars = {}
    loaded = 0
    for day, info in day_info.items():
        exp = info["expiry"]
        atm = info["atm"]
        for opt_type in ["CE", "PE"]:
            sym = build_option_symbol(exp, atm, opt_type)
            key = (sym, str(day))
            if key not in _OPTION_PREMIUM_CACHE:
                df = load_option_premiums_for_day(sym, day)
                loaded += 1

    logger.info(f"Preloaded {loaded} option-day combinations, cache size: {len(_OPTION_PREMIUM_CACHE)}")
    return day_info


def resolve_option_at_entry(
    index_price: float,
    timestamp: pd.Timestamp,
    direction: str,
    strike_gap: int = 50,
) -> Optional[dict]:
    """
    Resolve the ATM option contract at trade entry.
    
    Returns dict with:
      symbol, expiry, strike, premium (entry price), premium_df (for tracking)
    """
    ref_date = timestamp.date() if hasattr(timestamp, "date") else timestamp
    expiry = get_nearest_expiry(ref_date)
    if expiry is None:
        return None

    atm = get_atm_strike(index_price, strike_gap)
    opt_type = "CE" if direction == "CALL" else "PE"

    # Try ATM first, then nearby strikes if no data
    # Search outward: ATM, ±50, ±100, ... ±500
    premium_df = pd.DataFrame()
    actual_strike = atm
    offsets = [0]
    for i in range(1, 11):
        offsets.extend([i * strike_gap, -i * strike_gap])
    for offset in offsets:
        trial_strike = atm + offset
        sym = build_option_symbol(expiry, trial_strike, opt_type)
        pdf = load_option_premiums_for_day(sym, ref_date)
        if not pdf.empty:
            premium_df = pdf
            actual_strike = trial_strike
            break

    if premium_df.empty:
        return None

    symbol = build_option_symbol(expiry, actual_strike, opt_type)

    # Find the premium at entry timestamp
    ts = pd.to_datetime(timestamp)
    mask = (premium_df["timestamp"] - ts).abs() <= pd.Timedelta(minutes=1)
    matching = premium_df[mask]
    if matching.empty:
        return None

    entry_premium = float(matching.iloc[0]["premium"])
    dte = get_days_to_expiry(ref_date, expiry)

    return {
        "symbol": symbol,
        "expiry": expiry,
        "strike": atm,
        "opt_type": opt_type,
        "entry_premium": entry_premium,
        "dte": dte,
        "premium_df": premium_df,
    }


def clear_cache():
    """Clear the premium cache (call between backtest runs)."""
    global _OPTION_PREMIUM_CACHE
    _OPTION_PREMIUM_CACHE = {}
