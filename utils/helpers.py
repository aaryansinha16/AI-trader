"""
Utility Helpers
───────────────
Common utility functions used across the system.
"""

from datetime import datetime, time as dt_time

from config.settings import (
    MARKET_OPEN_HOUR,
    MARKET_OPEN_MINUTE,
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MINUTE,
)


def is_market_open(now: datetime = None) -> bool:
    """Check if NSE market is currently open (9:15 AM – 3:30 PM IST, Mon–Fri)."""
    now = now or datetime.now()

    # Weekend check
    if now.weekday() >= 5:
        return False

    market_open = dt_time(MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE)
    market_close = dt_time(MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE)

    return market_open <= now.time() <= market_close


def round_to_tick(price: float, tick_size: float = 0.05) -> float:
    """Round price to nearest tick size (NSE options tick = 0.05)."""
    return round(round(price / tick_size) * tick_size, 2)


def calculate_stop_loss(entry: float, atr: float, multiplier: float = 1.5) -> float:
    """Calculate stop loss based on ATR."""
    return round(entry - (atr * multiplier), 2)


def calculate_target(entry: float, atr: float, multiplier: float = 2.0) -> float:
    """Calculate target based on ATR (risk-reward)."""
    return round(entry + (atr * multiplier), 2)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division with zero-safety."""
    if denominator == 0 or denominator is None:
        return default
    return numerator / denominator