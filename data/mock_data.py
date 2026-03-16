"""
Mock Data Generator
───────────────────
Generates realistic synthetic market data for development and testing
before live TrueData / Kite connections are available.

Produces data matching the exact schema used by the real system:
  - Tick data (sub-second, with bid/ask/oi)
  - 1-minute OHLCV candles (6 months worth ≈ 28,000 rows per symbol)
  - Option chain snapshots
  - 5-day tick history (for Micro Model training)

All prices are realistic for NSE NIFTY (~22000-23000) and BANKNIFTY (~47000-49000).
"""

import random
from datetime import datetime, timedelta, time as dt_time
from typing import List, Optional

import numpy as np
import pandas as pd

from config.settings import SYMBOLS
from utils.logger import get_logger

logger = get_logger("mock_data")

# Realistic price ranges for NSE instruments
_BASE_PRICES = {
    "NIFTY": 22500.0,
    "BANKNIFTY": 48000.0,
}

_MARKET_OPEN = dt_time(9, 15)
_MARKET_CLOSE = dt_time(15, 30)
_TRADING_MINUTES = 375  # 6h 15m


def generate_mock_minute_bars(
    symbol: str = "NIFTY",
    trading_days: int = 125,
    start_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Generate ~6 months of 1-minute OHLCV data.
    Used for Macro ML Model training.

    Simulates realistic intraday price movement with:
      - Opening gap
      - Morning volatility
      - Lunch doldrums
      - Afternoon trend
      - Volume profile (U-shaped)
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=int(trading_days * 1.5))

    base_price = _BASE_PRICES.get(symbol, 22500.0)
    records = []

    current_date = start_date
    days_generated = 0

    while days_generated < trading_days:
        # Skip weekends
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue

        # Daily opening gap (-0.5% to +0.5%)
        daily_gap = random.uniform(-0.005, 0.005)
        price = base_price * (1 + daily_gap)

        # Daily trend bias
        daily_trend = random.uniform(-0.003, 0.003)

        for minute in range(_TRADING_MINUTES):
            ts = datetime.combine(
                current_date.date(),
                _MARKET_OPEN,
            ) + timedelta(minutes=minute)

            # Volatility varies through the day (higher at open/close)
            hour_frac = minute / _TRADING_MINUTES
            volatility = 0.001 * (
                1.5 if hour_frac < 0.1 or hour_frac > 0.9
                else 0.6 if 0.3 < hour_frac < 0.6
                else 1.0
            )

            # Price walk with trend
            change = np.random.normal(daily_trend / _TRADING_MINUTES, volatility)
            price *= (1 + change)

            # Generate OHLC from the minute
            intra_vol = volatility * 0.5
            high = price * (1 + abs(np.random.normal(0, intra_vol)))
            low = price * (1 - abs(np.random.normal(0, intra_vol)))
            open_p = price * (1 + np.random.normal(0, intra_vol * 0.3))
            close_p = price

            # U-shaped volume profile
            vol_mult = 2.0 if hour_frac < 0.1 or hour_frac > 0.85 else 0.7
            volume = int(np.random.exponential(5000) * vol_mult)

            records.append({
                "timestamp": ts,
                "symbol": symbol,
                "open": round(open_p, 2),
                "high": round(max(high, open_p, close_p), 2),
                "low": round(min(low, open_p, close_p), 2),
                "close": round(close_p, 2),
                "volume": max(volume, 100),
            })

        # Drift the base price for next day
        base_price = price
        days_generated += 1
        current_date += timedelta(days=1)

    df = pd.DataFrame(records)
    logger.info(
        f"Generated {len(df)} mock minute bars for {symbol} "
        f"({days_generated} trading days)."
    )
    return df


def generate_mock_tick_data(
    symbol: str = "NIFTY",
    trading_days: int = 5,
    ticks_per_second: int = 3,
    start_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Generate ~5 days of tick data for Microstructure Model training.

    Each tick includes: price, volume, bid/ask, oi
    Simulates realistic microstructure: bid-ask bounce, spread variation,
    volume clustering, OI drift.
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=7)

    base_price = _BASE_PRICES.get(symbol, 22500.0)
    records = []

    current_date = start_date
    days_generated = 0

    while days_generated < trading_days:
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue

        price = base_price * (1 + random.uniform(-0.003, 0.003))
        oi_base = random.randint(5_000_000, 15_000_000)

        total_seconds = _TRADING_MINUTES * 60

        for sec in range(total_seconds):
            n_ticks = random.randint(1, ticks_per_second * 2)

            for t in range(n_ticks):
                ms_offset = random.randint(0, 999)
                ts = (
                    datetime.combine(current_date.date(), _MARKET_OPEN)
                    + timedelta(seconds=sec, milliseconds=ms_offset)
                )

                # Microstructure: bid-ask bounce
                spread = random.uniform(0.5, 3.0)
                tick_change = np.random.normal(0, 0.5)
                price += tick_change
                bid = round(price - spread / 2, 2)
                ask = round(price + spread / 2, 2)

                # Volume: mostly small, occasional large trades
                if random.random() < 0.05:
                    vol = random.randint(500, 5000)  # Large trade
                else:
                    vol = random.randint(1, 100)

                # OI drifts slowly
                oi_base += random.randint(-100, 100)

                records.append({
                    "timestamp": ts,
                    "symbol": symbol,
                    "price": round(price, 2),
                    "volume": vol,
                    "bid_price": bid,
                    "ask_price": ask,
                    "bid_qty": random.randint(50, 2000),
                    "ask_qty": random.randint(50, 2000),
                    "oi": max(oi_base, 0),
                })

        base_price = price
        days_generated += 1
        current_date += timedelta(days=1)

    df = pd.DataFrame(records)
    logger.info(
        f"Generated {len(df)} mock ticks for {symbol} "
        f"({days_generated} trading days)."
    )
    return df


def generate_mock_option_chain(
    symbol: str = "NIFTY",
    spot_price: float = 22500.0,
    num_strikes: int = 20,
    timestamp: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Generate a realistic option chain snapshot.
    Strikes centered around ATM with CE and PE for each.
    """
    timestamp = timestamp or datetime.now()
    strike_gap = 50 if symbol == "NIFTY" else 100
    atm_strike = round(spot_price / strike_gap) * strike_gap

    strikes = [
        atm_strike + (i - num_strikes // 2) * strike_gap
        for i in range(num_strikes)
    ]

    records = []
    expiry = (timestamp + timedelta(days=(3 - timestamp.weekday()) % 7)).date()

    for strike in strikes:
        for opt_type in ["CE", "PE"]:
            moneyness = (spot_price - strike) / spot_price
            if opt_type == "PE":
                moneyness = -moneyness

            # Approximate premium based on moneyness
            itm = max(0, moneyness * spot_price)
            time_value = random.uniform(20, 150)
            ltp = round(itm + time_value, 2)

            # OI pattern: highest near ATM
            distance = abs(strike - atm_strike) / strike_gap
            oi = int(np.random.exponential(500000) / (1 + distance))
            oi_change = random.randint(-50000, 50000)

            # IV smile
            iv = 15 + distance * 2 + random.uniform(-1, 1)

            records.append({
                "timestamp": timestamp,
                "symbol": symbol,
                "expiry": expiry,
                "strike": strike,
                "option_type": opt_type,
                "ltp": ltp,
                "volume": random.randint(100, 50000),
                "oi": max(oi, 0),
                "oi_change": oi_change,
                "iv": round(iv, 2),
                "bid_price": round(ltp * 0.98, 2),
                "ask_price": round(ltp * 1.02, 2),
                "delta": round(random.uniform(-1, 1), 4),
                "gamma": round(random.uniform(0, 0.01), 6),
                "theta": round(random.uniform(-50, -1), 2),
                "vega": round(random.uniform(1, 30), 2),
            })

    df = pd.DataFrame(records)
    logger.info(f"Generated option chain: {len(df)} contracts for {symbol}.")
    return df


def generate_all_mock_data() -> dict:
    """
    Generate a complete mock dataset for all symbols.

    Returns dict with keys:
      - minute_bars: 6 months of 1m data (for Macro Model)
      - ticks: 5 days of tick data (for Micro Model)
      - option_chain: sample option chain snapshot
    """
    all_minutes = []
    all_ticks = []
    all_options = []

    for symbol in SYMBOLS:
        minute_df = generate_mock_minute_bars(symbol, trading_days=125)
        all_minutes.append(minute_df)

        tick_df = generate_mock_tick_data(symbol, trading_days=5, ticks_per_second=2)
        all_ticks.append(tick_df)

        spot = minute_df["close"].iloc[-1] if not minute_df.empty else _BASE_PRICES[symbol]
        option_df = generate_mock_option_chain(symbol, spot_price=spot)
        all_options.append(option_df)

    result = {
        "minute_bars": pd.concat(all_minutes, ignore_index=True),
        "ticks": pd.concat(all_ticks, ignore_index=True),
        "option_chain": pd.concat(all_options, ignore_index=True),
    }

    logger.info(
        f"Full mock dataset: "
        f"{len(result['minute_bars'])} minute bars, "
        f"{len(result['ticks'])} ticks, "
        f"{len(result['option_chain'])} option contracts."
    )
    return result
