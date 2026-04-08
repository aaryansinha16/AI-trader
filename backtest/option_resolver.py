"""
Option Contract Resolver
─────────────────────────
Resolves ATM option contracts from the DB for a given index price and timestamp.
Used by the backtest engine to trade actual option premiums instead of delta approximations.
"""

import re
import time
from datetime import date, datetime, timedelta
from typing import Optional, Tuple

import pandas as pd

from database.db import read_sql
from utils.logger import get_logger
from strategy.vol_surface import VolSurfaceModel

logger = get_logger("option_resolver")

# All historical expiry dates from our option data (DB-derived)
_EXPIRY_DATES = None
_OPTION_PREMIUM_CACHE = {}

# Live TrueData expiry list (cached for 1 hour to avoid hammering REST)
_LIVE_EXPIRIES: list[date] | None = None
_LIVE_EXPIRIES_FETCHED_AT: float = 0.0
_LIVE_EXPIRIES_TTL = 3600.0  # 1 hour


def _load_expiry_dates(force_reload: bool = False):
    """Load all historical expiry dates from option symbols in DB (cached)."""
    global _EXPIRY_DATES
    if _EXPIRY_DATES is not None and not force_reload:
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
    logger.info(f"Loaded {len(_EXPIRY_DATES)} historical expiry dates from DB")
    return _EXPIRY_DATES


def _fetch_live_expiries_from_truedata() -> list[date]:
    """
    Pull the authoritative upcoming-expiry list from TrueData REST.

    This is the only way to get the right answer because NSE periodically
    moves NIFTY weekly expiry day (Thu → Wed → Tue → Mon historically), and
    individual weeks may be holiday-adjusted. Hard-coding a weekday will
    eventually break.

    Cached for 1 hour. Returns empty list on failure (caller falls back to
    DB-derived expiries).
    """
    global _LIVE_EXPIRIES, _LIVE_EXPIRIES_FETCHED_AT
    now = time.time()
    if _LIVE_EXPIRIES is not None and (now - _LIVE_EXPIRIES_FETCHED_AT) < _LIVE_EXPIRIES_TTL:
        return _LIVE_EXPIRIES
    try:
        import requests
        from data.truedata_adapter import TrueDataAdapter
        td = TrueDataAdapter()
        if not td.authenticate():
            return _LIVE_EXPIRIES or []
        r = requests.get(
            "https://history.truedata.in/getSymbolExpiryList",
            params={"symbol": "NIFTY", "response": "csv"},
            headers=td._auth_header(),
            timeout=15,
        )
        r.raise_for_status()
        out: list[date] = []
        for ln in r.text.strip().split("\n")[1:]:  # skip "expiry" header
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = datetime.strptime(ln, "%Y-%m-%d").date()
                if d.year >= 2099:  # placeholder
                    continue
                out.append(d)
            except ValueError:
                continue
        out.sort()
        _LIVE_EXPIRIES = out
        _LIVE_EXPIRIES_FETCHED_AT = now
        logger.info(f"Fetched {len(out)} live NIFTY expiries from TrueData REST: next 5 = {out[:5]}")
        return out
    except Exception as e:
        logger.warning(f"Live expiry fetch failed: {e}")
        return _LIVE_EXPIRIES or []


def get_nearest_expiry(ref_date: date) -> Optional[date]:
    """
    Find the nearest NIFTY expiry on or after ref_date.

    Two distinct paths:
      • **Historical / backtest dates** (ref_date < today): look up the
        DB-derived list. This is what was actually traded on that date,
        which is what a faithful backtest needs. Live REST has no
        knowledge of contracts that already expired.
      • **Today or future dates**: use TrueData REST `getSymbolExpiryList`
        as the source of truth. NSE periodically moves NIFTY weekly expiry
        day (Thu → Wed → Tue → Mon historically), and individual weeks may
        be holiday-shifted, so any hard-coded weekday eventually breaks.

    NEVER returns a date in the past relative to ref_date. The previous
    `expiries[-1]` fallback was the bug on 2026-04-08: it returned the
    last DB entry (Apr 7) when called on Apr 8, causing the live collector
    to subscribe to contracts that had expired the previous day.

    NIFTY-only. BANKNIFTY/FINNIFTY use monthly expiries — see
    scripts/_market_data_lib.get_current_monthly_expiry() for those.
    """
    today = date.today()

    # ── Historical lookup (backtests) ─────────────────────────────────
    if ref_date < today:
        for e in _load_expiry_dates():
            if e >= ref_date:
                return e
        # Last-ditch reload
        for e in _load_expiry_dates(force_reload=True):
            if e >= ref_date:
                return e
        logger.error(f"No historical expiry found for ref_date={ref_date}")
        return None

    # ── Live lookup (today / future) ──────────────────────────────────
    live = _fetch_live_expiries_from_truedata()
    for e in live:
        if e >= ref_date:
            return e

    # TrueData unreachable — fall back to DB (best effort), but still
    # filter out expired dates so we never subscribe to a dead contract.
    for e in _load_expiry_dates():
        if e >= ref_date:
            return e

    logger.error(f"No live or DB expiry found for ref_date={ref_date}")
    return None


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
    """
    Load option premium data for a single day.

    Tries TICK data first (full intra-minute resolution), falls back to
    1-min candles if ticks aren't available.

    The returned DataFrame has a `_mode` attribute set to "tick" or "candle"
    so check_exit() can branch on resolution. Caller MUST honour this:

      - "tick" mode: rows are individual ticks. Each row has columns
            [timestamp, premium, bid, ask] (no high/low/volume aggregation).
            check_exit() walks every tick in chronological order.
      - "candle" mode: rows are 1-min OHLCV bars. Each row has columns
            [timestamp, open, high, low, premium (=close), volume, oi].
            check_exit() uses bar high/low approximations.

    Tick mode is dramatically more accurate for trailing-stop and SL logic
    because it doesn't have to guess the intra-minute price sequence.
    """
    cache_key = (symbol, str(trading_date))
    if cache_key in _OPTION_PREMIUM_CACHE:
        return _OPTION_PREMIUM_CACHE[cache_key]

    # ── Try ticks first ──────────────────────────────────────────────
    # Threshold: we want at least ~50 ticks to make tick-mode worthwhile.
    # Below that, the candles probably have richer info (TrueData backfills
    # candles even on illiquid strikes via vol-surface estimation).
    tick_df = read_sql(
        "SELECT timestamp, price as premium, bid_price as bid, ask_price as ask "
        "FROM tick_data "
        "WHERE symbol = :sym AND timestamp::date = :dt "
        "ORDER BY timestamp",
        {"sym": symbol, "dt": str(trading_date)},
    )

    if not tick_df.empty and len(tick_df) >= 50:
        tick_df["timestamp"] = pd.to_datetime(tick_df["timestamp"])
        # Bid/ask may be 0 on some feeds — fall back to mid (premium) so
        # downstream slippage logic works.
        tick_df["bid"] = tick_df["bid"].where(tick_df["bid"] > 0, tick_df["premium"])
        tick_df["ask"] = tick_df["ask"].where(tick_df["ask"] > 0, tick_df["premium"])
        tick_df.attrs["_mode"] = "tick"
        _OPTION_PREMIUM_CACHE[cache_key] = tick_df
        return tick_df

    # ── Fall back to candles ─────────────────────────────────────────
    df = read_sql(
        "SELECT timestamp, open, high, low, close as premium, volume, oi "
        "FROM minute_candles "
        "WHERE symbol = :sym AND timestamp::date = :dt "
        "ORDER BY timestamp",
        {"sym": symbol, "dt": str(trading_date)},
    )
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.attrs["_mode"] = "candle"
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


def resolve_option_with_vol_surface(
    index_price: float,
    timestamp: pd.Timestamp,
    direction: str,
    vol_model: VolSurfaceModel,
    strike_gap: int = 50,
) -> dict:
    """
    Resolve the optimal option contract using the volatility surface model.
    Falls back to ATM if vol surface selection fails.
    
    Returns same dict as resolve_option_at_entry plus vol_surface_info.
    """
    ref_date = timestamp.date() if hasattr(timestamp, "date") else timestamp
    expiry = get_nearest_expiry(ref_date)
    if expiry is None:
        return resolve_option_at_entry(index_price, timestamp, direction, strike_gap)

    # Load all option data for this expiry+day to build IV surface
    atm = get_atm_strike(index_price, strike_gap)
    exp_code = expiry.strftime("%y%m%d")
    opt_type = "CE" if direction == "CALL" else "PE"

    # Gather premium data for nearby strikes
    option_rows = []
    for offset in range(-vol_model.max_strike_offset, vol_model.max_strike_offset + 1):
        for ot in ["CE", "PE"]:
            trial_strike = atm + offset * strike_gap
            sym = build_option_symbol(expiry, trial_strike, ot)
            pdf = load_option_premiums_for_day(sym, ref_date)
            if pdf.empty:
                continue
            # Get premium at timestamp
            ts = pd.to_datetime(timestamp)
            mask = (pdf["timestamp"] - ts).abs() <= pd.Timedelta(minutes=1)
            matching = pdf[mask]
            if matching.empty:
                continue
            row = matching.iloc[0]
            option_rows.append({
                "symbol": sym,
                "close": float(row["premium"]),
                "oi": int(row.get("oi", 0)),
                "volume": int(row.get("volume", 0)),
            })

    if not option_rows:
        return resolve_option_at_entry(index_price, timestamp, direction, strike_gap)

    option_df = pd.DataFrame(option_rows)

    # Use vol surface model to select optimal strike
    selection = vol_model.select_optimal_strike(
        spot=index_price,
        direction=direction,
        expiry=expiry,
        ref_date=ref_date,
        option_data=option_df,
    )

    if selection is None:
        return resolve_option_at_entry(index_price, timestamp, direction, strike_gap)

    # Resolve the selected strike
    selected_strike = selection["strike"]
    sym = build_option_symbol(expiry, selected_strike, opt_type)
    pdf = load_option_premiums_for_day(sym, ref_date)

    if pdf.empty:
        return resolve_option_at_entry(index_price, timestamp, direction, strike_gap)

    ts = pd.to_datetime(timestamp)
    mask = (pdf["timestamp"] - ts).abs() <= pd.Timedelta(minutes=1)
    matching = pdf[mask]
    if matching.empty:
        return resolve_option_at_entry(index_price, timestamp, direction, strike_gap)

    entry_premium = float(matching.iloc[0]["premium"])
    dte = get_days_to_expiry(ref_date, expiry)

    result = {
        "symbol": sym,
        "expiry": expiry,
        "strike": selected_strike,
        "opt_type": opt_type,
        "entry_premium": entry_premium,
        "dte": dte,
        "premium_df": pdf,
        "vol_surface_info": selection,
    }
    return result


def clear_cache():
    """Clear the premium cache (call between backtest runs)."""
    global _OPTION_PREMIUM_CACHE
    _OPTION_PREMIUM_CACHE = {}
