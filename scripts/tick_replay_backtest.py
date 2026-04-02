"""
Tick-Level Replay Backtest
──────────────────────────
Streams historical ticks from DB for a given day and runs the FULL trading
pipeline exactly as it would operate live:

  tick → aggregate 1-min candle → compute features → detect regime
  → generate signals → ML scoring → options flow → composite score
  → resolve option contract → manage trade (SL / target / timeout)

Usage:
  python scripts/tick_replay_backtest.py                   # all available days
  python scripts/tick_replay_backtest.py 2026-03-10        # single day
  python scripts/tick_replay_backtest.py 2026-03-10 2026-03-11  # specific days
"""

import os, sys, argparse, json
from datetime import datetime, date, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd

from database.db import read_sql
from features.indicators import compute_all_macro_indicators
from strategy.signal_generator import generate_signals
from strategy.regime_detector import RegimeDetector, get_strategies_for_regime
from models.predict import Predictor
from models.strategy_models import StrategyPredictor
from backtest.option_resolver import (
    resolve_option_at_entry, resolve_option_with_vol_surface,
    load_option_premiums_for_day,
    clear_cache, get_nearest_expiry, get_days_to_expiry,
)
from strategy.vol_surface import VolSurfaceModel
from config.settings import (
    WEIGHT_ML_PROBABILITY, WEIGHT_OPTIONS_FLOW, WEIGHT_TECHNICAL_STRENGTH,
    SCORE_THRESHOLD,
)
from features.micro_features import compute_micro_features
from features.option_chain_features import OptionChainFeatureEngine
from strategy.regime_detector import MarketRegime
from data.news_sentiment import NewsSentimentEngine
from config.risk_profiles import get_risk_profile, RiskLevel, RiskProfile
from models.rl_exit_agent import RLExitAgent, compute_state as rl_compute_state
from utils.logger import get_logger

logger = get_logger("tick_replay")

# ── Parameters (defaults = MEDIUM risk, overridden by --risk) ────────────────
# These module-level vars are set by _apply_risk_profile() in main().
_PROFILE: RiskProfile = get_risk_profile(RiskLevel.MEDIUM)

BASE_LOT_SIZE   = _PROFILE.base_lot_size
SL_PCT          = _PROFILE.sl_pct
TGT_PCT         = _PROFILE.tgt_pct
COMMISSION      = 40.0       # ₹20/order × 2
MAX_HOLD_BARS   = _PROFILE.max_hold_bars
MAX_TRADES_DAY  = _PROFILE.max_trades_day
SKIP_FIRST_MIN  = _PROFILE.skip_first_min
SKIP_LAST_MIN   = _PROFILE.skip_last_min
MARKET_OPEN_MIN = 555        # 9:15 AM IST = 9*60+15
MAX_PREMIUM     = _PROFILE.max_premium
AFTERNOON_CUT   = _PROFILE.afternoon_cut
TRAILING_TRIGGER = _PROFILE.trailing_trigger
TRAILING_LOCK   = _PROFILE.trailing_lock

# News sentiment
NEWS_LOOKBACK_HOURS  = 4
NEWS_BLOCK_THRESHOLD = _PROFILE.news_block_threshold
NEWS_BOOST_THRESHOLD = _PROFILE.news_boost_threshold
NEWS_BOOST_AMOUNT    = _PROFILE.news_boost_amount

# Dynamic SL/Target: scale by ATR relative to median ATR
ATR_BASELINE    = 0.00065
SL_MIN_PCT      = _PROFILE.sl_min_pct
SL_MAX_PCT      = _PROFILE.sl_max_pct
TGT_MIN_PCT     = _PROFILE.tgt_min_pct
TGT_MAX_PCT     = _PROFILE.tgt_max_pct

# Regime-aware lot sizing — built from profile
def _build_regime_multipliers(profile: RiskProfile) -> dict:
    mapping = {
        "TRENDING_BULL": MarketRegime.TRENDING_BULL,
        "TRENDING_BEAR": MarketRegime.TRENDING_BEAR,
        "SIDEWAYS": MarketRegime.SIDEWAYS,
        "HIGH_VOLATILITY": MarketRegime.HIGH_VOLATILITY,
        "LOW_VOLATILITY": MarketRegime.LOW_VOLATILITY,
        "UNKNOWN": MarketRegime.UNKNOWN,
    }
    return {mapping[k]: v * profile.lot_multiplier for k, v in profile.regime_multipliers.items()}

REGIME_LOT_MULTIPLIER = _build_regime_multipliers(_PROFILE)

# Micro model entry confirmation
MICRO_MOMENTUM_THRESHOLD = 0.1

# Bid/Ask spread model — realistic slippage for NIFTY ATM options
# Entry: pay ask = close * (1 + HALF_SPREAD_PCT)
# Exit:  receive bid = close * (1 - HALF_SPREAD_PCT)
# ATM options spread ~₹0.15-0.30 on ₹40-100 premium → ~0.3% each side
HALF_SPREAD_PCT = 0.003


def apply_risk_profile(level: RiskLevel):
    """Apply a risk profile to all module-level trading parameters."""
    global _PROFILE, BASE_LOT_SIZE, SL_PCT, TGT_PCT, MAX_HOLD_BARS
    global MAX_TRADES_DAY, SKIP_FIRST_MIN, SKIP_LAST_MIN, MAX_PREMIUM
    global AFTERNOON_CUT, TRAILING_TRIGGER, TRAILING_LOCK
    global NEWS_BLOCK_THRESHOLD, NEWS_BOOST_THRESHOLD, NEWS_BOOST_AMOUNT
    global SL_MIN_PCT, SL_MAX_PCT, TGT_MIN_PCT, TGT_MAX_PCT
    global REGIME_LOT_MULTIPLIER

    _PROFILE = get_risk_profile(level)
    BASE_LOT_SIZE   = _PROFILE.base_lot_size
    SL_PCT          = _PROFILE.sl_pct
    TGT_PCT         = _PROFILE.tgt_pct
    MAX_HOLD_BARS   = _PROFILE.max_hold_bars
    MAX_TRADES_DAY  = _PROFILE.max_trades_day
    SKIP_FIRST_MIN  = _PROFILE.skip_first_min
    SKIP_LAST_MIN   = _PROFILE.skip_last_min
    MAX_PREMIUM     = _PROFILE.max_premium
    AFTERNOON_CUT   = _PROFILE.afternoon_cut
    TRAILING_TRIGGER = _PROFILE.trailing_trigger
    TRAILING_LOCK   = _PROFILE.trailing_lock
    NEWS_BLOCK_THRESHOLD = _PROFILE.news_block_threshold
    NEWS_BOOST_THRESHOLD = _PROFILE.news_boost_threshold
    NEWS_BOOST_AMOUNT    = _PROFILE.news_boost_amount
    SL_MIN_PCT      = _PROFILE.sl_min_pct
    SL_MAX_PCT      = _PROFILE.sl_max_pct
    TGT_MIN_PCT     = _PROFILE.tgt_min_pct
    TGT_MAX_PCT     = _PROFILE.tgt_max_pct
    REGIME_LOT_MULTIPLIER = _build_regime_multipliers(_PROFILE)


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_available_days() -> list:
    """Return dates that have meaningful tick data."""
    days = read_sql("""
        SELECT timestamp::date as day, COUNT(*) as ticks
        FROM tick_data WHERE symbol = 'NIFTY-I'
        GROUP BY 1 HAVING COUNT(*) > 500
        ORDER BY 1
    """)
    return list(days["day"])


def minutes_from_open(ts) -> int:
    """Minutes elapsed since market open (9:15 IST).
    
    Handles both IST-as-UTC timestamps (09:15+00:00) and real UTC
    timestamps (03:45+00:00) by detecting if the hour falls in the
    IST market window (9-16) or needs +5:30 conversion.
    """
    h, m = ts.hour, ts.minute
    # If hour < 4 or hour > 16, it's almost certainly real UTC — convert to IST (+5:30)
    # IST market hours: 9:15 to 15:30 → UTC: 3:45 to 10:00
    if h < 9:
        # Likely real UTC — add 5:30
        total_minutes_utc = h * 60 + m
        total_minutes_ist = total_minutes_utc + 330  # +5h30m
        return total_minutes_ist - MARKET_OPEN_MIN
    return h * 60 + m - MARKET_OPEN_MIN


def dynamic_sl_tgt(atr_pct: float, final_score: float = 0.0) -> tuple:
    """
    Scale SL and TGT percentages by current ATR and signal score.

    High vol → widen SL (give room) but also widen TGT (bigger moves possible)
    Low vol  → tighten SL and TGT (smaller moves, take profits quickly)
    Score-tiered boost: strong signals get tighter SL and wider TGT.
    """
    if atr_pct <= 0 or np.isnan(atr_pct):
        sl, tgt = SL_PCT, TGT_PCT
    else:
        ratio = atr_pct / ATR_BASELINE  # >1 = high vol, <1 = low vol
        sl = np.clip(SL_PCT * ratio, SL_MIN_PCT, SL_MAX_PCT)
        tgt = np.clip(TGT_PCT * ratio, TGT_MIN_PCT, TGT_MAX_PCT)

    # Score-based tgt boost: high-conviction signals deserve larger targets
    if final_score >= 0.80:
        sl = min(sl, 0.15)            # tighten SL cap on best signals
        tgt = max(tgt, TGT_MAX_PCT)   # aim for ceiling on best signals
    elif final_score >= 0.70:
        tgt = max(tgt, TGT_MIN_PCT + 0.10)  # nudge target up

    return round(sl, 3), round(tgt, 3)


def score_lot_multiplier(final_score: float) -> int:
    """Dynamic lot multiplier based on signal strength (aligned with live system)."""
    if final_score >= 0.80:
        return 3
    elif final_score >= 0.70:
        return 2
    return 1


def kelly_lot_size(
    regime: MarketRegime,
    entry_premium: float,
    equity: float,
    win_rate: float = 0.55,
    avg_win_pct: float = 0.45,
    avg_loss_pct: float = 0.30,
) -> int:
    """
    Capital-aware position sizing combining Kelly Criterion with regime scaling.

    Kelly fraction: f* = (p*b - q) / b
      p = win probability, q = 1-p
      b = avg_win / avg_loss ratio

    Applies half-Kelly for safety, then scales by regime multiplier and
    risk profile's max_capital_per_trade cap.
    Returns number of underlying units (multiple of 65).
    """
    if entry_premium <= 0 or equity <= 0:
        return BASE_LOT_SIZE

    # Kelly fraction
    b = avg_win_pct / max(avg_loss_pct, 0.01)
    q = 1.0 - win_rate
    kelly_f = (win_rate * b - q) / b
    kelly_f = max(0.0, kelly_f)          # never negative
    half_kelly = kelly_f * 0.5           # half-Kelly for safety

    # Cap by risk profile's max capital per trade
    effective_f = min(half_kelly, _PROFILE.max_capital_per_trade)

    # Capital to risk on this trade
    capital_at_risk = equity * effective_f

    # Max loss per unit = entry_premium * sl_pct (in ₹, per share)
    # NIFTY lot = 65 underlying units
    loss_per_lot = entry_premium * SL_PCT * BASE_LOT_SIZE
    if loss_per_lot <= 0:
        return BASE_LOT_SIZE

    raw_lots = capital_at_risk / loss_per_lot

    # Apply regime multiplier on top
    regime_mult = REGIME_LOT_MULTIPLIER.get(regime, 0.75)
    scaled_lots = raw_lots * regime_mult

    # Clamp: min 1 lot, max 5 lots (safety ceiling)
    clamped = max(1, min(5, round(scaled_lots)))
    return clamped * BASE_LOT_SIZE


def regime_lot_size(regime: MarketRegime) -> int:
    """Legacy fallback: regime-only sizing (used when equity not available)."""
    multiplier = REGIME_LOT_MULTIPLIER.get(regime, 0.75)
    lots = max(1, round(multiplier))
    return lots * BASE_LOT_SIZE


def check_micro_confirmation(minute_ticks: pd.DataFrame, direction: str) -> bool:
    """
    Use micro features on the current minute's ticks to confirm entry.
    
    For CALLs: want positive tick_momentum (buying pressure)
    For PUTs:  want negative tick_momentum (selling pressure)
    
    Returns True if micro features confirm the direction, or if we don't
    have enough tick data to compute (fail-open).
    """
    if len(minute_ticks) < 5:
        return True  # not enough ticks, allow entry

    try:
        # Build a minimal tick df for micro feature computation
        tick_df = minute_ticks.copy()
        # Use real bid/ask if available; fabricate only as fallback
        if "bid_price" not in tick_df.columns or tick_df["bid_price"].isna().all():
            tick_df["bid_price"] = tick_df["price"] - 0.5
            tick_df["ask_price"] = tick_df["price"] + 0.5
            tick_df["bid_qty"] = tick_df["volume"]
            tick_df["ask_qty"] = tick_df["volume"]
        else:
            # Fill NaN bid/ask with price-based estimate
            tick_df["bid_price"] = tick_df["bid_price"].fillna(tick_df["price"] - 0.5)
            tick_df["ask_price"] = tick_df["ask_price"].fillna(tick_df["price"] + 0.5)
            tick_df["bid_qty"] = tick_df["bid_qty"].fillna(tick_df["volume"])
            tick_df["ask_qty"] = tick_df["ask_qty"].fillna(tick_df["volume"])
        tick_df["symbol"] = "NIFTY-I"

        micro = compute_micro_features(tick_df, window_seconds=10)
        if micro.empty:
            return True

        last = micro.iloc[-1]
        momentum = last.get("tick_momentum", 0)
        if pd.isna(momentum):
            return True

        if direction == "CALL":
            return momentum > -MICRO_MOMENTUM_THRESHOLD  # not strongly selling
        else:  # PUT
            return momentum < MICRO_MOMENTUM_THRESHOLD   # not strongly buying
    except Exception:
        return True  # fail-open


# ── Trade Manager ────────────────────────────────────────────────────────────

class OpenTrade:
    """Tracks a single open trade with real option premium monitoring."""

    def __init__(self, entry_time, symbol, direction, strategy, entry_premium,
                 premium_df, ml_prob, strat_prob, flow_score, final_score,
                 regime, index_price, entry_bar_idx,
                 sl_pct=SL_PCT, tgt_pct=TGT_PCT, lot_size=BASE_LOT_SIZE,
                 rl_agent: RLExitAgent = None):
        self.entry_time = entry_time
        self.symbol = symbol
        self.direction = direction
        self.strategy = strategy
        self.entry_premium = entry_premium
        self.premium_df = premium_df
        self.ml_prob = ml_prob
        self.strat_prob = strat_prob
        self.flow_score = flow_score
        self.final_score = final_score
        self.regime = regime
        self.index_price = index_price
        self.entry_bar_idx = entry_bar_idx
        self.lot_size = lot_size
        self.sl_pct = sl_pct
        self.tgt_pct = tgt_pct
        self.rl_agent = rl_agent

        self.sl = entry_premium * (1 - sl_pct)
        self.target = entry_premium * (1 + tgt_pct)
        self.trailing_active = False
        self.peak_premium = entry_premium
        self.premium_history = [entry_premium]
        self.exit_time = None
        self.exit_premium = None
        self.result = None
        self.pnl = None
        # Per-bar journey: [{ts, premium, sl, nifty_price, bars_held}]
        self.journey = [{
            "ts": str(entry_time),
            "premium": round(entry_premium, 2),
            "sl": round(self.sl, 2),
            "nifty_price": round(index_price, 1),
            "bars_held": 0,
        }]

    def check_exit(self, current_minute, bar_idx, nifty_close: float = 0.0) -> bool:
        """Check SL/target/timeout against option premium at current_minute.
        
        If an RL agent is available, it can override HOLD decisions by choosing
        EXIT (take profit/cut loss early) or TIGHTEN (move SL up).
        Hard SL and TARGET are still enforced as safety rails.
        """
        ts = pd.to_datetime(current_minute)
        mask = (self.premium_df["timestamp"] - ts).abs() <= pd.Timedelta(minutes=1)
        row = self.premium_df[mask]
        if row.empty:
            return False

        p_high  = float(row.iloc[0].get("high", row.iloc[0]["premium"]))
        p_low   = float(row.iloc[0].get("low", row.iloc[0]["premium"]))
        p_close = float(row.iloc[0]["premium"])
        bars_held = bar_idx - self.entry_bar_idx

        # Apply bid-side slippage — we receive bid (= close - spread) when selling
        p_high_bid  = p_high  * (1 - HALF_SPREAD_PCT)
        p_low_bid   = p_low   * (1 - HALF_SPREAD_PCT)
        p_close_bid = p_close * (1 - HALF_SPREAD_PCT)

        self.premium_history.append(p_close)

        # Record journey point for this bar
        self.journey.append({
            "ts": str(current_minute),
            "premium": round(p_close, 2),
            "sl": round(self.sl, 2),
            "nifty_price": round(nifty_close, 1),
            "bars_held": bars_held,
        })

        # Track peak using mid price (internal reference, not adjusted)
        self.peak_premium = max(self.peak_premium, p_high)
        if not self.trailing_active:
            gain_pct = (self.peak_premium - self.entry_premium) / self.entry_premium
            if gain_pct >= TRAILING_TRIGGER:
                self.trailing_active = True
                # Lock SL at TRAILING_LOCK % above entry (not breakeven)
                lock_price = self.entry_premium * (1 + TRAILING_LOCK)
                self.sl = max(self.sl, lock_price)
        else:
            # Progressive trail: as peak rises, ratchet SL upward
            # Use a stepped retention: tighter as profit grows
            gain_from_entry = self.peak_premium - self.entry_premium
            gain_pct = gain_from_entry / self.entry_premium if self.entry_premium > 0 else 0
            if gain_pct >= TRAILING_TRIGGER * 2.5:
                retention = 0.55   # deep in profit → lock more
            elif gain_pct >= TRAILING_TRIGGER * 1.5:
                retention = 0.45   # moderate profit
            else:
                retention = 0.35   # early trailing → give room to run
            trail_sl = self.entry_premium + retention * gain_from_entry
            self.sl = max(self.sl, trail_sl)

        # Time-based SL tightening: as we approach timeout, reduce risk
        hold_pct = bars_held / max(MAX_HOLD_BARS, 1)
        if hold_pct >= 0.70 and not self.trailing_active:
            # After 70% of max hold, tighten SL toward breakeven
            tighten_progress = (hold_pct - 0.70) / 0.30  # 0→1 over last 30%
            be_price = self.entry_premium + (COMMISSION / self.lot_size)
            time_sl = self.sl + tighten_progress * (be_price - self.sl)
            if time_sl > self.sl:
                self.sl = time_sl

        exit_prem = None
        result = None

        # Hard safety rails — always enforced
        # Use bid prices (what we actually receive when selling)
        if p_low_bid <= self.sl:
            exit_prem = min(p_low_bid, self.sl)  # realistic: might gap below SL
            result = "TRAILING_SL" if self.trailing_active else "SL"
        elif p_high_bid >= self.target:
            exit_prem = self.target  # target fills at limit — receive bid = target level
            result = "TARGET"
        elif bars_held >= MAX_HOLD_BARS:
            exit_prem = p_close_bid  # market exit at bid
            result = "TIMEOUT"

        # RL agent override (only when no hard exit triggered)
        if exit_prem is None and self.rl_agent is not None and self.rl_agent.is_loaded:
            try:
                state = rl_compute_state(
                    entry_premium=self.entry_premium,
                    current_premium=p_close,
                    bars_held=bars_held,
                    max_hold_bars=MAX_HOLD_BARS,
                    sl=self.sl,
                    target=self.target,
                    trailing_active=self.trailing_active,
                    peak_premium=self.peak_premium,
                    premium_history=self.premium_history,
                )
                action = self.rl_agent.decide(state, explore=False)

                if action == "EXIT":
                    exit_prem = p_close_bid  # RL exit at bid
                    result = "RL_EXIT"
                elif action == "TIGHTEN":
                    if p_close > self.entry_premium:
                        new_sl = self.entry_premium + 0.5 * (p_close - self.entry_premium)
                        self.sl = max(self.sl, new_sl)
                        if not self.trailing_active:
                            self.trailing_active = True
            except Exception:
                pass  # fail-open: RL error → fall through to normal logic

        if exit_prem is not None:
            self.exit_time = current_minute
            self.exit_premium = round(exit_prem, 2)
            self.result = result
            self.pnl = round((exit_prem - self.entry_premium) * self.lot_size - COMMISSION, 2)
            return True
        return False

    def to_dict(self) -> dict:
        return {
            "entry_time": str(self.entry_time),
            "exit_time": str(self.exit_time),
            "symbol": self.symbol,
            "direction": self.direction,
            "strategy": self.strategy,
            "entry_premium": round(self.entry_premium, 2),
            "exit_premium": self.exit_premium,
            "sl": round(self.sl, 2),
            "target": round(self.target, 2),
            "sl_pct": self.sl_pct,
            "tgt_pct": self.tgt_pct,
            "lot_size": self.lot_size,
            "pnl": self.pnl,
            "result": self.result,
            "ml_prob": round(self.ml_prob, 4),
            "strat_prob": round(self.strat_prob, 4) if self.strat_prob else None,
            "flow_score": round(self.flow_score, 2),
            "final_score": round(self.final_score, 4),
            "regime": self.regime,
            "index_price": round(self.index_price, 1),
            "journey": self.journey,
        }


# ── Day Replay ───────────────────────────────────────────────────────────────

def replay_day(
    replay_date: date,
    predictor: Predictor,
    strategy_predictor: StrategyPredictor,
    regime_detector: RegimeDetector,
    warmup_candles: pd.DataFrame,
    news_engine: NewsSentimentEngine = None,
    oc_engine: OptionChainFeatureEngine = None,
    vol_model: VolSurfaceModel = None,
    rl_agent: RLExitAgent = None,
    equity: float = 50000.0,
    rolling_wins: list = None,
    verbose: bool = True,
) -> list:
    """
    Stream all ticks for replay_date through the full pipeline.
    Returns list of completed trade dicts.
    """
    clear_cache()

    # ── Load ticks ───────────────────────────────────────────────────────
    ticks = read_sql(
        "SELECT timestamp, price, volume, oi, bid_price, ask_price, bid_qty, ask_qty "
        "FROM tick_data WHERE symbol = 'NIFTY-I' AND timestamp::date = :dt "
        "ORDER BY timestamp",
        {"dt": str(replay_date)},
    )
    if ticks.empty:
        logger.warning(f"No ticks for {replay_date}")
        return []

    ticks["timestamp"] = pd.to_datetime(ticks["timestamp"])
    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Replaying {replay_date}  |  {len(ticks):,} ticks")
        print(f"{'─'*60}")

    # ── Group ticks into minutes ─────────────────────────────────────────
    ticks["minute"] = ticks["timestamp"].dt.floor("min")
    minute_groups = ticks.groupby("minute")
    minutes = sorted(minute_groups.groups.keys())

    # ── State ────────────────────────────────────────────────────────────
    candle_buffer = warmup_candles.copy()
    open_trade: OpenTrade = None
    completed_trades = []
    daily_trades = 0
    daily_pnl = 0.0
    signals_seen = 0
    signals_passed = 0
    # Rolling win/loss tracking for adaptive Kelly
    _wins = list(rolling_wins) if rolling_wins else []
    # Daily loss circuit breaker: stop trading if cumulative loss exceeds threshold
    daily_loss_limit = -(_PROFILE.max_capital_per_trade * equity * 3)
    # Consecutive SL circuit breaker: after 2 hard SL hits in a row, pause 30 bars
    # (2026-03-27: 3 SL hits in 71 min; 2026-03-30: 2 in 59 min — system stuck in wrong direction)
    consecutive_sl_hits = 0
    sl_pause_until_bar = -1  # bar index after which trading resumes

    # ── Stream minutes ───────────────────────────────────────────────────
    for bar_idx, minute_ts in enumerate(minutes):
        minute_ticks = minute_groups.get_group(minute_ts)

        # ── 1. Check open trade exit ─────────────────────────────────────
        nifty_close_now = float(minute_ticks["price"].iloc[-1])
        if open_trade is not None:
            if open_trade.check_exit(minute_ts, bar_idx, nifty_close=nifty_close_now):
                completed_trades.append(open_trade.to_dict())
                t = open_trade
                daily_pnl += t.pnl
                equity += t.pnl
                _wins.append(1 if t.pnl > 0 else 0)
                # Consecutive SL tracking: count hard SL hits, reset on any profit/trailing
                if t.result == "SL":
                    consecutive_sl_hits += 1
                    if consecutive_sl_hits >= 2:
                        sl_pause_until_bar = bar_idx + 30  # 30-min cooling off
                else:
                    consecutive_sl_hits = 0  # any non-SL exit resets the streak
                if verbose:
                    pnl_str = f"₹{t.pnl:+,.0f}"
                    color = "\033[92m" if t.pnl > 0 else "\033[91m"
                    reset = "\033[0m"
                    print(f"    EXIT  {t.result:7s}  {t.symbol}  {color}{pnl_str}{reset}")
                open_trade = None

        # ── 2. Build candle from ticks ───────────────────────────────────
        candle = {
            "timestamp": minute_ts,
            "symbol": "NIFTY-I",
            "open": float(minute_ticks["price"].iloc[0]),
            "high": float(minute_ticks["price"].max()),
            "low": float(minute_ticks["price"].min()),
            "close": float(minute_ticks["price"].iloc[-1]),
            "volume": int(minute_ticks["volume"].sum()),
            "vwap": 0,
            "oi": int(minute_ticks["oi"].iloc[-1]) if "oi" in minute_ticks.columns else 0,
        }
        candle_buffer = pd.concat(
            [candle_buffer, pd.DataFrame([candle])], ignore_index=True
        ).tail(500)

        # ── 3. Skip if in trade, max trades, circuit breaker, or not enough warmup
        if open_trade is not None:
            continue
        # if daily_trades >= MAX_TRADES_DAY:  # TEMP: disabled to collect more training data
        #     continue
        if daily_pnl <= daily_loss_limit:
            continue  # circuit breaker: stop trading after large intraday loss
        if bar_idx <= sl_pause_until_bar:
            continue  # cooling off after 2 consecutive SL hits
        if len(candle_buffer) < 250:
            continue

        # ── 4. Time-of-day filter ────────────────────────────────────────
        mfo = minutes_from_open(minute_ts)
        if mfo < SKIP_FIRST_MIN or mfo > (375 - SKIP_LAST_MIN):
            continue
        if mfo > AFTERNOON_CUT:
            continue  # no new entries after 12:30 IST

        # ── 4b. News sentiment gate ────────────────────────────────────
        news_sentiment = None
        news_boost = 0.0
        if news_engine is not None:
            try:
                news_sentiment = news_engine.get_market_sentiment(
                    lookback_hours=NEWS_LOOKBACK_HOURS,
                    as_of=pd.Timestamp(minute_ts).tz_localize("UTC") if pd.Timestamp(minute_ts).tz is None else pd.Timestamp(minute_ts),
                )
                if news_sentiment["should_block_trading"]:
                    continue  # critical negative event, skip
                if news_sentiment["score"] < NEWS_BLOCK_THRESHOLD:
                    continue  # very bearish news, skip
                if news_sentiment["score"] > NEWS_BOOST_THRESHOLD:
                    news_boost = NEWS_BOOST_AMOUNT
            except Exception:
                pass  # fail-open: if news unavailable, proceed

        # ── 5. Compute features ──────────────────────────────────────────
        try:
            featured = compute_all_macro_indicators(candle_buffer.tail(300).copy())
            if featured.empty:
                continue
            latest = featured.iloc[-1].to_dict()
        except Exception:
            continue

        # ── 5b. Overlay option chain features (fills NaN columns) ───────
        if oc_engine is not None:
            try:
                oc_feats = oc_engine.compute_for_timestamp(
                    timestamp=minute_ts,
                    spot_price=latest["close"],
                )
                for k, v in oc_feats.items():
                    if k in latest and (pd.isna(latest[k]) or latest[k] is None):
                        latest[k] = v
            except Exception:
                pass  # fail-open

        # ── 6. Detect regime ─────────────────────────────────────────────
        regime = MarketRegime.UNKNOWN
        regime_str = "UNKNOWN"
        regime_strategies = None
        try:
            rw = candle_buffer.tail(100)[["open", "high", "low", "close", "volume"]].copy()
            regime = regime_detector.detect(rw)
            regime_str = regime.value
            regime_strategies = get_strategies_for_regime(regime)
        except Exception:
            pass

        # ── 7. Generate signals ──────────────────────────────────────────
        signals = generate_signals(latest, "NIFTY-I")
        if not signals:
            continue

        # ── 8. Score each signal, take first qualifying ──────────────────
        for sig in signals:
            signals_seen += 1

            # 8a. General ML
            ml_prob = 0.5
            if predictor.is_loaded:
                p = predictor.predict_macro(latest)
                if p is not None:
                    ml_prob = p

            # 8c. Strategy-specific ML (fallback if out-of-distribution)
            strat_prob = strategy_predictor.predict(sig.strategy, latest)
            if strat_prob is None or strat_prob < 0.05:
                strat_prob = 0.5

            # 8d. Options flow score
            flow_score = 0.5
            pcr = latest.get("pcr")
            oi_change = latest.get("oi_change", 0)
            if pcr and not np.isnan(pcr):
                flow_score = 0.0
                if pcr > 1.2:
                    flow_score += 0.3
                if oi_change and not np.isnan(oi_change) and abs(oi_change) > 1e6:
                    flow_score += 0.3
                flow_score = min(flow_score + 0.2, 1.0)
            else:
                # OBV slope + MFI fallback when PCR is unavailable (early session, candle gaps)
                # Direction-aware: positive OBV/high MFI = bullish = good for CALL / bad for PUT
                obv_slope = latest.get("obv_slope", 0) or 0
                mfi = latest.get("mfi", 50) or 50
                if sig.direction == "CALL":
                    obv_contrib = 0.15 if obv_slope > 0 else (-0.10 if obv_slope < 0 else 0.0)
                    mfi_contrib = 0.15 if mfi > 60 else (-0.10 if mfi < 40 else 0.0)
                else:  # PUT
                    obv_contrib = 0.15 if obv_slope < 0 else (-0.10 if obv_slope > 0 else 0.0)
                    mfi_contrib = 0.15 if mfi < 40 else (-0.10 if mfi > 60 else 0.0)
                flow_score = max(0.20, min(1.0, 0.50 + obv_contrib + mfi_contrib))

            # 8e. Regime bonus / penalty
            regime_bonus = 0.05 if regime_strategies and sig.strategy in regime_strategies else 0.0
            # Penalise counter-trend strategies in volatile regimes
            if sig.strategy == "mean_reversion" and regime in (MarketRegime.HIGH_VOLATILITY, MarketRegime.TRENDING_BEAR):
                regime_bonus -= 0.08
            # CALLs in bearish regimes need extra conviction
            if sig.direction == "CALL" and regime == MarketRegime.TRENDING_BEAR:
                regime_bonus -= 0.05

            # 8f. Composite score (includes news sentiment boost)
            directional_prob = ml_prob if sig.direction == "CALL" else (1.0 - ml_prob)
            final_score = (
                WEIGHT_ML_PROBABILITY * directional_prob
                + WEIGHT_OPTIONS_FLOW * flow_score
                + WEIGHT_TECHNICAL_STRENGTH * sig.technical_strength
                + regime_bonus
                + news_boost
            )
            # Direction-based quality gate
            min_score = _PROFILE.put_score_threshold if sig.direction == "PUT" else _PROFILE.score_threshold

            # Strategy-specific overrides (evidence-based from backtest with real slippage)
            if sig.strategy == "mean_reversion":
                # Only fire in SIDEWAYS/LOW_VOLATILITY — in trending markets it fights the trend and loses
                # (2026-03-25: 2 SL hits with scores 0.90-0.94 in what was a TRENDING session)
                if regime not in (MarketRegime.SIDEWAYS, MarketRegime.LOW_VOLATILITY):
                    continue
                # ML must confirm direction: directional_prob < 0.40 means model is actively bearish/bullish
                # against the signal — counter-trend entries without ML backing have poor outcomes
                if directional_prob < 0.40:
                    continue
                min_score = max(min_score, 0.80)
            elif sig.strategy == "vwap_momentum_breakout":
                # Re-enabled: requires TRENDING_BULL regime and strong score (CALL only, bullish breakout)
                if regime not in (MarketRegime.TRENDING_BULL, MarketRegime.LOW_VOLATILITY):
                    continue
                min_score = max(min_score, 0.65)
            if final_score < min_score:
                if verbose:
                    print(
                        f"    SKIP  {sig.strategy} {sig.direction}  "
                        f"score={final_score:.3f} < {min_score}  "
                        f"(ml={directional_prob:.3f} flow={flow_score:.3f} tech={sig.technical_strength:.3f} "
                        f"strat={strat_prob:.3f} regime_bonus={regime_bonus:.2f} news={news_boost:.2f})"
                    )
                continue

            signals_passed += 1

            # ── 9a. Previous-bar NIFTY direction confirmation for expensive options ─
            # 10 of 13 MEDIUM SL hits were high-premium (>₹120) — the index often moved
            # against the signal direction in the previous bar, a "false breakout" indicator.
            # Require last completed bar to not be STRONGLY counter-directional (>0.1% move).
            if len(featured) >= 2:
                prev_bar = featured.iloc[-2]
                prev_move_pct = (float(prev_bar["close"]) - float(prev_bar["open"])) / max(float(prev_bar["open"]), 1)
                if sig.direction == "PUT" and prev_move_pct > 0.0010:
                    continue  # previous bar strongly bullish → skip PUT entry
                if sig.direction == "CALL" and prev_move_pct < -0.0010:
                    continue  # previous bar strongly bearish → skip CALL entry

            # ── 9b. Micro-level entry confirmation ────────────────────
            if not check_micro_confirmation(minute_ticks, sig.direction):
                continue  # tick momentum opposes our direction

            # ── 9b. Resolve option contract with real premium ─────────
            if vol_model is not None and _PROFILE.use_vol_surface:
                opt = resolve_option_with_vol_surface(
                    index_price=latest["close"],
                    timestamp=minute_ts,
                    direction=sig.direction,
                    vol_model=vol_model,
                )
            else:
                opt = resolve_option_at_entry(
                    index_price=latest["close"],
                    timestamp=minute_ts,
                    direction=sig.direction,
                )
            if opt is None:
                continue

            # Apply ask-side slippage — we pay ask (= close + spread) when buying
            entry_prem = opt["entry_premium"] * (1 + HALF_SPREAD_PCT)
            if entry_prem <= 0:
                continue
            # Regime-aware premium cap: tighter in volatile markets
            effective_max_prem = MAX_PREMIUM
            if regime == MarketRegime.HIGH_VOLATILITY:
                effective_max_prem = MAX_PREMIUM * 0.60
            elif regime == MarketRegime.UNKNOWN:
                effective_max_prem = MAX_PREMIUM * 0.80
            if entry_prem > effective_max_prem:
                continue

            # ── 9c. Dynamic SL/Target based on ATR + score ────────────
            atr_pct = latest.get("atr_pct", 0)
            sl_pct, tgt_pct = dynamic_sl_tgt(atr_pct, final_score)

            # ── 9d. Score-tiered lot sizing (aligned with live _lots_for_score) ─────
            # Kelly at ₹50K equity always resolves to 1 lot, then score-bonus adds +1 → every
            # trade was 2 lots regardless of conviction. Explicit tiers are transparent and match
            # the live backend exactly: 1 lot (0.60-0.70) / 2 lots (0.70-0.80) / 3 lots (0.80+)
            if final_score >= 0.80:
                lot_sz = BASE_LOT_SIZE * 3  # 195 units (3 lots)
            elif final_score >= 0.70:
                lot_sz = BASE_LOT_SIZE * 2  # 130 units (2 lots)
            else:
                lot_sz = BASE_LOT_SIZE      # 65 units (1 lot)

            # ── 10. Open trade ───────────────────────────────────────────
            open_trade = OpenTrade(
                entry_time=minute_ts,
                symbol=opt["symbol"],
                direction=sig.direction,
                strategy=sig.strategy,
                entry_premium=entry_prem,
                premium_df=opt["premium_df"],
                ml_prob=ml_prob,
                strat_prob=strat_prob,
                flow_score=flow_score,
                final_score=final_score,
                regime=regime_str,
                index_price=latest["close"],
                entry_bar_idx=bar_idx,
                sl_pct=sl_pct,
                tgt_pct=tgt_pct,
                lot_size=lot_sz,
                rl_agent=rl_agent,
            )
            daily_trades += 1

            if verbose:
                print(
                    f"    ENTRY {opt['symbol']}  {sig.direction}  "
                    f"prem=₹{entry_prem:.1f}  SL=₹{open_trade.sl:.1f}({sl_pct:.0%})  "
                    f"TGT=₹{open_trade.target:.1f}({tgt_pct:.0%})  "
                    f"lots={lot_sz}  regime={regime_str}  "
                    f"score={final_score:.2f}  strat={sig.strategy}"
                )
            break  # one trade at a time

    # ── Force-close any open trade at EOD ────────────────────────────────
    if open_trade is not None:
        # Use last available premium
        ts = pd.to_datetime(minutes[-1])
        mask = (open_trade.premium_df["timestamp"] - ts).abs() <= pd.Timedelta(minutes=2)
        row = open_trade.premium_df[mask]
        if not row.empty:
            exit_prem = float(row.iloc[-1]["premium"]) * (1 - HALF_SPREAD_PCT)  # receive bid at EOD
        else:
            exit_prem = open_trade.entry_premium  # flat
        open_trade.exit_time = minutes[-1]
        open_trade.exit_premium = round(exit_prem, 2)
        open_trade.result = "EOD_CLOSE"
        open_trade.pnl = round((exit_prem - open_trade.entry_premium) * open_trade.lot_size - COMMISSION, 2)
        completed_trades.append(open_trade.to_dict())
        if verbose:
            pnl_str = f"₹{open_trade.pnl:+,.0f}"
            print(f"    EXIT  EOD_CLOSE  {open_trade.symbol}  {pnl_str}")

    # ── Day summary ──────────────────────────────────────────────────────
    if verbose:
        day_pnl = sum(t["pnl"] for t in completed_trades)
        n = len(completed_trades)
        wins = sum(1 for t in completed_trades if t["pnl"] > 0)
        print(f"\n  Day result: {n} trades  |  {wins}W / {n-wins}L  |  P&L = ₹{day_pnl:+,.0f}")
        print(f"  Signals seen: {signals_seen}  |  Passed score: {signals_passed}")

    return completed_trades


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tick-level replay backtest")
    parser.add_argument("dates", nargs="*", help="Dates to replay (YYYY-MM-DD). Default: all available.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-trade output")
    parser.add_argument("--risk", choices=["low", "medium", "high"], default="medium",
                        help="Risk profile: low (conservative), medium (balanced), high (aggressive)")
    args = parser.parse_args()

    # Apply risk profile BEFORE any trading logic
    risk_level = RiskLevel(args.risk)
    apply_risk_profile(risk_level)

    print("=" * 60)
    print(f"  TICK-LEVEL REPLAY BACKTEST  [{_PROFILE.name.upper()} RISK]")
    print("  Streaming historical ticks through the full pipeline")
    print(f"  Risk:  {_PROFILE.level.value}  |  Lots: {_PROFILE.lot_multiplier:.2f}x  |  "
          f"SL: {SL_MIN_PCT:.0%}-{SL_MAX_PCT:.0%}  |  TGT: {TGT_MIN_PCT:.0%}-{TGT_MAX_PCT:.0%}")
    print(f"  Score: >={_PROFILE.score_threshold}  |  Max trades/day: {MAX_TRADES_DAY}  |  "
          f"Max premium: ₹{MAX_PREMIUM}")
    print("=" * 60)

    # ── Load models once ─────────────────────────────────────────────────
    predictor = Predictor()
    predictor.load()
    strategy_predictor = StrategyPredictor()
    strategy_predictor.load()
    regime_detector = RegimeDetector()

    # ── Initialize news sentiment engine ───────────────────────────────
    try:
        news_engine = NewsSentimentEngine()
        print(f"  News sentiment:  enabled")
    except Exception as e:
        news_engine = None
        print(f"  News sentiment:  disabled ({e})")

    # ── Initialize option chain feature engine ─────────────────────────
    try:
        oc_engine = OptionChainFeatureEngine()
        print(f"  Option chain:    enabled")
    except Exception as e:
        oc_engine = None
        print(f"  Option chain:    disabled ({e})")

    # ── Initialize volatility surface model ──────────────────────────
    vol_model = None
    if _PROFILE.use_vol_surface:
        vol_model = VolSurfaceModel(max_strike_offset=_PROFILE.max_strike_offset)
        print(f"  Vol surface:     enabled (±{_PROFILE.max_strike_offset} strikes)")
    else:
        print(f"  Vol surface:     disabled")

    # ── Initialize RL exit agent ───────────────────────────────────────
    rl_agent = RLExitAgent()
    if rl_agent.load():
        summary = rl_agent.policy_summary()
        print(f"  RL exit agent:   enabled ({summary['states']} states, {summary['episodes']} episodes)")
    else:
        rl_agent = None
        print(f"  RL exit agent:   disabled (no trained model)")

    print(f"  ML model loaded: {predictor.is_loaded}")
    print(f"  Strategy models: {strategy_predictor.available_strategies}")

    # ── Determine which days to replay ───────────────────────────────────
    if args.dates:
        replay_dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in args.dates]
    else:
        replay_dates = get_available_days()

    print(f"  Days to replay:  {len(replay_dates)}")
    for d in replay_dates:
        print(f"    {d}")

    # ── Load warmup candles (before earliest replay date) ────────────────
    earliest = min(replay_dates)
    warmup = read_sql(
        "SELECT timestamp, symbol, open, high, low, close, volume, vwap, oi "
        "FROM minute_candles WHERE symbol = 'NIFTY-I' "
        "AND timestamp < :dt ORDER BY timestamp DESC LIMIT 300",
        {"dt": str(earliest)},
    )
    warmup["timestamp"] = pd.to_datetime(warmup["timestamp"])
    warmup = warmup.sort_values("timestamp").reset_index(drop=True)
    print(f"  Warmup candles:  {len(warmup)}")

    # ── Replay each day ──────────────────────────────────────────────────
    from config.settings import INITIAL_CAPITAL
    all_trades = []
    running_equity = float(INITIAL_CAPITAL)
    rolling_wins: list = []
    for replay_date in replay_dates:
        if oc_engine is not None:
            oc_engine.clear_cache()  # fresh option data per day
        day_trades = replay_day(
            replay_date=replay_date,
            predictor=predictor,
            strategy_predictor=strategy_predictor,
            regime_detector=regime_detector,
            warmup_candles=warmup,
            news_engine=news_engine,
            oc_engine=oc_engine,
            vol_model=vol_model,
            rl_agent=rl_agent,
            equity=running_equity,
            rolling_wins=rolling_wins,
            verbose=not args.quiet,
        )
        all_trades.extend(day_trades)
        # Update equity and rolling win history for next day
        for t in day_trades:
            running_equity += t["pnl"]
            rolling_wins.append(1 if t["pnl"] > 0 else 0)
        rolling_wins = rolling_wins[-50:]  # keep last 50 trades

        # Carry forward: add this day's candles to warmup for next day
        day_candles = read_sql(
            "SELECT timestamp, symbol, open, high, low, close, volume, vwap, oi "
            "FROM minute_candles WHERE symbol = 'NIFTY-I' "
            "AND timestamp::date = :dt ORDER BY timestamp",
            {"dt": str(replay_date)},
        )
        if not day_candles.empty:
            day_candles["timestamp"] = pd.to_datetime(day_candles["timestamp"])
            warmup = pd.concat([warmup, day_candles], ignore_index=True).tail(300)

    # ── Final Report ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TICK REPLAY BACKTEST — FINAL REPORT")
    print("=" * 60)

    if not all_trades:
        print("  No trades executed.")
        # Write empty CSV so dashboard doesn't show stale results
        out_dir = Path("backtest_results")
        out_dir.mkdir(exist_ok=True)
        empty_df = pd.DataFrame(columns=[
            "entry_time", "exit_time", "symbol", "direction", "strategy",
            "entry_premium", "exit_premium", "sl", "target", "sl_pct", "tgt_pct",
            "lot_size", "pnl", "result", "ml_prob", "strat_prob", "flow_score",
            "final_score", "regime", "index_price",
        ])
        risk_csv = out_dir / f"trades_{_PROFILE.level.value}_risk.csv"
        empty_df.to_csv(risk_csv, index=False)
        print(f"  Empty results written to {risk_csv}")
        print("=" * 60)
        return

    df = pd.DataFrame(all_trades)
    total_pnl = df["pnl"].sum()
    n = len(df)
    wins = (df["pnl"] > 0).sum()
    losses = n - wins
    profitable = df[df["pnl"] > 0]
    unprofitable = df[df["pnl"] <= 0]

    print(f"\n  Total trades:      {n}")
    print(f"  Days replayed:     {len(replay_dates)}")
    print(f"  Trades / day:      {n / len(replay_dates):.1f}")
    print(f"\n  Total P&L:         ₹{total_pnl:+,.0f}")
    print(f"  Avg P&L / trade:   ₹{df['pnl'].mean():+,.0f}")
    print(f"  Avg P&L / day:     ₹{total_pnl / len(replay_dates):+,.0f}")
    print(f"\n  Wins:              {wins} ({wins/n*100:.0f}%)")
    print(f"  Losses:            {losses} ({losses/n*100:.0f}%)")
    if len(profitable) > 0 and len(unprofitable) > 0:
        print(f"  Avg winner:        ₹{profitable['pnl'].mean():+,.0f}")
        print(f"  Avg loser:         ₹{unprofitable['pnl'].mean():+,.0f}")
        print(f"  Risk-Reward:       {abs(profitable['pnl'].mean() / unprofitable['pnl'].mean()):.2f}")
    print(f"  Max win:           ₹{df['pnl'].max():+,.0f}")
    print(f"  Max loss:          ₹{df['pnl'].min():+,.0f}")

    # Equity curve
    df["cum_pnl"] = df["pnl"].cumsum()
    dd = (df["cum_pnl"] - df["cum_pnl"].cummax()).min()
    print(f"\n  Peak equity:       ₹{df['cum_pnl'].max():+,.0f}")
    print(f"  Max drawdown:      ₹{dd:+,.0f}")

    # By strategy
    print(f"\n  {'Strategy':<30s} {'Trades':>6s} {'WR':>5s} {'Total P&L':>10s} {'Avg':>8s}")
    print(f"  {'─'*30} {'─'*6} {'─'*5} {'─'*10} {'─'*8}")
    for strat, g in df.groupby("strategy"):
        wr = (g["pnl"] > 0).mean() * 100
        print(f"  {strat:<30s} {len(g):>6d} {wr:>4.0f}% {g['pnl'].sum():>+10,.0f} {g['pnl'].mean():>+8,.0f}")

    # By direction
    print(f"\n  {'Direction':<10s} {'Trades':>6s} {'WR':>5s} {'Total P&L':>10s}")
    print(f"  {'─'*10} {'─'*6} {'─'*5} {'─'*10}")
    for d, g in df.groupby("direction"):
        wr = (g["pnl"] > 0).mean() * 100
        print(f"  {d:<10s} {len(g):>6d} {wr:>4.0f}% {g['pnl'].sum():>+10,.0f}")

    # By result
    print(f"\n  {'Result':<12s} {'Count':>6s} {'Profitable':>10s} {'Total P&L':>10s}")
    print(f"  {'─'*12} {'─'*6} {'─'*10} {'─'*10}")
    for r, g in df.groupby("result"):
        prof = (g["pnl"] > 0).mean() * 100
        print(f"  {r:<12s} {len(g):>6d} {prof:>9.0f}% {g['pnl'].sum():>+10,.0f}")

    # Per-day breakdown
    print(f"\n  {'Date':<12s} {'Trades':>6s} {'W':>3s} {'L':>3s} {'P&L':>10s}")
    print(f"  {'─'*12} {'─'*6} {'─'*3} {'─'*3} {'─'*10}")
    df["day"] = pd.to_datetime(df["entry_time"]).dt.date
    for day, g in df.groupby("day"):
        w = (g["pnl"] > 0).sum()
        l = len(g) - w
        print(f"  {str(day):<12s} {len(g):>6d} {w:>3d} {l:>3d} {g['pnl'].sum():>+10,.0f}")

    # Export — write both a generic file and the per-risk file the dashboard API reads
    out_dir = Path("backtest_results")
    out_dir.mkdir(exist_ok=True)

    # Save journeys separately (lists can't go into CSV)
    journeys = {i: t.get("journey", []) for i, t in enumerate(all_trades)}
    journey_path = out_dir / f"journeys_{_PROFILE.level.value}_risk.json"
    with open(journey_path, "w") as f:
        json.dump(journeys, f)
    print(f"\n  Trade journeys:    {journey_path} ({len(journeys)} trades)")

    clean_df = df.drop(columns=["cum_pnl", "day", "journey"], errors="ignore")
    csv_path = out_dir / "tick_replay_trades.csv"
    clean_df.to_csv(csv_path, index=False)
    # Per-risk file consumed by /api/backtest/results
    risk_csv_path = out_dir / f"trades_{_PROFILE.level.value}_risk.csv"
    clean_df.to_csv(risk_csv_path, index=False)
    print(f"  Trades exported to {csv_path}")
    print(f"  Dashboard results: {risk_csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
