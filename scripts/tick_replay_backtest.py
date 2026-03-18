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

import os, sys, argparse
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
    resolve_option_at_entry, load_option_premiums_for_day,
    clear_cache, get_nearest_expiry, get_days_to_expiry,
)
from config.settings import (
    WEIGHT_ML_PROBABILITY, WEIGHT_OPTIONS_FLOW, WEIGHT_TECHNICAL_STRENGTH,
    SCORE_THRESHOLD,
)
from features.micro_features import compute_micro_features
from strategy.regime_detector import MarketRegime
from utils.logger import get_logger

logger = get_logger("tick_replay")

# ── Parameters ───────────────────────────────────────────────────────────────
BASE_LOT_SIZE   = 65
SL_PCT          = 0.30       # 30% of premium (default, scaled by ATR)
TGT_PCT         = 0.50       # 50% of premium (default, scaled by ATR)
COMMISSION      = 40.0       # ₹20/order × 2
MAX_HOLD_BARS   = 30         # timeout after 30 minutes
MAX_TRADES_DAY  = 5
SKIP_FIRST_MIN  = 5          # skip first 5 min after open
SKIP_LAST_MIN   = 15         # skip last 15 min before close
MARKET_OPEN_MIN = 555        # 9:15 AM IST = 9*60+15
MAX_PREMIUM     = 250        # don't buy options above ₹250
AFTERNOON_CUT   = 195        # no new trades after 12:30 IST (195 min from open)
TRAILING_TRIGGER = 0.15      # activate trailing stop after +15% move
TRAILING_LOCK   = 0.0        # once triggered, lock SL at breakeven (0% loss)

# Dynamic SL/Target: scale by ATR relative to median ATR
ATR_BASELINE    = 0.00065    # median ATR% across Mar 10-18 data
SL_MIN_PCT      = 0.20       # floor: never tighter than 20%
SL_MAX_PCT      = 0.40       # ceiling: never wider than 40%
TGT_MIN_PCT     = 0.35       # floor: never less than 35%
TGT_MAX_PCT     = 0.70       # ceiling: never more than 70%

# Regime-aware lot sizing
REGIME_LOT_MULTIPLIER = {
    MarketRegime.TRENDING_BULL:   1.25,  # high conviction trending
    MarketRegime.TRENDING_BEAR:   1.25,
    MarketRegime.SIDEWAYS:        0.75,  # lower conviction
    MarketRegime.HIGH_VOLATILITY: 0.50,  # reduce risk in chaos
    MarketRegime.LOW_VOLATILITY:  1.00,
    MarketRegime.UNKNOWN:         0.75,
}

# Micro model entry confirmation
MICRO_MOMENTUM_THRESHOLD = 0.1   # minimum tick_momentum to confirm entry


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
    """Minutes elapsed since market open (9:15 IST)."""
    return ts.hour * 60 + ts.minute - MARKET_OPEN_MIN


def dynamic_sl_tgt(atr_pct: float) -> tuple:
    """
    Scale SL and TGT percentages by current ATR relative to baseline.
    
    High vol → widen SL (give room) but also widen TGT (bigger moves possible)
    Low vol  → tighten SL and TGT (smaller moves, take profits quickly)
    """
    if atr_pct <= 0 or np.isnan(atr_pct):
        return SL_PCT, TGT_PCT

    ratio = atr_pct / ATR_BASELINE  # >1 = high vol, <1 = low vol
    sl = np.clip(SL_PCT * ratio, SL_MIN_PCT, SL_MAX_PCT)
    tgt = np.clip(TGT_PCT * ratio, TGT_MIN_PCT, TGT_MAX_PCT)
    return round(sl, 3), round(tgt, 3)


def regime_lot_size(regime: MarketRegime) -> int:
    """Adjust lot count based on market regime."""
    multiplier = REGIME_LOT_MULTIPLIER.get(regime, 0.75)
    # Round to nearest whole lot (NIFTY lot = 65)
    lots = max(1, round(multiplier))
    return lots * 65


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
                 sl_pct=SL_PCT, tgt_pct=TGT_PCT, lot_size=BASE_LOT_SIZE):
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

        self.sl = entry_premium * (1 - sl_pct)
        self.target = entry_premium * (1 + tgt_pct)
        self.trailing_active = False
        self.peak_premium = entry_premium
        self.exit_time = None
        self.exit_premium = None
        self.result = None
        self.pnl = None

    def check_exit(self, current_minute, bar_idx) -> bool:
        """Check SL/target/timeout against option premium at current_minute.
        
        Includes trailing stop: once premium is +15% from entry, lock SL at
        breakeven. This protects profitable trades from reversing to a loss.
        """
        ts = pd.to_datetime(current_minute)
        mask = (self.premium_df["timestamp"] - ts).abs() <= pd.Timedelta(minutes=1)
        row = self.premium_df[mask]
        if row.empty:
            return False

        p_high = float(row.iloc[0].get("high", row.iloc[0]["premium"]))
        p_low  = float(row.iloc[0].get("low", row.iloc[0]["premium"]))
        p_close = float(row.iloc[0]["premium"])
        bars_held = bar_idx - self.entry_bar_idx

        # Track peak and activate trailing stop
        self.peak_premium = max(self.peak_premium, p_high)
        if not self.trailing_active:
            gain_pct = (self.peak_premium - self.entry_premium) / self.entry_premium
            if gain_pct >= TRAILING_TRIGGER:
                self.trailing_active = True
                # Lock SL at breakeven + commission recovery
                self.sl = self.entry_premium + (COMMISSION / self.lot_size)

        exit_prem = None
        result = None

        if p_low <= self.sl:
            exit_prem = self.sl
            result = "TRAILING_SL" if self.trailing_active else "SL"
        elif p_high >= self.target:
            exit_prem = self.target
            result = "TARGET"
        elif bars_held >= MAX_HOLD_BARS:
            exit_prem = p_close
            result = "TIMEOUT"

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
        }


# ── Day Replay ───────────────────────────────────────────────────────────────

def replay_day(
    replay_date: date,
    predictor: Predictor,
    strategy_predictor: StrategyPredictor,
    regime_detector: RegimeDetector,
    warmup_candles: pd.DataFrame,
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
    signals_seen = 0
    signals_passed = 0

    # ── Stream minutes ───────────────────────────────────────────────────
    for bar_idx, minute_ts in enumerate(minutes):
        minute_ticks = minute_groups.get_group(minute_ts)

        # ── 1. Check open trade exit ─────────────────────────────────────
        if open_trade is not None:
            if open_trade.check_exit(minute_ts, bar_idx):
                completed_trades.append(open_trade.to_dict())
                t = open_trade
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

        # ── 3. Skip if in trade, max trades, or not enough warmup ───────
        if open_trade is not None:
            continue
        if daily_trades >= MAX_TRADES_DAY:
            continue
        if len(candle_buffer) < 250:
            continue

        # ── 4. Time-of-day filter ────────────────────────────────────────
        mfo = minutes_from_open(minute_ts)
        if mfo < SKIP_FIRST_MIN or mfo > (375 - SKIP_LAST_MIN):
            continue
        if mfo > AFTERNOON_CUT:
            continue  # no new entries after 12:30 IST

        # ── 5. Compute features ──────────────────────────────────────────
        try:
            featured = compute_all_macro_indicators(candle_buffer.tail(300).copy())
            if featured.empty:
                continue
            latest = featured.iloc[-1].to_dict()
        except Exception:
            continue

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

            # 8b. PUT gate (stricter — PUTs need strong bearish conviction)
            if sig.direction == "PUT" and ml_prob > 0.30:
                continue

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

            # 8e. Regime bonus
            regime_bonus = 0.05 if regime_strategies and sig.strategy in regime_strategies else 0.0

            # 8f. Composite score
            directional_prob = ml_prob if sig.direction == "CALL" else (1.0 - ml_prob)
            final_score = (
                WEIGHT_ML_PROBABILITY * directional_prob
                + WEIGHT_OPTIONS_FLOW * flow_score
                + WEIGHT_TECHNICAL_STRENGTH * sig.technical_strength
                + regime_bonus
            )
            # Higher bar for PUTs
            min_score = 0.70 if sig.direction == "PUT" else SCORE_THRESHOLD
            if final_score < min_score:
                continue

            signals_passed += 1

            # ── 9a. Micro-level entry confirmation ────────────────────
            if not check_micro_confirmation(minute_ticks, sig.direction):
                continue  # tick momentum opposes our direction

            # ── 9b. Resolve option contract with real premium ─────────
            opt = resolve_option_at_entry(
                index_price=latest["close"],
                timestamp=minute_ts,
                direction=sig.direction,
            )
            if opt is None:
                continue

            entry_prem = opt["entry_premium"]
            if entry_prem <= 0:
                continue
            if entry_prem > MAX_PREMIUM:
                continue  # expensive options have worse win rates

            # ── 9c. Dynamic SL/Target based on ATR ────────────────────
            atr_pct = latest.get("atr_pct", 0)
            sl_pct, tgt_pct = dynamic_sl_tgt(atr_pct)

            # ── 9d. Regime-aware lot sizing ───────────────────────────
            lot_sz = regime_lot_size(regime)

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
            exit_prem = float(row.iloc[-1]["premium"])
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
    args = parser.parse_args()

    print("=" * 60)
    print("  TICK-LEVEL REPLAY BACKTEST")
    print("  Streaming historical ticks through the full pipeline")
    print("=" * 60)

    # ── Load models once ─────────────────────────────────────────────────
    predictor = Predictor()
    predictor.load()
    strategy_predictor = StrategyPredictor()
    strategy_predictor.load()
    regime_detector = RegimeDetector()

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
    all_trades = []
    for replay_date in replay_dates:
        day_trades = replay_day(
            replay_date=replay_date,
            predictor=predictor,
            strategy_predictor=strategy_predictor,
            regime_detector=regime_detector,
            warmup_candles=warmup,
            verbose=not args.quiet,
        )
        all_trades.extend(day_trades)

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

    # Export
    out_dir = Path("backtest_results")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "tick_replay_trades.csv"
    df.drop(columns=["cum_pnl", "day"], errors="ignore").to_csv(csv_path, index=False)
    print(f"\n  Trades exported to {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
