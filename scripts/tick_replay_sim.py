"""
Tick-Level Replay Simulation (Paper Trading)
─────────────────────────────────────────────
Replays historical tick data through the full trading pipeline:

  1. Aggregate ticks into 1-min candles on-the-fly
  2. Maintain a rolling window of candles for feature computation
  3. Generate strategy signals per candle
  4. Score with trained ML model
  5. Simulate ATM option trades with SL/target on tick-level precision
  6. Log all trades and produce performance report

This simulates what the live system would do — but on historical ticks
for the last 2-3 days we have in the DB.

Run: python scripts/tick_replay_sim.py
"""

import os
import sys
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd

from database.db import read_sql
from features.indicators import compute_all_macro_indicators
from strategy.signal_generator import generate_signals
from models.predict import Predictor
from config.settings import (
    WEIGHT_ML_PROBABILITY,
    WEIGHT_OPTIONS_FLOW,
    WEIGHT_TECHNICAL_STRENGTH,
)
from utils.logger import get_logger

logger = get_logger("tick_replay")


@dataclass
class SimTrade:
    """A single simulated trade from tick replay."""
    entry_time: datetime = None
    exit_time: datetime = None
    symbol: str = ""
    direction: str = ""
    strategy: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0
    quantity: int = 25
    pnl: float = 0.0
    result: str = ""
    ml_score: float = 0.0
    final_score: float = 0.0
    entry_tick_idx: int = 0
    exit_tick_idx: int = 0


def aggregate_ticks_to_candle(ticks: pd.DataFrame) -> dict:
    """Aggregate a batch of ticks into a single 1-min OHLCV candle."""
    if ticks.empty:
        return None
    return {
        "timestamp": ticks["timestamp"].iloc[0].floor("min"),
        "open": ticks["price"].iloc[0],
        "high": ticks["price"].max(),
        "low": ticks["price"].min(),
        "close": ticks["price"].iloc[-1],
        "volume": ticks["volume"].sum(),
        "oi": ticks["oi"].iloc[-1] if "oi" in ticks.columns else 0,
    }


def main():
    print("\n" + "=" * 70)
    print("  TICK-LEVEL REPLAY SIMULATION (Paper Trading)")
    print("=" * 70)

    # ── Load tick data ────────────────────────────────────────────────────
    logger.info("Loading tick data from DB...")
    tick_df = read_sql("""
        SELECT timestamp, price, volume, oi, bid_price, ask_price, bid_qty, ask_qty
        FROM tick_data
        WHERE symbol = 'NIFTY-I'
        ORDER BY timestamp
    """)
    tick_df["timestamp"] = pd.to_datetime(tick_df["timestamp"])
    logger.info(f"Loaded {len(tick_df):,} ticks: {tick_df['timestamp'].min()} → {tick_df['timestamp'].max()}")

    # ── Load historical candles for feature warmup ────────────────────────
    # We need ~250 candles of history before the tick period starts for indicators
    tick_start = tick_df["timestamp"].min()
    warmup_start = tick_start - timedelta(days=5)

    logger.info("Loading warmup candles from DB...")
    warmup_df = read_sql(
        "SELECT timestamp, symbol, open, high, low, close, volume, vwap, oi "
        "FROM minute_candles WHERE symbol = 'NIFTY-I' "
        "AND timestamp < :ts ORDER BY timestamp",
        {"ts": str(tick_start)},
    )
    warmup_df["timestamp"] = pd.to_datetime(warmup_df["timestamp"])
    logger.info(f"Loaded {len(warmup_df)} warmup candles")

    # ── Load ML predictor ─────────────────────────────────────────────────
    predictor = Predictor()
    predictor.load()
    if predictor.is_loaded:
        logger.info("ML Predictor loaded.")
    else:
        logger.warning("No ML model. Using default 0.5 probability.")
        predictor = None

    # ── Simulation parameters ─────────────────────────────────────────────
    SL_MULT = 1.5
    TGT_MULT = 2.0
    SCORE_THRESHOLD = 0.60
    MAX_TRADES_PER_DAY = 3
    LOT_SIZE = 65
    ATM_DELTA = 0.5
    MAX_HOLD_SECONDS = 30 * 60  # 30 minutes
    COMMISSION_PER_ORDER = 20.0  # Zerodha: Rs20 per order
    COMMISSION_ROUND_TRIP = COMMISSION_PER_ORDER * 2  # entry + exit

    # ── Group ticks by minute ─────────────────────────────────────────────
    tick_df["minute"] = tick_df["timestamp"].dt.floor("min")
    minute_groups = tick_df.groupby("minute")
    minutes = sorted(minute_groups.groups.keys())

    logger.info(f"Replay period: {len(minutes)} minutes of tick data")

    # ── Replay loop ───────────────────────────────────────────────────────
    candle_history = warmup_df.copy()
    trades: List[SimTrade] = []
    current_trade: Optional[SimTrade] = None
    daily_trades = 0
    current_day = None
    tick_count = 0

    t0 = time.time()

    for minute_ts in minutes:
        minute_ticks = minute_groups.get_group(minute_ts)
        tick_count += len(minute_ticks)

        # Reset daily counter
        day = minute_ts.date() if hasattr(minute_ts, "date") else None
        if day != current_day:
            current_day = day
            daily_trades = 0

        # ── Check open trade against every tick (precision exit) ──────────
        if current_trade is not None:
            for _, tick in minute_ticks.iterrows():
                tick_price = tick["price"]
                tick_time = tick["timestamp"]
                elapsed = (tick_time - current_trade.entry_time).total_seconds()

                hit_target = False
                hit_stop = False

                if current_trade.direction == "CALL":
                    if tick_price >= current_trade.target:
                        hit_target = True
                    if tick_price <= current_trade.stop_loss:
                        hit_stop = True
                else:
                    if tick_price <= current_trade.target:
                        hit_target = True
                    if tick_price >= current_trade.stop_loss:
                        hit_stop = True

                if hit_stop:
                    current_trade.exit_price = current_trade.stop_loss
                    current_trade.result = "LOSS"
                elif hit_target:
                    current_trade.exit_price = current_trade.target
                    current_trade.result = "WIN"
                elif elapsed >= MAX_HOLD_SECONDS:
                    current_trade.exit_price = tick_price
                    current_trade.result = "TIMEOUT"

                if current_trade.result:
                    current_trade.exit_time = tick_time
                    # Delta-adjusted PnL
                    if current_trade.direction == "CALL":
                        idx_move = current_trade.exit_price - current_trade.entry_price
                    else:
                        idx_move = current_trade.entry_price - current_trade.exit_price
                    current_trade.pnl = round(
                        idx_move * ATM_DELTA * current_trade.quantity - COMMISSION_ROUND_TRIP, 2
                    )
                    trades.append(current_trade)
                    logger.info(
                        f"  EXIT: {current_trade.result} | "
                        f"{current_trade.direction} {current_trade.strategy} | "
                        f"PnL=₹{current_trade.pnl:,.0f} | "
                        f"held {elapsed:.0f}s"
                    )
                    current_trade = None
                    break

        # ── Build candle from this minute's ticks ─────────────────────────
        candle = aggregate_ticks_to_candle(minute_ticks)
        if candle is None:
            continue

        candle["symbol"] = "NIFTY-I"
        new_row = pd.DataFrame([candle])
        candle_history = pd.concat([candle_history, new_row], ignore_index=True)

        # Keep last 500 candles for feature computation
        if len(candle_history) > 500:
            candle_history = candle_history.tail(500).reset_index(drop=True)

        # ── Compute features on rolling window (every candle) ─────────────
        if current_trade is None and daily_trades < MAX_TRADES_PER_DAY:
            if len(candle_history) < 250:
                continue

            try:
                featured = compute_all_macro_indicators(candle_history.tail(300).copy())
                if featured.empty:
                    continue
                latest = featured.iloc[-1].to_dict()
            except Exception:
                continue

            # ── Generate signals ──────────────────────────────────────────
            signals = generate_signals(latest, "NIFTY-I")
            if not signals:
                continue

            sig = signals[0]  # Take first signal

            # ── ML scoring ────────────────────────────────────────────────
            ml_prob = 0.5
            if predictor:
                p = predictor.predict_macro(latest)
                if p is not None:
                    ml_prob = p

            # ML gate: PUT trades need model to predict low P(UP)
            if sig.direction == "PUT" and ml_prob > 0.40:
                continue

            # Directional probability for scoring
            directional_prob = ml_prob if sig.direction == "CALL" else (1.0 - ml_prob)

            final_score = (
                WEIGHT_ML_PROBABILITY * directional_prob
                + WEIGHT_OPTIONS_FLOW * 0.5
                + WEIGHT_TECHNICAL_STRENGTH * sig.technical_strength
            )

            if final_score < SCORE_THRESHOLD:
                continue

            # ── Open trade ────────────────────────────────────────────────
            atr = latest.get("atr", 0)
            if atr <= 0:
                continue

            entry_price = candle["close"]
            stop_dist = atr * SL_MULT

            if sig.direction == "CALL":
                sl = round(entry_price - stop_dist, 2)
                tgt = round(entry_price + atr * TGT_MULT, 2)
            else:
                sl = round(entry_price + stop_dist, 2)
                tgt = round(entry_price - atr * TGT_MULT, 2)

            current_trade = SimTrade(
                entry_time=minute_ts,
                symbol="NIFTY-I",
                direction=sig.direction,
                strategy=sig.strategy,
                entry_price=entry_price,
                stop_loss=sl,
                target=tgt,
                quantity=LOT_SIZE,
                ml_score=round(ml_prob, 4),
                final_score=round(final_score, 4),
            )
            daily_trades += 1

            logger.info(
                f"  ENTRY: {sig.direction} {sig.strategy} @ ₹{entry_price:,.1f} | "
                f"SL=₹{sl:,.1f} TGT=₹{tgt:,.1f} | "
                f"ML={ml_prob:.2f} Score={final_score:.2f}"
            )

    # Close any remaining trade
    if current_trade is not None:
        last_price = tick_df["price"].iloc[-1]
        current_trade.exit_price = last_price
        current_trade.exit_time = tick_df["timestamp"].iloc[-1]
        current_trade.result = "TIMEOUT"
        if current_trade.direction == "CALL":
            idx_move = current_trade.exit_price - current_trade.entry_price
        else:
            idx_move = current_trade.entry_price - current_trade.exit_price
        current_trade.pnl = round(idx_move * ATM_DELTA * current_trade.quantity - COMMISSION_ROUND_TRIP, 2)
        trades.append(current_trade)

    elapsed = time.time() - t0

    # ── Results ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TICK REPLAY SIMULATION RESULTS")
    print("=" * 70)
    print(f"  Ticks processed: {tick_count:,}")
    print(f"  Minutes:         {len(minutes)}")
    print(f"  Runtime:         {elapsed:.1f}s")
    print(f"  Total trades:    {len(trades)}")

    if trades:
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in trades)
        win_rate = len(wins) / len(trades)
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0

        print(f"  Wins:            {len(wins)}")
        print(f"  Losses:          {len(losses)}")
        print(f"  Win rate:        {win_rate:.1%}")
        print(f"  Gross PnL:       ₹{total_pnl:,.0f}")
        print(f"  Avg win:         ₹{avg_win:,.0f}")
        print(f"  Avg loss:        ₹{avg_loss:,.0f}")
        if avg_loss != 0:
            print(f"  Risk-reward:     {abs(avg_win/avg_loss):.2f}")

        # Strategy breakdown
        print("\n  By strategy:")
        for strat in set(t.strategy for t in trades):
            strat_trades = [t for t in trades if t.strategy == strat]
            strat_wins = sum(1 for t in strat_trades if t.pnl > 0)
            strat_pnl = sum(t.pnl for t in strat_trades)
            print(f"    {strat}: {len(strat_trades)} trades, {strat_wins} wins, PnL=₹{strat_pnl:,.0f}")

        # Export trades
        out_dir = Path("backtest_results")
        out_dir.mkdir(exist_ok=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        trades_data = [{
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "symbol": t.symbol,
            "direction": t.direction,
            "strategy": t.strategy,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "stop_loss": t.stop_loss,
            "target": t.target,
            "quantity": t.quantity,
            "pnl": t.pnl,
            "result": t.result,
            "ml_score": t.ml_score,
            "final_score": t.final_score,
        } for t in trades]
        pd.DataFrame(trades_data).to_csv(
            out_dir / f"tick_replay_{ts_str}.csv", index=False
        )
        print(f"\n  Trades exported to backtest_results/tick_replay_{ts_str}.csv")
    else:
        print("  No trades generated during replay period.")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
