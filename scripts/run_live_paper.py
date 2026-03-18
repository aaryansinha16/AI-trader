"""
Live Paper Trading Pipeline
────────────────────────────
Simulates real-time trading without placing actual orders.
Uses TrueData WebSocket for live tick streaming.

What happens on every tick:
  a) Save tick data to TimescaleDB (real-time persistence)
  b) Aggregate into 1-min candles on-the-fly
  c) Every minute: compute features → detect regime → generate signals
     → ML scoring (general + strategy-specific) → options flow → score
  d) If trade qualifies: log trade suggestion (paper mode) or execute (live mode)

Post-market (daily):
  e) Incremental retrain: macro model on new 1-min data
  f) Incremental retrain: micro model on new tick data
  g) Retrain strategy-specific models (weekly)

Run: python scripts/run_live_paper.py
"""

import os
import sys
import time
import threading
from datetime import datetime, date, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd

from config.settings import (
    SYMBOLS, TD_INDEX_SYMBOLS, SCAN_INTERVAL_SECONDS,
    WEIGHT_ML_PROBABILITY, WEIGHT_OPTIONS_FLOW, WEIGHT_TECHNICAL_STRENGTH,
    SCORE_THRESHOLD,
)
from data.tick_collector import TickCollector
from data.aggregator import AggregationEngine
from data.truedata_adapter import TrueDataAdapter
from database.db import read_sql, write_df, get_engine
from features.indicators import compute_all_macro_indicators
from features.micro_features import compute_micro_features
from strategy.signal_generator import generate_signals
from strategy.regime_detector import RegimeDetector, get_strategies_for_regime
from models.predict import Predictor
from models.strategy_models import StrategyPredictor
from backtest.option_resolver import get_nearest_expiry, get_days_to_expiry
from utils.logger import get_logger

logger = get_logger("live_paper")


class LivePaperTrader:
    """
    Paper trading engine that processes live ticks through the full pipeline.

    Architecture:
      TrueData WebSocket → TickCollector → AggregationEngine → FeatureEngine
      → RegimeDetector → SignalGenerator → ML Predictor → Trade Scorer
      → Trade Logger (paper) / Order Manager (live)
    """

    def __init__(self, mode: str = "paper"):
        self.mode = mode  # "paper" or "live"

        # Data layer
        self.tick_collector = TickCollector(buffer_size=200)
        self.aggregator = AggregationEngine()

        # Intelligence layer
        self.predictor = Predictor()
        self.strategy_predictor = StrategyPredictor()
        self.regime_detector = RegimeDetector()

        # State
        self._candle_buffer = pd.DataFrame()
        self._last_candle_time = None
        self._current_regime = None
        self._trade_log = []
        self._in_trade = False
        self._daily_trades = 0
        self._current_day = None

    def initialize(self):
        """Load models and warm up candle history."""
        logger.info("Initializing Live Paper Trader...")

        # Load ML models
        self.predictor.load()
        self.strategy_predictor.load()

        if self.predictor.is_loaded:
            logger.info("General ML Predictor loaded.")
        if self.strategy_predictor.available_strategies:
            logger.info(f"Strategy models: {self.strategy_predictor.available_strategies}")

        # Load recent candle history for feature computation
        warmup = read_sql(
            "SELECT timestamp, symbol, open, high, low, close, volume, vwap, oi "
            "FROM minute_candles WHERE symbol = 'NIFTY-I' "
            "ORDER BY timestamp DESC LIMIT 300"
        )
        if not warmup.empty:
            warmup["timestamp"] = pd.to_datetime(warmup["timestamp"])
            self._candle_buffer = warmup.sort_values("timestamp").reset_index(drop=True)
            logger.info(f"Warmed up with {len(warmup)} historical candles.")

        # Wire tick collector to process ticks
        self.tick_collector.add_listener(self._on_tick)
        logger.info("Initialization complete.")

    def _on_tick(self, tick: dict):
        """Called for every incoming tick."""
        # Ticks are automatically persisted by TickCollector.flush()
        # We just need to check if a new minute has started
        ts = tick.get("timestamp", datetime.now())
        minute = ts.replace(second=0, microsecond=0) if hasattr(ts, "replace") else ts

        if self._last_candle_time is None:
            self._last_candle_time = minute
            return

        # New minute → aggregate previous minute's ticks into a candle
        if minute > self._last_candle_time:
            self._on_new_minute(minute)
            self._last_candle_time = minute

    def _on_new_minute(self, current_minute):
        """Process a completed 1-minute candle."""
        # Get ticks for the previous minute from collector buffer
        tick_df = self.tick_collector.get_buffer_df(symbol="NIFTY-I")
        if tick_df.empty:
            return

        # Aggregate into a candle
        candle = {
            "timestamp": current_minute - timedelta(minutes=1),
            "symbol": "NIFTY-I",
            "open": tick_df["price"].iloc[0],
            "high": tick_df["price"].max(),
            "low": tick_df["price"].min(),
            "close": tick_df["price"].iloc[-1],
            "volume": tick_df["volume"].sum(),
            "vwap": 0,
            "oi": tick_df["oi"].iloc[-1] if "oi" in tick_df.columns else 0,
        }

        # Append to candle buffer
        new_row = pd.DataFrame([candle])
        self._candle_buffer = pd.concat(
            [self._candle_buffer, new_row], ignore_index=True
        ).tail(500)

        # Also persist the candle to DB
        try:
            write_df(new_row, "minute_candles")
        except Exception as e:
            logger.error(f"Failed to persist candle: {e}")

        # Run the trading pipeline
        self._run_pipeline(candle)

    def _run_pipeline(self, latest_candle: dict):
        """
        Full trading decision pipeline on the latest 1-min candle.

        This is where ALL the intelligence happens:
          1. Compute features
          2. Detect regime
          3. Generate signals
          4. ML scoring (general + strategy-specific)
          5. Options flow scoring
          6. Final trade decision
        """
        ts = latest_candle.get("timestamp", datetime.now())

        # Daily reset
        day = ts.date() if hasattr(ts, "date") else date.today()
        if day != self._current_day:
            self._current_day = day
            self._daily_trades = 0

        # Skip first 5 min and last 15 min
        if hasattr(ts, "hour"):
            mins = ts.hour * 60 + ts.minute - 555
            if mins < 5 or mins > 360:
                return

        if self._in_trade or self._daily_trades >= 3:
            return

        # 1. Compute features
        if len(self._candle_buffer) < 250:
            return

        try:
            featured = compute_all_macro_indicators(
                self._candle_buffer.tail(300).copy()
            )
            if featured.empty:
                return
            latest = featured.iloc[-1].to_dict()
        except Exception as e:
            logger.error(f"Feature computation error: {e}")
            return

        # 2. Detect regime
        try:
            regime_window = self._candle_buffer.tail(100)[
                ["open", "high", "low", "close", "volume"]
            ].copy()
            self._current_regime = self.regime_detector.detect(regime_window)
            regime_strategies = get_strategies_for_regime(self._current_regime)
        except Exception:
            regime_strategies = None

        # 3. Generate signals (all strategies)
        signals = generate_signals(latest, "NIFTY-I")
        if not signals:
            return

        for sig in signals:
            # 4a. General ML scoring
            ml_prob = 0.5
            if self.predictor.is_loaded:
                p = self.predictor.predict_macro(latest)
                if p is not None:
                    ml_prob = p

            # ML gate for PUTs
            if sig.direction == "PUT" and ml_prob > 0.40:
                continue

            # 4b. Strategy-specific ML scoring
            strat_prob = self.strategy_predictor.predict(sig.strategy, latest)
            if strat_prob is not None and strat_prob < 0.3:
                continue  # Strategy model says this won't work

            directional_prob = ml_prob if sig.direction == "CALL" else (1.0 - ml_prob)

            # 5. Options flow scoring (from latest features)
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

            # Regime bonus
            regime_bonus = 0.05 if regime_strategies and sig.strategy in regime_strategies else 0.0

            # 6. Final score
            final_score = (
                WEIGHT_ML_PROBABILITY * directional_prob
                + WEIGHT_OPTIONS_FLOW * flow_score
                + WEIGHT_TECHNICAL_STRENGTH * sig.technical_strength
                + regime_bonus
            )

            if final_score < SCORE_THRESHOLD:
                continue

            # Resolve option contract
            expiry = get_nearest_expiry(day)
            dte = get_days_to_expiry(day, expiry) if expiry else 0
            atm = round(latest.get("close", 0) / 50) * 50
            opt_type = "CE" if sig.direction == "CALL" else "PE"
            exp_code = expiry.strftime("%y%m%d") if expiry else "000000"
            opt_symbol = f"NIFTY{exp_code}{atm}{opt_type}"

            # Trade suggestion
            trade = {
                "timestamp": ts,
                "symbol": opt_symbol,
                "direction": sig.direction,
                "strategy": sig.strategy,
                "index_price": latest.get("close", 0),
                "atm_strike": atm,
                "expiry": str(expiry),
                "dte": dte,
                "ml_prob": round(ml_prob, 4),
                "strat_prob": round(strat_prob, 4) if strat_prob else None,
                "flow_score": round(flow_score, 2),
                "regime": self._current_regime.value if self._current_regime else "UNKNOWN",
                "final_score": round(final_score, 4),
                "tech_strength": sig.technical_strength,
            }
            self._trade_log.append(trade)
            self._daily_trades += 1

            logger.info(
                f"\n{'='*50}\n"
                f"  TRADE SUGGESTION\n"
                f"  {sig.direction} {opt_symbol}\n"
                f"  Strategy: {sig.strategy}\n"
                f"  Index: ₹{latest.get('close', 0):,.1f}  ATM: {atm}\n"
                f"  Expiry: {expiry}  DTE: {dte}\n"
                f"  ML: {ml_prob:.2f}  Strategy ML: {strat_prob or 'N/A'}\n"
                f"  Flow: {flow_score:.2f}  Regime: {self._current_regime}\n"
                f"  FINAL SCORE: {final_score:.2f}\n"
                f"{'='*50}"
            )

            if self.mode == "live":
                logger.info("  → Would execute via Zerodha (live mode)")
                # TODO: execution.order_manager.execute_trade(...)
            else:
                logger.info("  → PAPER MODE: trade logged, not executed")

            break  # One trade at a time

    def run_daily_retrain(self):
        """Post-market daily incremental retraining."""
        logger.info("Running daily incremental retrain...")
        today = date.today()

        try:
            from features.feature_engine import build_macro_features
            from features.micro_features import compute_micro_features
            from models.train_model import MacroModelTrainer, MicroModelTrainer, generate_macro_labels

            # Macro: incremental train on today's 1-min data
            today_bars = read_sql(
                "SELECT * FROM minute_candles "
                "WHERE symbol = 'NIFTY-I' AND timestamp::date = :dt "
                "ORDER BY timestamp",
                {"dt": str(today)},
            )
            if not today_bars.empty and len(today_bars) > 50:
                today_bars["timestamp"] = pd.to_datetime(today_bars["timestamp"])
                featured = compute_all_macro_indicators(today_bars)
                if not featured.empty:
                    trainer = MacroModelTrainer()
                    prepared = trainer.prepare_data(featured)
                    if len(prepared) > 10:
                        trainer.incremental_train(prepared)
                        trainer.save()
                        logger.info(f"Macro model updated with {len(prepared)} new samples.")

            # Micro: incremental train on today's tick data
            today_ticks = read_sql(
                "SELECT * FROM tick_data "
                "WHERE symbol = 'NIFTY-I' AND timestamp::date = :dt "
                "ORDER BY timestamp",
                {"dt": str(today)},
            )
            if not today_ticks.empty and len(today_ticks) > 100:
                today_ticks["timestamp"] = pd.to_datetime(today_ticks["timestamp"])
                micro_feats = compute_micro_features(today_ticks)
                if not micro_feats.empty:
                    trainer = MicroModelTrainer()
                    prepared = trainer.prepare_data(micro_feats)
                    if len(prepared) > 10:
                        trainer.incremental_train(prepared)
                        trainer.save()
                        logger.info(f"Micro model updated with {len(prepared)} new samples.")

            logger.info("Daily retrain complete.")

        except Exception as e:
            logger.error(f"Daily retrain failed: {e}", exc_info=True)

    def get_trade_log(self) -> pd.DataFrame:
        """Return trade suggestions as DataFrame."""
        if not self._trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self._trade_log)

    def export_trade_log(self, path: str = None):
        """Export trade log to CSV."""
        df = self.get_trade_log()
        if df.empty:
            logger.info("No trades to export.")
            return
        path = path or f"backtest_results/paper_trades_{date.today()}.csv"
        Path(path).parent.mkdir(exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Trade log exported to {path}")


def main():
    print("=" * 60)
    print("  LIVE PAPER TRADING MODE")
    print("=" * 60)
    print("  This connects to TrueData WebSocket for live ticks.")
    print("  Trades are logged but NOT executed.")
    print("  Press Ctrl+C to stop.")
    print("=" * 60)

    trader = LivePaperTrader(mode="paper")
    trader.initialize()

    # Connect to TrueData WebSocket
    try:
        td = TrueDataAdapter()
        td.authenticate()

        # For paper trading, we poll REST API every 30s instead of WebSocket
        # (WebSocket requires a persistent connection and subscription management)
        logger.info("Starting polling loop (30s intervals)...")

        while True:
            try:
                # Fetch latest ticks
                from datetime import timedelta
                end = datetime.now()
                start = end - timedelta(seconds=60)

                ticks = td.fetch_historical_ticks("NIFTY-I", start=start, end=end, days=1)
                if not ticks.empty:
                    for _, tick in ticks.iterrows():
                        trader.tick_collector.on_tick(tick.to_dict())

                    # Force flush to DB
                    trader.tick_collector.flush()

            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error(f"Polling error: {e}")

            time.sleep(SCAN_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        logger.info("\nStopping paper trader...")
        trader.tick_collector.flush()
        trader.export_trade_log()

        # Post-market retrain
        print("\nRunning post-market incremental retrain...")
        trader.run_daily_retrain()

        print("\nPaper trading session ended.")


if __name__ == "__main__":
    main()
