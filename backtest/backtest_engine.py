"""
Backtesting Engine
──────────────────
From the docs (Flow Diagrams §6, Learning Pipeline §7-8):

  Historical Data → Replay Engine → Strategy Engine → Signal Generator
                  → Trade Simulator → Portfolio Manager → Performance Metrics

  Walk-forward backtesting:
    Train on Jan–Mar → Test on Apr
    Train on Feb–Apr → Test on May
    Train → Test → Move window → Train → Test

  Metrics: Win rate, Profit factor, Sharpe ratio, Max drawdown, Expectancy

  Backtest primarily with minute data.
  Tick data is used only to refine entries.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import (
    INITIAL_CAPITAL,
    RISK_PER_TRADE,
    SCORE_THRESHOLD,
    WEIGHT_ML_PROBABILITY,
    WEIGHT_OPTIONS_FLOW,
    WEIGHT_TECHNICAL_STRENGTH,
)
from strategy.signal_generator import generate_signals, Signal
from utils.logger import get_logger

logger = get_logger("backtest")


@dataclass
class BacktestTrade:
    """A single simulated trade."""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    symbol: str = ""
    direction: str = ""
    strategy: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0
    quantity: int = 1
    pnl: float = 0.0
    result: str = ""       # WIN / LOSS / TIMEOUT
    ml_score: float = 0.0
    flow_score: float = 0.0
    tech_score: float = 0.0
    final_score: float = 0.0


@dataclass
class BacktestResult:
    """Summary of a backtest run."""
    trades: List[BacktestTrade]
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0


class BacktestEngine:
    """
    Replays historical 1-minute candle data through the full strategy pipeline.

    Simulates:
      - Signal generation
      - Trade scoring (with or without ML)
      - SL/target exit simulation
      - Performance metric calculation
    """

    def __init__(
        self,
        capital: float = INITIAL_CAPITAL,
        risk_per_trade: float = RISK_PER_TRADE,
        score_threshold: float = SCORE_THRESHOLD,
        max_trades_per_day: int = 5,
        sl_multiplier: float = 1.5,
        target_multiplier: float = 2.0,
        max_holding_periods: int = 30,
    ):
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.score_threshold = score_threshold
        self.max_trades_per_day = max_trades_per_day
        self.sl_multiplier = sl_multiplier
        self.target_multiplier = target_multiplier
        self.max_holding_periods = max_holding_periods

    def run(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        predictor=None,
        active_strategies: List[str] = None,
    ) -> BacktestResult:
        """
        Run a backtest on a DataFrame of 1-minute candles with features.

        Args:
            df: DataFrame with OHLCV + all macro features computed
            symbol: instrument symbol
            predictor: optional Predictor instance (for ML scoring)
            active_strategies: which strategies to run (None = all)

        Returns BacktestResult with all trades and metrics.
        """
        df = df.copy().reset_index(drop=True)

        if df.empty:
            logger.warning("Empty DataFrame for backtest.")
            return BacktestResult(trades=[])

        trades: List[BacktestTrade] = []
        in_trade = False
        current_trade: Optional[BacktestTrade] = None
        daily_trades = 0
        current_day = None

        logger.info(
            f"Starting backtest: {len(df)} candles, symbol={symbol}"
        )

        for i in range(50, len(df)):
            row = df.iloc[i].to_dict()
            ts = row.get("timestamp", i)

            # Reset daily trade counter
            if hasattr(ts, "date"):
                day = ts.date()
                if day != current_day:
                    current_day = day
                    daily_trades = 0

            # ── Check if current trade hit SL/target ─────────────────────────
            if in_trade and current_trade is not None:
                high = row.get("high", row.get("close", 0))
                low = row.get("low", row.get("close", 0))
                bars_held = i - current_trade.entry_time

                hit_target = False
                hit_stop = False

                if current_trade.direction == "CALL":
                    if high >= current_trade.target:
                        hit_target = True
                    if low <= current_trade.stop_loss:
                        hit_stop = True
                else:  # PUT
                    if low <= current_trade.target:
                        hit_target = True
                    if high >= current_trade.stop_loss:
                        hit_stop = True

                # Determine exit
                if hit_stop:
                    current_trade.exit_price = current_trade.stop_loss
                    current_trade.result = "LOSS"
                elif hit_target:
                    current_trade.exit_price = current_trade.target
                    current_trade.result = "WIN"
                elif bars_held >= self.max_holding_periods:
                    current_trade.exit_price = row["close"]
                    current_trade.result = "TIMEOUT"

                if current_trade.result:
                    current_trade.exit_time = ts
                    if current_trade.direction == "CALL":
                        current_trade.pnl = (
                            (current_trade.exit_price - current_trade.entry_price)
                            * current_trade.quantity
                        )
                    else:
                        current_trade.pnl = (
                            (current_trade.entry_price - current_trade.exit_price)
                            * current_trade.quantity
                        )
                    trades.append(current_trade)
                    in_trade = False
                    current_trade = None
                    continue

            # ── Generate signals if not in a trade ───────────────────────────
            if not in_trade and daily_trades < self.max_trades_per_day:
                signals = generate_signals(row, symbol, active_strategies)

                for sig in signals:
                    # ML scoring (optional)
                    ml_prob = 0.5
                    if predictor is not None:
                        p = predictor.predict_macro(row)
                        if p is not None:
                            ml_prob = p

                    # Composite score
                    final_score = (
                        WEIGHT_ML_PROBABILITY * ml_prob
                        + WEIGHT_OPTIONS_FLOW * 0.5
                        + WEIGHT_TECHNICAL_STRENGTH * sig.technical_strength
                    )

                    if final_score < self.score_threshold:
                        continue

                    # Position sizing
                    atr = row.get("atr", 0)
                    if atr <= 0:
                        continue

                    stop_dist = atr * self.sl_multiplier
                    risk_amount = self.capital * self.risk_per_trade
                    qty = max(1, int(risk_amount / stop_dist))

                    if sig.direction == "CALL":
                        sl = round(row["close"] - stop_dist, 2)
                        tgt = round(row["close"] + atr * self.target_multiplier, 2)
                    else:
                        sl = round(row["close"] + stop_dist, 2)
                        tgt = round(row["close"] - atr * self.target_multiplier, 2)

                    current_trade = BacktestTrade(
                        entry_time=i,
                        symbol=symbol,
                        direction=sig.direction,
                        strategy=sig.strategy,
                        entry_price=row["close"],
                        stop_loss=sl,
                        target=tgt,
                        quantity=qty,
                        ml_score=ml_prob,
                        flow_score=0.5,
                        tech_score=sig.technical_strength,
                        final_score=round(final_score, 4),
                    )

                    in_trade = True
                    daily_trades += 1
                    break  # One trade at a time

        # Close any remaining open trade at last price
        if in_trade and current_trade is not None:
            current_trade.exit_price = df.iloc[-1]["close"]
            current_trade.exit_time = df.iloc[-1].get("timestamp", len(df))
            current_trade.result = "TIMEOUT"
            if current_trade.direction == "CALL":
                current_trade.pnl = (
                    (current_trade.exit_price - current_trade.entry_price)
                    * current_trade.quantity
                )
            else:
                current_trade.pnl = (
                    (current_trade.entry_price - current_trade.exit_price)
                    * current_trade.quantity
                )
            trades.append(current_trade)

        result = self._compute_metrics(trades)
        self._log_summary(result)
        return result

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _compute_metrics(self, trades: List[BacktestTrade]) -> BacktestResult:
        """Compute all performance metrics from trade list."""
        if not trades:
            return BacktestResult(trades=[])

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        total_pnl = sum(t.pnl for t in trades)
        gross_wins = sum(t.pnl for t in wins) if wins else 0
        gross_losses = abs(sum(t.pnl for t in losses)) if losses else 0

        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")
        win_rate = len(wins) / len(trades) if trades else 0

        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0

        # Expectancy per trade
        expectancy = (
            win_rate * avg_win + (1 - win_rate) * avg_loss
        ) if trades else 0

        # Max drawdown
        cumulative = np.cumsum([t.pnl for t in trades])
        peak = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - peak
        max_dd = abs(drawdowns.min()) if len(drawdowns) > 0 else 0

        # Sharpe ratio (approximate, using trade returns)
        returns = [t.pnl for t in trades]
        sharpe = (
            np.mean(returns) / np.std(returns) * np.sqrt(252)
            if np.std(returns) > 0
            else 0
        )

        return BacktestResult(
            trades=trades,
            total_trades=len(trades),
            wins=len(wins),
            losses=len(losses),
            win_rate=round(win_rate, 4),
            gross_pnl=round(total_pnl, 2),
            net_pnl=round(total_pnl, 2),
            profit_factor=round(profit_factor, 2),
            max_drawdown=round(max_dd, 2),
            sharpe_ratio=round(sharpe, 2),
            expectancy=round(expectancy, 2),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
        )

    def _log_summary(self, result: BacktestResult):
        logger.info("=" * 50)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 50)
        logger.info(f"  Total trades:  {result.total_trades}")
        logger.info(f"  Wins:          {result.wins}")
        logger.info(f"  Losses:        {result.losses}")
        logger.info(f"  Win rate:      {result.win_rate:.1%}")
        logger.info(f"  Gross PnL:     ₹{result.gross_pnl:,.2f}")
        logger.info(f"  Profit factor: {result.profit_factor:.2f}")
        logger.info(f"  Max drawdown:  ₹{result.max_drawdown:,.2f}")
        logger.info(f"  Sharpe ratio:  {result.sharpe_ratio:.2f}")
        logger.info(f"  Expectancy:    ₹{result.expectancy:,.2f}/trade")
        logger.info(f"  Avg win:       ₹{result.avg_win:,.2f}")
        logger.info(f"  Avg loss:      ₹{result.avg_loss:,.2f}")
        logger.info("=" * 50)


def run_backtest(df: pd.DataFrame, symbol: str = "") -> BacktestResult:
    """Convenience function to run a backtest with default settings."""
    engine = BacktestEngine()
    return engine.run(df, symbol)