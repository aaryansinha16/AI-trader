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

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional
import json
from pathlib import Path

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

# Lazy import to avoid circular deps
def _get_option_resolver():
    from backtest.option_resolver import resolve_option_at_entry, get_days_to_expiry
    return resolve_option_at_entry, get_days_to_expiry


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

    def export_to_csv(self, filepath: str = "backtest_results.csv"):
        """Export all trades to CSV file."""
        if not self.trades:
            logger.warning("No trades to export.")
            return

        trades_data = []
        for t in self.trades:
            trades_data.append({
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
                "flow_score": t.flow_score,
                "tech_score": t.tech_score,
                "final_score": t.final_score,
            })

        df = pd.DataFrame(trades_data)
        df.to_csv(filepath, index=False)
        logger.info(f"Backtest trades exported to {filepath}")

    def export_to_json(self, filepath: str = "backtest_results.json"):
        """Export full backtest results (trades + metrics) to JSON file."""
        data = {
            "summary": {
                "total_trades": self.total_trades,
                "wins": self.wins,
                "losses": self.losses,
                "win_rate": self.win_rate,
                "gross_pnl": self.gross_pnl,
                "net_pnl": self.net_pnl,
                "profit_factor": self.profit_factor,
                "max_drawdown": self.max_drawdown,
                "sharpe_ratio": self.sharpe_ratio,
                "expectancy": self.expectancy,
                "avg_win": self.avg_win,
                "avg_loss": self.avg_loss,
            },
            "trades": []
        }

        for t in self.trades:
            trade_dict = asdict(t)
            # Convert datetime objects to strings
            if isinstance(trade_dict.get("entry_time"), datetime):
                trade_dict["entry_time"] = trade_dict["entry_time"].isoformat()
            else:
                trade_dict["entry_time"] = str(trade_dict["entry_time"])
            if isinstance(trade_dict.get("exit_time"), datetime):
                trade_dict["exit_time"] = trade_dict["exit_time"].isoformat()
            elif trade_dict.get("exit_time") is not None:
                trade_dict["exit_time"] = str(trade_dict["exit_time"])
            data["trades"].append(trade_dict)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Backtest results exported to {filepath}")

    def export_to_txt(self, filepath: str = "backtest_report.txt"):
        """Export formatted text report with summary and trade details."""
        lines = []
        lines.append("=" * 80)
        lines.append("BACKTEST REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append("SUMMARY METRICS")
        lines.append("-" * 80)
        lines.append(f"Total Trades:        {self.total_trades}")
        lines.append(f"Wins:                {self.wins}")
        lines.append(f"Losses:              {self.losses}")
        lines.append(f"Win Rate:            {self.win_rate:.2%}")
        lines.append(f"Gross P&L:           ₹{self.gross_pnl:,.2f}")
        lines.append(f"Net P&L:             ₹{self.net_pnl:,.2f}")
        lines.append(f"Profit Factor:       {self.profit_factor:.2f}")
        lines.append(f"Max Drawdown:        ₹{self.max_drawdown:,.2f}")
        lines.append(f"Sharpe Ratio:        {self.sharpe_ratio:.2f}")
        lines.append(f"Expectancy/Trade:    ₹{self.expectancy:,.2f}")
        lines.append(f"Average Win:         ₹{self.avg_win:,.2f}")
        lines.append(f"Average Loss:        ₹{self.avg_loss:,.2f}")
        lines.append("")
        lines.append("=" * 80)
        lines.append("TRADE DETAILS")
        lines.append("=" * 80)
        lines.append("")

        for i, t in enumerate(self.trades, 1):
            lines.append(f"Trade #{i}")
            lines.append("-" * 40)
            lines.append(f"  Symbol:        {t.symbol}")
            lines.append(f"  Direction:     {t.direction}")
            lines.append(f"  Strategy:      {t.strategy}")
            lines.append(f"  Entry Time:    {t.entry_time}")
            lines.append(f"  Exit Time:     {t.exit_time}")
            lines.append(f"  Entry Price:   ₹{t.entry_price:.2f}")
            lines.append(f"  Exit Price:    ₹{t.exit_price:.2f}")
            lines.append(f"  Stop Loss:     ₹{t.stop_loss:.2f}")
            lines.append(f"  Target:        ₹{t.target:.2f}")
            lines.append(f"  Quantity:      {t.quantity}")
            lines.append(f"  P&L:           ₹{t.pnl:,.2f}")
            lines.append(f"  Result:        {t.result}")
            lines.append(f"  ML Score:      {t.ml_score:.4f}")
            lines.append(f"  Flow Score:    {t.flow_score:.4f}")
            lines.append(f"  Tech Score:    {t.tech_score:.4f}")
            lines.append(f"  Final Score:   {t.final_score:.4f}")
            lines.append("")

        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)

        with open(filepath, "w") as f:
            f.write("\n".join(lines))
        logger.info(f"Backtest report exported to {filepath}")

    def export_all(self, base_name: str = "backtest", output_dir: str = "backtest_results"):
        """Export results to all formats (CSV, JSON, TXT) in specified directory."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        self.export_to_csv(str(output_path / f"{base_name}_trades.csv"))
        self.export_to_json(str(output_path / f"{base_name}_results.json"))
        self.export_to_txt(str(output_path / f"{base_name}_report.txt"))
        logger.info(f"All backtest results exported to {output_dir}/")


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
        slippage_pct: float = 0.0005,
        commission_per_order: float = 20.0,
        lot_size: int = 65,              # NIFTY lot size (65 since Jan 2026)
        atm_delta: float = 0.5,          # ATM option delta for PnL modeling
    ):
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.score_threshold = score_threshold
        self.max_trades_per_day = max_trades_per_day
        self.sl_multiplier = sl_multiplier
        self.target_multiplier = target_multiplier
        self.max_holding_periods = max_holding_periods
        self.slippage_pct = slippage_pct          # 0.05% default slippage per side
        self.commission_per_order = commission_per_order  # ₹20 per order (entry+exit = ₹40)
        self.lot_size = lot_size
        self.atm_delta = atm_delta

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
                    # PnL models ATM option premium change:
                    # option_premium_move ≈ index_move × delta
                    if current_trade.direction == "CALL":
                        index_move = current_trade.exit_price - current_trade.entry_price
                    else:
                        index_move = current_trade.entry_price - current_trade.exit_price

                    option_pnl = index_move * self.atm_delta * current_trade.quantity
                    # Slippage on option premium (not index price)
                    option_premium_est = abs(index_move * self.atm_delta)
                    slippage_cost = option_premium_est * self.slippage_pct * 2 * current_trade.quantity
                    commission_cost = self.commission_per_order * 2  # entry + exit
                    current_trade.pnl = round(option_pnl - slippage_cost - commission_cost, 2)
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

                    # ML gate: PUT trades need higher ML confidence
                    # The model predicts P(price UP). For PUTs we want P(UP) to be LOW.
                    # If ml_prob > 0.4 → model thinks price may go up → bad for PUTs
                    if sig.direction == "PUT" and ml_prob > 0.40:
                        continue

                    # For CALLs, use ml_prob directly; for PUTs, invert it
                    directional_prob = ml_prob if sig.direction == "CALL" else (1.0 - ml_prob)

                    # Composite score
                    final_score = (
                        WEIGHT_ML_PROBABILITY * directional_prob
                        + WEIGHT_OPTIONS_FLOW * 0.5
                        + WEIGHT_TECHNICAL_STRENGTH * sig.technical_strength
                    )

                    if final_score < self.score_threshold:
                        continue

                    # Position sizing: fixed lot size (options trading)
                    atr = row.get("atr", 0)
                    if atr <= 0:
                        continue

                    stop_dist = atr * self.sl_multiplier
                    # Use fixed lot size for options; cap lots by risk budget
                    risk_per_lot = stop_dist * self.atm_delta * self.lot_size
                    risk_amount = self.capital * self.risk_per_trade
                    n_lots = max(1, int(risk_amount / risk_per_lot)) if risk_per_lot > 0 else 1
                    qty = n_lots * self.lot_size

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
                index_move = current_trade.exit_price - current_trade.entry_price
            else:
                index_move = current_trade.entry_price - current_trade.exit_price
            option_pnl = index_move * self.atm_delta * current_trade.quantity
            option_premium_est = abs(index_move * self.atm_delta)
            slippage_cost = option_premium_est * self.slippage_pct * 2 * current_trade.quantity
            commission_cost = self.commission_per_order * 2
            current_trade.pnl = round(option_pnl - slippage_cost - commission_cost, 2)
            trades.append(current_trade)

        result = self._compute_metrics(trades)
        self._log_summary(result)
        return result

    # ── Option Premium Backtest ────────────────────────────────────────────────

    def run_with_premiums(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        predictor=None,
        active_strategies: List[str] = None,
    ) -> BacktestResult:
        """
        Backtest using ACTUAL option premium prices from DB.

        Instead of approximating PnL via index_move × delta, this method:
        1. At signal → resolves nearest expiry + ATM strike
        2. Looks up the real ATM option premium at that timestamp
        3. Enters at actual premium price
        4. Tracks premium OHLC bars for SL/target hits
        5. Exits at actual premium price

        SL/Target are set as percentage moves on the option premium:
          - SL: premium drops by sl_pct (default ~30% of entry premium)
          - Target: premium rises by tgt_pct (default ~50% of entry premium)

        Requires option 1-min bars in minute_candles table.
        """
        resolve_option, get_dte = _get_option_resolver()

        df = df.copy().reset_index(drop=True)
        if df.empty:
            return BacktestResult(trades=[])

        # Premium-based SL/Target as fraction of entry premium
        sl_pct = 0.30    # lose 30% of premium → stop out
        tgt_pct = 0.50   # gain 50% of premium → target hit

        trades: List[BacktestTrade] = []
        in_trade = False
        current_trade: Optional[BacktestTrade] = None
        option_info = None  # holds premium_df, symbol, expiry for current trade
        daily_trades = 0
        current_day = None

        logger.info(f"Starting PREMIUM backtest: {len(df)} candles, symbol={symbol}")

        for i in range(50, len(df)):
            row = df.iloc[i].to_dict()
            ts = row.get("timestamp", None)
            if ts is None:
                continue

            # Reset daily counter
            if hasattr(ts, "date"):
                day = ts.date()
                if day != current_day:
                    current_day = day
                    daily_trades = 0

            # ── Check open trade: track premium bar for SL/target ──────────
            if in_trade and current_trade is not None and option_info is not None:
                bars_held = i - current_trade.entry_time
                prem_df = option_info["premium_df"]

                # Find premium at current timestamp
                ts_pd = pd.to_datetime(ts)
                mask = (prem_df["timestamp"] - ts_pd).abs() <= pd.Timedelta(minutes=1)
                prem_row = prem_df[mask]

                if not prem_row.empty:
                    prem_high = float(prem_row.iloc[0].get("high", prem_row.iloc[0]["premium"]))
                    prem_low = float(prem_row.iloc[0].get("low", prem_row.iloc[0]["premium"]))
                    prem_close = float(prem_row.iloc[0]["premium"])

                    entry_prem = option_info["entry_premium"]
                    prem_sl = entry_prem * (1 - sl_pct)
                    prem_tgt = entry_prem * (1 + tgt_pct)

                    hit_target = prem_high >= prem_tgt
                    hit_stop = prem_low <= prem_sl

                    if hit_stop:
                        exit_prem = prem_sl
                        current_trade.result = "LOSS"
                    elif hit_target:
                        exit_prem = prem_tgt
                        current_trade.result = "WIN"
                    elif bars_held >= self.max_holding_periods:
                        exit_prem = prem_close
                        current_trade.result = "TIMEOUT"
                    else:
                        exit_prem = None

                    if exit_prem is not None:
                        current_trade.exit_time = ts
                        current_trade.exit_price = round(exit_prem, 2)
                        prem_move = exit_prem - entry_prem
                        raw_pnl = prem_move * current_trade.quantity
                        slippage = entry_prem * self.slippage_pct * 2 * current_trade.quantity
                        commission = self.commission_per_order * 2
                        current_trade.pnl = round(raw_pnl - slippage - commission, 2)
                        trades.append(current_trade)
                        in_trade = False
                        current_trade = None
                        option_info = None
                        continue

            # ── Generate signals ───────────────────────────────────────────
            if not in_trade and daily_trades < self.max_trades_per_day:
                signals = generate_signals(row, symbol, active_strategies)

                for sig in signals:
                    ml_prob = 0.5
                    if predictor is not None:
                        p = predictor.predict_macro(row)
                        if p is not None:
                            ml_prob = p

                    # ML gate for PUTs
                    if sig.direction == "PUT" and ml_prob > 0.40:
                        continue

                    directional_prob = ml_prob if sig.direction == "CALL" else (1.0 - ml_prob)
                    final_score = (
                        WEIGHT_ML_PROBABILITY * directional_prob
                        + WEIGHT_OPTIONS_FLOW * 0.5
                        + WEIGHT_TECHNICAL_STRENGTH * sig.technical_strength
                    )
                    if final_score < self.score_threshold:
                        continue

                    # Resolve actual ATM option contract
                    opt = resolve_option(
                        index_price=row["close"],
                        timestamp=ts,
                        direction=sig.direction,
                    )
                    if opt is None:
                        continue

                    entry_prem = opt["entry_premium"]
                    if entry_prem <= 0:
                        continue

                    # Position sizing: risk budget / (premium × sl_pct)
                    risk_per_lot = entry_prem * sl_pct * self.lot_size
                    risk_amount = self.capital * self.risk_per_trade
                    n_lots = max(1, int(risk_amount / risk_per_lot)) if risk_per_lot > 0 else 1
                    qty = n_lots * self.lot_size

                    current_trade = BacktestTrade(
                        entry_time=i,
                        symbol=opt["symbol"],
                        direction=sig.direction,
                        strategy=sig.strategy,
                        entry_price=round(entry_prem, 2),
                        stop_loss=round(entry_prem * (1 - sl_pct), 2),
                        target=round(entry_prem * (1 + tgt_pct), 2),
                        quantity=qty,
                        ml_score=ml_prob,
                        flow_score=0.5,
                        tech_score=sig.technical_strength,
                        final_score=round(final_score, 4),
                    )
                    option_info = opt
                    in_trade = True
                    daily_trades += 1

                    logger.debug(
                        f"ENTRY: {sig.direction} {opt['symbol']} "
                        f"prem=₹{entry_prem:.1f} SL=₹{entry_prem*(1-sl_pct):.1f} "
                        f"TGT=₹{entry_prem*(1+tgt_pct):.1f} DTE={opt['dte']}"
                    )
                    break

        # Close any remaining open trade
        if in_trade and current_trade is not None and option_info is not None:
            entry_prem = option_info["entry_premium"]
            prem_df = option_info["premium_df"]
            last_prem = float(prem_df.iloc[-1]["premium"]) if not prem_df.empty else entry_prem
            current_trade.exit_price = round(last_prem, 2)
            current_trade.exit_time = df.iloc[-1].get("timestamp", len(df))
            current_trade.result = "TIMEOUT"
            prem_move = last_prem - entry_prem
            raw_pnl = prem_move * current_trade.quantity
            slippage = entry_prem * self.slippage_pct * 2 * current_trade.quantity
            commission = self.commission_per_order * 2
            current_trade.pnl = round(raw_pnl - slippage - commission, 2)
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