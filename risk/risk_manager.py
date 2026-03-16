"""
Risk Management System
──────────────────────
From the Product Vision doc (§14):

  Rules:
    risk per trade = 1%
    max trades/day = 5
    max daily loss = 5%

  Position sizing example:
    account = ₹50,000
    risk per trade = ₹500

  Stop loss should be exchange-managed.
  Example: Entry=200, Stop=180, Target=230
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Optional

from config.settings import (
    INITIAL_CAPITAL,
    MAX_DAILY_LOSS,
    MAX_TRADES_PER_DAY,
    RISK_PER_TRADE,
)
from utils.helpers import calculate_stop_loss, calculate_target
from utils.logger import get_logger

logger = get_logger("risk_manager")


@dataclass
class RiskDecision:
    """Result from the risk manager's validation."""
    approved: bool
    quantity: int
    stop_loss: float
    target: float
    risk_amount: float
    rejection_reason: str = ""


class RiskManager:
    """
    Validates trades against risk rules before execution.

    Tracks:
      - Daily trade count
      - Daily P&L
      - Per-trade risk limits
    """

    def __init__(
        self,
        capital: float = INITIAL_CAPITAL,
        risk_per_trade: float = RISK_PER_TRADE,
        max_trades_per_day: int = MAX_TRADES_PER_DAY,
        max_daily_loss: float = MAX_DAILY_LOSS,
    ):
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.max_trades_per_day = max_trades_per_day
        self.max_daily_loss = max_daily_loss

        # Daily state
        self._today: date = date.today()
        self._trades_today: int = 0
        self._daily_pnl: float = 0.0
        self._open_positions: Dict[str, dict] = {}

    # ── Core Validation ───────────────────────────────────────────────────────

    def validate_trade(
        self,
        symbol: str,
        entry_price: float,
        atr: float,
        direction: str = "CALL",
        sl_multiplier: float = 1.5,
        target_multiplier: float = 2.0,
    ) -> RiskDecision:
        """
        Validate a trade against all risk rules.

        Returns RiskDecision with approval status, quantity, SL, and target.
        """
        self._reset_if_new_day()

        # Rule 1: Max trades per day
        if self._trades_today >= self.max_trades_per_day:
            return RiskDecision(
                approved=False,
                quantity=0,
                stop_loss=0,
                target=0,
                risk_amount=0,
                rejection_reason=f"Max trades/day reached ({self.max_trades_per_day})",
            )

        # Rule 2: Max daily loss
        max_loss_amount = self.capital * self.max_daily_loss
        if self._daily_pnl <= -max_loss_amount:
            return RiskDecision(
                approved=False,
                quantity=0,
                stop_loss=0,
                target=0,
                risk_amount=0,
                rejection_reason=f"Max daily loss reached (₹{abs(self._daily_pnl):.0f})",
            )

        # Rule 3: Don't trade if already in same symbol
        if symbol in self._open_positions:
            return RiskDecision(
                approved=False,
                quantity=0,
                stop_loss=0,
                target=0,
                risk_amount=0,
                rejection_reason=f"Already have open position in {symbol}",
            )

        # Calculate position sizing
        risk_amount = self.capital * self.risk_per_trade
        stop_distance = atr * sl_multiplier

        if stop_distance <= 0:
            return RiskDecision(
                approved=False,
                quantity=0,
                stop_loss=0,
                target=0,
                risk_amount=0,
                rejection_reason="Invalid stop distance (ATR=0)",
            )

        quantity = max(1, int(risk_amount / stop_distance))

        # Calculate SL and target
        if direction == "CALL":
            stop_loss = round(entry_price - stop_distance, 2)
            target = round(entry_price + (atr * target_multiplier), 2)
        else:
            stop_loss = round(entry_price + stop_distance, 2)
            target = round(entry_price - (atr * target_multiplier), 2)

        logger.info(
            f"Risk approved: {symbol} {direction} "
            f"qty={quantity}, entry={entry_price}, "
            f"SL={stop_loss}, target={target}, "
            f"risk=₹{risk_amount:.0f}"
        )

        return RiskDecision(
            approved=True,
            quantity=quantity,
            stop_loss=stop_loss,
            target=target,
            risk_amount=round(risk_amount, 2),
        )

    # ── Position Tracking ─────────────────────────────────────────────────────

    def register_entry(self, symbol: str, entry_price: float, quantity: int, direction: str):
        """Record a new open position."""
        self._trades_today += 1
        self._open_positions[symbol] = {
            "entry_price": entry_price,
            "quantity": quantity,
            "direction": direction,
            "entry_time": datetime.now(),
        }
        logger.info(f"Position opened: {symbol} ({self._trades_today}/{self.max_trades_per_day} today)")

    def register_exit(self, symbol: str, exit_price: float):
        """Record a position exit and update daily P&L."""
        pos = self._open_positions.pop(symbol, None)
        if pos is None:
            return

        if pos["direction"] == "CALL":
            pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
        else:
            pnl = (pos["entry_price"] - exit_price) * pos["quantity"]

        self._daily_pnl += pnl
        logger.info(
            f"Position closed: {symbol} PnL=₹{pnl:.2f} "
            f"(daily total: ₹{self._daily_pnl:.2f})"
        )

    # ── State ─────────────────────────────────────────────────────────────────

    @property
    def trades_today(self) -> int:
        self._reset_if_new_day()
        return self._trades_today

    @property
    def daily_pnl(self) -> float:
        self._reset_if_new_day()
        return self._daily_pnl

    @property
    def open_positions(self) -> Dict:
        return dict(self._open_positions)

    @property
    def can_trade(self) -> bool:
        """Quick check: are we allowed to take another trade?"""
        self._reset_if_new_day()
        if self._trades_today >= self.max_trades_per_day:
            return False
        if self._daily_pnl <= -(self.capital * self.max_daily_loss):
            return False
        return True

    def get_daily_summary(self) -> Dict:
        return {
            "date": self._today.isoformat(),
            "trades": self._trades_today,
            "max_trades": self.max_trades_per_day,
            "daily_pnl": round(self._daily_pnl, 2),
            "max_loss_limit": round(self.capital * self.max_daily_loss, 2),
            "open_positions": len(self._open_positions),
            "can_trade": self.can_trade,
        }

    def _reset_if_new_day(self):
        today = date.today()
        if today != self._today:
            logger.info(f"New trading day: {today}. Resetting daily counters.")
            self._today = today
            self._trades_today = 0
            self._daily_pnl = 0.0


def calculate_position_size(balance: float, stop_distance: float) -> int:
    """Legacy compatibility."""
    risk_amount = balance * RISK_PER_TRADE
    if stop_distance <= 0:
        return 0
    return int(risk_amount / stop_distance)