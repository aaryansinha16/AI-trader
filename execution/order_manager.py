"""
Order Manager
─────────────
Manages the full order lifecycle from the Product Vision doc (§15):

  Order flow:
    signal detected → risk validated → place entry order
                    → place stop loss → place target

  Stop loss should be exchange-managed.
  Example: Entry=200, Stop=180, Target=230
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from config.settings import SYMBOLS
from utils.logger import get_logger

logger = get_logger("order_manager")


@dataclass
class Order:
    """Represents a single order in the system."""
    order_id: str = ""
    symbol: str = ""
    option_symbol: str = ""
    direction: str = ""       # BUY / SELL
    order_type: str = "MARKET"
    quantity: int = 0
    price: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0
    status: str = "PENDING"   # PENDING, PLACED, FILLED, CANCELLED, REJECTED
    strategy: str = ""
    scores: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    broker_order_id: str = ""


class OrderManager:
    """
    Orchestrates order placement through the broker adapter.

    Responsibilities:
      - Build entry + SL + target order set from a ScoredTrade + RiskDecision
      - Track open orders
      - Handle order fills and cancellations
      - Log all activity to trade_log table
    """

    def __init__(self, broker_adapter=None):
        self._broker = broker_adapter
        self._orders: Dict[str, Order] = {}
        self._order_counter = 0

    def set_broker(self, broker_adapter):
        """Set or replace the broker adapter."""
        self._broker = broker_adapter

    # ── Order Creation ────────────────────────────────────────────────────────

    def create_order(
        self,
        symbol: str,
        direction: str,
        quantity: int,
        entry_price: float,
        stop_loss: float,
        target: float,
        strategy: str = "",
        scores: Dict = None,
    ) -> Order:
        """
        Create a new order (not yet placed).
        Maps direction CALL→BUY, PUT→BUY (we buy options).
        """
        self._order_counter += 1
        order_id = f"ORD-{self._order_counter:06d}"

        # For options, direction is always BUY (we buy calls or puts)
        side = "BUY"

        order = Order(
            order_id=order_id,
            symbol=symbol,
            direction=side,
            quantity=quantity,
            price=entry_price,
            stop_loss=stop_loss,
            target=target,
            strategy=strategy,
            scores=scores or {},
        )

        self._orders[order_id] = order
        logger.info(
            f"Order created: {order_id} {symbol} {side} "
            f"qty={quantity} @ {entry_price} "
            f"SL={stop_loss} T={target}"
        )
        return order

    # ── Order Placement ───────────────────────────────────────────────────────

    def place_order(self, order: Order) -> bool:
        """
        Place order through the broker adapter.
        Also places SL and target orders.

        Returns True if entry order was successfully placed.
        """
        if self._broker is None:
            logger.warning("No broker adapter set. Order not placed (dry run).")
            order.status = "DRY_RUN"
            return False

        try:
            # 1. Place entry order
            broker_id = self._broker.place_order(
                symbol=order.symbol,
                qty=order.quantity,
                side=order.direction,
                order_type="MARKET",
            )
            order.broker_order_id = broker_id or ""
            order.status = "PLACED"
            order.filled_at = datetime.now()
            logger.info(f"Entry order placed: {order.order_id} → broker={broker_id}")

            # 2. Place stop loss (exchange-managed)
            if order.stop_loss > 0:
                self._broker.place_sl_order(
                    symbol=order.symbol,
                    qty=order.quantity,
                    trigger_price=order.stop_loss,
                )
                logger.info(f"SL order placed for {order.symbol} @ {order.stop_loss}")

            # 3. Place target
            if order.target > 0:
                self._broker.place_target_order(
                    symbol=order.symbol,
                    qty=order.quantity,
                    price=order.target,
                )
                logger.info(f"Target order placed for {order.symbol} @ {order.target}")

            return True

        except Exception as e:
            order.status = "REJECTED"
            logger.error(f"Order placement failed: {order.order_id} – {e}")
            return False

    # ── Convenience: Full Execution Pipeline ──────────────────────────────────

    def execute_trade(
        self,
        symbol: str,
        direction: str,
        quantity: int,
        entry_price: float,
        stop_loss: float,
        target: float,
        strategy: str = "",
        scores: Dict = None,
    ) -> Order:
        """
        Create and immediately place an order.
        Returns the Order object (check .status for result).
        """
        order = self.create_order(
            symbol, direction, quantity, entry_price,
            stop_loss, target, strategy, scores,
        )
        self.place_order(order)
        return order

    # ── Order Tracking ────────────────────────────────────────────────────────

    def get_open_orders(self) -> List[Order]:
        return [o for o in self._orders.values() if o.status in ("PLACED", "FILLED")]

    def get_all_orders(self) -> List[Order]:
        return list(self._orders.values())

    def cancel_order(self, order_id: str):
        order = self._orders.get(order_id)
        if order and order.status == "PLACED":
            if self._broker:
                try:
                    self._broker.cancel_order(order.broker_order_id)
                except Exception as e:
                    logger.error(f"Cancel failed: {e}")
            order.status = "CANCELLED"
            logger.info(f"Order cancelled: {order_id}")