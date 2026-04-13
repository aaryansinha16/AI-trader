"""
Paper Trading Adapter
─────────────────────
Implements BrokerAdapter for simulated (paper) trading. This is the default
mode and maintains backward compatibility with the existing paper_positions
system in backend/app.py.

No real money is touched. Orders "fill" instantly at the requested price.
Positions are tracked in-memory and persisted to the JSONL trade log.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from broker.base_adapter import (
    BrokerAdapter,
    OrderRequest,
    OrderResponse,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from utils.logger import get_logger

logger = get_logger("paper_adapter")


class PaperAdapter(BrokerAdapter):
    """
    Simulated broker that fills orders instantly at the requested price.

    Positions are tracked in-memory. The OrderManager layer handles
    persistence to the JSONL trade log and the dashboard state.
    """

    def __init__(self):
        self._authenticated = False
        self._positions: dict[str, Position] = {}   # order_id → Position
        self._orders: list[OrderResponse] = []       # today's order history
        self._sl_orders: dict[str, OrderRequest] = {}  # order_id → pending SL

    # ── Authentication ────────────────────────────────────────────────

    def authenticate(self) -> bool:
        self._authenticated = True
        logger.info("Paper adapter authenticated (simulated)")
        return True

    @property
    def is_connected(self) -> bool:
        return self._authenticated

    @property
    def broker_name(self) -> str:
        return "Paper"

    # ── Order Placement ───────────────────────────────────────────────

    def place_order(self, request: OrderRequest) -> OrderResponse:
        order_id = f"PAPER-{uuid.uuid4().hex[:12]}"
        now = datetime.now()

        # Paper orders fill instantly at the requested price (or last known)
        fill_price = request.price if request.price > 0 else 0.0

        response = OrderResponse(
            order_id=order_id,
            status=OrderStatus.COMPLETE,
            filled_quantity=request.quantity,
            average_price=fill_price,
            message=f"Paper {request.side.value} filled at ₹{fill_price:.2f}",
            timestamp=now,
            raw={"symbol": request.symbol, "side": request.side.value, "tag": request.tag},
        )
        self._orders.append(response)

        # Track position
        if request.side == OrderSide.BUY:
            self._positions[order_id] = Position(
                symbol=request.symbol,
                exchange=request.exchange,
                quantity=request.quantity,
                average_price=fill_price,
                last_price=fill_price,
                pnl=0.0,
                product=request.product,
                order_id=order_id,
            )
            logger.info(f"Paper BUY: {request.symbol} ×{request.quantity} @ ₹{fill_price:.2f} [{order_id}]")
        elif request.side == OrderSide.SELL:
            # Find the matching position and close it
            closed = False
            for pid, pos in list(self._positions.items()):
                if pos.symbol == request.symbol and pos.quantity > 0:
                    pnl = (fill_price - pos.average_price) * min(request.quantity, pos.quantity)
                    pos.quantity -= request.quantity
                    pos.pnl += pnl
                    if pos.quantity <= 0:
                        del self._positions[pid]
                    closed = True
                    logger.info(
                        f"Paper SELL: {request.symbol} ×{request.quantity} @ ₹{fill_price:.2f} "
                        f"P&L=₹{pnl:+,.0f} [{order_id}]"
                    )
                    break
            if not closed:
                logger.warning(f"Paper SELL: no open position found for {request.symbol}")

        return response

    def modify_order(
        self,
        order_id: str,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        order_type: Optional[OrderType] = None,
    ) -> OrderResponse:
        # For paper trading, modify just updates the SL trigger in our tracking
        if order_id in self._sl_orders:
            if trigger_price is not None:
                self._sl_orders[order_id].trigger_price = trigger_price
            logger.info(f"Paper MODIFY: {order_id} trigger→₹{trigger_price:.2f}")
        return OrderResponse(
            order_id=order_id,
            status=OrderStatus.COMPLETE,
            message="Paper order modified",
        )

    def cancel_order(self, order_id: str) -> OrderResponse:
        self._sl_orders.pop(order_id, None)
        logger.info(f"Paper CANCEL: {order_id}")
        return OrderResponse(
            order_id=order_id,
            status=OrderStatus.CANCELLED,
            message="Paper order cancelled",
        )

    # ── Position & Order Queries ──────────────────────────────────────

    def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    def get_order_status(self, order_id: str) -> OrderResponse:
        for o in self._orders:
            if o.order_id == order_id:
                return o
        return OrderResponse(order_id=order_id, status=OrderStatus.ERROR, message="Not found")

    def get_orders_today(self) -> list[OrderResponse]:
        return list(self._orders)

    # ── Safety ────────────────────────────────────────────────────────

    def kill_switch(self) -> list[OrderResponse]:
        """Close all paper positions at current price (which is average for paper)."""
        responses = []
        for pid, pos in list(self._positions.items()):
            if pos.quantity > 0:
                resp = self.sell(pos.symbol, pos.quantity, price=pos.last_price, tag="KILL_SWITCH")
                responses.append(resp)
        self._sl_orders.clear()
        logger.warning(f"Paper KILL SWITCH: closed {len(responses)} position(s)")
        return responses

    # ── Paper-specific helpers ────────────────────────────────────────

    def update_price(self, symbol: str, price: float):
        """Update the last_price of an open position (called by tick monitor)."""
        for pos in self._positions.values():
            if pos.symbol == symbol and pos.quantity > 0:
                pos.last_price = price
                pos.pnl = (price - pos.average_price) * pos.quantity

    def reset(self):
        """Clear all positions and orders (e.g. start of new day)."""
        self._positions.clear()
        self._orders.clear()
        self._sl_orders.clear()
