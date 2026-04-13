"""
Abstract Broker Adapter
───────────────────────
Defines the interface that all broker implementations must follow.
Adding a new broker = implementing this interface + registering in __init__.py.

Every method that touches real money has explicit safety documentation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"           # stop-loss order
    SL_MARKET = "SL-M"  # stop-loss market


class OrderStatus(str, Enum):
    PENDING = "PENDING"       # submitted but not yet confirmed
    OPEN = "OPEN"             # confirmed by exchange, waiting to fill
    COMPLETE = "COMPLETE"     # fully filled
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"


class ProductType(str, Enum):
    """Zerodha/NSE product types."""
    NRML = "NRML"    # Normal (carry-forward) — for options this is standard
    MIS = "MIS"      # Margin Intraday — auto-squared-off at 15:20 IST
    CNC = "CNC"      # Cash and Carry (equity only)


@dataclass
class OrderRequest:
    """What we send TO the broker."""
    symbol: str                       # e.g. "NIFTY26041323800PE"
    exchange: str = "NFO"             # NFO for F&O, NSE for equity
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    product: ProductType = ProductType.MIS  # intraday by default
    quantity: int = 0                 # in units (not lots)
    price: float = 0.0               # limit price (ignored for MARKET)
    trigger_price: float = 0.0       # for SL/SL-M orders
    tag: str = ""                    # free-text tag for reconciliation


@dataclass
class OrderResponse:
    """What the broker returns AFTER placing an order."""
    order_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    average_price: float = 0.0
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    raw: dict = field(default_factory=dict)  # broker-specific raw response


@dataclass
class Position:
    """An open or closed position as reported by the broker."""
    symbol: str = ""
    exchange: str = "NFO"
    quantity: int = 0          # positive = long, negative = short
    average_price: float = 0.0
    last_price: float = 0.0
    pnl: float = 0.0
    product: ProductType = ProductType.MIS
    order_id: str = ""         # entry order ID


class BrokerAdapter(ABC):
    """
    Abstract base class for all broker integrations.

    Subclasses must implement every @abstractmethod. The OrderManager
    calls these methods — it never touches broker-specific APIs directly.

    Safety contract:
      - authenticate() must be called before any order method
      - buy/sell always return OrderResponse (never raise on order rejection)
      - kill_switch() MUST close all positions unconditionally
      - is_connected property must reflect real connectivity state
    """

    # ── Authentication ────────────────────────────────────────────────

    @abstractmethod
    def authenticate(self) -> bool:
        """
        Establish a session with the broker.

        For Zerodha: complete the OAuth2 flow (redirect URL → access_token).
        For Paper: always returns True.

        Returns True if authenticated, False otherwise.
        """
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """True if the adapter has a valid, authenticated session."""
        ...

    @property
    @abstractmethod
    def broker_name(self) -> str:
        """Human-readable broker name for display (e.g. 'Zerodha', 'Paper')."""
        ...

    # ── Order Placement ───────────────────────────────────────────────

    @abstractmethod
    def place_order(self, request: OrderRequest) -> OrderResponse:
        """
        Place a single order. Returns OrderResponse with order_id and status.

        This is the raw order placement — callers should use OrderManager's
        buy() / sell() wrappers which add logging, safety checks, and
        position tracking.

        Must NOT raise on order rejection — return OrderResponse with
        status=REJECTED and the rejection message.
        """
        ...

    @abstractmethod
    def modify_order(
        self,
        order_id: str,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        order_type: Optional[OrderType] = None,
    ) -> OrderResponse:
        """Modify an existing open order (e.g. update SL trigger price)."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> OrderResponse:
        """Cancel an existing open order."""
        ...

    # ── Position & Order Queries ──────────────────────────────────────

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Return all open positions from the broker."""
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderResponse:
        """Get the current status of a specific order."""
        ...

    @abstractmethod
    def get_orders_today(self) -> list[OrderResponse]:
        """Return all orders placed today (for reconciliation)."""
        ...

    # ── Safety ────────────────────────────────────────────────────────

    @abstractmethod
    def kill_switch(self) -> list[OrderResponse]:
        """
        EMERGENCY: Close ALL open positions at market price immediately.

        This is the nuclear option. It:
          1. Cancels all pending/open orders
          2. Places market SELL orders for all long positions
          3. Returns list of exit OrderResponses

        Must NEVER raise. Must NEVER skip a position. If a single exit
        fails, retry once, then log the failure and continue to the next.

        Called by:
          - Dashboard kill-switch button
          - Max daily loss circuit breaker
          - Manual API call: POST /api/broker/kill
        """
        ...

    # ── Convenience ───────────────────────────────────────────────────

    def buy(self, symbol: str, quantity: int, price: float = 0.0,
            order_type: OrderType = OrderType.MARKET, tag: str = "") -> OrderResponse:
        """Convenience: place a BUY order."""
        return self.place_order(OrderRequest(
            symbol=symbol, side=OrderSide.BUY, quantity=quantity,
            price=price, order_type=order_type, tag=tag,
        ))

    def sell(self, symbol: str, quantity: int, price: float = 0.0,
             order_type: OrderType = OrderType.MARKET, tag: str = "") -> OrderResponse:
        """Convenience: place a SELL order."""
        return self.place_order(OrderRequest(
            symbol=symbol, side=OrderSide.SELL, quantity=quantity,
            price=price, order_type=order_type, tag=tag,
        ))

    def place_sl_order(self, symbol: str, quantity: int,
                       trigger_price: float, tag: str = "") -> OrderResponse:
        """Convenience: place a stop-loss market order."""
        return self.place_order(OrderRequest(
            symbol=symbol, side=OrderSide.SELL, quantity=quantity,
            order_type=OrderType.SL_MARKET, trigger_price=trigger_price, tag=tag,
        ))
