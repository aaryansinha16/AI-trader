"""
Zerodha Kite Connect Adapter
─────────────────────────────
Real-money execution via Zerodha's Kite Connect API v3.

Authentication flow:
  1. Server generates a login URL → user opens in browser
  2. Zerodha redirects back with a `request_token`
  3. Server exchanges request_token for `access_token` (valid ~6am next day)
  4. All subsequent API calls use the access_token

Required env vars:
  ZERODHA_API_KEY       — from https://developers.kite.trade
  ZERODHA_API_SECRET    — from Kite developer console
  ZERODHA_ACCESS_TOKEN  — set after OAuth flow (or by /api/broker/auth callback)

Optional:
  ZERODHA_USER_ID       — for display purposes only

Install:
  pip install kiteconnect

Docs: https://kite.trade/docs/connect/v3/
"""

from __future__ import annotations

import os
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
    ProductType,
)
from utils.logger import get_logger

logger = get_logger("zerodha_adapter")

# Map our OrderType → Kite order_type string
_ORDER_TYPE_MAP = {
    OrderType.MARKET: "MARKET",
    OrderType.LIMIT: "LIMIT",
    OrderType.SL: "SL",
    OrderType.SL_MARKET: "SL-M",
}

# Map our ProductType → Kite product string
_PRODUCT_MAP = {
    ProductType.MIS: "MIS",
    ProductType.NRML: "NRML",
    ProductType.CNC: "CNC",
}


class ZerodhaAdapter(BrokerAdapter):
    """
    Kite Connect v3 broker adapter for real-money trading.

    Usage:
        adapter = ZerodhaAdapter()
        if adapter.authenticate():
            resp = adapter.buy("NIFTY26041323800PE", quantity=25, tag="bearish_momentum")
    """

    def __init__(self):
        self._api_key = os.getenv("ZERODHA_API_KEY", "")
        self._api_secret = os.getenv("ZERODHA_API_SECRET", "")
        self._access_token = os.getenv("ZERODHA_ACCESS_TOKEN", "")
        self._user_id = os.getenv("ZERODHA_USER_ID", "")
        self._kite = None  # kiteconnect.KiteConnect instance
        self._connected = False

    # ── Authentication ────────────────────────────────────────────────

    def authenticate(self) -> bool:
        """
        Authenticate with Kite Connect using the stored access_token.

        If access_token is not set, generate the login URL for the user.
        The access_token is typically obtained via the OAuth redirect
        callback at /api/broker/auth/callback.
        """
        if not self._api_key:
            logger.error("ZERODHA_API_KEY not set in .env")
            return False

        try:
            from kiteconnect import KiteConnect
        except ImportError:
            logger.error("kiteconnect package not installed. Run: pip install kiteconnect")
            return False

        self._kite = KiteConnect(api_key=self._api_key)

        if self._access_token:
            self._kite.set_access_token(self._access_token)
            # Verify the token is valid
            try:
                profile = self._kite.profile()
                self._user_id = profile.get("user_id", self._user_id)
                self._connected = True
                logger.info(f"Zerodha authenticated: {self._user_id}")
                return True
            except Exception as e:
                logger.error(f"Zerodha access_token invalid: {e}")
                self._connected = False
                return False
        else:
            login_url = self._kite.login_url()
            logger.warning(
                f"Zerodha: no access_token. Complete OAuth flow:\n"
                f"  1. Open: {login_url}\n"
                f"  2. Login and authorize\n"
                f"  3. Copy the request_token from the redirect URL\n"
                f"  4. POST it to /api/broker/auth/callback"
            )
            return False

    def generate_login_url(self) -> str:
        """Return the Kite Connect login URL for OAuth2 flow."""
        if self._kite is None:
            try:
                from kiteconnect import KiteConnect
                self._kite = KiteConnect(api_key=self._api_key)
            except ImportError:
                return ""
        return self._kite.login_url()

    def complete_auth(self, request_token: str) -> bool:
        """
        Exchange request_token for access_token (step 3 of OAuth flow).

        Called by /api/broker/auth/callback when user completes login.
        """
        if self._kite is None:
            logger.error("Kite not initialized — call authenticate() first")
            return False
        try:
            data = self._kite.generate_session(request_token, api_secret=self._api_secret)
            self._access_token = data["access_token"]
            self._kite.set_access_token(self._access_token)
            self._user_id = data.get("user_id", "")
            self._connected = True
            # Persist token to env so it survives restarts (until tomorrow 6am)
            os.environ["ZERODHA_ACCESS_TOKEN"] = self._access_token
            logger.info(f"Zerodha OAuth complete: user={self._user_id}, token set")
            return True
        except Exception as e:
            logger.error(f"Zerodha OAuth failed: {e}")
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected and self._kite is not None

    @property
    def broker_name(self) -> str:
        return "Zerodha"

    # ── Order Placement ───────────────────────────────────────────────

    def place_order(self, request: OrderRequest) -> OrderResponse:
        if not self.is_connected:
            return OrderResponse(status=OrderStatus.ERROR, message="Not connected to Zerodha")

        try:
            kite_params = {
                "tradingsymbol": request.symbol,
                "exchange": request.exchange,
                "transaction_type": request.side.value,
                "order_type": _ORDER_TYPE_MAP.get(request.order_type, "MARKET"),
                "product": _PRODUCT_MAP.get(request.product, "MIS"),
                "quantity": request.quantity,
                "validity": "DAY",
                "tag": request.tag[:20] if request.tag else "",  # Kite tag max 20 chars
            }

            if request.order_type in (OrderType.LIMIT, OrderType.SL):
                kite_params["price"] = request.price
            if request.order_type in (OrderType.SL, OrderType.SL_MARKET):
                kite_params["trigger_price"] = request.trigger_price

            order_id = self._kite.place_order(variety="regular", **kite_params)

            logger.info(
                f"Zerodha ORDER: {request.side.value} {request.symbol} "
                f"×{request.quantity} [{request.order_type.value}] → {order_id}"
            )

            return OrderResponse(
                order_id=str(order_id),
                status=OrderStatus.OPEN,
                filled_quantity=0,  # will be updated by get_order_status
                message=f"Order placed: {order_id}",
                timestamp=datetime.now(),
                raw=kite_params,
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Zerodha order FAILED: {request.symbol} {request.side.value} — {error_msg}")

            # Detect specific rejection reasons
            status = OrderStatus.REJECTED
            if "insufficient" in error_msg.lower():
                status = OrderStatus.REJECTED
            elif "network" in error_msg.lower() or "timeout" in error_msg.lower():
                status = OrderStatus.ERROR

            return OrderResponse(
                status=status,
                message=error_msg,
                timestamp=datetime.now(),
            )

    def modify_order(
        self,
        order_id: str,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        order_type: Optional[OrderType] = None,
    ) -> OrderResponse:
        if not self.is_connected:
            return OrderResponse(status=OrderStatus.ERROR, message="Not connected")

        try:
            params = {"order_id": order_id, "variety": "regular"}
            if quantity is not None:
                params["quantity"] = quantity
            if price is not None:
                params["price"] = price
            if trigger_price is not None:
                params["trigger_price"] = trigger_price
            if order_type is not None:
                params["order_type"] = _ORDER_TYPE_MAP.get(order_type, "MARKET")

            self._kite.modify_order(**params)
            logger.info(f"Zerodha MODIFY: {order_id} trigger=₹{trigger_price}")
            return OrderResponse(order_id=order_id, status=OrderStatus.OPEN, message="Modified")

        except Exception as e:
            logger.error(f"Zerodha MODIFY failed: {order_id} — {e}")
            return OrderResponse(order_id=order_id, status=OrderStatus.ERROR, message=str(e))

    def cancel_order(self, order_id: str) -> OrderResponse:
        if not self.is_connected:
            return OrderResponse(status=OrderStatus.ERROR, message="Not connected")

        try:
            self._kite.cancel_order(variety="regular", order_id=order_id)
            logger.info(f"Zerodha CANCEL: {order_id}")
            return OrderResponse(order_id=order_id, status=OrderStatus.CANCELLED, message="Cancelled")
        except Exception as e:
            logger.error(f"Zerodha CANCEL failed: {order_id} — {e}")
            return OrderResponse(order_id=order_id, status=OrderStatus.ERROR, message=str(e))

    # ── Position & Order Queries ──────────────────────────────────────

    def get_positions(self) -> list[Position]:
        if not self.is_connected:
            return []

        try:
            data = self._kite.positions()
            positions = []
            for pos in data.get("net", []):
                if pos["quantity"] != 0:
                    positions.append(Position(
                        symbol=pos["tradingsymbol"],
                        exchange=pos["exchange"],
                        quantity=pos["quantity"],
                        average_price=pos["average_price"],
                        last_price=pos["last_price"],
                        pnl=pos["pnl"],
                        product=ProductType(pos.get("product", "MIS")),
                    ))
            return positions
        except Exception as e:
            logger.error(f"Zerodha get_positions failed: {e}")
            return []

    def get_order_status(self, order_id: str) -> OrderResponse:
        if not self.is_connected:
            return OrderResponse(status=OrderStatus.ERROR, message="Not connected")

        try:
            history = self._kite.order_history(order_id)
            if not history:
                return OrderResponse(order_id=order_id, status=OrderStatus.ERROR, message="No history")

            latest = history[-1]
            status_map = {
                "COMPLETE": OrderStatus.COMPLETE,
                "REJECTED": OrderStatus.REJECTED,
                "CANCELLED": OrderStatus.CANCELLED,
                "OPEN": OrderStatus.OPEN,
                "PENDING": OrderStatus.PENDING,
            }
            return OrderResponse(
                order_id=order_id,
                status=status_map.get(latest.get("status", ""), OrderStatus.PENDING),
                filled_quantity=latest.get("filled_quantity", 0),
                average_price=latest.get("average_price", 0),
                message=latest.get("status_message", ""),
                raw=latest,
            )
        except Exception as e:
            logger.error(f"Zerodha order_status failed: {order_id} — {e}")
            return OrderResponse(order_id=order_id, status=OrderStatus.ERROR, message=str(e))

    def get_orders_today(self) -> list[OrderResponse]:
        if not self.is_connected:
            return []

        try:
            orders = self._kite.orders()
            return [
                OrderResponse(
                    order_id=str(o["order_id"]),
                    status=OrderStatus.COMPLETE if o["status"] == "COMPLETE" else OrderStatus.OPEN,
                    filled_quantity=o.get("filled_quantity", 0),
                    average_price=o.get("average_price", 0),
                    message=o.get("status_message", ""),
                    raw=o,
                )
                for o in orders
            ]
        except Exception as e:
            logger.error(f"Zerodha get_orders failed: {e}")
            return []

    # ── Safety ────────────────────────────────────────────────────────

    def kill_switch(self) -> list[OrderResponse]:
        """
        EMERGENCY: Cancel all open orders and close all positions at market.

        This is the nuclear option. Runs even if some individual operations
        fail — logs errors and continues.
        """
        if not self.is_connected:
            logger.error("KILL SWITCH: not connected to Zerodha!")
            return []

        responses = []

        # 1. Cancel all pending orders
        try:
            orders = self._kite.orders()
            for o in orders:
                if o["status"] in ("OPEN", "PENDING", "TRIGGER PENDING"):
                    try:
                        self._kite.cancel_order(variety="regular", order_id=o["order_id"])
                        logger.warning(f"KILL: cancelled order {o['order_id']}")
                    except Exception as e:
                        logger.error(f"KILL: cancel {o['order_id']} failed: {e}")
        except Exception as e:
            logger.error(f"KILL: fetch orders failed: {e}")

        # 2. Close all open positions
        try:
            positions = self.get_positions()
            for pos in positions:
                if pos.quantity > 0:
                    resp = self.sell(
                        pos.symbol, pos.quantity,
                        order_type=OrderType.MARKET,
                        tag="KILL_SWITCH",
                    )
                    responses.append(resp)
                    if resp.status == OrderStatus.ERROR:
                        # Retry once
                        logger.warning(f"KILL: retrying {pos.symbol}")
                        resp2 = self.sell(pos.symbol, pos.quantity, tag="KILL_RETRY")
                        responses.append(resp2)
                elif pos.quantity < 0:
                    resp = self.buy(
                        pos.symbol, abs(pos.quantity),
                        order_type=OrderType.MARKET,
                        tag="KILL_SWITCH",
                    )
                    responses.append(resp)
        except Exception as e:
            logger.error(f"KILL: close positions failed: {e}")

        logger.warning(f"KILL SWITCH COMPLETE: {len(responses)} exit orders placed")
        return responses

    # ── Helpers ────────────────────────────────────────────────────────

    def get_margins(self) -> dict:
        """Return available margin / funds."""
        if not self.is_connected:
            return {}
        try:
            return self._kite.margins()
        except Exception as e:
            logger.error(f"Margins fetch failed: {e}")
            return {}
