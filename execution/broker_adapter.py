"""
Broker Adapter (Zerodha Kite Connect)
─────────────────────────────────────
Handles all communication with the Zerodha Kite API.

Order types supported:
  - Market order (entry)
  - SL-M order  (exchange-managed stop loss)
  - Limit order  (target exit)
  - Cancel order

From the docs (§15):
  Stop loss should be exchange-managed.
"""

from typing import Optional

from config.settings import KITE_API_KEY, KITE_ACCESS_TOKEN
from utils.logger import get_logger

logger = get_logger("broker_adapter")


class BrokerAdapter:
    """
    Wraps Zerodha Kite Connect API for order execution.
    Lazy-initializes the KiteConnect client on first use.
    """

    def __init__(self):
        self._kite = None

    def _get_kite(self):
        if self._kite is None:
            try:
                from kiteconnect import KiteConnect

                self._kite = KiteConnect(api_key=KITE_API_KEY)
                self._kite.set_access_token(KITE_ACCESS_TOKEN)
                logger.info("Kite Connect initialized.")
            except ImportError:
                logger.error(
                    "kiteconnect package not installed. "
                    "Install with: pip install kiteconnect"
                )
                raise
            except Exception as e:
                logger.error(f"Kite Connect initialization failed: {e}")
                raise
        return self._kite

    # ── Entry Order ───────────────────────────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        qty: int,
        side: str = "BUY",
        order_type: str = "MARKET",
        price: float = 0,
        exchange: str = "NFO",
        product: str = "MIS",
    ) -> Optional[str]:
        """
        Place an entry order. Returns the Kite order ID.

        Args:
            symbol: trading symbol (e.g. "NIFTY2540322500CE")
            qty: quantity
            side: "BUY" or "SELL"
            order_type: "MARKET" or "LIMIT"
            price: limit price (only for LIMIT orders)
            exchange: "NFO" for F&O
            product: "MIS" for intraday
        """
        kite = self._get_kite()

        params = {
            "tradingsymbol": symbol,
            "exchange": exchange,
            "transaction_type": side,
            "quantity": qty,
            "order_type": order_type,
            "product": product,
            "variety": "regular",
        }

        if order_type == "LIMIT" and price > 0:
            params["price"] = price

        try:
            order_id = kite.place_order(**params)
            logger.info(f"Order placed: {symbol} {side} qty={qty} → {order_id}")
            return str(order_id)
        except Exception as e:
            logger.error(f"Order failed: {symbol} {side} qty={qty} – {e}")
            raise

    # ── Stop Loss Order (Exchange-Managed SL-M) ──────────────────────────────

    def place_sl_order(
        self,
        symbol: str,
        qty: int,
        trigger_price: float,
        side: str = "SELL",
        exchange: str = "NFO",
        product: str = "MIS",
    ) -> Optional[str]:
        """
        Place a stop-loss market (SL-M) order.
        This is exchange-managed – executes automatically when trigger is hit.
        """
        kite = self._get_kite()

        try:
            order_id = kite.place_order(
                tradingsymbol=symbol,
                exchange=exchange,
                transaction_type=side,
                quantity=qty,
                order_type="SL-M",
                trigger_price=trigger_price,
                product=product,
                variety="regular",
            )
            logger.info(f"SL order placed: {symbol} trigger={trigger_price} → {order_id}")
            return str(order_id)
        except Exception as e:
            logger.error(f"SL order failed: {symbol} trigger={trigger_price} – {e}")
            raise

    # ── Target Order (Limit) ──────────────────────────────────────────────────

    def place_target_order(
        self,
        symbol: str,
        qty: int,
        price: float,
        side: str = "SELL",
        exchange: str = "NFO",
        product: str = "MIS",
    ) -> Optional[str]:
        """Place a limit order as target exit."""
        return self.place_order(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type="LIMIT",
            price=price,
            exchange=exchange,
            product=product,
        )

    # ── Cancel Order ──────────────────────────────────────────────────────────

    def cancel_order(self, order_id: str, variety: str = "regular") -> bool:
        """Cancel an open order by its Kite order ID."""
        kite = self._get_kite()
        try:
            kite.cancel_order(variety=variety, order_id=order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Cancel failed: {order_id} – {e}")
            return False

    # ── Order Status ──────────────────────────────────────────────────────────

    def get_order_status(self, order_id: str) -> dict:
        """Fetch status of a specific order."""
        kite = self._get_kite()
        try:
            orders = kite.order_history(order_id)
            return orders[-1] if orders else {}
        except Exception as e:
            logger.error(f"Status fetch failed: {order_id} – {e}")
            return {}

    def get_positions(self) -> dict:
        """Fetch current positions."""
        kite = self._get_kite()
        try:
            return kite.positions()
        except Exception as e:
            logger.error(f"Positions fetch failed: {e}")
            return {}


# ── Legacy compatibility ──────────────────────────────────────────────────────

_adapter: Optional[BrokerAdapter] = None


def place_order(symbol: str, qty: int, side: str):
    """Legacy function."""
    global _adapter
    if _adapter is None:
        _adapter = BrokerAdapter()
    _adapter.place_order(symbol, qty, side)