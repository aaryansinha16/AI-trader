"""
Market Stream (Kite WebSocket)
──────────────────────────────
Live tick feed from Zerodha Kite Connect WebSocket.
Used during market hours for real-time data ingestion.

Kite provides:
  - LTP, volume, OHLC, market depth
  - ~100-500ms latency
  - No historical tick data (use TrueData for that)
"""

from datetime import datetime
from typing import Callable, List, Optional

from config.settings import KITE_API_KEY, KITE_ACCESS_TOKEN
from utils.logger import get_logger

logger = get_logger("market_stream")


class KiteStream:
    """Wraps Kite Connect WebSocket ticker for live market data."""

    def __init__(self):
        self._ticker = None
        self._callbacks: List[Callable] = []
        self._instrument_tokens: List[int] = []

    def connect(self, instrument_tokens: List[int]):
        """
        Connect to Kite WebSocket and subscribe to given instrument tokens.
        Instrument tokens map to specific NIFTY/BANKNIFTY option contracts.
        """
        self._instrument_tokens = instrument_tokens

        try:
            from kiteconnect import KiteTicker

            self._ticker = KiteTicker(KITE_API_KEY, KITE_ACCESS_TOKEN)
            self._ticker.on_ticks = self._on_ticks
            self._ticker.on_connect = self._on_connect
            self._ticker.on_close = self._on_close
            self._ticker.on_error = self._on_error

            logger.info("Connecting to Kite WebSocket...")
            self._ticker.connect(threaded=True)

        except ImportError:
            logger.warning(
                "kiteconnect not installed or not configured. "
                "Live Kite stream unavailable."
            )
        except Exception as e:
            logger.error(f"Kite WebSocket connection failed: {e}")

    def _on_connect(self, ws, response):
        logger.info("Kite WebSocket connected.")
        if self._instrument_tokens:
            ws.subscribe(self._instrument_tokens)
            ws.set_mode(ws.MODE_FULL, self._instrument_tokens)
            logger.info(
                f"Subscribed to {len(self._instrument_tokens)} instruments."
            )

    def _on_ticks(self, ws, ticks):
        for raw in ticks:
            tick = self._parse_kite_tick(raw)
            for cb in self._callbacks:
                try:
                    cb(tick)
                except Exception as e:
                    logger.error(f"Tick callback error: {e}")

    def _on_close(self, ws, code, reason):
        logger.warning(f"Kite WebSocket closed: {code} – {reason}")

    def _on_error(self, ws, code, reason):
        logger.error(f"Kite WebSocket error: {code} – {reason}")

    def add_callback(self, callback: Callable):
        """Register a function to receive parsed tick dicts."""
        self._callbacks.append(callback)

    def disconnect(self):
        if self._ticker:
            try:
                self._ticker.close()
            except Exception:
                pass
        logger.info("Kite WebSocket disconnected.")

    # ── Tick Parsing ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_kite_tick(raw: dict) -> dict:
        """
        Convert Kite tick payload into our standard tick format.
        Kite MODE_FULL provides: ltp, volume, oi, depth, ohlc, etc.
        """
        depth = raw.get("depth", {})
        buy_depth = depth.get("buy", [{}])
        sell_depth = depth.get("sell", [{}])

        return {
            "timestamp": raw.get("exchange_timestamp", datetime.now()),
            "symbol": str(raw.get("instrument_token", "")),
            "price": float(raw.get("last_price", 0)),
            "volume": int(raw.get("volume_traded", 0)),
            "bid_price": float(buy_depth[0].get("price", 0)) if buy_depth else 0.0,
            "ask_price": float(sell_depth[0].get("price", 0)) if sell_depth else 0.0,
            "bid_qty": int(buy_depth[0].get("quantity", 0)) if buy_depth else 0,
            "ask_qty": int(sell_depth[0].get("quantity", 0)) if sell_depth else 0,
            "oi": int(raw.get("oi", 0)),
        }