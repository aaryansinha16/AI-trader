"""
Tick Collector
──────────────
Central tick ingestion service. Receives ticks from any source
(Kite WebSocket, TrueData live stream, or mock generator) and:

  1. Writes raw ticks to tick_data table
  2. Notifies the aggregation engine for candle building
  3. Buffers ticks in-memory for micro-feature computation
"""

from datetime import datetime
from typing import Callable, Dict, List, Optional

import pandas as pd

from database.db import write_df
from utils.logger import get_logger

logger = get_logger("tick_collector")


class TickCollector:
    """Collects ticks and persists them, with optional in-memory buffer."""

    def __init__(self, buffer_size: int = 500):
        self._buffer: List[Dict] = []
        self._buffer_size = buffer_size
        self._listeners: List[Callable] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def on_tick(self, tick: dict):
        """
        Process a single tick. Expected keys:
          timestamp, symbol, price, volume,
          bid_price, ask_price, bid_qty, ask_qty, oi
        """
        tick.setdefault("timestamp", datetime.now())
        tick.setdefault("bid_price", None)
        tick.setdefault("ask_price", None)
        tick.setdefault("bid_qty", None)
        tick.setdefault("ask_qty", None)
        tick.setdefault("oi", None)

        self._buffer.append(tick)

        # Notify listeners (aggregation engine, micro-feature builder, etc.)
        for listener in self._listeners:
            try:
                listener(tick)
            except Exception as e:
                logger.error(f"Tick listener error: {e}")

        # Flush buffer when full
        if len(self._buffer) >= self._buffer_size:
            self.flush()

    def flush(self):
        """Persist buffered ticks to the database."""
        if not self._buffer:
            return

        df = pd.DataFrame(self._buffer)
        cols = [
            "timestamp", "symbol", "price", "volume",
            "bid_price", "ask_price", "bid_qty", "ask_qty", "oi",
        ]
        for c in cols:
            if c not in df.columns:
                df[c] = None

        try:
            write_df(df[cols], "tick_data")
            logger.info(f"Flushed {len(self._buffer)} ticks to database.")
        except Exception as e:
            logger.error(f"Failed to flush ticks: {e}")

        self._buffer.clear()

    def add_listener(self, callback: Callable):
        """Register a callback that receives every tick dict."""
        self._listeners.append(callback)

    def get_buffer(self) -> List[Dict]:
        """Return current in-memory buffer (for micro-feature computation)."""
        return list(self._buffer)

    def get_buffer_df(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """Return buffer as DataFrame, optionally filtered by symbol."""
        if not self._buffer:
            return pd.DataFrame()
        df = pd.DataFrame(self._buffer)
        if symbol:
            df = df[df["symbol"] == symbol]
        return df

    # ── Bulk Ingest (for loading historical ticks) ────────────────────────────

    def ingest_historical_ticks(self, df: pd.DataFrame):
        """
        Write a DataFrame of historical ticks directly to the database.
        Used when loading TrueData's 5-day tick history.
        """
        cols = [
            "timestamp", "symbol", "price", "volume",
            "bid_price", "ask_price", "bid_qty", "ask_qty", "oi",
        ]
        for c in cols:
            if c not in df.columns:
                df[c] = None

        try:
            write_df(df[cols], "tick_data")
            logger.info(f"Ingested {len(df)} historical ticks.")
        except Exception as e:
            logger.error(f"Failed to ingest historical ticks: {e}")