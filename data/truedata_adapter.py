"""
TrueData Adapter
─────────────────
Provides two modes of data access:

1. Historical data (REST)  – 6 months of 1m bars, 5 days of tick data
2. Real-time streaming     – live tick feed via WebSocket

Used for:
  - ML training (historical 1m bars  → Macro Model)
  - ML training (historical tick data → Micro Model)
  - Live tick ingestion during market hours
"""

import time
from datetime import datetime, timedelta
from typing import Callable, List, Optional

import pandas as pd

from config.settings import TRUEDATA_USER, TRUEDATA_PASSWORD, SYMBOLS
from utils.logger import get_logger

logger = get_logger("truedata")


class TrueDataAdapter:
    """Wrapper around the TrueData API (real-time + historical)."""

    def __init__(self):
        self._connected = False
        self._callbacks: List[Callable] = []
        self._td = None

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self):
        """Connect to TrueData. Requires truedata_ws package when live."""
        try:
            from truedata_ws.TD import TD

            self._td = TD(TRUEDATA_USER, TRUEDATA_PASSWORD, live_port=8082)
            self._connected = True
            logger.info("Connected to TrueData.")
        except ImportError:
            logger.warning(
                "truedata_ws not installed. Using mock mode. "
                "Install with: pip install truedata_ws"
            )
            self._connected = False
        except Exception as e:
            logger.error(f"TrueData connection failed: {e}")
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Historical Minute Bars (6 months) ─────────────────────────────────────

    def fetch_historical_minute_bars(
        self,
        symbol: str,
        days: int = 180,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical 1-minute OHLCV bars.
        This powers the **Macro ML Model** training.
        """
        if not self._connected:
            logger.warning("TrueData not connected; returning empty DataFrame.")
            return pd.DataFrame()

        end_date = end_date or datetime.now()
        start_date = end_date - timedelta(days=days)

        logger.info(
            f"Fetching 1m bars for {symbol}: "
            f"{start_date.date()} → {end_date.date()}"
        )

        try:
            hist = self._td.get_historic_data(
                symbol,
                start_time=start_date,
                end_time=end_date,
                bar_size="1 min",
            )
            df = pd.DataFrame(hist)
            df.columns = [c.lower().strip() for c in df.columns]

            rename_map = {
                "time": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
            }
            df.rename(columns=rename_map, inplace=True)
            df["symbol"] = symbol
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            logger.info(f"Fetched {len(df)} minute bars for {symbol}.")
            return df

        except Exception as e:
            logger.error(f"Error fetching minute bars for {symbol}: {e}")
            return pd.DataFrame()

    # ── Historical Tick Data (5 days) ─────────────────────────────────────────

    def fetch_historical_ticks(
        self,
        symbol: str,
        days: int = 5,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical tick data (up to 5 trading days from TrueData).
        This powers the **Microstructure Model** training.
        """
        if not self._connected:
            logger.warning("TrueData not connected; returning empty DataFrame.")
            return pd.DataFrame()

        end_date = end_date or datetime.now()
        start_date = end_date - timedelta(days=days)

        logger.info(
            f"Fetching ticks for {symbol}: "
            f"{start_date.date()} → {end_date.date()}"
        )

        try:
            hist = self._td.get_historic_data(
                symbol,
                start_time=start_date,
                end_time=end_date,
                bar_size="tick",
            )
            df = pd.DataFrame(hist)
            df.columns = [c.lower().strip() for c in df.columns]

            rename_map = {
                "time": "timestamp",
                "ltp": "price",
                "ltq": "volume",
                "bidprice": "bid_price",
                "askprice": "ask_price",
                "bidqty": "bid_qty",
                "askqty": "ask_qty",
            }
            df.rename(columns=rename_map, inplace=True)
            df["symbol"] = symbol
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            logger.info(f"Fetched {len(df)} ticks for {symbol}.")
            return df

        except Exception as e:
            logger.error(f"Error fetching ticks for {symbol}: {e}")
            return pd.DataFrame()

    # ── Real-time Tick Streaming ──────────────────────────────────────────────

    def subscribe_live(self, symbols: List[str], callback: Callable):
        """
        Subscribe to real-time tick stream.
        Callback receives a dict per tick:
          {timestamp, symbol, price, volume, bid_price, ask_price, bid_qty, ask_qty, oi}
        """
        if not self._connected:
            logger.warning("TrueData not connected; cannot subscribe.")
            return

        self._callbacks.append(callback)

        try:
            req_ids = self._td.start_live_data(symbols)
            logger.info(f"Subscribed to live feed: {symbols}")

            while True:
                live = self._td.live_data
                for sym in symbols:
                    if sym in live:
                        tick = live[sym]
                        tick_dict = {
                            "timestamp": datetime.now(),
                            "symbol": sym,
                            "price": float(getattr(tick, "ltp", 0)),
                            "volume": int(getattr(tick, "ltq", 0)),
                            "bid_price": float(getattr(tick, "bidprice", 0)),
                            "ask_price": float(getattr(tick, "askprice", 0)),
                            "bid_qty": int(getattr(tick, "bidqty", 0)),
                            "ask_qty": int(getattr(tick, "askqty", 0)),
                            "oi": int(getattr(tick, "oi", 0)),
                        }
                        for cb in self._callbacks:
                            cb(tick_dict)
                time.sleep(0.25)

        except Exception as e:
            logger.error(f"Live stream error: {e}")

    # ── Fetch All Historical Data (convenience) ───────────────────────────────

    def fetch_all_historical(self) -> dict:
        """
        Fetch both minute bars (6mo) and tick data (5d) for all symbols.
        Returns {"minute_bars": DataFrame, "ticks": DataFrame}.
        """
        all_minutes = []
        all_ticks = []

        for symbol in SYMBOLS:
            minute_df = self.fetch_historical_minute_bars(symbol, days=180)
            if not minute_df.empty:
                all_minutes.append(minute_df)

            tick_df = self.fetch_historical_ticks(symbol, days=5)
            if not tick_df.empty:
                all_ticks.append(tick_df)

        return {
            "minute_bars": pd.concat(all_minutes, ignore_index=True) if all_minutes else pd.DataFrame(),
            "ticks": pd.concat(all_ticks, ignore_index=True) if all_ticks else pd.DataFrame(),
        }

    def disconnect(self):
        if self._td:
            try:
                self._td.disconnect()
            except Exception:
                pass
        self._connected = False
        logger.info("TrueData disconnected.")
