"""
Aggregation Engine
──────────────────
Converts raw tick data into multi-timeframe candles:
  - 1-second candles  → used for micro-feature signals
  - 1-minute candles  → primary ML training timeframe (Macro Model)
  - 5-minute candles  → regime detection timeframe

Architecture (from docs):
  Tick Feed → Tick Database → Aggregation Engine → 1s / 1m / 5m candles

Supports two modes:
  1. Real-time: aggregate ticks streaming through TickCollector
  2. Batch: aggregate historical ticks already in the database
"""

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from database.db import read_sql, write_df
from utils.logger import get_logger

logger = get_logger("aggregator")


class AggregationEngine:
    """Builds OHLCV candles at multiple timeframes from raw ticks."""

    # ── Batch Aggregation (from DB) ───────────────────────────────────────────

    def aggregate_from_db(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ):
        """
        Read ticks from the database and produce 1s, 1m, 5m candles.
        Writes results back to the respective candle tables.
        """
        where = "WHERE symbol = :symbol"
        params = {"symbol": symbol}

        if start:
            where += " AND timestamp >= :start"
            params["start"] = start
        if end:
            where += " AND timestamp <= :end"
            params["end"] = end

        query = f"SELECT * FROM tick_data {where} ORDER BY timestamp"
        df = read_sql(query, params)

        if df.empty:
            logger.warning(f"No ticks found for {symbol} to aggregate.")
            return

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()

        # Generate all timeframes
        self._build_and_store(df, symbol, "1s", "second_candles")
        self._build_and_store(df, symbol, "1min", "minute_candles")
        self._build_and_store(df, symbol, "5min", "five_minute_candles")

    # ── DataFrame-based Aggregation (for ticks already in memory) ─────────────

    def aggregate_ticks_df(self, tick_df: pd.DataFrame, symbol: str):
        """
        Aggregate a DataFrame of ticks (e.g. from TickCollector buffer).
        Returns dict of {"1s": df, "1m": df, "5m": df}.
        """
        if tick_df.empty:
            return {}

        tick_df = tick_df.copy()
        tick_df["timestamp"] = pd.to_datetime(tick_df["timestamp"])
        tick_df = tick_df.set_index("timestamp").sort_index()

        results = {}
        for freq, label in [("1s", "1s"), ("1min", "1m"), ("5min", "5m")]:
            candles = self._resample(tick_df, freq, symbol)
            if not candles.empty:
                results[label] = candles

        return results

    # ── Store Historical Minute Bars Directly (from TrueData) ─────────────────

    def ingest_minute_bars(self, df: pd.DataFrame):
        """
        Write pre-built 1-minute bars (e.g. from TrueData historical API)
        directly into the minute_candles table.
        Used for the 6-month Macro Model training dataset.
        """
        required = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                logger.error(f"Missing column '{col}' in minute bars DataFrame.")
                return

        df = df.copy()
        if "vwap" not in df.columns:
            df["vwap"] = self._compute_vwap(df)

        try:
            write_df(df[required + ["vwap"]], "minute_candles")
            logger.info(f"Ingested {len(df)} minute bars into database.")
        except Exception as e:
            logger.error(f"Failed to ingest minute bars: {e}")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _resample(
        self, df: pd.DataFrame, freq: str, symbol: str
    ) -> pd.DataFrame:
        """Resample tick DataFrame into OHLCV candles at given frequency."""
        if "price" not in df.columns:
            return pd.DataFrame()

        ohlcv = df["price"].resample(freq).ohlc()
        ohlcv.columns = ["open", "high", "low", "close"]
        ohlcv["volume"] = df["volume"].resample(freq).sum()
        ohlcv = ohlcv.dropna(subset=["open"])
        ohlcv["symbol"] = symbol

        # Compute VWAP for minute+ candles
        if freq in ("1min", "5min"):
            ohlcv["vwap"] = self._compute_vwap(ohlcv)

        ohlcv = ohlcv.reset_index()
        ohlcv.rename(columns={"index": "timestamp"}, inplace=True)
        return ohlcv

    def _build_and_store(
        self, df: pd.DataFrame, symbol: str, freq: str, table: str
    ):
        """Resample and write to DB."""
        candles = self._resample(df, freq, symbol)
        if candles.empty:
            logger.warning(f"No {freq} candles produced for {symbol}.")
            return

        cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        if "vwap" in candles.columns:
            cols.append("vwap")

        try:
            write_df(candles[cols], table)
            logger.info(
                f"Wrote {len(candles)} {freq} candles for {symbol} → {table}"
            )
        except Exception as e:
            logger.error(f"Failed to write {freq} candles: {e}")

    @staticmethod
    def _compute_vwap(df: pd.DataFrame) -> pd.Series:
        """
        Approximate VWAP from OHLCV data.
        VWAP = cumulative(typical_price * volume) / cumulative(volume)
        """
        typical = (df["high"] + df["low"] + df["close"]) / 3
        cum_tp_vol = (typical * df["volume"]).cumsum()
        cum_vol = df["volume"].cumsum()
        vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
        return vwap