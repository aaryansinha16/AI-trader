"""
Market Regime Detector
──────────────────────
Detects the current market environment from 5-minute candle data.

From the docs (Product Vision §8):
  Possible regimes:
    TRENDING_BULL
    TRENDING_BEAR
    SIDEWAYS
    HIGH_VOLATILITY
    LOW_VOLATILITY

  Model types: RandomForest, HMM, Gradient Boosting
  Strategy selection adapts to regime.

  Example output: 10:15 AM → TRENDING BULL

For now this uses a rule-based detector. Once real data is available,
it can be upgraded to an ML-based classifier (RandomForest / HMM).
"""

from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("regime_detector")


class MarketRegime(str, Enum):
    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    UNKNOWN = "UNKNOWN"


class RegimeDetector:
    """
    Detects market regime using a combination of:
      - EMA trend (20 vs 50 on 5m candles)
      - ATR percentile (volatility)
      - ADX-like directional strength
      - Price range compression

    Returns a MarketRegime enum value.
    """

    def __init__(
        self,
        ema_short: int = 20,
        ema_long: int = 50,
        atr_period: int = 14,
        lookback: int = 50,
        vol_high_pct: float = 75,
        vol_low_pct: float = 25,
        trend_threshold: float = 0.002,
    ):
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.atr_period = atr_period
        self.lookback = lookback
        self.vol_high_pct = vol_high_pct
        self.vol_low_pct = vol_low_pct
        self.trend_threshold = trend_threshold

    def detect(self, df: pd.DataFrame) -> MarketRegime:
        """
        Detect regime from a DataFrame of 5-minute candles.
        Requires columns: open, high, low, close, volume.
        Uses the most recent `lookback` candles.
        """
        if df.empty or len(df) < self.ema_long + 5:
            logger.warning("Not enough data for regime detection.")
            return MarketRegime.UNKNOWN

        df = df.tail(max(self.lookback, self.ema_long + 20)).copy()

        # ── EMA Trend ────────────────────────────────────────────────────────
        df["_ema_s"] = df["close"].ewm(span=self.ema_short, adjust=False).mean()
        df["_ema_l"] = df["close"].ewm(span=self.ema_long, adjust=False).mean()

        ema_diff = (df["_ema_s"].iloc[-1] - df["_ema_l"].iloc[-1]) / df["_ema_l"].iloc[-1]
        ema_slope = (df["_ema_s"].iloc[-1] - df["_ema_s"].iloc[-5]) / df["_ema_s"].iloc[-5]

        # ── ATR / Volatility ─────────────────────────────────────────────────
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)

        atr = tr.rolling(self.atr_period).mean()
        atr_pct = atr / df["close"]  # ATR as % of price

        current_atr_pct = atr_pct.iloc[-1]
        atr_history = atr_pct.dropna()

        vol_high_threshold = np.percentile(atr_history, self.vol_high_pct)
        vol_low_threshold = np.percentile(atr_history, self.vol_low_pct)

        # ── Price Range Compression (sideways detection) ─────────────────────
        recent_high = df["high"].tail(20).max()
        recent_low = df["low"].tail(20).min()
        range_pct = (recent_high - recent_low) / df["close"].iloc[-1]

        # ── Classification Logic ─────────────────────────────────────────────

        # Priority 1: Extreme volatility
        if current_atr_pct > vol_high_threshold:
            regime = MarketRegime.HIGH_VOLATILITY
        elif current_atr_pct < vol_low_threshold:
            regime = MarketRegime.LOW_VOLATILITY
        # Priority 2: Clear trend
        elif ema_diff > self.trend_threshold and ema_slope > 0:
            regime = MarketRegime.TRENDING_BULL
        elif ema_diff < -self.trend_threshold and ema_slope < 0:
            regime = MarketRegime.TRENDING_BEAR
        # Priority 3: Sideways (range-bound)
        elif range_pct < 0.01:
            regime = MarketRegime.SIDEWAYS
        # Default
        elif ema_diff > 0:
            regime = MarketRegime.TRENDING_BULL
        elif ema_diff < 0:
            regime = MarketRegime.TRENDING_BEAR
        else:
            regime = MarketRegime.SIDEWAYS

        logger.info(
            f"Regime: {regime.value} "
            f"(ema_diff={ema_diff:.4f}, atr%={current_atr_pct:.4f}, "
            f"range%={range_pct:.4f})"
        )
        return regime

    def detect_with_details(self, df: pd.DataFrame) -> dict:
        """
        Detect regime and return full diagnostics.
        Useful for logging and dashboard display.
        """
        regime = self.detect(df)

        df = df.tail(max(self.lookback, self.ema_long + 20)).copy()

        df["_ema_s"] = df["close"].ewm(span=self.ema_short, adjust=False).mean()
        df["_ema_l"] = df["close"].ewm(span=self.ema_long, adjust=False).mean()

        return {
            "regime": regime.value,
            "ema_short": round(float(df["_ema_s"].iloc[-1]), 2),
            "ema_long": round(float(df["_ema_l"].iloc[-1]), 2),
            "last_close": round(float(df["close"].iloc[-1]), 2),
            "recent_high": round(float(df["high"].tail(20).max()), 2),
            "recent_low": round(float(df["low"].tail(20).min()), 2),
        }


# ── Strategy Regime Mapping ───────────────────────────────────────────────────

REGIME_STRATEGIES = {
    # Trending regimes: primary directional strategy gets the bonus
    MarketRegime.TRENDING_BULL: ["vwap_momentum_breakout"],
    MarketRegime.TRENDING_BEAR: ["bearish_momentum"],
    # Sideways: mean_reversion is primary, but momentum breakouts/breakdowns are valid too
    # (NIFTY can grind down from a sideways range — bearish_momentum is a real sideways setup)
    MarketRegime.SIDEWAYS: ["mean_reversion", "bearish_momentum", "vwap_momentum_breakout"],
    # High-vol: both reversion and momentum are valid (fast moves in both directions)
    MarketRegime.HIGH_VOLATILITY: ["mean_reversion", "bearish_momentum", "vwap_momentum_breakout"],
    MarketRegime.LOW_VOLATILITY: ["vwap_momentum_breakout"],
    MarketRegime.UNKNOWN: ["vwap_momentum_breakout", "bearish_momentum", "mean_reversion"],
}


def get_strategies_for_regime(regime: MarketRegime) -> list:
    """Return list of strategy names appropriate for the given regime."""
    return REGIME_STRATEGIES.get(regime, REGIME_STRATEGIES[MarketRegime.UNKNOWN])
