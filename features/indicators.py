"""
Technical Indicators
────────────────────
Computes all technical indicators defined in the Product Vision doc:

Price indicators:  RSI, MACD, EMA 20, EMA 50, VWAP, Bollinger Bands, ATR
Volume signals:    relative volume, volume spikes, volume SMA
Options signals:   OI change, PCR, IV, ATM premium momentum

These feed into the Macro Feature set for the Macro ML Model.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta

from utils.logger import get_logger

logger = get_logger("indicators")


def compute_price_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all price-based technical indicators.
    Input: DataFrame with columns [open, high, low, close, volume]
    """
    df = df.copy()

    # ── Momentum ──────────────────────────────────────────────────────────────
    df["rsi"] = ta.rsi(df["close"], length=14)

    macd_result = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd_result is not None and not macd_result.empty:
        df["macd"] = macd_result.iloc[:, 0]
        df["macd_signal"] = macd_result.iloc[:, 2]
    else:
        df["macd"] = np.nan
        df["macd_signal"] = np.nan

    # ── Trend ─────────────────────────────────────────────────────────────────
    df["ema20"] = ta.ema(df["close"], length=20)
    df["ema50"] = ta.ema(df["close"], length=50)
    df["ema20_slope"] = df["ema20"].diff(5) / 5

    # ── VWAP ──────────────────────────────────────────────────────────────────
    # pandas_ta.vwap requires a DatetimeIndex
    had_dt_index = isinstance(df.index, pd.DatetimeIndex)
    if not had_dt_index and "timestamp" in df.columns:
        df = df.set_index("timestamp")

    vwap = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
    if vwap is not None and not vwap.empty:
        df["vwap"] = vwap
    else:
        typical = (df["high"] + df["low"] + df["close"]) / 3
        cum_tp_vol = (typical * df["volume"]).cumsum()
        cum_vol = df["volume"].cumsum().replace(0, np.nan)
        df["vwap"] = cum_tp_vol / cum_vol

    if not had_dt_index:
        df = df.reset_index()

    df["vwap_dist"] = (df["close"] - df["vwap"]) / df["vwap"]

    # ── Volatility ────────────────────────────────────────────────────────────
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    bbands = ta.bbands(df["close"], length=20, std=2)
    if bbands is not None and not bbands.empty:
        df["bollinger_upper"] = bbands.iloc[:, 2]
        df["bollinger_lower"] = bbands.iloc[:, 0]
        df["bollinger_width"] = (
            (df["bollinger_upper"] - df["bollinger_lower"]) / df["close"]
        )
    else:
        df["bollinger_upper"] = np.nan
        df["bollinger_lower"] = np.nan
        df["bollinger_width"] = np.nan

    return df


def compute_volume_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volume-based signals.
    Input: DataFrame with column [volume]
    """
    df = df.copy()

    df["volume_sma20"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma20"].replace(0, np.nan)

    # Volume spike: volume > 2x 20-period average
    df["volume_spike"] = (df["volume_ratio"] > 2.0).astype(int)

    return df


def compute_options_signals(
    df: pd.DataFrame,
    option_chain_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Merge options-derived signals into the feature DataFrame.
    If option_chain_df is provided, compute PCR, aggregated OI change, IV.
    Otherwise, fill with NaN (will be populated during live trading).
    """
    df = df.copy()

    if option_chain_df is not None and not option_chain_df.empty:
        oc = option_chain_df.copy()

        # Put-Call Ratio
        ce_oi = oc[oc["option_type"] == "CE"]["oi"].sum()
        pe_oi = oc[oc["option_type"] == "PE"]["oi"].sum()
        pcr = pe_oi / ce_oi if ce_oi > 0 else np.nan

        # Net OI change
        oi_change = oc["oi_change"].sum()

        # Average IV (ATM strikes)
        avg_iv = oc["iv"].mean()

        df["pcr"] = pcr
        df["oi_change"] = oi_change
        df["iv"] = avg_iv
    else:
        df["pcr"] = np.nan
        df["oi_change"] = np.nan
        df["iv"] = np.nan

    return df


def compute_all_macro_indicators(
    df: pd.DataFrame,
    option_chain_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Full macro indicator pipeline: price + volume + options.
    Returns DataFrame ready for Macro ML Model feature extraction.
    """
    logger.info(f"Computing macro indicators on {len(df)} rows...")

    df = compute_price_indicators(df)
    df = compute_volume_signals(df)
    df = compute_options_signals(df, option_chain_df)

    logger.info(f"Macro indicators computed. Columns: {list(df.columns)}")
    return df