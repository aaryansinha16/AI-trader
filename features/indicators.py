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
        df["macd_hist"] = macd_result.iloc[:, 1]
    else:
        df["macd"] = np.nan
        df["macd_signal"] = np.nan
        df["macd_hist"] = np.nan

    # ── Additional Momentum Indicators ────────────────────────────────────────
    # Stochastic RSI
    stoch = ta.stochrsi(df["close"], length=14)
    if stoch is not None and not stoch.empty:
        df["stoch_rsi_k"] = stoch.iloc[:, 0]
        df["stoch_rsi_d"] = stoch.iloc[:, 1]
    else:
        df["stoch_rsi_k"] = np.nan
        df["stoch_rsi_d"] = np.nan

    # Williams %R
    willr = ta.willr(df["high"], df["low"], df["close"], length=14)
    df["williams_r"] = willr if willr is not None else np.nan

    # Rate of Change (10-period and 20-period)
    df["roc_10"] = ta.roc(df["close"], length=10)
    df["roc_20"] = ta.roc(df["close"], length=20)

    # ADX (trend strength)
    adx_result = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx_result is not None and not adx_result.empty:
        df["adx"] = adx_result.iloc[:, 0]
        df["di_plus"] = adx_result.iloc[:, 1]
        df["di_minus"] = adx_result.iloc[:, 2]
    else:
        df["adx"] = np.nan
        df["di_plus"] = np.nan
        df["di_minus"] = np.nan

    # CCI (Commodity Channel Index)
    df["cci"] = ta.cci(df["high"], df["low"], df["close"], length=20)

    # ── Trend ─────────────────────────────────────────────────────────────────
    df["ema9"] = ta.ema(df["close"], length=9)
    df["ema20"] = ta.ema(df["close"], length=20)
    df["ema50"] = ta.ema(df["close"], length=50)
    df["sma200"] = ta.sma(df["close"], length=200)
    df["ema20_slope"] = df["ema20"].diff(5) / 5

    # EMA crossover signals
    df["ema9_20_cross"] = (df["ema9"] - df["ema20"]) / df["close"]
    df["ema20_50_cross"] = (df["ema20"] - df["ema50"]) / df["close"]
    df["close_above_sma200"] = (df["close"] > df["sma200"]).astype(int)

    # ── VWAP ──────────────────────────────────────────────────────────────────
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
    df["atr_pct"] = df["atr"] / df["close"]

    bbands = ta.bbands(df["close"], length=20, std=2)
    if bbands is not None and not bbands.empty:
        df["bollinger_upper"] = bbands.iloc[:, 2]
        df["bollinger_lower"] = bbands.iloc[:, 0]
        df["bollinger_width"] = (
            (df["bollinger_upper"] - df["bollinger_lower"]) / df["close"]
        )
        df["bollinger_pct"] = (
            (df["close"] - df["bollinger_lower"])
            / (df["bollinger_upper"] - df["bollinger_lower"]).replace(0, np.nan)
        )
    else:
        df["bollinger_upper"] = np.nan
        df["bollinger_lower"] = np.nan
        df["bollinger_width"] = np.nan
        df["bollinger_pct"] = np.nan

    # Rolling volatility (std of returns)
    df["returns_1m"] = df["close"].pct_change()
    df["volatility_20"] = df["returns_1m"].rolling(20).std()
    df["volatility_60"] = df["returns_1m"].rolling(60).std()
    df["vol_regime"] = df["volatility_20"] / df["volatility_60"].replace(0, np.nan)

    # ── Candle Pattern Features ───────────────────────────────────────────────
    body = df["close"] - df["open"]
    full_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["candle_body_pct"] = body / full_range
    df["upper_shadow_pct"] = (df["high"] - df[["open", "close"]].max(axis=1)) / full_range
    df["lower_shadow_pct"] = (df[["open", "close"]].min(axis=1) - df["low"]) / full_range

    # ── Multi-timeframe (5-min and 15-min lookbacks via rolling) ──────────────
    df["rsi_5m"] = ta.rsi(df["close"], length=70)      # 14 * 5
    df["rsi_15m"] = ta.rsi(df["close"], length=210)     # 14 * 15
    df["ema20_5m"] = ta.ema(df["close"], length=100)    # 20 * 5
    df["atr_5m"] = ta.atr(df["high"], df["low"], df["close"], length=70)

    # ── Session / Time Features ───────────────────────────────────────────────
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        # Minutes since market open (09:15 IST = 03:45 UTC)
        df["minutes_since_open"] = ts.dt.hour * 60 + ts.dt.minute - 225
        df["minutes_since_open"] = df["minutes_since_open"].clip(lower=0)
        # Normalize to 0-1 (375 min session)
        df["session_progress"] = df["minutes_since_open"] / 375.0
        # Day of week (0=Mon, 4=Fri)
        df["day_of_week"] = ts.dt.dayofweek
        # Is it first/last hour?
        df["is_first_hour"] = (df["minutes_since_open"] <= 60).astype(int)
        df["is_last_hour"] = (df["minutes_since_open"] >= 315).astype(int)

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

    # Volume momentum: change in volume vs previous bar
    df["volume_change"] = df["volume"].pct_change()

    # Cumulative volume delta (proxy: classify bars as buy/sell by close vs open)
    df["volume_delta"] = np.where(
        df["close"] >= df["open"], df["volume"], -df["volume"]
    )
    df["cum_volume_delta_20"] = df["volume_delta"].rolling(20).sum()

    # On-Balance Volume (OBV) normalized
    obv = ta.obv(df["close"], df["volume"])
    if obv is not None:
        df["obv"] = obv
        df["obv_slope"] = df["obv"].diff(10) / df["obv"].rolling(10).mean().replace(0, np.nan)
    else:
        df["obv"] = np.nan
        df["obv_slope"] = np.nan

    # Money Flow Index
    mfi = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)
    df["mfi"] = mfi if mfi is not None else np.nan

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
        # Only fill with NaN if not already enriched (e.g. by option_chain_builder)
        if "pcr" not in df.columns:
            df["pcr"] = np.nan
        if "oi_change" not in df.columns:
            df["oi_change"] = np.nan
        if "iv" not in df.columns:
            df["iv"] = np.nan

    return df


def compute_options_aware_features(
    df: pd.DataFrame,
    option_chain_df: pd.DataFrame = None,
    expiry=None,
    relative_strike: int = 0,
) -> pd.DataFrame:
    """
    Add options-structure features: relative_strike, days_to_expiry,
    theta_pressure, and cross-strike aggregate features.

    Args:
        df: DataFrame with 'timestamp' column
        option_chain_df: option chain with relative_strike, oi, option_type, iv
        expiry: expiry date (date object) for DTE calculation
        relative_strike: which relative strike this data represents (0=ATM)
    """
    from features.options_features import (
        compute_days_to_expiry,
        compute_theta_pressure,
        compute_cross_strike_features,
    )

    df = df.copy()

    # ── relative_strike (static per symbol) ─────────────────────────────
    if "relative_strike" not in df.columns:
        df["relative_strike"] = relative_strike

    # ── days_to_expiry + theta_pressure (per row) ───────────────────────
    if expiry is not None and "timestamp" in df.columns:
        df["days_to_expiry"] = df["timestamp"].apply(
            lambda ts: compute_days_to_expiry(ts, expiry)
        )
        df["theta_pressure"] = df["days_to_expiry"].apply(compute_theta_pressure)
    else:
        if "days_to_expiry" not in df.columns:
            df["days_to_expiry"] = np.nan
        if "theta_pressure" not in df.columns:
            df["theta_pressure"] = np.nan

    # ── Cross-strike aggregate features (same for all rows in a snapshot) ─
    # Only compute from option_chain_df if columns aren't already present
    cross_feats = compute_cross_strike_features(option_chain_df)
    for feat_name, feat_val in cross_feats.items():
        if feat_name not in df.columns:
            df[feat_name] = feat_val

    return df


def compute_all_macro_indicators(
    df: pd.DataFrame,
    option_chain_df: pd.DataFrame = None,
    expiry=None,
    relative_strike: int = 0,
) -> pd.DataFrame:
    """
    Full macro indicator pipeline: price + volume + options + options-structure.
    Returns DataFrame ready for Macro ML Model feature extraction.

    Args:
        df: 1-minute OHLCV DataFrame
        option_chain_df: option chain for cross-strike features
        expiry: expiry date for DTE calculation
        relative_strike: which relative strike this data represents
    """
    logger.info(f"Computing macro indicators on {len(df)} rows...")

    df = compute_price_indicators(df)
    df = compute_volume_signals(df)
    df = compute_options_signals(df, option_chain_df)
    df = compute_options_aware_features(df, option_chain_df, expiry, relative_strike)

    logger.info(f"Macro indicators computed. Columns: {list(df.columns)}")
    return df