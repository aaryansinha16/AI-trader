"""
Feature Engine
──────────────
Orchestrates the two-model feature pipeline:

  1. Macro Features  (from 1m candles)  → features_macro table
  2. Micro Features  (from tick data)   → features_micro table

From the docs (Elaborated Challenges):
  Historical 1m data (6 months) → Macro ML Model
  Tick data (5 days + ongoing)  → Microstructure Model
  Both outputs                  → Final trade decision
"""

from typing import Optional

import pandas as pd

from config.settings import FEATURE_COLUMNS_MACRO, FEATURE_COLUMNS_MICRO
from database.db import read_sql, write_df
from features.indicators import compute_all_macro_indicators
from features.micro_features import compute_micro_features
from utils.logger import get_logger

logger = get_logger("feature_engine")


def build_macro_features(
    symbol: str,
    option_chain_df: Optional[pd.DataFrame] = None,
    limit: int = 0,
) -> pd.DataFrame:
    """
    Build macro feature set from 1-minute candles.
    Reads from minute_candles table, computes all indicators,
    writes to features_macro table, and returns the DataFrame.
    """
    query = (
        "SELECT * FROM minute_candles "
        "WHERE symbol = :symbol ORDER BY timestamp"
    )
    df = read_sql(query, {"symbol": symbol})

    if df.empty:
        logger.warning(f"No minute candles found for {symbol}.")
        return pd.DataFrame()

    if limit > 0:
        df = df.tail(limit)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Compute all macro indicators
    df = compute_all_macro_indicators(df, option_chain_df)

    # Select feature columns that exist
    available_cols = ["timestamp", "symbol"] + [
        c for c in FEATURE_COLUMNS_MACRO if c in df.columns
    ]
    features_df = df[available_cols].copy()

    # Persist to DB
    try:
        write_df(features_df, "features_macro", if_exists="replace")
        logger.info(
            f"Wrote {len(features_df)} macro feature rows for {symbol}."
        )
    except Exception as e:
        logger.error(f"Failed to write macro features: {e}")

    return features_df


def build_micro_features(
    symbol: str,
    tick_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build micro feature set from tick data.
    If tick_df is provided, use it directly. Otherwise read from DB.
    Writes to features_micro table and returns the DataFrame.
    """
    if tick_df is None or tick_df.empty:
        query = (
            "SELECT * FROM tick_data "
            "WHERE symbol = :symbol ORDER BY timestamp"
        )
        tick_df = read_sql(query, {"symbol": symbol})

    if tick_df.empty:
        logger.warning(f"No tick data found for {symbol}.")
        return pd.DataFrame()

    # Compute micro features
    features_df = compute_micro_features(tick_df)

    if features_df.empty:
        return features_df

    # Persist to DB
    available_cols = ["timestamp", "symbol"] + [
        c for c in FEATURE_COLUMNS_MICRO if c in features_df.columns
    ]
    out = features_df[available_cols].copy()

    try:
        write_df(out, "features_micro", if_exists="replace")
        logger.info(
            f"Wrote {len(out)} micro feature rows for {symbol}."
        )
    except Exception as e:
        logger.error(f"Failed to write micro features: {e}")

    return out


def build_all_features(
    symbol: str,
    option_chain_df: Optional[pd.DataFrame] = None,
    tick_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Build both macro and micro features for a symbol.
    Returns {"macro": DataFrame, "micro": DataFrame}.
    """
    macro = build_macro_features(symbol, option_chain_df)
    micro = build_micro_features(symbol, tick_df)

    logger.info(
        f"Features built for {symbol}: "
        f"{len(macro)} macro rows, {len(micro)} micro rows."
    )
    return {"macro": macro, "micro": micro}


def build_feature_dataset():
    """
    Legacy compatibility wrapper.
    Builds macro features for all configured symbols.
    """
    from config.settings import SYMBOLS
    for symbol in SYMBOLS:
        build_macro_features(symbol)