"""
Option Chain Builder
────────────────────
Builds a time-aligned option chain from per-symbol 1-min bars stored in
minute_candles.  This produces per-minute PCR, OI change, and aggregate
option signals that enrich the index features for the Macro ML Model.

The output is a DataFrame indexed by timestamp with columns:
  pcr, oi_change, total_ce_oi, total_pe_oi, ce_volume, pe_volume,
  oi_skew, pcr_near_atm, pcr_far, oi_concentration,
  call_oi_gradient, put_oi_gradient
"""

import re
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from database.db import read_sql
from utils.logger import get_logger

logger = get_logger("option_chain_builder")

# Regex to parse NIFTY option symbols: NIFTY<YYMMDD><strike><CE|PE>
_OPT_RE = re.compile(r"NIFTY(\d{6})(\d+)(CE|PE)")


def parse_option_symbol(symbol: str) -> Optional[dict]:
    """Extract expiry_str, strike, option_type from an option symbol."""
    m = _OPT_RE.match(symbol)
    if not m:
        return None
    exp_str, strike_str, opt_type = m.groups()
    return {
        "expiry_str": exp_str,
        "strike": int(strike_str),
        "option_type": opt_type,
    }


def build_option_chain_timeseries(
    index_df: pd.DataFrame = None,
    strike_gap: int = 50,
    atm_range: int = 3,
) -> pd.DataFrame:
    """
    Build per-minute option chain features from all option bars in the DB.
    Uses vectorized pandas operations for speed (~881K rows).

    Args:
        index_df: Optional index DataFrame with 'timestamp' and 'close'
                  to determine ATM strike per minute.
        strike_gap: gap between strikes (50 for NIFTY)
        atm_range: number of strikes around ATM to consider "near"

    Returns:
        DataFrame with timestamp index and option chain feature columns.
    """
    logger.info("Loading all option bars from DB...")

    query = """
        SELECT timestamp, symbol, close, volume, oi
        FROM minute_candles
        WHERE symbol LIKE 'NIFTY%%CE' OR symbol LIKE 'NIFTY%%PE'
        ORDER BY timestamp
    """
    opt_df = read_sql(query)

    if opt_df.empty:
        logger.warning("No option bars found in DB.")
        return pd.DataFrame()

    logger.info(f"Loaded {len(opt_df):,} option bar rows. Parsing symbols...")

    # Vectorized symbol parsing
    extracted = opt_df["symbol"].str.extract(r"NIFTY(\d{6})(\d+)(CE|PE)")
    opt_df["expiry_str"] = extracted[0]
    opt_df["strike"] = extracted[1].astype(float).astype("Int64")
    opt_df["option_type"] = extracted[2]
    opt_df = opt_df.dropna(subset=["option_type"])
    opt_df["timestamp"] = pd.to_datetime(opt_df["timestamp"])

    # ── Basic per-minute aggregations (vectorized) ──────────────────────
    logger.info("Computing per-minute option chain aggregates...")

    ce_mask = opt_df["option_type"] == "CE"
    pe_mask = opt_df["option_type"] == "PE"

    ce_agg = opt_df[ce_mask].groupby("timestamp").agg(
        total_ce_oi=("oi", "sum"),
        ce_volume=("volume", "sum"),
    )
    pe_agg = opt_df[pe_mask].groupby("timestamp").agg(
        total_pe_oi=("oi", "sum"),
        pe_volume=("volume", "sum"),
    )

    chain_df = ce_agg.join(pe_agg, how="outer").fillna(0)
    chain_df["total_oi"] = chain_df["total_ce_oi"] + chain_df["total_pe_oi"]
    chain_df["pcr"] = np.where(
        chain_df["total_ce_oi"] > 0,
        chain_df["total_pe_oi"] / chain_df["total_ce_oi"],
        np.nan,
    )
    chain_df["oi_skew"] = np.where(
        chain_df["total_oi"] > 0,
        (chain_df["total_ce_oi"] - chain_df["total_pe_oi"]) / chain_df["total_oi"],
        0.0,
    )
    chain_df["oi_change"] = chain_df["total_oi"].diff()

    # ── ATM-relative features ───────────────────────────────────────────
    if index_df is not None and not index_df.empty:
        logger.info("Computing ATM-relative option features...")

        idx = index_df[["timestamp", "close"]].copy()
        idx["timestamp"] = pd.to_datetime(idx["timestamp"])
        idx["atm_strike"] = (idx["close"] / strike_gap).round() * strike_gap
        idx["atm_strike"] = idx["atm_strike"].astype(int)

        # Merge ATM into option bars
        opt_with_atm = opt_df.merge(
            idx[["timestamp", "atm_strike"]], on="timestamp", how="inner"
        )
        opt_with_atm["relative_strike"] = (
            (opt_with_atm["strike"] - opt_with_atm["atm_strike"]) / strike_gap
        ).astype(int)

        # Near-ATM (|rel| <= 1) and far (|rel| >= 2) masks
        near = opt_with_atm[opt_with_atm["relative_strike"].abs() <= 1]
        far = opt_with_atm[opt_with_atm["relative_strike"].abs() >= 2]

        # PCR near ATM
        near_ce = near[near["option_type"] == "CE"].groupby("timestamp")["oi"].sum()
        near_pe = near[near["option_type"] == "PE"].groupby("timestamp")["oi"].sum()
        pcr_near = (near_pe / near_ce.replace(0, np.nan)).rename("pcr_near_atm")

        # PCR far
        far_ce = far[far["option_type"] == "CE"].groupby("timestamp")["oi"].sum()
        far_pe = far[far["option_type"] == "PE"].groupby("timestamp")["oi"].sum()
        pcr_far = (far_pe / far_ce.replace(0, np.nan)).rename("pcr_far")

        # OI concentration at ATM ±1
        near_total = near.groupby("timestamp")["oi"].sum().rename("near_oi")
        oi_conc = (near_total / chain_df["total_oi"].replace(0, np.nan)).rename("oi_concentration")

        # Max OI call/put relative strike
        ce_with_rel = opt_with_atm[opt_with_atm["option_type"] == "CE"]
        pe_with_rel = opt_with_atm[opt_with_atm["option_type"] == "PE"]

        max_ce_idx = ce_with_rel.groupby("timestamp")["oi"].idxmax().dropna()
        max_oi_call = ce_with_rel.loc[max_ce_idx].set_index("timestamp")["relative_strike"].rename("max_oi_call_rel")

        max_pe_idx = pe_with_rel.groupby("timestamp")["oi"].idxmax().dropna()
        max_oi_put = pe_with_rel.loc[max_pe_idx].set_index("timestamp")["relative_strike"].rename("max_oi_put_rel")

        # Join all ATM-relative features
        for feat in [pcr_near, pcr_far, oi_conc, max_oi_call, max_oi_put]:
            chain_df = chain_df.join(feat, how="left")

    chain_df = chain_df.reset_index()
    chain_df = chain_df.sort_values("timestamp")

    logger.info(f"Built option chain timeseries: {len(chain_df):,} rows, "
                f"columns: {list(chain_df.columns)}")

    return chain_df


def enrich_index_with_options(
    index_df: pd.DataFrame,
    strike_gap: int = 50,
) -> pd.DataFrame:
    """
    Main entry point: enrich an index minute-bar DataFrame with
    time-aligned option chain features.

    Returns the enriched DataFrame with option columns merged.
    """
    chain_df = build_option_chain_timeseries(index_df, strike_gap)

    if chain_df.empty:
        return index_df

    # Merge on timestamp
    index_df = index_df.copy()
    index_df["timestamp"] = pd.to_datetime(index_df["timestamp"])

    opt_cols = [c for c in chain_df.columns if c != "timestamp"]
    merged = index_df.merge(
        chain_df[["timestamp"] + opt_cols],
        on="timestamp",
        how="left",
    )

    logger.info(f"Enriched index data: {len(merged)} rows with "
                f"{len(opt_cols)} option chain columns.")

    return merged
