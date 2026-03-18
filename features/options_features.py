"""
Options-Aware Features
──────────────────────
Features that capture the structure of the options chain relative to ATM.
These are critical for the ML model to understand moneyness, expiry proximity,
and cross-strike relationships.

New features:
  1. relative_strike     – 0=ATM, +1=ATM+1, -1=ATM-1, etc.
  2. days_to_expiry      – trading days remaining until expiry
  3. oi_skew             – (total call OI - total put OI) / total OI
  4. pcr_near_atm        – PCR computed only from ATM ±1
  5. pcr_far             – PCR computed from ATM ±2 to ±3
  6. max_oi_call_rel     – relative strike with highest call OI
  7. max_oi_put_rel      – relative strike with highest put OI
  8. oi_concentration    – % of total OI at ATM ±1 (high = pinning risk)
  9. call_oi_gradient    – OI slope from ATM to ATM+3 (rising = resistance)
  10. put_oi_gradient    – OI slope from ATM to ATM-3 (rising = support)
  11. iv_skew            – difference in IV between OTM puts and OTM calls
  12. theta_pressure     – exponential decay factor as expiry approaches
"""

from datetime import date, datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("options_features")


def compute_days_to_expiry(
    timestamp: datetime,
    expiry: date,
) -> int:
    """
    Compute trading days to expiry (approximate).
    Excludes weekends. Does not account for NSE holidays.
    """
    if expiry is None:
        return -1

    ref_date = timestamp.date() if isinstance(timestamp, datetime) else timestamp
    if ref_date >= expiry:
        return 0

    days = 0
    current = ref_date
    from datetime import timedelta
    while current < expiry:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Mon-Fri
            days += 1
    return days


def compute_theta_pressure(days_to_expiry: int) -> float:
    """
    Exponential theta pressure factor.
    Theta decay accelerates as expiry approaches.
    Returns 0.0 (far from expiry) to 1.0 (expiry day).
    """
    if days_to_expiry <= 0:
        return 1.0
    # Exponential decay: pressure = exp(-days/5)
    # At 0 days → 1.0, at 5 days → 0.37, at 10 days → 0.14
    return float(np.exp(-days_to_expiry / 5.0))


def compute_cross_strike_features(
    option_chain_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute aggregate cross-strike features from the full option chain.

    Expects columns: strike, option_type (CE/PE), oi, volume, iv, relative_strike
    At minimum needs: option_type, oi, relative_strike

    Returns dict of feature name → value.
    """
    features = {}

    if option_chain_df is None or option_chain_df.empty:
        return _empty_cross_strike_features()

    oc = option_chain_df.copy()

    # Ensure required columns
    if "option_type" not in oc.columns or "oi" not in oc.columns:
        return _empty_cross_strike_features()

    ce = oc[oc["option_type"] == "CE"]
    pe = oc[oc["option_type"] == "PE"]

    total_ce_oi = ce["oi"].sum()
    total_pe_oi = pe["oi"].sum()
    total_oi = total_ce_oi + total_pe_oi

    # ── OI Skew: (call OI - put OI) / total OI ──────────────────────────
    # Positive = more call OI (bearish), Negative = more put OI (bullish)
    features["oi_skew"] = (
        (total_ce_oi - total_pe_oi) / total_oi if total_oi > 0 else 0.0
    )

    # ── PCR near ATM (±1 strikes only) ──────────────────────────────────
    has_rel = "relative_strike" in oc.columns
    if has_rel:
        near_mask = oc["relative_strike"].abs() <= 1
        near_ce_oi = oc.loc[near_mask & (oc["option_type"] == "CE"), "oi"].sum()
        near_pe_oi = oc.loc[near_mask & (oc["option_type"] == "PE"), "oi"].sum()
        features["pcr_near_atm"] = near_pe_oi / near_ce_oi if near_ce_oi > 0 else np.nan

        # ── PCR far (±2 to ±3 strikes) ──────────────────────────────────
        far_mask = oc["relative_strike"].abs() >= 2
        far_ce_oi = oc.loc[far_mask & (oc["option_type"] == "CE"), "oi"].sum()
        far_pe_oi = oc.loc[far_mask & (oc["option_type"] == "PE"), "oi"].sum()
        features["pcr_far"] = far_pe_oi / far_ce_oi if far_ce_oi > 0 else np.nan

        # ── Max OI call/put relative strike ──────────────────────────────
        if not ce.empty and has_rel:
            features["max_oi_call_rel"] = float(
                ce.loc[ce["oi"].idxmax(), "relative_strike"]
            )
        else:
            features["max_oi_call_rel"] = np.nan

        if not pe.empty and has_rel:
            features["max_oi_put_rel"] = float(
                pe.loc[pe["oi"].idxmax(), "relative_strike"]
            )
        else:
            features["max_oi_put_rel"] = np.nan

        # ── OI concentration at ATM ±1 ──────────────────────────────────
        # High concentration = gamma pinning risk
        atm_oi = oc.loc[near_mask, "oi"].sum()
        features["oi_concentration"] = atm_oi / total_oi if total_oi > 0 else 0.0

        # ── Call OI gradient (ATM → ATM+3) ──────────────────────────────
        # Rising OI at higher strikes = resistance building
        call_oi_by_strike = (
            ce[ce["relative_strike"] >= 0]
            .groupby("relative_strike")["oi"]
            .sum()
            .sort_index()
        )
        if len(call_oi_by_strike) >= 2:
            features["call_oi_gradient"] = float(np.polyfit(
                call_oi_by_strike.index.astype(float),
                call_oi_by_strike.values.astype(float),
                1,
            )[0])
        else:
            features["call_oi_gradient"] = 0.0

        # ── Put OI gradient (ATM → ATM-3) ───────────────────────────────
        # Rising OI at lower strikes = support building
        put_oi_by_strike = (
            pe[pe["relative_strike"] <= 0]
            .groupby("relative_strike")["oi"]
            .sum()
            .sort_index(ascending=False)
        )
        if len(put_oi_by_strike) >= 2:
            features["put_oi_gradient"] = float(np.polyfit(
                put_oi_by_strike.index.astype(float),
                put_oi_by_strike.values.astype(float),
                1,
            )[0])
        else:
            features["put_oi_gradient"] = 0.0
    else:
        features["pcr_near_atm"] = np.nan
        features["pcr_far"] = np.nan
        features["max_oi_call_rel"] = np.nan
        features["max_oi_put_rel"] = np.nan
        features["oi_concentration"] = 0.0
        features["call_oi_gradient"] = 0.0
        features["put_oi_gradient"] = 0.0

    # ── IV Skew ──────────────────────────────────────────────────────────
    # OTM puts vs OTM calls IV difference (fear gauge)
    if "iv" in oc.columns and has_rel:
        otm_put_iv = pe.loc[pe["relative_strike"] < 0, "iv"].mean()
        otm_call_iv = ce.loc[ce["relative_strike"] > 0, "iv"].mean()
        features["iv_skew"] = (
            (otm_put_iv - otm_call_iv) if pd.notna(otm_put_iv) and pd.notna(otm_call_iv) else np.nan
        )
    else:
        features["iv_skew"] = np.nan

    return features


def _empty_cross_strike_features() -> Dict[str, float]:
    """Return dict of all cross-strike features with NaN values."""
    return {
        "oi_skew": np.nan,
        "pcr_near_atm": np.nan,
        "pcr_far": np.nan,
        "max_oi_call_rel": np.nan,
        "max_oi_put_rel": np.nan,
        "oi_concentration": np.nan,
        "call_oi_gradient": np.nan,
        "put_oi_gradient": np.nan,
        "iv_skew": np.nan,
    }


# ── Feature column names for settings.py ────────────────────────────────────

OPTIONS_FEATURE_COLUMNS = [
    "relative_strike",
    "days_to_expiry",
    "theta_pressure",
    "oi_skew",
    "pcr_near_atm",
    "pcr_far",
    "max_oi_call_rel",
    "max_oi_put_rel",
    "oi_concentration",
    "call_oi_gradient",
    "put_oi_gradient",
    "iv_skew",
]
