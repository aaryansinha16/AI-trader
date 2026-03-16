"""
Options Flow Detector
─────────────────────
Detects institutional positioning from option chain data.

From the Product Vision doc (§10):

  Key signals:
    Long Build Up    – price ↑, OI ↑  → Bullish
    Short Covering   – price ↑, OI ↓  → Strong rallies
    Long Unwinding   – price ↓, OI ↓  → Weak market
    Short Build Up   – price ↓, OI ↑  → Bearish
    Gamma Pinning    – large OI near a strike → price gravitates

  Inputs: option chain data, OI change, volume spikes, strike concentration
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("options_flow")


class FlowSignal(str, Enum):
    LONG_BUILD_UP = "LONG_BUILD_UP"
    SHORT_COVERING = "SHORT_COVERING"
    LONG_UNWINDING = "LONG_UNWINDING"
    SHORT_BUILD_UP = "SHORT_BUILD_UP"
    GAMMA_PINNING = "GAMMA_PINNING"
    NEUTRAL = "NEUTRAL"


@dataclass
class FlowResult:
    """Result from the options flow analysis."""
    signal: FlowSignal
    score: float           # 0.0 – 1.0
    pcr: float             # Put-Call Ratio
    max_oi_strike: float   # Strike with largest OI
    net_oi_change: float
    details: Dict


class OptionsFlowDetector:
    """Analyses option chain snapshots to detect institutional activity."""

    def analyze(
        self,
        option_chain_df: pd.DataFrame,
        spot_price: float,
        prev_spot_price: float = None,
    ) -> FlowResult:
        """
        Analyze an option chain snapshot.

        Args:
            option_chain_df: DataFrame with columns:
                strike, option_type (CE/PE), oi, oi_change, volume, ltp, iv
            spot_price: current underlying price
            prev_spot_price: previous candle's underlying price (for trend)

        Returns FlowResult with signal, score, and details.
        """
        if option_chain_df.empty:
            return FlowResult(
                signal=FlowSignal.NEUTRAL,
                score=0.0,
                pcr=0.0,
                max_oi_strike=0.0,
                net_oi_change=0.0,
                details={},
            )

        oc = option_chain_df.copy()

        # ── Put-Call Ratio ────────────────────────────────────────────────────
        ce_oi = oc[oc["option_type"] == "CE"]["oi"].sum()
        pe_oi = oc[oc["option_type"] == "PE"]["oi"].sum()
        pcr = pe_oi / ce_oi if ce_oi > 0 else 1.0

        # ── Max OI Strike (Gamma Pinning) ────────────────────────────────────
        oi_by_strike = oc.groupby("strike")["oi"].sum()
        max_oi_strike = oi_by_strike.idxmax() if not oi_by_strike.empty else spot_price

        # ── Net OI Change ────────────────────────────────────────────────────
        net_oi_change = oc["oi_change"].sum() if "oi_change" in oc.columns else 0

        ce_oi_change = oc[oc["option_type"] == "CE"]["oi_change"].sum()
        pe_oi_change = oc[oc["option_type"] == "PE"]["oi_change"].sum()

        # ── Volume Spikes ────────────────────────────────────────────────────
        vol_mean = oc["volume"].mean() if "volume" in oc.columns else 0
        high_vol_contracts = oc[oc["volume"] > vol_mean * 2] if vol_mean > 0 else pd.DataFrame()
        has_volume_spike = len(high_vol_contracts) > 0

        # ── Price Trend ──────────────────────────────────────────────────────
        if prev_spot_price is not None and prev_spot_price > 0:
            price_change = (spot_price - prev_spot_price) / prev_spot_price
            price_up = price_change > 0.001
            price_down = price_change < -0.001
        else:
            price_up = False
            price_down = False
            price_change = 0

        # ── Signal Classification ────────────────────────────────────────────
        oi_increasing = net_oi_change > 0
        oi_decreasing = net_oi_change < 0

        # Check gamma pinning: is max OI strike near spot?
        gamma_pin_distance = abs(max_oi_strike - spot_price) / spot_price
        is_gamma_pinned = gamma_pin_distance < 0.005  # within 0.5%

        if price_up and oi_increasing:
            flow_signal = FlowSignal.LONG_BUILD_UP
        elif price_up and oi_decreasing:
            flow_signal = FlowSignal.SHORT_COVERING
        elif price_down and oi_decreasing:
            flow_signal = FlowSignal.LONG_UNWINDING
        elif price_down and oi_increasing:
            flow_signal = FlowSignal.SHORT_BUILD_UP
        elif is_gamma_pinned:
            flow_signal = FlowSignal.GAMMA_PINNING
        else:
            flow_signal = FlowSignal.NEUTRAL

        # ── Score Calculation ────────────────────────────────────────────────
        score = 0.0

        # PCR extremes (bullish if PCR > 1.2, bearish if < 0.7)
        if pcr > 1.2:
            score += 0.25   # Contrarian bullish (lots of puts = support)
        elif pcr < 0.7:
            score += 0.15

        # OI change strength
        if abs(net_oi_change) > 0:
            score += 0.25

        # Volume spikes in options
        if has_volume_spike:
            score += 0.25

        # Directional alignment
        if flow_signal in (FlowSignal.LONG_BUILD_UP, FlowSignal.SHORT_COVERING):
            score += 0.25  # Bullish alignment
        elif flow_signal in (FlowSignal.SHORT_BUILD_UP, FlowSignal.LONG_UNWINDING):
            score += 0.20

        score = min(score, 1.0)

        details = {
            "ce_oi": int(ce_oi),
            "pe_oi": int(pe_oi),
            "ce_oi_change": int(ce_oi_change),
            "pe_oi_change": int(pe_oi_change),
            "volume_spike": has_volume_spike,
            "gamma_pin_distance": round(gamma_pin_distance, 4),
            "price_change": round(price_change, 4) if price_change else 0,
        }

        logger.info(
            f"Flow: {flow_signal.value} (score={score:.2f}, "
            f"PCR={pcr:.2f}, max_OI_strike={max_oi_strike})"
        )

        return FlowResult(
            signal=flow_signal,
            score=round(score, 2),
            pcr=round(pcr, 2),
            max_oi_strike=max_oi_strike,
            net_oi_change=net_oi_change,
            details=details,
        )


def options_flow_score(option_chain: dict) -> float:
    """Legacy compatibility wrapper."""
    score = 0.0
    if option_chain.get("oi_change", 0) > 10:
        score += 0.4
    if option_chain.get("volume_spike"):
        score += 0.3
    if option_chain.get("price_momentum"):
        score += 0.3
    return score