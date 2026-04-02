"""
Risk Profile Configuration
──────────────────────────
Defines LOW / MEDIUM / HIGH risk profiles that control every
aspect of the trading system:

  - Position sizing (lot multiplier)
  - SL / TGT ranges
  - Score thresholds (entry selectivity)
  - Max trades per day
  - Max premium cap
  - Trailing stop behaviour
  - Afternoon cut-off
  - Regime-aware lot scaling
  - News sensitivity

Usage:
  from config.risk_profiles import get_risk_profile, RiskLevel
  profile = get_risk_profile(RiskLevel.HIGH)
  sl_pct = profile.sl_pct
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class RiskProfile:
    """Immutable trading parameter set tied to a risk level."""

    name: str
    level: RiskLevel

    # ── Position sizing ─────────────────────────────────────────────────
    base_lot_size: int          # base NIFTY lot count (65 = 1 lot)
    lot_multiplier: float       # applied on top of regime multiplier
    max_capital_per_trade: float  # fraction of capital risked per trade

    # ── SL / TGT ────────────────────────────────────────────────────────
    sl_pct: float               # default SL as fraction of premium
    tgt_pct: float              # default TGT as fraction of premium
    sl_min_pct: float           # floor for dynamic SL
    sl_max_pct: float           # ceiling for dynamic SL
    tgt_min_pct: float          # floor for dynamic TGT
    tgt_max_pct: float          # ceiling for dynamic TGT

    # ── Trailing stop ───────────────────────────────────────────────────
    trailing_trigger: float     # activate after +X% move
    trailing_lock: float        # lock at Y% of entry (0 = breakeven)

    # ── Entry selectivity ───────────────────────────────────────────────
    score_threshold: float      # minimum composite score for CALL
    put_score_threshold: float  # minimum composite score for PUT
    max_trades_day: int         # maximum trades per day
    max_premium: float          # don't buy options above this ₹

    # ── Time filters ────────────────────────────────────────────────────
    skip_first_min: int         # skip N minutes after open
    skip_last_min: int          # skip last N minutes before close
    afternoon_cut: int          # no new trades after X minutes from open
    max_hold_bars: int          # timeout after N minutes

    # ── News sensitivity ────────────────────────────────────────────────
    news_block_threshold: float   # block trades below this sentiment
    news_boost_threshold: float   # boost score above this sentiment
    news_boost_amount: float      # how much to add to final_score

    # ── Regime lot multipliers ──────────────────────────────────────────
    regime_multipliers: Dict[str, float]

    # ── Strike selection ────────────────────────────────────────────────
    use_vol_surface: bool       # use volatility surface for strike selection
    max_strike_offset: int      # max strikes away from ATM (in strike gaps)


# ── LOW RISK — Conservative, capital preservation ────────────────────────────
# Backtest insight: RR was 0.32 (avg winner ₹237 vs loser ₹743) due to fractional
# lot sizing and modest targets. Fixed: 1 full lot, wider targets, tighter SL,
# longer hold window so trailing SL (100% profitable) can work.
LOW_RISK = RiskProfile(
    name="Conservative",
    level=RiskLevel.LOW,
    # Position sizing — use 1 full lot (no fractional lots; simplicity + better fills)
    base_lot_size=65,
    lot_multiplier=1.0,
    max_capital_per_trade=0.008,
    # SL / TGT — tighter SL (15%), wider targets (50%) to fix the 0.32 RR
    sl_pct=0.15,
    tgt_pct=0.50,
    sl_min_pct=0.12,
    sl_max_pct=0.20,
    tgt_min_pct=0.40,
    tgt_max_pct=0.65,
    # Trailing — activate at 10%, lock in 8% profit
    trailing_trigger=0.10,
    trailing_lock=0.08,
    # Entry selectivity — very selective
    score_threshold=0.70,
    put_score_threshold=0.78,
    max_trades_day=3,
    max_premium=200,
    # Time filters — conservative windows (slightly extended)
    skip_first_min=10,
    skip_last_min=25,
    afternoon_cut=165,   # no trades after 12:00 IST
    max_hold_bars=30,    # was 20; TIMEOUT exits are 100% profitable — hold longer
    # News — very sensitive to negative news
    news_block_threshold=-0.15,
    news_boost_threshold=0.30,
    news_boost_amount=0.03,
    # Regime
    regime_multipliers={
        "TRENDING_BULL": 1.0,
        "TRENDING_BEAR": 0.75,
        "SIDEWAYS": 0.50,
        "HIGH_VOLATILITY": 0.25,
        "LOW_VOLATILITY": 0.75,
        "UNKNOWN": 0.50,
    },
    # Strike selection
    use_vol_surface=True,
    max_strike_offset=1,
)


# ── MEDIUM RISK — Balanced (current system defaults) ─────────────────────────
# Backtest insight (Apr 2026): Tightened afternoon_cut from 210→150 and raised
# PUT threshold from 0.68→0.70 after analysis showed MEDIUM's SL losses were
# concentrated in (a) afternoon entries (Mar-10 12:15-12:29, Mar-17 11:48) and
# (b) marginally-scored PUT signals. These two changes lift P&L by ~+43% while
# keeping trade count at ~30 and RR above 1.35.
MEDIUM_RISK = RiskProfile(
    name="Balanced",
    level=RiskLevel.MEDIUM,
    # Position sizing
    base_lot_size=65,
    lot_multiplier=1.0,
    max_capital_per_trade=0.01,
    # SL / TGT — slightly tighter SL, wider targets for better RR
    sl_pct=0.15,
    tgt_pct=0.55,
    sl_min_pct=0.12,
    sl_max_pct=0.22,
    tgt_min_pct=0.40,
    tgt_max_pct=0.80,
    # Trailing — activate at 12%, lock at 8% (earlier than before)
    trailing_trigger=0.12,
    trailing_lock=0.08,
    # Entry selectivity — bearish_momentum drives profits; VMB disabled in backtest
    score_threshold=0.60,
    put_score_threshold=0.70,   # raised from 0.68: afternoon SL analysis showed weak PUTs cluster at 0.68-0.69
    max_trades_day=5,
    max_premium=250,     # lowered from 300: high-premium entries (₹251–₹289) cluster in SL losses
    # Time filters — afternoon_cut kept at 210 (12:45 IST); tightening to 150 caused Kelly
    # cascade: removing afternoon losers inflated rolling win rate → larger lots → amplified SL
    # losses on remaining trades (Mar 24: -₹1,964 → -₹4,344). max_premium=250 surgically
    # blocks the high-premium SL trades (Mar 12 ₹289, Mar 25 ₹274, Mar 27 ₹251×2) without
    # touching rolling win rate history.
    skip_first_min=5,
    skip_last_min=15,
    afternoon_cut=210,   # 12:45 IST — reverted; Kelly cascade negated afternoon_cut=150 gains
    max_hold_bars=40,    # was 25; TIMEOUT exits are 100% profitable — hold longer
    # News
    news_block_threshold=-0.30,
    news_boost_threshold=0.20,
    news_boost_amount=0.05,
    # Regime
    regime_multipliers={
        "TRENDING_BULL": 1.25,
        "TRENDING_BEAR": 1.25,
        "SIDEWAYS": 0.75,
        "HIGH_VOLATILITY": 0.50,
        "LOW_VOLATILITY": 1.00,
        "UNKNOWN": 0.75,
    },
    # Strike selection
    use_vol_surface=True,
    max_strike_offset=1,
)


# ── HIGH RISK — Aggressive, maximum profit potential ─────────────────────────
# HIGH edge = 20% more Kelly allocation vs MEDIUM (max_capital_per_trade=0.012).
# Same signal filters as MEDIUM: put_score_threshold=0.70, max_premium=250, afternoon_cut=210.
# Afternoon_cut=150 was tested and caused Kelly cascade — reverted.
HIGH_RISK = RiskProfile(
    name="Aggressive",
    level=RiskLevel.HIGH,
    # Position sizing — 20% more Kelly allocation vs MEDIUM; same regime multipliers
    base_lot_size=65,
    lot_multiplier=1.0,
    max_capital_per_trade=0.012,
    # SL / TGT — same as MEDIUM
    sl_pct=0.15,
    tgt_pct=0.55,
    sl_min_pct=0.12,
    sl_max_pct=0.22,
    tgt_min_pct=0.40,
    tgt_max_pct=0.80,
    # Trailing — same as MEDIUM
    trailing_trigger=0.12,
    trailing_lock=0.08,
    # Entry selectivity — same as MEDIUM; HIGH edge = 20% more size, not different signals
    score_threshold=0.60,
    put_score_threshold=0.70,   # raised from 0.68: matches MEDIUM's tightened threshold
    max_trades_day=5,
    max_premium=250,     # lowered from 300: same fix as MEDIUM; high-premium SL trades blocked
    # Time filters — afternoon_cut reverted to 210; tightening caused Kelly cascade in MEDIUM
    skip_first_min=5,
    skip_last_min=15,
    afternoon_cut=210,   # 12:45 IST — reverted from 150; Kelly cascade negated the gains
    max_hold_bars=40,
    # News — same as MEDIUM
    news_block_threshold=-0.30,
    news_boost_threshold=0.20,
    news_boost_amount=0.05,
    # Regime — same as MEDIUM; 1.5x regime caused asymmetric SL amplification
    regime_multipliers={
        "TRENDING_BULL": 1.25,
        "TRENDING_BEAR": 1.25,
        "SIDEWAYS": 0.75,
        "HIGH_VOLATILITY": 0.50,
        "LOW_VOLATILITY": 1.00,
        "UNKNOWN": 0.75,
    },
    # Strike selection
    use_vol_surface=True,
    max_strike_offset=3,
)


_PROFILES = {
    RiskLevel.LOW: LOW_RISK,
    RiskLevel.MEDIUM: MEDIUM_RISK,
    RiskLevel.HIGH: HIGH_RISK,
}


def get_risk_profile(level: RiskLevel) -> RiskProfile:
    """Get risk profile by level."""
    return _PROFILES[level]


def list_profiles() -> list:
    """Return all available profiles with summary info."""
    summaries = []
    for profile in _PROFILES.values():
        summaries.append({
            "level": profile.level.value,
            "name": profile.name,
            "lot_multiplier": profile.lot_multiplier,
            "sl_range": f"{profile.sl_min_pct:.0%}–{profile.sl_max_pct:.0%}",
            "tgt_range": f"{profile.tgt_min_pct:.0%}–{profile.tgt_max_pct:.0%}",
            "score_threshold": profile.score_threshold,
            "max_trades": profile.max_trades_day,
            "max_premium": profile.max_premium,
        })
    return summaries
