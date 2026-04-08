#!/usr/bin/env python3
"""Diagnostic: mirrors the exact backend scan_market() logic to show why signals do/don't fire."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
from database.db import read_sql
from features.indicators import compute_all_macro_indicators
from strategy.signal_generator import generate_signals
from strategy.regime_detector import RegimeDetector, MarketRegime, get_strategies_for_regime
from models.predict import Predictor
from models.strategy_models import StrategyPredictor
from backtest.option_resolver import get_nearest_expiry
from config.risk_profiles import get_risk_profile, RiskLevel
from datetime import date

# Must match backend/app.py constants
SCORE_THRESHOLD = 0.60
WEIGHT_ML_PROBABILITY      = 0.50
WEIGHT_OPTIONS_FLOW        = 0.30
WEIGHT_TECHNICAL_STRENGTH  = 0.20

# 1. Load data
df = read_sql(
    "SELECT timestamp, symbol, open, high, low, close, volume, vwap, oi "
    "FROM minute_candles WHERE symbol='NIFTY-I' ORDER BY timestamp DESC LIMIT 300"
)
if df.empty or len(df) < 50:
    print("ERROR: Not enough candle data")
    sys.exit(1)

df = df.sort_values('timestamp').reset_index(drop=True)
df['timestamp'] = pd.to_datetime(df['timestamp'])
featured = compute_all_macro_indicators(df)
latest = featured.iloc[-1].to_dict()

# 2. Market state
print("=" * 55)
print("MARKET STATE")
print("=" * 55)
print(f"Latest bar  : {latest.get('timestamp')}")
print(f"Close       : {latest.get('close'):.2f}")
print(f"RSI         : {latest.get('rsi', 0):.1f}")
print(f"VWAP        : {latest.get('vwap', 0):.2f}  dist={latest.get('vwap_dist', 0):.4f}")
print(f"MACD hist   : {latest.get('macd_hist', 0):.3f}")
print(f"ADX         : {latest.get('adx', 0):.1f}")
print(f"ATR%        : {latest.get('atr_pct', 0):.4f}")
print(f"OBV slope   : {latest.get('obv_slope', 0):.4f}")
print(f"MFI         : {latest.get('mfi', 50):.1f}")
print(f"PCR         : {latest.get('pcr', None)}")
print(f"mins open   : {latest.get('minutes_since_open', 0):.0f}")

# 3. Regime
regime_detector = RegimeDetector()
regime_window = df.tail(100)[["open", "high", "low", "close", "volume"]].copy()
regime = regime_detector.detect(regime_window)
regime_strategies = get_strategies_for_regime(regime)
print(f"\nRegime      : {regime.value}")
print(f"Regime strats: {regime_strategies}")

# 4. Raw signals
print("\n" + "=" * 55)
print("SIGNALS")
print("=" * 55)
signals = generate_signals(latest, 'NIFTY-I')
print(f"generate_signals() returned: {len(signals)} signals")
for s in signals:
    print(f"  {s.strategy} -> {s.direction}  strength={s.technical_strength:.2f}")

if not signals:
    row = latest
    print("\nNo signals — why each strategy didn't fire:")
    close = row.get('close', 0)
    vwap  = row.get('vwap', 0)
    macd  = row.get('macd_hist', 0)
    rsi   = row.get('rsi', 50)
    adx   = row.get('adx', 0)
    print(f"\n  vwap_momentum_breakout (needs CALL: price>VWAP, MACD>0, RSI 50-70, ADX>20):")
    print(f"    price>VWAP : {close:.1f} > {vwap:.1f} = {close > vwap}")
    print(f"    MACD>0     : {macd:.4f} = {macd > 0}")
    print(f"    RSI 50-70  : {rsi:.1f} = {50 <= rsi <= 70}")
    print(f"    ADX>20     : {adx:.1f} = {adx > 20}")
    print(f"\n  bearish_momentum (needs PUT: price<VWAP, MACD<0, RSI 30-50, ADX>20):")
    print(f"    price<VWAP : {close:.1f} < {vwap:.1f} = {close < vwap}")
    print(f"    MACD<0     : {macd:.4f} = {macd < 0}")
    print(f"    RSI 30-50  : {rsi:.1f} = {30 <= rsi <= 50}")
    print(f"    ADX>20     : {adx:.1f} = {adx > 20}")
    print(f"\n  mean_reversion (needs RSI<30 or RSI>70):")
    print(f"    RSI        : {rsi:.1f}  oversold={rsi<30}  overbought={rsi>70}")
    sys.exit(0)

# 5. Score each signal exactly as backend does
print("\n" + "=" * 55)
print("SCORING (mirroring scan_market logic)")
print("=" * 55)
predictor = Predictor()
predictor.load()
strategy_predictor = StrategyPredictor()
strategy_predictor.load()
expiry = get_nearest_expiry(date.today())
_med_profile = get_risk_profile(RiskLevel.MEDIUM)
print(f"Expiry: {expiry}")

for sig in signals:
    print(f"\n--- {sig.strategy} {sig.direction} (strength={sig.technical_strength:.2f}) ---")

    # ML probability
    ml_prob = 0.5
    if predictor.is_loaded:
        p = predictor.predict_macro(latest)
        if p is not None:
            ml_prob = p
    print(f"  ml_prob (raw)      : {ml_prob:.4f}")

    # Strategy model gate
    strat_prob = strategy_predictor.predict(sig.strategy, latest) or 0.5
    print(f"  strat_prob         : {strat_prob:.4f}  (gate: >=0.02 = {strat_prob >= 0.02})")
    if strat_prob < 0.02:
        print(f"  ✗ BLOCKED by strategy model gate (strat_prob {strat_prob:.4f} < 0.02)")
        continue

    # Directional probability
    directional_prob = ml_prob if sig.direction == 'CALL' else (1.0 - ml_prob)
    print(f"  directional_prob   : {directional_prob:.4f}")

    # Flow score
    pcr = latest.get('pcr')
    if pcr and not np.isnan(float(pcr if pcr else 0)):
        flow_score = min(0.3 * (pcr > 1.2) + 0.2, 1.0)
        print(f"  flow_score (PCR)   : {flow_score:.4f}  PCR={pcr:.3f}")
    else:
        obv_slope = latest.get('obv_slope', 0) or 0
        mfi = latest.get('mfi', 50) or 50
        if sig.direction == 'CALL':
            obv_contrib = 0.15 if obv_slope > 0 else (-0.10 if obv_slope < 0 else 0.0)
            mfi_contrib = 0.15 if mfi > 60 else (-0.10 if mfi < 40 else 0.0)
        else:
            obv_contrib = 0.15 if obv_slope < 0 else (-0.10 if obv_slope > 0 else 0.0)
            mfi_contrib = 0.15 if mfi < 40 else (-0.10 if mfi > 60 else 0.0)
        flow_score = max(0.20, min(1.0, 0.50 + obv_contrib + mfi_contrib))
        print(f"  flow_score (OBV/MFI): {flow_score:.4f}  obv={obv_slope:.4f} mfi={mfi:.1f}")

    # Regime bonus
    regime_bonus = 0.05 if regime_strategies and sig.strategy in regime_strategies else 0.0
    print(f"  regime_bonus       : {regime_bonus}")

    # Final score
    final_score = (
        WEIGHT_ML_PROBABILITY * directional_prob
        + WEIGHT_OPTIONS_FLOW * flow_score
        + WEIGHT_TECHNICAL_STRENGTH * sig.technical_strength
        + regime_bonus
    )
    print(f"  final_score        : {final_score:.4f}  = 0.5×{directional_prob:.3f} + 0.3×{flow_score:.3f} + 0.2×{sig.technical_strength:.3f} + {regime_bonus}")

    # Effective threshold
    if regime in (MarketRegime.SIDEWAYS, MarketRegime.LOW_VOLATILITY):
        effective_threshold = SCORE_THRESHOLD * 0.85
    elif regime in (MarketRegime.HIGH_VOLATILITY,):
        effective_threshold = SCORE_THRESHOLD * 0.90
    else:
        effective_threshold = SCORE_THRESHOLD
    effective_threshold = max(effective_threshold, _med_profile.put_score_threshold)
    print(f"  base threshold     : {effective_threshold:.4f}  (regime={regime.value})")

    # Strategy-specific gates
    if sig.strategy == 'mean_reversion':
        if regime not in (MarketRegime.SIDEWAYS, MarketRegime.LOW_VOLATILITY):
            print(f"  ✗ BLOCKED: mean_reversion only fires in SIDEWAYS/LOW_VOL (regime={regime.value})")
            continue
        if directional_prob < 0.40:
            print(f"  ✗ BLOCKED: mean_reversion directional_prob {directional_prob:.3f} < 0.40")
            continue
        effective_threshold = max(effective_threshold, 0.80)
        print(f"  threshold (mean_rev): {effective_threshold:.4f}")
    elif sig.strategy == 'vwap_momentum_breakout':
        if regime not in (MarketRegime.TRENDING_BULL, MarketRegime.LOW_VOLATILITY):
            print(f"  ✗ BLOCKED: vwap_momentum_breakout only fires in TRENDING_BULL/LOW_VOL (regime={regime.value})")
            continue
        effective_threshold = max(effective_threshold, 0.65)
        print(f"  threshold (vwap)   : {effective_threshold:.4f}")

    # Final gate
    if final_score >= effective_threshold:
        print(f"  ✓ PASSES threshold! Would generate trade suggestion.")
    else:
        gap = effective_threshold - final_score
        print(f"  ✗ BELOW threshold by {gap:.4f}  ({final_score:.4f} < {effective_threshold:.4f})")
