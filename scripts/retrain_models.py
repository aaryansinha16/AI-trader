"""
Retrain All ML Models on Recent Data
─────────────────────────────────────
Retrains macro model, micro model, and strategy-specific models using
all available candle data from the DB (Sep 2025 – present).

Usage:
  python scripts/retrain_models.py              # full retrain
  python scripts/retrain_models.py --recent 60  # only last 60 days
"""

import os, sys, argparse
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from database.db import read_sql
from features.indicators import compute_all_macro_indicators
from models.train_model import MacroModelTrainer, train_all_models
from models.strategy_models import train_all_strategy_models, StrategyPredictor
from utils.logger import get_logger

logger = get_logger("retrain")


def main():
    parser = argparse.ArgumentParser(description="Retrain ML models")
    parser.add_argument("--recent", type=int, default=0,
                        help="Only use last N days of data (0 = all)")
    args = parser.parse_args()

    print("=" * 60)
    print("  ML MODEL RETRAINING")
    print("=" * 60)

    # ── Load candle data ─────────────────────────────────────────────────
    if args.recent > 0:
        cutoff = (datetime.now() - timedelta(days=args.recent)).strftime("%Y-%m-%d")
        where = f"AND timestamp >= '{cutoff}'"
        print(f"  Using data from last {args.recent} days (since {cutoff})")
    else:
        where = ""
        print("  Using ALL available candle data")

    candles = read_sql(
        f"SELECT timestamp, symbol, open, high, low, close, volume, vwap, oi "
        f"FROM minute_candles WHERE symbol = 'NIFTY-I' {where} "
        f"ORDER BY timestamp"
    )
    if candles.empty:
        print("  ERROR: No candle data found.")
        return

    candles["timestamp"] = pd.to_datetime(candles["timestamp"])
    print(f"  Candles loaded: {len(candles):,}")
    print(f"  Date range: {candles['timestamp'].min().date()} → {candles['timestamp'].max().date()}")
    days = candles["timestamp"].dt.date.nunique()
    print(f"  Trading days: {days}")

    # ── Compute features ─────────────────────────────────────────────────
    print("\n  Computing features...")
    featured = compute_all_macro_indicators(candles)
    print(f"  Featured rows: {len(featured):,}")
    print(f"  Feature columns: {len(featured.columns)}")

    # ── Train macro model ────────────────────────────────────────────────
    print("\n  --- MACRO MODEL ---")
    macro_trainer = MacroModelTrainer()
    macro_df = macro_trainer.prepare_data(featured.copy())
    if len(macro_df) > 100:
        metrics = macro_trainer.train(macro_df, walk_forward=True, n_splits=5)
        macro_trainer.save()
        print(f"  Macro model trained: {len(macro_df):,} samples")
        print(f"  Metrics: {metrics}")

        fi = macro_trainer.get_feature_importance()
        print(f"  Top 10 features:")
        for _, row in fi.head(10).iterrows():
            print(f"    {row['feature']:30s}  {row['importance']:.4f}")
    else:
        print(f"  Not enough data ({len(macro_df)} samples). Need 100+.")

    # ── Train strategy-specific models ───────────────────────────────────
    print("\n  --- STRATEGY MODELS ---")
    try:
        strat_results = train_all_strategy_models(featured.copy())
        for name, info in strat_results.items():
            if info:
                print(f"  {name}: {info.get('metrics', 'trained')}")
            else:
                print(f"  {name}: skipped (insufficient data)")
    except Exception as e:
        print(f"  Strategy model training error: {e}")

    # ── Verify loaded models ─────────────────────────────────────────────
    print("\n  --- VERIFICATION ---")
    from models.predict import Predictor
    p = Predictor()
    p.load()
    sp = StrategyPredictor()
    sp.load()
    print(f"  Macro model loaded: {p.is_loaded}")
    print(f"  Strategy models: {sp.available_strategies}")

    # Quick prediction test
    if p.is_loaded and not featured.empty:
        test_row = featured.iloc[-1].to_dict()
        prob = p.predict_macro(test_row)
        print(f"  Test prediction on last row: {prob:.4f}" if prob else "  Test prediction: None")

    print("\n" + "=" * 60)
    print("  RETRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
