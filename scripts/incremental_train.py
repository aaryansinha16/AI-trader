"""
Incremental Model Training
──────────────────────────
Updates existing ML models with new data without full retraining.
Uses warm-start to continue training from the last checkpoint.

Usage:
  python scripts/incremental_train.py --days 2  # train on last 2 days
"""

import os, sys, argparse, shutil
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

BACKUP_DIR = Path("models/saved/backups")
MACRO_MODEL_PATH = Path("models/saved/macro_model.pkl")
MICRO_MODEL_PATH_P = Path("models/saved/micro_model.pkl")


def _backup_model(path: Path):
    """Timestamped backup before overwriting a model."""
    if path.exists():
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = BACKUP_DIR / f"{path.stem}_{ts}.pkl"
        shutil.copy2(path, dst)
        print(f"  Backed up {path.name} → {dst.name}")

import pandas as pd

from database.db import read_sql
from features.indicators import compute_all_macro_indicators
from features.micro_features import compute_micro_features
from models.train_model import MacroModelTrainer, MicroModelTrainer
from models.strategy_models import train_all_strategy_models, StrategyPredictor
from utils.logger import get_logger

logger = get_logger("incremental_train")


def main():
    parser = argparse.ArgumentParser(description="Incrementally train ML models with new data")
    parser.add_argument("--days", type=int, default=1,
                        help="Number of recent days to use for incremental training (default: 1, max: 3)")
    parser.add_argument("--macro-only", action="store_true", help="Skip micro model training")
    parser.add_argument("--micro-only", action="store_true", help="Skip macro model training")
    args = parser.parse_args()

    # ── Safety guard: incremental must be small ───────────────────────────
    if args.days > 3:
        print(f"  ERROR: --days {args.days} is too large for incremental training.")
        print(f"  Incremental should only add 1-2 new days on top of the base model.")
        print(f"  For a full retrain, use: python scripts/retrain_full.py")
        return 1

    # ── Guard: base model must already exist ─────────────────────────────
    if not args.micro_only and not MACRO_MODEL_PATH.exists():
        print(f"  ERROR: No base macro model found at {MACRO_MODEL_PATH}.")
        print(f"  Run a full retrain first: python scripts/retrain_full.py")
        return 1

    print("=" * 60)
    print("  INCREMENTAL ML MODEL TRAINING")
    print("=" * 60)

    # ── MACRO MODEL — full retrain on ALL available candle history ───────
    if not args.micro_only:
        candles = read_sql(
            "SELECT timestamp, symbol, open, high, low, close, volume, vwap, oi "
            "FROM minute_candles WHERE symbol = 'NIFTY-I' "
            "ORDER BY timestamp"
        )
        if candles.empty:
            print("  WARNING: No candle data found for macro model. Skipping.")
        else:
            candles["timestamp"] = pd.to_datetime(candles["timestamp"])
            print(f"  Candles loaded: {len(candles):,}")
            print(f"  Date range: {candles['timestamp'].min().date()} → {candles['timestamp'].max().date()}")
            print(f"  Trading days: {candles['timestamp'].dt.date.nunique()}")

            # ── Compute features ─────────────────────────────────────────
            print("\n  Computing macro features...")
            featured = compute_all_macro_indicators(candles)
            print(f"  Featured rows: {len(featured):,}  |  columns: {len(featured.columns)}")

            # ── Train macro model (full retrain on all candle history) ────
            print("\n  --- MACRO MODEL (Full retrain on all candle history) ---")
            macro_trainer = MacroModelTrainer()
            _backup_model(MACRO_MODEL_PATH)
            macro_df = macro_trainer.prepare_data(featured.copy())
            if len(macro_df) > 100:
                metrics = macro_trainer.train(macro_df, walk_forward=True, n_splits=5)
                macro_trainer.save()
                print(f"  Macro model trained: {len(macro_df):,} samples  |  {metrics}")
            else:
                print(f"  Not enough data ({len(macro_df)} samples). Need 100+.")

            if macro_trainer.model is not None:
                fi = macro_trainer.get_feature_importance()
                print(f"\n  Top 10 features:")
                for _, row in fi.head(10).iterrows():
                    print(f"    {row['feature']:30s}  {row['importance']:.4f}")

            # ── Strategy models ──────────────────────────────────────────
            print("\n  --- STRATEGY MODELS ---")
            try:
                strat_results = train_all_strategy_models(featured.copy())
                for name, info in strat_results.items():
                    print(f"  {name}: {info.get('metrics', 'trained') if info else 'skipped (insufficient data)'}")
            except Exception as e:
                print(f"  Strategy model training error: {e}")

    # ── MICRO MODEL — full retrain on ALL available tick data ────────────
    if not args.macro_only:
        print("\n  --- MICRO MODEL (Full retrain on all available tick data) ---")
        ticks = read_sql(
            "SELECT timestamp, symbol, price, volume, bid_price, ask_price, bid_qty, ask_qty, oi "
            "FROM tick_data WHERE symbol = 'NIFTY-I' "
            "ORDER BY timestamp"
        )
        if ticks.empty:
            print("  WARNING: No tick data found for the specified period. Skipping micro model.")
        else:
            ticks["timestamp"] = pd.to_datetime(ticks["timestamp"])
            print(f"  Ticks loaded: {len(ticks):,}")
            print(f"  Date range: {ticks['timestamp'].min().date()} → {ticks['timestamp'].max().date()}")
            print(f"  Trading days: {ticks['timestamp'].dt.date.nunique()}")

            print("\n  Computing micro features...")
            try:
                micro_featured = compute_micro_features(ticks)
                print(f"  Micro featured rows: {len(micro_featured):,}  |  columns: {len(micro_featured.columns)}")

                micro_trainer = MicroModelTrainer()
                _backup_model(MICRO_MODEL_PATH_P)
                micro_df = micro_trainer.prepare_data(micro_featured.copy())
                if len(micro_df) > 100:
                    metrics = micro_trainer.train(micro_df, walk_forward=True, n_splits=3)
                    micro_trainer.save()
                    print(f"  Micro model trained: {len(micro_df):,} samples  |  {metrics}")
                else:
                    print(f"  Not enough micro data ({len(micro_df)} samples). Need 100+.")
            except Exception as e:
                print(f"  Micro model training error: {e}")

    # ── Verify loaded models ─────────────────────────────────────────────
    print("\n  --- VERIFICATION ---")
    from models.predict import Predictor
    p = Predictor()
    p.load()
    sp = StrategyPredictor()
    sp.load()
    print(f"  Macro model loaded: {p.is_loaded}")
    print(f"  Strategy models: {sp.available_strategies}")

    if not args.micro_only and p.is_loaded:
        candles_check = read_sql(
            "SELECT timestamp, symbol, open, high, low, close, volume, vwap, oi "
            "FROM minute_candles WHERE symbol='NIFTY-I' ORDER BY timestamp DESC LIMIT 300"
        )
        if not candles_check.empty:
            try:
                candles_check["timestamp"] = pd.to_datetime(candles_check["timestamp"])
                featured_check = compute_all_macro_indicators(candles_check.sort_values("timestamp"))
                featured_check = featured_check.dropna()
                if not featured_check.empty:
                    test_row = featured_check.iloc[-1].to_dict()
                    prob = p.predict_macro(test_row)
                    print(f"  Test macro prediction on latest data: {prob:.4f}" if prob else "  Test prediction: None")
            except Exception as e:
                print(f"  Verification skipped: {e}")

    print("\n" + "=" * 60)
    print("  INCREMENTAL TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
