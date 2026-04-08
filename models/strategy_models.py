"""
Strategy-Specific ML Models
────────────────────────────
Instead of one model predicting "will price go up?", train separate models for:

  1. VWAP Breakout Model  → "will VWAP breakout strategy succeed?"
  2. Bearish Momentum Model → "will bearish momentum strategy succeed?"
  3. Mean Reversion Model → "will mean reversion strategy succeed?"

Each model is trained only on samples where its strategy WOULD have fired,
with labels based on whether the trade would have been profitable.

This dramatically improves prediction accuracy because each model learns
the specific conditions that make its strategy work.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

from config.settings import FEATURE_COLUMNS_MACRO, MODEL_DIR
from strategy.signal_generator import (
    vwap_momentum_breakout,
    bearish_momentum,
    mean_reversion,
    STRATEGY_MAP,
)
from utils.logger import get_logger

logger = get_logger("strategy_models")

STRATEGY_MODEL_DIR = MODEL_DIR / "strategy"
STRATEGY_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _get_model(model_type: str = "xgboost", **kwargs):
    """Create a fresh model instance."""
    if model_type == "xgboost":
        import xgboost as xgb
        defaults = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "random_state": 42,
            "scale_pos_weight": 1,
        }
        defaults.update(kwargs)
        return xgb.XGBClassifier(**defaults)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def generate_strategy_labels(
    df: pd.DataFrame,
    strategy_name: str,
    forward_periods: int = 40,
    sl_pct: float = 0.003,
    tgt_pct: float = 0.008,
) -> pd.DataFrame:
    """
    Generate labels specific to a strategy.

    For each row where the strategy signal fires:
      - Look forward `forward_periods` bars (matches max_hold_bars=40)
      - Check if NIFTY price hit target before stop
      - sl_pct=0.3%, tgt_pct=0.8% approximate real option RR (~2.5x) via delta
        (ATM delta≈0.45: 15% option SL ≈ 0.3% NIFTY move; 50% option TGT ≈ 0.8%)
      - Direction comes from the actual signal (handles mean_reversion CALL/PUT)
      - Label 1 if strategy would have been profitable, 0 otherwise

    For rows where the strategy doesn't fire: excluded from training.
    """
    df = df.copy()
    strategy_func = STRATEGY_MAP.get(strategy_name)
    if strategy_func is None:
        logger.error(f"Unknown strategy: {strategy_name}")
        return pd.DataFrame()

    # Check which rows trigger this strategy and record the signal direction
    fires = []
    directions = []
    for i, row in df.iterrows():
        row_dict = row.to_dict()
        signal = strategy_func(row_dict, "NIFTY-I")
        fires.append(signal is not None)
        directions.append(signal.direction if signal is not None else None)

    df["_fires"] = fires
    df["_direction"] = directions
    df = df[df["_fires"]].copy()
    df.drop(columns=["_fires"], inplace=True)

    if df.empty:
        logger.warning(f"Strategy {strategy_name} never fires. No training data.")
        return pd.DataFrame()

    # Generate forward return labels
    # Direction-aware: use actual signal direction (PUT or CALL) for each bar.
    # For PUT: win if NIFTY drops tgt_pct before rising sl_pct (bearish edge).
    # For CALL: win if NIFTY rises tgt_pct before dropping sl_pct (bullish edge).
    labels = []
    close = df["close"].values
    signal_dirs = df["_direction"].values
    indices = df.index.tolist()

    for j, idx in enumerate(indices):
        pos = df.index.get_loc(idx)
        entry = close[pos]
        direction = signal_dirs[j]

        # Look forward
        won = False
        for k in range(1, min(forward_periods + 1, len(close) - pos)):
            future_price = close[pos + k]
            ret = (future_price - entry) / entry

            if direction == "PUT":
                # PUT: profit if price drops by tgt_pct before rising sl_pct
                if ret < -tgt_pct:
                    won = True
                    break
                if ret > sl_pct:
                    break  # stopped out
            else:
                # CALL: profit if price rises by tgt_pct before dropping sl_pct
                if ret > tgt_pct:
                    won = True
                    break
                if ret < -sl_pct:
                    break  # stopped out

        labels.append(1 if won else 0)

    df["target"] = labels
    df.drop(columns=["_direction"], inplace=True, errors="ignore")

    pos = sum(labels)
    total = len(labels)
    logger.info(
        f"Strategy {strategy_name}: {total} signal rows, "
        f"{pos} wins ({pos/total*100:.1f}%)"
    )
    return df


def train_strategy_model(
    features_df: pd.DataFrame,
    strategy_name: str,
    model_type: str = "xgboost",
) -> Optional[Dict]:
    """Train a strategy-specific model."""
    logger.info(f"\n{'='*40}")
    logger.info(f"Training model for: {strategy_name}")
    logger.info(f"{'='*40}")

    df = generate_strategy_labels(features_df, strategy_name)
    if df.empty or len(df) < 50:
        logger.warning(f"Not enough data for {strategy_name} ({len(df)} samples)")
        return None

    # Select features
    available = [c for c in FEATURE_COLUMNS_MACRO if c in df.columns]
    non_null = [c for c in available if df[c].notna().any()]
    df[non_null] = df[non_null].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    df = df.dropna(subset=non_null + ["target"])

    if len(df) < 50:
        logger.warning(f"After cleanup: only {len(df)} rows for {strategy_name}")
        return None

    X = df[non_null]
    y = df["target"]

    # Auto class weight
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    spw = round(neg / pos, 2) if pos > 0 else 1
    logger.info(f"  Samples: {len(df)}, pos: {pos} ({pos/len(df)*100:.1f}%), scale_pos_weight: {spw}")

    model = _get_model(model_type, scale_pos_weight=spw)

    # Walk-forward validation (3 splits for smaller datasets)
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    n_splits = min(3, max(2, len(df) // 100))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics_list = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        m = _get_model(model_type, scale_pos_weight=spw)
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
        y_prob = m.predict_proba(X_te)[:, 1]
        try:
            auc = round(roc_auc_score(y_te, y_prob), 4)
        except:
            auc = 0.0
        fold_metrics = {
            "precision": round(precision_score(y_te, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_te, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_te, y_pred, zero_division=0), 4),
            "auc_roc": auc,
        }
        metrics_list.append(fold_metrics)
        logger.info(f"  Fold {fold+1}: {fold_metrics}")

    avg_metrics = {k: round(np.mean([m[k] for m in metrics_list]), 4) for k in metrics_list[0]}
    logger.info(f"  Avg: {avg_metrics}")

    # Train final model on all data
    model.fit(X, y)

    # Save
    path = STRATEGY_MODEL_DIR / f"{strategy_name}_model.pkl"
    joblib.dump({
        "model": model,
        "features": non_null,
        "strategy": strategy_name,
        "metrics": avg_metrics,
        "n_samples": len(df),
        "trained_at": datetime.now().isoformat(),
    }, path)
    logger.info(f"  Saved to {path}")

    return avg_metrics


def train_all_strategy_models(features_df: pd.DataFrame) -> Dict:
    """Train models for all strategies."""
    results = {}
    for name in STRATEGY_MAP:
        metrics = train_strategy_model(features_df, name)
        if metrics:
            results[name] = metrics
    return results


class StrategyPredictor:
    """Loads and serves strategy-specific models."""

    def __init__(self):
        self._models = {}  # strategy_name -> {model, features}

    def load(self):
        """Load all available strategy models."""
        for name in STRATEGY_MAP:
            path = STRATEGY_MODEL_DIR / f"{name}_model.pkl"
            if path.exists():
                try:
                    data = joblib.load(path)
                    self._models[name] = data
                    metrics = data.get("metrics") or {}
                    auc = metrics.get("auc_roc", data.get("cv_auc", "?"))
                    n = data.get("n_samples", "?")
                    logger.info(
                        f"Loaded {name} model ({len(data['features'])} features, "
                        f"n={n}, AUC={auc})"
                    )
                except Exception as e:
                    logger.error(f"Failed to load {name} model: {e}")

    def predict(self, strategy_name: str, features: dict) -> Optional[float]:
        """Get P(strategy success) for a single feature row."""
        if strategy_name not in self._models:
            return None
        data = self._models[strategy_name]
        try:
            df = pd.DataFrame([features])
            available = [c for c in data["features"] if c in df.columns]
            if not available:
                return None
            X = df[available].fillna(0)
            return float(data["model"].predict_proba(X)[0][1])
        except Exception as e:
            logger.error(f"Strategy prediction error ({strategy_name}): {e}")
            return None

    @property
    def available_strategies(self) -> list:
        return list(self._models.keys())
