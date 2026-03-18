"""
ML Training Pipeline
────────────────────
Two separate trainers following the dual-model architecture:

  1. MacroModelTrainer  – trains on 1-minute candle features (6 months)
     Target: "Did price move +0.4% in next 10 minutes?"
     Or better: "Did the strategy hit target before stop?"

  2. MicroModelTrainer  – trains on tick/second-level features (5 days+)
     Target: "Did breakout occur within next 2 minutes?"

Both use walk-forward validation to avoid overfitting.

Models: XGBoost (primary), LightGBM (alternative)

From docs:
  - ML does NOT generate trades. It evaluates probability.
  - Never let ML decide trades alone.
  - Train on strategy success, not raw price movement.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit

from config.settings import (
    FEATURE_COLUMNS_MACRO,
    FEATURE_COLUMNS_MICRO,
    MACRO_MODEL_PATH,
    MICRO_MODEL_PATH,
    MODEL_DIR,
)
from utils.logger import get_logger

logger = get_logger("train_model")


# ═══════════════════════════════════════════════════════════════════════════════
# Label Generation
# ═══════════════════════════════════════════════════════════════════════════════


def generate_macro_labels(
    df: pd.DataFrame,
    forward_periods: int = 15,
    threshold: float = 0.001,
) -> pd.DataFrame:
    """
    Generate labels for the Macro Model.

    Default target: "Did price move +0.1% in next 15 minutes?"
    Tuned for ~13% positive rate to avoid extreme class imbalance.
    Can be swapped to strategy-outcome labels once trade log exists.

    Args:
        df: DataFrame with 'close' column (1-minute candles with features)
        forward_periods: how many candles to look ahead
        threshold: minimum return to count as positive
    """
    df = df.copy()
    future_return = df["close"].shift(-forward_periods) / df["close"] - 1
    df["target"] = (future_return > threshold).astype(int)
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)

    pos = df["target"].sum()
    neg = len(df) - pos
    logger.info(
        f"Macro labels: {len(df)} samples, "
        f"{pos} positive ({pos / len(df) * 100:.1f}%), "
        f"{neg} negative"
    )
    return df


def generate_micro_labels(
    df: pd.DataFrame,
    forward_seconds: int = 60,
    threshold: float = 0.001,
) -> pd.DataFrame:
    """
    Generate labels for the Micro Model.

    Default target: "Did price breakout (+0.1%) within next 60 seconds?"
    Tuned for better class balance on tick-level data.
    """
    df = df.copy()

    # Micro features are at 1-second resolution
    # forward_seconds rows ahead
    if "price" in df.columns:
        price_col = "price"
    elif "close" in df.columns:
        price_col = "close"
    else:
        logger.error("No price column found for micro label generation.")
        return df

    future_return = df[price_col].shift(-forward_seconds) / df[price_col] - 1
    df["target"] = (future_return > threshold).astype(int)
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)

    pos = df["target"].sum()
    logger.info(
        f"Micro labels: {len(df)} samples, "
        f"{pos} positive ({pos / len(df) * 100:.1f}%)"
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Walk-Forward Validation
# ═══════════════════════════════════════════════════════════════════════════════


def walk_forward_split(
    df: pd.DataFrame,
    n_splits: int = 5,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Time-series walk-forward splits.

    Example with 5 splits on 6 months of data:
      Split 1: Train Jan–Feb,  Test Mar
      Split 2: Train Jan–Mar,  Test Apr
      Split 3: Train Jan–Apr,  Test May
      ...

    Returns list of (train_df, test_df) tuples.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []

    for train_idx, test_idx in tscv.split(df):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        splits.append((train, test))

    return splits


# ═══════════════════════════════════════════════════════════════════════════════
# Model Training
# ═══════════════════════════════════════════════════════════════════════════════


def _get_model(model_type: str = "xgboost", **kwargs):
    """Create a fresh model instance."""
    if model_type == "xgboost":
        import xgboost as xgb

        defaults = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "random_state": 42,
            "scale_pos_weight": 1,  # overridden per-call when imbalanced
        }
        defaults.update(kwargs)
        return xgb.XGBClassifier(**defaults)

    elif model_type == "lightgbm":
        import lightgbm as lgb

        defaults = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbose": -1,
        }
        defaults.update(kwargs)
        return lgb.LGBMClassifier(**defaults)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _evaluate(model, X_test, y_test) -> Dict:
    """Evaluate a trained model and return metrics dict."""
    from sklearn.metrics import roc_auc_score

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    try:
        auc = round(roc_auc_score(y_test, y_proba), 4)
    except ValueError:
        auc = 0.0

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "auc_roc": auc,
    }
    return metrics


class MacroModelTrainer:
    """
    Trains the Macro ML Model on 1-minute candle features.
    Uses walk-forward validation.
    """

    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = None
        self.feature_cols = FEATURE_COLUMNS_MACRO
        self.metrics: Dict = {}

    def prepare_data(
        self,
        features_df: pd.DataFrame,
        forward_periods: int = 15,
        threshold: float = 0.001,
    ) -> pd.DataFrame:
        """Add labels and drop rows with NaN features."""
        df = generate_macro_labels(features_df, forward_periods, threshold)

        available = [c for c in self.feature_cols if c in df.columns]

        # Drop features that are entirely NaN (e.g. options features for index)
        non_null = [c for c in available if df[c].notna().any()]
        dropped = set(available) - set(non_null)
        if dropped:
            logger.info(f"Dropping all-NaN features: {dropped}")
        self.feature_cols = non_null

        # Fill remaining sporadic NaNs (e.g. first few rows for rolling indicators)
        df[non_null] = df[non_null].ffill().bfill()
        df = df.dropna(subset=non_null + ["target"])
        logger.info(f"Prepared {len(df)} samples with {len(non_null)} features.")
        return df

    def train(
        self,
        df: pd.DataFrame,
        walk_forward: bool = True,
        n_splits: int = 5,
        **model_kwargs,
    ) -> Dict:
        """
        Train the macro model.

        If walk_forward=True, uses walk-forward validation and trains
        the final model on the full dataset.

        Returns metrics dict.
        """
        X = df[self.feature_cols]
        y = df["target"]

        # Auto-compute class weight for imbalanced data
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        if pos_count > 0 and "scale_pos_weight" not in model_kwargs:
            model_kwargs["scale_pos_weight"] = round(neg_count / pos_count, 2)
            logger.info(f"Auto scale_pos_weight={model_kwargs['scale_pos_weight']} (neg/pos={neg_count}/{pos_count})")

        if walk_forward:
            logger.info(f"Walk-forward validation with {n_splits} splits...")
            splits = walk_forward_split(df, n_splits)
            all_metrics = []

            for i, (train_df, test_df) in enumerate(splits):
                X_tr = train_df[self.feature_cols]
                y_tr = train_df["target"]
                X_te = test_df[self.feature_cols]
                y_te = test_df["target"]

                model = _get_model(self.model_type, **model_kwargs)
                model.fit(X_tr, y_tr)
                m = _evaluate(model, X_te, y_te)
                all_metrics.append(m)
                logger.info(f"  Split {i + 1}: {m}")

            # Average metrics across folds
            self.metrics = {
                k: round(np.mean([m[k] for m in all_metrics]), 4)
                for k in all_metrics[0]
            }
            logger.info(f"Walk-forward avg metrics: {self.metrics}")

        # Train final model on full data
        logger.info("Training final macro model on full dataset...")
        self.model = _get_model(self.model_type, **model_kwargs)
        self.model.fit(X, y)

        if not walk_forward:
            self.metrics = _evaluate(self.model, X, y)

        return self.metrics

    def incremental_train(
        self,
        new_df: pd.DataFrame,
        existing_model_path: str = None,
        **model_kwargs,
    ) -> Dict:
        """
        Incremental (warm-start) training for daily macro model updates.
        Loads existing model and continues boosting on new day's data.

        Args:
            new_df: new day's feature data (already prepared with labels)
            existing_model_path: path to existing .pkl model file
        """
        existing_path = existing_model_path or MACRO_MODEL_PATH

        try:
            existing_data = joblib.load(existing_path)
            self.model = existing_data["model"]
            self.feature_cols = existing_data.get("features", self.feature_cols)
            logger.info(f"Loaded existing macro model from {existing_path}")
        except (FileNotFoundError, Exception) as e:
            logger.warning(f"No existing model ({e}). Training from scratch.")
            return self.train(new_df, walk_forward=False, **model_kwargs)

        X_new = new_df[self.feature_cols]
        y_new = new_df["target"]

        if len(X_new) < 10:
            logger.warning(f"Only {len(X_new)} new samples. Skipping incremental train.")
            return self.metrics

        try:
            if self.model_type == "xgboost":
                new_model = _get_model(self.model_type, n_estimators=50, **model_kwargs)
                new_model.fit(X_new, y_new, xgb_model=self.model.get_booster())
                self.model = new_model
            elif self.model_type == "lightgbm":
                new_model = _get_model(self.model_type, n_estimators=50, **model_kwargs)
                new_model.fit(X_new, y_new, init_model=self.model)
                self.model = new_model
            else:
                self.model = _get_model(self.model_type, **model_kwargs)
                self.model.fit(X_new, y_new)

            self.metrics = _evaluate(self.model, X_new, y_new)
            logger.info(
                f"Macro incremental training on {len(X_new)} samples. "
                f"Metrics: {self.metrics}"
            )
        except Exception as e:
            logger.error(f"Macro incremental training failed: {e}. Keeping existing model.")

        return self.metrics

    def save(self, path: str = None):
        """Save trained model to disk."""
        path = path or MACRO_MODEL_PATH
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "features": self.feature_cols, "metrics": self.metrics},
            path,
        )
        logger.info(f"Macro model saved to {path}")

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance as a sorted DataFrame."""
        if self.model is None:
            return pd.DataFrame()

        importance = self.model.feature_importances_
        fi = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": importance,
        }).sort_values("importance", ascending=False)
        return fi


class MicroModelTrainer:
    """
    Trains the Microstructure Model on tick/second-level features.
    Uses walk-forward validation.
    Supports incremental (warm-start) training for daily updates.
    """

    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = None
        self.feature_cols = FEATURE_COLUMNS_MICRO
        self.metrics: Dict = {}

    def prepare_data(
        self,
        features_df: pd.DataFrame,
        forward_seconds: int = 60,
        threshold: float = 0.001,
    ) -> pd.DataFrame:
        """Add labels and drop NaN rows."""
        df = generate_micro_labels(features_df, forward_seconds, threshold)

        available = [c for c in self.feature_cols if c in df.columns]
        non_null = [c for c in available if df[c].notna().any()]
        dropped = set(available) - set(non_null)
        if dropped:
            logger.info(f"Dropping all-NaN micro features: {dropped}")
        self.feature_cols = non_null

        df[non_null] = df[non_null].ffill().bfill()
        df = df.dropna(subset=non_null + ["target"])
        logger.info(f"Prepared {len(df)} micro samples with {len(non_null)} features.")
        return df

    def train(
        self,
        df: pd.DataFrame,
        walk_forward: bool = True,
        n_splits: int = 5,
        **model_kwargs,
    ) -> Dict:
        """Train the micro model with optional walk-forward validation."""
        X = df[self.feature_cols]
        y = df["target"]

        # Auto-compute class weight for imbalanced data
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        if pos_count > 0 and "scale_pos_weight" not in model_kwargs:
            model_kwargs["scale_pos_weight"] = round(neg_count / pos_count, 2)
            logger.info(f"Micro auto scale_pos_weight={model_kwargs['scale_pos_weight']}")

        if walk_forward:
            logger.info(f"Micro walk-forward with {n_splits} splits...")
            splits = walk_forward_split(df, n_splits)
            all_metrics = []

            for i, (train_df, test_df) in enumerate(splits):
                X_tr = train_df[self.feature_cols]
                y_tr = train_df["target"]
                X_te = test_df[self.feature_cols]
                y_te = test_df["target"]

                model = _get_model(self.model_type, **model_kwargs)
                model.fit(X_tr, y_tr)
                m = _evaluate(model, X_te, y_te)
                all_metrics.append(m)
                logger.info(f"  Micro split {i + 1}: {m}")

            self.metrics = {
                k: round(np.mean([m[k] for m in all_metrics]), 4)
                for k in all_metrics[0]
            }
            logger.info(f"Micro walk-forward avg: {self.metrics}")

        logger.info("Training final micro model on full dataset...")
        self.model = _get_model(self.model_type, **model_kwargs)
        self.model.fit(X, y)

        if not walk_forward:
            self.metrics = _evaluate(self.model, X, y)

        return self.metrics

    def incremental_train(
        self,
        new_df: pd.DataFrame,
        existing_model_path: str = None,
        **model_kwargs,
    ) -> Dict:
        """
        Incremental (warm-start) training: load existing model and continue
        training on new data. This avoids retraining from scratch daily.

        For XGBoost: uses xgb_model parameter to continue boosting.
        For LightGBM: uses init_model parameter.

        Args:
            new_df: new day's data (already prepared with labels)
            existing_model_path: path to existing .pkl model file
        """
        existing_path = existing_model_path or MICRO_MODEL_PATH

        # Load existing model
        existing_data = None
        try:
            existing_data = joblib.load(existing_path)
            self.model = existing_data["model"]
            self.feature_cols = existing_data.get("features", self.feature_cols)
            logger.info(f"Loaded existing micro model from {existing_path}")
        except (FileNotFoundError, Exception) as e:
            logger.warning(f"No existing model found ({e}). Training from scratch.")
            return self.train(new_df, walk_forward=False, **model_kwargs)

        X_new = new_df[self.feature_cols]
        y_new = new_df["target"]

        if len(X_new) < 10:
            logger.warning(f"Only {len(X_new)} new samples. Skipping incremental train.")
            return self.metrics

        try:
            if self.model_type == "xgboost":
                # XGBoost warm-start: pass existing booster
                new_model = _get_model(self.model_type, n_estimators=50, **model_kwargs)
                new_model.fit(X_new, y_new, xgb_model=self.model.get_booster())
                self.model = new_model
            elif self.model_type == "lightgbm":
                # LightGBM warm-start: pass init_model
                new_model = _get_model(self.model_type, n_estimators=50, **model_kwargs)
                new_model.fit(X_new, y_new, init_model=self.model)
                self.model = new_model
            else:
                # Fallback: retrain from scratch
                self.model = _get_model(self.model_type, **model_kwargs)
                self.model.fit(X_new, y_new)

            self.metrics = _evaluate(self.model, X_new, y_new)
            logger.info(
                f"Incremental training complete on {len(X_new)} new samples. "
                f"Metrics: {self.metrics}"
            )
        except Exception as e:
            logger.error(f"Incremental training failed: {e}. Keeping existing model.")

        return self.metrics

    def save(self, path: str = None):
        """Save trained model to disk."""
        path = path or MICRO_MODEL_PATH
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "features": self.feature_cols, "metrics": self.metrics},
            path,
        )
        logger.info(f"Micro model saved to {path}")

    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame()
        importance = self.model.feature_importances_
        return pd.DataFrame({
            "feature": self.feature_cols,
            "importance": importance,
        }).sort_values("importance", ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience: Full Training Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def train_all_models(
    macro_features_df: pd.DataFrame,
    micro_features_df: pd.DataFrame = None,
    model_type: str = "xgboost",
) -> Dict:
    """
    Train both macro and micro models from DataFrames.
    Returns dict of metrics for both models.

    NOTE: Call this ONLY with real data (TrueData), never with mock data.
    """
    results = {}

    # Macro model
    macro_trainer = MacroModelTrainer(model_type)
    macro_df = macro_trainer.prepare_data(macro_features_df)
    if len(macro_df) > 100:
        macro_metrics = macro_trainer.train(macro_df)
        macro_trainer.save()
        results["macro"] = macro_metrics
        logger.info(f"Macro model feature importance:\n{macro_trainer.get_feature_importance()}")
    else:
        logger.warning("Not enough macro data to train. Need at least 100 samples.")

    # Micro model
    if micro_features_df is not None and not micro_features_df.empty:
        micro_trainer = MicroModelTrainer(model_type)
        micro_df = micro_trainer.prepare_data(micro_features_df)
        if len(micro_df) > 100:
            micro_metrics = micro_trainer.train(micro_df)
            micro_trainer.save()
            results["micro"] = micro_metrics
        else:
            logger.warning("Not enough micro data to train.")
    else:
        logger.info("No micro features provided; skipping micro model.")

    return results