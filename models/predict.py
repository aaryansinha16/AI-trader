"""
Dual-Model Predictor
────────────────────
Loads both Macro and Micro models and provides unified prediction.

From the docs (Elaborated Challenges):
  trade_score = 0.5 * minute_model + 0.3 * options_flow + 0.2 * tick_model

This module provides the ML probability component (macro + micro combined).
The options_flow and technical scores are added by the Trade Scoring Engine.
"""

from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

from config.settings import MACRO_MODEL_PATH, MICRO_MODEL_PATH
from utils.logger import get_logger

logger = get_logger("predictor")


class Predictor:
    """
    Loads and manages both macro and micro models for inference.
    Gracefully handles missing models (returns None for unavailable predictions).
    """

    def __init__(self):
        self._macro_model = None
        self._macro_features: list = []
        self._micro_model = None
        self._micro_features: list = []
        self._loaded = False

    def load(
        self,
        macro_path: str = None,
        micro_path: str = None,
    ):
        """Load both models from disk. Either can be missing."""
        macro_path = macro_path or MACRO_MODEL_PATH
        micro_path = micro_path or MICRO_MODEL_PATH

        # Load macro model
        if Path(macro_path).exists():
            try:
                data = joblib.load(macro_path)
                self._macro_model = data["model"]
                self._macro_features = data["features"]
                logger.info(
                    f"Macro model loaded ({len(self._macro_features)} features)."
                )
            except Exception as e:
                logger.error(f"Failed to load macro model: {e}")
        else:
            logger.warning(f"No macro model at {macro_path}")

        # Load micro model
        if Path(micro_path).exists():
            try:
                data = joblib.load(micro_path)
                self._micro_model = data["model"]
                self._micro_features = data["features"]
                logger.info(
                    f"Micro model loaded ({len(self._micro_features)} features)."
                )
            except Exception as e:
                logger.error(f"Failed to load micro model: {e}")
        else:
            logger.warning(f"No micro model at {micro_path}")

        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded and (
            self._macro_model is not None or self._micro_model is not None
        )

    # ── Individual Predictions ────────────────────────────────────────────────

    def predict_macro(self, features: dict) -> Optional[float]:
        """
        Get Macro Model probability for a single feature row.
        Returns P(success) between 0 and 1, or None if model unavailable.
        """
        if self._macro_model is None:
            return None

        try:
            df = pd.DataFrame([features])
            # Select only the features the model was trained on
            available = [c for c in self._macro_features if c in df.columns]
            if not available:
                logger.warning("No matching macro features in input.")
                return None

            X = df[available]
            prob = float(self._macro_model.predict_proba(X)[0][1])
            return prob
        except Exception as e:
            logger.error(f"Macro prediction error: {e}")
            return None

    def predict_micro(self, features: dict) -> Optional[float]:
        """
        Get Micro Model probability for a single feature row.
        Returns P(breakout) between 0 and 1, or None if model unavailable.
        """
        if self._micro_model is None:
            return None

        try:
            df = pd.DataFrame([features])
            available = [c for c in self._micro_features if c in df.columns]
            if not available:
                logger.warning("No matching micro features in input.")
                return None

            X = df[available]
            prob = float(self._micro_model.predict_proba(X)[0][1])
            return prob
        except Exception as e:
            logger.error(f"Micro prediction error: {e}")
            return None

    # ── Combined Prediction ───────────────────────────────────────────────────

    def predict_combined(
        self,
        macro_features: dict,
        micro_features: dict = None,
        macro_weight: float = 0.7,
        micro_weight: float = 0.3,
    ) -> Dict:
        """
        Combined ML probability from both models.

        If only macro model is available, returns macro probability alone.
        If both are available, returns weighted combination.

        Returns dict:
          {
            "macro_prob": float or None,
            "micro_prob": float or None,
            "combined_ml_prob": float or None,
          }
        """
        macro_prob = self.predict_macro(macro_features)
        micro_prob = (
            self.predict_micro(micro_features)
            if micro_features is not None
            else None
        )

        # Compute combined probability
        if macro_prob is not None and micro_prob is not None:
            combined = macro_weight * macro_prob + micro_weight * micro_prob
        elif macro_prob is not None:
            combined = macro_prob
        elif micro_prob is not None:
            combined = micro_prob
        else:
            combined = None

        return {
            "macro_prob": macro_prob,
            "micro_prob": micro_prob,
            "combined_ml_prob": combined,
        }

    # ── Batch Prediction (for backtesting) ────────────────────────────────────

    def predict_macro_batch(self, df: pd.DataFrame) -> pd.Series:
        """Run macro model on a DataFrame. Returns Series of probabilities."""
        if self._macro_model is None:
            return pd.Series(dtype=float)

        available = [c for c in self._macro_features if c in df.columns]
        if not available:
            return pd.Series(dtype=float)

        X = df[available].copy()
        X = X.fillna(0)
        probs = self._macro_model.predict_proba(X)[:, 1]
        return pd.Series(probs, index=df.index)


# ── Legacy compatibility ──────────────────────────────────────────────────────

_predictor: Optional[Predictor] = None


def predict(features: dict) -> float:
    """
    Legacy function for backward compatibility.
    Loads macro model on first call.
    """
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
        _predictor.load()

    result = _predictor.predict_macro(features)
    return result if result is not None else 0.0