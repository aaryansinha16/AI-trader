"""
Model Registry
──────────────
Manages model versioning, storage, and retrieval.
Tracks all trained models with their metrics in the model_registry DB table.

Supports:
  - Registering new models (macro / micro)
  - Activating/deactivating models
  - Loading the currently active model for inference
  - Version history
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import joblib
import pandas as pd

from config.settings import MACRO_MODEL_PATH, MICRO_MODEL_PATH, MODEL_DIR
from database.db import execute_sql, read_sql
from utils.logger import get_logger

logger = get_logger("model_registry")


class ModelRegistry:
    """Persistent model registry backed by the model_registry DB table."""

    def register(
        self,
        model_name: str,
        model_type: str,
        file_path: str,
        metrics: Dict,
        train_start: str = None,
        train_end: str = None,
        metadata: Dict = None,
        activate: bool = True,
    ) -> int:
        """
        Register a newly trained model.

        Args:
            model_name: e.g. "macro_xgboost"
            model_type: "macro" or "micro"
            file_path: path to the saved .pkl file
            metrics: dict with accuracy, precision, recall, f1
            activate: if True, deactivate previous models of same type
        """
        # Determine next version
        existing = read_sql(
            "SELECT COALESCE(MAX(version), 0) as max_v "
            "FROM model_registry WHERE model_name = :name",
            {"name": model_name},
        )
        next_version = int(existing["max_v"].iloc[0]) + 1 if not existing.empty else 1

        if activate:
            # Deactivate all models of the same type
            execute_sql(
                "UPDATE model_registry SET is_active = FALSE "
                "WHERE model_type = :mtype",
                {"mtype": model_type},
            )

        import json
        execute_sql(
            """
            INSERT INTO model_registry
                (model_name, model_type, version, train_start, train_end,
                 accuracy, precision_score, recall_score, f1_score,
                 file_path, is_active, metadata)
            VALUES
                (:name, :mtype, :version, :train_start, :train_end,
                 :accuracy, :precision, :recall, :f1,
                 :file_path, :is_active, :metadata)
            """,
            {
                "name": model_name,
                "mtype": model_type,
                "version": next_version,
                "train_start": train_start,
                "train_end": train_end,
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "file_path": file_path,
                "is_active": activate,
                "metadata": json.dumps(metadata) if metadata else None,
            },
        )

        logger.info(
            f"Registered model: {model_name} v{next_version} "
            f"(type={model_type}, active={activate})"
        )
        return next_version

    def get_active_model_path(self, model_type: str) -> Optional[str]:
        """Get the file path of the currently active model for a given type."""
        df = read_sql(
            "SELECT file_path FROM model_registry "
            "WHERE model_type = :mtype AND is_active = TRUE "
            "ORDER BY trained_at DESC LIMIT 1",
            {"mtype": model_type},
        )
        if df.empty:
            return None
        return df["file_path"].iloc[0]

    def get_history(self, model_type: str = None) -> pd.DataFrame:
        """Get version history for all or a specific model type."""
        if model_type:
            return read_sql(
                "SELECT * FROM model_registry WHERE model_type = :mtype "
                "ORDER BY trained_at DESC",
                {"mtype": model_type},
            )
        return read_sql(
            "SELECT * FROM model_registry ORDER BY trained_at DESC"
        )

    def load_model(self, model_type: str) -> Optional[Dict]:
        """
        Load the active model for a given type.
        Returns dict with keys: model, features, metrics.
        Falls back to default file paths if DB is unavailable.
        """
        # Try DB registry first
        try:
            path = self.get_active_model_path(model_type)
        except Exception:
            path = None

        # Fallback to default paths
        if not path:
            path = MACRO_MODEL_PATH if model_type == "macro" else MICRO_MODEL_PATH

        if not Path(path).exists():
            logger.warning(f"No {model_type} model found at {path}")
            return None

        try:
            data = joblib.load(path)
            logger.info(f"Loaded {model_type} model from {path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {e}")
            return None