"""
Model Performance Monitor
─────────────────────────
Tracks ML model accuracy over time and detects concept drift.

Concept drift = when the statistical properties of the target variable
change over time, causing model predictions to degrade.

Monitors:
  - Daily prediction accuracy (predicted vs actual outcome)
  - Rolling accuracy window (7-day, 30-day)
  - Accuracy alerts when performance drops below threshold
  - Feature distribution shifts (mean/std drift)

Usage:
  monitor = ModelMonitor()
  monitor.log_prediction(symbol, "macro", predicted_prob=0.72, actual_outcome=1)
  monitor.log_prediction(symbol, "macro", predicted_prob=0.65, actual_outcome=0)
  report = monitor.get_daily_report()
  alerts = monitor.check_alerts()
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger

logger = get_logger("model_monitor")


@dataclass
class PredictionRecord:
    """A single prediction with its actual outcome."""
    timestamp: datetime
    symbol: str
    model_type: str        # "macro" or "micro"
    predicted_prob: float
    actual_outcome: int    # 1 = profitable, 0 = not
    predicted_class: int = 0   # computed in __post_init__
    correct: bool = False

    def __post_init__(self):
        self.predicted_class = 1 if self.predicted_prob >= 0.5 else 0
        self.correct = self.predicted_class == self.actual_outcome


@dataclass
class DailyModelReport:
    """Model performance for a single day."""
    date: date
    model_type: str
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    avg_confidence: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0


class ModelMonitor:
    """
    Tracks prediction accuracy and detects concept drift.

    Alerts are triggered when:
      - Daily accuracy drops below min_accuracy_threshold
      - Rolling 7-day accuracy drops below rolling_threshold
      - Prediction confidence diverges from actual hit rate (calibration drift)
    """

    def __init__(
        self,
        min_accuracy_threshold: float = 0.45,
        rolling_threshold: float = 0.48,
        rolling_window: int = 7,
    ):
        self.min_accuracy_threshold = min_accuracy_threshold
        self.rolling_threshold = rolling_threshold
        self.rolling_window = rolling_window

        # Store predictions by date and model_type
        self._records: Dict[str, List[PredictionRecord]] = defaultdict(list)
        # key = f"{date}_{model_type}"

        self._daily_reports: List[DailyModelReport] = []

    # ── Logging ─────────────────────────────────────────────────────────

    def log_prediction(
        self,
        symbol: str,
        model_type: str,
        predicted_prob: float,
        actual_outcome: int,
        timestamp: Optional[datetime] = None,
    ):
        """
        Log a single prediction with its actual outcome.

        Args:
            symbol: instrument symbol
            model_type: "macro" or "micro"
            predicted_prob: model's predicted probability (0.0 to 1.0)
            actual_outcome: 1 if the trade was profitable, 0 otherwise
            timestamp: when the prediction was made
        """
        ts = timestamp or datetime.now()
        record = PredictionRecord(
            timestamp=ts,
            symbol=symbol,
            model_type=model_type,
            predicted_prob=predicted_prob,
            actual_outcome=actual_outcome,
        )

        key = f"{ts.date()}_{model_type}"
        self._records[key].append(record)

    # ── Reports ─────────────────────────────────────────────────────────

    def compute_daily_report(
        self, target_date: date, model_type: str
    ) -> DailyModelReport:
        """Compute accuracy metrics for a specific day and model."""
        key = f"{target_date}_{model_type}"
        records = self._records.get(key, [])

        if not records:
            return DailyModelReport(date=target_date, model_type=model_type)

        total = len(records)
        correct = sum(1 for r in records if r.correct)
        avg_conf = np.mean([r.predicted_prob for r in records])

        tp = sum(1 for r in records if r.predicted_class == 1 and r.actual_outcome == 1)
        fp = sum(1 for r in records if r.predicted_class == 1 and r.actual_outcome == 0)
        tn = sum(1 for r in records if r.predicted_class == 0 and r.actual_outcome == 0)
        fn = sum(1 for r in records if r.predicted_class == 0 and r.actual_outcome == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        report = DailyModelReport(
            date=target_date,
            model_type=model_type,
            total_predictions=total,
            correct_predictions=correct,
            accuracy=round(correct / total, 4) if total > 0 else 0.0,
            avg_confidence=round(float(avg_conf), 4),
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            precision=round(precision, 4),
            recall=round(recall, 4),
        )

        self._daily_reports.append(report)
        return report

    def get_rolling_accuracy(
        self, model_type: str, window: int = None
    ) -> float:
        """Compute rolling accuracy over the last N days."""
        window = window or self.rolling_window
        relevant = [
            r for r in self._daily_reports
            if r.model_type == model_type and r.total_predictions > 0
        ]
        recent = relevant[-window:] if len(relevant) >= window else relevant

        if not recent:
            return 0.0

        total = sum(r.total_predictions for r in recent)
        correct = sum(r.correct_predictions for r in recent)
        return round(correct / total, 4) if total > 0 else 0.0

    # ── Alerts ──────────────────────────────────────────────────────────

    def check_alerts(self, model_type: str = "macro") -> List[str]:
        """
        Check for performance degradation alerts.
        Returns list of alert messages (empty = all OK).
        """
        alerts = []

        # Check latest daily accuracy
        relevant = [
            r for r in self._daily_reports
            if r.model_type == model_type and r.total_predictions > 0
        ]
        if relevant:
            latest = relevant[-1]
            if latest.accuracy < self.min_accuracy_threshold:
                alerts.append(
                    f"ALERT: {model_type} daily accuracy {latest.accuracy:.1%} "
                    f"below threshold {self.min_accuracy_threshold:.1%} "
                    f"on {latest.date}"
                )

        # Check rolling accuracy
        rolling_acc = self.get_rolling_accuracy(model_type)
        if rolling_acc > 0 and rolling_acc < self.rolling_threshold:
            alerts.append(
                f"ALERT: {model_type} {self.rolling_window}-day rolling accuracy "
                f"{rolling_acc:.1%} below threshold {self.rolling_threshold:.1%}. "
                f"Consider retraining."
            )

        # Check calibration drift: avg confidence vs actual hit rate
        if relevant and len(relevant) >= 3:
            recent = relevant[-3:]
            avg_confidence = np.mean([r.avg_confidence for r in recent])
            avg_accuracy = np.mean([r.accuracy for r in recent])
            calibration_gap = abs(avg_confidence - avg_accuracy)
            if calibration_gap > 0.15:
                alerts.append(
                    f"ALERT: {model_type} calibration drift detected. "
                    f"Avg confidence={avg_confidence:.1%}, "
                    f"Avg accuracy={avg_accuracy:.1%}, "
                    f"gap={calibration_gap:.1%}. Model may need retraining."
                )

        for alert in alerts:
            logger.warning(alert)

        return alerts

    # ── Summary ─────────────────────────────────────────────────────────

    def summary(self, model_type: str = "macro") -> str:
        """Human-readable performance summary."""
        relevant = [
            r for r in self._daily_reports
            if r.model_type == model_type and r.total_predictions > 0
        ]

        if not relevant:
            return f"No predictions logged for {model_type} model."

        latest = relevant[-1]
        rolling = self.get_rolling_accuracy(model_type)
        alerts = self.check_alerts(model_type)

        lines = [
            f"Model Monitor: {model_type}",
            f"  Latest day ({latest.date}): {latest.accuracy:.1%} accuracy "
            f"({latest.correct_predictions}/{latest.total_predictions})",
            f"  Precision: {latest.precision:.1%}, Recall: {latest.recall:.1%}",
            f"  Rolling {self.rolling_window}-day: {rolling:.1%}",
            f"  Avg confidence: {latest.avg_confidence:.1%}",
            f"  Alerts: {len(alerts)}",
        ]
        for a in alerts:
            lines.append(f"    ⚠ {a}")

        return "\n".join(lines)
