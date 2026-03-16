"""
Trade Scoring & Ranking Engine
──────────────────────────────
From the Product Vision doc (§12, §13):

  Trade Score =
    0.5 * ML probability
  + 0.3 * options flow score
  + 0.2 * technical strength

  Every scan cycle:
    Scan market → Generate signals → Score trades → Select top 3

  Example output:
    1. NIFTY 22500 CE   score 0.71
    2. BANKNIFTY 47000 PE  score 0.69
    3. ICICI BANK CALL   score 0.63
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config.settings import (
    SCORE_THRESHOLD,
    WEIGHT_ML_PROBABILITY,
    WEIGHT_OPTIONS_FLOW,
    WEIGHT_TECHNICAL_STRENGTH,
)
from strategy.signal_generator import Signal
from utils.logger import get_logger

logger = get_logger("trade_scorer")


@dataclass
class ScoredTrade:
    """A fully scored trade candidate ready for risk validation."""
    signal: Signal
    ml_probability: float
    flow_score: float
    technical_strength: float
    final_score: float
    rank: int = 0
    regime: str = ""
    details: Dict = field(default_factory=dict)

    @property
    def symbol(self) -> str:
        return self.signal.symbol

    @property
    def direction(self) -> str:
        return self.signal.direction

    @property
    def strategy(self) -> str:
        return self.signal.strategy

    @property
    def entry_price(self) -> float:
        return self.signal.entry_price


class TradeScorer:
    """
    Computes composite trade scores and ranks candidates.

    Scoring formula (from docs):
      final_score = w_ml * ml_prob + w_flow * flow_score + w_tech * tech_strength

    Default weights: 0.5, 0.3, 0.2
    """

    def __init__(
        self,
        w_ml: float = WEIGHT_ML_PROBABILITY,
        w_flow: float = WEIGHT_OPTIONS_FLOW,
        w_tech: float = WEIGHT_TECHNICAL_STRENGTH,
        threshold: float = SCORE_THRESHOLD,
        max_trades: int = 3,
    ):
        self.w_ml = w_ml
        self.w_flow = w_flow
        self.w_tech = w_tech
        self.threshold = threshold
        self.max_trades = max_trades

    def score_signal(
        self,
        signal: Signal,
        ml_probability: float = 0.0,
        flow_score: float = 0.0,
        regime: str = "",
    ) -> ScoredTrade:
        """
        Score a single signal.

        Args:
            signal: Signal from the strategy engine
            ml_probability: combined ML prob (macro + micro)
            flow_score: options flow detector score
            regime: current market regime string

        Returns ScoredTrade with computed final_score.
        """
        tech_strength = signal.technical_strength

        final_score = (
            self.w_ml * ml_probability
            + self.w_flow * flow_score
            + self.w_tech * tech_strength
        )

        return ScoredTrade(
            signal=signal,
            ml_probability=round(ml_probability, 4),
            flow_score=round(flow_score, 4),
            technical_strength=round(tech_strength, 4),
            final_score=round(final_score, 4),
            regime=regime,
            details={
                "w_ml": self.w_ml,
                "w_flow": self.w_flow,
                "w_tech": self.w_tech,
            },
        )

    def rank_trades(
        self,
        signals: List[Signal],
        ml_probabilities: Dict[str, float] = None,
        flow_scores: Dict[str, float] = None,
        regime: str = "",
    ) -> List[ScoredTrade]:
        """
        Score all signals and return ranked list (top N above threshold).

        Args:
            signals: list of Signal objects from strategy engine
            ml_probabilities: {symbol: probability} from Predictor
            flow_scores: {symbol: flow_score} from OptionsFlowDetector
            regime: current market regime

        Returns list of ScoredTrade sorted by final_score descending,
        filtered by threshold, limited to max_trades.
        """
        ml_probabilities = ml_probabilities or {}
        flow_scores = flow_scores or {}

        scored = []
        for sig in signals:
            ml_prob = ml_probabilities.get(sig.symbol, 0.5)
            flow = flow_scores.get(sig.symbol, 0.0)

            trade = self.score_signal(sig, ml_prob, flow, regime)
            scored.append(trade)

        # Sort by score descending
        scored.sort(key=lambda t: t.final_score, reverse=True)

        # Filter by threshold
        qualified = [t for t in scored if t.final_score >= self.threshold]

        # Select top N
        top = qualified[: self.max_trades]

        # Assign ranks
        for i, t in enumerate(top):
            t.rank = i + 1

        # Log results
        if top:
            logger.info(f"Trade ranking ({len(top)} qualified):")
            for t in top:
                logger.info(
                    f"  #{t.rank} {t.symbol} {t.direction} "
                    f"score={t.final_score:.2f} "
                    f"(ml={t.ml_probability:.2f}, "
                    f"flow={t.flow_score:.2f}, "
                    f"tech={t.technical_strength:.2f}) "
                    f"[{t.strategy}]"
                )
        else:
            logger.info("No trades qualified this cycle.")

        return top
