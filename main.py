"""
AI Trading System - Main Entry Point

Supports five modes:

  1. mock      - Generate mock data, build features (no DB required)
  2. ingest    - Load historical data from TrueData into DB, build features
  3. train     - Train ML models from DB data (requires ingest first)
  4. backtest  - Run backtest on mock data with full pipeline
  5. live      - Real-time trading loop

Usage:
  python main.py mock
  python main.py ingest
  python main.py train
  python main.py backtest
  python main.py live
"""

import sys
import time

from config.settings import SYMBOLS, SCAN_INTERVAL_SECONDS
from utils.logger import get_logger

logger = get_logger("main")


def run_mock():
    """
    Generate mock data and run the feature pipeline end-to-end.
    No database required – operates entirely in-memory.
    """
    from data.mock_data import generate_all_mock_data
    from data.aggregator import AggregationEngine
    from features.indicators import compute_all_macro_indicators
    from features.micro_features import compute_micro_features

    logger.info("=" * 60)
    logger.info("MODE: MOCK DATA – generating synthetic dataset")
    logger.info("=" * 60)

    # 1. Generate mock data
    mock = generate_all_mock_data()
    minute_bars = mock["minute_bars"]
    ticks = mock["ticks"]
    option_chain = mock["option_chain"]

    logger.info(
        f"Mock data generated: "
        f"{len(minute_bars)} minute bars, "
        f"{len(ticks)} ticks, "
        f"{len(option_chain)} option contracts."
    )

    # 2. Demonstrate aggregation from ticks
    agg = AggregationEngine()
    for symbol in SYMBOLS:
        sym_ticks = ticks[ticks["symbol"] == symbol]
        candles = agg.aggregate_ticks_df(sym_ticks, symbol)
        for tf, df in candles.items():
            logger.info(f"  {symbol} {tf} candles: {len(df)} rows")

    # 3. Build macro features (from minute bars)
    for symbol in SYMBOLS:
        sym_minutes = minute_bars[minute_bars["symbol"] == symbol].copy()
        sym_options = option_chain[option_chain["symbol"] == symbol]

        macro_df = compute_all_macro_indicators(sym_minutes, sym_options)
        logger.info(
            f"  {symbol} macro features: {len(macro_df)} rows, "
            f"columns: {[c for c in macro_df.columns if c not in ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]}"
        )

    # 4. Build micro features (from ticks)
    for symbol in SYMBOLS:
        sym_ticks = ticks[ticks["symbol"] == symbol].copy()
        micro_df = compute_micro_features(sym_ticks)
        logger.info(
            f"  {symbol} micro features: {len(micro_df)} rows, "
            f"columns: {[c for c in micro_df.columns if c not in ['timestamp', 'symbol']]}"
        )

    logger.info("=" * 60)
    logger.info("Mock pipeline complete. All layers functional.")
    logger.info("=" * 60)


def run_ingest():
    """
    Load historical data from TrueData into the database and build features.
    Requires: TimescaleDB running + TrueData credentials in .env
    """
    from database.db import init_db
    from data.truedata_adapter import TrueDataAdapter
    from data.aggregator import AggregationEngine
    from data.tick_collector import TickCollector
    from features.feature_engine import build_all_features

    logger.info("=" * 60)
    logger.info("MODE: INGEST – loading historical data from TrueData")
    logger.info("=" * 60)

    # 1. Initialize database
    init_db()

    # 2. Connect to TrueData
    td = TrueDataAdapter()
    td.connect()

    if not td.is_connected:
        logger.error(
            "Cannot connect to TrueData. "
            "Check credentials in .env or run 'python main.py mock' instead."
        )
        return

    agg = AggregationEngine()
    collector = TickCollector()

    for symbol in SYMBOLS:
        # 3a. Fetch 6 months of 1m bars → Macro Model training data
        minute_df = td.fetch_historical_minute_bars(symbol, days=180)
        if not minute_df.empty:
            agg.ingest_minute_bars(minute_df)

        # 3b. Fetch 5 days of tick data → Micro Model training data
        tick_df = td.fetch_historical_ticks(symbol, days=5)
        if not tick_df.empty:
            collector.ingest_historical_ticks(tick_df)
            agg.aggregate_from_db(symbol)

        # 4. Build features
        build_all_features(symbol)

    td.disconnect()
    logger.info("Historical data ingestion complete.")


def run_train():
    """
    Train ML models from data in the database.
    Requires: TimescaleDB running + data ingested via 'ingest' mode.

    NOTE: Only call this with real TrueData, never with mock data.
    """
    from features.feature_engine import build_macro_features, build_micro_features
    from models.train_model import train_all_models

    logger.info("=" * 60)
    logger.info("MODE: TRAIN - training ML models from DB data")
    logger.info("=" * 60)

    for symbol in SYMBOLS:
        logger.info(f"Building features for {symbol}...")
        macro_df = build_macro_features(symbol)
        micro_df = build_micro_features(symbol)

        if macro_df.empty:
            logger.warning(f"No macro features for {symbol}. Run 'ingest' first.")
            continue

        logger.info(f"Training models for {symbol}...")
        results = train_all_models(macro_df, micro_df)

        for model_type, metrics in results.items():
            logger.info(f"  {symbol} {model_type}: {metrics}")

    logger.info("Training complete.")


def run_backtest():
    """
    Run backtest on mock data through the full pipeline:
      Mock data -> Features -> Strategy signals -> Scoring -> SL/Target sim -> Metrics

    This demonstrates the entire system end-to-end without needing real data or DB.
    No ML models are trained on mock data - uses default 0.5 probability.
    """
    from data.mock_data import generate_mock_minute_bars
    from features.indicators import compute_all_macro_indicators
    from backtest.backtest_engine import BacktestEngine

    logger.info("=" * 60)
    logger.info("MODE: BACKTEST - running strategy backtest on mock data")
    logger.info("=" * 60)
    logger.info("NOTE: ML probability fixed at 0.5 (no model trained on mock data)")

    engine = BacktestEngine()

    for symbol in SYMBOLS:
        logger.info(f"\nBacktesting {symbol}...")

        # Generate mock minute bars
        minute_df = generate_mock_minute_bars(symbol, trading_days=125)

        # Compute features
        featured_df = compute_all_macro_indicators(minute_df)

        # Run backtest (no predictor = uses default 0.5 ML prob)
        result = engine.run(featured_df, symbol=symbol, predictor=None)

        logger.info(f"{symbol} backtest: {result.total_trades} trades, "
                     f"win_rate={result.win_rate:.1%}, PnL={result.gross_pnl:,.0f}")

    logger.info("Backtest complete.")


def run_live():
    """
    Live trading loop.
    Requires: TimescaleDB + TrueData/Kite + trained ML models.

    System loop (from docs):
      while market_open:
        fetch market data -> update indicators -> detect regime
        -> generate signals -> compute options flow -> run ML model
        -> rank trades -> execute trades

    Cycle time: 30-60 seconds
    """
    from models.predict import Predictor
    from strategy.regime_detector import RegimeDetector, get_strategies_for_regime
    from strategy.signal_generator import generate_signals
    from strategy.options_flow_detector import OptionsFlowDetector
    from strategy.trade_scorer import TradeScorer
    from risk.risk_manager import RiskManager
    from execution.order_manager import OrderManager
    from execution.broker_adapter import BrokerAdapter
    from utils.helpers import is_market_open

    logger.info("=" * 60)
    logger.info("MODE: LIVE TRADING")
    logger.info("=" * 60)

    # Initialize all components
    predictor = Predictor()
    predictor.load()

    if not predictor.is_loaded:
        logger.error(
            "No ML models found. Train models first with: python main.py train"
        )
        return

    regime_detector = RegimeDetector()
    flow_detector = OptionsFlowDetector()
    scorer = TradeScorer()
    risk_mgr = RiskManager()
    order_mgr = OrderManager(broker_adapter=BrokerAdapter())

    logger.info("All components initialized. Entering trading loop...")

    while True:
        if not is_market_open():
            logger.info("Market closed. Waiting...")
            time.sleep(60)
            continue

        if not risk_mgr.can_trade:
            logger.info(f"Risk limit reached. {risk_mgr.get_daily_summary()}")
            time.sleep(SCAN_INTERVAL_SECONDS)
            continue

        try:
            from features.feature_engine import build_macro_features
            from database.db import read_sql

            all_signals = []
            ml_probs = {}
            flow_scores = {}

            for symbol in SYMBOLS:
                # 1. Build latest features
                macro_df = build_macro_features(symbol, limit=100)
                if macro_df.empty:
                    continue

                latest = macro_df.iloc[-1].to_dict()

                # 2. Detect market regime (from 5m candles)
                fivemin_df = read_sql(
                    "SELECT * FROM five_minute_candles "
                    "WHERE symbol = :sym ORDER BY timestamp DESC LIMIT 100",
                    {"sym": symbol},
                )
                regime = regime_detector.detect(fivemin_df)
                active_strats = get_strategies_for_regime(regime)

                # 3. Generate signals
                signals = generate_signals(latest, symbol, active_strats)

                # 4. ML prediction
                ml_result = predictor.predict_combined(latest)
                ml_probs[symbol] = ml_result.get("combined_ml_prob", 0.5)

                # 5. Options flow
                oc_df = read_sql(
                    "SELECT * FROM option_chain "
                    "WHERE symbol = :sym ORDER BY timestamp DESC LIMIT 100",
                    {"sym": symbol},
                )
                if not oc_df.empty:
                    flow = flow_detector.analyze(oc_df, latest.get("close", 0))
                    flow_scores[symbol] = flow.score

                all_signals.extend(signals)

            # 6. Rank and select top trades
            top_trades = scorer.rank_trades(
                all_signals, ml_probs, flow_scores, regime.value
            )

            # 7. Execute approved trades
            for scored_trade in top_trades:
                atr = latest.get("atr", 0)
                risk_decision = risk_mgr.validate_trade(
                    symbol=scored_trade.symbol,
                    entry_price=scored_trade.entry_price,
                    atr=atr,
                    direction=scored_trade.direction,
                )

                if risk_decision.approved:
                    order = order_mgr.execute_trade(
                        symbol=scored_trade.symbol,
                        direction=scored_trade.direction,
                        quantity=risk_decision.quantity,
                        entry_price=scored_trade.entry_price,
                        stop_loss=risk_decision.stop_loss,
                        target=risk_decision.target,
                        strategy=scored_trade.strategy,
                        scores={
                            "ml": scored_trade.ml_probability,
                            "flow": scored_trade.flow_score,
                            "tech": scored_trade.technical_strength,
                            "final": scored_trade.final_score,
                        },
                    )
                    risk_mgr.register_entry(
                        scored_trade.symbol,
                        scored_trade.entry_price,
                        risk_decision.quantity,
                        scored_trade.direction,
                    )
                else:
                    logger.info(
                        f"Trade rejected: {scored_trade.symbol} - "
                        f"{risk_decision.rejection_reason}"
                    )

        except Exception as e:
            logger.error(f"Trading loop error: {e}", exc_info=True)

        logger.info(f"Cycle complete. Sleeping {SCAN_INTERVAL_SECONDS}s...")
        time.sleep(SCAN_INTERVAL_SECONDS)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "mock"

    modes = {
        "mock": run_mock,
        "ingest": run_ingest,
        "train": run_train,
        "backtest": run_backtest,
        "live": run_live,
    }

    if mode in modes:
        modes[mode]()
    else:
        logger.error(f"Unknown mode: {mode}. Use: {' | '.join(modes.keys())}")
        sys.exit(1)


if __name__ == "__main__":
    main()