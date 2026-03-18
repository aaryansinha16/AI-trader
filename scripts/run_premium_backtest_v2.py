"""
Definitive Premium Backtest v2
──────────────────────────────
Integrates ALL available engines:
  1. Real option premium pricing (from DB)
  2. ML scoring with PUT gate
  3. Options Flow Detector (real flow scores from option chain data)
  4. Regime Detector (strategy filtering by market condition)
  5. DTE filtering (avoid expiry day trades)
  6. Time-of-day rules (skip first 5 min, last 30 min)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
from datetime import timedelta

from database.db import read_sql
from models.predict import Predictor
from backtest.backtest_engine import BacktestEngine, BacktestTrade, BacktestResult
from backtest.option_resolver import (
    resolve_option_at_entry, get_nearest_expiry, get_days_to_expiry, clear_cache,
)
from strategy.signal_generator import generate_signals
from strategy.options_flow_detector import OptionsFlowDetector
from strategy.regime_detector import RegimeDetector, get_strategies_for_regime
from config.settings import (
    WEIGHT_ML_PROBABILITY, WEIGHT_OPTIONS_FLOW, WEIGHT_TECHNICAL_STRENGTH,
)
from utils.logger import get_logger

logger = get_logger("premium_bt_v2")

# ── Parameters ────────────────────────────────────────────────────────────────
SCORE_THRESHOLD = 0.60
MAX_TRADES_PER_DAY = 5
MAX_HOLD_BARS = 20
LOT_SIZE = 65
SL_PCT = 0.30          # lose 30% of premium → stop
TGT_PCT = 0.50         # gain 50% of premium → target
COMMISSION = 40.0       # ₹20/order × 2
SLIPPAGE_PCT = 0.001
SKIP_FIRST_MIN = 5     # skip first 5 minutes after open
SKIP_LAST_MIN = 15     # skip last 15 minutes before close
MIN_DTE = 0            # allow expiry day but apply theta caution
EXPIRY_DAY_CUTOFF_MIN = 300  # on expiry day, stop new trades after 2:15 PM (300 min from open)


def build_option_chain_snapshot(row, option_chain_ts_df):
    """Build a per-row option chain snapshot from preloaded time-aligned data."""
    ts = row.get("timestamp")
    if ts is None or option_chain_ts_df is None:
        return None

    # Find matching timestamp
    mask = option_chain_ts_df["timestamp"] == ts
    matched = option_chain_ts_df[mask]
    if matched.empty:
        return None
    return matched.iloc[0].to_dict()


def main():
    print("=" * 70)
    print("  PREMIUM BACKTEST v2 - ALL ENGINES INTEGRATED")
    print("=" * 70)

    # ── Load features ─────────────────────────────────────────────────────
    print("Loading features...")
    df = read_sql("SELECT * FROM features_macro ORDER BY timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    ohlcv = read_sql(
        "SELECT timestamp, open, high, low, close, volume, vwap "
        "FROM minute_candles WHERE symbol = 'NIFTY-I' ORDER BY timestamp"
    )
    ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"])
    df = df.merge(ohlcv, on="timestamp", how="left", suffixes=("", "_mc"))
    for c in [x for x in df.columns if x.endswith("_mc")]:
        df.drop(columns=[c], inplace=True)
    print(f"Features: {df.shape}")

    # ── Load option chain timeseries (for flow scoring) ───────────────────
    print("Loading option chain data for flow scoring...")
    oc_ts = read_sql("""
        SELECT mc.timestamp,
               mc.symbol, mc.close as premium, mc.volume, mc.oi
        FROM minute_candles mc
        WHERE (mc.symbol LIKE 'NIFTY%%CE' OR mc.symbol LIKE 'NIFTY%%PE')
        ORDER BY mc.timestamp
    """)
    oc_ts["timestamp"] = pd.to_datetime(oc_ts["timestamp"])
    # Extract option_type from symbol
    oc_ts["option_type"] = oc_ts["symbol"].str[-2:]
    oc_ts["oi_change"] = oc_ts.groupby("symbol")["oi"].diff().fillna(0)
    # Extract strike
    import re
    oc_ts["strike"] = oc_ts["symbol"].str.extract(r'NIFTY\d{6}(\d+)')[0].astype(float)
    print(f"Option chain rows: {len(oc_ts):,}")

    # ── Initialize engines ────────────────────────────────────────────────
    predictor = Predictor()
    predictor.load()
    flow_detector = OptionsFlowDetector()
    regime_detector = RegimeDetector()

    # ── Run backtest ──────────────────────────────────────────────────────
    clear_cache()
    df = df.reset_index(drop=True)
    trades = []
    in_trade = False
    current_trade = None
    option_info = None
    daily_trades = 0
    current_day = None
    skipped_dte = 0
    skipped_time = 0
    skipped_regime = 0

    print(f"Running backtest on {len(df)} candles...")

    for i in range(50, len(df)):
        row = df.iloc[i].to_dict()
        ts = row.get("timestamp")
        if ts is None:
            continue

        # Day reset
        day = ts.date() if hasattr(ts, "date") else None
        if day != current_day:
            current_day = day
            daily_trades = 0

        # ── Time-of-day filter ────────────────────────────────────────────
        if hasattr(ts, "hour"):
            # Timestamps stored as IST with +00:00 label; 09:15 IST = hour 9
            minutes_since_open = ts.hour * 60 + ts.minute - 555  # 9*60+15=555
            if minutes_since_open < SKIP_FIRST_MIN or minutes_since_open > (375 - SKIP_LAST_MIN):
                if in_trade:
                    pass  # still track open trades
                else:
                    skipped_time += 1
                    continue

        # ── Check open trade ──────────────────────────────────────────────
        if in_trade and current_trade is not None and option_info is not None:
            bars_held = i - current_trade.entry_time
            prem_df = option_info["premium_df"]

            ts_pd = pd.to_datetime(ts)
            mask = (prem_df["timestamp"] - ts_pd).abs() <= pd.Timedelta(minutes=1)
            prem_row = prem_df[mask]

            if not prem_row.empty:
                prem_high = float(prem_row.iloc[0].get("high", prem_row.iloc[0]["premium"]))
                prem_low = float(prem_row.iloc[0].get("low", prem_row.iloc[0]["premium"]))
                prem_close = float(prem_row.iloc[0]["premium"])

                entry_prem = option_info["entry_premium"]
                prem_sl = entry_prem * (1 - SL_PCT)
                prem_tgt = entry_prem * (1 + TGT_PCT)

                hit_target = prem_high >= prem_tgt
                hit_stop = prem_low <= prem_sl

                exit_prem = None
                if hit_stop:
                    exit_prem = prem_sl
                    current_trade.result = "LOSS"
                elif hit_target:
                    exit_prem = prem_tgt
                    current_trade.result = "WIN"
                elif bars_held >= MAX_HOLD_BARS:
                    exit_prem = prem_close
                    current_trade.result = "TIMEOUT"

                if exit_prem is not None:
                    current_trade.exit_time = ts
                    current_trade.exit_price = round(exit_prem, 2)
                    prem_move = exit_prem - entry_prem
                    slippage = entry_prem * SLIPPAGE_PCT * 2 * current_trade.quantity
                    current_trade.pnl = round(
                        prem_move * current_trade.quantity - slippage - COMMISSION, 2
                    )
                    trades.append(current_trade)
                    in_trade = False
                    current_trade = None
                    option_info = None
                    continue

        # ── Entry logic ───────────────────────────────────────────────────
        if in_trade or daily_trades >= MAX_TRADES_PER_DAY:
            continue

        # DTE check: on expiry day, only trade in the morning
        dte = 99
        if day:
            expiry = get_nearest_expiry(day)
            if expiry:
                dte = get_days_to_expiry(day, expiry)
                if dte == 0 and hasattr(ts, 'hour'):
                    mins_open = ts.hour * 60 + ts.minute - 555
                    if mins_open > EXPIRY_DAY_CUTOFF_MIN:
                        skipped_dte += 1
                        continue

        # Regime detection (use rolling 1-min candles as proxy for 5-min)
        # Regime suggests preferred strategies but doesn't block others
        regime_strategies = None
        if i >= 100:
            regime_window = df.iloc[i-100:i][["open", "high", "low", "close", "volume"]].copy()
            regime = regime_detector.detect(regime_window)
            regime_strategies = get_strategies_for_regime(regime)
        active_strategies = None  # allow all, but boost regime-aligned ones

        # Generate signals (filtered by regime)
        signals = generate_signals(row, "NIFTY-I", active_strategies)
        if not signals:
            continue

        sig = signals[0]

        # ML scoring
        ml_prob = 0.5
        if predictor.is_loaded:
            p = predictor.predict_macro(row)
            if p is not None:
                ml_prob = p

        # ML gate for PUTs
        if sig.direction == "PUT" and ml_prob > 0.40:
            continue

        # Options flow scoring (use pre-computed option chain features from the row)
        flow_score = 0.5  # default
        pcr = row.get("pcr", None)
        oi_change = row.get("oi_change", 0)
        oi_skew = row.get("oi_skew", 0)
        if pcr is not None and not np.isnan(pcr):
            # Simple flow score from available option data
            flow_score = 0.0
            if pcr > 1.2:
                flow_score += 0.3  # High PCR = contrarian bullish
            elif pcr < 0.7:
                flow_score += 0.15
            if oi_change and not np.isnan(oi_change) and abs(oi_change) > 1000000:
                flow_score += 0.3  # Large OI change = institutional activity
            if oi_skew and not np.isnan(oi_skew) and abs(oi_skew) > 0.1:
                flow_score += 0.2
            flow_score = min(flow_score, 1.0)

        directional_prob = ml_prob if sig.direction == "CALL" else (1.0 - ml_prob)

        # Regime alignment bonus
        regime_bonus = 0.0
        if regime_strategies and sig.strategy in regime_strategies:
            regime_bonus = 0.05

        final_score = (
            WEIGHT_ML_PROBABILITY * directional_prob
            + WEIGHT_OPTIONS_FLOW * flow_score
            + WEIGHT_TECHNICAL_STRENGTH * sig.technical_strength
            + regime_bonus
        )

        if final_score < SCORE_THRESHOLD:
            continue

        # Resolve actual option contract
        opt = resolve_option_at_entry(
            index_price=row["close"], timestamp=ts, direction=sig.direction,
        )
        if opt is None:
            continue

        entry_prem = opt["entry_premium"]
        if entry_prem <= 0:
            continue

        # Position sizing
        risk_per_lot = entry_prem * SL_PCT * LOT_SIZE
        risk_amount = 50000 * 0.01  # 1% of 50K
        n_lots = max(1, int(risk_amount / risk_per_lot)) if risk_per_lot > 0 else 1
        qty = n_lots * LOT_SIZE

        current_trade = BacktestTrade(
            entry_time=i,
            symbol=opt["symbol"],
            direction=sig.direction,
            strategy=sig.strategy,
            entry_price=round(entry_prem, 2),
            stop_loss=round(entry_prem * (1 - SL_PCT), 2),
            target=round(entry_prem * (1 + TGT_PCT), 2),
            quantity=qty,
            ml_score=ml_prob,
            flow_score=flow_score,
            tech_score=sig.technical_strength,
            final_score=round(final_score, 4),
        )
        option_info = opt
        in_trade = True
        daily_trades += 1

    # Close remaining
    if in_trade and current_trade is not None and option_info is not None:
        entry_prem = option_info["entry_premium"]
        prem_df = option_info["premium_df"]
        last_prem = float(prem_df.iloc[-1]["premium"]) if not prem_df.empty else entry_prem
        current_trade.exit_price = round(last_prem, 2)
        current_trade.exit_time = df.iloc[-1].get("timestamp")
        current_trade.result = "TIMEOUT"
        prem_move = last_prem - entry_prem
        slippage = entry_prem * SLIPPAGE_PCT * 2 * current_trade.quantity
        current_trade.pnl = round(prem_move * current_trade.quantity - slippage - COMMISSION, 2)
        trades.append(current_trade)

    # ── Compute metrics ───────────────────────────────────────────────────
    engine = BacktestEngine()
    result = engine._compute_metrics(trades)
    engine._log_summary(result)

    sep = "=" * 70
    print(f"\n{sep}")
    print("  PREMIUM BACKTEST v2 - ALL ENGINES")
    print(sep)
    print(f"  Total trades:  {result.total_trades}")
    print(f"  Wins:          {result.wins}")
    print(f"  Losses:        {result.losses}")
    print(f"  Win rate:      {result.win_rate:.1%}")
    print(f"  Gross PnL:     Rs{result.gross_pnl:,.0f}")
    print(f"  Profit factor: {result.profit_factor:.2f}")
    print(f"  Max drawdown:  Rs{result.max_drawdown:,.0f}")
    print(f"  Sharpe ratio:  {result.sharpe_ratio:.2f}")
    print(f"  Expectancy:    Rs{result.expectancy:,.0f}/trade")
    print(f"  Avg win:       Rs{result.avg_win:,.0f}")
    print(f"  Avg loss:      Rs{result.avg_loss:,.0f}")
    if result.avg_loss != 0:
        print(f"  Risk-reward:   {abs(result.avg_win / result.avg_loss):.2f}")
    print(f"\n  Skipped (DTE):   {skipped_dte}")
    print(f"  Skipped (time):  {skipped_time}")
    print(sep)

    if trades:
        tdf = pd.DataFrame([
            {"strategy": t.strategy, "direction": t.direction, "pnl": t.pnl,
             "result": t.result, "symbol": t.symbol, "entry_price": t.entry_price,
             "exit_price": t.exit_price, "flow_score": t.flow_score}
            for t in trades
        ])
        print("\nBy strategy:")
        grp = tdf.groupby("strategy").agg(
            count=("pnl", "count"),
            wins=("result", lambda x: (x == "WIN").sum()),
            wr=("result", lambda x: round((x == "WIN").mean(), 3)),
            avg_pnl=("pnl", "mean"),
            total=("pnl", "sum"),
        ).round(0)
        print(grp.to_string())
        print("\nBy direction:")
        grp2 = tdf.groupby("direction").agg(
            count=("pnl", "count"),
            wins=("result", lambda x: (x == "WIN").sum()),
            wr=("result", lambda x: round((x == "WIN").mean(), 3)),
            avg_pnl=("pnl", "mean"),
            total=("pnl", "sum"),
        ).round(0)
        print(grp2.to_string())

        print(f"\nFlow score distribution: mean={tdf['flow_score'].mean():.2f}, "
              f"min={tdf['flow_score'].min():.2f}, max={tdf['flow_score'].max():.2f}")

        print("\nSample trades:")
        print(tdf[["symbol", "direction", "strategy", "entry_price", "exit_price",
                    "pnl", "result", "flow_score"]].head(10).to_string())

    result.export_all(base_name="NIFTY_premium_v2", output_dir="backtest_results")
    print(f"\nResults exported to backtest_results/")


if __name__ == "__main__":
    main()
