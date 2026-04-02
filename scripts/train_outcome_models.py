"""
Outcome-Based Strategy Model Training
──────────────────────────────────────
Trains strategy-specific XGBoost models using ACTUAL trade outcomes
from backtest CSVs, instead of synthetic forward-return labels.

For each trade in the CSVs:
  - Fetch the NIFTY-I candle at entry_time from minute_candles
  - Compute macro indicators → 50-feature vector (same as live scoring)
  - Label: 1 = profitable (TRAILING_SL / TIMEOUT / RL_EXIT / TARGET / pnl>0)
           0 = loss (SL hit / pnl<=0)

This gives the models real discrimination power: "given exactly this market
state, does this strategy's signal lead to a win or a loss?"

Usage:
  python scripts/train_outcome_models.py
  python scripts/train_outcome_models.py --min-samples 20   # lower threshold
  python scripts/train_outcome_models.py --evaluate         # show stats only
"""

import os, sys, argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import joblib

from database.db import read_sql
from features.indicators import compute_all_macro_indicators
from config.settings import FEATURE_COLUMNS_MACRO, MODEL_DIR
from utils.logger import get_logger

logger = get_logger("train_outcome_models")

STRATEGY_MODEL_DIR = MODEL_DIR / "strategy"
STRATEGY_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# CSVs to load — all tick-replay style (have entry_time, pnl, result, strategy)
BACKTEST_CSVS = [
    Path("backtest_results/trades_high_risk.csv"),
    Path("backtest_results/trades_medium_risk.csv"),
    Path("backtest_results/trades_low_risk.csv"),
]

WIN_RESULTS = {"TRAILING_SL", "TIMEOUT", "RL_EXIT", "EOD_CLOSE", "TARGET", "WIN"}


def _parse_entry_time(ts_str: str) -> pd.Timestamp:
    """
    Parse entry_time from backtest CSV.

    Both the CSV and the DB store timestamps as IST but labelled as +00:00
    (timezone-naive IST with wrong UTC marker). No offset conversion needed —
    just strip tzinfo and compare naively.
    E.g. '2026-03-10 12:21:00+00:00' = IST 12:21 PM in both CSV and DB.
    """
    ts = pd.Timestamp(ts_str)
    if ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)
    return ts


def load_trades() -> pd.DataFrame:
    """Load and deduplicate all backtest trade records."""
    dfs = []
    for csv_path in BACKTEST_CSVS:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["source_file"] = csv_path.name
            dfs.append(df)
            logger.info(f"  Loaded {len(df)} trades from {csv_path.name}")
        else:
            logger.warning(f"  Missing: {csv_path}")

    if not dfs:
        raise FileNotFoundError("No backtest CSVs found in backtest_results/")

    all_trades = pd.concat(dfs, ignore_index=True)
    before = len(all_trades)
    all_trades = all_trades.drop_duplicates(
        subset=["entry_time", "symbol", "direction", "entry_premium"]
    )
    logger.info(f"  Deduplicated: {before} → {len(all_trades)} trades")
    return all_trades


def build_label(row: pd.Series) -> int:
    """1 = win, 0 = loss."""
    if row.get("result") in WIN_RESULTS:
        return 1
    if pd.notna(row.get("pnl")) and float(row["pnl"]) > 0:
        return 1
    return 0


def fetch_features_at_entry(entry_times: list) -> pd.DataFrame:
    """
    Fetch a window of NIFTY-I candles ending at each entry time and
    compute macro indicators. Returns a DataFrame with a '_entry_time' column.

    Both CSV and DB timestamps are IST stored with +00:00 label — no conversion.
    """
    if not entry_times:
        return pd.DataFrame()

    earliest = min(entry_times)
    warm_start = earliest - timedelta(hours=6)

    logger.info(f"  Fetching candles from {warm_start.strftime('%Y-%m-%d %H:%M')}...")
    candles = read_sql(
        """
        SELECT * FROM minute_candles
        WHERE symbol = :sym
          AND timestamp >= :from_ts
          AND timestamp <= :to_ts
        ORDER BY timestamp
        """,
        {
            "sym": "NIFTY-I",
            "from_ts": warm_start,
            "to_ts": max(entry_times) + timedelta(minutes=2),
        },
    )
    if candles.empty:
        logger.error("No candles fetched from DB!")
        return pd.DataFrame()

    logger.info(f"  Loaded {len(candles)} candles, computing indicators...")
    featured = compute_all_macro_indicators(candles)
    # Normalize timestamp to tz-naive for comparison
    featured["timestamp"] = pd.to_datetime(featured["timestamp"]).dt.tz_localize(None)
    featured = featured.set_index("timestamp").sort_index()

    # For each entry time, find closest prior candle (within 2 min)
    result_rows = []
    for ts in entry_times:
        ts_pd = pd.Timestamp(ts)
        available = featured.index[featured.index <= ts_pd]
        if len(available) == 0:
            logger.warning(f"  No candle found for {ts_pd}")
            continue
        closest = available[-1]
        if (ts_pd - closest).total_seconds() > 120:
            logger.warning(f"  Gap too large for {ts_pd}: nearest={closest}")
            continue
        row = featured.loc[closest].copy()
        row["_entry_time"] = ts_pd
        result_rows.append(row)

    if not result_rows:
        return pd.DataFrame()

    feature_df = pd.DataFrame(result_rows)
    logger.info(f"  Matched {len(feature_df)} trade entries to candle features")
    return feature_df


def train_strategy_outcome_model(
    strategy_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    min_samples: int = 15,
) -> dict | None:
    """Train a single strategy model on outcome-labeled features."""
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import roc_auc_score

    if len(X) < min_samples:
        logger.warning(
            f"  {strategy_name}: only {len(X)} samples (need {min_samples}), skipping"
        )
        return None

    pos = int(y.sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        logger.warning(f"  {strategy_name}: all samples same class ({pos}W/{neg}L), skipping")
        return None

    spw = round(neg / pos, 2)
    logger.info(f"  {strategy_name}: {len(X)} samples | {pos}W / {neg}L | scale_pos_weight={spw}")

    # Conservative hyperparams for small datasets
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=2,           # shallow — prevent overfitting on small dataset
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.5,  # use half features per tree
        min_child_weight=3,    # require 3 samples per leaf
        reg_alpha=0.5,         # L1 regularization
        reg_lambda=2.0,        # L2 regularization
        scale_pos_weight=spw,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
    )

    # Cross-val if enough samples
    if len(X) >= 30:
        n_splits = min(3, pos)  # can't have more splits than positives
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=False)
            try:
                aucs = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
                logger.info(f"    CV AUC: {aucs.round(3)} | avg={aucs.mean():.3f}")
            except Exception as e:
                logger.warning(f"    CV failed: {e}")

    # Train final model on all data
    model.fit(X, y)

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=X.columns)
    top5 = imp.nlargest(5)
    logger.info(f"    Top features: {dict(top5.round(3))}")

    # Save
    path = STRATEGY_MODEL_DIR / f"{strategy_name}_model.pkl"
    # Back up existing model
    if path.exists():
        from datetime import datetime
        import shutil
        backup_dir = Path("models/saved/backups") / datetime.now().strftime("%Y%m%d")
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_dir / f"{strategy_name}_pre_outcome_{datetime.now().strftime('%H%M%S')}.pkl")

    joblib.dump({
        "model": model,
        "features": list(X.columns),
        "strategy": strategy_name,
        "n_samples": len(X),
        "n_wins": pos,
        "n_losses": neg,
        "trained_at": datetime.now().isoformat(),
        "training_method": "outcome_labels",
    }, path)
    logger.info(f"    Saved → {path}")
    return {"n_samples": len(X), "wins": pos, "losses": neg}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-samples", type=int, default=15)
    parser.add_argument("--evaluate", action="store_true", help="Show stats only, don't train")
    args = parser.parse_args()

    print("=" * 60)
    print("  OUTCOME-BASED STRATEGY MODEL TRAINING")
    print("=" * 60)

    # 1. Load trades
    print("\n  Loading backtest trade records...")
    trades = load_trades()
    trades["label"] = trades.apply(build_label, axis=1)

    print(f"\n  Total trades: {len(trades)}")
    print(f"  Win-labeled:  {trades['label'].sum()}")
    print(f"  Loss-labeled: {(trades['label'] == 0).sum()}")
    print(f"  Strategies:   {trades['strategy'].value_counts().to_dict()}")

    if args.evaluate:
        print("\n  [Evaluate-only mode — no training]")
        for strat, grp in trades.groupby("strategy"):
            pos = grp["label"].sum()
            neg = (grp["label"] == 0).sum()
            print(f"    {strat}: {len(grp)} trades, {pos}W/{neg}L ({pos/len(grp)*100:.0f}% win rate)")
        return

    # 2. Parse entry times
    print("\n  Parsing entry timestamps...")
    trades["entry_time_parsed"] = trades["entry_time"].apply(_parse_entry_time)
    entry_times = sorted(trades["entry_time_parsed"].tolist())

    # 3. Fetch features from DB
    print("  Fetching features from DB...")
    feature_df = fetch_features_at_entry(entry_times)

    if feature_df.empty:
        print("  ERROR: No features fetched. Aborting.")
        return

    # 4. Join features to trade labels by matching _entry_time → entry_time_parsed
    # Use integer index alignment (not timestamp join) to avoid duplicates
    trades_reset = trades.reset_index(drop=True)
    feature_df = feature_df.reset_index(drop=True)

    # Match each feature row to its trade by nearest timestamp
    trade_meta = []
    for _, feat_row in feature_df.iterrows():
        feat_ts = feat_row["_entry_time"]
        # Find the trade whose entry_time is closest to this feature ts
        diffs = (trades_reset["entry_time_parsed"] - feat_ts).abs()
        best_idx = diffs.idxmin()
        if diffs[best_idx].total_seconds() <= 120:
            trade_meta.append({
                "strategy": trades_reset.at[best_idx, "strategy"],
                "label": trades_reset.at[best_idx, "label"],
                "direction": trades_reset.at[best_idx, "direction"],
                "pnl": trades_reset.at[best_idx, "pnl"],
                "result": trades_reset.at[best_idx, "result"],
            })
        else:
            trade_meta.append(None)

    feature_df["_meta"] = trade_meta
    feature_df = feature_df[feature_df["_meta"].notna()].copy()
    for col in ["strategy", "label", "direction", "pnl", "result"]:
        feature_df[col] = feature_df["_meta"].apply(lambda m: m[col])
    feature_df = feature_df.drop(columns=["_meta", "_entry_time"])

    joined = feature_df
    print(f"  Joined {len(joined)} trade-feature pairs")

    # 5. Train per-strategy models
    available_features = [c for c in FEATURE_COLUMNS_MACRO if c in joined.columns]
    # Drop all-NaN columns
    available_features = [
        c for c in available_features
        if joined[c].notna().any() and not np.isinf(joined[c].replace([np.inf, -np.inf], np.nan)).all()
    ]
    joined[available_features] = (
        joined[available_features]
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .bfill()
    )

    print(f"\n  Feature columns available: {len(available_features)}")
    print("\n" + "=" * 60)

    results = {}
    for strategy_name, grp in joined.groupby("strategy"):
        print(f"\n  Strategy: {strategy_name}")
        X = grp[available_features].dropna()
        y = grp.loc[X.index, "label"]
        result = train_strategy_outcome_model(
            strategy_name, X, y, min_samples=args.min_samples
        )
        if result:
            results[strategy_name] = result

    print("\n" + "=" * 60)
    print("  OUTCOME TRAINING COMPLETE")
    print("=" * 60)
    for name, r in results.items():
        print(f"  {name}: {r['n_samples']} samples, {r['wins']}W/{r['losses']}L")

    if not results:
        print("\n  WARNING: No models trained — insufficient samples.")
        print("  Run more backtests or lower --min-samples threshold.")


if __name__ == "__main__":
    main()
