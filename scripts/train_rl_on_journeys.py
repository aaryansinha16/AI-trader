"""
RL Exit Agent — Retrain on All Journey Data
─────────────────────────────────────────────
Retrains the Q-learning exit agent using the saved journey JSON files
(premium trajectories) from all completed backtests.

This decouples the RL agent from specific entry timing: instead of learning
only from trajectories the agent happened to see during a single backtest run,
it trains on every recorded trade from every risk profile, across all replayed
days. The state features (pnl_pct, bars_held_norm, momentum, etc.) are all
trade-relative — there's no dependency on which exact bar entry happened.

Journey files: backtest_results/journeys_{risk}_risk.json
Trade files:   backtest_results/trades_{risk}_risk.csv  (entry_premium, sl, target)

Usage:
  python scripts/train_rl_on_journeys.py             # default 30 epochs
  python scripts/train_rl_on_journeys.py --epochs 50
  python scripts/train_rl_on_journeys.py --evaluate  # show policy summary only
  python scripts/train_rl_on_journeys.py --fresh      # reset Q-table, train from scratch
"""

import os, sys, argparse, json, shutil
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd

from models.rl_exit_agent import RLExitAgent
from utils.logger import get_logger

logger = get_logger("train_rl_journeys")

BACKTEST_DIR = Path("backtest_results")
MODEL_DIR = Path("models/saved")
BACKUP_DIR = MODEL_DIR / "backups"

RISK_LEVELS = ["high", "medium", "low"]


def load_journey_data() -> list[dict]:
    """
    Load all journey + trade pairs from every risk level.
    Returns list of dicts:
      {
        'risk': str,
        'trade_idx': int,
        'entry_premium': float,
        'sl': float,
        'target': float,
        'lot_size': int,
        'result': str,
        'pnl': float,
        'journey': list[dict],  # [{ts, premium, sl, nifty_price, bars_held}, ...]
      }
    """
    all_data = []

    for risk in RISK_LEVELS:
        journey_path = BACKTEST_DIR / f"journeys_{risk}_risk.json"
        trades_path = BACKTEST_DIR / f"trades_{risk}_risk.csv"

        if not journey_path.exists():
            logger.warning(f"  Missing journey file: {journey_path}")
            continue
        if not trades_path.exists():
            logger.warning(f"  Missing trades file: {trades_path}")
            continue

        with open(journey_path) as f:
            journeys = json.load(f)  # {str(idx): [list of bar dicts]}

        trades_df = pd.read_csv(trades_path)

        loaded = 0
        for idx_str, journey_bars in journeys.items():
            idx = int(idx_str)
            if idx >= len(trades_df):
                continue
            if not journey_bars:
                continue

            trade = trades_df.iloc[idx]
            entry_premium = float(trade.get("entry_premium", 0))
            sl = float(trade.get("sl", 0))
            target = float(trade.get("target", 0))
            lot_size = int(trade.get("lot_size", 65))
            result = str(trade.get("result", ""))
            pnl = float(trade.get("pnl", 0))

            if entry_premium <= 0 or sl <= 0 or target <= 0:
                continue
            if len(journey_bars) < 2:
                continue

            # Extract premium trajectory from journey
            # Journey bars have either 'premium' (backtest) or 'option_price' (live)
            premiums = []
            for bar in journey_bars:
                p = bar.get("premium") or bar.get("option_price")
                if p is not None and float(p) > 0:
                    premiums.append(float(p))

            if len(premiums) < 2:
                continue

            all_data.append({
                "risk": risk,
                "trade_idx": idx,
                "entry_premium": entry_premium,
                "sl": sl,
                "target": target,
                "lot_size": lot_size,
                "result": result,
                "pnl": pnl,
                "premiums": premiums,
            })
            loaded += 1

        logger.info(f"  {risk} risk: loaded {loaded}/{len(journeys)} journeys")

    return all_data


def evaluate_agent(agent: RLExitAgent, trades: list[dict]):
    """Simulate agent on all journeys without updating Q-table. Reports stats."""
    exits = {"HOLD_till_end": 0, "EXIT": 0, "TIGHTEN": 0}
    total_pnl_pct = 0.0

    for t in trades:
        ep = t["entry_premium"]
        sl = t["sl"]
        target = t["target"]
        premiums = t["premiums"]
        max_bars = len(premiums)

        peak = ep
        trailing_active = False
        trailing_sl = sl
        history = [ep]

        final_pnl_pct = (premiums[-1] - ep) / ep
        agent_action = "HOLD_till_end"

        from models.rl_exit_agent import compute_state
        for bar_idx, prem in enumerate(premiums):
            history.append(prem)
            peak = max(peak, prem)

            gain_pct = (peak - ep) / ep
            if gain_pct >= 0.15 and not trailing_active:
                trailing_active = True
                trailing_sl = ep

            eff_sl = max(sl, trailing_sl) if trailing_active else sl

            state = compute_state(
                entry_premium=ep, current_premium=prem,
                bars_held=bar_idx, max_hold_bars=max_bars,
                sl=eff_sl, target=target,
                trailing_active=trailing_active,
                peak_premium=peak, premium_history=history,
            )

            action = agent.decide(state, explore=False)
            if action == "EXIT":
                final_pnl_pct = (prem - ep) / ep
                agent_action = "EXIT"
                break
            elif action == "TIGHTEN":
                agent_action = "TIGHTEN"
                # keep going (tighten adjusts SL but doesn't exit)

        exits[agent_action] += 1
        total_pnl_pct += final_pnl_pct

    n = len(trades)
    print(f"\n    Agent policy on {n} trades:")
    print(f"      EXIT early: {exits['EXIT']} ({exits['EXIT']/n*100:.0f}%)")
    print(f"      TIGHTEN:    {exits['TIGHTEN']} ({exits['TIGHTEN']/n*100:.0f}%)")
    print(f"      Hold→end:   {exits['HOLD_till_end']} ({exits['HOLD_till_end']/n*100:.0f}%)")
    print(f"      Avg P&L%%:  {total_pnl_pct/n*100:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training passes over all journeys")
    parser.add_argument("--evaluate", action="store_true",
                        help="Show policy stats only, don't train")
    parser.add_argument("--fresh", action="store_true",
                        help="Reset Q-table and train from scratch")
    parser.add_argument("--epsilon", type=float, default=0.10,
                        help="Exploration rate during training (default 0.10)")
    args = parser.parse_args()

    print("=" * 60)
    print("  RL EXIT AGENT — JOURNEY-BASED RETRAINING")
    print("=" * 60)

    # Load journey data
    print("\n  Loading journey data...")
    trades = load_journey_data()
    if not trades:
        print("  ERROR: No journey data found. Run backtest first.")
        return

    total_bars = sum(len(t["premiums"]) for t in trades)
    print(f"\n  Loaded {len(trades)} trade journeys")
    print(f"  Total bars:   {total_bars}")
    print(f"  Avg length:   {total_bars/len(trades):.1f} bars/trade")

    # Per-risk breakdown
    for risk in RISK_LEVELS:
        subset = [t for t in trades if t["risk"] == risk]
        if subset:
            wins = sum(1 for t in subset if t["pnl"] > 0)
            print(f"  {risk.capitalize():6s}: {len(subset)} trades, {wins}W/{len(subset)-wins}L")

    # Initialize agent
    agent = RLExitAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=args.epsilon,
    )

    if args.fresh:
        print("\n  [Fresh start — Q-table reset]")
    else:
        agent.load()
        if agent.is_loaded:
            print(f"\n  Loaded existing Q-table: {len(agent.q_table)} states, "
                  f"{agent.training_episodes} prior episodes")
        else:
            print("\n  No existing model — training from scratch")

    if args.evaluate:
        print("\n  [Evaluate-only mode]")
        evaluate_agent(agent, trades)
        summary = agent.policy_summary()
        print(f"\n  Q-table: {summary['states']} states")
        print(f"  Policy distribution: {summary.get('policy_distribution', {})}")
        return

    # Pre-training eval
    print("\n  Pre-training agent performance:")
    evaluate_agent(agent, trades)

    # Training loop
    print(f"\n  Training for {args.epochs} epochs ({args.epochs * len(trades)} episodes total)...")
    print(f"  Epsilon (exploration): {args.epsilon}")

    best_q_table = None
    best_avg_reward = -float("inf")

    for epoch in range(args.epochs):
        epoch_rewards = []
        # Shuffle trade order each epoch for better generalization
        rng = np.random.default_rng(seed=epoch)
        order = rng.permutation(len(trades))

        for i in order:
            t = trades[i]
            stats = agent.train_on_trajectory(
                premium_trajectory=t["premiums"],
                entry_premium=t["entry_premium"],
                sl=t["sl"],
                target=t["target"],
                max_hold_bars=len(t["premiums"]),
                lot_size=t["lot_size"],
            )
            epoch_rewards.append(stats["reward"])

        avg_reward = np.mean(epoch_rewards)
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_q_table = {k: v.copy() for k, v in agent.q_table.items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}/{args.epochs} | "
                f"avg_reward={avg_reward:.4f} | "
                f"states={len(agent.q_table)} | "
                f"episodes={agent.training_episodes}"
            )

    # Restore best Q-table
    if best_q_table:
        agent.q_table = best_q_table
        print(f"\n  Restored best Q-table (avg_reward={best_avg_reward:.4f})")

    # Post-training eval
    print("\n  Post-training agent performance:")
    evaluate_agent(agent, trades)

    # Save
    rl_path = MODEL_DIR / "rl_exit_agent.pkl"
    if rl_path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = BACKUP_DIR / datetime.now().strftime("%Y%m%d")
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rl_path, backup_dir / f"rl_exit_agent_pre_journey_{ts}.pkl")
        print(f"\n  Backed up existing RL model → backups/{datetime.now().strftime('%Y%m%d')}/")

    agent.save()
    summary = agent.policy_summary()
    print(f"  Saved: {summary['states']} states, {agent.training_episodes} episodes")
    print(f"  Policy: {summary.get('policy_distribution', {})}")

    print("\n" + "=" * 60)
    print("  RL JOURNEY TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
