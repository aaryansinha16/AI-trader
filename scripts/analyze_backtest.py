"""Quick profitability analysis of premium backtest v2 results."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

df = pd.read_csv("backtest_results/NIFTY_premium_v2_trades.csv")

print("=" * 60)
print("PREMIUM BACKTEST v2 — PROFITABILITY REPORT")
print("=" * 60)
print(f"Total trades:     {len(df)}")
print(f"Total P&L:        Rs {df['pnl'].sum():,.0f}")
print(f"Avg P&L/trade:    Rs {df['pnl'].mean():,.0f}")
print()

wins = df[df["result"] == "WIN"]
losses = df[df["result"] == "LOSS"]
timeouts = df[df["result"] == "TIMEOUT"]
print(f"WINs:     {len(wins)} ({len(wins)/len(df)*100:.1f}%)")
print(f"LOSSes:   {len(losses)} ({len(losses)/len(df)*100:.1f}%)")
print(f"TIMEOUTs: {len(timeouts)} ({len(timeouts)/len(df)*100:.1f}%)")
print()

profitable = df[df["pnl"] > 0]
unprofitable = df[df["pnl"] <= 0]
print(f"Profitable trades: {len(profitable)} ({len(profitable)/len(df)*100:.1f}%)")
print(f"Avg winner:        Rs {profitable['pnl'].mean():,.0f}")
print(f"Avg loser:         Rs {unprofitable['pnl'].mean():,.0f}")
rr = abs(profitable["pnl"].mean() / unprofitable["pnl"].mean()) if len(unprofitable) > 0 else 0
print(f"Risk-Reward:       {rr:.2f}")
print(f"Max single win:    Rs {df['pnl'].max():,.0f}")
print(f"Max single loss:   Rs {df['pnl'].min():,.0f}")
print()

# Equity curve
df["cum_pnl"] = df["pnl"].cumsum()
peak = df["cum_pnl"].cummax()
drawdown = (df["cum_pnl"] - peak).min()
print(f"Peak equity:       Rs {df['cum_pnl'].max():,.0f}")
print(f"Max drawdown:      Rs {drawdown:,.0f}")
print()

# By strategy
print("--- By Strategy ---")
for strat, g in df.groupby("strategy"):
    wr = (g["pnl"] > 0).mean() * 100
    print(f"  {strat:30s}  trades={len(g):3d}  WR={wr:.0f}%  total=Rs{g['pnl'].sum():>8,.0f}  avg=Rs{g['pnl'].mean():>6,.0f}")
print()

# By direction
print("--- By Direction ---")
for d, g in df.groupby("direction"):
    wr = (g["pnl"] > 0).mean() * 100
    print(f"  {d:10s}  trades={len(g):3d}  WR={wr:.0f}%  total=Rs{g['pnl'].sum():>8,.0f}  avg=Rs{g['pnl'].mean():>6,.0f}")
print()

# By result type
print("--- By Result ---")
for r, g in df.groupby("result"):
    wr = (g["pnl"] > 0).mean() * 100
    print(f"  {r:10s}  trades={len(g):3d}  profitable={wr:.0f}%  total=Rs{g['pnl'].sum():>8,.0f}  avg=Rs{g['pnl'].mean():>6,.0f}")
print()

# Monthly breakdown
print("--- Monthly P&L ---")
df["exit_dt"] = pd.to_datetime(df["exit_time"], errors="coerce")
df["month"] = df["exit_dt"].dt.to_period("M")
for m, g in df.dropna(subset=["month"]).groupby("month"):
    wr = (g["pnl"] > 0).mean() * 100
    print(f"  {str(m):10s}  trades={len(g):3d}  P&L=Rs{g['pnl'].sum():>8,.0f}  WR={wr:.0f}%")
print()

# Annualized
days = 122
print(f"Expectancy per trade: Rs {df['pnl'].mean():,.0f}")
print(f"Trading days:         {days}")
print(f"P&L per day:          Rs {df['pnl'].sum()/days:,.0f}")
print(f"Annualized (250 days): Rs {df['pnl'].sum()/days*250:,.0f}")

# Win streak / lose streak
streaks = []
current = 0
for p in df["pnl"]:
    if p > 0:
        current = max(current, 0) + 1
    else:
        current = min(current, 0) - 1
    streaks.append(current)
print(f"Max win streak:    {max(streaks)}")
print(f"Max lose streak:   {abs(min(streaks))}")
