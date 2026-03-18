"""Quick backtest runner with ML gate for PUTs."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from database.db import read_sql
from models.predict import Predictor
from backtest.backtest_engine import BacktestEngine

print("Loading features from DB...")
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

predictor = Predictor()
predictor.load()

engine = BacktestEngine(
    sl_multiplier=1.5,
    target_multiplier=2.0,
    score_threshold=0.60,
    max_trades_per_day=3,
    max_holding_periods=30,
    lot_size=25,
    atm_delta=0.5,
    commission_per_order=20.0,
)

result = engine.run(df, symbol="NIFTY-I", predictor=predictor)

sep = "=" * 60
print(f"\n{sep}")
print("  BACKTEST v5 - ML GATE FOR PUTS")
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
    rr = abs(result.avg_win / result.avg_loss)
    print(f"  Risk-reward:   {rr:.2f}")
print(sep)

trades_data = [
    {"strategy": t.strategy, "direction": t.direction, "pnl": t.pnl, "result": t.result}
    for t in result.trades
]
if trades_data:
    tdf = pd.DataFrame(trades_data)
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

result.export_all(base_name="NIFTY_v5_mlgate", output_dir="backtest_results")
