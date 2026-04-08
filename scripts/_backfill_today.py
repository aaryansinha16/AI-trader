#!/usr/bin/env python3
"""One-shot: backfill today's missing NIFTY-I candles and ATM option candles from 9:15 to now."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv; load_dotenv()

from datetime import datetime, date
import pandas as pd
from data.truedata_adapter import TrueDataAdapter
from database.db import upsert_candles, read_sql
from backtest.option_resolver import get_nearest_expiry

today = date.today()
day_start = datetime(today.year, today.month, today.day, 9, 15, 0)
end_dt = datetime.now()

print(f"Backfilling today ({today}) from 09:15 to {end_dt.strftime('%H:%M')}...")

td = TrueDataAdapter()
if not td.authenticate():
    print("ERROR: TrueData auth failed")
    sys.exit(1)
print("Auth OK")

# 1. NIFTY-I candles
bars = td.fetch_historical_bars("NIFTY-I", day_start, end_dt, interval="1min")
if bars.empty:
    print("WARNING: No NIFTY-I bars returned")
    nifty_close = 0
else:
    for col in ["vwap", "oi"]:
        if col not in bars.columns:
            bars[col] = 0
    bars = bars[["timestamp", "symbol", "open", "high", "low", "close", "volume", "vwap", "oi"]]
    upsert_candles(bars)
    nifty_close = float(bars.iloc[-1]["close"])
    first_ts = str(bars.iloc[0]["timestamp"])
    last_ts = str(bars.iloc[-1]["timestamp"])
    print(f"NIFTY-I: upserted {len(bars)} candles  [{first_ts} → {last_ts}]  last={nifty_close:.1f}")

# 2. ATM option candles
if nifty_close > 0:
    atm = round(nifty_close / 50) * 50
    expiry = get_nearest_expiry(today)
    if not expiry:
        print("WARNING: No expiry found, skipping options")
        sys.exit(0)
    exp_code = expiry.strftime("%y%m%d")
    print(f"\nATM={atm}, Expiry={expiry}")
    expected_bars = max(1, int((end_dt - day_start).total_seconds() / 60))

    for delta in range(-3, 4):
        strike = atm + delta * 50
        for opt in ["CE", "PE"]:
            sym = f"NIFTY{exp_code}{strike}{opt}"
            chk = read_sql(
                "SELECT COUNT(*) as cnt FROM minute_candles WHERE symbol=:s AND timestamp::date=:d",
                {"s": sym, "d": today.isoformat()}
            )
            cnt = int(chk.iloc[0]["cnt"]) if not chk.empty else 0
            if cnt >= expected_bars * 0.8:
                print(f"  {sym}: already has {cnt}/{expected_bars} bars, skipping")
                continue
            try:
                ob = td.fetch_historical_bars(sym, day_start, end_dt, interval="1min")
                if not ob.empty:
                    for col in ["vwap", "oi"]:
                        if col not in ob.columns:
                            ob[col] = 0
                    ob = ob[["timestamp", "symbol", "open", "high", "low", "close", "volume", "vwap", "oi"]]
                    upsert_candles(ob)
                    print(f"  {sym}: upserted {len(ob)} bars (had {cnt})")
                else:
                    print(f"  {sym}: no data returned")
                time.sleep(1.1)
            except Exception as e:
                print(f"  {sym}: ERROR {e}")

print("\nDone.")
