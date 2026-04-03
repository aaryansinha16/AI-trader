# Signal Pipeline: Backtest vs Live

## Overview

The AI Trader uses the **same core pipeline** for both backtesting and live trading. The difference is in how data is sourced and how trades are managed.

```
tick → aggregate 1-min candle → compute features → detect regime
→ generate signals → ML scoring → options flow → composite score
→ resolve option contract → manage trade (SL / target / trailing / timeout)
```

---

## 1. Backtest Mode (`scripts/tick_replay_backtest.py`)

### Data Flow

| Step | What Happens | Frequency |
|------|-------------|-----------|
| **Load ticks** | All historical ticks for a day are loaded from `tick_data` table | Once per day |
| **Group into minutes** | Ticks are grouped by `floor(timestamp, 1min)` | — |
| **Build 1-min candle** | For each minute: OHLCV is computed from that minute's ticks | Every 1 minute |
| **Compute features** | 75+ macro indicators computed on rolling 300-candle buffer | Every 1 minute |
| **Detect regime** | Market regime (trending, volatile, sideways) from last 100 candles | Every 1 minute |
| **Generate signals** | Technical strategies scan the latest candle for entry patterns | Every 1 minute |
| **ML scoring** | Macro model + strategy-specific model score each signal | Per signal |
| **Composite score** | ML + options flow + technical strength + regime bonus | Per signal |
| **Score threshold** | Signal must exceed risk profile's `score_threshold` | Per signal |
| **Micro confirmation** | Tick-level momentum check on the current minute's raw ticks | Per qualifying signal |
| **Option resolution** | Look up real option premium from `minute_candles` for ATM strike | Per qualifying signal |
| **Trade management** | SL/TGT/trailing/timeout checked against option premium each minute | Every 1 minute |

### Why It Finishes in ~2 Seconds

The backtest is **not slow because it's using 1-min candles** — it's fast because:

1. **All tick data is pre-loaded** from the database in one SQL query (~10K ticks/day)
2. **Candle aggregation is instant** — just `groupby` + `agg` on in-memory DataFrames
3. **No network latency** — all data is local (DB + cached option premiums)
4. **~300 minutes per trading day** — so only ~300 iterations of the pipeline per day

For 2 days (Mar 23-24), that's ~600 iterations — trivially fast.

### Is 1-Min Candle Resolution a Problem?

**For signal generation: No.** Signals are based on indicators (RSI, MACD, Bollinger, VWAP, etc.) which are computed on 1-min candles. These indicators don't change meaningfully within a single minute.

**For trade management: Mostly no, but with caveats:**
- SL/TGT are checked against the option premium's OHLC for each minute
- The `high` catches the best price and `low` catches the worst price within that minute
- This means intra-minute spikes that hit SL then recover (or vice versa) are captured
- The only limitation: if price hits both SL and TGT in the same minute, the backtest assumes SL was hit first (conservative)

**For entry timing:** The backtest enters at the option premium's closing price for the signal minute. In live trading, the entry might be at a slightly different price due to order execution latency.

---

## 2. Live Mode (`backend/app.py`)

### Signal Scanning

| Component | Frequency | What It Does |
|-----------|-----------|-------------|
| **Background scanner** (`background_scanner`) | Every **30 seconds** | Loads latest 300 candles from DB, computes features, generates signals, scores them |
| **Tick collector** (`_ensure_collector`) | Continuous during market hours | Receives real-time ticks via TrueData WebSocket, writes to `tick_data` table |
| **Candle aggregator** | Every 1 minute | Aggregates buffered ticks into 1-min candles, writes to `minute_candles` table |
| **Tick monitor** (`_tick_monitor_loop`) | Every **1 second** | Reads latest tick prices, updates open position P&L, checks trailing SL |

### Live Signal Flow

```
[Every 30 seconds]
1. Load latest 300 1-min candles from DB
2. Compute all macro indicators (75+ features)
3. Detect market regime
4. Generate strategy signals (bearish_momentum, vwap_momentum_breakout, mean_reversion)
5. Score each signal: ML prob + options flow + technical strength + regime bonus
6. If score >= threshold → emit TRADE SUGGESTION to dashboard
7. User (or auto-mode) decides whether to take the trade

[Continuous — every 1 second]
8. Tick monitor reads latest price for open positions
9. Updates trailing SL, checks hard SL/TGT
10. Auto-exits if SL/TGT/timeout triggered
```

### Key Differences from Backtest

| Aspect | Backtest | Live |
|--------|----------|------|
| **Signal check frequency** | Every 1 minute (per candle) | Every 30 seconds |
| **Data source** | Historical ticks from DB | Real-time WebSocket ticks |
| **Trade execution** | Instant (simulated) | Requires order placement via broker API |
| **SL/TGT monitoring** | Per-minute candle OHLC | Per-second tick price |
| **Option premium** | Historical from DB | Real-time from market |
| **Slippage** | None (uses exact candle prices) | Real market slippage |
| **Commission** | Fixed ₹40 (₹20 × 2) | Actual brokerage charges |

---

## 3. Data Architecture

### Tick Data (`tick_data` table)

```
timestamp (TIMESTAMPTZ) | symbol | price | volume | oi | bid_price | ask_price | bid_qty | ask_qty
```

- **Source**: TrueData WebSocket (live) or REST API (historical backfill)
- **Granularity**: Every tick (~1-5 per second during market hours)
- **Retention**: All ticks stored permanently for replay

### Minute Candles (`minute_candles` table)

```
timestamp (TIMESTAMPTZ) | symbol | open | high | low | close | volume | vwap | oi
```

- **Source**: Aggregated from ticks (live) or fetched via REST (backfill)
- **Granularity**: 1-minute bars
- **Used by**: Both live scanner and backtest engine

### Timestamp Convention

- **Target**: All timestamps should be IST stored as `TIMESTAMPTZ`
- **Current state**: Mixed — older data (Mar 10-20) stored as IST-labeled-UTC, newer data (Mar 23+) stored as actual UTC
- **Handling**: The backtest's `minutes_from_open()` auto-detects and converts both formats

---

## 4. Why Premium Varies Every Few Seconds (Your Question)

You're right that option premiums change every few seconds in the live market. Here's how we handle it:

### In Backtest
- We use **1-minute OHLC of the option premium** (from `minute_candles`)
- The `high` and `low` of each minute capture the full intra-minute range
- SL is checked against `low`, TGT against `high` — so intra-minute extremes are accounted for
- Entry/exit prices use the `close` of the signal minute

### In Live Trading
- The **tick monitor** runs every **1 second**, reading the latest option premium
- SL/TGT are checked against the real-time premium, not 1-min candles
- This gives sub-second reaction to SL hits in live mode

### Gap Between Backtest and Live
The backtest may miss some scenarios:
- A premium that spikes up then crashes within 1 minute (SL might not have been hit in live)
- A brief premium dip that recovers (live tick monitor would exit, backtest might not)
- These are edge cases and the 1-min resolution is industry-standard for options backtesting

---

## 5. Complete Signal Generation Strategies

Three strategies generate entry signals:

| Strategy | Logic | Best Regime |
|----------|-------|-------------|
| **bearish_momentum** | Strong downtrend + volume confirmation → PUT | Trending Bear, High Volatility |
| **vwap_momentum_breakout** | Price breaks above/below VWAP with momentum → CALL/PUT | Trending Bull/Bear |
| **mean_reversion** | Oversold/overbought bounce at Bollinger bands → CALL/PUT | Sideways, Low Volatility |

Each signal is then scored by the composite formula:

```
final_score = 0.40 × ML_probability
            + 0.25 × options_flow_score
            + 0.35 × technical_strength
            + regime_bonus (±0.05 to ±0.08)
            + news_boost (0 or +0.03)
```

The signal must exceed the risk profile's `score_threshold` (0.58 for high risk, 0.60 for medium, 0.70 for low) to generate a trade.
