# AI Trader — NSE F&O Algorithmic Trading Research System

> **Full-stack intraday options trading research platform for NIFTY** — tick-level replay backtest engine, XGBoost + RL ML models, dynamic risk management, and a live retro terminal dashboard.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Next.js 16](https://img.shields.io/badge/Next.js-16-black.svg)](https://nextjs.org/)
[![TimescaleDB](https://img.shields.io/badge/TimescaleDB-PostgreSQL%2017-blue.svg)](https://www.timescale.com/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost%20%2B%20RL-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---
## ⚠️ Important Disclaimer

> **This software is an algorithmic trading research and execution platform intended for educational, research, and controlled live trading use.**

* This system **can integrate with broker APIs and execute real trades**, but it is **not a licensed trading platform, broker, or investment advisor**
* All trading strategies, signals, and ML models are **experimental in nature** and may perform unpredictably in live markets
* Backtested and simulated results are **not indicative of future performance**
* Financial markets, especially derivatives and options, involve **substantial risk of loss**, including loss exceeding initial capital
* Users are solely responsible for:

  * their trading decisions
  * risk management
  * regulatory compliance (including SEBI/NSE rules for algo trading in India)
* The system does **not guarantee profitability**, consistency, or protection from losses
* Use of automated execution features should be done **with caution, proper safeguards, and preferably in phased deployment (paper → small capital → scale)**
* The authors and contributors **assume no liability** for financial losses, system failures, or regulatory issues arising from usage of this software

> ⚠️ **Trade responsibly. Start small. Assume everything can fail.**


---

## What This System Is

This is a **complete algorithmic trading research platform** that covers the full development lifecycle of a trading strategy:

| Layer | Capability |
|---|---|
| **Data** | Live tick collection via TrueData WebSocket + REST backfill into TimescaleDB |
| **Backtest** | Tick-level replay engine — same pipeline as live, on real historical tick data |
| **Feature Engineering** | 80 macro indicators + 5 micro tick features computed per bar |
| **ML Models** | XGBoost macro/micro/per-strategy models + Q-learning RL exit agent |
| **Signal Detection** | 3 rule-based strategies, scored and filtered by ML probability |
| **Risk Management** | Kelly criterion lot sizing, dynamic SL/target, trailing stops, regime gating |
| **Paper Trading** | Auto/manual paper trade execution with live position monitoring |
| **Dashboard** | Next.js terminal UI — live positions, charts, backtest runner, trade history |

**The paper-trading mode** is one component of the system — the platform is equally designed for strategy research, model training, and backtesting with real market data.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  DATA COLLECTION (scripts/collect_ticks.py)                            │
│  TrueData WebSocket → TickCollector → TimescaleDB                      │
│  · NIFTY-I futures (continuous) ticks                                  │
│  · ATM ±3 strikes × CE+PE = 14 option contracts                        │
│  · Dynamic re-subscription if NIFTY drifts 100+ pts from ATM           │
│  · 1-min candles aggregated in memory, flushed every minute             │
│  · Live price cache: /tmp/td_live_prices.json (1s refresh)              │
└────────────────────────────┬────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  FEATURE ENGINEERING (features/)                                       │
│  compute_all_macro_indicators(df) → 80 features from 1m candles        │
│  · Momentum: RSI, MACD, StochRSI, Williams%R, ROC(10/20), CCI          │
│  · Trend: EMA(9/20/50), SMA200, VWAP distance, ADX, DI+/DI-            │
│  · Volatility: ATR, Bollinger Bands, vol regime, volatility(20/60)      │
│  · Volume: OBV slope, MFI, volume ratio, volume delta, VWAP             │
│  · Multi-timeframe: RSI/EMA at 5m and 15m resolution                   │
│  · Options: PCR, OI change, IV, days_to_expiry, theta_pressure          │
│  · Session: minutes_since_open, session_progress, is_first/last_hour    │
│  compute_micro_features() → 5 tick-level features                      │
│  · bid_ask_spread, order_imbalance, trade_size_spike,                   │
│    volume_burst, tick_momentum                                          │
└────────────────────────────┬────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  ML MODELS (models/)                                                   │
│  ┌─────────────────────────────────────────────────────────┐           │
│  │ Macro Model (macro_model.pkl)                           │           │
│  │ · XGBoost binary classifier on 50 features              │           │
│  │ · Target: will NIFTY rise ≥ 0.1% in next 15 mins       │           │
│  │ · Walk-forward validation: 5 splits                     │           │
│  │ · Training samples: 6+ months of 1m candles             │           │
│  │ · Output: P(bullish) — used as directional gate         │           │
│  └─────────────────────────────────────────────────────────┘           │
│  ┌─────────────────────────────────────────────────────────┐           │
│  │ Micro Model (micro_model.pkl)                           │           │
│  │ · XGBoost on 5 tick-level features (per-second data)    │           │
│  │ · Target: net buying pressure in next 30 ticks          │           │
│  │ · Walk-forward: 3 splits, ~143K samples                 │           │
│  │ · Output: P(tick momentum bullish) — entry confirmation │           │
│  └─────────────────────────────────────────────────────────┘           │
│  ┌─────────────────────────────────────────────────────────┐           │
│  │ Strategy Outcome Models (strategy/*.pkl)                │           │
│  │ · One XGBoost per strategy                              │           │
│  │ · Trained on ACTUAL trade outcomes (WIN/LOSS) from      │           │
│  │   backtest CSVs, not synthetic forward-return labels    │           │
│  │ · Features: market state at entry (same 50 features)    │           │
│  └─────────────────────────────────────────────────────────┘           │
│  ┌─────────────────────────────────────────────────────────┐           │
│  │ RL Exit Agent (rl_exit_agent.pkl)                       │           │
│  │ · Tabular Q-learning (8-feature state space)            │           │
│  │ · Actions: HOLD / EXIT / TIGHTEN (SL tightening)        │           │
│  │ · Trained on premium trajectories from all journeys     │           │
│  │ · Decoupled from entry timing — trade-relative state    │           │
│  │ · 254K+ training episodes across 79 trade journeys      │           │
│  └─────────────────────────────────────────────────────────┘           │
└────────────────────────────┬────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  SIGNAL DETECTION (strategy/signal_generator.py)                       │
│  Every 30 seconds, checks 3 rule-based strategies:                     │
│                                                                         │
│  1. VWAP Momentum Breakout → CALL                                      │
│     price > VWAP + RSI > 55 + EMA20 > EMA50 + volume_spike (≥3/4)     │
│                                                                         │
│  2. Bearish Momentum → PUT                                              │
│     price < VWAP + RSI < 45 + EMA20 < EMA50 + volume_spike (≥3/4)     │
│                                                                         │
│  3. Mean Reversion → CALL or PUT                                        │
│     RSI < 30 (CALL) or RSI > 70 (PUT) + Bollinger touch + VWAP dist   │
│                                                                         │
│  Each signal has a technical_strength (0–1) from the # conditions met  │
└────────────────────────────┬────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  SCORING & FILTERING (backend/app.py: scan_market)                    │
│                                                                         │
│  final_score = 0.5 × directional_prob                                  │
│              + 0.3 × flow_score                                        │
│              + 0.2 × technical_strength                                │
│              + regime_bonus                                             │
│                                                                         │
│  directional_prob = ML_prob (CALL) or 1 − ML_prob (PUT)                │
│  flow_score = PCR-based: 0.5 default; boosts on PCR > 1.2              │
│  regime_bonus = +0.05 if strategy matches current regime               │
└────────────────────────────┬────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  POSITION MANAGEMENT (backend/app.py: _tick_monitor_loop)             │
│  Runs every 1 second from live price cache                             │
│                                                                         │
│  Kelly + Regime Lot Sizing:                                            │
│  · Kelly fraction from rolling 20-trade win rate + avg W/L ratios      │
│  · Scaled by regime multiplier (0.25x HIGH_VOL → 1.25x TRENDING)      │
│  · Bonus +1 lot for signals with score ≥ 0.70                         │
│  · Capped at 5 lots maximum                                            │
│                                                                         │
│  Dynamic SL/Target (ATR + score):                                      │
│  · SL: 12%–22% range scaled by ATR percentile + signal score           │
│  · Target: 40%–80% range                                               │
│                                                                         │
│  Trailing SL:                                                          │
│  · Activates after +12% move, locks in 8% profit                      │
│                                                                         │
│  RL Exit Agent:                                                        │
│  · Called every bar — HOLD / EXIT early / TIGHTEN SL                  │
│  · Fires early exit when Q-values favor taking profit                  │
│                                                                         │
│  Exits: SL_HIT / TARGET_HIT / TRAILING_SL / RL_EXIT / EOD_CLOSE       │
└────────────────────────────┬────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  DASHBOARD (dashboard/ — Next.js + Flask API)                          │
│  · Live page: positions, suggestions, auto/manual toggle, SSE stream   │
│  · Charts: NIFTY candles, option chain, tick charts                    │
│  · Backtest: run + view results, equity curve                          │
│  · Trades: full history, P&L, strategy breakdown                       │
│  · Settings: risk profile selector                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Technology | Details |
|---|---|---|
| **Frontend** | Next.js 16 + React 19 | Retro terminal UI, dark theme |
| **Styling** | Tailwind CSS v4 + Recharts | Custom dark palette, live charts |
| **Backend API** | Flask (Python 3.13) | SSE stream, REST endpoints, port 5050 |
| **Database** | TimescaleDB (PostgreSQL 17) | Hypertables for tick/candle time-series |
| **ML Models** | XGBoost 2.x + Q-learning | Binary classifiers + tabular RL agent |
| **Feature Pipeline** | scikit-learn + pandas | 80 macro + 5 micro features |
| **Data Feed** | TrueData REST + WebSocket | `wss://push.truedata.in:8084` |
| **ORM** | SQLAlchemy (read_sql/write_df) | Never raw psycopg2 |

---

## Prerequisites & Setup

### Requirements

| Requirement | Version | Purpose |
|---|---|---|
| Python | 3.13+ | Backend + ML pipeline |
| Node.js | 18+ | Next.js dashboard |
| PostgreSQL | 17 | With TimescaleDB extension |
| TrueData API | — | Live + historical market data |

### 1. Clone & Python environment

```bash
git clone https://github.com/yourusername/ai-trader.git
cd ai-trader
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Database setup

```bash
# macOS (Homebrew)
brew install postgresql@17 timescaledb

# Start PostgreSQL
brew services start postgresql@17

# Create database
createdb trading
psql -U postgres -d trading -c "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"

# Run schema (creates all hypertables)
python -c "from database.db import init_db; init_db()"
```

### 3. Environment variables

Create `.env` in the project root:

```env
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading
DB_USER=postgres
DB_PASSWORD=postgres

# TrueData (required for live data)
TRUEDATA_USER=your_username
TRUEDATA_PASSWORD=your_password

# Trading parameters (optional overrides)
INITIAL_CAPITAL=50000
ATM_RANGE=3              # strikes ±N from ATM (default 3 → 14 option contracts)
MAX_SYMBOLS=50           # TrueData plan max
SCORE_THRESHOLD=0.6      # minimum composite score to suggest a trade
LOG_LEVEL=INFO
MODEL_DIR=models/saved
```

### 4. Dashboard

```bash
cd dashboard
npm install
```

---

## Running the System

### Start everything (market hours only — 9:15–15:30 IST)

**Terminal 1 — Flask API backend:**
```bash
source .venv/bin/activate
python backend/app.py
# Serves on http://localhost:5050
# Auto-starts tick collector at market open
```

**Terminal 2 — Next.js dashboard:**
```bash
cd dashboard && npm run dev
# Open http://localhost:3000
```

### Backtest

```bash
# Tick-level replay backtest (most accurate — uses real historical ticks)
python scripts/tick_replay_backtest.py --risk medium

# Specific dates only
python scripts/tick_replay_backtest.py --risk high 2026-03-23 2026-03-24 2026-03-25

# All three profiles
python scripts/tick_replay_backtest.py --risk low
python scripts/tick_replay_backtest.py --risk medium
python scripts/tick_replay_backtest.py --risk high
```

### Model Training Workflow (Post-Market)

```bash
# 1. Train per-strategy models on actual trade outcomes (safe, recommended daily):
python scripts/train_outcome_models.py

# 2. Retrain RL exit agent on all saved trade journeys:
python scripts/train_rl_on_journeys.py --epochs 50

# 3. Full macro/micro retrain (only when backtest baseline is confirmed stable):
python scripts/incremental_train.py
```

### Backfill missing data

```bash
# Fill today's candles + ticks for NIFTY-I + all ATM options via REST
python scripts/backfill_today.py

# Fill specific date for single symbol
python scripts/fetch_missing_ticks.py --dates 2026-03-25 --symbol NIFTY-I
```

---

## ML Models — Deep Dive

### How signals are detected and scored

The signal pipeline runs every **30 seconds** during market hours:

```
1. Load last 300 1-minute candles from TimescaleDB
2. Compute 80 macro features (RSI, MACD, EMA, VWAP, ATR, PCR, IV, etc.)
3. Run 3 rule-based strategy checks → candidate signals
4. For each signal:
   a. Feed 50 features to XGBoost macro model → P(bullish) in [0,1]
   b. directional_prob = P(bullish) for CALL, 1-P(bullish) for PUT
   c. flow_score from Put-Call Ratio + OI change
   d. final_score = 0.5×directional_prob + 0.3×flow + 0.2×technical + regime_bonus
5. If final_score ≥ threshold AND strategy outcome model confirms → suggest/enter
```

### Macro Model — `models/saved/macro_model.pkl`

| Property | Value |
|---|---|
| Algorithm | XGBoost binary classifier |
| Training data | `minute_candles` for NIFTY-I, all available days |
| Features | 50 of 80 computed (`FEATURE_COLUMNS_MACRO` in settings.py) |
| Target label | Will price rise ≥ 0.1% in next 15 candles? (~14% positive rate) |
| Validation | 5-fold walk-forward (chronological splits) |
| Output | Float in [0,1]: P(bullish 15-min move) |

> **Key calibration note**: Label threshold must stay at `0.001/15 bars`. Higher thresholds compress positive rate to <5%, collapsing all outputs near 0 and destroying directional discrimination.

### Micro Model — `models/saved/micro_model.pkl`

| Property | Value |
|---|---|
| Features | `bid_ask_spread, order_imbalance, trade_size_spike, volume_burst, tick_momentum` |
| Target | Net buying pressure in next 30 ticks |
| Use | Entry confirmation — micro model must agree with signal direction before entry |

### Strategy Outcome Models — `models/saved/strategy/*.pkl`

Trained on **actual trade outcomes** (WIN/LOSS) from backtest CSVs — not synthetic forward-return labels:

| Model | Training Samples | Win Rate in Data | Notes |
|---|---|---|---|
| `bearish_momentum` | 38 | 76% | Primary strategy; PUT signals |
| `mean_reversion` | <15 | — | Skipped until more trades accumulated |

> These models need 15+ samples per strategy to train. Run more backtests → more outcome data → better discrimination.

### RL Exit Agent — `models/saved/rl_exit_agent.pkl`

| Property | Value |
|---|---|
| Algorithm | Tabular Q-learning (dict-based Q-table) |
| State space | 8 features: pnl_pct, bars_held_norm, momentum, volatility, dist_to_sl, dist_to_tgt, trailing_active, peak_gain |
| Action space | HOLD / EXIT / TIGHTEN |
| Training data | 79 premium trajectories from all backtest journeys (high/medium/low) |
| Episodes | 254,000+ |
| Policy | 23% early EXIT, 29% TIGHTEN, 48% HOLD till natural exit |

The RL agent is decoupled from entry timing — all state features are trade-relative. It learns when to exit early vs hold based purely on the shape of the premium trajectory.

---

## Strategies — How Signals Are Generated

### Strategy 1: VWAP Momentum Breakout → CALL

```
Conditions (need ≥ 3 of 4):
  ✓ close > VWAP               (price above intraday average)
  ✓ RSI > 55                   (momentum not overbought yet)
  ✓ EMA20 > EMA50              (short-term trend above medium-term)
  ✓ volume_spike OR ratio > 1.5x
```
Active primarily in bullish trending markets. Requires minimum score ≥ 0.65.

### Strategy 2: Bearish Momentum → PUT

```
Conditions (need ≥ 3 of 4):
  ✓ close < VWAP
  ✓ RSI < 45
  ✓ EMA20 < EMA50
  ✓ volume_spike OR ratio > 1.5x
```
Best-performing strategy in current backtests: 60–76% win rate across all risk profiles.

### Strategy 3: Mean Reversion → CALL or PUT

```
CALL (oversold):        PUT (overbought):
  RSI < 30                RSI > 70
  close ≤ BB_lower        close ≥ BB_upper
  VWAP dist > 0.3%        VWAP dist > 0.3%
```
Rare signals but high average gain when conditions align. Filtered to SIDEWAYS/LOW_VOL regimes only.

---

## Risk Profiles

Three profiles selectable from the Settings page.

### LOW (Conservative)

```
Entry threshold:  score ≥ 0.70 (CALL) / ≥ 0.78 (PUT)
Max premium:      ₹200 per option
SL range:         12%–20% (dynamic, ATR-scaled)
Target range:     40%–65%
Trailing trigger: +12% move → activate
Hold timeout:     30 minutes
Afternoon cut:    12:30 IST
```

### MEDIUM (Balanced) — Default

```
Entry threshold:  score ≥ 0.60 (CALL) / ≥ 0.70 (PUT)
Max premium:      ₹250 per option
SL range:         12%–22% (dynamic)
Target range:     40%–80%
Trailing trigger: +12% move → activate; lock at +8%
Hold timeout:     40 minutes
Afternoon cut:    12:45 IST
```

### HIGH (Aggressive)

```
Entry threshold:  score ≥ 0.60 (CALL) / ≥ 0.70 (PUT)
Max premium:      ₹250 per option
SL range:         12%–22%
Target range:     40%–80%
Hold timeout:     40 minutes
Afternoon cut:    12:45 IST
Kelly sizing:     20% more capital per trade vs MEDIUM
```

---

## Backtest Results (March–April 2026, ~11 Trading Days)

These results are from `tick_replay_backtest.py` using **actual historical tick data** — the same pipeline runs in live mode. All three risk profiles were tested on the same date range.

### HIGH Risk

| Metric | Value |
|---|---|
| Total Trades | 37 |
| Win Rate | **62%** |
| Net P&L | **+₹20,434** |
| Avg per Trade | +₹552 |
| Avg Win | +₹1,773 |
| Avg Loss | -₹1,453 |

Exit breakdown: TRAILING_SL 46%, SL 27%, RL_EXIT 14%, TIMEOUT 11%, EOD 3%

### MEDIUM Risk

| Metric | Value |
|---|---|
| Total Trades | 32 |
| Win Rate | **72%** |
| Net P&L | **+₹29,689** |
| Avg per Trade | +₹928 |
| Avg Win | +₹1,884 |
| Avg Loss | -₹1,516 |

Exit breakdown: TRAILING_SL 47%, SL 22%, RL_EXIT 16%, TIMEOUT 12%, EOD 3%

### LOW Risk

| Metric | Value |
|---|---|
| Total Trades | 9 |
| Win Rate | **78%** |
| Net P&L | **+₹9,408** |
| Avg per Trade | +₹1,045 |
| Avg Win | +₹1,400 |
| Avg Loss | -₹197 |

Exit breakdown: TRAILING_SL 89%, TIMEOUT 11%

> **Note**: The RL_EXIT agent contributes ~14–16% of exits (profitable early exits). Trailing SL is the most common profitable exit type. The backtest was run with the max_trades/day gate disabled to collect maximum training data.

---

## Database Schema

All timestamps stored as `TIMESTAMPTZ` (UTC). Displayed as IST (+5:30) in frontend.

| Table | Type | Key Columns | Notes |
|---|---|---|---|
| `tick_data` | Hypertable | `timestamp, symbol, price, volume, oi, bid_price, ask_price` | ~8K ticks/day/symbol |
| `minute_candles` | Hypertable | `timestamp, symbol, open, high, low, close, volume, vwap, oi` | Primary ML training source |
| `option_chain` | Hypertable | `timestamp, symbol, strike, option_type, ltp, oi, iv, delta` | Snapshots |
| `symbol_master` | Regular | `symbol, expiry, strike, option_type, lot_size` | TrueData F&O universe |
| `trade_log` | Regular | `entry_time, exit_time, symbol, side, entry_price, pnl, ml_score` | Paper trade history |
| `features_macro` | Regular | 17 feature columns | Computed features |
| `features_micro` | Regular | 5 feature columns | Tick-level features |
| `daily_performance` | Regular | `date, total_trades, wins, net_pnl, win_rate` | EOD summary |

---

## Flask API Routes

All served on `http://localhost:5050`:

| Route | Method | Description |
|---|---|---|
| `/api/stream` | GET | SSE: live price + state updates every ~1s |
| `/api/state` | GET | Full scanner state (regime, suggestions, positions) |
| `/api/scan` | POST | Manually trigger one scan cycle |
| `/api/paper/enter` | POST | Enter a paper trade |
| `/api/paper/exit` | POST | Exit an open position |
| `/api/paper/positions` | GET | Open positions |
| `/api/paper/clear` | POST | Clear all positions |
| `/api/auto_trade` | GET/POST | Get or set AUTO/MANUAL mode |
| `/api/live/prices` | GET | Live prices from tick cache |
| `/api/trades/history` | GET | Past completed trades |
| `/api/equity/curve` | GET | Daily equity curve |
| `/api/risk/profiles` | GET | List risk profile configs |
| `/api/market/candles` | GET | Last N candles for symbol |
| `/api/backtest/run` | POST | Run tick replay backtest |
| `/api/backtest/results` | GET | Saved backtest results |
| `/api/backtest/progress` | GET | SSE: backtest progress |

---

## TrueData Integration

### Symbol Naming

```
NIFTY-I              → NIFTY continuous futures (historical + live ticks)
NIFTY 50             → NIFTY spot index (WebSocket only)
NIFTY{YYMMDD}{STRIKE}{CE|PE}  → Options e.g. NIFTY26040122400PE
```

**Weekly expiry: Tuesdays** (confirmed from `symbol_master` table).

### WebSocket Flow

```
Connect → wss://push.truedata.in:8084?user=X&password=Y
Auth response → { "success": true, "maxsymbols": 50 }
Subscribe → { "method": "addsymbol", "symbols": [...] }
Snapshot → { "symbollist": [[symbol, symbolID, ts, LTP, ...], ...] }  (18 fields)
Live tick → { "trade": [symbolID, ts, LTP, LTQ, ATP, OI, ...] }       (no symbol name!)
                ↑ symbolID is mapped to name via _symbol_id_map built during subscribe
```

---

## Project File Structure

```
ai-trader/
├── backend/app.py          # Flask API (port 5050) — main backend
│
├── dashboard/               # Next.js 16 frontend (port 3000)
│   ├── app/live/            # Live page: positions, suggestions, auto/manual toggle
│   ├── app/charts/          # Candle charts, option chain viewer
│   ├── app/backtest/        # Backtest runner + results viewer
│   ├── app/trades/          # Trade history with P&L analytics
│   └── app/settings/        # Risk profile selector
│
├── scripts/
│   ├── collect_ticks.py          # Live tick collector (WebSocket, market hours)
│   ├── incremental_train.py      # Daily macro/micro model retraining
│   ├── train_outcome_models.py   # Per-strategy models on actual trade outcomes
│   ├── train_rl_on_journeys.py   # RL exit agent on all journey data
│   ├── tick_replay_backtest.py   # Tick-level replay backtest engine
│   ├── fetch_missing_ticks.py    # Backfill single symbol via REST
│   └── backfill_today.py         # Backfill all today's symbols
│
├── models/
│   ├── train_model.py       # MacroModelTrainer + MicroModelTrainer
│   ├── strategy_models.py   # Per-strategy outcome model training
│   ├── rl_exit_agent.py     # RLExitAgent (Q-learning, 8-feature state)
│   ├── predict.py           # Predictor (load + infer)
│   └── saved/
│       ├── macro_model.pkl
│       ├── micro_model.pkl
│       ├── rl_exit_agent.pkl
│       ├── strategy/        # bearish_momentum_model.pkl, etc.
│       └── backups/YYYYMMDD/ # Date-organized backups before each retrain
│
├── features/
│   ├── indicators.py        # compute_all_macro_indicators() — 80 features
│   └── micro_features.py    # compute_micro_features() — 5 features
│
├── strategy/
│   ├── signal_generator.py  # 3 rule-based strategies → Signal objects
│   ├── trade_scorer.py      # Composite score = ML + flow + technical
│   ├── regime_detector.py   # EMA/ATR-based regime classification
│   └── options_flow_detector.py  # PCR, OI flow analysis
│
├── config/
│   ├── settings.py          # All constants: DB URL, symbols, weights, thresholds
│   └── risk_profiles.py     # LOW / MEDIUM / HIGH RiskProfile dataclasses
│
├── data/
│   ├── truedata_adapter.py  # TrueData REST + WebSocket client
│   └── tick_collector.py    # TickCollector (buffers 200 ticks → DB flush)
│
├── backtest/
│   ├── backtest_engine.py   # Simple candle-level backtest
│   └── option_resolver.py   # get_nearest_expiry() + premium lookup
│
└── database/
    ├── db.py                # read_sql / write_df / upsert_candles / init_db
    └── schema.sql           # Full TimescaleDB schema
```

---

## Known Limitations & Real-World Gaps

1. **Slippage**: Modeled at flat ₹40 commission per trade. Real options slippage can be 0.5–2%+ for illiquid strikes — actual P&L would be lower
2. **Bid-ask spread**: System uses close/LTP price, not actual fill price. For wide-spread options, fills may be worse
3. **Data gaps**: If TrueData WebSocket drops, option premiums go stale → REST fallback (1 min delay)
4. **Model drift**: XGBoost trained on 6-month history. During regime changes (budget, elections, global risk-off), accuracy degrades
5. **Expiry day behavior**: On Tuesdays (expiry day), extreme theta decay and gamma spikes are only partially represented in training data
6. **Outcome model sample size**: Only ~50 unique backtest trades so far → outcome models have AUC ≈ 0.50. More backtests needed before they add value
7. **Lot sizing ceiling**: Kelly + score bonus resolves to 2 lots (130 units) for virtually all signals at ₹50K starting capital. True variation requires more equity or explicit score-tiered sizing

---

## Common Issues

| Problem | Cause | Fix |
|---|---|---|
| No trade suggestions | Score < threshold or strat_prob < 0.02 | Check regime, ML prob in logs |
| SL hit instantly | Stale option price | DB candle age check + REST fallback |
| Live price not updating | NIFTY drifted, no subscription | Collector auto re-subscribes every 2 min |
| `inf` values in XGBoost | Feature computation | Fixed: `replace([inf, -inf], nan)` before training |
| Duplicate candle inserts | Raw `write_df` append | Use `upsert_candles()` always |
| Collector dies at shell exit | Started without `nohup` | Always: `nohup .venv/bin/python scripts/collect_ticks.py &` |
| Outcome models AUC = 0.50 | Too few training samples | Run more backtests on historical data |

---

## License

MIT — see [LICENSE](LICENSE)

---

## Full Disclaimer

This software is provided for **educational, research, and paper-trading purposes only**.

- **Not financial advice**: Nothing in this codebase constitutes investment, trading, or financial advice
- **No warranty**: Provided "as-is" with no guarantees of accuracy, fitness, or profitability
- **High risk**: Options trading involves the potential for total loss and more; past backtested performance does not guarantee future results
- **Regulatory compliance**: Live algorithmic trading in India requires SEBI registration and compliance with NSE/BSE/SEBI exchange regulations — this system does **not** provide or constitute regulatory compliance
- **The authors and contributors accept no liability** for financial losses, trading errors, regulatory violations, or any other damages arising from the use of this software
