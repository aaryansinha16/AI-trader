# AI Trader — Startup Guide

## Quick Start (Paper Trading Tomorrow)

### Prerequisites
- PostgreSQL + TimescaleDB running (`pg_isready` should show "accepting connections")
- Python virtual environment activated

### Step 1: Start TimescaleDB (if not already running)
```bash
brew services start postgresql@17
```

### Step 2: Start Paper Trading Dashboard
```bash
cd /Users/aaryansinha/Dev/Projects/ai-trader
source .venv/bin/activate
python backend/app.py
```

Open **http://localhost:5050** in your browser.

The dashboard will:
- Auto-scan every 30 seconds using data already in TimescaleDB
- Show NIFTY price, market regime, and trade suggestions
- Display ML probabilities and strategy scores

### Step 3: (Optional) Feed Live Ticks During Market Hours
In a **second terminal**:
```bash
cd /Users/aaryansinha/Dev/Projects/ai-trader
source .venv/bin/activate
python scripts/run_live_paper.py
```
This connects to TrueData and feeds live tick data into TimescaleDB.

---

## Do I Need to Retrain Models?

**No.** All models are saved as `.pkl` files and will load automatically:

| Model | File | Size |
|---|---|---|
| Macro (general) | `models/saved/macro_model.pkl` | 1.1 MB |
| Micro (tick) | `models/saved/micro_model.pkl` | 1.1 MB |
| VWAP Breakout | `models/saved/strategy/vwap_momentum_breakout_model.pkl` | 389 KB |
| Bearish Momentum | `models/saved/strategy/bearish_momentum_model.pkl` | 398 KB |
| Mean Reversion | `models/saved/strategy/mean_reversion_model.pkl` | 427 KB |

Models are trained on 6 months of NIFTY data (Sep 2025 – Mar 2026).

### When to Retrain
- **Daily (incremental)**: After market close, run `python scripts/run_live_paper.py` and press Ctrl+C — it auto-retrains on the day's new data
- **Weekly (full)**: `python main.py train` — retrains from scratch on all DB data
- **Strategy models**: `python -c "from models.strategy_models import train_all_strategy_models; ..."`

---

## Available Modes

| Command | What it does |
|---|---|
| `python backend/app.py` | Paper trading dashboard (web UI) |
| `python main.py backtest-real` | Backtest on real historical data with ML |
| `python main.py train` | Full model retraining from DB |
| `python main.py ingest` | Fetch fresh data from TrueData |
| `python scripts/run_premium_backtest_v2.py` | Premium backtest with all engines |
| `python scripts/tick_replay_sim.py` | Replay tick data simulation |

---

## System Architecture (What Happens Per Scan)

```
Every 30 seconds:
  1. Load latest 300 minute candles from TimescaleDB
  2. Compute 58 technical features (RSI, MACD, EMA, ATR, Bollinger, etc.)
  3. Detect market regime (Trending Bull/Bear, Sideways, High/Low Vol)
  4. Generate strategy signals (VWAP Breakout, Bearish Momentum, Mean Reversion)
  5. ML scoring:
     - General model: P(price UP) — filters bad PUTs
     - Strategy-specific model: P(this strategy succeeds)
  6. Options flow scoring from PCR, OI change, skew
  7. Composite score = 0.5×ML + 0.3×flow + 0.2×technical + regime bonus
  8. If score > 0.60 → show trade suggestion on dashboard
```

---

## Trading Parameters

| Parameter | Value | Notes |
|---|---|---|
| Lot size | 65 | NIFTY lot (since Jan 2026) |
| Commission | ₹40/trade | ₹20/order × 2 (Zerodha) |
| Max trades/day | 5 | Risk management |
| SL | 30% of premium | |
| Target | 50% of premium | |
| Max hold | 30 minutes | |
| Score threshold | 0.60 | Minimum score to suggest trade |
| PUT gate | ML > 0.40 → block | Prevents bad bearish trades |

---

## Troubleshooting

### "No ML models found"
Models are in `models/saved/`. If missing, retrain: `python main.py train`

### "Database connection failed"
```bash
pg_isready -h localhost -p 5432
# If not running:
brew services start postgresql@17
```

### Dashboard shows no data
The scanner reads from `minute_candles` table. If no recent data, run:
```bash
python main.py ingest  # fetches from TrueData
```
