# AI Trader — CLAUDE.md

Intraday NIFTY options paper-trading system. Collects live ticks, trains XGBoost ML models, scores trade signals, and presents everything through a Next.js dashboard backed by a Flask API.

---

## Stack

| Layer | Tech |
|---|---|
| Frontend | Next.js 16 + React 19 + Tailwind v4 + Recharts |
| Backend API | Flask (Python 3.13), SSE stream at `/api/stream` |
| Database | TimescaleDB (PostgreSQL 17) — hypertables for tick/candle data |
| ML | XGBoost + LightGBM (scikit-learn pipeline), joblib `.pkl` models |
| Data Feed | TrueData REST + WebSocket (`wss://push.truedata.in:8084`) |
| Runtime | macOS, Python venv at `.venv/`, Node in `dashboard/node_modules/` |

---

## Directory Structure

```
ai-trader/
├── frontend/app.py          # Flask API server (port 5050) — the main backend
├── dashboard/               # Next.js frontend (port 3000)
│   └── app/
│       ├── live/            # Live trading page (SSE stream, positions, suggestions)
│       ├── charts/          # Option chain + candle charts
│       ├── backtest/        # Backtest runner + results
│       ├── trades/          # Trade history
│       ├── ai/              # AI chat
│       └── settings/        # Risk profile + config
├── scripts/
│   ├── collect_ticks.py     # LIVE tick collector (runs during market hours)
│   ├── incremental_train.py # Daily model retraining after market close
│   ├── fetch_missing_ticks.py  # Backfill single symbol via REST
│   └── backfill_today.py    # Backfill all today's symbols via REST
├── models/
│   ├── train_model.py       # MacroModelTrainer + MicroModelTrainer classes
│   ├── strategy_models.py   # Strategy-specific XGBoost models
│   ├── predict.py           # Predictor wrapper (load + infer)
│   └── saved/               # .pkl files (macro_model.pkl, micro_model.pkl, backups/)
├── data/
│   ├── truedata_adapter.py  # TrueData REST + WebSocket client
│   └── tick_collector.py    # TickCollector (buffers 200 ticks → DB flush)
├── features/
│   ├── indicators.py        # compute_all_macro_indicators() — 58 features
│   └── micro_features.py    # compute_micro_features() — 5 features
├── strategy/
│   ├── signal_generator.py  # Generates BUY/SELL signals per strategy
│   ├── trade_scorer.py      # Composite score = 0.5×ML + 0.3×flow + 0.2×tech
│   ├── regime_detector.py   # TRENDING_BULL/BEAR/SIDEWAYS/HIGH_VOL/LOW_VOL
│   └── options_flow_detector.py
├── config/
│   ├── settings.py          # All constants (DB URL, symbols, thresholds)
│   └── risk_profiles.py     # LOW/MEDIUM/HIGH RiskProfile dataclasses
├── database/
│   ├── db.py                # read_sql / write_df / upsert_candles / init_db
│   └── schema.sql           # Full TimescaleDB schema
└── backtest/
    ├── backtest_engine.py
    └── option_resolver.py   # get_nearest_expiry(date) → next Tuesday expiry
```

---

## How to Run

### Prerequisites
```bash
brew services start postgresql@17   # TimescaleDB must be running
cd /Users/aaryansinha/Dev/Projects/ai-trader
source .venv/bin/activate
```

### Backend (Flask API)
```bash
python frontend/app.py              # http://localhost:5050
```
Auto-starts `collect_ticks.py` during market hours via `_ensure_collector()`.

### Frontend (Next.js dashboard)
```bash
cd dashboard && npm run dev         # http://localhost:3000
```

### Live Tick Collector (market hours only: 9:15–15:30 IST)
```bash
# Flask auto-starts this. To run manually:
nohup .venv/bin/python scripts/collect_ticks.py >> logs/tick_collector_YYYYMMDD.log 2>&1 &
```
Writes to `/tmp/td_live_prices.json` every 1s (Flask reads this for streaming prices).

### Retrain Models (after market close)
```bash
# Full retrain on ALL historical data (preferred):
python scripts/incremental_train.py

# Incremental only (1–2 new days, uses warm-start):
python scripts/incremental_train.py --days 1

# Full retrain from scratch:
python main.py train
```

### Fill Missing Data
```bash
# Fill today's candles + ticks for NIFTY-I + all ATM options via REST:
python scripts/backfill_today.py

# Fill specific date for a single symbol:
python scripts/fetch_missing_ticks.py --dates 2026-03-25 --symbol NIFTY-I
```

### DB Init / Schema
```bash
python -c "from database.db import init_db; init_db()"
```

---

## Database Schema (TimescaleDB Hypertables)

| Table | Key Columns | Notes |
|---|---|---|
| `tick_data` | `timestamp, symbol, price, volume, oi, bid_price, ask_price` | Hypertable; ~8k ticks/day/symbol |
| `minute_candles` | `timestamp, symbol, open, high, low, close, volume, vwap, oi` | Primary ML training source; upsert by (timestamp, symbol) |
| `option_chain` | `timestamp, symbol, underlying, expiry, strike, option_type, ltp, oi, iv, delta` | Snapshots |
| `symbol_master` | `symbol, underlying, expiry, strike, option_type, lot_size` | TrueData F&O universe |
| `trade_log` | `entry_time, exit_time, symbol, side, entry_price, pnl, ml_score, final_score` | Paper trade history |
| `features_macro` | 17 feature columns | Computed from minute_candles |
| `features_micro` | 5 feature columns | bid_ask_spread, order_imbalance, etc. |
| `daily_performance` | `date, total_trades, wins, net_pnl, win_rate` | EOD summary |

All timestamps are `TIMESTAMPTZ` (stored UTC, displayed as IST +5:30).

---

## TrueData Integration

### Symbol Naming
```
NIFTY-I            → NIFTY continuous futures (historical bars, live ticks)
NIFTY 50           → NIFTY spot index (websocket subscription only)
NIFTY{YYMMDD}{STRIKE}{CE|PE}  → Options e.g. NIFTY26033022950PE
```
NIFTY weekly expiry is on **Tuesdays** (confirmed from symbol_master table).

### WebSocket (live streaming)
- URL: `wss://push.truedata.in:8084?user=X&password=Y`
- Auth response: `{"success": true, "segments": ["FO","IND",""], "maxsymbols": 50}`
- Subscribe: `{"method": "addsymbol", "symbols": ["NIFTY-I", ...]}`
- Subscription snapshot: `{"symbollist": [[symbol, symbolID, ts, LTP, ...], ...]}` (18 fields, starts with symbol name)
- **Live streaming ticks**: `{"trade": [symbolID, ts, LTP, LTQ, ATP, TTQ, O, H, L, prevclose, OI, prevOI, turnover, tag, bid_qty, bid, ask_qty, ask]}` — **no symbol name**, symbolID must be mapped
- Heartbeats: `{"message": "HeartBeat", "timestamp": "..."}`
- `_symbol_id_map` in `TrueDataAdapter` maps `symbolID → symbol name` built during `ws_subscribe()`

### Rate Limits
- REST: 1 request/second (`TD_RATE_LIMIT_RPS = 1`)
- WebSocket: max 50 symbols per connection

### REST Endpoints
```
GET  https://history.truedata.in/getbars        → OHLCV bars (1min/5min/eod)
GET  https://history.truedata.in/getticks       → tick data
GET  https://history.truedata.in/getlastnbars   → last N bars (max 200)
GET  https://history.truedata.in/getlastnticks  → last N ticks (max 200)
POST https://auth.truedata.in/token             → bearer token
```

---

## ML Pipeline

### Models
| Model | File | Algorithm | Training data |
|---|---|---|---|
| Macro | `models/saved/macro_model.pkl` | XGBoost | All minute_candles for NIFTY-I (6+ months) |
| Micro | `models/saved/micro_model.pkl` | XGBoost | All tick_data (all available days) |
| Strategy models | `models/saved/strategy/*.pkl` | XGBoost | Per-strategy labeled trades |

### Training
- **MacroModelTrainer**: `train(df, walk_forward=True, n_splits=5)` — 5-fold walk-forward validation on 58 features
- **MicroModelTrainer**: `train(df, walk_forward=True, n_splits=3)` — 3-fold on 5 micro features
- Always backup existing model before retraining: `models/saved/backups/macro_model_YYYYMMDD_HHMMSS.pkl`

### Features (58 macro, 5 micro)
**Macro** (from `FEATURE_COLUMNS_MACRO` in settings.py): RSI, MACD, EMA(9/20/50), SMA200, VWAP distance, Bollinger, ATR, StochRSI, Williams%R, ROC, ADX, CCI, OBV slope, MFI, PCR, IV, OI change, days_to_expiry, multi-timeframe RSI/EMA (5m/15m), session time features.

**Micro** (from `FEATURE_COLUMNS_MICRO`): `bid_ask_spread, order_imbalance, trade_size_spike, volume_burst, tick_momentum`.

### Trade Scoring
```
final_score = 0.5 × ML_prob + 0.3 × flow_score + 0.2 × technical_strength ± regime_bonus
```

---

## Risk Profiles

| Profile | Score Threshold (CALL/PUT) | SL | Target | Max Trades/Day | Afternoon Cut |
|---|---|---|---|---|---|
| LOW (Conservative) | 0.70 / 0.80 | 20% | 35% | 3 | 11:45 IST |
| MEDIUM (Balanced) | 0.60 / 0.70 | 20% | 50% | 5 | 12:30 IST |
| HIGH (Aggressive) | 0.58 / 0.65 | 20% | 80% | 8 | 13:15 IST |

NIFTY lot size = 65 units. Suggestion SL/target in `frontend/app.py`:
```python
INITIAL_SL_PCT = 0.15   # SL = entry × 0.85
TGT_PCT = 0.50          # Target = entry × 1.50
SUGGESTION_COOLDOWN_SECS = 300  # 5 min between re-suggesting same signal
```

---

## Flask API Routes (frontend/app.py — port 5050)

```
GET  /api/state                → full scanner state (regime, suggestions, positions)
GET  /api/stream               → SSE stream (NIFTY price, state updates every ~1s)
POST /api/scan                 → manual trigger scan
POST /api/paper/enter          → enter paper trade
POST /api/paper/exit           → exit paper trade
GET  /api/paper/positions      → open positions
POST /api/paper/clear          → clear all positions
GET  /api/live/prices          → live prices from tick cache
GET  /api/trades/history       → past trades
GET  /api/equity/curve         → equity curve for chart
GET  /api/risk/profiles        → list risk profiles
GET  /api/days                 → dates with candle data
GET  /api/market/candles       → last N candles for symbol
GET  /api/market/candles/date  → candles for specific date+symbol
GET  /api/market/ticks/date    → ticks for specific date+symbol
GET  /api/candle_dates         → dates available in minute_candles
GET  /api/options/expiries     → available option expiry dates
GET  /api/options/chain        → option chain for date+expiry (from minute_candles)
GET  /api/options/ticks        → tick chart for a specific option symbol+date
POST /api/backtest/run         → run backtest
GET  /api/backtest/results     → saved backtest results
GET  /api/backtest/progress    → backtest progress SSE
```

### Key State Variables (in-memory, resets on Flask restart)
```python
state = {
    "trade_suggestions": [...],  # expires after SUGGESTION_COOLDOWN_SECS
    "positions": [...],          # open paper trades
    "regime": "...",
    "status": "idle|scanning|market_closed"
}
```

### Background Threads
- `background_scanner` — runs every 30s, calls `scan_market()` + `_ensure_collector()`
- `_tick_monitor_loop` — runs every 1s, updates open position prices from tick cache
- `collect_ticks.py` — separate subprocess, auto-started by `_ensure_collector()`

---

## Live Price Cache

**File**: `/tmp/td_live_prices.json`

**Format**: `{"NIFTY-I": {"price": 22950.0, "ts": "2026-03-27T13:42:47.362923"}, ...}`

- Written every 1s by `collect_ticks.py`'s `_flush_price_cache` thread
- `ts` = **wall-clock time when tick was received** (NOT the tick's market timestamp)
- Flask reads this in `_tick_monitor_loop` and the SSE stream
- `_cache_prices_are_fresh(max_age_secs=90)` returns False if file >30s old or all ts >90s old → triggers collector restart

---

## Key Conventions

### Python
- All DB calls via `read_sql(sql, params)` / `write_df(df, table)` / `upsert_candles(df)` — never raw psycopg2
- `from dotenv import load_dotenv; load_dotenv()` at top of every script (not inside functions)
- Logger via `from utils.logger import get_logger; logger = get_logger("name")`
- Timestamps: DB stores UTC `TIMESTAMPTZ`, Python uses naive local IST for comparisons
- Option symbols: `NIFTY{YYMMDD}{STRIKE}{CE|PE}` e.g. `NIFTY26033022950PE`
- Expiry from `backtest.option_resolver.get_nearest_expiry(date)` — queries symbol_master table

### Frontend (Next.js / TypeScript)
- API calls via `dashboard/lib/api.ts` helper
- SSE stream consumed in `dashboard/app/live/page.tsx`
- All times displayed in IST
- Trade table columns include: Time, Symbol, Type, Strategy, Risk (LOW/MED/HIGH badge), SL ₹, Target ₹, Expiry, Score, ML%, Action

---

## What NOT To Do

- **Don't use `incremental_train.py --days N` where N > 3** — safety guard rejects it; use `main.py train` for full retrains
- **Don't write candles with raw `write_df(..., if_exists='append')`** — use `upsert_candles()` which handles `ON CONFLICT DO NOTHING` on (timestamp, symbol). Bare append causes duplicate key errors.
- **Don't delete today's candles and re-insert inside `upsert_candles()`** — do the DELETE separately before calling upsert if you need a clean replace
- **Don't start `collect_ticks.py` as a plain `&` subprocess from a Bash shell** — it gets SIGHUP when the shell exits. Always use `nohup ... &`
- **Don't store the tick's market timestamp as `ts` in live_price_cache** — for illiquid options, the market timestamp can be hours old, making Flask think prices are stale and killing the collector. Use `datetime.now().isoformat()` (wall-clock receipt time)
- **Don't call `ws_start_streaming(callback)` multiple times with the same callback** — it deduplicates now, but prior reconnect loops created duplicate stream threads. The `_stream_loop` handles reconnection internally; only call `ws_start_streaming()` once at startup
- **Don't assume TrueData sends streaming ticks as plain JSON arrays** — they changed the format to `{"trade": [symbolID, ts, LTP, ...]}` (dict with no symbol name). The symbolID→name map is built in `ws_subscribe()` and used in `_stream_loop`
- **Don't parse `minute_candles` timestamps as naive datetimes** — they're stored as UTC TIMESTAMPTZ; when comparing to `datetime.now()` (IST), the 5:30h offset matters
- **Don't add `print()` statements to Flask routes** — use `logger` (Flask stdout is not shown in production)
- **Don't run Flask in debug mode with the reloader** — the reloader forks processes and double-starts background threads (scanner + tick monitor run twice)

---

## Environment Variables (.env)

```env
# Required
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading
DB_USER=postgres
DB_PASSWORD=postgres

TRUEDATA_USER=your_username
TRUEDATA_PASSWORD=your_password

# Optional overrides
INITIAL_CAPITAL=50000
ATM_RANGE=3              # strikes ±N from ATM to subscribe (default 3 → 14 option contracts)
MAX_SYMBOLS=50           # TrueData plan limit
SCORE_THRESHOLD=0.6
LOG_LEVEL=INFO
MODEL_DIR=models/saved
```

---

## Important Operational Notes

- **Market hours**: 9:15–15:30 IST weekdays. `_is_market_hours()` guards scanner and collector startup
- **NIFTY lot size**: 65 units (1 lot = 65 shares)
- **NIFTY weekly expiry**: Tuesdays (confirmed from symbol_master in DB)
- **ATM strike gap**: 50 points for NIFTY
- **TickCollector buffer**: flushes to `tick_data` every 200 ticks
- **Candle aggregation**: done by `collect_ticks.py` in-memory per minute, written via `upsert_candles()`
- **Model backups**: stored in `models/saved/backups/` with timestamp suffix before each retrain
- **Logs**: `logs/tick_collector_YYYYMMDD.log` (one file per day), `logs/trading.log`
- **Backtest results**: `backtest_results/*.json` and `*.csv`
