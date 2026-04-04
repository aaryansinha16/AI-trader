# Flow Diagrams and Designs

Below is an updated architecture package for the current AI Trader system.

This version is aligned to the code paths that are actively used today, especially:

- `scripts/collect_ticks.py`
- `backend/app.py`
- `scripts/tick_replay_backtest.py`
- `models/predict.py`
- `models/strategy_models.py`
- `models/rl_exit_agent.py`
- `dashboard/app/*`

Important alignment note:

- The current primary runtime is a **paper-trading / research system**
- Live market data, backtesting, model training, and dashboarding are active
- Zerodha execution modules exist in the repo, but they are **not the main active path**
- The main backend flow currently ends in **trade suggestions + paper positions + monitoring**, not exchange execution

# 1. Current High-Level Architecture

Core idea:

The system has two main worlds:

```text
Offline Research Layer (training + replay backtests)

Live Paper-Trading Layer (real-time scanning + monitoring + dashboard)
```

Current top-level system flow:

```mermaid
flowchart TD

A["TrueData WebSocket + REST"] --> B["scripts/collect_ticks.py"]

B --> C1["tick_data (TimescaleDB)"]
B --> C2["minute_candles (TimescaleDB)"]
B --> C3["/tmp/td_live_prices.json"]

C2 --> D["Feature Engineering<br/>compute_all_macro_indicators()"]
D --> E["Regime Detection"]
D --> F["Signal Generation"]

F --> G["Macro Model Probability"]
D --> H["Strategy Model Gate"]
D --> I["Flow Score<br/>(PCR or OBV/MFI fallback)"]

E --> J["Final Trade Scoring + Regime Filters"]
G --> J
H --> J
I --> J
F --> J

J --> K["Trade Suggestions"]
K --> L["Auto or Manual Paper Entry"]
L --> M["Open Paper Positions"]

C3 --> N["Tick Monitor Loop"]
M --> N
N --> O["SL / Target / Trailing / RL Exit / Regime Tighten"]
O --> P["Closed Trades + Journey Persistence"]

K --> Q["Flask API + SSE"]
M --> Q
P --> Q
Q --> R["Next.js Dashboard"]
```

# 2. Data Pipeline Architecture

This diagram focuses on what the data layer does today.

```mermaid
flowchart LR

A["TrueData WebSocket"] --> B["Tick Collector"]
A --> C["REST Backfill / Last Bar Fallback"]

B --> D["tick_data"]
B --> E["In-memory 1m Aggregation"]
E --> F["minute_candles"]

B --> G["Live Price Cache JSON"]
C --> F

F --> H["Feature Computation"]
H --> I["Scanner / Backtest / Training"]
G --> J["Position Monitoring + SSE Prices"]
```

Purpose:

| Component | Current role |
| --- | --- |
| Tick Collector | captures live futures + option ticks |
| Minute Aggregation | builds and upserts 1-minute candles |
| Live Price Cache | shares freshest prices with Flask |
| REST Backfill | fills stale/missing market data |
| Feature Computation | derives macro indicators for scan/backtest/train |

# 3. Current Model Training Pipeline

This is the offline training path that matches the current scripts and saved models.

```mermaid
flowchart TD

A["minute_candles"] --> B["Macro Features"]
B --> C["Macro Label Generation"]
C --> D["Macro Model Training<br/>(XGBoost walk-forward)"]
D --> E["models/saved/macro_model.pkl"]

F["tick_data"] --> G["Micro Features"]
G --> H["Micro Label Generation"]
H --> I["Micro Model Training<br/>(XGBoost walk-forward)"]
I --> J["models/saved/micro_model.pkl"]

K["backtest_results/trades_*.csv"] --> L["Outcome Dataset Builder"]
L --> M["Per-Strategy Outcome Model Training"]
M --> N["models/saved/strategy/*.pkl"]

O["backtest_results/journeys_*.json"] --> P["Journey Loader"]
P --> Q["RL Exit Agent Retraining"]
Q --> R["models/saved/rl_exit_agent.pkl"]
```

Training loop:

```mermaid
flowchart TD

A["Historical / Outcome Data"] --> B["Train Candidate Model"]
B --> C["Walk-Forward Validation"]
C --> D{"Good Enough?"}

D -- "Yes" --> E["Save Model + Backup Existing"]
D -- "No" --> F["Tune Labels / Filters / Parameters"]
F --> B
```

# 4. Current Live Paper-Trading Execution Pipeline

This is the primary active runtime path in `backend/app.py`.

```mermaid
flowchart TD

A["Background Scan Trigger"] --> B["Load Latest NIFTY-I minute_candles"]
B --> C["Compute Macro Indicators"]
C --> D["Detect Market Regime"]
C --> E["Generate Strategy Signals"]

E --> F["Predict Macro Probability"]
E --> G["Predict Strategy Success Probability"]
C --> H["Compute Flow Score"]
D --> I["Apply Regime Strategy Map + Bonuses"]

F --> J["Final Score"]
G --> J
H --> J
I --> J
E --> J

J --> K{"Pass gates and thresholds?"}
K -- "No" --> L["Drop Signal"]
K -- "Yes" --> M["Resolve ATM Option Symbol"]

M --> N["Fetch Entry Premium<br/>(cache -> DB candle -> REST)"]
N --> O["Create Trade Suggestion"]
O --> P{"Auto trade enabled?"}

P -- "Yes" --> Q["Open Paper Position"]
P -- "No" --> R["Wait for Manual Entry"]
R --> Q
```

# 5. Position Monitoring and Exit Logic

This is the active trade-management path in the Flask backend.

```mermaid
flowchart TD

A["Open Paper Position"] --> B["Tick Monitor Loop (1s)"]
B --> C["Read /tmp/td_live_prices.json"]
C --> D{"Fresh option price available?"}

D -- "Yes" --> E["Use bid / live price"]
D -- "No" --> F["REST last-bar fallback"]
F --> E

E --> G["Update unrealised P&L"]
G --> H["Record journey point"]
H --> I["Breakeven / Trailing / Regime Tightening"]
I --> J["Optional RL Exit Decision"]

J --> K{"Exit condition hit?"}
K -- "No" --> B
K -- "Yes" --> L["Close Position"]
L --> M["Persist closed trade JSONL"]
M --> N["Expose via API / SSE / Dashboard"]
```

Current exit sources:

```text
SL_HIT
TARGET_HIT
TRAILING_SL
RL_EXIT
TIMEOUT
EOD_CLOSE
```

# 6. Current Decision Engine Logic

Scoring logic as implemented today:

```mermaid
flowchart TD

A["Raw Strategy Signal"] --> E["Score Composer"]
B["Directional Macro Probability"] --> E
C["Flow Score"] --> E
D["Regime Bonus / Filter"] --> E

E --> F["Final Score"]
F --> G["Strategy-specific gates"]
G --> H["Suggestion or Reject"]
```

Primary score formula:

```text
final_score =
0.5 * directional_prob
+ 0.3 * flow_score
+ 0.2 * technical_strength
+ regime_bonus
```

Important current behavior:

- `strategy_prob` is used mainly as a **gate**, not as a weighted score term
- PUT directional confidence is computed as `1 - ml_prob`
- regime-specific thresholds are looser/tighter depending on market regime
- some strategies have extra regime restrictions before they are allowed through

# 7. Current Backtesting Architecture

This aligns with `scripts/tick_replay_backtest.py`.

```mermaid
flowchart TD

A["Historical tick_data"] --> B["Replay by trading day"]
B --> C["Build minute candles during replay"]
C --> D["Compute indicators"]
D --> E["Detect regime"]
D --> F["Generate signals"]

F --> G["Macro model"]
F --> H["Strategy model gate"]
D --> I["Flow score + optional enrichments"]
E --> J["Final score + risk-profile rules"]

J --> K{"Trade qualifies?"}
K -- "No" --> L["Continue replay"]
K -- "Yes" --> M["Resolve option contract + premium path"]

M --> N["Simulate entry/exit with spread assumptions"]
N --> O["Dynamic SL / Target / Trailing / RL Exit"]
O --> P["Trade record + journey"]
P --> Q["CSV / JSON / Report / Equity Curve"]
```

Metrics produced today:

```text
Trade list
P&L
Win rate
Profit factor
Avg win / avg loss
Equity curve
Journey traces per trade
```

# 8. Current Low-Level Architecture (LLD)

```mermaid
flowchart TD

subgraph DataLayer["Data Layer"]
  A1["TrueDataAdapter"]
  A2["collect_ticks.py"]
  A3["database/db.py"]
  A4["tick_data / minute_candles / option_chain / symbol_master"]
  A5["/tmp/td_live_prices.json"]
end

subgraph IntelligenceLayer["Intelligence Layer"]
  B1["features/indicators.py"]
  B2["strategy/regime_detector.py"]
  B3["strategy/signal_generator.py"]
  B4["models/predict.py"]
  B5["models/strategy_models.py"]
  B6["models/rl_exit_agent.py"]
end

subgraph RuntimeLayer["Runtime Layer"]
  C1["backend/app.py scanner"]
  C2["trade suggestions"]
  C3["paper positions"]
  C4["tick monitor"]
  C5["closed trade persistence"]
end

subgraph PresentationLayer["Presentation Layer"]
  D1["Flask REST + SSE"]
  D2["Next.js dashboard"]
end

subgraph OfflineResearch["Offline Research"]
  E1["scripts/tick_replay_backtest.py"]
  E2["scripts/incremental_train.py"]
  E3["scripts/train_outcome_models.py"]
  E4["scripts/train_rl_on_journeys.py"]
end

A1 --> A2
A2 --> A3
A3 --> A4
A2 --> A5

A4 --> B1
B1 --> B2
B1 --> B3
B1 --> B4
B1 --> B5
B6 --> C4

B2 --> C1
B3 --> C1
B4 --> C1
B5 --> C1
A5 --> C4
C1 --> C2
C2 --> C3
C3 --> C4
C4 --> C5

C2 --> D1
C3 --> D1
C5 --> D1
D1 --> D2

A4 --> E1
A4 --> E2
E1 --> E3
E1 --> E4
```

# 9. Service-Level Architecture

If the current system is further modularized, this is the split that best matches the current codebase.

```mermaid
flowchart LR

A["Market Data Service"]
B["Storage Service"]
C["Feature + Signal Service"]
D["Scoring Service"]
E["Paper Execution Service"]
F["Monitoring Service"]
G["Research / Backtest Service"]
H["Dashboard API Service"]

A --> B
B --> C
C --> D
D --> E
E --> F

B --> G
C --> G
D --> G
F --> H
E --> H
D --> H
```

# 10. Current Storage Design

### Tick Data

```text
tick_data

timestamp
symbol
price
volume
oi
bid_price
ask_price
bid_qty
ask_qty
```

### Minute Candles

```text
minute_candles

timestamp
symbol
open
high
low
close
volume
vwap
oi
```

### Options / Symbol Metadata

```text
option_chain
symbol_master
```

### Trade Outputs

```text
trade_log                  # DB-level trade history / backtest usage
paper_trades/*.jsonl       # persisted Flask paper trades
backtest_results/*.csv
backtest_results/*.json
backtest_results/journeys_*.json
```

### Model Artifacts

```text
models/saved/macro_model.pkl
models/saved/micro_model.pkl
models/saved/strategy/*.pkl
models/saved/rl_exit_agent.pkl
models/saved/backups/YYYYMMDD/*
```

# 11. Actual Project Runtime Loop

```mermaid
flowchart TD

A["Every scan interval"] --> B["Load latest minute candles"]
B --> C["Compute indicators"]
C --> D["Detect regime"]
C --> E["Generate candidate signals"]
E --> F["Score and filter"]

F --> G{"Qualified?"}
G -- "No" --> H["Wait next scan"]
G -- "Yes" --> I["Create suggestion"]
I --> J{"Auto-enter?"}
J -- "Yes" --> K["Open paper trade"]
J -- "No" --> L["Dashboard manual enter"]

K --> M["Tick monitor every 1s"]
L --> M
M --> N["Manage exits + record journey"]
N --> O["Push state via SSE"]
H --> O
```

Current cadence:

```text
Scanner: ~30 to 60 seconds depending on path
Tick monitor: 1 second
Live cache flush: 1 second
Journey capture: ~5 seconds
```

# 12. Dashboard and API Flow

```mermaid
flowchart TD

A["Flask state"] --> B["/api/state"]
A --> C["/api/stream (SSE)"]
A --> D["/api/paper/*"]
A --> E["/api/backtest/*"]
A --> F["/api/market/*"]

C --> G["dashboard/app/live/page.tsx"]
D --> H["dashboard/app/trades/page.tsx"]
E --> I["dashboard/app/backtest/page.tsx"]
F --> J["dashboard/app/charts/page.tsx"]
```

# 13. Optional / Secondary Paths in the Repo

These exist in the repo but are not the primary current system path:

```text
execution/order_manager.py
execution/broker_adapter.py
models/dqn_exit_agent.py
models/model_registry.py
main.py live execution path
```

They should be treated as optional or future-facing unless the runtime is explicitly switched to them.

# 14. Final Current System Summary

The current AI Trader system is best described as:

```text
Live TrueData ingestion
+ minute-candle feature pipeline
+ rule-based strategy generation
+ ML scoring and strategy gating
+ RL-assisted paper-trade exits
+ tick-replay backtesting
+ dashboard-driven monitoring
```

That is the architecture the Mermaid diagrams in this document now reflect.
