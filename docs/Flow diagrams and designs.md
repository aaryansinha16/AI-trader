# Flow diagrams and designs

Below is a **complete architecture package** for the AI intraday trading system we designed together.

I’ll provide:

1️⃣ **High Level Design (HLD)**

2️⃣ **System Data Flow**

3️⃣ **Model Training Pipeline**

4️⃣ **Live Trading Execution Pipeline**

5️⃣ **Low Level Design (LLD)**

6️⃣ **Database Schema**

7️⃣ **Complete Mermaid Diagrams**

Everything here can be pasted into **Mermaid renderers / Notion / GitHub / Obsidian / draw.io**.

# 1. High Level Architecture (HLD)

Core idea:

Your system has **two worlds**:

```
Offline Intelligence Layer (training/backtesting)

Live Trading Layer (real-time decisions)
```

Data pipeline:

```mermaid
flowchart TD

A[TrueData Tick Stream] --> B[Tick Data Collector]

B --> C[Tick Database]

C --> D[Aggregation Engine]

D --> E1[1 Second Candles]
D --> E2[1 Minute Candles]
D --> E3[5 Minute Candles]

E2 --> F[Feature Engineering Engine]

F --> G1[Technical Indicators]
F --> G2[Options Flow Signals]
F --> G3[Volume Signals]

G1 --> H[ML Prediction Engine]
G2 --> H
G3 --> H

H --> I[Trade Scoring Engine]

I --> J[Risk Manager]

J --> K[Order Execution Engine]

K --> L[Broker API - Zerodha]

L --> M[Exchange]
```

# 2. Data Pipeline Architecture

This diagram focuses only on **data ingestion and processing**.

```mermaid
flowchart LR

A[TrueData WebSocket] --> B[Tick Collector Service]

B --> C[Tick Storage]

C --> D[Aggregation Service]

D --> E1[1s Bars]
D --> E2[1m Bars]
D --> E3[5m Bars]

E2 --> F[Feature Generator]

F --> G[Feature Store]

G --> H1[Backtesting Engine]
G --> H2[ML Training Engine]
G --> H3[Live Trading Engine]
```

Purpose:

| Component | Role |
| --- | --- |
| Tick Collector | capture streaming ticks |
| Aggregation | generate multiple timeframes |
| Feature Generator | compute indicators |
| Feature Store | ML dataset |

# 3. Machine Learning Training Pipeline

This pipeline runs **offline**.

```mermaid
flowchart TD

A[Historical Market Data] --> B[Feature Engineering]

B --> C[Feature Dataset]

C --> D[Label Generation]

D --> E[Training Dataset]

E --> F[Model Training]

F --> G[Model Validation]

G --> H[Model Registry]

H --> I[Production Model]
```

Training loop:

```mermaid
flowchart TD

A[Historical Dataset] --> B[Train Model]

B --> C[Test Model]

C --> D{Performance OK?}

D -- Yes --> E[Deploy Model]

D -- No --> B
```

# 4. Live Trading Execution Pipeline

Real-time system.

```mermaid
flowchart TD

A[Market Tick Arrives] --> B[Update Indicators]

B --> C[Generate Strategy Signals]

C --> D[Options Flow Detector]

D --> E[ML Probability Prediction]

E --> F[Trade Scoring]

F --> G{Score > Threshold?}

G -- Yes --> H[Risk Manager]

H --> I[Place Order]

I --> J[Broker API]

J --> K[Exchange]

G -- No --> L[Ignore Signal]
```

# 5. Decision Engine Logic

Signal scoring logic:

```mermaid
flowchart TD

A[Technical Signal] --> D[Score Calculator]

B[Options Flow Signal] --> D

C[ML Probability] --> D

D --> E[Final Trade Score]

E --> F{Score > 0.6 ?}

F -->|Yes| G[Execute Trade]

F -->|No| H[Reject]
```

Score formula:

```
TradeScore =
0.5 * ML_Probability
+ 0.3 * OptionsFlowScore
+ 0.2 * TechnicalSignalStrength
```

---

# 6. Backtesting Architecture

Used to validate strategies.

```mermaid
flowchart TD

A[Historical Data] --> B[Replay Engine]

B --> C[Strategy Engine]

C --> D[Signal Generator]

D --> E[Trade Simulator]

E --> F[Portfolio Manager]

F --> G[Performance Metrics]
```

Metrics calculated:

```
Win rate
Profit factor
Sharpe ratio
Max drawdown
Expectancy
```

---

# 7. Low Level Architecture (LLD)

System components.

```mermaid
flowchart TD

subgraph Data Layer
A1[Tick Collector]
A2[Aggregation Engine]
A3[Feature Engine]
end

subgraph Intelligence Layer
B1[Technical Indicator Module]
B2[Options Flow Detector]
B3[ML Prediction Engine]
end

subgraph Decision Layer
C1[Trade Scoring Engine]
C2[Risk Manager]
end

subgraph Execution Layer
D1[Order Manager]
D2[Broker Adapter]
end

subgraph Storage
E1[Tick Database]
E2[Feature Store]
E3[Trade Logs]
end

A1 --> E1
A1 --> A2
A2 --> A3
A3 --> E2

A3 --> B1
A3 --> B2
A3 --> B3

B1 --> C1
B2 --> C1
B3 --> C1

C1 --> C2
C2 --> D1
D1 --> D2
D2 --> BrokerAPI
```

# 8. Service-Level Architecture

If you modularize the system:

```mermaid
flowchart LR

A[Market Data Service]
B[Feature Service]
C[ML Service]
D[Signal Service]
E[Execution Service]
F[Risk Service]
G[Backtest Service]

A --> B
B --> C
C --> D
D --> F
F --> E

B --> G
C --> G
```

# 9. Database Schema Design

### Tick Data Table

```
tick_data

timestamp
symbol
price
volume
bid_price
ask_price
bid_qty
ask_qty
```

---

### Minute Candle Table

```
minute_candles

timestamp
symbol
open
high
low
close
volume
vwap
```

---

### Feature Table

```
features

timestamp
symbol
rsi
ema20
ema50
vwap_dist
volume_ratio
oi_change
pcr
atr
```

---

### Trade Log Table

```
trade_log

trade_id
timestamp
symbol
entry_price
exit_price
stop_loss
target
result
pnl
```

---

# 10. Project Folder Structure

```
ai_trading_system

data/
    tick_collector.py
    aggregation.py

features/
    indicators.py
    feature_engine.py

models/
    train_model.py
    predictor.py

strategy/
    signal_generator.py
    options_flow_detector.py

risk/
    risk_manager.py

execution/
    broker_adapter.py
    order_manager.py

backtest/
    backtest_engine.py

database/
    schema.sql

main.py
```

---

# 11. System Runtime Loop

```mermaid
flowchart TD

A[Tick Stream] --> B[Update Market Data]

B --> C[Update Indicators]

C --> D[Generate Signals]

D --> E[ML Prediction]

E --> F[Trade Ranking]

F --> G{Trade Valid?}

G -->|Yes| H[Execute Order]

G -->|No| I[Wait Next Tick]
```

Loop interval:

```
30–60 seconds
```

---

# 12. Final System Overview

```mermaid
flowchart TD

subgraph Market
A[TrueData Tick Feed]
end

subgraph DataPipeline
B[Tick Collector]
C[Tick DB]
D[Aggregation Engine]
end

subgraph Intelligence
E[Feature Engine]
F[Options Flow Detector]
G[ML Prediction]
end

subgraph Decision
H[Trade Scoring]
I[Risk Manager]
end

subgraph Execution
J[Order Manager]
K[Zerodha API]
end

A --> B
B --> C
C --> D
D --> E
E --> F
E --> G
F --> H
G --> H
H --> I
I --> J
J --> K
```

---

# 13. Final Concept

Your system ultimately becomes a **mini quant trading engine** with:

```
Market Data
+ Technical Signals
+ Options Flow Analysis
+ Machine Learning
+ Risk Control
+ Automated Execution
```

All running locally on your laptop.