# AI-Powered Options Trading System

> **Intraday NSE F&O options trading system** combining dual-model ML architecture, institutional options flow analysis, and regime-adaptive strategies for NIFTY and BANKNIFTY.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Overview

This is a **production-ready algorithmic trading system** designed for Indian equity derivatives (NSE F&O). It processes tick-level and minute-level market data through a sophisticated ML pipeline to generate high-probability intraday options trades.

### Key Features

- **Dual-Model ML Architecture**
  - **Macro Model**: XGBoost trained on 6 months of 1-minute candle data (18 technical indicators)
  - **Micro Model**: XGBoost trained on 5 days of tick data (order flow microstructure)
  - Walk-forward validation to prevent overfitting

- **Institutional Options Flow Detection**
  - Long Build Up, Short Covering, Long Unwinding, Short Build Up
  - Gamma Pinning detection
  - Put-Call Ratio (PCR) analysis
  - OI change tracking and volume spike detection

- **Market Regime Detection**
  - Trending Bull/Bear, Sideways, High/Low Volatility
  - Regime-adaptive strategy selection
  - EMA trend + ATR percentile + range compression

- **Three Core Strategies**
  1. **VWAP Momentum Breakout** — Bullish breakouts (Buy ATM Call)
  2. **Bearish Momentum** — Bearish breakdowns (Buy ATM Put)
  3. **Mean Reversion** — Extreme RSI + Bollinger Band touches

- **Composite Trade Scoring**
  ```
  Trade Score = 0.5 × ML Probability + 0.3 × Options Flow + 0.2 × Technical Strength
  ```
  Top 3 trades selected per scan cycle (30-60s)

- **Robust Risk Management**
  - 1% risk per trade
  - Max 5 trades/day
  - 5% daily loss cap
  - ATR-based stop loss and targets
  - Exchange-managed SL-M orders

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                              │
│  TrueData API → Tick Collector → Aggregation Engine            │
│  (6mo 1m bars + 5d ticks) → TimescaleDB → Feature Store        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      FEATURE LAYER                              │
│  Macro: RSI, MACD, EMA, VWAP, Bollinger, ATR, Volume, PCR, IV  │
│  Micro: Bid-Ask Spread, Order Imbalance, Tick Momentum         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    INTELLIGENCE LAYER                           │
│  Regime Detector → Strategy Engine → Options Flow Detector     │
│  Macro Model (1m) + Micro Model (tick) → Predictor             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     EXECUTION LAYER                             │
│  Trade Scorer → Risk Manager → Order Manager → Kite Connect    │
│  (Entry + SL-M + Target orders)                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Data Requirements

| Data Type | Source | Duration | Purpose |
|-----------|--------|----------|---------|
| **1-minute candles** | TrueData | 6 months | Macro Model training |
| **Tick data** | TrueData | 5 days | Micro Model training |
| **Option chain** | TrueData | Real-time | Options flow analysis |
| **Live ticks** | Kite WebSocket | Real-time | Trade execution |

**Symbols**: NIFTY, BANKNIFTY (expandable to FINNIFTY, MIDCPNIFTY)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- PostgreSQL with TimescaleDB extension
- TrueData API subscription
- Zerodha Kite Connect API credentials

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-trader.git
cd ai-trader

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your credentials
```

### Configuration

Edit `.env`:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/trading_db

# TrueData
TRUEDATA_USERNAME=your_username
TRUEDATA_PASSWORD=your_password

# Kite Connect
KITE_API_KEY=your_api_key
KITE_ACCESS_TOKEN=your_access_token

# Trading Parameters
INITIAL_CAPITAL=50000
RISK_PER_TRADE=0.01
MAX_TRADES_PER_DAY=5
MAX_DAILY_LOSS=0.05
```

### Database Setup

```bash
# Install TimescaleDB (macOS)
brew install timescaledb

# Initialize database
psql -U postgres -c "CREATE DATABASE trading_db;"
psql -U postgres -d trading_db -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"

# Run schema
psql -U postgres -d trading_db -f database/schema.sql
```

---

## 💻 Usage

### 5 Operating Modes

#### 1. Mock Mode (Development/Testing)
```bash
python main.py mock
```
Generates synthetic data and runs the full feature pipeline in-memory. No database required.

#### 2. Ingest Mode (Data Loading)
```bash
python main.py ingest
```
Loads 6 months of 1-minute bars + 5 days of tick data from TrueData into TimescaleDB.

#### 3. Train Mode (ML Training)
```bash
python main.py train
```
Trains both Macro and Micro models using walk-forward validation. **Only use with real TrueData.**

#### 4. Backtest Mode (Strategy Testing)
```bash
python main.py backtest
```
Runs full strategy backtest on mock data. Tests signal generation, scoring, and SL/target simulation.

**Results are automatically exported to `backtest_results/` directory in 3 formats:**
- **CSV** (`*_trades.csv`): All trades with entry/exit prices, P&L, scores - open in Excel
- **JSON** (`*_results.json`): Full results including summary metrics and trade details - for programmatic analysis
- **TXT** (`*_report.txt`): Human-readable formatted report with summary and trade-by-trade breakdown

#### 5. Live Mode (Production Trading)
```bash
python main.py live
```
Real-time trading loop:
```
while market_open:
    fetch data → update indicators → detect regime
    → generate signals → compute options flow → run ML model
    → rank trades → execute top 3 trades
    sleep 30-60s
```

---

## 📈 Performance Metrics

The backtest engine computes:

- **Win Rate** — % of profitable trades
- **Profit Factor** — Gross wins / Gross losses
- **Sharpe Ratio** — Risk-adjusted returns
- **Max Drawdown** — Largest peak-to-trough decline
- **Expectancy** — Average P&L per trade
- **Avg Win / Avg Loss** — Trade distribution

Example output:
```
==================================================
BACKTEST RESULTS
==================================================
  Total trades:  625
  Wins:          260
  Losses:        365
  Win rate:      41.6%
  Gross PnL:     ₹-8,990.97
  Profit factor: 0.94
  Max drawdown:  ₹14,355.88
  Sharpe ratio:  -0.47
  Expectancy:    ₹-14.39/trade
  Avg win:       ₹557.37
  Avg loss:      ₹-421.66
==================================================
```

---

## 🛡️ Risk Management

| Rule | Value |
|------|-------|
| Risk per trade | 1% of capital |
| Max trades/day | 5 |
| Max daily loss | 5% of capital |
| Stop loss | 1.5 × ATR |
| Target | 2.0 × ATR |
| Position sizing | Risk amount / Stop distance |

**Stop Loss Type**: Exchange-managed SL-M orders (no manual monitoring required)

---

## 📁 Project Structure

```
ai-trader/
├── backtest/           # Backtesting engine
│   └── backtest_engine.py
├── config/             # Configuration & settings
│   └── settings.py
├── data/               # Data ingestion & aggregation
│   ├── truedata_adapter.py
│   ├── market_stream.py
│   ├── tick_collector.py
│   ├── aggregator.py
│   └── mock_data.py
├── database/           # TimescaleDB schema & connection
│   ├── schema.sql
│   └── db.py
├── docs/               # Documentation
├── execution/          # Order management & broker API
│   ├── order_manager.py
│   └── broker_adapter.py
├── features/           # Feature engineering
│   ├── indicators.py
│   ├── micro_features.py
│   └── feature_engine.py
├── models/             # ML training & prediction
│   ├── train_model.py
│   ├── predict.py
│   └── model_registry.py
├── risk/               # Risk management
│   └── risk_manager.py
├── strategy/           # Trading strategies & signals
│   ├── regime_detector.py
│   ├── signal_generator.py
│   ├── options_flow_detector.py
│   └── trade_scorer.py
├── utils/              # Helpers & logging
│   ├── logger.py
│   └── helpers.py
├── main.py             # Entry point
├── requirements.txt
└── README.md
```

---

## 🔧 Technical Stack

- **Language**: Python 3.13+
- **Database**: PostgreSQL + TimescaleDB
- **ML Framework**: XGBoost, LightGBM, scikit-learn
- **Data Processing**: pandas, NumPy
- **Technical Analysis**: pandas-ta
- **Broker API**: Kite Connect (Zerodha)
- **Data Provider**: TrueData API

---

## 📝 Development Roadmap

- [x] Layer 1: Infrastructure (DB, config, logging)
- [x] Layer 2: Data pipeline (TrueData, Kite, aggregation)
- [x] Layer 3: Feature engineering (macro + micro)
- [x] Layer 4: ML training & prediction (dual-model)
- [x] Layer 5: Strategy execution & risk management
- [ ] Layer 6: Dashboard & monitoring (Grafana/Streamlit)
- [ ] Layer 7: Telegram alerts & notifications
- [ ] Layer 8: Multi-symbol expansion (FINNIFTY, MIDCPNIFTY)
- [ ] Layer 9: Advanced ML (LSTM, Transformers)
- [ ] Layer 10: Portfolio optimization

---

## ⚠️ Important Notes

### Do NOT Train on Mock Data
Mock data is synthetic noise with no real market patterns. Training ML models on it will produce garbage results. Always wait for real TrueData before running `python main.py train`.

### Paper Trading First
Before going live with real capital:
1. Run backtests on historical data
2. Test in paper trading mode (dry-run)
3. Verify all components work end-to-end
4. Start with small capital (₹10,000-₹25,000)

### Market Hours
System automatically checks market hours (9:15 AM - 3:30 PM IST). Outside market hours, the live loop sleeps.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚖️ Disclaimer

**This software is for educational purposes only.** Trading in derivatives involves substantial risk of loss. Past performance is not indicative of future results. The authors and contributors are not responsible for any financial losses incurred through the use of this software.

**Always:**
- Understand the risks before trading
- Start with paper trading
- Never risk more than you can afford to lose
- Comply with all applicable regulations

---

## 📧 Contact

For questions, issues, or collaboration:
- Open an issue on GitHub
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- **TrueData** for market data API
- **Zerodha** for Kite Connect API
- **TimescaleDB** for time-series database
- **pandas-ta** for technical analysis library

---

**Built with ❤️ for algorithmic traders**
