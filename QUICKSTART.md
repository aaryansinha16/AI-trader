# Quick Start Guide

Get the AI Trading System up and running in 10 minutes.

## 🚀 Installation (5 minutes)

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/aaryansinha16/ai-trader.git
cd ai-trader

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Test with Mock Data (No Setup Required)

```bash
# Run the mock pipeline - generates synthetic data and tests all features
python main.py mock
```

You should see output like:
```
2026-03-16 14:52:38 | INFO | MODE: MOCK DATA – generating synthetic dataset
2026-03-16 14:52:40 | INFO | Mock data generated: 93750 minute bars, 562798 ticks
2026-03-16 14:52:40 | INFO | Mock pipeline complete. All layers functional.
```

### 3. Run a Backtest

```bash
# Test the full strategy pipeline with mock data
python main.py backtest
```

This will:
- Generate mock market data
- Compute technical indicators
- Generate trading signals
- Simulate trades with SL/target
- Display performance metrics

Expected output:
```
==================================================
BACKTEST RESULTS
==================================================
  Total trades:  625
  Wins:          260
  Win rate:      41.6%
  Gross PnL:     ₹-8,990.97
  Profit factor: 0.94
==================================================
```

**Note:** Negative PnL on mock data is expected - it's random synthetic data. Real performance requires real market data and trained models.

---

## 📊 Production Setup (30 minutes)

### 1. Install TimescaleDB

**macOS:**
```bash
brew install timescaledb
brew services start postgresql
```

**Ubuntu:**
```bash
sudo apt install postgresql-14 timescaledb-2-postgresql-14
sudo systemctl start postgresql
```

**Docker:**
```bash
docker run -d --name timescaledb -p 5432:5432 \
  -e POSTGRES_PASSWORD=password \
  timescale/timescaledb:latest-pg14
```

### 2. Create Database

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database and enable TimescaleDB
CREATE DATABASE trading_db;
\c trading_db
CREATE EXTENSION IF NOT EXISTS timescaledb;
\q

# Load schema
psql -U postgres -d trading_db -f database/schema.sql
```

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your credentials
nano .env  # or use your preferred editor
```

Required settings:
```bash
# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/trading_db

# TrueData (get from https://www.truedata.in/)
TRUEDATA_USERNAME=your_username
TRUEDATA_PASSWORD=your_password

# Kite Connect (get from https://kite.trade/)
KITE_API_KEY=your_api_key
KITE_ACCESS_TOKEN=your_access_token

# Trading Parameters
INITIAL_CAPITAL=50000
RISK_PER_TRADE=0.01
MAX_TRADES_PER_DAY=5
```

### 4. Load Historical Data

```bash
# This will fetch 6 months of 1-minute bars + 5 days of tick data
python main.py ingest
```

Expected duration: 10-20 minutes depending on your internet speed.

### 5. Train ML Models

```bash
# Train both Macro and Micro models
python main.py train
```

This will:
- Load features from database
- Perform walk-forward validation (5 splits)
- Train final models on full dataset
- Save models to `models/` directory

Expected output:
```
2026-03-16 15:30:00 | INFO | Training final macro model on full dataset...
2026-03-16 15:30:45 | INFO | Macro model saved to models/macro_model.pkl
2026-03-16 15:31:00 | INFO | Training final micro model on full dataset...
2026-03-16 15:31:30 | INFO | Micro model saved to models/micro_model.pkl
```

### 6. Go Live! 🎉

```bash
# Start the live trading loop
python main.py live
```

The system will:
- Check market hours (9:15 AM - 3:30 PM IST)
- Fetch real-time data every 30-60 seconds
- Detect market regime
- Generate signals from active strategies
- Score trades using ML + options flow + technicals
- Execute top 3 trades via Kite Connect
- Manage risk (SL/target, daily limits)

---

## 🎯 Your First Trade

Once live mode is running, here's what happens:

1. **Market Scan** (every 30-60s)
   ```
   Fetching data for NIFTY, BANKNIFTY...
   Regime: TRENDING_BULL
   Active strategies: vwap_momentum_breakout
   ```

2. **Signal Generation**
   ```
   Signal: vwap_momentum_breakout → CALL for NIFTY (strength=0.75)
   ```

3. **ML Prediction**
   ```
   Macro prob: 0.68, Micro prob: 0.72, Combined: 0.69
   ```

4. **Trade Scoring**
   ```
   #1 NIFTY CALL score=0.71 (ml=0.69, flow=0.65, tech=0.75)
   ```

5. **Risk Validation**
   ```
   Risk approved: NIFTY CALL qty=2, entry=150, SL=135, target=180
   ```

6. **Order Execution**
   ```
   Entry order placed: ORD-000001 → broker=240316000001
   SL order placed for NIFTY @ 135
   Target order placed for NIFTY @ 180
   ```

---

## 🛡️ Safety Checklist

Before going live with real money:

- [ ] Test with mock data (`python main.py mock`)
- [ ] Run backtests on historical data (`python main.py backtest`)
- [ ] Verify database connection works
- [ ] Confirm TrueData API is active
- [ ] Test Kite Connect API (place a test order)
- [ ] Start with small capital (₹10,000-₹25,000)
- [ ] Monitor first few trades manually
- [ ] Check logs regularly (`logs/ai_trader.log`)
- [ ] Set up alerts (Telegram/email) - optional but recommended

---

## 📱 Monitoring

### Check Logs
```bash
# Real-time log monitoring
tail -f logs/ai_trader.log

# Search for errors
grep ERROR logs/ai_trader.log
```

### Database Queries
```sql
-- Check recent trades
SELECT * FROM trade_log ORDER BY entry_time DESC LIMIT 10;

-- Daily performance
SELECT * FROM daily_performance ORDER BY date DESC LIMIT 7;

-- Active positions
SELECT symbol, direction, entry_price, quantity 
FROM trade_log 
WHERE exit_time IS NULL;
```

### System Status
```bash
# Check if TimescaleDB is running
pg_isready

# Check Python processes
ps aux | grep python

# Check disk space (logs can grow)
df -h
```

---

## ❓ Troubleshooting

### "No module named 'xgboost'"
```bash
pip install -r requirements.txt
```

### "Connection refused" (Database)
```bash
# Check if PostgreSQL is running
brew services list  # macOS
systemctl status postgresql  # Linux

# Restart if needed
brew services restart postgresql
```

### "TrueData authentication failed"
- Verify credentials in `.env`
- Check if subscription is active
- Test login at https://www.truedata.in/

### "Kite access token expired"
- Kite tokens expire daily
- Generate new token from https://kite.trade/
- Update `KITE_ACCESS_TOKEN` in `.env`

### "No ML models found"
```bash
# Train models first
python main.py train
```

### "Market closed"
- System only trades 9:15 AM - 3:30 PM IST
- Outside hours, it will sleep and wait

---

## 🎓 Next Steps

1. **Read the Documentation**
   - [Architecture Overview](docs/ARCHITECTURE.md)
   - [Product Vision](docs/Product_vision.md)
   - [API Reference](docs/API_REFERENCE.md)

2. **Customize Strategies**
   - Edit `strategy/signal_generator.py`
   - Adjust parameters in `config/settings.py`

3. **Add More Symbols**
   - Update `SYMBOLS` in `config/settings.py`
   - Re-run `python main.py ingest`

4. **Optimize Models**
   - Tune hyperparameters in `models/train_model.py`
   - Experiment with different features

5. **Build a Dashboard**
   - Use Streamlit or Grafana
   - Visualize trades, P&L, metrics

---

## 💡 Pro Tips

- **Start Small**: Begin with 1-2 symbols and small capital
- **Monitor Closely**: Watch first 10-20 trades manually
- **Keep Logs**: Logs are your best debugging tool
- **Retrain Regularly**: Update models weekly/monthly with fresh data
- **Backtest Changes**: Always backtest before deploying strategy changes
- **Paper Trade First**: Test in dry-run mode before going live

---

## 📞 Need Help?

- **Issues**: https://github.com/yourusername/ai-trader/issues
- **Discussions**: https://github.com/yourusername/ai-trader/discussions
- **Email**: your.email@example.com

---

**Happy Trading! 📈**
