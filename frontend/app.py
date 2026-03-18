"""
Paper Trading Dashboard
───────────────────────
A simple Flask web app that shows live paper trading status.

Features:
  - Current market regime and price
  - Trade suggestions with ML scores
  - Trade log with P&L
  - System status (models loaded, DB connected, etc.)

Run: python frontend/app.py
Open: http://localhost:5050
"""

import os
import sys
import json
import threading
import time
from datetime import datetime, date, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
from flask import Flask, render_template_string, jsonify

from database.db import read_sql, get_engine
from features.indicators import compute_all_macro_indicators
from strategy.signal_generator import generate_signals
from strategy.regime_detector import RegimeDetector, get_strategies_for_regime, MarketRegime
from models.predict import Predictor
from models.strategy_models import StrategyPredictor
from backtest.option_resolver import get_nearest_expiry, get_days_to_expiry
from config.settings import (
    WEIGHT_ML_PROBABILITY, WEIGHT_OPTIONS_FLOW, WEIGHT_TECHNICAL_STRENGTH,
    SCORE_THRESHOLD,
)
from utils.logger import get_logger

logger = get_logger("dashboard")

app = Flask(__name__)

# ── Global State ──────────────────────────────────────────────────────────────
state = {
    "status": "initializing",
    "last_scan": None,
    "last_price": 0,
    "regime": "UNKNOWN",
    "models_loaded": False,
    "strategy_models_loaded": [],
    "db_connected": False,
    "trade_suggestions": [],
    "scan_count": 0,
    "signals_checked": 0,
    "trades_today": 0,
}

predictor = Predictor()
strategy_predictor = StrategyPredictor()
regime_detector = RegimeDetector()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Trader - Paper Trading</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e1e4e8; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1 { color: #58a6ff; margin-bottom: 5px; font-size: 24px; }
        .subtitle { color: #8b949e; margin-bottom: 20px; font-size: 14px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 20px; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
        .card h3 { color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
        .card .value { font-size: 28px; font-weight: 700; }
        .green { color: #3fb950; }
        .red { color: #f85149; }
        .yellow { color: #d29922; }
        .blue { color: #58a6ff; }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }
        .badge-green { background: #0d1117; border: 1px solid #3fb950; color: #3fb950; }
        .badge-red { background: #0d1117; border: 1px solid #f85149; color: #f85149; }
        .badge-yellow { background: #0d1117; border: 1px solid #d29922; color: #d29922; }
        .badge-blue { background: #0d1117; border: 1px solid #58a6ff; color: #58a6ff; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th { text-align: left; padding: 8px 12px; border-bottom: 2px solid #30363d; color: #8b949e; font-size: 12px; text-transform: uppercase; }
        td { padding: 8px 12px; border-bottom: 1px solid #21262d; font-size: 13px; }
        tr:hover { background: #1c2128; }
        .status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; }
        .dot-green { background: #3fb950; }
        .dot-red { background: #f85149; }
        .dot-yellow { background: #d29922; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        .paper-badge { background: #d29922; color: #0d1117; padding: 4px 12px; border-radius: 4px; font-weight: 700; font-size: 12px; }
        .refresh-info { color: #484f58; font-size: 12px; }
        .pnl-positive { color: #3fb950; font-weight: 600; }
        .pnl-negative { color: #f85149; font-weight: 600; }
    </style>
    <script>
        function refreshData() {
            fetch('/api/state')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('status').innerHTML = data.status === 'scanning' ?
                        '<span class="dot-green status-dot"></span>Live' :
                        '<span class="dot-yellow status-dot"></span>' + data.status;
                    document.getElementById('price').textContent = data.last_price ? '₹' + data.last_price.toLocaleString('en-IN', {maximumFractionDigits: 1}) : '--';
                    document.getElementById('regime').textContent = data.regime;
                    document.getElementById('regime').className = 'value ' +
                        (data.regime.includes('BULL') ? 'green' : data.regime.includes('BEAR') ? 'red' : 'yellow');
                    document.getElementById('scans').textContent = data.scan_count;
                    document.getElementById('signals').textContent = data.signals_checked;
                    document.getElementById('trades-today').textContent = data.trades_today;
                    document.getElementById('last-scan').textContent = data.last_scan || '--';
                    document.getElementById('models').innerHTML =
                        (data.models_loaded ? '<span class="badge badge-green">ML Loaded</span> ' : '<span class="badge badge-red">No ML</span> ') +
                        data.strategy_models_loaded.map(s => '<span class="badge badge-blue">' + s + '</span>').join(' ');

                    // Trade table
                    let html = '';
                    data.trade_suggestions.slice().reverse().forEach(t => {
                        const pnlClass = t.estimated_pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
                        html += `<tr>
                            <td>${t.time || '--'}</td>
                            <td><strong>${t.symbol}</strong></td>
                            <td><span class="badge ${t.direction === 'CALL' ? 'badge-green' : 'badge-red'}">${t.direction}</span></td>
                            <td>${t.strategy}</td>
                            <td>₹${t.entry_premium || '--'}</td>
                            <td>${t.expiry || '--'} (${t.dte}d)</td>
                            <td>${(t.ml_prob * 100).toFixed(0)}%</td>
                            <td>${(t.strat_prob * 100).toFixed(0)}%</td>
                            <td>${(t.final_score * 100).toFixed(0)}%</td>
                            <td>${t.regime}</td>
                        </tr>`;
                    });
                    document.getElementById('trades-body').innerHTML = html || '<tr><td colspan="10" style="text-align:center;color:#484f58">No trade suggestions yet. Waiting for signals...</td></tr>';
                });
        }
        setInterval(refreshData, 3000);
        refreshData();
    </script>
</head>
<body>
    <div class="container">
        <div class="nav" style="margin-bottom:16px">
            <a href="/" style="color:#58a6ff;text-decoration:none;margin-right:16px;font-size:14px;font-weight:bold">Live Paper Trading</a>
            <a href="/replay" style="color:#58a6ff;text-decoration:none;font-size:14px">Replay Simulation →</a>
        </div>
        <div class="header">
            <div>
                <h1>AI Trader Dashboard</h1>
                <div class="subtitle">NIFTY Options Paper Trading System</div>
            </div>
            <div>
                <span class="paper-badge">PAPER MODE</span>
                <div class="refresh-info" style="margin-top:4px">Auto-refreshes every 3s</div>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h3>System Status</h3>
                <div class="value" id="status"><span class="dot-yellow status-dot"></span>Starting...</div>
            </div>
            <div class="card">
                <h3>NIFTY Index Price</h3>
                <div class="value blue" id="price">--</div>
            </div>
            <div class="card">
                <h3>Market Regime</h3>
                <div class="value yellow" id="regime">UNKNOWN</div>
            </div>
            <div class="card">
                <h3>Trades Today</h3>
                <div class="value green" id="trades-today">0</div>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h3>Scan Count</h3>
                <div class="value" id="scans">0</div>
            </div>
            <div class="card">
                <h3>Signals Checked</h3>
                <div class="value" id="signals">0</div>
            </div>
            <div class="card">
                <h3>Last Scan</h3>
                <div class="value" style="font-size:16px" id="last-scan">--</div>
            </div>
            <div class="card">
                <h3>Models</h3>
                <div id="models" style="margin-top:4px"></div>
            </div>
        </div>

        <div class="card" style="margin-top:16px">
            <h3>Trade Suggestions</h3>
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Contract</th>
                        <th>Direction</th>
                        <th>Strategy</th>
                        <th>Premium</th>
                        <th>Expiry</th>
                        <th>ML Prob</th>
                        <th>Strat ML</th>
                        <th>Score</th>
                        <th>Regime</th>
                    </tr>
                </thead>
                <tbody id="trades-body">
                    <tr><td colspan="10" style="text-align:center;color:#484f58">No trade suggestions yet. Waiting for signals...</td></tr>
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""


def initialize():
    """Load models and verify DB connection."""
    global state
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(__import__('sqlalchemy').text("SELECT 1"))
        state["db_connected"] = True
    except Exception as e:
        logger.error(f"DB connection failed: {e}")

    predictor.load()
    state["models_loaded"] = predictor.is_loaded

    strategy_predictor.load()
    state["strategy_models_loaded"] = strategy_predictor.available_strategies

    state["status"] = "ready"
    logger.info("Dashboard initialized.")


def scan_market():
    """Run one scan cycle — compute features, check signals, score trades."""
    global state

    try:
        # Load latest 300 candles
        df = read_sql(
            "SELECT timestamp, symbol, open, high, low, close, volume, vwap, oi "
            "FROM minute_candles WHERE symbol = 'NIFTY-I' "
            "ORDER BY timestamp DESC LIMIT 300"
        )
        if df.empty or len(df) < 250:
            return

        df = df.sort_values("timestamp").reset_index(drop=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Compute features
        featured = compute_all_macro_indicators(df)
        if featured.empty:
            return

        latest = featured.iloc[-1].to_dict()
        state["last_price"] = float(latest.get("close", 0))
        state["last_scan"] = datetime.now().strftime("%H:%M:%S")
        state["scan_count"] += 1

        # Regime
        regime_window = df.tail(100)[["open", "high", "low", "close", "volume"]].copy()
        regime = regime_detector.detect(regime_window)
        state["regime"] = regime.value
        regime_strategies = get_strategies_for_regime(regime)

        # Signals
        signals = generate_signals(latest, "NIFTY-I")
        state["signals_checked"] += len(signals) if signals else 0
        state["status"] = "scanning"

        if not signals:
            return

        today = date.today()
        expiry = get_nearest_expiry(today)
        dte = get_days_to_expiry(today, expiry) if expiry else 0

        for sig in signals:
            ml_prob = 0.5
            if predictor.is_loaded:
                p = predictor.predict_macro(latest)
                if p is not None:
                    ml_prob = p

            if sig.direction == "PUT" and ml_prob > 0.40:
                continue

            strat_prob = strategy_predictor.predict(sig.strategy, latest) or 0.5
            if strat_prob < 0.30:
                continue

            directional_prob = ml_prob if sig.direction == "CALL" else (1.0 - ml_prob)

            flow_score = 0.5
            pcr = latest.get("pcr")
            if pcr and not np.isnan(pcr):
                flow_score = min(0.3 * (pcr > 1.2) + 0.2, 1.0)

            regime_bonus = 0.05 if regime_strategies and sig.strategy in regime_strategies else 0.0
            final_score = (
                WEIGHT_ML_PROBABILITY * directional_prob
                + WEIGHT_OPTIONS_FLOW * flow_score
                + WEIGHT_TECHNICAL_STRENGTH * sig.technical_strength
                + regime_bonus
            )

            if final_score < SCORE_THRESHOLD:
                continue

            atm = round(latest.get("close", 0) / 50) * 50
            opt_type = "CE" if sig.direction == "CALL" else "PE"
            exp_code = expiry.strftime("%y%m%d") if expiry else "000000"
            opt_symbol = f"NIFTY{exp_code}{atm}{opt_type}"

            trade = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "symbol": opt_symbol,
                "direction": sig.direction,
                "strategy": sig.strategy,
                "entry_premium": None,
                "expiry": str(expiry),
                "dte": dte,
                "ml_prob": round(ml_prob, 4),
                "strat_prob": round(strat_prob, 4),
                "flow_score": round(flow_score, 2),
                "final_score": round(final_score, 4),
                "regime": regime.value,
                "index_price": round(latest.get("close", 0), 1),
            }

            state["trade_suggestions"].append(trade)
            state["trades_today"] += 1

            logger.info(
                f"TRADE SUGGESTION: {sig.direction} {opt_symbol} | "
                f"ML={ml_prob:.2f} Strat={strat_prob:.2f} Score={final_score:.2f}"
            )
            break

    except Exception as e:
        logger.error(f"Scan error: {e}", exc_info=True)


def background_scanner():
    """Background thread that scans every 30 seconds."""
    while True:
        try:
            scan_market()
        except Exception as e:
            logger.error(f"Scanner error: {e}")
        time.sleep(30)


replay_state = {
    "status": "idle",        # idle, running, done
    "date": None,
    "progress": 0,
    "total_minutes": 0,
    "current_time": None,
    "current_price": 0,
    "regime": "UNKNOWN",
    "trades": [],
    "total_pnl": 0,
    "ticks_processed": 0,
}


def run_replay(replay_date: str):
    """Run tick replay for a specific date in background."""
    global replay_state
    from backtest.option_resolver import get_nearest_expiry, get_days_to_expiry, clear_cache

    replay_state = {
        "status": "running", "date": replay_date, "progress": 0,
        "total_minutes": 0, "current_time": None, "current_price": 0,
        "regime": "UNKNOWN", "trades": [], "total_pnl": 0, "ticks_processed": 0,
    }
    clear_cache()

    try:
        # Load ticks for this day
        ticks = read_sql(
            "SELECT timestamp, price, volume, oi, bid_price, ask_price, bid_qty, ask_qty "
            "FROM tick_data WHERE symbol = 'NIFTY-I' AND timestamp::date = :dt "
            "ORDER BY timestamp",
            {"dt": replay_date},
        )
        if ticks.empty:
            replay_state["status"] = "done"
            return

        ticks["timestamp"] = pd.to_datetime(ticks["timestamp"])

        # Load warmup candles
        warmup = read_sql(
            "SELECT timestamp, symbol, open, high, low, close, volume, vwap, oi "
            "FROM minute_candles WHERE symbol = 'NIFTY-I' "
            "AND timestamp < :dt ORDER BY timestamp DESC LIMIT 300",
            {"dt": replay_date},
        )
        warmup["timestamp"] = pd.to_datetime(warmup["timestamp"])
        candle_buffer = warmup.sort_values("timestamp").reset_index(drop=True)

        # Group by minute
        ticks["minute"] = ticks["timestamp"].dt.floor("min")
        minute_groups = ticks.groupby("minute")
        minutes = sorted(minute_groups.groups.keys())
        replay_state["total_minutes"] = len(minutes)

        in_trade = False
        current_trade = None
        option_info = None
        daily_trades = 0
        LOT_SIZE = 65
        SL_PCT = 0.30
        TGT_PCT = 0.50
        COMMISSION = 40.0

        for idx, minute_ts in enumerate(minutes):
            minute_ticks = minute_groups.get_group(minute_ts)
            replay_state["ticks_processed"] += len(minute_ticks)
            replay_state["progress"] = int((idx + 1) / len(minutes) * 100)
            replay_state["current_time"] = str(minute_ts)
            replay_state["current_price"] = float(minute_ticks["price"].iloc[-1])

            # Check open trade against ticks
            if in_trade and current_trade and option_info:
                entry_prem = option_info["entry_premium"]
                prem_df = option_info["premium_df"]
                ts_pd = pd.to_datetime(minute_ts)
                mask = (prem_df["timestamp"] - ts_pd).abs() <= pd.Timedelta(minutes=1)
                prem_row = prem_df[mask]

                if not prem_row.empty:
                    p_high = float(prem_row.iloc[0].get("high", prem_row.iloc[0]["premium"]))
                    p_low = float(prem_row.iloc[0].get("low", prem_row.iloc[0]["premium"]))
                    p_close = float(prem_row.iloc[0]["premium"])
                    bars_held = idx - current_trade.get("entry_idx", 0)

                    exit_prem = None
                    result = None
                    if p_low <= entry_prem * (1 - SL_PCT):
                        exit_prem = entry_prem * (1 - SL_PCT)
                        result = "LOSS"
                    elif p_high >= entry_prem * (1 + TGT_PCT):
                        exit_prem = entry_prem * (1 + TGT_PCT)
                        result = "WIN"
                    elif bars_held >= 20:
                        exit_prem = p_close
                        result = "TIMEOUT"

                    if exit_prem:
                        pnl = round((exit_prem - entry_prem) * LOT_SIZE - COMMISSION, 2)
                        current_trade["exit_time"] = str(minute_ts)
                        current_trade["exit_price"] = round(exit_prem, 2)
                        current_trade["pnl"] = pnl
                        current_trade["result"] = result
                        replay_state["trades"].append(current_trade)
                        replay_state["total_pnl"] = round(
                            sum(t["pnl"] for t in replay_state["trades"]), 2
                        )
                        in_trade = False
                        current_trade = None
                        option_info = None

            # Build candle
            candle = {
                "timestamp": minute_ts,
                "symbol": "NIFTY-I",
                "open": float(minute_ticks["price"].iloc[0]),
                "high": float(minute_ticks["price"].max()),
                "low": float(minute_ticks["price"].min()),
                "close": float(minute_ticks["price"].iloc[-1]),
                "volume": int(minute_ticks["volume"].sum()),
                "vwap": 0, "oi": 0,
            }
            candle_buffer = pd.concat(
                [candle_buffer, pd.DataFrame([candle])], ignore_index=True
            ).tail(500)

            # Signal + ML if not in trade
            if not in_trade and daily_trades < 5 and len(candle_buffer) >= 250:
                try:
                    featured = compute_all_macro_indicators(candle_buffer.tail(300).copy())
                    if featured.empty:
                        continue
                    latest = featured.iloc[-1].to_dict()
                except Exception:
                    continue

                # Time filter
                if hasattr(minute_ts, "hour"):
                    mins = minute_ts.hour * 60 + minute_ts.minute - 555
                    if mins < 5 or mins > 360:
                        continue

                # Regime
                try:
                    rw = candle_buffer.tail(100)[["open","high","low","close","volume"]].copy()
                    regime = regime_detector.detect(rw)
                    replay_state["regime"] = regime.value
                except Exception:
                    pass

                signals = generate_signals(latest, "NIFTY-I")
                if not signals:
                    continue

                sig = signals[0]
                ml_prob = 0.5
                if predictor.is_loaded:
                    p = predictor.predict_macro(latest)
                    if p is not None:
                        ml_prob = p

                if sig.direction == "PUT" and ml_prob > 0.40:
                    continue

                strat_prob = strategy_predictor.predict(sig.strategy, latest)
                if strat_prob is None or strat_prob < 0.05:
                    strat_prob = 0.5  # fallback when model unavailable / out-of-distribution

                dp = ml_prob if sig.direction == "CALL" else (1.0 - ml_prob)
                score = 0.5 * dp + 0.3 * strat_prob + 0.2 * sig.technical_strength
                if score < 0.55:
                    continue

                # Resolve option
                from backtest.option_resolver import resolve_option_at_entry
                opt = resolve_option_at_entry(
                    index_price=latest["close"], timestamp=minute_ts,
                    direction=sig.direction,
                )
                if opt is None:
                    continue

                entry_prem = opt["entry_premium"]
                if entry_prem <= 0:
                    continue

                current_trade = {
                    "entry_time": str(minute_ts),
                    "symbol": opt["symbol"],
                    "direction": sig.direction,
                    "strategy": sig.strategy,
                    "entry_price": round(entry_prem, 2),
                    "sl": round(entry_prem * (1 - SL_PCT), 2),
                    "target": round(entry_prem * (1 + TGT_PCT), 2),
                    "ml_prob": round(ml_prob, 3),
                    "strat_prob": round(strat_prob, 3),
                    "score": round(score, 3),
                    "entry_idx": idx,
                    "exit_time": None, "exit_price": None, "pnl": None, "result": "OPEN",
                }
                option_info = opt
                in_trade = True
                daily_trades += 1

            time.sleep(0.20)  # Delay for UI polling (200ms × ~375 min ≈ 75s per day)

        replay_state["status"] = "done"
        replay_state["progress"] = 100

    except Exception as e:
        logger.error(f"Replay error: {e}", exc_info=True)
        replay_state["status"] = "done"


REPLAY_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Trader - Replay Simulation</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e1e4e8; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1 { color: #58a6ff; margin-bottom: 5px; font-size: 24px; }
        .subtitle { color: #8b949e; margin-bottom: 20px; font-size: 14px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 16px; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; }
        .card h3 { color: #8b949e; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
        .card .value { font-size: 24px; font-weight: 700; }
        .green { color: #3fb950; } .red { color: #f85149; } .yellow { color: #d29922; } .blue { color: #58a6ff; }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }
        .badge-green { background: #0d1117; border: 1px solid #3fb950; color: #3fb950; }
        .badge-red { background: #0d1117; border: 1px solid #f85149; color: #f85149; }
        .badge-yellow { background: #0d1117; border: 1px solid #d29922; color: #d29922; }
        table { width: 100%; border-collapse: collapse; margin-top: 8px; }
        th { text-align: left; padding: 6px 10px; border-bottom: 2px solid #30363d; color: #8b949e; font-size: 11px; text-transform: uppercase; }
        td { padding: 6px 10px; border-bottom: 1px solid #21262d; font-size: 13px; }
        tr:hover { background: #1c2128; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
        .sim-badge { background: #a371f7; color: #0d1117; padding: 4px 12px; border-radius: 4px; font-weight: 700; font-size: 12px; }
        select, button { background: #21262d; color: #e1e4e8; border: 1px solid #30363d; padding: 8px 16px; border-radius: 6px; font-size: 14px; cursor: pointer; }
        button:hover { background: #30363d; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .progress-bar { width: 100%; height: 6px; background: #21262d; border-radius: 3px; margin-top: 8px; }
        .progress-fill { height: 100%; background: #58a6ff; border-radius: 3px; transition: width 0.3s; }
        .nav { margin-bottom: 16px; }
        .nav a { color: #58a6ff; text-decoration: none; margin-right: 16px; font-size: 14px; }
        .nav a:hover { text-decoration: underline; }
    </style>
    <script>
        function startReplay() {
            const day = document.getElementById('day-select').value;
            if (!day) return;
            document.getElementById('start-btn').disabled = true;
            fetch('/api/replay/start?date=' + day, {method: 'POST'}).then(() => pollReplay());
        }
        function pollReplay() {
            fetch('/api/replay/state').then(r => r.json()).then(data => {
                document.getElementById('status').textContent = data.status;
                document.getElementById('time').textContent = data.current_time ? data.current_time.split(' ')[1] || data.current_time : '--';
                document.getElementById('price').textContent = data.current_price ? '₹' + data.current_price.toLocaleString('en-IN', {maximumFractionDigits:1}) : '--';
                document.getElementById('regime').textContent = data.regime;
                document.getElementById('regime').className = 'value ' + (data.regime.includes('BULL') ? 'green' : data.regime.includes('BEAR') ? 'red' : 'yellow');
                document.getElementById('ticks').textContent = data.ticks_processed.toLocaleString();
                document.getElementById('n-trades').textContent = data.trades.length;
                document.getElementById('pnl').textContent = '₹' + data.total_pnl.toLocaleString('en-IN');
                document.getElementById('pnl').className = 'value ' + (data.total_pnl >= 0 ? 'green' : 'red');
                document.getElementById('progress-fill').style.width = data.progress + '%';
                document.getElementById('progress-text').textContent = data.progress + '%';

                let wins = data.trades.filter(t => t.pnl > 0).length;
                let total = data.trades.length;
                document.getElementById('wr').textContent = total > 0 ? Math.round(wins/total*100) + '%' : '--';

                let html = '';
                data.trades.slice().reverse().forEach(t => {
                    const cls = t.result === 'WIN' ? 'badge-green' : t.result === 'LOSS' ? 'badge-red' : 'badge-yellow';
                    const pnlCls = t.pnl >= 0 ? 'green' : 'red';
                    html += '<tr>' +
                        '<td>' + (t.entry_time ? t.entry_time.split(' ')[1] || t.entry_time.substring(11,19) : '') + '</td>' +
                        '<td><strong>' + t.symbol + '</strong></td>' +
                        '<td><span class="badge ' + (t.direction==='CALL'?'badge-green':'badge-red') + '">' + t.direction + '</span></td>' +
                        '<td>' + t.strategy + '</td>' +
                        '<td>₹' + t.entry_price + '</td>' +
                        '<td>' + (t.exit_price || '--') + '</td>' +
                        '<td class="' + pnlCls + '">₹' + (t.pnl || 0) + '</td>' +
                        '<td><span class="badge ' + cls + '">' + t.result + '</span></td>' +
                        '<td>' + (t.ml_prob*100).toFixed(0) + '%</td>' +
                        '<td>' + (t.score*100).toFixed(0) + '%</td></tr>';
                });
                document.getElementById('trades-body').innerHTML = html || '<tr><td colspan="10" style="text-align:center;color:#484f58">Waiting for trades...</td></tr>';

                if (data.status === 'running') setTimeout(pollReplay, 500);
                else document.getElementById('start-btn').disabled = false;
            });
        }
    </script>
</head>
<body>
    <div class="container">
        <div class="nav">
            <a href="/">← Live Paper Trading</a>
            <a href="/replay"><strong>Replay Simulation</strong></a>
        </div>
        <div class="header">
            <div>
                <h1>Tick Replay Simulation</h1>
                <div class="subtitle">Watch the AI trade a full historical day in fast-forward</div>
            </div>
            <span class="sim-badge">SIMULATION</span>
        </div>

        <div class="card" style="margin-bottom:16px;display:flex;align-items:center;gap:16px">
            <div>
                <h3>Select Day</h3>
                <select id="day-select">
                    <option value="">-- pick a day --</option>
                    DAYS_OPTIONS
                </select>
            </div>
            <button id="start-btn" onclick="startReplay()">▶ Start Replay</button>
            <div style="flex:1">
                <div style="display:flex;justify-content:space-between;font-size:12px;color:#8b949e">
                    <span id="status">idle</span>
                    <span id="progress-text">0%</span>
                </div>
                <div class="progress-bar"><div class="progress-fill" id="progress-fill" style="width:0%"></div></div>
            </div>
        </div>

        <div class="grid">
            <div class="card"><h3>Time</h3><div class="value blue" id="time">--</div></div>
            <div class="card"><h3>NIFTY Price</h3><div class="value blue" id="price">--</div></div>
            <div class="card"><h3>Regime</h3><div class="value yellow" id="regime">--</div></div>
            <div class="card"><h3>Ticks</h3><div class="value" id="ticks">0</div></div>
            <div class="card"><h3>Trades</h3><div class="value" id="n-trades">0</div></div>
            <div class="card"><h3>Win Rate</h3><div class="value green" id="wr">--</div></div>
            <div class="card"><h3>Day P&L</h3><div class="value green" id="pnl">₹0</div></div>
        </div>

        <div class="card">
            <h3>Trades</h3>
            <table>
                <thead><tr>
                    <th>Time</th><th>Contract</th><th>Dir</th><th>Strategy</th>
                    <th>Entry</th><th>Exit</th><th>P&L</th><th>Result</th><th>ML</th><th>Score</th>
                </tr></thead>
                <tbody id="trades-body">
                    <tr><td colspan="10" style="text-align:center;color:#484f58">Select a day and click Start</td></tr>
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/replay")
def replay_page():
    # Get available days from tick data
    days = read_sql("""
        SELECT timestamp::date as day, COUNT(*) as ticks
        FROM tick_data WHERE symbol = 'NIFTY-I'
        GROUP BY 1 HAVING COUNT(*) > 100
        ORDER BY 1
    """)
    options = ""
    for _, r in days.iterrows():
        options += f'<option value="{r["day"]}">{r["day"]} ({r["ticks"]:,} ticks)</option>\n'
    html = REPLAY_HTML.replace("DAYS_OPTIONS", options)
    return render_template_string(html)


@app.route("/api/state")
def api_state():
    return jsonify(state)


@app.route("/api/replay/state")
def api_replay_state():
    return jsonify(replay_state)


@app.route("/api/replay/start", methods=["POST"])
def api_replay_start():
    from flask import request
    replay_date = request.args.get("date")
    if not replay_date:
        return jsonify({"error": "date required"}), 400
    thread = threading.Thread(target=run_replay, args=(replay_date,), daemon=True)
    thread.start()
    return jsonify({"status": "started", "date": replay_date})


@app.route("/api/scan", methods=["POST"])
def api_scan():
    scan_market()
    return jsonify({"status": "scanned"})


if __name__ == "__main__":
    initialize()

    # Start background scanner
    scanner_thread = threading.Thread(target=background_scanner, daemon=True)
    scanner_thread.start()

    print("\n" + "=" * 50)
    print("  AI Trader Paper Trading Dashboard")
    print("  Open: http://localhost:5050")
    print("  Press Ctrl+C to stop")
    print("=" * 50 + "\n")

    app.run(host="0.0.0.0", port=5050, debug=False)
