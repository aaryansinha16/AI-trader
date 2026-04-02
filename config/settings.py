import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / os.getenv("MODEL_DIR", "models/saved")
LOG_DIR = BASE_DIR / os.getenv("LOG_DIR", "logs")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Database (TimescaleDB) ─────────────────────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "trading")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_URL = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    if DB_PASSWORD
    else f"postgresql://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ── Zerodha Kite Connect ──────────────────────────────────────────────────────
KITE_API_KEY = os.getenv("KITE_API_KEY", "")
KITE_API_SECRET = os.getenv("KITE_API_SECRET", "")
KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", "")

# ── TrueData ──────────────────────────────────────────────────────────────────
TRUEDATA_USER = os.getenv("TRUEDATA_USER", "")
TRUEDATA_PASSWORD = os.getenv("TRUEDATA_PASSWORD", "")

# TrueData API Endpoints
TD_AUTH_URL = "https://auth.truedata.in/token"
TD_HISTORY_URL = "https://history.truedata.in"
TD_SYMBOL_MASTER_URL = "https://api.truedata.in"
TD_ANALYTICS_URL = "https://analytics.truedata.in"
TD_TCP_HOST = "push.truedata.in"
TD_TCP_PORT = int(os.getenv("TD_TCP_PORT", "8084"))

# TrueData rate limit (requests per second)
TD_RATE_LIMIT_RPS = 1

# ── Symbols ───────────────────────────────────────────────────────────────────
SYMBOLS = ["NIFTY"]
# SYMBOLS = ["NIFTY", "BANKNIFTY"]  # Uncomment when ready to add BANKNIFTY

# TrueData symbol name mappings
# Spot index (for live quotes / real-time tick stream)
TD_INDEX_SPOT_SYMBOLS = {
    "NIFTY": "NIFTY 50",
    "BANKNIFTY": "NIFTY BANK",
}
# Continuous futures (for historical data / charting)
TD_INDEX_FUTURES_SYMBOLS = {
    "NIFTY": "NIFTY-I",
    "BANKNIFTY": "BANKNIFTY-I",
}
# Backwards-compat alias used by main.py ingest
TD_INDEX_SYMBOLS = TD_INDEX_FUTURES_SYMBOLS

# Strike gap per underlying (NIFTY = 50pt, BANKNIFTY = 100pt)
STRIKE_GAP = {
    "NIFTY": 50,
    "BANKNIFTY": 100,
}

# Number of strikes above/below ATM to track (±3 = 7 strikes per CE/PE)
ATM_RANGE = int(os.getenv("ATM_RANGE", "3"))

# Maximum symbols to subscribe (plan limit)
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "50"))

# ── Trading Parameters ────────────────────────────────────────────────────────
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "50000"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "5"))
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "0.05"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.6"))

# ── Model Paths ───────────────────────────────────────────────────────────────
MACRO_MODEL_PATH = str(BASE_DIR / os.getenv("MACRO_MODEL_PATH", "models/saved/macro_model.pkl"))
MICRO_MODEL_PATH = str(BASE_DIR / os.getenv("MICRO_MODEL_PATH", "models/saved/micro_model.pkl"))

# ── Feature Configuration ─────────────────────────────────────────────────────
FEATURE_COLUMNS_MACRO = [
    # Price / Momentum (core)
    "rsi", "macd", "macd_signal", "macd_hist",
    "ema9", "ema20", "ema50", "sma200",
    "vwap_dist", "bollinger_upper", "bollinger_lower", "bollinger_width",
    "atr", "volume_ratio", "volume_sma20",
    # Momentum derivatives
    "stoch_rsi_k", "stoch_rsi_d", "williams_r",
    "roc_10", "roc_20", "adx", "di_plus", "di_minus", "cci",
    # Trend crossovers
    "ema9_20_cross", "ema20_50_cross", "close_above_sma200",
    # Volatility
    "atr_pct", "bollinger_pct", "returns_1m",
    "volatility_20", "volatility_60", "vol_regime",
    # Candle patterns
    "candle_body_pct", "upper_shadow_pct", "lower_shadow_pct",
    # Multi-timeframe
    "rsi_5m", "rsi_15m", "ema20_5m", "atr_5m",
    # Session / time
    "minutes_since_open", "session_progress", "day_of_week",
    "is_first_hour", "is_last_hour",
    # Volume signals (expanded)
    "volume_change", "cum_volume_delta_20", "obv_slope", "mfi",
    # Options basics
    "oi_change", "pcr", "iv",
    # Options-aware (relative strike, expiry, cross-strike)
    "relative_strike", "days_to_expiry", "theta_pressure",
    "oi_skew", "pcr_near_atm", "pcr_far",
    "max_oi_call_rel", "max_oi_put_rel", "oi_concentration",
    "call_oi_gradient", "put_oi_gradient", "iv_skew",
]

FEATURE_COLUMNS_MICRO = [
    "bid_ask_spread", "order_imbalance", "trade_size_spike",
    "volume_burst", "tick_momentum",
]

# ── Trade Scoring Weights ─────────────────────────────────────────────────────
WEIGHT_ML_PROBABILITY = 0.50
WEIGHT_OPTIONS_FLOW = 0.30
WEIGHT_TECHNICAL_STRENGTH = 0.20
# WEIGHT_STRATEGY_PROB = 0.25  # reserved — strat_prob is used as gate only until models improve
STRAT_PROB_SCALE = 0.06             # normalize strat_prob raw output to [0,1] when needed

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── Market Hours (IST) ───────────────────────────────────────────────────────
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30

# ── Scan Cycle ────────────────────────────────────────────────────────────────
SCAN_INTERVAL_SECONDS = 60