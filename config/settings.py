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
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ── Zerodha Kite Connect ──────────────────────────────────────────────────────
KITE_API_KEY = os.getenv("KITE_API_KEY", "")
KITE_API_SECRET = os.getenv("KITE_API_SECRET", "")
KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", "")

# ── TrueData ──────────────────────────────────────────────────────────────────
TRUEDATA_USER = os.getenv("TRUEDATA_USER", "")
TRUEDATA_PASSWORD = os.getenv("TRUEDATA_PASSWORD", "")

# ── Symbols ───────────────────────────────────────────────────────────────────
SYMBOLS = ["NIFTY", "BANKNIFTY"]

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
    "rsi", "macd", "macd_signal", "ema20", "ema50",
    "vwap_dist", "bollinger_upper", "bollinger_lower", "bollinger_width",
    "atr", "volume_ratio", "volume_sma20",
    "oi_change", "pcr", "iv",
]

FEATURE_COLUMNS_MICRO = [
    "bid_ask_spread", "order_imbalance", "trade_size_spike",
    "volume_burst", "tick_momentum",
]

# ── Trade Scoring Weights (from docs) ─────────────────────────────────────────
WEIGHT_ML_PROBABILITY = 0.5
WEIGHT_OPTIONS_FLOW = 0.3
WEIGHT_TECHNICAL_STRENGTH = 0.2

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── Market Hours (IST) ───────────────────────────────────────────────────────
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30

# ── Scan Cycle ────────────────────────────────────────────────────────────────
SCAN_INTERVAL_SECONDS = 60