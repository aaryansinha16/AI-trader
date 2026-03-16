-- ============================================================================
-- AI Trading System – TimescaleDB Schema
-- ============================================================================
-- Requires: CREATE EXTENSION IF NOT EXISTS timescaledb;
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ── Tick Data (raw ticks from Kite / TrueData WebSocket) ─────────────────────
CREATE TABLE IF NOT EXISTS tick_data (
    timestamp   TIMESTAMPTZ     NOT NULL,
    symbol      TEXT            NOT NULL,
    price       DOUBLE PRECISION NOT NULL,
    volume      BIGINT          NOT NULL DEFAULT 0,
    bid_price   DOUBLE PRECISION,
    ask_price   DOUBLE PRECISION,
    bid_qty     BIGINT,
    ask_qty     BIGINT,
    oi          BIGINT
);

SELECT create_hypertable('tick_data', 'timestamp', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_tick_symbol_ts ON tick_data (symbol, timestamp DESC);

-- ── Second Candles (aggregated from ticks) ───────────────────────────────────
CREATE TABLE IF NOT EXISTS second_candles (
    timestamp   TIMESTAMPTZ     NOT NULL,
    symbol      TEXT            NOT NULL,
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      BIGINT          NOT NULL DEFAULT 0
);

SELECT create_hypertable('second_candles', 'timestamp', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_second_symbol_ts ON second_candles (symbol, timestamp DESC);

-- ── Minute Candles (primary ML training timeframe) ───────────────────────────
CREATE TABLE IF NOT EXISTS minute_candles (
    timestamp   TIMESTAMPTZ     NOT NULL,
    symbol      TEXT            NOT NULL,
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      BIGINT          NOT NULL DEFAULT 0,
    vwap        DOUBLE PRECISION
);

SELECT create_hypertable('minute_candles', 'timestamp', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_minute_symbol_ts ON minute_candles (symbol, timestamp DESC);

-- ── 5-Minute Candles (regime detection timeframe) ────────────────────────────
CREATE TABLE IF NOT EXISTS five_minute_candles (
    timestamp   TIMESTAMPTZ     NOT NULL,
    symbol      TEXT            NOT NULL,
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      BIGINT          NOT NULL DEFAULT 0,
    vwap        DOUBLE PRECISION
);

SELECT create_hypertable('five_minute_candles', 'timestamp', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_5min_symbol_ts ON five_minute_candles (symbol, timestamp DESC);

-- ── Option Chain Snapshots ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS option_chain (
    timestamp       TIMESTAMPTZ     NOT NULL,
    symbol          TEXT            NOT NULL,
    expiry          DATE            NOT NULL,
    strike          DOUBLE PRECISION NOT NULL,
    option_type     TEXT            NOT NULL,  -- CE / PE
    ltp             DOUBLE PRECISION,
    volume          BIGINT          DEFAULT 0,
    oi              BIGINT          DEFAULT 0,
    oi_change       BIGINT          DEFAULT 0,
    iv              DOUBLE PRECISION,
    bid_price       DOUBLE PRECISION,
    ask_price       DOUBLE PRECISION,
    delta           DOUBLE PRECISION,
    gamma           DOUBLE PRECISION,
    theta           DOUBLE PRECISION,
    vega            DOUBLE PRECISION
);

SELECT create_hypertable('option_chain', 'timestamp', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_oc_symbol_ts ON option_chain (symbol, timestamp DESC);

-- ── Macro Features (computed from 1m candles – for Macro ML Model) ───────────
CREATE TABLE IF NOT EXISTS features_macro (
    timestamp       TIMESTAMPTZ     NOT NULL,
    symbol          TEXT            NOT NULL,
    rsi             DOUBLE PRECISION,
    macd            DOUBLE PRECISION,
    macd_signal     DOUBLE PRECISION,
    ema20           DOUBLE PRECISION,
    ema50           DOUBLE PRECISION,
    vwap            DOUBLE PRECISION,
    vwap_dist       DOUBLE PRECISION,
    bollinger_upper DOUBLE PRECISION,
    bollinger_lower DOUBLE PRECISION,
    bollinger_width DOUBLE PRECISION,
    atr             DOUBLE PRECISION,
    volume_ratio    DOUBLE PRECISION,
    volume_sma20    DOUBLE PRECISION,
    oi_change       DOUBLE PRECISION,
    pcr             DOUBLE PRECISION,
    iv              DOUBLE PRECISION
);

SELECT create_hypertable('features_macro', 'timestamp', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_feat_macro_ts ON features_macro (symbol, timestamp DESC);

-- ── Micro Features (computed from tick/second data – for Microstructure Model)
CREATE TABLE IF NOT EXISTS features_micro (
    timestamp       TIMESTAMPTZ     NOT NULL,
    symbol          TEXT            NOT NULL,
    bid_ask_spread  DOUBLE PRECISION,
    order_imbalance DOUBLE PRECISION,
    trade_size_spike DOUBLE PRECISION,
    volume_burst    DOUBLE PRECISION,
    tick_momentum   DOUBLE PRECISION
);

SELECT create_hypertable('features_micro', 'timestamp', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_feat_micro_ts ON features_micro (symbol, timestamp DESC);

-- ── Trade Log ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trade_log (
    trade_id        SERIAL          PRIMARY KEY,
    entry_time      TIMESTAMPTZ     NOT NULL,
    exit_time       TIMESTAMPTZ,
    symbol          TEXT            NOT NULL,
    option_symbol   TEXT,
    side            TEXT            NOT NULL,  -- BUY / SELL
    entry_price     DOUBLE PRECISION NOT NULL,
    exit_price      DOUBLE PRECISION,
    stop_loss       DOUBLE PRECISION,
    target          DOUBLE PRECISION,
    quantity        INT             NOT NULL,
    pnl             DOUBLE PRECISION,
    result          TEXT,           -- WIN / LOSS / OPEN
    strategy        TEXT,
    ml_score        DOUBLE PRECISION,
    flow_score      DOUBLE PRECISION,
    tech_score      DOUBLE PRECISION,
    final_score     DOUBLE PRECISION,
    regime          TEXT
);

-- ── Model Registry ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_registry (
    id              SERIAL          PRIMARY KEY,
    model_name      TEXT            NOT NULL,
    model_type      TEXT            NOT NULL,  -- macro / micro
    version         INT             NOT NULL,
    trained_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    train_start     DATE,
    train_end       DATE,
    accuracy        DOUBLE PRECISION,
    precision_score DOUBLE PRECISION,
    recall_score    DOUBLE PRECISION,
    f1_score        DOUBLE PRECISION,
    file_path       TEXT            NOT NULL,
    is_active       BOOLEAN         DEFAULT FALSE,
    metadata        JSONB
);

-- ── Daily Performance ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS daily_performance (
    date            DATE            PRIMARY KEY,
    total_trades    INT             DEFAULT 0,
    wins            INT             DEFAULT 0,
    losses          INT             DEFAULT 0,
    gross_pnl       DOUBLE PRECISION DEFAULT 0,
    net_pnl         DOUBLE PRECISION DEFAULT 0,
    max_drawdown    DOUBLE PRECISION DEFAULT 0,
    win_rate        DOUBLE PRECISION DEFAULT 0,
    profit_factor   DOUBLE PRECISION DEFAULT 0
);