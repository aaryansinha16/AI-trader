export const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5050";

export const SSE_STREAM_URL = `${API_BASE}/api/stream`;

export interface StreamPayload {
  state: {
    last_price: number;
    spot_price?: number;
    regime: string;
    status: string;
    last_scan: string | null;
    scan_count: number;
    trade_suggestions: TradeSuggestion[];
    auto_trade_enabled?: boolean;
  };
  positions_by_mode: { test: PaperPosition[]; live: PaperPosition[] };
  tick_cache: Record<string, { price: number; ts: string }>;
  tick_cache_age: number | null;
  total_open_pnl: number;
  total_closed_pnl: number;
  total_pnl: number;
  total_open_pnl_test: number;
  total_closed_pnl_test: number;
  total_pnl_test: number;
  total_open_pnl_live: number;
  total_closed_pnl_live: number;
  total_pnl_live: number;
}

export async function fetchJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`API ${path} → ${res.status}`);
  return res.json();
}

export async function postJSON<T>(path: string, body?: object): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`API ${path} → ${res.status}`);
  return res.json();
}

export type RiskLevel = "low" | "medium" | "high";

export interface LiveState {
  status: string;
  last_scan: string | null;
  last_price: number;
  spot_price?: number;
  regime: string;
  models_loaded: boolean;
  strategy_models_loaded: string[];
  db_connected: boolean;
  trade_suggestions: TradeSuggestion[];
  scan_count: number;
  signals_checked: number;
  trades_today: number;
  scanner_enabled?: boolean;
  auto_trade_enabled?: boolean;
}

export interface TradeSuggestion {
  time: string;
  symbol: string;
  direction: "CALL" | "PUT";
  strategy: string;
  entry_premium: number | null;
  sl_price: number | null;
  target_price: number | null;
  risk_label: "LOW" | "MEDIUM" | "HIGH" | null;
  expiry: string;
  dte: number;
  ml_prob: number;
  strat_prob: number;
  flow_score: number;
  final_score: number;
  regime: string;
  index_price: number;
  lots?: number;
}

export interface Trade {
  entry_time: string;
  exit_time: string;
  symbol: string;
  direction: "CALL" | "PUT";
  strategy: string;
  entry_premium: number;
  exit_premium: number;
  sl: number;
  target: number;
  sl_pct: number;
  tgt_pct: number;
  lot_size: number;
  pnl: number;
  result: string;
  ml_prob: number;
  strat_prob: number;
  flow_score: number;
  final_score: number;
  regime: string;
  index_price: number;
}

/** A closed live paper trade (persisted across Flask restarts). */
export interface LiveTrade {
  id: number;
  entry_time: string;         // HH:MM:SS
  entry_time_dt: string;      // full ISO datetime
  exit_time: string | null;
  symbol: string;
  direction: "CALL" | "PUT";
  strategy: string;
  entry_premium: number;
  exit_premium: number | null;
  current_premium: number;
  sl: number;
  initial_sl: number;
  target: number;
  lot_size: number;
  ml_prob: number;
  final_score: number;
  realised_pnl: number | null;
  exit_reason: string | null;
  status: "OPEN" | "CLOSED";
  mode: string;
  regime?: string;
  index_price: number;
  journey: JourneyPoint[];
  breakeven_locked?: boolean;
  trailing_active?: boolean;
}

export interface JourneyPoint {
  ts: string;
  option_price?: number;   // live trades use option_price
  premium?: number;        // backtest uses premium
  nifty_price: number;
  sl: number;
  unrealised_pnl?: number;
  bars_held?: number;
}

export interface BacktestProfile {
  trades: number;
  pnl: number;
  win_rate: number;
  avg_win: number;
  avg_loss: number;
  max_dd: number;
  rr: number;
  equity_curve: number[];
  trade_list: Trade[];
}

export interface BacktestResults {
  low?: BacktestProfile;
  medium?: BacktestProfile;
  high?: BacktestProfile;
}

export interface EquityCurvePoint {
  time: string;
  equity: number;
}

export interface Candle {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface RLStatus {
  tabular?: { states: number; episodes: number; policy: Record<string, number> };
  dqn?: { episodes: number; training_steps: number; epsilon: number; params: number };
}

export interface RiskProfile {
  name: string;
  base_lot_size: number;
  lot_multiplier: number;
  sl_pct: number;
  tgt_pct: number;
  score_threshold: number;
  max_trades_day: number;
  max_premium: number;
  max_capital_per_trade: number;
}

export interface CandleDateInfo {
  day: string;
  bars: number;
  ticks: number;
}

export interface OptionChainRow {
  symbol: string;
  strike: number;
  type: "CE" | "PE";
  last_price: number;
  volume: number;
  oi: number;
}

export interface OptionTickData {
  data: Record<string, unknown>[];
  source: "ticks" | "candles";
}

export interface PaperPosition {
  id: number;
  entry_time: string;
  symbol: string;
  direction: "CALL" | "PUT";
  strategy: string;
  entry_premium: number;
  sl: number;
  initial_sl: number;
  target: number;
  max_premium: number;
  trailing_active: boolean;
  lot_size: number;
  ml_prob: number;
  final_score: number;
  index_price: number;
  expiry: string;
  status: "OPEN" | "CLOSED";
  current_premium: number;
  unrealised_pnl: number;
  exit_time: string | null;
  exit_premium: number | null;
  realised_pnl: number | null;
  exit_reason: string | null;
}

export interface PaperPositionsResponse {
  positions: PaperPosition[];
  total_open_pnl: number;
  total_closed_pnl: number;
  total_pnl: number;
}

export type TradingMode = "test" | "live";

export async function enterPaperTrade(suggestion: TradeSuggestion, mode: TradingMode = "test"): Promise<PaperPosition> {
  return postJSON<PaperPosition>("/api/paper/enter", {
    symbol: suggestion.symbol,
    direction: suggestion.direction,
    strategy: suggestion.strategy,
    entry_premium: suggestion.entry_premium,
    expiry: suggestion.expiry,
    ml_prob: suggestion.ml_prob,
    final_score: suggestion.final_score,
    index_price: suggestion.index_price,
    sl_pct: (suggestion as unknown as Record<string, number>).sl_pct,
    target_pct: (suggestion as unknown as Record<string, number>).target_pct,
    mode,
  });
}

export async function exitPaperTrade(id: number, mode: TradingMode = "test"): Promise<PaperPosition> {
  return postJSON<PaperPosition>("/api/paper/exit", { id, mode });
}

export async function getPaperPositions(mode: TradingMode = "test"): Promise<PaperPositionsResponse> {
  return fetchJSON<PaperPositionsResponse>(`/api/paper/positions?mode=${mode}`);
}

export async function clearClosedPositions(mode: TradingMode = "test"): Promise<void> {
  await postJSON("/api/paper/clear", { mode });
}

export async function setAutoTrade(enabled: boolean): Promise<{ auto_trade_enabled: boolean }> {
  return postJSON<{ auto_trade_enabled: boolean }>("/api/auto_trade", { enabled });
}
