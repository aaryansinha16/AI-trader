"use client";

import React, { useEffect, useState, useRef } from "react";
import Sidebar from "@/components/Sidebar";
import Badge from "@/components/Badge";
import {
  fetchJSON, postJSON,
  enterPaperTrade, exitPaperTrade, getPaperPositions, clearClosedPositions, setAutoTrade,
  SSE_STREAM_URL,
  type LiveState, type TradeSuggestion, type PaperPosition, type StreamPayload,
} from "@/lib/api";
import { RefreshCw, Zap, TrendingUp, TrendingDown, X, Trash2, Radio, Bot, Hand, Activity } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ReferenceLine, ResponsiveContainer, CartesianGrid } from "recharts";

type JourneyPoint = { ts: string; option_price: number; nifty_price: number; sl: number; unrealised_pnl: number };
type JourneyData = { id: number; symbol: string; direction: string; entry_premium: number; initial_sl: number; target: number; status: string; journey: JourneyPoint[] };
import { useTradingMode } from "@/contexts/TradingModeContext";

export default function LivePage() {
  const { mode } = useTradingMode();
  const [state, setState] = useState<LiveState | null>(null);
  const [scanning, setScanning] = useState(false);
  const [lastUpdate, setLastUpdate] = useState("--");
  const [positionsByMode, setPositionsByMode] = useState<{ test: PaperPosition[]; live: PaperPosition[] }>({ test: [], live: [] });
  const [pnlByMode, setPnlByMode] = useState<Record<string, { open: number; closed: number; total: number }>>({
    test: { open: 0, closed: 0, total: 0 },
    live: { open: 0, closed: 0, total: 0 },
  });
  const [autoTrade, setAutoTradeState] = useState<boolean>(true);
  const [togglingAuto, setTogglingAuto] = useState(false);
  const [entering, setEntering] = useState<string | null>(null);
  const [exiting, setExiting] = useState<number | null>(null);
  const [enterError, setEnterError] = useState<string | null>(null);
  const [tickCacheAge, setTickCacheAge] = useState<number | null>(null);
  const [livePrices, setLivePrices] = useState<Record<string, { price: number; ts: string }>>({});
  const [sseConnected, setSseConnected] = useState(false);
  const esRef = useRef<EventSource | null>(null);

  // SSE connection — single stream replaces all HTTP polling
  useEffect(() => {
    const es = new EventSource(SSE_STREAM_URL);
    esRef.current = es;

    es.onopen = () => setSseConnected(true);
    es.onerror = () => setSseConnected(false);

    es.onmessage = (event) => {
      try {
        const d: StreamPayload = JSON.parse(event.data);
        if (d.state) {
          setState(prev => ({
            ...(prev ?? {} as LiveState),
            ...d.state,
          } as LiveState));
          setLastUpdate(new Date().toLocaleTimeString("en-IN"));
          if (d.state.auto_trade_enabled !== undefined) {
            setAutoTradeState(d.state.auto_trade_enabled);
          }
        }
        if (d.positions_by_mode) setPositionsByMode(d.positions_by_mode);
        if (d.tick_cache) setLivePrices(d.tick_cache);
        if (d.tick_cache_age !== undefined) setTickCacheAge(d.tick_cache_age);
        setPnlByMode({
          test: { open: d.total_open_pnl_test ?? 0, closed: d.total_closed_pnl_test ?? 0, total: d.total_pnl_test ?? 0 },
          live: { open: d.total_open_pnl_live ?? 0, closed: d.total_closed_pnl_live ?? 0, total: d.total_pnl_live ?? 0 },
        });
      } catch {}
    };

    // Fallback: load initial state via HTTP in case SSE takes a moment
    fetchJSON<LiveState>("/api/state").then(d => { if (d) setState(d); }).catch(() => {});

    return () => { es.close(); esRef.current = null; };
  }, []);

  // Derived: positions and PnL for current mode
  const positions = positionsByMode[mode] ?? [];
  const totalOpenPnl = pnlByMode[mode]?.open ?? 0;
  const totalClosedPnl = pnlByMode[mode]?.closed ?? 0;
  const totalPnl = pnlByMode[mode]?.total ?? 0;

  const triggerScan = async () => {
    setScanning(true);
    try { await postJSON("/api/scan"); }
    finally { setScanning(false); }
  };

  const handleToggleAutoTrade = async () => {
    setTogglingAuto(true);
    try {
      const res = await setAutoTrade(!autoTrade);
      setAutoTradeState(res.auto_trade_enabled);
    } finally {
      setTogglingAuto(false);
    }
  };

  const handleEnter = async (t: TradeSuggestion) => {
    setEntering(t.symbol + t.direction);
    setEnterError(null);
    try {
      await enterPaperTrade(t, mode);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setEnterError(msg.includes("Could not determine") ? "No live premium data for this contract yet" : msg);
    } finally {
      setEntering(null);
    }
  };

  const handleExit = async (id: number) => {
    setExiting(id);
    try { await exitPaperTrade(id, mode); }
    finally { setExiting(null); }
  };

  const handleClear = async () => {
    await clearClosedPositions(mode);
  };

  const openPositions = positions.filter(p => p.status === "OPEN");
  const closedPositions = positions.filter(p => p.status === "CLOSED");

  // Journey chart state
  const [journeyId, setJourneyId] = useState<number | null>(null);
  const [journeyData, setJourneyData] = useState<JourneyData | null>(null);
  const journeyInterval = useRef<NodeJS.Timeout | null>(null);

  const fetchJourney = async (id: number) => {
    try {
      const data = await fetchJSON(`/api/paper/journey/${id}`);
      setJourneyData(data as JourneyData);
    } catch { /* silent */ }
  };

  const toggleJourney = (id: number) => {
    if (journeyId === id) {
      // Close
      setJourneyId(null);
      setJourneyData(null);
      if (journeyInterval.current) clearInterval(journeyInterval.current);
    } else {
      setJourneyId(id);
      fetchJourney(id);
      if (journeyInterval.current) clearInterval(journeyInterval.current);
      journeyInterval.current = setInterval(() => fetchJourney(id), 5000);
    }
  };

  // Clean up interval on unmount
  useEffect(() => () => { if (journeyInterval.current) clearInterval(journeyInterval.current); }, []);

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 p-5 overflow-y-auto">

        {/* Header */}
        <div className="flex items-center justify-between mb-5">
          <div>
            <h1 className="text-sm font-bold uppercase tracking-wider" style={{ color: '#00e87b' }}>Live Trading</h1>
            <p className="text-[10px] mt-0.5" style={{ color: mode === "live" ? '#ff3e3e' : '#3d4450' }}>
              {mode === "live" ? "LIVE MODE — REAL ZERODHA EXECUTIONS" : "TEST MODE — SIMULATED PAPER ORDERS"}
            </p>
          </div>
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1.5 text-[10px]" style={{ color: sseConnected ? '#00e87b' : '#ff3e3e' }}>
              <Radio className="w-3 h-3" />
              {sseConnected ? "LIVE" : "RECONNECTING..."}
            </span>
            <span className="text-[10px]" style={{ color: '#3d4450' }}>{lastUpdate}</span>

            {/* Auto / Manual toggle */}
            <button
              onClick={handleToggleAutoTrade}
              disabled={togglingAuto}
              className="flex items-center gap-1.5 text-[10px] font-semibold px-3 py-1.5 disabled:opacity-50 transition-all"
              style={{
                border: `1px solid ${autoTrade ? '#00e87b' : '#4da6ff'}`,
                color: autoTrade ? '#00e87b' : '#4da6ff',
                background: autoTrade ? '#0a2a18' : '#0a1a2a',
                letterSpacing: '0.08em',
              }}
            >
              {autoTrade ? <Bot className="w-3 h-3" /> : <Hand className="w-3 h-3" />}
              {autoTrade ? "AUTO" : "MANUAL"}
            </button>

            <button onClick={triggerScan} disabled={scanning} className="t-btn flex items-center gap-1.5 disabled:opacity-50">
              {scanning ? <RefreshCw className="w-3 h-3 animate-spin" /> : <Zap className="w-3 h-3" />}
              {scanning ? "SCANNING..." : "SCAN NOW"}
            </button>
          </div>
        </div>

        {/* Status bar */}
        <div className="t-panel p-4 mb-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-[9px] uppercase tracking-[1.5px] mb-1" style={{ color: '#5a6270' }}>System Status</p>
              <div className="flex items-center gap-2">
                <span className={`w-[6px] h-[6px] ${state?.status === "scanning" ? "bg-[#00e87b] t-pulse" : "bg-[#e8c300]"}`} />
                <span className="font-semibold text-sm uppercase">{state?.status ?? "..."}</span>
              </div>
            </div>
            <div>
              <p className="text-[9px] uppercase tracking-[1.5px] mb-1" style={{ color: '#5a6270' }}>
                NIFTY <span style={{ color: '#3d4450' }}>· FUTURES</span>
              </p>
              {(() => {
                // System trades on futures only — no spot subscription
                const futPrice = state?.last_price ?? null;
                return (
                  <p className="text-xl font-bold" style={{ color: '#4da6ff' }}>
                    {futPrice
                      ? `₹${futPrice.toLocaleString("en-IN", { maximumFractionDigits: 1 })}`
                      : "--"}
                  </p>
                );
              })()}
            </div>
            <div>
              <p className="text-[9px] uppercase tracking-[1.5px] mb-1" style={{ color: '#5a6270' }}>Market Regime</p>
              <p className="text-xl font-bold" style={{
                color: state?.regime?.includes("BULL") ? '#00e87b' : state?.regime?.includes("BEAR") ? '#ff3e3e' : '#e8c300'
              }}>
                {state?.regime ?? "--"}
              </p>
            </div>
            <div>
              <p className="text-[9px] uppercase tracking-[1.5px] mb-1" style={{ color: '#5a6270' }}>Session P&L</p>
              <p className="text-xl font-bold" style={{ color: totalPnl >= 0 ? '#00e87b' : '#ff3e3e' }}>
                ₹{totalPnl.toLocaleString("en-IN")}
              </p>
            </div>
          </div>
        </div>

        {/* Open Positions */}
        {openPositions.length > 0 && (
          <div className="t-panel p-4 mb-4" style={{ borderColor: '#1a3a1a' }}>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-3">
                <h3 className="text-[11px] font-semibold uppercase tracking-wider" style={{ color: '#00e87b' }}>
                  Open Positions ({openPositions.length})
                </h3>
                {tickCacheAge !== null && tickCacheAge < 30 ? (
                  <span className="text-[9px] px-1.5 py-0.5" style={{ background: '#0a2a18', border: '1px solid #1a5c3a', color: '#00e87b' }}>
                    ⚡ TICK LIVE · {tickCacheAge.toFixed(0)}s
                  </span>
                ) : (
                  <span className="text-[9px] px-1.5 py-0.5" style={{ background: '#2a2a0a', border: '1px solid #5c5c1a', color: '#e8c300' }}>
                    ⏱ REST ~1MIN
                  </span>
                )}
              </div>
              <span className="text-xs font-bold" style={{ color: totalOpenPnl >= 0 ? '#00e87b' : '#ff3e3e' }}>
                Unrealised: ₹{totalOpenPnl.toLocaleString("en-IN")}
              </span>
            </div>
            <div className="overflow-x-auto">
              <table>
                <thead>
                  <tr>
                    {["Time", "Contract", "Dir", "Entry ₹", "Current ₹", "SL ₹", "Target ₹", "Lots", "Unrealised P&L", ""].map(h => (
                      <th key={h}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {openPositions.map(p => {
                    // Use tick cache price if available and fresh, else fall back to server value
                    const tickEntry = livePrices[p.symbol];
                    const currentPrem = (tickEntry && tickCacheAge !== null && tickCacheAge < 30)
                      ? tickEntry.price
                      : p.current_premium;
                    const unrealisedPnl = Math.round((currentPrem - p.entry_premium) * p.lot_size - 40);
                    const pnlColor = unrealisedPnl >= 0 ? '#00e87b' : '#ff3e3e';
                    const pct = ((currentPrem - p.entry_premium) / p.entry_premium * 100);
                    const slPct = ((p.sl - p.entry_premium) / p.entry_premium * 100);
                    const tgtPct = ((p.target - p.entry_premium) / p.entry_premium * 100);
                    return (
                      <React.Fragment key={p.id}>
                      <tr>
                        <td style={{ color: '#5a6270' }}>{p.entry_time}</td>
                        <td style={{ fontWeight: 600 }}>{p.symbol}</td>
                        <td><Badge label={p.direction} variant={p.direction === "CALL" ? "green" : "red"} /></td>
                        <td>₹{p.entry_premium}</td>
                        <td style={{ color: pnlColor, fontWeight: 600 }}>
                          ₹{currentPrem.toFixed(1)}
                          <span className="ml-1 text-[10px]" style={{ color: pnlColor }}>
                            ({pct >= 0 ? "+" : ""}{pct.toFixed(1)}%)
                          </span>
                        </td>
                        <td style={{ color: p.trailing_active ? '#e8c300' : (p as any).breakeven_locked ? '#00b4ff' : '#ff3e3e' }}>
                          ₹{p.sl}
                          <span className="text-[9px] ml-0.5">({slPct.toFixed(0)}%)</span>
                          {p.trailing_active && (
                            <span className="text-[8px] ml-1 px-1 py-0" style={{ background: '#2a2a0a', border: '1px solid #5c5c1a', color: '#e8c300' }}>TRAIL</span>
                          )}
                          {(p as any).breakeven_locked && !p.trailing_active && (
                            <span className="text-[8px] ml-1 px-1 py-0" style={{ background: '#001a2a', border: '1px solid #005580', color: '#00b4ff' }}>BE</span>
                          )}
                        </td>
                        <td style={{ color: '#00e87b' }}>₹{p.target} <span className="text-[9px]">(+{tgtPct.toFixed(0)}%)</span></td>
                        <td style={{ color: p.lot_size > 65 ? '#e8c300' : '#5a6270', fontWeight: p.lot_size > 65 ? 700 : 400 }}>
                          {Math.round(p.lot_size / 65)}×
                        </td>
                        <td style={{ color: pnlColor, fontWeight: 700 }}>
                          {unrealisedPnl >= 0 ? "+" : ""}₹{unrealisedPnl.toLocaleString("en-IN")}
                        </td>
                        <td>
                          <div className="flex gap-1">
                            <button
                              onClick={() => toggleJourney(p.id)}
                              className="t-btn flex items-center gap-1 text-[10px]"
                              style={{ borderColor: journeyId === p.id ? '#00b4ff' : '#2a3040', color: journeyId === p.id ? '#00b4ff' : '#5a6270', padding: '2px 8px' }}
                              title="View trade journey chart"
                            >
                              <Activity className="w-3 h-3" />
                            </button>
                            <button
                              onClick={() => handleExit(p.id)}
                              disabled={exiting === p.id}
                              className="t-btn flex items-center gap-1 text-[10px] disabled:opacity-50"
                              style={{ borderColor: '#ff3e3e', color: '#ff3e3e', padding: '2px 8px' }}
                            >
                              <X className="w-3 h-3" />
                              {exiting === p.id ? "..." : "EXIT"}
                            </button>
                          </div>
                        </td>
                      </tr>
                      {/* Journey chart row */}
                      {journeyId === p.id && journeyData && (
                        <tr key={`journey-${p.id}`}>
                          <td colSpan={10} style={{ padding: '12px 8px', background: '#0d1117', borderBottom: '1px solid #1e2530' }}>
                            <div style={{ marginBottom: 6, fontSize: 11, color: '#5a6270', display: 'flex', gap: 16, alignItems: 'center' }}>
                              <span style={{ color: '#e8e8e8', fontWeight: 600 }}>{journeyData.symbol} Journey</span>
                              <span>{journeyData.journey.length} data points</span>
                              <span style={{ color: '#00b4ff' }}>Entry ₹{journeyData.entry_premium}</span>
                              <span style={{ color: '#ff3e3e' }}>SL ₹{journeyData.initial_sl}</span>
                              <span style={{ color: '#00e87b' }}>Target ₹{journeyData.target}</span>
                              {journeyData.status === "OPEN" && <span style={{ color: '#e8c300', fontSize: 10 }}>● LIVE — refreshing every 5s</span>}
                            </div>
                            {journeyData.journey.length < 2 ? (
                              <div style={{ color: '#5a6270', fontSize: 11, padding: '8px 0' }}>Collecting data... check back in a few seconds.</div>
                            ) : (
                              <ResponsiveContainer width="100%" height={180}>
                                <LineChart data={journeyData.journey.map(pt => ({
                                  ...pt,
                                  time: pt.ts.slice(11, 19),
                                  entry: journeyData.entry_premium,
                                }))}>
                                  <CartesianGrid strokeDasharray="3 3" stroke="#1e2530" />
                                  <XAxis dataKey="time" tick={{ fontSize: 9, fill: '#5a6270' }} interval="preserveStartEnd" />
                                  <YAxis domain={['auto', 'auto']} tick={{ fontSize: 9, fill: '#5a6270' }} width={45} />
                                  <Tooltip
                                    contentStyle={{ background: '#0d1117', border: '1px solid #1e2530', fontSize: 11 }}
                                    formatter={(val: unknown, name: unknown) => [`₹${Number(val).toFixed(2)}`, String(name)]}
                                  />
                                  {/* Entry line */}
                                  <ReferenceLine y={journeyData.entry_premium} stroke="#00b4ff" strokeDasharray="4 2" label={{ value: 'Entry', fill: '#00b4ff', fontSize: 9 }} />
                                  {/* Initial SL line */}
                                  <ReferenceLine y={journeyData.initial_sl} stroke="#ff3e3e" strokeDasharray="4 2" label={{ value: 'SL', fill: '#ff3e3e', fontSize: 9 }} />
                                  {/* Target line */}
                                  <ReferenceLine y={journeyData.target} stroke="#00e87b" strokeDasharray="4 2" label={{ value: 'TGT', fill: '#00e87b', fontSize: 9 }} />
                                  {/* Option price */}
                                  <Line type="monotone" dataKey="option_price" stroke="#e8c300" dot={false} strokeWidth={2} name="Option ₹" />
                                  {/* Live SL (trailing) */}
                                  <Line type="stepAfter" dataKey="sl" stroke="#ff7a00" dot={false} strokeWidth={1} strokeDasharray="3 2" name="Live SL" />
                                </LineChart>
                              </ResponsiveContainer>
                            )}
                          </td>
                        </tr>
                      )}
                      </React.Fragment>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Signal feed */}
        <div className="t-panel p-4 mb-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-3">
              <h3 className="text-[11px] font-semibold uppercase tracking-wider" style={{ color: '#5a6270' }}>Trade Suggestions</h3>
              <span
                className="flex items-center gap-1 text-[9px] font-semibold px-2 py-0.5"
                style={{
                  border: `1px solid ${autoTrade ? '#00e87b55' : '#4da6ff55'}`,
                  color: autoTrade ? '#00e87b' : '#4da6ff',
                  background: autoTrade ? '#0a2a18' : '#0a1a2a',
                }}
              >
                {autoTrade ? <><Bot className="w-2.5 h-2.5" /> AUTO TRADING</> : <><Hand className="w-2.5 h-2.5" /> MANUAL</>}
              </span>
            </div>
            <span className="text-[10px]" style={{ color: '#3d4450' }}>AUTO-REFRESH 3S · 5MIN COOLDOWN PER SIGNAL</span>
          </div>
          {enterError && (
            <div className="mb-3 px-3 py-2 text-[11px]" style={{ background: '#2a0a0a', border: '1px solid #5c1a1a', color: '#ff3e3e' }}>
              ⚠ {enterError}
            </div>
          )}
          {!state?.trade_suggestions?.length ? (
            <div className="py-10 text-center">
              <p className="text-[11px]" style={{ color: '#3d4450' }}>NO SIGNALS — SCANNING EVERY 30S</p>
              <p className="text-[10px] mt-1" style={{ color: '#252a33' }}>SIGNALS APPEAR WHEN AI FINDS HIGH-PROBABILITY SETUPS</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table>
                <thead>
                  <tr>
                    {["Time", "Contract", "Dir", "Strategy", "Risk", "Entry ₹", "SL ₹", "Target ₹", "Lots", "Expiry", "ML%", "Score", "Regime", ""].map(h => (
                      <th key={h}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[...state.trade_suggestions].reverse().map((t, i) => {
                    const key = t.symbol + t.direction;
                    const isEntering = entering === key;
                    return (
                      <tr key={i}>
                        <td style={{ color: '#5a6270' }}>{t.time}</td>
                        <td style={{ color: '#c8cdd5', fontWeight: 600 }}>{t.symbol}</td>
                        <td><Badge label={t.direction} variant={t.direction === "CALL" ? "green" : "red"} /></td>
                        <td style={{ color: '#5a6270' }}>{t.strategy?.replace(/_/g, " ")}</td>
                        <td>
                          <span style={{
                            fontSize: '10px', fontWeight: 700, padding: '2px 6px',
                            borderRadius: '3px', letterSpacing: '0.05em',
                            background: t.risk_label === 'LOW' ? '#0d2b1a' : t.risk_label === 'MEDIUM' ? '#2a2000' : '#2a0a0a',
                            color: t.risk_label === 'LOW' ? '#00e87b' : t.risk_label === 'MEDIUM' ? '#e8c300' : '#ff3e3e',
                            border: `1px solid ${t.risk_label === 'LOW' ? '#00e87b33' : t.risk_label === 'MEDIUM' ? '#e8c30033' : '#ff3e3e33'}`,
                          }}>
                            {t.risk_label ?? '--'}
                          </span>
                        </td>
                        <td>{t.entry_premium ? `₹${t.entry_premium}` : "--"}</td>
                        <td style={{ color: '#ff3e3e' }}>{t.sl_price ? `₹${t.sl_price}` : "--"}</td>
                        <td style={{ color: '#00e87b' }}>{t.target_price ? `₹${t.target_price}` : "--"}</td>
                        <td style={{ color: t.lots && t.lots > 1 ? '#e8c300' : '#5a6270', fontWeight: t.lots && t.lots > 1 ? 700 : 400 }}>
                          {t.lots ?? 1}×
                        </td>
                        <td style={{ color: '#5a6270' }}>{t.expiry} ({t.dte}d)</td>
                        <td>{(t.ml_prob * 100).toFixed(0)}%</td>
                        <td>
                          <div className="flex items-center gap-2">
                            <div className="h-1 w-14 overflow-hidden" style={{ background: '#1e222c' }}>
                              <div className="h-full" style={{ width: `${t.final_score * 100}%`, background: '#4da6ff' }} />
                            </div>
                            <span className="text-[10px]">{(t.final_score * 100).toFixed(0)}%</span>
                          </div>
                        </td>
                        <td style={{ color: '#5a6270' }}>{t.regime}</td>
                        <td>
                          {autoTrade ? (
                            <span
                              className="flex items-center gap-1 text-[9px] font-semibold px-2 py-1"
                              style={{ color: '#00e87b', background: '#0a2a18', border: '1px solid #00e87b33' }}
                            >
                              <Bot className="w-2.5 h-2.5" /> AUTO
                            </span>
                          ) : (
                            <button
                              onClick={() => handleEnter(t)}
                              disabled={isEntering}
                              className="t-btn flex items-center gap-1 text-[10px] disabled:opacity-50"
                              style={{ borderColor: '#00e87b', color: '#00e87b', padding: '2px 8px', whiteSpace: 'nowrap' }}
                            >
                              {t.direction === "CALL"
                                ? <TrendingUp className="w-3 h-3" />
                                : <TrendingDown className="w-3 h-3" />}
                              {isEntering ? "..." : "ENTER"}
                            </button>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Closed Positions */}
        {closedPositions.length > 0 && (
          <div className="t-panel p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-[11px] font-semibold uppercase tracking-wider" style={{ color: '#5a6270' }}>
                Closed Positions ({closedPositions.length})
              </h3>
              <div className="flex items-center gap-3">
                <span className="text-xs font-bold" style={{ color: totalClosedPnl >= 0 ? '#00e87b' : '#ff3e3e' }}>
                  Realised: ₹{totalClosedPnl.toLocaleString("en-IN")}
                </span>
                <button onClick={handleClear} className="t-btn flex items-center gap-1 text-[10px]" style={{ padding: '2px 8px' }}>
                  <Trash2 className="w-3 h-3" /> CLEAR
                </button>
              </div>
            </div>
            <div className="overflow-x-auto">
              <table>
                <thead>
                  <tr>
                    {["Entry", "Exit", "Contract", "Dir", "Entry ₹", "Exit ₹", "Realised P&L", "Reason"].map(h => (
                      <th key={h}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[...closedPositions].reverse().map(p => {
                    const pnlColor = (p.realised_pnl ?? 0) >= 0 ? '#00e87b' : '#ff3e3e';
                    const reasonColor = p.exit_reason === "TARGET_HIT" ? '#00e87b'
                      : p.exit_reason === "SL_HIT" ? '#ff3e3e'
                      : p.exit_reason === "TRAILING_SL" ? '#e8c300'
                      : '#e8c300';
                    return (
                      <tr key={p.id}>
                        <td style={{ color: '#5a6270' }}>{p.entry_time}</td>
                        <td style={{ color: '#5a6270' }}>{p.exit_time}</td>
                        <td style={{ fontWeight: 600 }}>{p.symbol}</td>
                        <td><Badge label={p.direction} variant={p.direction === "CALL" ? "green" : "red"} /></td>
                        <td>₹{p.entry_premium}</td>
                        <td>₹{p.exit_premium}</td>
                        <td style={{ color: pnlColor, fontWeight: 700 }}>
                          {(p.realised_pnl ?? 0) >= 0 ? "+" : ""}₹{(p.realised_pnl ?? 0).toLocaleString("en-IN")}
                        </td>
                        <td style={{ color: reasonColor, fontSize: '10px', fontWeight: 600 }}>
                          {p.exit_reason?.replace(/_/g, " ")}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Model status — collapsed at bottom */}
        <div className="t-panel p-4 mt-4">
          <h3 className="text-[11px] font-semibold uppercase tracking-wider mb-3" style={{ color: '#5a6270' }}>Model & System Status</h3>
          <div className="flex flex-wrap gap-2">
            <span className="t-badge" style={{
              borderColor: state?.models_loaded ? '#1a5c3a' : '#5c1a1a',
              color: state?.models_loaded ? '#00e87b' : '#ff3e3e',
              background: state?.models_loaded ? '#0a2a18' : '#2a0a0a',
            }}>
              ML {state?.models_loaded ? "LOADED" : "NOT LOADED"}
            </span>
            <span className="t-badge" style={{
              borderColor: state?.db_connected ? '#1a5c3a' : '#5c1a1a',
              color: state?.db_connected ? '#00e87b' : '#ff3e3e',
              background: state?.db_connected ? '#0a2a18' : '#2a0a0a',
            }}>
              DB {state?.db_connected ? "OK" : "DOWN"}
            </span>
            {state?.strategy_models_loaded?.map(s => (
              <span key={s} className="t-badge" style={{ borderColor: '#1a3a5c', color: '#4da6ff', background: '#0a1a2a' }}>{s}</span>
            ))}
            <span className="t-badge ml-4" style={{ borderColor: '#2a2a1a', color: '#5a6270', background: '#1a1a0a' }}>
              Scans: {state?.scan_count ?? 0}
            </span>
            <span className="t-badge" style={{ borderColor: '#2a2a1a', color: '#5a6270', background: '#1a1a0a' }}>
              Last: {state?.last_scan ?? "--"}
            </span>
          </div>
        </div>

      </main>
    </div>
  );
}
