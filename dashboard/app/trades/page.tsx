"use client";

import React, { useEffect, useState, useCallback, useRef } from "react";
import Sidebar from "@/components/Sidebar";
import TradeTable from "@/components/TradeTable";
import PnlBarChart from "@/components/PnlBarChart";
import { fetchJSON, type Trade, type LiveTrade, type JourneyPoint } from "@/lib/api";
import { toDateStr, toISTTimeFull } from "@/lib/time";
import { useTradingMode } from "@/contexts/TradingModeContext";
import { Download, Activity } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ReferenceLine, ResponsiveContainer, CartesianGrid } from "recharts";

type RiskLevel = "low" | "medium" | "high";
type TabMode = "backtest" | "live";

const riskColors: Record<string, string> = { low: "#4da6ff", medium: "#e8c300", high: "#00e87b" };
const pnlFmt = (v: number) =>
  `₹${v >= 0 ? "+" : ""}${v.toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;

function resultVariant(r: string) {
  if (r === "TARGET" || r === "TARGET_HIT") return "green";
  if (r === "SL" || r === "TRAILING_SL" || r === "SL_HIT") return "red";
  if (r === "RL_EXIT" || r === "DQN_EXIT") return "purple";
  return "yellow";
}

// ── Journey chart (shared for both live and backtest trades) ─────────────────
function JourneyChart({ journey, entryPremium, initialSl, target, symbol }: {
  journey: JourneyPoint[];
  entryPremium: number;
  initialSl: number;
  target: number;
  symbol: string;
}) {
  if (journey.length < 2) {
    return <div style={{ color: "#5a6270", fontSize: 11, padding: "8px 0" }}>Not enough data points.</div>;
  }

  const data = journey.map((pt, i) => ({
    ...pt,
    premium: pt.option_price ?? pt.premium ?? 0,
    // Format time label
    time: pt.ts.length > 10 ? pt.ts.slice(11, 16) : `Bar ${i}`,
    entry: entryPremium,
  }));

  return (
    <div>
      <div style={{ marginBottom: 6, fontSize: 11, color: "#5a6270", display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
        <span style={{ color: "#e8e8e8", fontWeight: 600 }}>{symbol} Journey</span>
        <span>{journey.length} data points</span>
        <span style={{ color: "#00b4ff" }}>Entry ₹{entryPremium}</span>
        <span style={{ color: "#ff3e3e" }}>SL ₹{initialSl.toFixed(1)}</span>
        <span style={{ color: "#00e87b" }}>Target ₹{target.toFixed(1)}</span>
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e2530" />
          <XAxis dataKey="time" tick={{ fontSize: 9, fill: "#5a6270" }} interval="preserveStartEnd" />
          <YAxis domain={["auto", "auto"]} tick={{ fontSize: 9, fill: "#5a6270" }} width={50} />
          <Tooltip
            contentStyle={{ background: "#0d1117", border: "1px solid #1e2530", fontSize: 11 }}
            formatter={(val: unknown, name: unknown) => [`₹${Number(val).toFixed(2)}`, String(name)]}
          />
          <ReferenceLine y={entryPremium} stroke="#00b4ff" strokeDasharray="4 2" label={{ value: "Entry", fill: "#00b4ff", fontSize: 9 }} />
          <ReferenceLine y={initialSl} stroke="#ff3e3e" strokeDasharray="4 2" label={{ value: "SL", fill: "#ff3e3e", fontSize: 9 }} />
          <ReferenceLine y={target} stroke="#00e87b" strokeDasharray="4 2" label={{ value: "TGT", fill: "#00e87b", fontSize: 9 }} />
          <Line type="monotone" dataKey="premium" stroke="#e8c300" dot={false} strokeWidth={2} name="Option ₹" />
          <Line type="stepAfter" dataKey="sl" stroke="#ff7a00" dot={false} strokeWidth={1} strokeDasharray="3 2" name="Live SL" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Live trades table with inline journey ────────────────────────────────────
function LiveTradeRows({ trades }: { trades: LiveTrade[] }) {
  const [journeyId, setJourneyId] = useState<number | null>(null);
  const [journeyData, setJourneyData] = useState<{ journey: JourneyPoint[]; entry_premium: number; initial_sl: number; target: number; symbol: string } | null>(null);

  const toggleJourney = async (t: LiveTrade) => {
    if (journeyId === t.id) {
      setJourneyId(null);
      setJourneyData(null);
      return;
    }
    setJourneyId(t.id);
    // Use the journey embedded in the trade dict (already loaded)
    setJourneyData({
      journey: t.journey || [],
      entry_premium: t.entry_premium,
      initial_sl: t.initial_sl,
      target: t.target,
      symbol: t.symbol,
    });
  };

  if (!trades.length) {
    return <tr><td colSpan={11} style={{ color: "#3d4450", textAlign: "center", padding: "24px" }}>NO LIVE TRADES TO DISPLAY</td></tr>;
  }

  return (
    <>
      {trades.map((t) => {
        const pnl = t.realised_pnl ?? 0;
        const pnlColor = pnl > 0 ? "#00e87b" : pnl < 0 ? "#ff3e3e" : "#5a6270";
        const dateStr = t.entry_time_dt ? t.entry_time_dt.slice(0, 10) : "—";
        const timeStr = t.entry_time || "—";
        return (
          <React.Fragment key={t.id}>
            <tr>
              <td style={{ color: "#5a6270" }}>{dateStr}</td>
              <td style={{ color: "#5a6270" }}>{timeStr}</td>
              <td style={{ fontWeight: 600 }}>{t.symbol}</td>
              <td>
                <span style={{
                  fontSize: 9, fontWeight: 700, padding: "2px 5px", letterSpacing: "0.5px",
                  background: t.direction === "CALL" ? "#0a2a18" : "#2a0a0a",
                  color: t.direction === "CALL" ? "#00e87b" : "#ff3e3e",
                  border: `1px solid ${t.direction === "CALL" ? "#1a5c3a" : "#5c1a1a"}`,
                }}>{t.direction}</span>
              </td>
              <td style={{ color: "#5a6270" }}>{t.strategy?.replace(/_/g, " ")}</td>
              <td>₹{t.entry_premium.toFixed(1)}</td>
              <td>₹{(t.exit_premium ?? t.current_premium)?.toFixed(1) ?? "—"}</td>
              <td style={{ color: pnlColor, fontWeight: 600 }}>
                {pnlFmt(pnl)}
              </td>
              <td>
                <span style={{
                  fontSize: 9, fontWeight: 700, padding: "2px 5px",
                  background: t.exit_reason === "TARGET_HIT" ? "#0a2a18" : t.exit_reason === "SL_HIT" || t.exit_reason === "TRAILING_SL" ? "#2a0a0a" : "#1a1a2a",
                  color: t.exit_reason === "TARGET_HIT" ? "#00e87b" : t.exit_reason === "SL_HIT" || t.exit_reason === "TRAILING_SL" ? "#ff3e3e" : "#e8c300",
                  border: `1px solid ${t.exit_reason === "TARGET_HIT" ? "#1a5c3a" : t.exit_reason === "SL_HIT" || t.exit_reason === "TRAILING_SL" ? "#5c1a1a" : "#333a45"}`,
                }}>{t.exit_reason ?? "OPEN"}</span>
              </td>
              <td style={{ color: "#5a6270" }}>{(t.final_score * 100).toFixed(0)}%</td>
              <td>
                <button
                  onClick={() => toggleJourney(t)}
                  style={{
                    background: "transparent", border: `1px solid ${journeyId === t.id ? "#00b4ff" : "#252a33"}`,
                    color: journeyId === t.id ? "#00b4ff" : "#5a6270", padding: "2px 6px", cursor: "pointer",
                    display: "flex", alignItems: "center", gap: 4, fontSize: 10,
                  }}
                  title="View trade journey"
                >
                  <Activity size={10} /> Journey
                </button>
              </td>
            </tr>
            {journeyId === t.id && journeyData && (
              <tr>
                <td colSpan={11} style={{ padding: "12px 8px", background: "#0d1117", borderBottom: "1px solid #1e2530" }}>
                  <JourneyChart
                    journey={journeyData.journey}
                    entryPremium={journeyData.entry_premium}
                    initialSl={journeyData.initial_sl}
                    target={journeyData.target}
                    symbol={journeyData.symbol}
                  />
                </td>
              </tr>
            )}
          </React.Fragment>
        );
      })}
    </>
  );
}

// ── Backtest trade rows with inline journey ──────────────────────────────────
function BacktestTradeRows({ trades, risk }: { trades: Trade[]; risk: RiskLevel }) {
  const [journeyIdx, setJourneyIdx] = useState<number | null>(null);
  const [journeyData, setJourneyData] = useState<{ journey: JourneyPoint[]; entry_premium: number; initial_sl: number; target: number; symbol: string } | null>(null);
  const [loadingJourney, setLoadingJourney] = useState(false);

  const toggleJourney = async (t: Trade, idx: number) => {
    if (journeyIdx === idx) {
      setJourneyIdx(null);
      setJourneyData(null);
      return;
    }
    setJourneyIdx(idx);
    setLoadingJourney(true);
    try {
      const data = await fetchJSON<{ journey: JourneyPoint[] }>(`/api/backtest/journey/${risk}/${idx}`);
      setJourneyData({
        journey: data.journey,
        entry_premium: t.entry_premium,
        initial_sl: t.sl,
        target: t.target,
        symbol: t.symbol,
      });
    } catch {
      setJourneyData({ journey: [], entry_premium: t.entry_premium, initial_sl: t.sl, target: t.target, symbol: t.symbol });
    } finally {
      setLoadingJourney(false);
    }
  };

  if (!trades.length) {
    return <tr><td colSpan={12} style={{ color: "#3d4450", textAlign: "center", padding: "24px" }}>NO TRADES TO DISPLAY</td></tr>;
  }

  return (
    <>
      {[...trades].reverse().map((t, i) => {
        const origIdx = trades.length - 1 - i; // original 0-based index for journey lookup
        return (
          <React.Fragment key={i}>
            <tr>
              <td style={{ color: "#5a6270" }}>{toDateStr(String(t.entry_time))}</td>
              <td style={{ color: "#5a6270" }}>{toISTTimeFull(String(t.entry_time))}</td>
              <td style={{ color: "#c8cdd5" }}>{t.symbol}</td>
              <td>
                <span style={{
                  fontSize: 9, fontWeight: 700, padding: "2px 5px",
                  background: t.direction === "CALL" ? "#0a2a18" : "#2a0a0a",
                  color: t.direction === "CALL" ? "#00e87b" : "#ff3e3e",
                  border: `1px solid ${t.direction === "CALL" ? "#1a5c3a" : "#5c1a1a"}`,
                }}>{t.direction}</span>
              </td>
              <td style={{ color: "#5a6270" }}>{t.strategy?.replace(/_/g, " ")}</td>
              <td>₹{t.entry_premium?.toFixed(1)}</td>
              <td>₹{t.exit_premium?.toFixed(1) ?? "—"}</td>
              <td style={{ color: t.pnl > 0 ? "#00e87b" : t.pnl < 0 ? "#ff3e3e" : "#5a6270", fontWeight: 600 }}>
                {pnlFmt(t.pnl)}
              </td>
              <td>
                <span style={{
                  fontSize: 9, fontWeight: 700, padding: "2px 5px",
                  background: t.result === "TARGET" ? "#0a2a18" : t.result === "SL" || t.result === "TRAILING_SL" ? "#2a0a0a" : "#1a1a2a",
                  color: t.result === "TARGET" ? "#00e87b" : t.result === "SL" || t.result === "TRAILING_SL" ? "#ff3e3e" : "#e8c300",
                  border: `1px solid ${t.result === "TARGET" ? "#1a5c3a" : t.result === "SL" || t.result === "TRAILING_SL" ? "#5c1a1a" : "#333a45"}`,
                }}>{t.result}</span>
              </td>
              <td>{(t.final_score * 100).toFixed(0)}%</td>
              <td style={{ color: "#5a6270" }}>{t.regime}</td>
              <td>
                <button
                  onClick={() => toggleJourney(t, origIdx)}
                  style={{
                    background: "transparent", border: `1px solid ${journeyIdx === origIdx ? "#00b4ff" : "#252a33"}`,
                    color: journeyIdx === origIdx ? "#00b4ff" : "#5a6270", padding: "2px 6px", cursor: "pointer",
                    display: "flex", alignItems: "center", gap: 4, fontSize: 10,
                  }}
                  title="View trade journey"
                >
                  <Activity size={10} /> {loadingJourney && journeyIdx === origIdx ? "..." : "Journey"}
                </button>
              </td>
            </tr>
            {journeyIdx === origIdx && journeyData && (
              <tr>
                <td colSpan={12} style={{ padding: "12px 8px", background: "#0d1117", borderBottom: "1px solid #1e2530" }}>
                  <JourneyChart
                    journey={journeyData.journey}
                    entryPremium={journeyData.entry_premium}
                    initialSl={journeyData.initial_sl}
                    target={journeyData.target}
                    symbol={journeyData.symbol}
                  />
                </td>
              </tr>
            )}
          </React.Fragment>
        );
      })}
    </>
  );
}

// ── Main page ────────────────────────────────────────────────────────────────
export default function TradesPage() {
  const { mode: tradingMode } = useTradingMode();
  const [tabMode, setTabMode] = useState<TabMode>("backtest");
  const [risk, setRisk] = useState<RiskLevel>("high");
  const [filter, setFilter] = useState<"ALL" | "CALL" | "PUT" | "WIN" | "LOSS" | "RL_EXIT">("ALL");

  // Backtest
  const [btTrades, setBtTrades] = useState<Trade[]>([]);
  const [btLoading, setBtLoading] = useState(true);

  // Live
  const [liveTrades, setLiveTrades] = useState<LiveTrade[]>([]);
  const [liveLoading, setLiveLoading] = useState(true);

  const loadBacktest = useCallback(async (r: RiskLevel) => {
    setBtLoading(true);
    try {
      const data = await fetchJSON<Trade[]>(`/api/trades/history?risk=${r}`);
      setBtTrades(Array.isArray(data) ? data : []);
    } finally {
      setBtLoading(false);
    }
  }, []);

  const loadLive = useCallback(async () => {
    setLiveLoading(true);
    try {
      const data = await fetchJSON<LiveTrade[]>(`/api/paper/trades?mode=${tradingMode}`);
      setLiveTrades(Array.isArray(data) ? data : []);
    } finally {
      setLiveLoading(false);
    }
  }, [tradingMode]);

  useEffect(() => { loadBacktest(risk); }, [risk, loadBacktest]);
  useEffect(() => { if (tabMode === "live") loadLive(); }, [tabMode, loadLive]);

  // Filtered backtest trades
  const filteredBt = btTrades.filter(t => {
    if (filter === "ALL") return true;
    if (filter === "CALL" || filter === "PUT") return t.direction === filter;
    if (filter === "WIN") return t.pnl > 0;
    if (filter === "LOSS") return t.pnl <= 0;
    if (filter === "RL_EXIT") return t.result === "RL_EXIT" || t.result === "DQN_EXIT";
    return true;
  });

  // Filtered live trades
  const filteredLive = liveTrades.filter(t => {
    if (filter === "ALL") return true;
    if (filter === "CALL" || filter === "PUT") return t.direction === filter;
    if (filter === "WIN") return (t.realised_pnl ?? 0) > 0;
    if (filter === "LOSS") return (t.realised_pnl ?? 0) <= 0;
    return true;
  });

  // Stats
  const activeCount = tabMode === "backtest" ? filteredBt.length : filteredLive.length;
  const totalPnl = tabMode === "backtest"
    ? filteredBt.reduce((s, t) => s + t.pnl, 0)
    : filteredLive.reduce((s, t) => s + (t.realised_pnl ?? 0), 0);
  const wins = tabMode === "backtest"
    ? filteredBt.filter(t => t.pnl > 0).length
    : filteredLive.filter(t => (t.realised_pnl ?? 0) > 0).length;
  const losses = activeCount - wins;
  const winRate = activeCount > 0 ? (wins / activeCount * 100).toFixed(1) : "--";

  // Backtest strategy breakdown
  const strategies = Array.from(new Set(btTrades.map(t => t.strategy)));
  const byStrategy = strategies.map(s => ({
    strategy: s,
    trades: btTrades.filter(t => t.strategy === s).length,
    pnl: btTrades.filter(t => t.strategy === s).reduce((sum, t) => sum + t.pnl, 0),
    wr: btTrades.filter(t => t.strategy === s).length > 0
      ? (btTrades.filter(t => t.strategy === s && t.pnl > 0).length / btTrades.filter(t => t.strategy === s).length * 100).toFixed(0)
      : "0",
  }));

  const isLoading = tabMode === "backtest" ? btLoading : liveLoading;

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 p-5 overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-5">
          <div>
            <h1 className="text-sm font-bold uppercase tracking-wider" style={{ color: "#00e87b" }}>Trade History</h1>
            <p className="text-[10px] mt-0.5" style={{ color: "#3d4450" }}>
              {tabMode === "backtest" ? "COMPLETED TRADES FROM BACKTEST" : `LIVE PAPER TRADES — ${tradingMode.toUpperCase()} MODE`}
            </p>
          </div>
          <a
            href={tabMode === "backtest"
              ? `http://localhost:5050/api/trades/history?risk=${risk}`
              : `http://localhost:5050/api/paper/trades?mode=${tradingMode}`}
            download={tabMode === "backtest" ? `bt_trades_${risk}.json` : `live_trades_${tradingMode}.json`}
            className="t-btn flex items-center gap-1.5"
          >
            <Download className="w-3 h-3" /> EXPORT
          </a>
        </div>

        {/* ── Mode toggle ─────────────────────────────────────────────────── */}
        <div className="flex gap-[1px] mb-4">
          {([["backtest", "Backtest"], ["live", "Live Trades"]] as [TabMode, string][]).map(([m, label]) => (
            <button key={m} onClick={() => setTabMode(m)}
              className="px-5 py-[6px] text-[10px] font-semibold tracking-wider uppercase transition-all"
              style={{
                background: tabMode === m ? "#00e87b" : "#181c24",
                color: tabMode === m ? "#000" : "#5a6270",
                border: `1px solid ${tabMode === m ? "#00e87b" : "#252a33"}`,
              }}>
              {label}
            </button>
          ))}
        </div>

        {/* ── Backtest risk tabs (backtest mode only) ──────────────────────── */}
        {tabMode === "backtest" && (
          <div className="flex gap-[1px] mb-4">
            {(["low", "medium", "high"] as RiskLevel[]).map(r => (
              <button key={r} onClick={() => setRisk(r)}
                className="px-4 py-[6px] text-[10px] font-semibold tracking-wider uppercase transition-all"
                style={{
                  background: risk === r ? riskColors[r] : "#181c24",
                  color: risk === r ? "#000" : "#5a6270",
                  border: `1px solid ${risk === r ? riskColors[r] : "#252a33"}`,
                }}>
                {r} Risk
              </button>
            ))}
          </div>
        )}

        {/* ── Summary stats ────────────────────────────────────────────────── */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-[1px] mb-4">
          {[
            { label: "Total Trades", value: activeCount, color: "#c8cdd5" },
            { label: "Total P&L", value: pnlFmt(totalPnl), color: totalPnl >= 0 ? "#00e87b" : "#ff3e3e" },
            { label: "Win Rate", value: `${winRate}%`, color: "#c8cdd5" },
            { label: "Winners", value: wins, color: "#00e87b" },
            { label: "Losers", value: losses, color: "#ff3e3e" },
          ].map(({ label, value, color }) => (
            <div key={label} className="t-panel p-3">
              <p className="text-[9px] uppercase tracking-[1.5px] mb-1" style={{ color: "#5a6270" }}>{label}</p>
              <p className="text-xl font-bold" style={{ color }}>{value}</p>
            </div>
          ))}
        </div>

        {/* ── Filter buttons ───────────────────────────────────────────────── */}
        <div className="flex gap-[1px] mb-4">
          {(["ALL", "CALL", "PUT", "WIN", "LOSS", ...(tabMode === "backtest" ? ["RL_EXIT"] : [])] as const).map(f => (
            <button key={f} onClick={() => setFilter(f as typeof filter)}
              className="px-3 py-[5px] text-[10px] font-semibold tracking-wider transition-all"
              style={{
                background: filter === f ? "#252a33" : "#181c24",
                color: filter === f ? "#c8cdd5" : "#3d4450",
                border: `1px solid ${filter === f ? "#333a45" : "#252a33"}`,
              }}>
              {f}
            </button>
          ))}
        </div>

        {/* ── P&L bar chart (backtest only for now) ────────────────────────── */}
        {tabMode === "backtest" && filteredBt.length > 0 && (
          <div className="t-panel p-4 mb-4">
            <h3 className="text-[11px] font-semibold mb-3 uppercase tracking-wider" style={{ color: "#5a6270" }}>Per-Trade P&L</h3>
            <PnlBarChart trades={filteredBt} />
          </div>
        )}

        {/* ── Strategy breakdown (backtest only) ──────────────────────────── */}
        {tabMode === "backtest" && byStrategy.length > 0 && (
          <div className="t-panel p-4 mb-4">
            <h3 className="text-[11px] font-semibold mb-3 uppercase tracking-wider" style={{ color: "#5a6270" }}>Strategy Breakdown</h3>
            <table>
              <thead>
                <tr>{["Strategy", "Trades", "P&L", "Win Rate"].map(h => <th key={h}>{h}</th>)}</tr>
              </thead>
              <tbody>
                {byStrategy.map(s => (
                  <tr key={s.strategy}>
                    <td style={{ color: "#c8cdd5" }}>{s.strategy?.replace(/_/g, " ")}</td>
                    <td>{s.trades}</td>
                    <td style={{ color: s.pnl >= 0 ? "#00e87b" : "#ff3e3e", fontWeight: 600 }}>{pnlFmt(s.pnl)}</td>
                    <td>{s.wr}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* ── Trade table ─────────────────────────────────────────────────── */}
        <div className="t-panel p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-[11px] font-semibold uppercase tracking-wider" style={{ color: "#5a6270" }}>
              {tabMode === "backtest" ? `All Backtest Trades${filter !== "ALL" ? ` (${filter})` : ""} — ${filteredBt.length} records`
                : `Live Paper Trades — ${filteredLive.length} records`}
            </h3>
            {tabMode === "live" && (
              <button
                onClick={loadLive}
                className="t-btn text-[10px]"
                style={{ borderColor: "#252a33", color: "#5a6270", padding: "2px 8px" }}
              >
                Refresh
              </button>
            )}
          </div>

          {isLoading ? (
            <div className="h-32 flex items-center justify-center text-[11px]" style={{ color: "#3d4450" }}>LOADING...</div>
          ) : tabMode === "backtest" ? (
            <div className="overflow-x-auto">
              <table>
                <thead>
                  <tr>
                    {["Date", "Time", "Symbol", "Dir", "Strategy", "Entry", "Exit", "P&L", "Result", "Score", "Regime", ""].map(h => (
                      <th key={h}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  <BacktestTradeRows trades={filteredBt} risk={risk} />
                </tbody>
              </table>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table>
                <thead>
                  <tr>
                    {["Date", "Time", "Symbol", "Dir", "Strategy", "Entry", "Exit", "P&L", "Reason", "Score", ""].map(h => (
                      <th key={h}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  <LiveTradeRows trades={filteredLive} />
                </tbody>
              </table>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
