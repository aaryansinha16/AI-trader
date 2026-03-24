"use client";

import { useEffect, useState, useCallback } from "react";
import Sidebar from "@/components/Sidebar";
import StatCard from "@/components/StatCard";
import EquityChart from "@/components/EquityChart";
import TradeTable from "@/components/TradeTable";
import PnlBarChart from "@/components/PnlBarChart";
import { fetchJSON, type BacktestResults, type LiveState, type EquityCurvePoint } from "@/lib/api";
import { RefreshCw } from "lucide-react";

const pnlFmt = (v: number) =>
  `₹${v >= 0 ? "+" : ""}${v.toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;

export default function Home() {
  const [live, setLive] = useState<LiveState | null>(null);
  const [results, setResults] = useState<BacktestResults>({});
  const [curves, setCurves] = useState<Record<string, EquityCurvePoint[]>>({});
  const [activeRisk, setActiveRisk] = useState<"low" | "medium" | "high">("high");
  const [loading, setLoading] = useState(true);
  const [lastRefresh, setLastRefresh] = useState<string>("--");

  const load = useCallback(async () => {
    try {
      const [liveData, backtestData, curveData] = await Promise.all([
        fetchJSON<LiveState>("/api/state").catch(() => null),
        fetchJSON<BacktestResults>("/api/backtest/results").catch(() => ({})),
        fetchJSON<Record<string, EquityCurvePoint[]>>("/api/equity/curve").catch(() => ({})),
      ]);
      if (liveData) setLive(liveData);
      setResults(backtestData);
      setCurves(curveData);
      setLastRefresh(new Date().toLocaleTimeString("en-IN"));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, 5000);
    return () => clearInterval(id);
  }, [load]);

  const p = results[activeRisk];

  const riskColors: Record<string, string> = { low: "#4da6ff", medium: "#e8c300", high: "#00e87b" };

  return (
    <div className="flex min-h-screen">
      <Sidebar />

      <div className="flex-1 flex flex-col min-h-screen overflow-hidden">
        {/* ── Ticker Bar (top) ─────────────────────────────────── */}
        <div className="ticker-bar flex items-center gap-6 px-5 py-2">
          <div className="flex items-center gap-2">
            <span className={`w-[6px] h-[6px] ${live?.status === "scanning" ? "bg-[#00e87b] t-pulse" : live?.status === "idle" ? "bg-[#4da6ff]" : "bg-[#3d4450]"}`} />
            <span className="text-[10px] uppercase tracking-wider" style={{ color: '#5a6270' }}>
              {live?.status ?? "..."}
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] uppercase" style={{ color: '#3d4450' }}>NIFTY</span>
            <span className="text-[12px] font-bold" style={{ color: '#00e87b' }}>
              {live?.last_price ? `₹${live.last_price.toLocaleString("en-IN", { maximumFractionDigits: 1 })}` : "--"}
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] uppercase" style={{ color: '#3d4450' }}>Regime</span>
            <span className="text-[12px] font-semibold" style={{
              color: live?.regime?.includes("BULL") ? '#00e87b' : live?.regime?.includes("BEAR") ? '#ff3e3e' : '#e8c300'
            }}>
              {live?.regime ?? "--"}
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] uppercase" style={{ color: '#3d4450' }}>Scans</span>
            <span className="text-[12px]">{live?.scan_count ?? 0}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] uppercase" style={{ color: '#3d4450' }}>Signals</span>
            <span className="text-[12px]">{live?.signals_checked ?? 0}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] uppercase" style={{ color: '#3d4450' }}>Trades</span>
            <span className="text-[12px] font-bold" style={{ color: '#00e87b' }}>{live?.trades_today ?? 0}</span>
          </div>
          <div className="ml-auto flex items-center gap-3">
            <span className="text-[10px]" style={{ color: '#3d4450' }}>{lastRefresh}</span>
            <button onClick={load} className="t-btn flex items-center gap-1.5">
              <RefreshCw className="w-3 h-3" /> REFRESH
            </button>
          </div>
        </div>

        {/* ── Main content ─────────────────────────────────────── */}
        <main className="flex-1 p-5 overflow-y-auto">

          {/* Risk profile tabs */}
          <div className="flex gap-[1px] mb-5">
            {(["low", "medium", "high"] as const).map(r => (
              <button
                key={r}
                onClick={() => setActiveRisk(r)}
                className="px-4 py-[6px] text-[10px] font-semibold tracking-wider uppercase transition-all"
                style={{
                  background: activeRisk === r ? riskColors[r] : '#181c24',
                  color: activeRisk === r ? '#000' : '#5a6270',
                  border: `1px solid ${activeRisk === r ? riskColors[r] : '#252a33'}`,
                }}
              >
                {r} Risk
              </button>
            ))}
          </div>

          {/* KPI cards */}
          {loading ? (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-[1px] mb-5">
              {[...Array(8)].map((_, i) => (
                <div key={i} className="t-panel p-3 h-16 animate-pulse" />
              ))}
            </div>
          ) : p ? (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-[1px] mb-5">
              <StatCard label="Total P&L" value={pnlFmt(p.pnl)} color={p.pnl >= 0 ? "green" : "red"} />
              <StatCard label="Win Rate" value={`${p.win_rate}%`} color={p.win_rate >= 55 ? "green" : p.win_rate >= 45 ? "yellow" : "red"} />
              <StatCard label="Total Trades" value={p.trades} sub={`${(p.pnl / Math.max(p.trades, 1)).toFixed(0)} avg/trade`} />
              <StatCard label="Risk-Reward" value={`${p.rr}×`} color={p.rr >= 1.5 ? "green" : p.rr >= 1.0 ? "yellow" : "red"} />
              <StatCard label="Avg Winner" value={pnlFmt(p.avg_win)} color="green" />
              <StatCard label="Avg Loser" value={pnlFmt(p.avg_loss)} color="red" />
              <StatCard label="Max Drawdown" value={pnlFmt(p.max_dd)} color="red" />
              <StatCard label="Profit Factor" value={p.avg_loss !== 0 ? (p.avg_win / Math.abs(p.avg_loss)).toFixed(2) : "∞"} color="blue" />
            </div>
          ) : (
            <div className="t-panel p-5 mb-5 text-center text-[11px]" style={{ color: '#3d4450' }}>
              NO BACKTEST DATA FOR <span style={{ color: riskColors[activeRisk] }}>{activeRisk.toUpperCase()}</span> RISK.
              RUN <code style={{ color: '#4da6ff' }}>python scripts/tick_replay_backtest.py --risk {activeRisk}</code>
            </div>
          )}

          {/* Charts row */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-[1px] mb-5">
            <div className="t-panel p-4">
              <h3 className="text-[11px] font-semibold mb-3 uppercase tracking-wider" style={{ color: '#5a6270' }}>
                Equity Curves
              </h3>
              <EquityChart curves={curves} selected="all" />
            </div>
            <div className="t-panel p-4">
              <h3 className="text-[11px] font-semibold mb-3 uppercase tracking-wider" style={{ color: '#5a6270' }}>
                Per-Trade P&L — <span style={{ color: riskColors[activeRisk] }}>{activeRisk}</span>
              </h3>
              {p?.trade_list ? <PnlBarChart trades={p.trade_list} /> : (
                <div className="h-48 flex items-center justify-center text-[11px]" style={{ color: '#3d4450' }}>NO DATA</div>
              )}
            </div>
          </div>

          {/* Risk profile comparison */}
          <div className="t-panel p-4 mb-5">
            <h3 className="text-[11px] font-semibold mb-3 uppercase tracking-wider" style={{ color: '#5a6270' }}>
              Risk Profile Comparison
            </h3>
            <div className="overflow-x-auto">
              <table>
                <thead>
                  <tr>
                    {["Profile", "Trades", "Total P&L", "Win Rate", "R:R", "Avg/Trade", "Max DD"].map(h => (
                      <th key={h}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(["low", "medium", "high"] as const).map(r => {
                    const rp = results[r];
                    if (!rp) return null;
                    return (
                      <tr key={r} style={{ background: activeRisk === r ? '#1e222c' : undefined }}>
                        <td>
                          <span className="font-semibold uppercase" style={{ color: riskColors[r] }}>{r}</span>
                        </td>
                        <td>{rp.trades}</td>
                        <td style={{ color: rp.pnl >= 0 ? '#00e87b' : '#ff3e3e', fontWeight: 600 }}>{pnlFmt(rp.pnl)}</td>
                        <td>{rp.win_rate}%</td>
                        <td>{rp.rr}×</td>
                        <td>{pnlFmt(rp.pnl / Math.max(rp.trades, 1))}</td>
                        <td style={{ color: '#ff3e3e' }}>{pnlFmt(rp.max_dd)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Recent trades */}
          <div className="t-panel p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-[11px] font-semibold uppercase tracking-wider" style={{ color: '#5a6270' }}>
                Recent Trades — <span style={{ color: riskColors[activeRisk] }}>{activeRisk}</span>
              </h3>
              <a href="/trades" className="text-[10px] uppercase tracking-wider" style={{ color: '#4da6ff' }}>View all →</a>
            </div>
            {p?.trade_list ? (
              <TradeTable trades={p.trade_list} maxRows={10} />
            ) : (
              <p className="text-[11px] text-center py-6" style={{ color: '#3d4450' }}>NO TRADES</p>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
