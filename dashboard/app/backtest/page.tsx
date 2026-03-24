"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import Sidebar from "@/components/Sidebar";
import EquityChart from "@/components/EquityChart";
import RiskProfileCard from "@/components/RiskProfileCard";
import TradeTable from "@/components/TradeTable";
import { fetchJSON, postJSON, type BacktestResults, type RiskProfile, type EquityCurvePoint } from "@/lib/api";
import { Play, RefreshCw, Terminal, Calendar } from "lucide-react";

type RiskLevel = "low" | "medium" | "high";
const riskColors: Record<string, string> = { low: "#4da6ff", medium: "#e8c300", high: "#00e87b" };

const pnlFmt = (v: number) =>
  `₹${v >= 0 ? "+" : ""}${v.toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;

interface BacktestProgress {
  running: boolean;
  risk: string | null;
  status: string;
  output_lines: string[];
  started?: string;
  finished?: string;
  exit_code?: number;
  start_date?: string | null;
  end_date?: string | null;
}

interface AvailableDay { day: string; ticks: number; }

export default function BacktestPage() {
  const [results, setResults] = useState<BacktestResults>({});
  const [curves, setCurves] = useState<Record<string, EquityCurvePoint[]>>({});
  const [profiles, setProfiles] = useState<Record<RiskLevel, RiskProfile> | null>(null);
  const [selectedRisk, setSelectedRisk] = useState<RiskLevel>("medium");
  const [loading, setLoading] = useState(true);
  const [progress, setProgress] = useState<BacktestProgress | null>(null);
  const termRef = useRef<HTMLDivElement>(null);

  // Date range
  const [availDays, setAvailDays] = useState<AvailableDay[]>([]);
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");

  const datesInitRef = useRef(false);
  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [r, c, p, days] = await Promise.all([
        fetchJSON<BacktestResults>("/api/backtest/results").catch(() => ({})),
        fetchJSON<Record<string, EquityCurvePoint[]>>("/api/equity/curve").catch(() => ({})),
        fetchJSON<Record<RiskLevel, RiskProfile>>("/api/risk/profiles").catch(() => null),
        fetchJSON<AvailableDay[]>("/api/days").catch(() => []),
      ]);
      setResults(r);
      setCurves(c);
      if (p) setProfiles(p as Record<RiskLevel, RiskProfile>);
      if (days.length > 0) {
        setAvailDays(days);
        if (!datesInitRef.current) {
          datesInitRef.current = true;
          setStartDate(String(days[0].day));
          setEndDate(String(days[days.length - 1].day));
        }
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  // Poll backtest progress when running
  useEffect(() => {
    const poll = async () => {
      const p = await fetchJSON<BacktestProgress>("/api/backtest/progress").catch(() => null);
      if (p) {
        setProgress(p);
        // Auto-scroll terminal
        if (termRef.current) termRef.current.scrollTop = termRef.current.scrollHeight;
        // Reload results when done
        if (!p.running && p.status === "done") {
          load();
        }
      }
    };
    poll();
    const id = setInterval(poll, 1000);
    return () => clearInterval(id);
  }, [load]);

  const runBacktest = async (risk: RiskLevel) => {
    try {
      const body: Record<string, string> = { risk };
      if (startDate) body.start_date = startDate;
      if (endDate) body.end_date = endDate;
      const res = await postJSON<{ status?: string; error?: string }>("/api/backtest/run", body);
      if (res.error) {
        setProgress(prev => prev ? { ...prev, output_lines: [...(prev.output_lines || []), `ERROR: ${res.error}`] } : null);
      }
    } catch {
      // Will be picked up by progress polling
    }
  };

  const isRunning = progress?.running === true;
  const p = results[selectedRisk];

  // Count available days in selected range
  const daysInRange = availDays.filter(d => {
    const ds = String(d.day);
    return (!startDate || ds >= startDate) && (!endDate || ds <= endDate);
  }).length;

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 p-5 overflow-y-auto">
        <div className="flex items-center justify-between mb-5">
          <div>
            <h1 className="text-sm font-bold uppercase tracking-wider" style={{ color: '#00e87b' }}>Backtest</h1>
            <p className="text-[10px] mt-0.5" style={{ color: '#3d4450' }}>RUN & COMPARE ACROSS RISK PROFILES</p>
          </div>
          <button onClick={load} className="t-btn flex items-center gap-1.5">
            <RefreshCw className="w-3 h-3" /> REFRESH
          </button>
        </div>

        {/* Date range picker */}
        <div className="t-panel px-4 py-3 mb-5 flex items-center gap-4 flex-wrap">
          <div className="flex items-center gap-2">
            <Calendar className="w-3.5 h-3.5" style={{ color: '#4da6ff' }} />
            <span className="text-[10px] font-bold uppercase tracking-wider" style={{ color: '#5a6270' }}>DATE RANGE</span>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-[9px] uppercase tracking-wider" style={{ color: '#3d4450' }}>FROM</label>
            <input
              type="date"
              value={startDate}
              onChange={e => setStartDate(e.target.value)}
              className="text-[11px] px-2 py-1"
              style={{
                background: '#181c24', border: '1px solid #252a33', color: '#c8cdd5',
                fontFamily: "'JetBrains Mono', monospace",
              }}
            />
          </div>
          <div className="flex items-center gap-2">
            <label className="text-[9px] uppercase tracking-wider" style={{ color: '#3d4450' }}>TO</label>
            <input
              type="date"
              value={endDate}
              onChange={e => setEndDate(e.target.value)}
              className="text-[11px] px-2 py-1"
              style={{
                background: '#181c24', border: '1px solid #252a33', color: '#c8cdd5',
                fontFamily: "'JetBrains Mono', monospace",
              }}
            />
          </div>
          <span className="text-[9px]" style={{ color: '#5a6270' }}>
            {daysInRange} TICK-DATA DAYS IN RANGE
            {availDays.length > 0 && (
              <span style={{ color: '#3d4450' }}> / {availDays.length} TOTAL</span>
            )}
          </span>
          {availDays.length > 0 && (
            <button
              onClick={() => {
                setStartDate(String(availDays[0].day));
                setEndDate(String(availDays[availDays.length - 1].day));
              }}
              className="text-[9px] font-bold uppercase tracking-wider px-2 py-0.5"
              style={{ background: '#181c24', border: '1px solid #252a33', color: '#4da6ff' }}
            >
              ALL DATES
            </button>
          )}
        </div>

        {/* Risk profile cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-[1px] mb-5">
          {profiles && (["low", "medium", "high"] as RiskLevel[]).map(r => (
            <div key={r} className="space-y-[1px]">
              <RiskProfileCard
                level={r}
                profile={profiles[r]}
                active={selectedRisk === r}
                onSelect={setSelectedRisk}
              />
              <button
                onClick={() => runBacktest(r)}
                disabled={isRunning}
                className="w-full flex items-center justify-center gap-2 px-3 py-2 text-[10px] font-bold uppercase tracking-wider transition-all"
                style={{
                  background: isRunning && progress?.risk === r ? '#252a33' : riskColors[r],
                  color: '#000',
                  opacity: isRunning && progress?.risk !== r ? 0.4 : 1,
                }}
              >
                {isRunning && progress?.risk === r ? (
                  <><RefreshCw className="w-3 h-3 animate-spin" /> RUNNING...</>
                ) : (
                  <><Play className="w-3 h-3" /> RUN {r.toUpperCase()}</>
                )}
              </button>
            </div>
          ))}
        </div>

        {/* Backtest progress terminal */}
        {progress && progress.status !== "idle" && (
          <div className="t-panel mb-5" style={{
            borderColor: isRunning ? '#e8c300' : progress.status === "done" ? '#00e87b' : '#ff3e3e',
          }}>
            <div className="flex items-center justify-between px-4 py-2" style={{
              background: '#0e1117',
              borderBottom: `1px solid ${isRunning ? '#e8c300' : progress.status === "done" ? '#00e87b' : '#ff3e3e'}`,
            }}>
              <div className="flex items-center gap-2">
                <Terminal className="w-3 h-3" style={{ color: isRunning ? '#e8c300' : progress.status === "done" ? '#00e87b' : '#ff3e3e' }} />
                <span className="text-[10px] font-bold uppercase tracking-wider" style={{
                  color: isRunning ? '#e8c300' : progress.status === "done" ? '#00e87b' : '#ff3e3e',
                }}>
                  BACKTEST {progress.risk?.toUpperCase()} — {progress.status.toUpperCase()}
                </span>
              </div>
              <div className="flex items-center gap-3 text-[9px]" style={{ color: '#5a6270' }}>
                {progress.started && <span>STARTED {progress.started}</span>}
                {progress.finished && <span>FINISHED {progress.finished}</span>}
                {isRunning && <RefreshCw className="w-3 h-3 animate-spin" style={{ color: '#e8c300' }} />}
              </div>
            </div>
            <div
              ref={termRef}
              className="px-4 py-3 overflow-y-auto font-mono"
              style={{
                background: '#0a0c10',
                maxHeight: 200,
                fontSize: 10,
                lineHeight: '1.6',
              }}
            >
              {progress.output_lines.length === 0 ? (
                <span style={{ color: '#3d4450' }}>WAITING FOR OUTPUT...</span>
              ) : (
                progress.output_lines.map((line, i) => (
                  <div key={i} style={{
                    color: line.includes("ERROR") ? '#ff3e3e'
                      : line.includes("WARN") ? '#e8c300'
                      : line.includes("✓") || line.includes("done") || line.includes("Done") ? '#00e87b'
                      : '#5a6270',
                  }}>
                    <span style={{ color: '#3d4450', marginRight: 8 }}>{String(i + 1).padStart(3, ' ')}</span>
                    {line}
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {/* Selected profile results */}
        {p ? (
          <>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-[1px] mb-4">
              {[
                { label: "Total P&L",    value: pnlFmt(p.pnl),        color: p.pnl >= 0 ? '#00e87b' : '#ff3e3e' },
                { label: "Win Rate",     value: `${p.win_rate}%`,      color: '#c8cdd5' },
                { label: "Risk-Reward",  value: `${p.rr}×`,            color: '#4da6ff' },
                { label: "Total Trades", value: p.trades,              color: '#c8cdd5' },
                { label: "Avg Winner",   value: pnlFmt(p.avg_win),     color: '#00e87b' },
                { label: "Avg Loser",    value: pnlFmt(p.avg_loss),    color: '#ff3e3e' },
                { label: "Max Drawdown", value: pnlFmt(p.max_dd),      color: '#ff3e3e' },
                { label: "Avg / Trade",  value: pnlFmt(p.pnl / Math.max(p.trades, 1)), color: p.pnl >= 0 ? '#00e87b' : '#ff3e3e' },
              ].map(({ label, value, color }) => (
                <div key={label} className="t-panel p-3">
                  <p className="text-[9px] uppercase tracking-[1.5px] mb-1" style={{ color: '#5a6270' }}>{label}</p>
                  <p className="text-xl font-bold" style={{ color }}>{value}</p>
                </div>
              ))}
            </div>

            <div className="t-panel p-4 mb-4">
              <h3 className="text-[11px] font-semibold mb-3 uppercase tracking-wider" style={{ color: '#5a6270' }}>
                Equity Curve — <span style={{ color: riskColors[selectedRisk] }}>{selectedRisk}</span>
              </h3>
              <EquityChart curves={curves} selected={selectedRisk} />
            </div>

            {/* Trade list for selected risk */}
            {p.trade_list && p.trade_list.length > 0 && (
              <div className="t-panel p-4 mb-4">
                <h3 className="text-[11px] font-semibold mb-3 uppercase tracking-wider" style={{ color: '#5a6270' }}>
                  Trade List — <span style={{ color: riskColors[selectedRisk] }}>{selectedRisk}</span>
                  <span className="ml-2 font-normal" style={{ color: '#3d4450' }}>{p.trade_list.length} trades</span>
                </h3>
                <TradeTable trades={p.trade_list} />
              </div>
            )}
          </>
        ) : (
          <div className="t-panel p-6 mb-4 text-center text-[11px]" style={{ color: '#3d4450' }}>
            NO RESULTS FOR <span style={{ color: riskColors[selectedRisk] }}>{selectedRisk.toUpperCase()}</span>. CLICK RUN ABOVE.
          </div>
        )}

        {/* All-profile comparison */}
        <div className="t-panel p-4">
          <h3 className="text-[11px] font-semibold mb-3 uppercase tracking-wider" style={{ color: '#5a6270' }}>
            All Profiles — Equity Curves
          </h3>
          <EquityChart curves={curves} selected="all" />

          <table className="mt-4">
            <thead>
              <tr>
                {["Profile", "Trades", "P&L", "Win Rate", "R:R", "Max DD", "Avg/Trade"].map(h => (
                  <th key={h}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(["low", "medium", "high"] as RiskLevel[]).map(r => {
                const rp = results[r];
                if (!rp) return (
                  <tr key={r}>
                    <td className="uppercase font-semibold" style={{ color: riskColors[r] }}>{r}</td>
                    <td colSpan={6} style={{ color: '#3d4450' }}>NO DATA</td>
                  </tr>
                );
                return (
                  <tr key={r} style={{ background: selectedRisk === r ? '#1e222c' : undefined }}>
                    <td className="uppercase font-semibold" style={{ color: riskColors[r] }}>{r}</td>
                    <td>{rp.trades}</td>
                    <td style={{ color: rp.pnl >= 0 ? '#00e87b' : '#ff3e3e', fontWeight: 600 }}>{pnlFmt(rp.pnl)}</td>
                    <td>{rp.win_rate}%</td>
                    <td>{rp.rr}×</td>
                    <td style={{ color: '#ff3e3e' }}>{pnlFmt(rp.max_dd)}</td>
                    <td>{pnlFmt(rp.pnl / Math.max(rp.trades, 1))}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </main>
    </div>
  );
}
