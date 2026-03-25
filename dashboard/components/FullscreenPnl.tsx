"use client";

import { useEffect, useState, useCallback } from "react";
import { X } from "lucide-react";
import { fetchJSON, type BacktestResults, type LiveState } from "@/lib/api";

const pnlFmt = (v: number) =>
  `${v >= 0 ? "+" : ""}${v.toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;

export default function FullscreenPnl({ onClose }: { onClose: () => void }) {
  const [pnl, setPnl] = useState<number>(0);
  const [winRate, setWinRate] = useState<string>("--");
  const [trades, setTrades] = useState<number>(0);
  const [regime, setRegime] = useState<string>("--");
  const [price, setPrice] = useState<number>(0);
  const [time, setTime] = useState<string>("--:--:--");

  const load = useCallback(async () => {
    const [results, live] = await Promise.all([
      fetchJSON<BacktestResults>("/api/backtest/results").catch(() => ({})),
      fetchJSON<LiveState>("/api/state").catch(() => null),
    ]);
    const high = (results as BacktestResults).high;
    if (high) {
      setPnl(high.pnl);
      setWinRate(`${high.win_rate}%`);
      setTrades(high.trades);
    }
    if (live) {
      setRegime(live.regime);
      setPrice(live.last_price);
    }
    setTime(new Date().toLocaleTimeString("en-IN", { hour12: false }));
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, 3000);
    return () => clearInterval(id);
  }, [load]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape" || e.key === "F11") {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  const isPositive = pnl >= 0;

  return (
    <div className="fullscreen-pnl">
      <button
        onClick={onClose}
        className="absolute top-6 right-6 z-10 transition-colors"
        style={{ color: '#3d4450' }}
      >
        <X className="w-8 h-8" />
      </button>

      <p className="text-lg tracking-[8px] uppercase mb-2 relative z-10" style={{ color: '#3d4450' }}>{time}</p>

      <p className="pnl-label mb-4">TOTAL P&L</p>

      <div
        className="pnl-value"
        style={{
          color: isPositive ? '#00e87b' : '#ff3e3e',
          textShadow: isPositive
            ? '0 0 40px rgba(0,232,123,0.3), 0 0 80px rgba(0,232,123,0.1)'
            : '0 0 40px rgba(255,62,62,0.3), 0 0 80px rgba(255,62,62,0.1)',
        }}
      >
        {pnlFmt(pnl)}
      </div>

      <div className="flex gap-16 mt-12 text-center relative z-10">
        <div>
          <p className="text-[11px] tracking-[2px] uppercase mb-2" style={{ color: '#3d4450' }}>Win Rate</p>
          <p className="text-2xl font-bold" style={{ color: '#4da6ff' }}>{winRate}</p>
        </div>
        <div>
          <p className="text-[11px] tracking-[2px] uppercase mb-2" style={{ color: '#3d4450' }}>Trades</p>
          <p className="text-2xl font-bold" style={{ color: '#c8cdd5' }}>{trades}</p>
        </div>
        <div>
          <p className="text-[11px] tracking-[2px] uppercase mb-2" style={{ color: '#3d4450' }}>NIFTY</p>
          <p className="text-2xl font-bold" style={{ color: '#4da6ff' }}>
            {price ? price.toLocaleString("en-IN", { maximumFractionDigits: 1 }) : "--"}
          </p>
        </div>
        <div>
          <p className="text-[11px] tracking-[2px] uppercase mb-2" style={{ color: '#3d4450' }}>Regime</p>
          <p className="text-2xl font-bold" style={{
            color: regime?.includes("BULL") ? '#00e87b' : regime?.includes("BEAR") ? '#ff3e3e' : '#e8c300'
          }}>
            {regime}
          </p>
        </div>
      </div>

      <p className="bottom-6 mt-4 text-[10px] tracking-[3px] z-10" style={{ color: '#3d4450' }}>
        PRESS ESC TO CLOSE
      </p>
    </div>
  );
}
