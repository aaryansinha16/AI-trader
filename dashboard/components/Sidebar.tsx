"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import {
  LayoutDashboard, TrendingUp, List, BarChart2,
  Settings, Radio, FlaskConical, Brain, Power, Maximize2,
} from "lucide-react";
import { fetchJSON, postJSON, type LiveState } from "@/lib/api";
import { useTradingMode } from "@/contexts/TradingModeContext";
import { useFullscreenPnl } from "@/app/providers";

const nav = [
  { href: "/",         label: "DASHBOARD",   icon: LayoutDashboard },
  { href: "/live",     label: "LIVE",        icon: Radio },
  { href: "/trades",   label: "TRADES",      icon: List },
  { href: "/backtest", label: "BACKTEST",     icon: FlaskConical },
  { href: "/charts",   label: "CHARTS",       icon: BarChart2 },
  { href: "/ai",       label: "AI MODELS",    icon: Brain },
  { href: "/settings", label: "SETTINGS",     icon: Settings },
];

export default function Sidebar() {
  const pathname = usePathname();
  const [status, setStatus] = useState<string>("connecting");
  const [enabled, setEnabled] = useState<boolean>(true);
  const [toggling, setToggling] = useState(false);
  const { mode, setMode } = useTradingMode();
  const { show: onFullscreenPnl } = useFullscreenPnl();

  useEffect(() => {
    const poll = () => {
      fetchJSON<LiveState>("/api/state").then(d => {
        setStatus(d.status);
        setEnabled(d.scanner_enabled ?? true);
      }).catch(() => setStatus("offline"));
    };
    poll();
    const id = setInterval(poll, 4000);
    return () => clearInterval(id);
  }, []);

  const toggleSystem = async () => {
    setToggling(true);
    try {
      const endpoint = enabled ? "/api/system/stop" : "/api/system/start";
      await postJSON(endpoint);
      setEnabled(!enabled);
      setStatus(enabled ? "stopped" : "idle");
    } catch { /* ignore */ }
    setToggling(false);
  };

  const statusDot = status === "scanning" ? "bg-[#00e87b]"
    : status === "idle" ? "bg-[#4da6ff]"
    : status === "stopped" ? "bg-[#ff3e3e]"
    : "bg-[#3d4450]";

  return (
    <aside className="w-52 min-h-screen flex flex-col" style={{ background: '#0e1117', borderRight: '1px solid #252a33' }}>
      {/* Brand */}
      <div className="px-4 py-4" style={{ borderBottom: '1px solid #252a33' }}>
        <div className="flex items-center gap-2">
          <TrendingUp className="w-4 h-4" style={{ color: '#00e87b' }} />
          <span className="text-sm font-bold tracking-wide" style={{ color: '#00e87b' }}>
            AI TRADER
          </span>
        </div>
        <p className="text-[9px] mt-1 tracking-widest" style={{ color: '#3d4450' }}>NIFTY OPTIONS SYSTEM</p>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-2 py-3 space-y-[2px]">
        {nav.map(({ href, label, icon: Icon }) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              className="flex items-center gap-3 px-3 py-[7px] text-[11px] font-medium tracking-wide transition-all"
              style={{
                background: active ? '#181c24' : 'transparent',
                color: active ? '#00e87b' : '#5a6270',
                borderLeft: active ? '2px solid #00e87b' : '2px solid transparent',
              }}
            >
              <Icon className="w-3.5 h-3.5 flex-shrink-0" />
              {label}
            </Link>
          );
        })}

        {/* Full P&L shortcut */}
        <button
          onClick={onFullscreenPnl}
          className="flex items-center gap-3 px-3 py-[7px] text-[11px] font-medium tracking-wide w-full text-left mt-4 transition-all"
          style={{ color: '#e8c300', borderLeft: '2px solid transparent' }}
        >
          <Maximize2 className="w-3.5 h-3.5 flex-shrink-0" />
          FULL P&L [Ctrl+K]
        </button>
      </nav>

      {/* Mode toggle */}
      <div className="px-3 py-3" style={{ borderTop: '1px solid #252a33' }}>
        <p className="text-[8px] uppercase tracking-[2px] mb-2 px-1" style={{ color: '#3d4450' }}>TRADING MODE</p>
        <div className="flex" style={{ border: '1px solid #252a33', background: '#0e1117' }}>
          <button
            onClick={() => setMode("test")}
            className="flex-1 py-[5px] text-[10px] font-bold uppercase tracking-wider text-center transition-all"
            style={{
              background: mode === "test" ? '#e8c300' : 'transparent',
              color: mode === "test" ? '#000' : '#5a6270',
              borderRight: '1px solid #252a33',
            }}
          >
            TEST
          </button>
          <button
            onClick={() => setMode("live")}
            className="flex-1 py-[5px] text-[10px] font-bold uppercase tracking-wider text-center transition-all"
            style={{
              background: mode === "live" ? '#ff3e3e' : 'transparent',
              color: mode === "live" ? '#000' : '#5a6270',
            }}
          >
            LIVE
          </button>
        </div>
        <p className="text-[8px] mt-1.5 px-1" style={{ color: mode === "live" ? '#ff3e3e' : '#3d4450' }}>
          {mode === "live" ? "⚠ REAL EXECUTIONS VIA ZERODHA" : "SIMULATED PAPER TRADES ONLY"}
        </p>
      </div>

      {/* System control */}
      <div className="px-3 py-3 space-y-2" style={{ borderTop: '1px solid #252a33' }}>
        <button
          onClick={toggleSystem}
          disabled={toggling}
          className={`flex items-center justify-center gap-2 w-full py-[6px] text-[10px] font-semibold tracking-wider uppercase transition-all disabled:opacity-50 ${enabled ? "t-btn-red" : "t-btn-green"}`}
        >
          <Power className="w-3 h-3" />
          {toggling ? "..." : enabled ? "STOP SYSTEM" : "START SYSTEM"}
        </button>

        <div className="flex items-center gap-2 text-[10px] px-1">
          <span className={`w-[6px] h-[6px] ${statusDot} ${status === "scanning" ? "t-pulse" : ""}`} />
          <span style={{ color: status === "scanning" ? '#00e87b' : status === "idle" ? '#4da6ff' : status === "stopped" ? '#ff3e3e' : '#3d4450' }}>
            {status === "scanning" ? "SCANNING..." : status.toUpperCase()}
          </span>
        </div>
      </div>
    </aside>
  );
}
