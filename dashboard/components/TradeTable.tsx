"use client";

import Badge from "./Badge";
import type { Trade } from "@/lib/api";
import { toDateStr, toISTTimeFull } from "@/lib/time";

interface Props {
  trades: Trade[];
  maxRows?: number;
}

const resultVariant = (r: string) => {
  if (r === "TARGET") return "green";
  if (r === "SL" || r === "TRAILING_SL") return "red";
  if (r === "RL_EXIT" || r === "DQN_EXIT") return "purple";
  return "yellow";
};

const fmt = (p: number) =>
  `₹${p >= 0 ? "+" : ""}${p.toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;

export default function TradeTable({ trades, maxRows }: Props) {
  const rows = maxRows ? trades.slice(-maxRows).reverse() : [...trades].reverse();

  if (!rows.length) {
    return (
      <p className="text-[11px] text-center py-8" style={{ color: '#3d4450' }}>NO TRADES TO DISPLAY</p>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table>
        <thead>
          <tr>
            {["Date", "Time", "Symbol", "Dir", "Strategy", "Entry", "Exit", "P&L", "Result", "Score", "Regime"].map(h => (
              <th key={h}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((t, i) => (
            <tr key={i}>
              <td style={{ color: '#5a6270' }}>
                {toDateStr(String(t.entry_time))}
              </td>
              <td style={{ color: '#5a6270' }}>
                {toISTTimeFull(String(t.entry_time))}
              </td>
              <td style={{ color: '#c8cdd5' }}>{t.symbol}</td>
              <td>
                <Badge label={t.direction} variant={t.direction === "CALL" ? "green" : "red"} />
              </td>
              <td style={{ color: '#5a6270' }}>{t.strategy?.replace(/_/g, " ")}</td>
              <td>₹{t.entry_premium?.toFixed(1)}</td>
              <td>₹{t.exit_premium?.toFixed(1) ?? "--"}</td>
              <td style={{ color: t.pnl > 0 ? '#00e87b' : t.pnl < 0 ? '#ff3e3e' : '#5a6270', fontWeight: 600 }}>
                {fmt(t.pnl)}
              </td>
              <td>
                <Badge label={t.result} variant={resultVariant(t.result)} />
              </td>
              <td>{(t.final_score * 100).toFixed(0)}%</td>
              <td style={{ color: '#5a6270' }}>{t.regime}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
