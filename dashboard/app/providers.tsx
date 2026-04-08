"use client";

import { createContext, useContext, useEffect, useRef, useState } from "react";
import { TradingModeProvider, useTradingMode } from "@/contexts/TradingModeContext";
import RetroDialog from "@/components/RetroDialog";
import FullscreenPnl from "@/components/FullscreenPnl";
import { SSE_STREAM_URL, type StreamPayload } from "@/lib/api";
import { playCoinChime, primeAudio, requestNotificationPermission, showTradeNotification } from "@/lib/tradeAlert";

/* ── Global trade-alert: coin chime + browser notification on new signals ── */
function GlobalTradeAlert() {
  const seenRef = useRef<Set<string>>(new Set());
  const primedRef = useRef(false);

  // Prime audio + request notification permission on first click anywhere
  useEffect(() => {
    const prime = () => {
      if (primedRef.current) return;
      primedRef.current = true;
      primeAudio();
      requestNotificationPermission();
    };
    window.addEventListener("click", prime, { once: true });
    window.addEventListener("keydown", prime, { once: true });
    return () => {
      window.removeEventListener("click", prime);
      window.removeEventListener("keydown", prime);
    };
  }, []);

  // Watch SSE stream for new trade_suggestions
  useEffect(() => {
    const es = new EventSource(SSE_STREAM_URL);

    let seeded = false;

    es.onmessage = (event) => {
      try {
        const d: StreamPayload = JSON.parse(event.data);
        const suggestions = d?.state?.trade_suggestions ?? [];

        if (!seeded) {
          // First message: silently record all current suggestions, no alert
          suggestions.forEach((s) => {
            seenRef.current.add(`${s.symbol}-${s.direction}-${s.time ?? ""}`);
          });
          seeded = true;
          return;
        }

        // Subsequent messages: alert only on genuinely new suggestions
        suggestions.forEach((s) => {
          const key = `${s.symbol}-${s.direction}-${s.time ?? ""}`;
          if (!seenRef.current.has(key)) {
            seenRef.current.add(key);
            playCoinChime();
            showTradeNotification(s.symbol, s.direction, s.final_score ?? 0);
          }
        });
      } catch {}
    };

    return () => es.close();
  }, []);

  return null;
}

/* ── Global fullscreen P&L context ────────────────────────────────────────── */
const FullscreenPnlContext = createContext<{ show: () => void }>({ show: () => {} });
export function useFullscreenPnl() { return useContext(FullscreenPnlContext); }

function GlobalDialogs() {
  const { showDialog, setShowDialog, dialogError } = useTradingMode();
  return (
    <RetroDialog
      open={showDialog}
      onClose={() => setShowDialog(false)}
      title="CONFIGURATION ERROR"
      message={dialogError ?? ""}
      type="error"
    />
  );
}

function GlobalShortcuts({ children }: { children: React.ReactNode }) {
  const [showFullPnl, setShowFullPnl] = useState(false);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.key === "k") {
        e.preventDefault();
        setShowFullPnl(p => !p);
      }
      if (e.key === "Escape") setShowFullPnl(false);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  return (
    <FullscreenPnlContext.Provider value={{ show: () => setShowFullPnl(true) }}>
      {showFullPnl && <FullscreenPnl onClose={() => setShowFullPnl(false)} />}
      {children}
    </FullscreenPnlContext.Provider>
  );
}

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <TradingModeProvider>
      <GlobalTradeAlert />
      <GlobalShortcuts>
        {children}
      </GlobalShortcuts>
      <GlobalDialogs />
    </TradingModeProvider>
  );
}
