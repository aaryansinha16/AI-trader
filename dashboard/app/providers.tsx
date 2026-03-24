"use client";

import { createContext, useContext, useEffect, useState } from "react";
import { TradingModeProvider, useTradingMode } from "@/contexts/TradingModeContext";
import RetroDialog from "@/components/RetroDialog";
import FullscreenPnl from "@/components/FullscreenPnl";

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
      <GlobalShortcuts>
        {children}
      </GlobalShortcuts>
      <GlobalDialogs />
    </TradingModeProvider>
  );
}
