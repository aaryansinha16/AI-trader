# AI Trader Dashboard

Retro terminal-style monitoring dashboard for the AI Trader NSE F&O system.

**Stack:** Next.js 15 · TypeScript · Tailwind CSS v4 · Recharts · JetBrains Mono

---

## Prerequisites

- Node.js 18+
- Flask API running on `http://localhost:5050` (see `../backend/app.py`)

## Setup

```bash
npm install
```

## Running

```bash
npm run dev       # Dev server → http://localhost:3000
npm run build     # Production build
npm run start     # Serve production build
```

> **The Flask API must be running first.** Start it from the project root:
> ```bash
> python backend/app.py
> ```

---

## Pages

| Route | Description |
|---|---|
| `/` | Main dashboard — live stats, equity curve, recent trades, ticker |
| `/live` | System status, NIFTY price, market regime, trade suggestions |
| `/trades` | Trade history, P&L bar chart, strategy breakdown |
| `/charts` | NIFTY candles, option chain table, tick charts, analytics |
| `/backtest` | Run backtests per risk profile, compare equity curves |
| `/ai` | ML model status (LightGBM, RL agents), Kelly sizer info |
| `/settings` | Risk profile selector, system info, CLI reference |

## Key Components

| Component | Description |
|---|---|
| `Sidebar` | Navigation + system start/stop control |
| `StatCard` | Neon metric card with optional pulse indicator |
| `EquityChart` | Recharts equity curve for all 3 risk profiles |
| `PnlBarChart` | Per-trade P&L bar chart (green/red) |
| `TradeTable` | Paginated trade history table |
| `FullscreenPnl` | F11 fullscreen live P&L overlay |
| `RiskProfileCard` | Selectable risk profile card |
| `Badge` | Neon-colored status badge |

## API Endpoints (proxied from Flask :5050)

All calls go through `lib/api.ts` → `fetchJSON("/api/...")` which is proxied to `http://localhost:5050`.

| Endpoint | Description |
|---|---|
| `GET /api/state` | Live system state (price, regime, status) |
| `GET /api/trades` | All recorded trades |
| `GET /api/backtest/results` | Backtest results per risk profile |
| `POST /api/backtest/run` | Trigger a backtest run |
| `GET /api/equity/curve` | Equity curve data points |
| `GET /api/risk/profiles` | Risk profile configurations |
| `GET /api/rl/status` | RL agent load status |
| `GET /api/charts/nifty` | NIFTY candle data |
| `GET /api/charts/option-chain` | Live option chain |
| `POST /api/scanner/toggle` | Start/stop the background scanner |

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `F11` | Toggle fullscreen P&L overlay |
