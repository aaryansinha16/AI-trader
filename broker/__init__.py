"""
Broker Execution Adapters
─────────────────────────
Broker-agnostic execution layer for the AI Trader signal pipeline.

Architecture:
  scan_market() → OrderManager → BrokerAdapter.buy/sell/modify_sl

Adapters:
  PaperAdapter    — simulated trades (default, current behavior)
  ZerodhaAdapter  — Kite Connect API (real money)

Config:
  TRADE_MODE env var: "paper" (default) | "zerodha"
"""
