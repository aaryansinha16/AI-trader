"""
Order Manager
─────────────
Bridges the signal pipeline (scan_market) to the broker adapter.

Responsibilities:
  - Receives trade suggestions from the scanner
  - Applies safety checks (max daily loss, concurrent position limit, cooldowns)
  - Delegates order placement to the active BrokerAdapter
  - Tracks open orders, manages SL orders, handles exits
  - Provides a unified interface for the dashboard

The OrderManager does NOT decide WHEN to trade — that's the scanner's job.
It decides WHETHER to execute (safety), HOW to execute (broker API), and
tracks WHAT happened (position state).
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional

from broker.base_adapter import (
    BrokerAdapter,
    OrderResponse,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from broker.paper_adapter import PaperAdapter
from broker.zerodha_adapter import ZerodhaAdapter
from utils.logger import get_logger

logger = get_logger("order_manager")


@dataclass
class ManagedPosition:
    """Internal position tracking with our signal metadata."""
    order_id: str
    symbol: str
    direction: str                     # "CALL" or "PUT"
    strategy: str
    entry_time: datetime
    entry_price: float
    quantity: int
    sl_price: float
    target_price: float
    sl_order_id: Optional[str] = None  # broker's SL order ID
    current_price: float = 0.0
    pnl: float = 0.0
    status: str = "OPEN"               # OPEN, CLOSED
    exit_time: Optional[datetime] = None
    exit_price: float = 0.0
    exit_reason: str = ""
    final_score: float = 0.0
    ml_prob: float = 0.0
    journey: list = field(default_factory=list)


class OrderManager:
    """
    Central execution coordinator.

    Config via env vars:
      TRADE_MODE           — "paper" (default) or "zerodha"
      MAX_DAILY_LOSS       — max cumulative loss before auto-stop (default: -5000)
      MAX_CONCURRENT_POSITIONS — max open positions at once (default: 1)
      ORDER_CONFIRMATION   — "auto" (default) or "manual" (requires dashboard click)
    """

    def __init__(self):
        self._mode = os.getenv("TRADE_MODE", "paper").lower()
        self._max_daily_loss = float(os.getenv("MAX_DAILY_LOSS", "-5000"))
        self._max_concurrent = int(os.getenv("MAX_CONCURRENT_POSITIONS", "1"))
        self._confirmation_mode = os.getenv("ORDER_CONFIRMATION", "auto").lower()

        # Initialize the appropriate adapter
        if self._mode == "zerodha":
            self._adapter: BrokerAdapter = ZerodhaAdapter()
        else:
            self._adapter: BrokerAdapter = PaperAdapter()

        # State
        self._positions: list[ManagedPosition] = []
        self._daily_pnl: float = 0.0
        self._trade_count: int = 0
        self._halted: bool = False      # set by kill switch or max loss
        self._halt_reason: str = ""
        self._pending_signals: list[dict] = []  # for manual confirmation mode
        self._lock = threading.Lock()

        logger.info(f"OrderManager initialized: mode={self._mode}, "
                    f"max_loss=₹{self._max_daily_loss}, max_concurrent={self._max_concurrent}, "
                    f"confirmation={self._confirmation_mode}")

    # ── Properties ────────────────────────────────────────────────────

    @property
    def adapter(self) -> BrokerAdapter:
        return self._adapter

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def trade_count(self) -> int:
        return self._trade_count

    @property
    def open_positions(self) -> list[ManagedPosition]:
        return [p for p in self._positions if p.status == "OPEN"]

    @property
    def closed_positions(self) -> list[ManagedPosition]:
        return [p for p in self._positions if p.status == "CLOSED"]

    @property
    def pending_signals(self) -> list[dict]:
        return list(self._pending_signals)

    # ── Lifecycle ─────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Authenticate with the broker."""
        ok = self._adapter.authenticate()
        if ok:
            logger.info(f"Connected to {self._adapter.broker_name}")
        else:
            logger.error(f"Failed to connect to {self._adapter.broker_name}")
        return ok

    def reset_daily(self):
        """Call at start of each trading day to reset counters."""
        self._daily_pnl = 0.0
        self._trade_count = 0
        self._halted = False
        self._halt_reason = ""
        self._pending_signals.clear()
        logger.info("OrderManager: daily reset")

    # ── Entry ─────────────────────────────────────────────────────────

    def submit_signal(self, signal: dict) -> dict:
        """
        Receive a trade signal from scan_market and decide what to do.

        signal dict must contain:
          symbol, direction, strategy, entry_premium, sl_price, target_price,
          lots, final_score, ml_prob, expiry, index_price

        Returns dict with:
          action: "executed" | "queued" | "rejected"
          reason: human-readable explanation
          order_id: (if executed)
        """
        with self._lock:
            # ── Safety checks ─────────────────────────────────────────

            # 1. Halted?
            if self._halted:
                return {"action": "rejected", "reason": f"Trading halted: {self._halt_reason}"}

            # 2. Max daily loss?
            if self._daily_pnl <= self._max_daily_loss:
                self._halted = True
                self._halt_reason = f"Max daily loss breached (₹{self._daily_pnl:,.0f})"
                logger.warning(f"HALT: {self._halt_reason}")
                return {"action": "rejected", "reason": self._halt_reason}

            # 3. Concurrent position limit?
            open_count = len(self.open_positions)
            if open_count >= self._max_concurrent:
                return {
                    "action": "rejected",
                    "reason": f"Max concurrent positions ({self._max_concurrent}) reached"
                }

            # 4. Same-direction duplicate?
            for pos in self.open_positions:
                if pos.direction == signal.get("direction"):
                    return {
                        "action": "rejected",
                        "reason": f"Already have an open {pos.direction} position ({pos.symbol})"
                    }

            # 5. Manual confirmation mode?
            if self._confirmation_mode == "manual":
                self._pending_signals.append({
                    **signal,
                    "received_at": datetime.now().isoformat(),
                })
                logger.info(f"Signal QUEUED for manual confirmation: {signal.get('symbol')} "
                           f"{signal.get('direction')} {signal.get('strategy')}")
                return {"action": "queued", "reason": "Waiting for manual confirmation on dashboard"}

            # ── Execute ───────────────────────────────────────────────
            return self._execute_entry(signal)

    def confirm_signal(self, index: int) -> dict:
        """
        Confirm a pending signal (manual mode). Called from dashboard.
        """
        with self._lock:
            if index < 0 or index >= len(self._pending_signals):
                return {"action": "rejected", "reason": "Invalid signal index"}
            signal = self._pending_signals.pop(index)
            return self._execute_entry(signal)

    def reject_signal(self, index: int) -> dict:
        """Reject a pending signal (manual mode)."""
        with self._lock:
            if index < 0 or index >= len(self._pending_signals):
                return {"action": "rejected", "reason": "Invalid signal index"}
            signal = self._pending_signals.pop(index)
            logger.info(f"Signal REJECTED by user: {signal.get('symbol')}")
            return {"action": "rejected", "reason": "Rejected by user"}

    def _execute_entry(self, signal: dict) -> dict:
        """Place the actual entry order via the adapter."""
        symbol = signal["symbol"]
        quantity = signal.get("lots", 65)
        entry_premium = signal.get("entry_premium", 0)

        resp = self._adapter.buy(
            symbol=symbol,
            quantity=quantity,
            price=entry_premium,
            tag=signal.get("strategy", "")[:20],
        )

        if resp.status in (OrderStatus.COMPLETE, OrderStatus.OPEN):
            fill_price = resp.average_price if resp.average_price > 0 else entry_premium
            pos = ManagedPosition(
                order_id=resp.order_id,
                symbol=symbol,
                direction=signal.get("direction", ""),
                strategy=signal.get("strategy", ""),
                entry_time=datetime.now(),
                entry_price=fill_price,
                quantity=quantity,
                sl_price=signal.get("sl_price", fill_price * 0.85),
                target_price=signal.get("target_price", fill_price * 1.50),
                final_score=signal.get("final_score", 0),
                ml_prob=signal.get("ml_prob", 0),
            )
            self._positions.append(pos)
            self._trade_count += 1

            logger.info(
                f"ENTRY: {signal.get('direction')} {symbol} ×{quantity} @ ₹{fill_price:.2f} "
                f"[{signal.get('strategy')}] score={signal.get('final_score', 0):.2f} "
                f"→ {resp.order_id}"
            )

            return {
                "action": "executed",
                "reason": f"Order placed: {resp.order_id}",
                "order_id": resp.order_id,
                "fill_price": fill_price,
            }
        else:
            logger.warning(f"ENTRY FAILED: {symbol} — {resp.message}")
            return {"action": "rejected", "reason": f"Order failed: {resp.message}"}

    # ── Exit ──────────────────────────────────────────────────────────

    def exit_position(self, order_id: str, price: float = 0, reason: str = "MANUAL") -> dict:
        """Exit a specific position by its order_id."""
        with self._lock:
            pos = next((p for p in self._positions if p.order_id == order_id and p.status == "OPEN"), None)
            if not pos:
                return {"action": "rejected", "reason": f"No open position with ID {order_id}"}

            resp = self._adapter.sell(
                symbol=pos.symbol,
                quantity=pos.quantity,
                price=price,
                tag=reason[:20],
            )

            if resp.status in (OrderStatus.COMPLETE, OrderStatus.OPEN):
                fill = resp.average_price if resp.average_price > 0 else price
                pnl = (fill - pos.entry_price) * pos.quantity
                pos.status = "CLOSED"
                pos.exit_time = datetime.now()
                pos.exit_price = fill
                pos.exit_reason = reason
                pos.pnl = pnl
                self._daily_pnl += pnl

                logger.info(
                    f"EXIT: {pos.symbol} ×{pos.quantity} @ ₹{fill:.2f} "
                    f"P&L=₹{pnl:+,.0f} [{reason}]"
                )

                # Check max daily loss after exit
                if self._daily_pnl <= self._max_daily_loss:
                    self._halted = True
                    self._halt_reason = f"Max daily loss breached (₹{self._daily_pnl:,.0f})"
                    logger.warning(f"HALT: {self._halt_reason}")

                return {"action": "executed", "pnl": pnl, "order_id": resp.order_id}
            else:
                logger.error(f"EXIT FAILED: {pos.symbol} — {resp.message}")
                return {"action": "rejected", "reason": resp.message}

    # ── SL Management ─────────────────────────────────────────────────

    def update_sl(self, order_id: str, new_sl: float):
        """Update the SL price for a managed position."""
        for pos in self._positions:
            if pos.order_id == order_id and pos.status == "OPEN":
                old_sl = pos.sl_price
                if new_sl > old_sl:
                    pos.sl_price = new_sl
                    # If there's a live SL order on the broker, modify it
                    if pos.sl_order_id and self._mode != "paper":
                        self._adapter.modify_order(
                            pos.sl_order_id,
                            trigger_price=new_sl,
                        )
                    logger.debug(f"SL updated: {pos.symbol} ₹{old_sl:.2f} → ₹{new_sl:.2f}")
                break

    def check_sl_target(self, symbol: str, current_price: float):
        """Check if any open position has hit SL or target. Called by tick monitor."""
        with self._lock:
            for pos in self._positions:
                if pos.symbol != symbol or pos.status != "OPEN":
                    continue
                pos.current_price = current_price
                pos.pnl = (current_price - pos.entry_price) * pos.quantity

                if current_price <= pos.sl_price:
                    self.exit_position(pos.order_id, price=current_price, reason="SL_HIT")
                elif current_price >= pos.target_price:
                    self.exit_position(pos.order_id, price=current_price, reason="TARGET_HIT")

    # ── Safety ────────────────────────────────────────────────────────

    def kill_switch(self) -> dict:
        """
        EMERGENCY: Close all positions, cancel all orders, halt trading.
        """
        with self._lock:
            self._halted = True
            self._halt_reason = "KILL SWITCH activated"
            logger.warning("=" * 50)
            logger.warning("  KILL SWITCH ACTIVATED")
            logger.warning("=" * 50)

            # Close managed positions
            results = []
            for pos in self._positions:
                if pos.status == "OPEN":
                    result = self.exit_position(pos.order_id, price=pos.current_price, reason="KILL_SWITCH")
                    results.append(result)

            # Also trigger adapter's own kill switch (catches anything we missed)
            adapter_results = self._adapter.kill_switch()

            return {
                "halted": True,
                "positions_closed": len(results),
                "adapter_exits": len(adapter_results),
                "daily_pnl": self._daily_pnl,
            }

    def resume(self) -> dict:
        """Resume trading after a halt (manual action)."""
        with self._lock:
            self._halted = False
            self._halt_reason = ""
            logger.info("Trading RESUMED")
            return {"halted": False}

    # ── Reconciliation ────────────────────────────────────────────────

    def reconcile(self) -> dict:
        """
        Compare our internal state vs the broker's actual positions.
        Returns discrepancies for manual review.
        """
        broker_positions = self._adapter.get_positions()
        our_open = {p.symbol: p for p in self.open_positions}
        broker_open = {p.symbol: p for p in broker_positions}

        discrepancies = []
        # Positions we think are open but broker doesn't have
        for sym, pos in our_open.items():
            if sym not in broker_open:
                discrepancies.append({
                    "type": "PHANTOM",
                    "symbol": sym,
                    "our_qty": pos.quantity,
                    "broker_qty": 0,
                    "message": f"We think {sym} is open but broker has no position",
                })

        # Positions broker has but we don't track
        for sym, bpos in broker_open.items():
            if sym not in our_open:
                discrepancies.append({
                    "type": "ORPHAN",
                    "symbol": sym,
                    "our_qty": 0,
                    "broker_qty": bpos.quantity,
                    "message": f"Broker has {sym} ×{bpos.quantity} but we don't track it",
                })

        # Quantity mismatches
        for sym in set(our_open) & set(broker_open):
            if our_open[sym].quantity != broker_open[sym].quantity:
                discrepancies.append({
                    "type": "QTY_MISMATCH",
                    "symbol": sym,
                    "our_qty": our_open[sym].quantity,
                    "broker_qty": broker_open[sym].quantity,
                    "message": f"{sym}: we have ×{our_open[sym].quantity}, broker has ×{broker_open[sym].quantity}",
                })

        if discrepancies:
            logger.warning(f"RECONCILIATION: {len(discrepancies)} discrepancy(ies)")
            for d in discrepancies:
                logger.warning(f"  {d['type']}: {d['message']}")
        else:
            logger.debug("Reconciliation: all positions match")

        return {"discrepancies": discrepancies, "our_open": len(our_open), "broker_open": len(broker_open)}

    # ── State for dashboard ───────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return full state for the dashboard API."""
        return {
            "mode": self._mode,
            "broker": self._adapter.broker_name,
            "connected": self._adapter.is_connected,
            "halted": self._halted,
            "halt_reason": self._halt_reason,
            "daily_pnl": round(self._daily_pnl, 2),
            "trade_count": self._trade_count,
            "max_daily_loss": self._max_daily_loss,
            "max_concurrent": self._max_concurrent,
            "confirmation_mode": self._confirmation_mode,
            "open_positions": [
                {
                    "order_id": p.order_id,
                    "symbol": p.symbol,
                    "direction": p.direction,
                    "strategy": p.strategy,
                    "entry_time": p.entry_time.isoformat(),
                    "entry_price": p.entry_price,
                    "quantity": p.quantity,
                    "sl_price": p.sl_price,
                    "target_price": p.target_price,
                    "current_price": p.current_price,
                    "pnl": round(p.pnl, 2),
                    "final_score": p.final_score,
                }
                for p in self.open_positions
            ],
            "closed_positions": [
                {
                    "order_id": p.order_id,
                    "symbol": p.symbol,
                    "direction": p.direction,
                    "strategy": p.strategy,
                    "entry_price": p.entry_price,
                    "exit_price": p.exit_price,
                    "pnl": round(p.pnl, 2),
                    "exit_reason": p.exit_reason,
                }
                for p in self.closed_positions
            ],
            "pending_signals": self._pending_signals,
        }
